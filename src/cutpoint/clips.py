"""``cutpoint clips``: export selected highlight moments as uniformly-encoded files.

:func:`export_clips` runs three steps: :func:`plan_clips` (moments to
:class:`ClipPlan`), :func:`render_clips` (encode) and :func:`write_manifest`.
"""

from __future__ import annotations

import logging
import ntpath
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any

from cutpoint import boundaries, cache, jsonout, sources, srt
from cutpoint.audio_tracks import export_tracks
from cutpoint.color import SKIP_360, SKIP_LOG, SKIP_NOT_UNDERWATER, AutoGrade, auto_grade
from cutpoint.config import (
    COLOR_MODES,
    color_default,
    is_absolute_any,
    path_key,
    remap_source,
    source_path,
    unc_path,
)
from cutpoint.config import export_workers as configured_workers
from cutpoint.crop import Crop, detect, parse_spec, project_crop
from cutpoint.encode import (
    GRADE_KEYS,
    Grade,
    Lut,
    OutputFormat,
    bt709_filter,
    choose_format,
    export_clip_checked,
    plan_makeup,
    plan_peaks,
    warn_if_log,
)
from cutpoint.events.tails import EventTailSettings, event_tail, load_events
from cutpoint.ffmpeg_utils import (
    SOFT_STOP_S,
    STOP_GRACE_S,
    ChildScope,
    FFmpegError,
    MediaInfo,
    check_not_sources,
    child_scope,
    clip_momentary,
    output_dir,
    probe,
    recover_swap,
    same_file,
    source_keys,
    staged_dir,
    swap_in,
    swap_lock,
    video_encoder,
)
from cutpoint.planning import (
    MANIFEST_GENERATOR,
    Padding,
    check_own_manifest,
    file_size,
    kept_why,
    manifest_entries,
    pad,
    report_clips,
    select_moments,
    speech_pass,
)
from cutpoint.styles import Style, get_style
from cutpoint.tail import TailResult, TailSettings, smart_tail
from cutpoint.utils import clip_files, seconds_keys, slugify

if TYPE_CHECKING:
    from cutpoint.pano.render import PanoExporter

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_FADE_S = 0.25  # audio fade in/out baked into every exported clip
MIN_EXPORT_S = 0.1  # shortest clip export_clips accepts
POLL_S = 0.1  # parallel export: the main thread waits in steps this long (Ctrl+C on Windows)


ARC_GROUPS = ("lift", "scenery", "run", "jump", "crash", "social")  # within a stretch of the day
ARC_OPENER_S = 5.0  # the opener is trimmed to this (around its middle)
ARC_MAX_REPEAT = 2  # same-category clips in a row
ARC_WINDOW = 4  # how far ahead a local swap may reach (clips)
REACTION_LABEL = "crash reaction"  # ``cutpoint mix`` label of a crash's reaction segment
BOUND_KEYS = ("min_start", "max_end")  # source-time limits of a segment's lead-in and tails
# Free space check of a temporary export (export_clips(temporary=True)): a clip's size is
# estimated at this rate for 1920x1080 at 30 fps (scaled by its pixels, and by its frame rate
# above 30), and the drive must hold the clips, the final video and its staging copy, with
# this much to spare.
CLIP_BYTES_PER_S_1080P = 2.5e6  # 20 Mb/s, the typical rate at 1080p30
TEMP_COPIES = 3
TEMP_SPARE_GB = 1.0
_WINDOWS = sys.platform == "win32"  # D:foo is drive-relative only here (elsewhere a file name)

Unit = list[dict[str, Any]]  # a clip, or a crash plus its reaction


def order_moments(moments: list[dict[str, Any]], order: str) -> list[dict[str, Any]]:
    """Order moments for playback.

    ``story``: hook (2nd-best) first, best moment last, the rest chronological.
    ``arc``: for ``category``-labelled moments (``cutpoint mix``), see :func:`_arc`;
    without any category it is ``story``.
    """
    chrono = sorted(moments, key=lambda m: (m.get("source_index", 0), m["source"], m["start"]))
    if order == "chronological":
        return chrono
    by_score = sorted(moments, key=lambda m: -(m["score"] or 0))
    if order == "score":
        return by_score
    if order == "arc" and any(m.get("category") for m in moments):
        return _arc(chrono)
    if order not in ("story", "arc"):
        raise ValueError(f"Unknown order {order!r}")
    if len(by_score) < 2:
        return by_score
    best, hook = by_score[0], by_score[1]
    middle = [m for m in chrono if m is not best and m is not hook]
    return [hook, *middle, best]


def index_sources(moments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Number sources by first appearance (``source_index``, in place), so a segments
    file in day order (as ``cutpoint mix`` writes it) keeps that order."""
    first: dict[str, int] = {}
    for m in moments:
        m["source_index"] = first.setdefault(m["source"], len(first))
    return moments


def _cat(u: Unit) -> str | None:
    return u[0].get("category")


def _streak(seq: list[Unit], cat: str | None) -> int:
    """Clips of ``cat`` at the end of ``seq`` (0 for uncategorised)."""
    n = 0
    for u in reversed(seq):
        if cat is None or _cat(u) != cat:
            break
        n += len(u)
    return n


def _max_streak(seq: list[Unit]) -> int:
    """Longest run of same-category clips in ``seq`` (uncategorised clips count as 0)."""
    best = n = 0
    for i, u in enumerate(seq):
        same = i > 0 and _cat(u) is not None and _cat(u) == _cat(seq[i - 1])
        n = (n if same else 0) + (len(u) if _cat(u) is not None else 0)
        best = max(best, n)
    return best


def _arc(chrono: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Arc order of chronological, categorised moments.

    Opener: the best lift / scenery clip of the first third of the day (else the
    earliest one), trimmed to :data:`ARC_OPENER_S`. Body: chronological by
    stretch of the day (``chapter``, else source), each stretch grouped
    lift -> scenery -> run / jump -> crash -> social, with local swaps so no more
    than :data:`ARC_MAX_REPEAT` clips of a category play in a row. A crash is
    always followed by its reaction. Finale: the best crash (else run / jump),
    then the best scenery clip of the last third as outro. Ties go to the earlier
    clip; uncategorised moments stay in the body.
    """
    n = len(chrono)
    units: list[Unit] = []
    for m in chrono:
        host = next(
            (
                u
                for u in reversed(units)
                if m.get("label") == REACTION_LABEL
                and len(u) == 1
                and _cat(u) == "crash"
                and u[0]["source"] == m["source"]
                and u[0].get("label") != REACTION_LABEL
            ),
            None,
        )
        if host is None:
            units.append([m])
        else:
            host.append(m)
    pos = {id(u[0]): i for i, u in enumerate(units)}
    at = {id(m): i for i, m in enumerate(chrono)}

    def best(pool: list[Unit]) -> Unit | None:
        return max(pool, key=lambda u: (u[0].get("score") or 0, -pos[id(u[0])]), default=None)

    def third(u: Unit) -> int:
        return int(3 * at[id(u[0])] / n)

    views = [u for u in units if _cat(u) in ("lift", "scenery")]
    opener = best([u for u in views if third(u) == 0]) or (views[0] if views else None)
    finale = best([u for u in units if _cat(u) == "crash"]) or best(
        [u for u in units if _cat(u) in ("run", "jump")]
    )
    outro = best([u for u in views if _cat(u) == "scenery" and third(u) == 2 and u is not opener])
    fixed = [u for u in (opener, finale, outro) if u is not None]
    body = [u for u in units if all(u is not f for f in fixed)]

    stretch: dict[int, Any] = {}  # unit -> its chapter, carried forward per source
    last: dict[Any, Any] = {}
    for u in body:
        ch = u[0].get("chapter") or last.get(u[0]["source"])
        last[u[0]["source"]] = ch
        stretch[id(u)] = ch or u[0]["source"]
    first: dict[Any, int] = {}
    for u in body:
        first.setdefault(stretch[id(u)], len(first))
    rank = {c: i for i, c in enumerate(ARC_GROUPS)}
    body.sort(
        key=lambda u: (first[stretch[id(u)]], rank.get(_cat(u) or "", len(rank)), pos[id(u[0])])
    )
    head = [opener] if opener else []
    seq, pending = list(head), body
    while pending:  # local swaps: the first clip ahead that keeps the streak short
        k = next(
            (
                i
                for i, u in enumerate(pending[:ARC_WINDOW])
                if _streak(seq, _cat(u)) + len(u) <= ARC_MAX_REPEAT
            ),
            0,
        )
        seq.append(pending.pop(k))
    if finale and _streak(seq, _cat(finale)) + len(finale) > ARC_MAX_REPEAT:
        now = _max_streak([*seq, finale])
        for j in range(len(seq) - 1, max(len(head) - 1, len(seq) - 1 - ARC_WINDOW), -1):
            cand = [*seq[:j], *seq[j + 1 :], seq[j]]
            if _cat(seq[j]) != _cat(finale) and _max_streak([*cand, finale]) < now:
                seq = cand  # never joins two streaks into a longer one
                break
    seq += [u for u in (finale, outro) if u]
    out = [m for u in seq for m in u]
    if opener and opener[0]["end"] - opener[0]["start"] > ARC_OPENER_S:
        o = dict(opener[0])
        o["start"] = round((o["start"] + o["end"] - ARC_OPENER_S) / 2, 3)
        o["end"] = round(o["start"] + ARC_OPENER_S, 3)
        o["duration"] = ARC_OPENER_S
        out[0] = o
    return out


def segment_source(source: str | Path, base: Path) -> tuple[str, Path]:
    """``(identity, read path)`` of a segments file's ``source``.

    A relative ``source`` is taken against ``base`` (the file's folder). The read path
    is its :func:`~cutpoint.config.remap_source`; the identity (recorded, matched) stays
    the original path, resolved, or only made absolute when the remap applies (then in
    UNC form on a mapped drive, :func:`~cutpoint.config.unc_path`, as ``analyze`` and a
    resolved reachable file record it, also while the drive is disconnected). A Windows
    path read elsewhere (``D:\\Videos\\a.insv``, as a project written on Windows records
    it) is absolute too; its identity is its Windows form, as Windows records it.
    """
    src = Path(source)
    if not src.is_absolute() and is_absolute_any(source):  # resolve() would prefix the cwd
        return ntpath.normpath(str(source)), remap_source(str(source))
    src = src if src.is_absolute() else (base / src)
    io_path = remap_source(src)
    if io_path == src:
        return str(src.resolve()), io_path
    # Remapped: abspath normalises '..'; Z:\ -> \\server\share (cached lookup).
    return unc_path(os.path.abspath(src)), io_path  # noqa: PTH100


def source_key(source: str | Path) -> str:
    """An absolute ``source`` as a lookup key, normalised as :func:`segment_source` records
    it (resolved, so through a symlink too), for matching exclusions to segments.

    A mapped drive is in UNC form (:func:`~cutpoint.config.unc_path`), so ``Z:\\F\\a.mp4``
    recorded while the drive was disconnected matches ``\\\\server\\share\\F\\a.mp4``.
    """
    return path_key(unc_path(segment_source(source, Path.cwd())[0]))


def drive_relative(source: str) -> str:
    """The drive of a drive-relative Windows ``source`` (``D:foo``, relative to drive D's
    current folder) on Windows, else ``""``."""
    if not _WINDOWS:
        return ""
    p = PureWindowsPath(source)
    return p.drive if p.drive and not p.root else ""


def drive_relative_error(source: str, drive: str, base: str = "the segments file") -> str:
    """Why a drive-relative ``source`` on ``drive`` is refused (``base``: what a relative
    path is taken against)."""
    return (
        f"source {source!r} is relative to the current folder of drive {drive}; give the "
        f"full path ({drive}\\...) or one relative to {base}"
    )


def _letters(k: int) -> str:
    """``k`` (1-based) as ``a``, ``b``, ..., ``z``, ``aa``, ``ab``, ..."""
    out = ""
    while k:
        k, r = divmod(k - 1, 26)
        out = chr(ord("a") + r) + out
    return out


def segment_ids(items: list[Any]) -> list[Any]:
    """The id of each segments-file entry: its own ``id``, else ``sNNN`` of its place
    in the list; a generated id another entry already has gets a suffix (``s002b``,
    ``s002c``, ...).

    Raises:
        ValueError: If two entries give the same ``id``.
    """
    given: dict[Any, int] = {}
    for i, item in enumerate(items, 1):
        if isinstance(item, dict) and "id" in item:
            try:
                j = given.setdefault(item["id"], i)
            except TypeError:  # a list or object
                raise ValueError(f"Segment {i}: id must be a string or a number") from None
            if j != i:
                raise ValueError(f"Segments {j} and {i} have the same id {item['id']!r}")
    taken = set(given)
    ids = []
    for i, item in enumerate(items, 1):
        if isinstance(item, dict) and "id" in item:
            ids.append(item["id"])
            continue
        sid, k = f"s{i:03d}", 1
        while sid in taken:
            k += 1
            sid = f"s{i:03d}{_letters(k)}"
        taken.add(sid)
        ids.append(sid)
    return ids


def read_json_file(path: Path, what: str = "segments file") -> Any:
    """JSON from the file ``path`` (a UTF-8 BOM is allowed); ``what`` names it in errors.

    Raises:
        IsADirectoryError: If ``path`` is a folder.
        FileNotFoundError: If ``path`` is missing.
        ValueError: If the file cannot be read, is not UTF-8 text or not valid JSON
            (see :func:`cutpoint.cache.read_json`).
    """
    if path.is_dir():
        raise IsADirectoryError(f"{path} is a folder, not a {what}")
    if not path.is_file():
        raise FileNotFoundError(f"{what[:1].upper()}{what[1:]} not found: {path}")
    return cache.read_json(path)


def load_segments(path: Path) -> list[dict[str, Any]]:
    """Load a hand-picked segment list: ``[{source, start, end, label}, ...]`` in play order.

    Relative ``source`` paths are resolved against the JSON file's folder.
    Optional per-segment keys: ``"kind"`` (the moment's dominant feature: 360
    director tweaks), ``"category"`` / ``"chapter"`` (``cutpoint mix``: ``arc``
    order), ``"tail"`` (seconds: extend the end by exactly this much, no
    detection), ``"smart_tail": false``
    (no smart/event tail), ``"speech_aware": false`` (keep both boundaries out
    of the speech-aware pass), ``"crop"`` (``auto``, ``none`` or ``W:H:X:Y``), ``"extend"`` /
    ``"extend_start"`` / ``"extend_end"`` (padding in seconds, overriding
    ``--extend*``) and ``"speech_max_extend"`` / ``"speech_max_lead"``
    (this segment's speech-aware caps, overriding ``--speech-max-*``) and
    ``"min_start"`` / ``"max_end"`` (source seconds the clip never extends past
    with lead-in, tails or padding; ``cutpoint 360 repair``).

    Raises:
        ValueError: If the file is malformed, a segment is invalid or two share an id.
        FileNotFoundError: If the file or a source video is missing.
        IsADirectoryError: If ``path`` is a folder.
    """
    items = read_json_file(path)
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    ids = segment_ids(items)
    moments = []
    for i, item in enumerate(items, 1):
        try:
            src = Path(item["source"])
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Segment {i} needs source, start and end: {item!r}") from None
        if drive := drive_relative(str(item["source"])):  # Windows: against drive's current folder
            raise ValueError(f"Segment {i}: {drive_relative_error(item['source'], drive)}")
        identity, io_path = segment_source(src, path.parent)
        if not io_path.is_file():
            raise FileNotFoundError(f"Segment {i}: source not found: {io_path}")
        if end <= start or start < 0:
            raise ValueError(f"Segment {i}: invalid range {start}-{end}")
        moment = {
            "id": ids[i - 1],
            "source": identity,
            "start": start,
            "end": end,
            "duration": round(end - start, 3),
            "score": item.get("score"),
            "label": str(item.get("label", "segment")),
        }
        for key in ("kind", "category"):
            if item.get(key):
                moment[key] = str(item[key])
        if isinstance(item.get("chapter"), int):
            moment["chapter"] = item["chapter"]
        moment.update(seconds_keys(item, ("tail",), f"Segment {i}"))
        for key in ("smart_tail", "speech_aware"):
            if key in item:
                if not isinstance(item[key], bool):
                    raise ValueError(f"Segment {i}: {key} must be true or false")
                moment[key] = item[key]
        if "crop" in item:
            parse_spec(item["crop"])  # validate early
            moment["crop"] = item["crop"]
        moment.update(seconds_keys(item, ("extend", "extend_start", "extend_end"), f"Segment {i}"))
        moment.update(boundaries.clip_limits(item, f"Segment {i}"))
        bounds = seconds_keys(item, BOUND_KEYS, f"Segment {i}", allow_negative=True)
        for key, val in bounds.items():
            if val > start if key == "min_start" else val < end:
                raise ValueError(f"Segment {i}: {key} must not cut into {start}-{end}")
        moment.update(bounds)
        moments.append(moment)
    return moments


def _tail_limits(ordered: list[dict[str, Any]]) -> list[float | None]:
    """For each moment, the start of the next selected moment in the same source.

    A tail must not run into footage that another clip already shows.
    """
    limits: list[float | None] = []
    for m in ordered:
        later = [
            o["start"]
            for o in ordered
            if o is not m and o["source"] == m["source"] and o["start"] >= m["end"] - 0.01
        ]
        limits.append(min(later) if later else None)
    return limits


def _resolve_crop(
    m: dict[str, Any],
    src: Path,
    info: MediaInfo,
    cli_crop: str | None,
    project_cfg: dict[str, Any] | None,
    detected: dict[Path, Crop | None],
) -> tuple[Crop | None, str]:
    """Crop for one moment and where it came from.

    Precedence: segment ``"crop"`` > ``--crop`` > project.json > none.
    """
    if "crop" in m:
        spec, origin = m["crop"], "segment"
    elif cli_crop is not None:
        spec, origin = cli_crop, "cli"
    else:
        spec = project_crop(project_cfg, Path(m["source"]))  # recorded path, not remap
        origin = "project.json" if spec is not None else "default"
    parsed = parse_spec(spec)
    if parsed == "auto":
        if src not in detected:
            detected[src] = detect(src, info, sources.readable_end(src, 0.0, info.duration))
        return detected[src], f"auto ({origin})"
    if isinstance(parsed, Crop) and (
        parsed.x + parsed.w > info.width or parsed.y + parsed.h > info.height
    ):
        raise ValueError(f"Crop {parsed} does not fit {src.name} ({info.width}x{info.height})")
    return parsed, origin


@dataclass
class ClipPlan:
    """One clip of a ``cutpoint clips`` run, ready to render (see :func:`plan_clips`).

    Attributes:
        name: Output file name (``NNN_label.mp4``).
        entry: The moment (highlights.json candidate or segment).
        src: Source file actually read (``source_remap`` applied).
        info: Probe of ``src``.
        start: In point (absolute source time; padded, speech-adjusted).
        end: Out point.
        keep: The picked moment after tails, before padding.
        tail: Smart / event tail decision.
        payoffs: Payoff events covered by the event tail.
        pad_rec: Padding record (manifest ``extend``); None without padding.
        speech: Speech-pass record (manifest ``speech``); None if not looked at.
        pano: Rendered by the 360 exporter (never cropped).
        crop: Crop box applied before scaling (None: no crop).
        crop_source: Where the crop came from (manifest ``crop_source``).
    """

    name: str
    entry: dict[str, Any]
    src: Path
    info: MediaInfo
    start: float
    end: float
    keep: tuple[float, float]
    tail: TailResult
    payoffs: list[str] = field(default_factory=list)
    pad_rec: dict[str, Any] | None = None
    speech: dict[str, Any] | None = None
    pano: bool = False
    crop: Crop | None = None
    crop_source: str = "default"


@dataclass(frozen=True)
class RenderResult:
    """What rendering one :class:`ClipPlan` produced.

    Attributes:
        out: The encoded clip file.
        end: Final out point (earlier than planned after a read error).
        read_error: Read-error record (manifest ``read_error``), or None.
        pano: 360 render record (manifest ``pano``), or None.
        srt: Name of the subtitles written next to the clip (manifest ``srt``), or None.
    """

    out: Path
    end: float
    read_error: dict[str, Any] | None = None
    pano: dict[str, Any] | None = None
    srt: str | None = None


def plan_clips(
    ordered: list[dict[str, Any]],
    project: Path,
    *,
    io_paths: dict[str, Path],
    infos: dict[str, MediaInfo],
    tail: TailSettings,
    events: dict[str, list[dict[str, Any]]] | None,
    event_settings: EventTailSettings,
    speech: boundaries.SpeechSettings,
    padding: Padding | None = None,
    crop: str | None = None,
    project_cfg: dict[str, Any] | None = None,
    pano: PanoExporter | None = None,
    language: str | None = None,
) -> list[ClipPlan]:
    """Plan the clips of ``ordered`` moments: names, tails, padding, speech, crops.

    Reads cached analysis data (transcripts, audio, read errors) and, for an
    ``auto`` crop, a few source frames; writes no clips.

    Args:
        ordered: Moments in output order.
        project: Project folder (events for the speech maps).
        io_paths: ``{recorded source: file to read}``.
        infos: ``{recorded source: probe}``.
        tail: Smart-tail settings of the run (a moment's ``tail`` /
            ``smart_tail`` override them).
        events: The project's events.json by source, or None (no event tails).
        event_settings: Event-tail settings.
        speech: Speech-aware boundary settings.
        padding: Padding for every clip (a moment's ``extend*`` override it).
        crop: ``--crop`` spec for every source (None: segment / project.json).
        project_cfg: The project's project.json.
        pano: 360 exporter; the moments it handles are not cropped.
        language: Transcript language the analysis forced (None: the detected one).

    Returns:
        One :class:`ClipPlan` per moment, in order.

    Raises:
        ValueError: If a moment is outside its source or a crop does not fit.
    """
    words_cache: dict[Path, list[dict[str, Any]] | None] = {}
    plans: list[ClipPlan] = []
    for idx, (m, nxt) in enumerate(zip(ordered, _tail_limits(ordered), strict=True), 1):
        name = f"{idx:03d}_{slugify(m.get('label', ''), 40)}.mp4"
        src = io_paths[m["source"]]
        info = infos[m["source"]]
        start, end = m["start"], min(m["end"], info.duration)
        if end - start < MIN_EXPORT_S:
            raise ValueError(f"{m['id']}: {start:.2f}-{m['end']:.2f} is outside {src.name}")
        limit = sources.readable_end(src, start, info.duration)
        if nxt is not None:
            limit = min(limit, nxt)
        limit = min(limit, m.get("max_end", limit))

        settings = tail
        if m.get("smart_tail") is False:
            settings = TailSettings(enabled=False)
        if "tail" in m:
            settings = TailSettings(fixed=m["tail"])
        if src not in words_cache and settings.enabled and settings.fixed is None:
            words_cache[src] = sources.source_words(src, language)

        def audio(t0: float, t1: float, s: Path = src, i: MediaInfo = info):
            return sources.audio_window(s, i, t0, t1)

        res = smart_tail(start, end, words_cache.get(src), audio, settings, max(limit, end))
        payoffs: list[str] = []
        if event_settings.enabled and m.get("smart_tail") is not False and "tail" not in m:
            if src not in words_cache:
                words_cache[src] = sources.source_words(src, language)
            ev_end, ev_reason, payoffs = event_tail(
                start,
                res.end,
                words_cache.get(src),
                events.get(str(Path(m["source"]))) if events else None,
                event_settings,
                max(limit, end),
            )
            if ev_reason:
                res.end, res.reason = ev_end, ev_reason
                res.extension = round(ev_end - end, 3)
        end = res.end

        keep_start, keep_end = start, end
        seg_pad = (padding or Padding()).override(m)
        pad_rec = None
        if seg_pad.active:
            start, end, pad_rec = pad(src, info, start, end, seg_pad, m)
        plans.append(
            ClipPlan(name, m, src, info, start, end, (keep_start, keep_end), res, payoffs, pad_rec)
        )

    speech_recs = speech_pass(
        plans,
        speech,
        project,
        fit_end=lambda e: "tail" not in e,
        words_cache=words_cache,
        language=language,
    )
    detected: dict[Path, Crop | None] = {}
    for plan in plans:
        plan.speech = speech_recs.get(plan.name)
        if pano is not None and pano.handles(plan.entry["source"]):
            plan.pano, plan.crop_source = True, "360"
        else:
            plan.crop, plan.crop_source = _resolve_crop(
                plan.entry, plan.src, plan.info, crop, project_cfg, detected
            )
    return plans


def export_workers(plans: list[ClipPlan]) -> int:
    """Clips :func:`render_clips` exports at once (#464): ``export_workers`` in config.json, or
    for auto 2 with a GPU encoder and 1 with libx264, which already uses every core
    (:func:`~cutpoint.ffmpeg_utils.video_encoder`, chosen here once instead of by every worker;
    an encoder that cannot be chosen leaves its error to the first export, exported alone).
    1 for fewer than two clips or a run with a 360 clip.
    """
    n = configured_workers()
    if len(plans) < 2 or any(p.pano for p in plans) or n == 1:
        return 1
    try:
        cpu = video_encoder().kind == "cpu"
    except FFmpegError:
        return 1
    return n or (1 if cpu else 2)


class _ExportPool:
    """Runs the encodes of :func:`render_clips` on ``workers`` threads (#464).

    Encodes start in plan order and their results are taken in plan order, so names, logs,
    subtitles and manifest.json are those of a sequential export; each clip gets the same
    ffmpeg command. ffmpeg runs in a :class:`~cutpoint.ffmpeg_utils.ChildScope`: the first
    failure kills the encodes in flight, starts no more and is raised. On Ctrl+C or Ctrl+Break
    (KeyboardInterrupt in this thread; ffmpeg got the same event) the running encodes get
    ``SOFT_STOP_S`` to exit and are then killed (a second Ctrl+C kills them at once), and every
    worker has removed its partial file before the exception leaves, as sequentially.
    """

    def __init__(self, jobs: list[Callable[[], dict[str, Any] | None]], workers: int) -> None:
        self.scope = ChildScope()
        self.failure: BaseException | None = None
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(workers, "cutpoint-export")
        self.futures = [self._pool.submit(self._run, job) for job in jobs]

    def _run(self, job: Callable[[], dict[str, Any] | None]) -> dict[str, Any] | None:
        with child_scope(self.scope):
            self.scope.check()  # not started after a failure or Ctrl+C
            try:
                return job()
            except BaseException as err:
                if not isinstance(err, KeyboardInterrupt):
                    self.scope.wait_cancelled(STOP_GRACE_S)  # the error of a Ctrl+C is no failure
                with self._lock:  # what a stop causes in the other encodes is no failure
                    first = self.failure is None and not self.scope.cancelled
                    if first:
                        self.failure = err
                if first:
                    self.scope.kill()
                raise

    def result(self, i: int) -> dict[str, Any] | None:
        """The read-error record of encode ``i`` (see :func:`export_clip_checked`).

        Raises:
            BaseException: The first failure of any encode; a KeyboardInterrupt of this
                thread (Ctrl+C) as it is, never replaced by a failure it caused in an encode.
        """
        f = self.futures[i]
        while self.failure is None and not f.done():
            wait([f], POLL_S)
        if self.failure is not None:
            raise self.failure
        return f.result()

    def __enter__(self) -> _ExportPool:
        return self

    def __exit__(self, kind: object, err: BaseException | None, tb: object) -> None:
        """Stop what still runs after a failure or Ctrl+C, then wait for every worker."""
        deadline = None
        if isinstance(err, KeyboardInterrupt):
            self.scope.cancel()  # ffmpeg got the same Ctrl+C: let it exit first
            deadline = time.monotonic() + SOFT_STOP_S
        elif err is not None:
            self.scope.kill()
        for f in self.futures:
            f.cancel()  # not started yet
        while not all(f.done() for f in self.futures):
            try:
                wait(self.futures, POLL_S)
                if deadline is not None and time.monotonic() > deadline:
                    self.scope.kill()
            except KeyboardInterrupt:
                self.scope.kill()
        self._pool.shutdown()


def _warm_source_checks(plans: list[ClipPlan]) -> None:
    """Run the once-per-source checks of an export (the audio track comparison and the
    log/flat footage check) on this thread, before the encodes run in parallel (#464), so
    each runs and logs once and a Ctrl+C stops them as in a sequential export. The color
    conversion notes of :func:`~cutpoint.encode.bt709_filter` are logged here too, once."""
    seen: set[Path] = set()
    for plan in plans:
        if plan.src in seen:
            continue
        seen.add(plan.src)
        warn_if_log(plan.src, plan.info, plan.start)
        bt709_filter(plan.info)
        if plan.info.audio_streams > 1:
            export_tracks(
                plan.src, plan.info.audio_streams, plan.info.audio_duration or plan.info.duration
            )


def render_clips(
    plans: list[ClipPlan],
    out_dir: Path,
    *,
    fmt: OutputFormat,
    vertical: bool,
    audio_fade: float,
    pano: PanoExporter | None = None,
    protected_srts: set[str] | None = None,
    clips_dir: Path | None = None,
    peaks: list[str] | None = None,
    language: str | None = None,
    grades: list[Grade | None] | None = None,
    workers: int = 1,
) -> list[RenderResult]:
    """Encode every planned clip into ``out_dir`` (360 clips through ``pano``), each with
    its ``.srt`` subtitles from the cached source transcript (none, with one INFO line,
    for sources without one; see :mod:`cutpoint.srt`).

    ``peaks``: per-plan peak compression filter (:func:`peak_filters`; 360 clips get none).

    ``protected_srts``: :func:`~cutpoint.config.path_key` of ``.srt`` names in the clips
    folder that an earlier run did not write; a clip with such a name gets no ``.srt``
    (one WARNING names them). ``clips_dir``: that clips folder (default ``out_dir``; the
    staging folder is not it).

    ``language``: transcript language of the subtitles (None: the detected one).

    ``grades``: per-plan grade (:class:`~cutpoint.encode.Grade`, None for none), 360 clips
    included.

    ``workers``: clips encoded at once (:func:`export_workers`; 360 plans are always encoded
    one at a time); the results, logs and subtitles stay in plan order.

    Returns:
        One :class:`RenderResult` per plan, in order.

    Raises:
        ValueError: If a 360 plan is given without ``pano``.
    """
    results: list[RenderResult] = []
    subs = srt.ClipSubtitles(protected_srts, language, clips_dir or out_dir)
    peaks = peaks or [""] * len(plans)
    grades = grades or [None] * len(plans)

    def encode(plan: ClipPlan, peak: str, grade: Grade | None) -> dict[str, Any] | None:
        return export_clip_checked(
            plan.src,
            plan.info,
            plan.start,
            plan.end,
            out_dir / plan.name,
            fmt,
            vertical,
            plan.crop,
            audio_fade,
            peak,
            grade,
        )

    pool = None
    if workers > 1 and not any(p.pano for p in plans):
        _warm_source_checks(plans)
        jobs = [
            lambda p=p, k=k, g=g: encode(p, k, g)
            for p, k, g in zip(plans, peaks, grades, strict=True)
        ]
        pool = _ExportPool(jobs, workers)
    with pool or nullcontext():
        for i, (plan, peak, grade) in enumerate(zip(plans, peaks, grades, strict=True)):
            m, start, end, tail = plan.entry, plan.start, plan.end, plan.tail
            out = out_dir / plan.name
            pano_rec = None
            if plan.pano:
                if pano is None:
                    raise ValueError(f"{plan.name}: 360 plan needs a PanoExporter")
                read_err, pano_rec = pano.export(
                    m["source"],
                    start,
                    end,
                    out,
                    audio_fade=audio_fade,
                    kind=m.get("kind"),
                    grade=grade,
                )
            else:
                read_err = pool.result(i) if pool is not None else encode(plan, peak, grade)
            if read_err and read_err["end"] < end:
                end = read_err["end"]
            logger.info(
                "  %s  <- %s [%.2f-%.2f] tail +%.2fs (%s)%s%s",
                plan.name,
                plan.src.name,
                start,
                end,
                max(0.0, tail.extension),
                tail.reason
                + (", trimmed by read error" if read_err and read_err["end"] < tail.end else ""),
                f" pad -{plan.pad_rec['applied']['start']:.2f}/+{plan.pad_rec['applied']['end']:.2f}"
                if plan.pad_rec
                else "",
                f" crop {plan.crop}" if plan.crop else "",
            )
            sub = subs.write(out, m["source"], start, end)
            results.append(RenderResult(out, end, read_err, pano_rec, sub))
    subs.report(len(plans))
    return results


def peak_filters(plans: list[ClipPlan], preset: str) -> tuple[list[str], list[float | None]]:
    """Per-plan :meth:`~cutpoint.encode.PeakPlan.filter` for ``preset`` (all "" for ``off``)
    and each one's makeup gain (dB, None for a clip left alone).

    Each clip's span of its source is measured with the tracks the export mixes, then again
    through its gain and compressor for the makeup (:func:`~cutpoint.encode.plan_makeup`).
    360 clips are rendered by the 360 exporter and are not compressed (one INFO line says so).
    """
    none: list[float | None] = [None] * len(plans)
    if preset == "off":
        return [""] * len(plans), none
    flat = [not p.pano for p in plans]
    if not all(flat):
        logger.info("Compress peaks: %d 360 clip(s) are not compressed", flat.count(False))
    series = [
        clip_momentary(p.src, p.info.audio_streams, p.start, p.end - p.start) if f else []
        for p, f in zip(plans, flat, strict=True)
    ]
    plan = plan_peaks(preset, series, [p.end - p.start for p in plans], flat)
    if plan is None:
        logger.info("Compress peaks: no clip has audio above silence")
        return [""] * len(plans), none
    compressed = [
        clip_momentary(p.src, p.info.audio_streams, p.start, p.end - p.start, chain)
        if (chain := plan.measure_chain(i))
        else []
        for i, p in enumerate(plans)
    ]
    plan = plan_makeup(plan, series, compressed)
    logger.info(
        "Compress peaks (%s): compressor from %.1f dB RMS; clip gains %s; makeup %s",
        preset,
        plan.threshold,
        " ".join("-" if g is None else f"{g:+.1f}" for g in plan.gains),
        " ".join("-" if m is None else f"{m:+.1f}" for m in plan.makeup),
    )
    return [plan.filter(i) for i in range(len(plans))], list(plan.makeup)


def _entry(
    plan: ClipPlan,
    res: RenderResult,
    audio_fade: float,
    peaks: str = "",
    makeup: float | None = None,
    size: int | None = None,
    grade: Grade | None = None,
) -> dict[str, Any]:
    """manifest.json record of one rendered clip (``peaks``: its :func:`peak_filters` preset,
    recorded as ``compress_peaks`` when the clip got compression, with its ``makeup`` gain as
    ``compress_peaks_makeup_db``; ``size``: the clip's size in bytes, see
    :func:`manifest_entries`; ``grade``: its grade, recorded with its ``to_json()`` keys, such as
    ``lut`` and ``lut_strength``)."""
    m, src, start, end = plan.entry, plan.src, plan.start, res.end
    return {
        "file": plan.name,
        "size": size,
        "id": m["id"],
        "source": m["source"],
        **({"read_from": str(src)} if str(src) != m["source"] else {}),
        "start": start,
        "end": end,
        "duration": round(end - start, 3),
        "original_end": m["end"],
        "pick": {"start": round(plan.keep[0], 3), "end": round(plan.keep[1], 3)},
        "tail": plan.tail.to_json() | ({"payoff_events": plan.payoffs} if plan.payoffs else {}),
        "crop": str(plan.crop) if plan.crop else None,
        "crop_source": plan.crop_source,
        **(grade.to_json() if grade else {}),
        "score": m["score"],
        "label": m.get("label", ""),
        **({"extend": plan.pad_rec} if plan.pad_rec else {}),
        **({"speech": plan.speech} if plan.speech is not None else {}),
        **({"speech_aware": False} if m.get("speech_aware") is False else {}),
        **{k: m[k] for k in (*boundaries.LIMIT_KEYS, *BOUND_KEYS) if m.get(k) is not None},
        "audio_fade": audio_fade,
        **({"compress_peaks": peaks} if peaks else {}),
        **({"compress_peaks_makeup_db": makeup or 0.0} if peaks else {}),
        **({"kind": m["kind"]} if m.get("kind") else {}),
        **({"category": m["category"]} if m.get("category") else {}),
        **({"pano": res.pano} if res.pano else {}),
        **({"read_error": res.read_error} if res.read_error else {}),
        **({"srt": res.srt} if res.srt else {}),
    }


def _smart_tail_used(tail: TailSettings, moments: list[dict[str, Any]]) -> bool:
    """Whether any of ``moments`` gets a smart tail: the run's setting is on and at least one
    moment did not turn it off (``"smart_tail": false`` or a fixed ``"tail"``)."""
    return tail.enabled and any(
        m.get("smart_tail") is not False and "tail" not in m for m in moments
    )


def write_manifest(
    out_dir: Path,
    project: Path,
    plans: list[ClipPlan],
    results: list[RenderResult],
    *,
    order: str,
    fmt: OutputFormat,
    vertical: bool,
    style: Style,
    tail: TailSettings,
    event_settings: EventTailSettings,
    padding: Padding | None,
    speech: boundaries.SpeechSettings,
    audio_fade: float,
    crop: str | None,
    compress_peaks: str = "off",
    peaks: list[str] | None = None,
    makeup: list[float | None] | None = None,
    grades: list[Grade | None] | None = None,
) -> None:
    """Write ``<out_dir>/manifest.json`` for the rendered clips (and log the tail total).

    ``compress_peaks`` is recorded when not ``off``, and on each clip ``peaks`` gave a filter,
    with its ``makeup`` (:func:`peak_filters`). ``grades``: each clip's grade, recorded on it.
    """
    total_ext = 0.0
    for plan in plans:
        total_ext += plan.tail.extension
    for plan, res in zip(plans, results, strict=True):
        if res.read_error and res.read_error["end"] < plan.end:  # trimmed by a read error
            total_ext -= plan.end - max(res.end, min(plan.end, plan.entry["end"]))
    peaks = peaks or [""] * len(plans)
    makeup = makeup or [None] * len(plans)
    grades = grades or [None] * len(plans)
    entries = [
        _entry(p, r, audio_fade, compress_peaks if f else "", m, file_size(out_dir / p.name), g)
        for p, r, f, m, g in zip(plans, results, peaks, makeup, grades, strict=True)
    ]
    logger.info(
        "Smart tail added %.1fs over %d of %d clips",
        total_ext,
        sum(1 for e in entries if e["tail"]["extension"] > 0),
        len(entries),
    )
    padding = padding or Padding()
    # The header says what was applied: off when every clip turned it off ("smart_tail": false
    # or a fixed "tail"; "speech_aware": false), as a sequence's segments can.
    tails_on = _smart_tail_used(tail, [p.entry for p in plans])
    speech_on = speech.enabled and any(p.entry.get("speech_aware") is not False for p in plans)
    cache.save_json(
        out_dir / "manifest.json",
        {
            "generator": MANIFEST_GENERATOR,
            "created": datetime.now().isoformat(timespec="seconds"),
            "project": str(project.resolve()),
            "order": order,
            "format": {
                "width": fmt.width,
                "height": fmt.height,
                "fps": fmt.fps,
                "vertical": vertical,
            },
            "smart_tail": {
                "style": style.name,
                "enabled": tails_on,
                "tail_min": tail.tail_min,
                "tail_max": tail.tail_max,
                "settle": tail.settle,
            },
            "event_tails": event_settings.to_json(),
            "extend": {"start": padding.start, "end": padding.end},
            "speech_aware": speech.to_json() | {"enabled": speech_on},
            "audio_fade": audio_fade,
            **({"compress_peaks": compress_peaks} if compress_peaks != "off" else {}),
            "crop": crop,
            "clips": entries,
        },
    )


def _replaced_files(
    out_dir: Path, new: list[str], sources: set[Any]
) -> tuple[list[str], list[str], list[str], bool, set[str], dict[str, str]]:
    """What ``--force`` replaces in ``out_dir`` besides the clip names ``new`` this run writes.

    Only files an earlier run made are replaced: the clips and their ``.srt`` subtitles
    listed in the old manifest.json and leftover ``.<name>.part.mp4`` files. A file this
    run reads (``sources``, see :func:`source_keys`) is never replaced; neither is any
    other ``.mp4``, so a name in ``new`` that is such a file stops the run
    (:func:`~cutpoint.srt.check_unlisted_clips`). Every other ``.srt`` (all of them without
    a readable manifest.json) is protected: :func:`render_clips` writes no subtitles under
    its name. A manifest.json cutpoint did not write stops the run
    (:func:`check_own_manifest`).

    Returns:
        ``(names to replace, .mp4 kept as sources of this run, other .mp4 kept, whether
        the old manifest.json was readable, path_key of the protected .srt names, the
        exported name of each kept clip renamed during review)``.

    Raises:
        ValueError: If a name in ``new`` is a source of this run (before any export).
        FileExistsError: If a name in ``new`` is another ``.mp4`` the old manifest.json
            does not list, or that manifest.json is not one cutpoint wrote (before any
            export).
    """
    check_not_sources(out_dir, new, sources)
    if not out_dir.is_dir():
        return [], [], [], True, set(), {}
    check_own_manifest(out_dir)
    listed = srt.listed_files(out_dir)
    srt.check_unlisted_clips(out_dir, new, listed)
    if listed is None and any(out_dir.iterdir()):
        logger.warning(
            "%s has no readable manifest.json: --force adds the clips this run writes and "
            "keeps every other file",
            out_dir,
        )
    parts = [p.name for p in out_dir.glob(".*.part.mp4")]
    # Only a name spelled on disk as listed: a clip renamed in case only still matches its
    # listed name on a case-insensitive file system and must stay (kept, with a warning).
    # A listed name spelled otherwise on disk goes, under that spelling, only if this run
    # writes or replaces its clip: an .srt beside it would keep the old cues, unlisted.
    present = {p.name for p in out_dir.iterdir()}
    on_disk = {path_key(n, out_dir): n for n in present}
    names = list(dict.fromkeys([*(listed or []), *parts]))
    replace: list[str] = []
    kept_sources: list[str] = []

    def add(n: str) -> None:
        f = out_dir / n
        if f.is_file() or f.is_symlink():
            (kept_sources if same_file(f, sources) else replace).append(n)

    for n in names:
        if n in present:
            add(n)
    gone = {path_key(n, out_dir) for n in (*new, *replace)}
    for n in names:
        if (
            n not in present
            and (d := on_disk.get(path_key(n, out_dir))) is not None
            and path_key(Path(n).with_suffix(".mp4").name, out_dir) in gone
        ):
            add(d)
    done = {path_key(n, out_dir) for n in (*new, *replace, *kept_sources)}
    kept: list[str] = []
    for p in sorted(clip_files(out_dir)):
        if path_key(p.name, out_dir) not in done:
            (kept_sources if same_file(p, sources) else kept).append(p.name)
    entries = manifest_entries(out_dir, [out_dir / n for n in kept])
    renamed = {n: e["file"] for n, e in entries.items() if e["file"] != n}
    return (
        replace,
        kept_sources,
        kept,
        listed is not None,
        srt.unlisted_srts(out_dir, listed),
        renamed,
    )


# The option that made each manifest.json grade key (:data:`~cutpoint.encode.GRADE_KEYS`).
_GRADE_OPTIONS = {"lut": "--lut", "color": "--color"}


def _graded_keys(entry: Any) -> list[str]:
    """The :data:`~cutpoint.encode.GRADE_KEYS` a manifest entry was graded with (a ``color``
    record the clip got no grade from, ``skip`` set, does not count)."""
    if not isinstance(entry, dict):
        return []
    color = entry.get("color")
    return [
        k
        for k in GRADE_KEYS
        if entry.get(k) and not (k == "color" and isinstance(color, dict) and color.get("skip"))
    ]


def _warn_dropped_grade(out_dir: Path) -> None:
    """WARNING (and a ``grade_dropped`` --json warning) when the manifest.json ``--force``
    replaces graded clips (a ``lut`` or ``color`` key, :data:`~cutpoint.encode.GRADE_KEYS`)
    and this run grades none: the grade is not carried over, the flags of this run decide."""
    manifest = cache.load_json(out_dir / "manifest.json")
    listed = manifest.get("clips") if isinstance(manifest, dict) else None
    graded = [e for e in listed or [] if _graded_keys(e)]
    if graded:
        keys = sorted({k for e in graded for k in _graded_keys(e)})
        again = " or ".join(_GRADE_OPTIONS[k] for k in keys)
        message = (
            f"{out_dir}: {len(graded)} clip(s) of the replaced manifest.json were graded "
            f"({', '.join(keys)}); this run grades none, so the new clips are ungraded (give "
            f"{again} again to keep the grade)"
        )
        logger.warning("%s", message)
        jsonout.warn("grade_dropped", message, [out_dir / "manifest.json"])


def _color_mode(
    color: str | None, lut: Lut | None, style: Style, *, known_style: bool = True
) -> str:
    """The ``--color`` mode of a run: ``color`` as given, else ``color`` in config.json; the
    config's is never applied with a ``lut`` or to ``gaming`` footage (an INFO line says so),
    a given one on ``gaming`` footage gets a WARNING (a stronger one for ``underwater``) and a
    ``color_on_gaming`` ``--json`` warning. ``known_style`` False (no analysis style, as for
    segments without highlights.json): ``style`` is a fallback, not the footage's, so it never
    counts as gaming.

    Raises:
        jsonout.ArgumentError: For a ``color`` other than ``off`` with a ``lut``, or one not in
            :data:`~cutpoint.config.COLOR_MODES`.
    """
    gaming = known_style and style.base == "gaming"
    if color is None:
        mode = color_default()
        if mode == "off" or lut is not None:
            return "off"
        if gaming:
            logger.info(
                "color %s in config.json is not applied to gaming footage (style %s); "
                "give --color %s to grade it",
                mode,
                style.name,
                mode,
            )
            return "off"
        return mode
    if color not in COLOR_MODES:
        raise jsonout.ArgumentError(f"--color {color}: expected one of {', '.join(COLOR_MODES)}")
    if color != "off" and lut is not None:
        raise jsonout.ArgumentError(f"--color {color} cannot be combined with --lut; use one")
    if color != "off" and gaming:
        if color == "underwater":
            message = (
                f"--color underwater on gaming footage (style {style.name}): it is made for "
                "video shot underwater and grades game graphics too, so the clips may look "
                "worse (clips that do not look underwater stay ungraded, not_underwater); "
                "--color auto or a LUT (--lut) fits games better"
            )
        else:
            message = (
                f"--color {color} on gaming footage (style {style.name}): game graphics are "
                "graded too; check the clips"
            )
        logger.warning("%s", message)
        jsonout.warn("color_on_gaming", message)
    return color


# Plain words after the skip code of a clip :func:`_color_grades` does not grade.
_SKIP_HINTS = {
    SKIP_LOG: "looks like log/flat footage; use --lut with the camera's LUT",
    SKIP_NOT_UNDERWATER: (
        "it does not look like murky or color-cast underwater footage; --color auto may suit it"
    ),
}


def _color_grades(
    plans: list[ClipPlan], mode: str, strength: float, fmt: OutputFormat, vertical: bool
) -> list[Grade | None]:
    """Each plan's automatic grade (:func:`~cutpoint.color.auto_grade`, measured through the
    plan's crop and ``vertical`` as its export), 360 clips skipped
    (:data:`~cutpoint.color.SKIP_360`). An INFO line names each clip not graded with its skip
    code (and a hint, :data:`_SKIP_HINTS`: ``--lut`` for log/flat clips, ``--color auto`` for
    ``not_underwater``); for ``auto`` another names the clips that look underwater, and a last
    one sums them up."""
    logger.info(
        "Measuring the color of %d clip(s) (--color %s, strength %g)", len(plans), mode, strength
    )
    grades: list[AutoGrade] = []
    for plan in plans:
        if plan.pano:
            grade = AutoGrade.skipped(mode, SKIP_360, strength)
        else:
            grade = auto_grade(
                plan.src,
                plan.info,
                plan.start,
                plan.end,
                mode,
                fmt,
                strength=strength,
                vertical=vertical,
                crop=plan.crop,
            )
        logger.debug("  %s: color %s", plan.name, grade.to_json()["color"])
        grades.append(grade)
    names = [p.name for p in plans]
    for name, g in zip(names, grades, strict=True):
        if g.skip:
            why = _SKIP_HINTS.get(g.skip)
            logger.info("Color: %s not graded (%s)%s", name, g.skip, f": {why}" if why else "")
    if hint := [n for n, g in zip(names, grades, strict=True) if g.underwater_hint]:
        logger.info(
            "Color: %s look underwater; --color underwater corrects underwater colors "
            "(--color auto does not)",
            ", ".join(hint),
        )
    skipped: dict[str, int] = {}
    for g in grades:
        if g.skip:
            skipped[g.skip] = skipped.get(g.skip, 0) + 1
    logger.info(
        "Color %s: %d of %d clips graded%s",
        mode,
        sum(1 for g in grades if g.filter()),
        len(grades),
        "; not graded: " + ", ".join(f"{k} {n}" for k, n in skipped.items()) if skipped else "",
    )
    return list(grades)


def check_free_space(folder: Path, plans: list[ClipPlan], fmt: OutputFormat) -> None:
    """Refuse a temporary export of ``plans`` at ``fmt`` into ``folder`` when its drive has too
    little free space for the clips, the final video and its staging copy
    (:data:`TEMP_COPIES`, :data:`CLIP_BYTES_PER_S_1080P`) plus :data:`TEMP_SPARE_GB`, as
    ``360 stitch`` checks before a stitch. An unknown free space is not checked.

    Raises:
        ValueError: Naming the free and the needed space.
    """
    from cutpoint.pano.stitch import free_gb  # noqa: PLC0415

    free = free_gb(folder)
    if free is None:
        return
    seconds = sum(p.end - p.start for p in plans)
    fps = max(1.0, float(Fraction(fmt.fps)) / 30)
    rate = CLIP_BYTES_PER_S_1080P * fmt.width * fmt.height / (1920 * 1080) * fps
    need = seconds * rate * TEMP_COPIES / 1e9
    logger.debug("Temporary clips: %.1f GB free, about %.1f GB needed", free, need)
    if free - need < TEMP_SPARE_GB:
        raise ValueError(
            f"{folder}: {free:.1f} GB free, the temporary clips and the final video need about "
            f"{need:.1f} GB (and {TEMP_SPARE_GB:g} GB to spare)"
        )


@contextmanager
def _claimed(out_dir: Path, force: bool) -> Iterator[None]:
    """Hold the lock of ``out_dir`` (:func:`swap_lock`: one run at a time, as another would
    remove this one's staging folder), finish a hard-killed earlier replace there
    (:func:`recover_swap`), then refuse a non-empty ``out_dir`` without ``force``.

    Raises:
        FolderBusyError: If another run is using ``out_dir``.
        FileExistsError: If ``out_dir`` is not empty and ``force`` is False.
    """
    with swap_lock(out_dir):
        recover_swap(out_dir)  # before the folder is looked at
        if not force and out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"{out_dir} is not empty; use --force to replace its clips")
        yield


def export_clips(
    project: Path,
    *,
    top: int | None = None,
    ids: list[str] | None = None,
    segments: Path | None = None,
    order: str | None = None,
    max_total: float | None = None,
    out_dir: Path | None = None,
    vertical: bool = False,
    force: bool = False,
    crop: str | None = None,
    smart_tail_on: bool | None = None,
    tail_min: float | None = None,
    tail_max: float | None = None,
    event_tails: bool | None = None,
    event_window: float | None = None,
    extend: Padding | None = None,
    speech_aware: bool | None = None,
    speech_max_extend: float | None = None,
    speech_max_lead: float | None = None,
    audio_fade: float = DEFAULT_AUDIO_FADE_S,
    pano_source: str = "auto",
    pano_cmd_every: int | None = None,
    size: tuple[int, int] | None = None,
    compress_peaks: str = "off",
    lut: Lut | None = None,
    color: str | None = None,
    color_strength: float = 1.0,
    temporary: bool = False,
) -> Path:
    """Export moments as normalised, numbered clips plus manifest.json.

    Selection: ``segments`` (hand-picked JSON, exported exactly, in its order;
    ``order="arc"`` reorders a ``cutpoint mix`` file, see :func:`order_moments`),
    else ``ids``, else the top ``top`` by score, else the analysis'
    ``suggested_top``. ``max_total`` keeps the best-scoring moments whose
    summed length fits (e.g. 60 s for a Short). Candidates marked
    ``must_include`` (confident crashes, ``cutpoint crashes``) are always added to
    an automatic selection and kept under ``max_total``.

    Speech-ended clips get a smart tail (see :mod:`cutpoint.tail`) using the
    style's settings (segments: the project's analysis style, else gaming),
    overridden by ``smart_tail_on`` / ``tail_min`` / ``tail_max`` and by
    per-segment ``tail`` / ``smart_tail``.

    Event tails (see :mod:`cutpoint.events.tails`): when
    ``<project>/events.json`` exists (or ``event_tails`` is True), a clip whose
    speech is followed within ``event_window`` s (default 10) by a payoff event
    (player kill, teamkill, round banner, laughter, strong VLM moment) is
    extended to cover it plus 1.5 s. ``event_tails=False`` disables this; a
    per-segment ``"tail"`` or ``"smart_tail": false`` also disables it.

    ``crop``: ``auto``, ``none`` or ``W:H:X:Y`` for every source; None uses
    ``<project>/project.json``'s per-source ``crop`` map, else no crop.

    ``extend``: padding added to both ends of every clip *after* the tails
    (see :mod:`cutpoint.extend`), clamped to the source and its readable
    range; per-segment ``extend`` / ``extend_start`` / ``extend_end`` override
    it. The pre-pad in/out points are recorded.

    Speech-aware boundaries (see :mod:`cutpoint.boundaries`; default: the
    style's ``speech_aware``, on for every built-in style) run last: an in or
    out point inside an utterance moves to its start/end (at most
    ``speech_max_lead`` / ``speech_max_extend`` s), and clips of the same
    source are kept from overlapping. Per-segment ``"speech_aware": false``
    opts out; a fixed ``"tail"`` keeps its end. Without the pass, padded clips
    may overlap.

    ``audio_fade``: seconds of audio fade-in/out baked into every clip (0 = off).

    ``compress_peaks``: ``off``, ``light`` or ``strong``
    (:data:`~cutpoint.encode.PEAK_PRESETS`): loud peaks are compressed and the clips leveled
    toward each other, then a true-peak limiter ends each clip's audio (manifest
    ``compress_peaks``, so ``assemble`` does not compress them again).

    ``lut`` (:func:`~cutpoint.encode.check_lut`): every clip is graded with it after the
    BT.709 conversion (:func:`~cutpoint.encode.lut_filter`), recorded per clip as ``lut`` and
    ``lut_strength`` so ``extend`` grades the same way; ``assemble`` leaves the grade as is.

    ``color`` (``off``, ``auto`` or ``underwater``; None: ``color`` in config.json, never
    for ``gaming`` footage nor with ``lut``, see :func:`_color_mode`) at ``color_strength``
    (0-1): each clip gets its own automatic grade (:func:`~cutpoint.color.auto_grade`),
    recorded per clip as ``color`` (skipped clips too, with the reason) so ``extend`` grades
    the same way without measuring again. It cannot be combined with ``lut``.

    360 sources (Insta360 X3 clips registered in ``<project>/pano.json`` by
    ``cutpoint 360 scan``) are rendered to flat video along a director path (see
    :mod:`cutpoint.pano.render`): ``pano_source`` picks the input
    (``auto``: SDK stitch > Studio export > raw dual fisheye) and
    ``pano_cmd_every`` the camera update interval in frames. Everything else
    (numbering, tails, padding, speech-aware boundaries, fades) is unchanged.

    ``size`` ``(width, height)`` forces the output resolution instead of the
    largest source's (useful when 16:9 and taller sources such as GoPro 8:7
    are mixed; give the taller ones a ``crop`` so they fill the frame).

    ``temporary``: ``out_dir`` is a :func:`cutpoint.assemble.temporary_clips` folder, joined
    and removed right after: the free space for it and the final video is checked before any
    export (:func:`check_free_space`), and ``--json`` gets no clip outputs.

    Returns:
        The clips directory.

    Raises:
        FileNotFoundError: If highlights.json / segments / a source is missing.
        FileExistsError: If the clips folder is non-empty and ``force`` is False.
        ValueError: If the selection or a crop is invalid, or (``temporary``) the drive has
            too little free space.
        jsonout.ArgumentError: For ``color`` with ``lut``, or ``color_strength`` outside 0-1.
    """
    if not 0 <= color_strength <= 1:
        raise jsonout.ArgumentError(f"--color-strength {color_strength:g}: must be from 0 to 1")
    if crop is not None:
        try:
            parse_spec(crop)  # fail before any work
        except ValueError as err:
            raise jsonout.ArgumentError(str(err)) from None
    highlights = cache.load_json(project / "highlights.json")
    language = ((highlights or {}).get("settings") or {}).get("language")
    known_style = isinstance(highlights, dict) and "style" in highlights
    style_name = (highlights or {}).get("style", "gaming")
    if segments is not None:
        ordered = load_segments(segments)
        if order == "arc":  # the file's order is the day's order
            ordered = order_moments(index_sources(ordered), "arc")
        else:
            order = "segments"
    else:
        chosen, style_name = select_moments(project, top=top, ids=ids, max_total=max_total)
        order = order or get_style(style_name).default_order
        ordered = order_moments(chosen, order)
    style: Style = get_style(style_name)
    color_mode = _color_mode(color, lut, style, known_style=known_style)
    base_tail = TailSettings.from_style(
        style, enabled=smart_tail_on, tail_min=tail_min, tail_max=tail_max
    )
    speech = boundaries.SpeechSettings.from_style(
        style, enabled=speech_aware, max_extend=speech_max_extend, max_lead=speech_max_lead
    )
    project_events = load_events(project) if event_tails is not False else None
    if event_tails and project_events is None:
        logger.warning(
            "--event-tails: %s has no events.json (run `cutpoint events`); "
            "continuing without event tails",
            project,
        )
    ev_settings = EventTailSettings(
        enabled=project_events is not None, **({"window": event_window} if event_window else {})
    )
    project_cfg = cache.load_json(project / "project.json")
    out_dir = out_dir or project / ("clips_vertical" if vertical else "clips")
    # The clips are exported into a hidden staging folder and swapped in only when all
    # succeeded, so a failed or cancelled run (--force included) leaves the folder as it was.
    # A run that fails leaves no empty folders behind. staged_dir refuses a drive root, a
    # mount point or a bind mount before any export; only an empty one is exported into
    # directly (each clip under a temporary name until complete; a failed run keeps the
    # finished clips).
    with (
        _claimed(out_dir, force),
        output_dir(out_dir.parent),
        staged_dir(out_dir, direct_if_empty=True) as work,
    ):
        from cutpoint.pano.render import exporter_for  # noqa: PLC0415

        pano = exporter_for(project, out_dir, vertical, pano_source, pano_cmd_every)
        if pano is not None and not any(pano.handles(m["source"]) for m in ordered):
            pano = None

        # Sources keep their recorded path as identity (events, tails, manifest);
        # the file actually read may be a remapped copy (config ``source_remap``).
        io_paths = {s: remap_source(s) for s in {m["source"] for m in ordered}}
        for s, p in io_paths.items():
            if str(p) != s:
                logger.info("Reading %s from %s (source_remap)", source_path(s).name, p)
        infos = {s: probe(p) for s, p in io_paths.items()}
        fmt = choose_format(
            [i for s, i in infos.items() if pano is None or not pano.handles(s)]
            or list(infos.values()),
            vertical,
        )
        if size is not None and not vertical:
            w, h = size
            if w <= 0 or h <= 0 or w % 2 or h % 2:
                raise jsonout.ArgumentError(
                    f"--size {w}x{h}: width and height must be positive and even"
                )
            fmt = OutputFormat(width=w, height=h, fps=fmt.fps)
        if pano is not None:  # 360 renders are 1920x1080 (or 1080x1920): match them
            fmt = OutputFormat(pano.fmt.width, pano.fmt.height, fmt.fps)
            pano.fmt = type(pano.fmt)(fmt.width, fmt.height, fmt.fps)
        logger.info(
            "Exporting %d clips at %dx%d@%d (%s order); smart tail %s",
            len(ordered),
            fmt.width,
            fmt.height,
            fmt.fps,
            order,
            f"on (min {base_tail.tail_min:.1f}s, max {base_tail.tail_max:.1f}s)"
            if _smart_tail_used(base_tail, ordered)
            else "off",
        )
        if lut is not None:
            logger.info("Grading every clip with %s (strength %g)", lut.path, lut.strength)

        plans = plan_clips(
            ordered,
            project,
            io_paths=io_paths,
            infos=infos,
            tail=base_tail,
            events=project_events,
            event_settings=ev_settings,
            speech=speech,
            padding=extend,
            crop=crop,
            project_cfg=project_cfg,
            pano=pano,
            language=language,
        )
        if temporary:
            check_free_space(out_dir.parent, plans, fmt)
        # --force replaces only what an earlier run made, never a file this run reads.
        replace, kept_sources, kept, had_manifest, protected_srts, renamed = _replaced_files(
            out_dir,
            [p.name for p in plans],
            source_keys((s, p, pano.inputs(s) if pano else []) for s, p in io_paths.items()),
        )
        peaks, makeup = peak_filters(plans, compress_peaks)
        grades: list[Grade | None] = [lut] * len(plans)
        if color_mode != "off":
            grades = _color_grades(plans, color_mode, color_strength, fmt, vertical)
        if force and not any(grades):
            _warn_dropped_grade(out_dir)
        workers = export_workers(plans)
        if workers > 1:
            logger.info("Exporting %d clips at a time (export_workers)", workers)
        results = render_clips(
            plans,
            work,
            fmt=fmt,
            vertical=vertical,
            audio_fade=audio_fade,
            pano=pano,
            protected_srts=protected_srts,
            clips_dir=out_dir,
            peaks=peaks,
            language=language,
            grades=grades,
            workers=workers,
        )
        write_manifest(
            work,
            project,
            plans,
            results,
            order=order,
            fmt=fmt,
            vertical=vertical,
            style=style,
            tail=base_tail,
            event_settings=ev_settings,
            padding=extend,
            speech=speech,
            audio_fade=audio_fade,
            crop=crop,
            compress_peaks=compress_peaks,
            peaks=peaks,
            makeup=makeup,
            grades=grades,
        )
        if work != out_dir:  # staged, not exported into directly
            swap_in(work, out_dir, [*replace, "manifest.json"])
    hint = "assemble includes them, so move them out if they are not part of the edit"
    if kept_sources:
        message = (
            f"Kept {', '.join(kept_sources)} in {out_dir}: a file this run reads as a source is "
            f"never replaced. They are not in the new manifest.json and {hint}"
        )
        logger.warning("%s", message)
        jsonout.warn("kept_sources", message, [out_dir / n for n in kept_sources])
    if kept:
        why = kept_why(kept, renamed, had_manifest)
        message = f"Kept {', '.join(kept)} in {out_dir}: {why}; {hint}"
        logger.warning("%s", message)
        jsonout.warn("kept_clips", message, [out_dir / n for n in kept])
    if not temporary:
        report_clips(out_dir)
    return out_dir
