"""``cutpoint clips``: export selected highlight moments as uniformly-encoded files.

:func:`export_clips` runs three steps: :func:`plan_clips` (moments to
:class:`ClipPlan`), :func:`render_clips` (encode) and :func:`write_manifest`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cutpoint import boundaries, cache, sources
from cutpoint.config import remap_source, source_path
from cutpoint.crop import Crop, detect, parse_spec, project_crop
from cutpoint.encode import OutputFormat, choose_format, export_clip_checked
from cutpoint.events.tails import EventTailSettings, event_tail, load_events
from cutpoint.ffmpeg_utils import (
    MediaInfo,
    output_dir,
    probe,
    recover_swap,
    staged_dir,
    swap_in,
)
from cutpoint.planning import Padding, pad, select_moments, speech_pass
from cutpoint.styles import Style, get_style
from cutpoint.tail import TailResult, TailSettings, smart_tail
from cutpoint.utils import slugify

if TYPE_CHECKING:
    from cutpoint.pano.render import PanoExporter

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_FADE_S = 0.25  # audio fade in/out baked into every exported clip


ARC_GROUPS = ("lift", "scenery", "run", "jump", "crash", "social")  # within a stretch of the day
ARC_OPENER_S = 5.0  # the opener is trimmed to this (around its middle)
ARC_MAX_REPEAT = 2  # same-category clips in a row
ARC_WINDOW = 4  # how far ahead a local swap may reach (clips)
REACTION_LABEL = "crash reaction"  # ``cutpoint mix`` label of a crash's reaction segment
BOUND_KEYS = ("min_start", "max_end")  # source-time limits of a segment's lead-in and tails

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
        ValueError: If the file is malformed or a segment is invalid.
        FileNotFoundError: If the file or a source video is missing.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Segments file not found: {path}")
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"{path} is not valid JSON: {err}") from None
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    moments = []
    for i, item in enumerate(items, 1):
        try:
            src = Path(item["source"])
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Segment {i} needs source, start and end: {item!r}") from None
        src = src if src.is_absolute() else (path.parent / src)
        io_path = remap_source(src)
        if not io_path.is_file():
            raise FileNotFoundError(f"Segment {i}: source not found: {io_path}")
        # Identity stays the original path; a remapped source is only read elsewhere.
        identity = str(src.resolve()) if io_path == src else os.path.abspath(src)  # noqa: PTH100 - normalises '..', keeps drive letters
        if end <= start or start < 0:
            raise ValueError(f"Segment {i}: invalid range {start}-{end}")
        moment = {
            "id": item.get("id", f"s{i:03d}"),
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
        if "tail" in item:
            try:
                moment["tail"] = float(item["tail"])
            except (TypeError, ValueError):
                raise ValueError(f"Segment {i}: tail must be a number of seconds") from None
            if moment["tail"] < 0:
                raise ValueError(f"Segment {i}: tail must not be negative")
        for key in ("smart_tail", "speech_aware"):
            if key in item:
                if not isinstance(item[key], bool):
                    raise ValueError(f"Segment {i}: {key} must be true or false")
                moment[key] = item[key]
        if "crop" in item:
            parse_spec(item["crop"])  # validate early
            moment["crop"] = item["crop"]
        for key in ("extend", "extend_start", "extend_end"):
            if key in item:
                try:
                    moment[key] = float(item[key])
                except (TypeError, ValueError):
                    raise ValueError(f"Segment {i}: {key} must be a number of seconds") from None
                if moment[key] < 0:
                    raise ValueError(f"Segment {i}: {key} must not be negative")
        moment.update(boundaries.clip_limits(item, f"Segment {i}"))
        for key in BOUND_KEYS:
            if item.get(key) is None:
                continue
            try:
                moment[key] = float(item[key])
            except (TypeError, ValueError):
                raise ValueError(f"Segment {i}: {key} must be a number of seconds") from None
            if moment[key] > start if key == "min_start" else moment[key] < end:
                raise ValueError(f"Segment {i}: {key} must not cut into {start}-{end}")
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
    """

    out: Path
    end: float
    read_error: dict[str, Any] | None = None
    pano: dict[str, Any] | None = None


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
        if end - start < 0.1:
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
            words_cache[src] = sources.source_words(src)

        def audio(t0: float, t1: float, s: Path = src, i: MediaInfo = info):
            return sources.audio_window(s, i, t0, t1)

        res = smart_tail(start, end, words_cache.get(src), audio, settings, max(limit, end))
        payoffs: list[str] = []
        if event_settings.enabled and m.get("smart_tail") is not False and "tail" not in m:
            if src not in words_cache:
                words_cache[src] = sources.source_words(src)
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
        plans, speech, project, fit_end=lambda e: "tail" not in e, words_cache=words_cache
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


def render_clips(
    plans: list[ClipPlan],
    out_dir: Path,
    *,
    fmt: OutputFormat,
    vertical: bool,
    audio_fade: float,
    pano: PanoExporter | None = None,
) -> list[RenderResult]:
    """Encode every planned clip into ``out_dir`` (360 clips through ``pano``).

    Returns:
        One :class:`RenderResult` per plan, in order.

    Raises:
        ValueError: If a 360 plan is given without ``pano``.
    """
    results: list[RenderResult] = []
    for plan in plans:
        m, start, end, tail = plan.entry, plan.start, plan.end, plan.tail
        out = out_dir / plan.name
        pano_rec = None
        if plan.pano:
            if pano is None:
                raise ValueError(f"{plan.name}: 360 plan needs a PanoExporter")
            read_err, pano_rec = pano.export(
                m["source"], start, end, out, audio_fade=audio_fade, kind=m.get("kind")
            )
        else:
            read_err = export_clip_checked(
                plan.src, plan.info, start, end, out, fmt, vertical, plan.crop, audio_fade
            )
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
        results.append(RenderResult(out, end, read_err, pano_rec))
    return results


def _entry(plan: ClipPlan, res: RenderResult, audio_fade: float) -> dict[str, Any]:
    """manifest.json record of one rendered clip."""
    m, src, start, end = plan.entry, plan.src, plan.start, res.end
    return {
        "file": plan.name,
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
        "score": m["score"],
        "label": m.get("label", ""),
        **({"extend": plan.pad_rec} if plan.pad_rec else {}),
        **({"speech": plan.speech} if plan.speech is not None else {}),
        **({"speech_aware": False} if m.get("speech_aware") is False else {}),
        **{k: m[k] for k in (*boundaries.LIMIT_KEYS, *BOUND_KEYS) if m.get(k) is not None},
        "audio_fade": audio_fade,
        **({"kind": m["kind"]} if m.get("kind") else {}),
        **({"category": m["category"]} if m.get("category") else {}),
        **({"pano": res.pano} if res.pano else {}),
        **({"read_error": res.read_error} if res.read_error else {}),
    }


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
) -> None:
    """Write ``<out_dir>/manifest.json`` for the rendered clips (and log the tail total)."""
    total_ext = 0.0
    for plan in plans:
        total_ext += plan.tail.extension
    for plan, res in zip(plans, results, strict=True):
        if res.read_error and res.read_error["end"] < plan.end:  # trimmed by a read error
            total_ext -= plan.end - max(res.end, min(plan.end, plan.entry["end"]))
    entries = [_entry(p, r, audio_fade) for p, r in zip(plans, results, strict=True)]
    logger.info(
        "Smart tail added %.1fs over %d of %d clips",
        total_ext,
        sum(1 for e in entries if e["tail"]["extension"] > 0),
        len(entries),
    )
    padding = padding or Padding()
    cache.save_json(
        out_dir / "manifest.json",
        {
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
                "enabled": tail.enabled,
                "tail_min": tail.tail_min,
                "tail_max": tail.tail_max,
                "settle": tail.settle,
            },
            "event_tails": event_settings.to_json(),
            "extend": {"start": padding.start, "end": padding.end},
            "speech_aware": speech.to_json(),
            "audio_fade": audio_fade,
            "crop": crop,
            "clips": entries,
        },
    )


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

    360 sources (Insta360 X3 clips registered in ``<project>/pano.json`` by
    ``cutpoint 360 scan``) are rendered to flat video along a director path (see
    :mod:`cutpoint.pano.render`): ``pano_source`` picks the input
    (``auto``: SDK stitch > Studio export > raw dual fisheye) and
    ``pano_cmd_every`` the camera update interval in frames. Everything else
    (numbering, tails, padding, speech-aware boundaries, fades) is unchanged.

    ``size`` ``(width, height)`` forces the output resolution instead of the
    largest source's (useful when 16:9 and taller sources such as GoPro 8:7
    are mixed; give the taller ones a ``crop`` so they fill the frame).

    Returns:
        The clips directory.

    Raises:
        FileNotFoundError: If highlights.json / segments / a source is missing.
        FileExistsError: If the clips folder is non-empty and ``force`` is False.
        ValueError: If the selection or a crop is invalid.
    """
    if crop is not None:
        parse_spec(crop)  # fail before any work
    highlights = cache.load_json(project / "highlights.json")
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
    recover_swap(out_dir)  # a hard-killed earlier replace, before the folder is looked at
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(f"{out_dir} is not empty; use --force to replace its clips")
    # The clips are exported into a hidden staging folder and swapped in only when all
    # succeeded, so a failed or cancelled run (--force included) leaves the folder as it was.
    # A run that fails leaves no empty folders behind.
    with output_dir(out_dir.parent), staged_dir(out_dir) as work:
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
                raise ValueError(f"--size {w}x{h}: width and height must be positive and even")
            fmt = OutputFormat(width=w, height=h, fps=fmt.fps)
        if pano is not None:  # 360 renders are 1920x1080 (or 1080x1920): match them
            fmt = OutputFormat(pano.fmt.width, pano.fmt.height, fmt.fps)
            pano.fmt = type(pano.fmt)(fmt.width, fmt.height, fmt.fps)
        logger.info(
            "Exporting %d clips at %dx%d@%d (%s order); smart tail %s (min %.1fs, max %.1fs)",
            len(ordered),
            fmt.width,
            fmt.height,
            fmt.fps,
            order,
            "on" if base_tail.enabled else "off",
            base_tail.tail_min,
            base_tail.tail_max,
        )

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
        )
        results = render_clips(
            plans, work, fmt=fmt, vertical=vertical, audio_fade=audio_fade, pano=pano
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
        )
        # --force replaces the clips and manifest.json; anything else in the folder stays.
        swap_in(work, out_dir, [p.name for p in out_dir.glob("*.mp4")] + ["manifest.json"])
    return out_dir
