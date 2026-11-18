"""Projects, their sources, moments, exports and resume state for ``cutpoint ui``."""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from cutpoint import cache, folders, insta360, jsonout, progress, projectdelete, runstate, stage
from cutpoint.assemble import JOIN_TRANSITIONS
from cutpoint.assistant import context as ai_context
from cutpoint.assistant import ops as ai_ops
from cutpoint.clips import MIN_EXPORT_S
from cutpoint.config import get_config, projects_dir, remap_source
from cutpoint.events import scoring
from cutpoint.pano import x3
from cutpoint.segments import MAX_SEGMENTS, check_segments, span_id
from cutpoint.ui.common import (
    BUSY_TEXT,
    EVENT_KINDS,
    EVENTS_MAX,
    FINAL_FILE,
    MAX_EXCLUDES,
    MAX_GLOB,
    MAX_INPUTS,
    MEDIA_LOOKUP_S,
    RESUME_CHECK_S,
    SAFE_NAME,
    SOURCES_CACHE_MAX,
    SOURCES_CACHE_S,
    UI_DIR,
    bad,
    cutpoint_cmd,
    finite_number,
    job_env,
    logger,
)
from cutpoint.ui.files import (
    _highlights,
    _inside,
    _is_project,
    _load,
    _own,
    _typed_path,
    _ui_folder,
    _values,
    picked_filters,
    picked_inputs,
    read_meta,
    save_analysis_skip,
)
from cutpoint.ui.jobs import parse_result
from cutpoint.ui.media import Media
from cutpoint.utils import (
    PANO_EXTS,
    VIDEO_EXTS,
    clip_files,
    file_stamp,
    find_videos,
    matches_glob,
    safe_is_file,
)


def project_dir(pid: str) -> Path:
    """The project folder ``pid`` (a direct child of the projects folder), else 404; 409 while
    it is being deleted."""
    root = projects_dir()
    if not pid or pid.startswith(".") or any(c in pid for c in "/\\\0") or pid != pid.strip():
        raise bad("unknown project", 404)
    p = root / pid
    if p in DELETING:
        raise bad("this project is being deleted", 409)
    try:
        ok = p.is_dir() and p.resolve().parent == root.resolve()
    except OSError:
        ok = False
    if not ok or not _is_project(p):
        raise bad("unknown project", 404)
    return p


DELETING: set[Path] = set()  # projects being deleted now (under ``Jobs.lock``): see project_dir
# projects whose ranking is being applied now (under ``Jobs.lock``): no job starts on them
RERANKING: set[Path] = set()


def protected_sources() -> dict[str, str]:
    """``{normcased real path: project name}`` of every source video or folder that a project
    under the projects folder reads (its picked inputs and the sources of its analysis): deleting
    a project never removes one."""
    found: dict[str, str] = {}
    try:
        projects = sorted(d for d in projects_dir().iterdir() if d.is_dir())
    except OSError:
        return found
    for d in projects:
        meta, hl = read_meta(d), _highlights(d) or {}
        raw = [
            *picked_inputs(meta),
            *((hl.get("settings") or {}).get("inputs") or []),
            *(s.get("path") for s in hl.get("sources") or [] if isinstance(s, dict)),
        ]
        for path in raw:
            if isinstance(path, str) and path:
                with contextlib.suppress(OSError, ValueError):
                    key = projectdelete.source_key(path)
                    found.setdefault(key, str(meta.get("name") or d.name))
    return found


def _under(path: str, roots: list[str]) -> bool:
    """``path`` is one of ``roots`` or inside one (checked on both spelled and real paths)."""

    def forms(x: str) -> set[str]:
        return {os.path.normcase(os.path.abspath(x)), os.path.normcase(os.path.realpath(x))}  # noqa: PTH100

    for root in roots:
        for r in forms(root):
            if any(c == r or c.startswith(r.rstrip(os.sep) + os.sep) for c in forms(path)):
                return True
    return False


def _listed_folder(folder: Path) -> Path:
    """Where a picked folder is listed: in place, else at its ``source_remap`` target, as
    ``analyze`` lists it."""
    return folder if os.path.exists(folder) else remap_source(folder)  # noqa: PTH110 - any OSError: missing


def _pano_folder(source: str | Path) -> Path:
    """The camera folder of a 360 project's picked source: the folder, or a picked file's."""
    src = Path(str(source))
    return src.parent if src.suffix.lower() in PANO_EXTS else src


def find_picked(inputs: list[str], exclude: list[str], recursive: bool) -> list[Path]:
    """The videos ``analyze`` finds in ``inputs``, de-duplicated in input order. As ``analyze``
    does, a missing input is listed at its ``source_remap`` target: the videos keep the original
    spelling, and ``exclude`` is matched against it.

    Raises:
        OSError: If an input cannot be listed (missing, unreadable, stale mount).
    """
    found: list[Path] = []
    for x in inputs:
        target = Path(x)
        io = _listed_folder(target)
        if io == target:
            found += find_videos([target], exclude=exclude, recursive=recursive)
            continue
        base, orig = io.resolve(), target.resolve()
        for read in find_videos([io], recursive=recursive):
            f = orig / read.relative_to(base) if read.is_relative_to(base) else read
            # source_remap_exclude: it would be read at the missing original
            if not matches_glob(f, exclude) and remap_source(f) != f:
                found.append(f)
    return list(dict.fromkeys(found))


def source_entries(
    p: Path, hl: dict[str, Any] | None, listed: list[Path | str] | None = None
) -> list[dict[str, Any]]:
    """The project's playable-in-principle sources: ``[{path, id, name, duration}]``. The picked
    inputs it lists itself (before their analysis) are added to ``listed``, followed by the error
    message when that listing failed."""
    meta = read_meta(p)
    pano = meta.get("kind") == "360"
    picked = picked_inputs(meta)
    if picked:
        roots = [str(_pano_folder(x)) for x in picked] if pano else picked
    else:  # command-line project: the inputs its analysis recorded
        roots = [str(x) for x in ((hl or {}).get("settings") or {}).get("inputs") or []]
    paths: dict[str, float | None] = {}
    for s in (hl or {}).get("sources") or []:
        if isinstance(s, dict) and isinstance(s.get("path"), str):
            dur = s.get("duration")
            paths.setdefault(s["path"], dur if isinstance(dur, (int, float)) else None)
    if not paths and picked:
        if listed is not None:
            listed.extend(map(Path, picked))
        try:
            if pano:  # one .lrv per clip 360 analysis can use (those of meta.clips, if set)
                only = meta.get("clips") or None
                _, clips, _ = inspect_source(Path(picked[0]))
                found = [
                    g["lrv"]
                    for g in (clips[n] for n in _usable(clips))
                    if not only
                    or {g["num"], x3.clip_key(g["date"], g["time"], g["num"])} & set(only)
                ]
            else:
                found = find_picked(picked, *picked_filters(meta))
            paths = dict.fromkeys(map(str, found))
        except OSError as err:  # find_videos: missing, unreadable, stale mount
            if listed is not None:
                listed.append(str(err))
    return [
        {
            "path": path,
            "id": hashlib.sha1(path.encode()).hexdigest()[:12],
            "name": Path(path).name,
            "duration": dur,
        }
        for path, dur in paths.items()
        if Path(path).is_absolute()
        and Path(path).suffix.lower() in (VIDEO_EXTS | {".lrv"} if pano else VIDEO_EXTS)
        and _under(path, roots)
    ]


class Sources:
    """Each project's :func:`source_entries`, kept while its inputs are unchanged, so media
    requests (each Range request, each still) do not reread and rescan a network share.

    The list depends only on ``ui/meta.json``, ``highlights.json`` and, before an analysis, the
    entries of the picked inputs; a folder's mtime changes when an entry directly inside it is
    added, removed or renamed. So a hit costs a stat per file and input, and any change to them
    rebuilds it, as does a change to ``source_remap`` or ``source_remap_exclude`` (``get_config``
    rereads an edited ``config.json``). ``SOURCES_CACHE_S`` bounds what stats cannot see (a
    sub-folder of a ``--recursive`` project changed, a link under a source retargeted, a
    filesystem with coarse mtimes). Callers must not change the entries."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # project folder -> (built at, files stamped, their stamps, remap settings, entries);
        # oldest use first
        self.kept: dict[str, tuple[float, list[Path], list[Any], Any, list[dict[str, Any]]]] = {}

    def entries(self, p: Path) -> list[dict[str, Any]]:
        return self.listing(p)[0]

    def listing(self, p: Path) -> tuple[list[dict[str, Any]], str | None]:
        """``(entries, why the picked folder could not be listed or None)``."""
        key = str(p)
        with self.lock:
            hit = self.kept.get(key)
        cfg = get_config()
        remap = (cfg.source_remap, cfg.source_remap_exclude)
        if (
            hit is not None
            and time.monotonic() - hit[0] < SOURCES_CACHE_S
            and hit[3] == remap
            and [file_stamp(f) for f in hit[1]] == hit[2]
        ):
            with self.lock:
                if self.kept.get(key) is hit:
                    self.kept[key] = self.kept.pop(key)  # recently used: evicted last
            return hit[4], None
        # Each file is stamped before it is read: a change made meanwhile shows on the next request.
        files = [p / f"{UI_DIR}/meta.json", p / "highlights.json"]
        at, stamps = time.monotonic(), [file_stamp(f) for f in files]
        meta = read_meta(p)
        picked = picked_inputs(meta)
        if picked and meta.get("kind") == "360":
            picked = [str(_pano_folder(picked[0]))]  # its .lrv files are listed, a file's too
        files += [_listed_folder(Path(x)) for x in picked]
        stamps += [file_stamp(f) for f in files[2:]]
        listed: list[Path | str] = []
        entries = source_entries(p, _highlights(p), listed)
        n = 2 + len(picked) if listed else 2  # the picked inputs count only while listed
        error = next((x for x in listed if isinstance(x, str)), None)
        with self.lock:
            self.kept.pop(key, None)
            if error is not None:  # an input could not be listed: the next request lists again
                return entries, error
            self.kept[key] = (at, files[:n], stamps[:n], remap, entries)
            while len(self.kept) > SOURCES_CACHE_MAX:
                del self.kept[next(iter(self.kept))]
        return entries, None


class MediaLookups:
    """The project folder (:func:`project_dir`) and source list (:class:`Sources`) the media
    routes serve from, kept for ``MEDIA_LOOKUP_S`` per project: a browser opens a fragmented MP4
    with hundreds of Range requests, and on a network share each folder check costs ~10 ms, so
    rechecking it for each one stalled playback ~16 s (#1014).

    Only found projects are kept: an unknown project is checked on every request (404). A project
    being deleted answers 409 at once (``DELETING`` is checked before the kept answer), and
    :meth:`forget` drops it once deleted, so it is 404 right after an in-app delete. A folder
    deleted, moved or changed outside the app (or a source added or removed meanwhile) is seen
    within ``MEDIA_LOOKUP_S``."""

    def __init__(self, sources: Sources) -> None:
        self.sources = sources
        self.lock = threading.Lock()
        # (projects folder, project id) -> (checked at, project folder, its source entries, the
        # content keys of its sources seen so far: source path -> :meth:`Media.key`)
        self.kept: dict[
            tuple[str, str], tuple[float, Path, list[dict[str, Any]], dict[str, str]]
        ] = {}
        # bumped by each forget: a check that started before one keeps nothing
        self.gen = 0

    def find(self, pid: str, sid: str) -> tuple[Path, dict[str, Any]]:
        """``(project folder, source entry)``, else 404 (unknown project or source) or 409."""
        key = (str(projects_dir()), pid)
        now = time.monotonic()
        with self.lock:
            hit, gen = self.kept.get(key), self.gen
        if hit is not None and now - hit[0] < MEDIA_LOOKUP_S and hit[1] not in DELETING:
            p, entries = hit[1], hit[2]
        else:
            p = project_dir(pid)  # 404 or 409: nothing kept
            entries = self.sources.entries(p)
            with self.lock:
                self.kept = {k: v for k, v in self.kept.items() if now - v[0] < MEDIA_LOOKUP_S}
                if gen == self.gen:  # no delete finished meanwhile: this answer still holds
                    self.kept[key] = (now, p, entries, {})
        entry = next((s for s in entries if s["id"] == sid), None)
        if entry is None:
            raise bad("unknown source", 404)
        return p, entry

    def content_key(self, pid: str, path: str) -> str | None:
        """:meth:`Media.key` of the source ``path`` of project ``pid`` (after :meth:`find`),
        kept as long as the project's kept answer is (``MEDIA_LOOKUP_S``), so repeated requests
        stat the source once. A source that goes missing is seen at once by the route's own stat;
        a changed one within ``MEDIA_LOOKUP_S``. A missing source is not kept."""
        key = (str(projects_dir()), pid)
        with self.lock:
            hit = self.kept.get(key)
            if hit is not None and time.monotonic() - hit[0] < MEDIA_LOOKUP_S:
                seen = hit[3].get(path)
                if seen is not None:
                    return seen
            else:
                hit = None
        found = Media.key(remap_source(path))
        if found is not None and hit is not None:
            with self.lock:
                hit[3][path] = found
        return found

    def forget(self, p: Path) -> None:
        """Drop the kept answer for the project folder ``p`` (it was deleted)."""
        with self.lock:
            self.gen += 1
            self.kept = {k: v for k, v in self.kept.items() if v[1] != p}


_REMOTE: dict[str, bool] = {}  # project folder -> stage.is_remote of it, for the page (#1014)


DRIVE_REMOVABLE, DRIVE_REMOTE = 2, 4  # GetDriveTypeW


def on_network(p: Path) -> bool:
    """Whether the project folder ``p`` is likely on a network or removable drive, checked once
    per folder and run. Windows asks the drive type (a UNC path is remote), so a local drive
    other than C: to E: is not named; elsewhere, or if Windows cannot say,
    :func:`cutpoint.stage.is_remote`."""
    key = str(p)
    if key not in _REMOTE:
        try:
            if sys.platform == "win32":
                try:
                    kind = projectdelete._drive_type(p.anchor)
                    _REMOTE[key] = kind in (DRIVE_REMOVABLE, DRIVE_REMOTE)
                    return _REMOTE[key]
                except OSError:
                    pass
            _REMOTE[key] = stage.is_remote(p)
        except (OSError, ValueError):
            _REMOTE[key] = False
    return _REMOTE[key]


def source_folder(p: Path) -> Path | None:
    """The folder holding the project's videos: the deepest folder shared by its picked inputs
    (a file's parent), or for a command-line project by its recorded sources. Never client
    input."""
    picked = picked_inputs(read_meta(p))
    try:
        if picked:
            # A picked folder that is gone must not fall back to its parent; only files step up.
            folders = []
            for source in picked:
                x = remap_source(Path(source))
                missing_file = not x.exists() and x.suffix.lower() in VIDEO_EXTS | PANO_EXTS
                is_file = x.is_file() or missing_file
                folders.append(x.parent if is_file else x)
        else:
            entries = source_entries(p, _highlights(p))
            folders = [remap_source(Path(s["path"])).parent for s in entries]
        try:
            folder = Path(os.path.commonpath(folders)) if folders else None
        except ValueError:  # on different drives: there is no one folder to show
            return None
        return folder if folder is not None and folder.is_dir() else None
    except OSError:  # unreadable or stale (permissions, network share): treat as gone
        return None


def bit_description(c: dict[str, Any], events: Any) -> str:
    """What the dialogue reader said happens in the moment ``c`` if it is a dialogue moment (one a
    talk stretch nominated or widened, or that has a ``dialogue_title``): the ``description`` of
    the most confident dialogue event of ``events`` (the events.json object) inside it, else the
    ``dialogue_description`` highlights.json keeps; "" for any other moment."""
    nominated = ((c.get("signals") or {}).get("events") or {}).get("nominated") or []
    if not (
        any(str(n).startswith("dialogue-") for n in nominated)
        or c.get("widened_from")
        or c.get("dialogue_title")
        or c.get("dialogue_description")
    ):
        return ""
    best, conf = "", -1.0
    for e in (events.get("events") or []) if isinstance(events, dict) else []:
        if (
            isinstance(e, dict)
            and e.get("kind") == "text"
            and e.get("type") == scoring.DIALOGUE_TYPE
            and e.get("source") == c.get("source")
            and isinstance(e.get("start"), int | float)
            and isinstance(e.get("end"), int | float)
            and e["start"] >= c["start"] - 0.01
            and e["end"] <= c["end"] + 0.01
            and e.get("description")
            and (e.get("confidence") or 0) > conf
        ):
            best, conf = str(e["description"]), e.get("confidence") or 0
    return best or str(c.get("dialogue_description") or "")


def moments_view(hl: dict[str, Any] | None, events: Any = None) -> dict[str, Any] | None:
    """Keep saved trims attached to source intervals, not mutable CLI ranks; ``cli_id`` keeps the
    CLI id (h001) that the cards show and ``clips --ids`` takes. ``display_label`` is the label
    the moments list shows: a generic one (``action-moment``) gives way to what the vision model
    saw (:func:`cutpoint.assistant.context.shown_label`, as in AI Chat); ``label`` stays the
    analysis label, which a moment added to the sequence takes. ``long_bit`` marks a long
    dialogue bit (the "Bit" badge and "Long bits" filter), with ``bit_description`` what
    happens in a dialogue moment (:func:`bit_description`; ``events``: the project's events.json
    object). A
    dialogue moment's ``display_label`` is its readable ``dialogue_title``, not the slug label."""
    if hl is None:
        return None
    reals: dict[str, str] = {}  # a source's real path, looked up once (A742: thousands of moments)

    def real(source: str) -> str:
        if source not in reals:
            reals[source] = os.path.realpath(source)
        return reals[source]

    return {
        **hl,
        "candidates": [
            {
                **c,
                "cli_id": c.get("id"),
                "display_label": c.get("dialogue_title")
                or ai_context.shown_label(c.get("label"), c),
                "long_bit": scoring.is_long_dialogue(c),
                "bit_description": bit_description(c, events),
                "id": span_id(real(c["source"]), c["start"], c["end"]),
            }
            for c in hl.get("candidates") or []
        ],
    }


def _secs(v: Any) -> float | None:
    """``v`` as a positive, finite number of seconds, else None."""
    v = finite_number(v)
    return v if v is not None and v > 0 else None


def analysis_took(p: Path, meta: dict[str, Any], hl: dict[str, Any] | None) -> float | None:
    """Wall-clock seconds of the last analysis. ``timings.total_s`` covers the flat analysis
    only (a re-rank keeps it and adds ``timings.rerank_s``), not the scan, events and camera
    paths of ``360 analyze``, so a 360 project's time is ``total_s`` in
    ``pano_timings.json`` (every analyze run since #564; a ``--views-only`` run, which leaves
    ``highlights.json`` alone, keeps the earlier analysis keys and time and adds
    ``views_only_s``); None when they are missing, as in an older file."""
    if meta.get("kind") == "360":
        pano = _load(p, "pano_timings.json")
        pano = pano if isinstance(pano, dict) else {}
        if _secs(pano.get("analyze_s")) is not None and (whole := _secs(pano.get("total_s"))):
            return whole
        return None
    timings = (hl or {}).get("timings")
    return _secs(timings.get("total_s")) if isinstance(timings, dict) else None


def analysis_stats(hl: dict[str, Any] | None, took: float | None) -> dict[str, float] | None:
    """Seconds of video the last analysis read, from highlights.json, and the seconds it
    ``took`` (#459, see ``analysis_took``); None when either is unknown."""
    # analyzed_s: the part a recovered or cut-short read covered, else the whole source
    parts = [
        _secs(s.get("analyzed_s", s.get("duration")))
        for s in (hl or {}).get("sources") or []
        if isinstance(s, dict)
    ]
    video = [v for v in parts if v is not None]
    if took is None or not video or len(video) < len(parts):
        return None
    return {"video_s": round(sum(video), 3), "took_s": took}


def validate_segments(
    raw: Any, sources: list[dict[str, Any]], media: Media
) -> list[dict[str, Any]]:
    """Check a review against the project's sources; return the cleaned segments (``exact``
    kept only when true)."""
    by_path = {s["path"]: s for s in sources}

    def duration(path: str) -> float | None:
        return by_path[path]["duration"] or media.duration(remap_source(path))

    def fail(rule: str, i: int, row: Any, **info: Any) -> HTTPException:
        at = f"segment {i}: "
        match rule:
            case "list":
                return bad("segments must be a list")
            case "count":
                return bad(f"at most {MAX_SEGMENTS} segments")
            case "object":
                return bad(f"{at}object expected")
            case "id":
                return bad(f"{at}id must be a unique string of 1-64 characters")
            case "source":
                return bad(f"{at}source is not part of this project")
            case "number":
                return bad(f"{at}{info['key']} must be a number")
            case "range":
                return bad(f"{at}needs 0 <= start and end at least {MIN_EXPORT_S}s after start")
            case "unreadable":
                return bad(f"{at}source cannot be read")
            case "past":
                return bad(f"{at}end {info['end']:g}s is past the source's {info['duration']:g}s")
            case "label":
                return bad(f"{at}label must be text of at most {ai_ops.LABEL_MAX} characters")
        return bad(f"{at}exact must be true or false")

    return check_segments(raw, has_source=by_path.__contains__, duration=duration, fail=fail)


def validate_joins(raw: Any) -> list[dict[str, str]]:
    """The cleaned ``joins`` of a review: a clip pair by segment id (``a`` then ``b``) and its
    ``transition``. The ids need not be in the sequence now: a pair a reorder separated keeps
    its choice for when the two are next to each other again."""
    if not isinstance(raw, list):
        raise bad("joins must be a list")
    if len(raw) > MAX_SEGMENTS:
        raise bad(f"at most {MAX_SEGMENTS} joins")
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise bad(f"join {i}: object expected")
        a, b, kind = item.get("a"), item.get("b"), item.get("transition")
        if not all(isinstance(s, str) and 0 < len(s) <= 64 for s in (a, b)):
            raise bad(f"join {i}: a and b must be segment ids of 1-64 characters")
        if (a, b) in seen:
            raise bad(f"join {i}: the pair {a}, {b} is already set")
        seen.add((a, b))
        if kind not in JOIN_TRANSITIONS:
            raise bad(f"join {i}: transition must be 'cut' or 'fade'")
        out.append({"a": a, "b": b, "transition": kind})
    return out


def _joins_of(review: Any) -> list[dict[str, str]]:
    """The joins saved in a ``review.json`` document; [] when it has none or they are invalid."""
    try:
        return validate_joins(review.get("joins", []) if isinstance(review, dict) else [])
    except HTTPException:
        return []


def clip_joins(folder: Path, joins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    r"""Each join of the clips in ``folder``, in clip order: the two files (``from``, ``to``),
    their manifest ``id``\ s (``a``, ``b``; None for a clip it does not list) and the review's
    ``transition`` for the pair, None when ``joins`` has no choice for it."""
    from cutpoint.planning import manifest_entries  # noqa: PLC0415

    files = clip_files(folder)
    entries = manifest_entries(folder, files)
    chosen = {(j["a"], j["b"]): j["transition"] for j in joins}
    out = []
    for x, y in pairwise(files):
        a, b = entries.get(x.name, {}).get("id"), entries.get(y.name, {}).get("id")
        out.append({"from": x.name, "to": y.name, "a": a, "b": b, "transition": chosen.get((a, b))})
    return out


def export_joins(folder: Path, joins: list[dict[str, str]]) -> list[dict[str, str]]:
    r"""The review's ``joins`` as ``assemble --joins`` entries for the clips in ``folder``,
    through its manifest's ``id``\ s. Only pairs next to each other in the clip order: the
    others use the global transition, which is no mismatch to warn about."""
    return [
        {"from": j["from"], "to": j["to"], "transition": j["transition"]}
        for j in clip_joins(folder, joins)
        if j["transition"]
    ]


def load_exports(p: Path) -> list[dict[str, Any]]:
    data = _load(p, f"{UI_DIR}/exports.json")
    items = data.get("exports") if isinstance(data, dict) else None
    return [
        e
        for e in items or []
        if isinstance(e, dict) and isinstance(e.get("dir"), str) and SAFE_NAME.fullmatch(e["dir"])
    ]


def _events_sig(p: Path) -> str | None:
    """``<mtime_ns>-<size>`` of the project's ``events.json`` and ``crashes.json`` (joined by
    ``+``, a missing file skipped), which changes when a run rewrites either; None when both are
    missing."""
    parts = []
    for name in ("events.json", "crashes.json"):
        try:
            st = (p / name).stat()
        except OSError:
            continue
        parts.append(f"{st.st_mtime_ns}-{st.st_size}")
    return "+".join(parts) or None


def _rail_events(p: Path) -> list[dict[str, Any]]:
    """The time-located events of ``events.json`` the trim rail marks, cut to ``{source, start,
    end, kind, type, label, confidence}``: sounds (``audio``), on-screen text (``ocr``), camera
    telemetry such as crashes and talk stretches (``text``, labelled with their
    ``dialogue_title`` when they have one), plus the crashes of ``crashes.json`` that no
    telemetry crash of events.json already covers (``kind`` telemetry, ``type`` crash). No model
    descriptions and no coverage spans. At most ``EVENTS_MAX`` (the most confident), by start.
    [] when there are none or the files are unreadable."""
    doc = _load(p, "events.json")
    raw = doc.get("events") if isinstance(doc, dict) else None
    out = []
    for e in raw if isinstance(raw, list) else []:
        if not isinstance(e, dict) or not isinstance(e.get("source"), str):
            continue
        if e.get("kind") not in EVENT_KINDS or str(e.get("type", "")).startswith("coverage"):
            continue
        start, end = finite_number(e.get("start")), finite_number(e.get("end"))
        conf = finite_number(e.get("confidence"))
        if start is None or end is None:
            continue
        label = e.get("type") if e["kind"] == "telemetry" else e.get("label")
        if e["kind"] == "text" and isinstance(e.get("title"), str) and e["title"].strip():
            label = e["title"].strip()
        out.append(
            {
                "source": e["source"],
                "start": start,
                "end": max(start, end),
                "kind": e["kind"],
                "type": str(e.get("type", ""))[:40],
                "label": str(label or e.get("type") or e["kind"])[:60],
                "confidence": None if conf is None else round(conf, 3),
            }
        )
    out += _crash_events(p, out)
    if len(out) > EVENTS_MAX:
        out = sorted(out, key=lambda e: -(e["confidence"] or 0.0))[:EVENTS_MAX]
    return sorted(out, key=lambda e: (e["source"], e["start"]))


CRASH_SAME_S = 2.0  # a crashes.json entry this close to a telemetry crash event is the same crash


def _crash_events(p: Path, have: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ``crashes.json`` entries as telemetry crash events for the rail, without those within
    CRASH_SAME_S of a telemetry crash in ``have`` (same source) and those below the confidence
    floor (:func:`cutpoint.crash.crash_floor` of every confidence in the file)."""
    from cutpoint.crash import crash_floor  # noqa: PLC0415

    doc = _load(p, "crashes.json")
    raw = doc.get("crashes") if isinstance(doc, dict) else None
    known = [
        (e["source"], e["start"], e["end"])
        for e in have
        if e["kind"] == "telemetry" and e["type"] == "crash"
    ]
    entries = [
        c
        for c in (raw if isinstance(raw, list) else [])
        if isinstance(c, dict)
        and isinstance(c.get("source"), str)
        and finite_number(c.get("confidence")) is not None
    ]
    floor = crash_floor(c["confidence"] for c in entries)
    out = []
    for c in entries:
        start, end, conf = finite_number(c.get("t0")), finite_number(c.get("t1")), c["confidence"]
        if conf < floor:
            continue
        if start is None or end is None:
            continue
        end = max(start, end)
        if any(
            src == c["source"] and start <= b + CRASH_SAME_S and end >= a - CRASH_SAME_S
            for src, a, b in known
        ):
            continue
        out.append(
            {
                "source": c["source"],
                "start": start,
                "end": end,
                "kind": "telemetry",
                "type": "crash",
                "label": "crash",
                "confidence": round(conf, 3),
            }
        )
    return out


TARGET_MAX_S = 36000  # the sequence's target length: up to 10 hours


def _valid_target(v: Any) -> bool:
    """Whether ``v`` is a usable target length (a number above 0, at most TARGET_MAX_S)."""
    v = finite_number(v)
    return v is not None and 0 < v <= TARGET_MAX_S


def _target_of(review: Any) -> float | None:
    """The target length saved in a ``review.json`` document, else None."""
    v = review.get("target") if isinstance(review, dict) else None
    return float(v) if _valid_target(v) else None


def run_state(p: Path, analyzed: bool) -> dict[str, Any] | None:
    """The project's last recorded run (``progress.json``, :func:`cutpoint.runstate.read`) as
    the page shows it: ``command``, ``status`` and whether Resume continues it (``resumable``:
    interrupted or stopped for memory; also failed while there is no analysis, ``analyzed``
    False, which only Resume can finish). None without a readable record."""
    rec = runstate.read(p) if _inside(p, runstate.FILE) else None
    command = (rec or {}).get("command")
    if rec is None or command not in runstate.COMMANDS:
        return None
    status = rec.get("status")
    resumable = status in runstate.RESUMABLE or (status == "failed" and not analyzed)
    return {"command": command, "status": status, "resumable": resumable}


CLIPS_UI_RESUME = re.compile(r"\.(clips_ui_\d{8}-\d{6}(?:-\d+)?)\.cutpoint-resume")


def prune_resume_folders(p: Path) -> None:
    """Remove the hidden resume folders (:mod:`cutpoint.clipresume`) of stopped UI exports that
    can never be resumed (#716). Every UI export writes a new ``clips_ui_<stamp>`` folder and
    Resume replays only the project's last run, so once a later export has superseded a stopped
    one its ``.clips_ui_<stamp>.cutpoint-resume`` only takes disk space.

    Only a folder with Cutpoint's naming and its ``resume.json`` goes. It stays when
    ``progress.json`` still names its clips folder in the recorded command line of a run that
    has not finished, when a run holds the project's lock or the clips folder's, and when it
    cannot be told apart from what Cutpoint wrote. Never raises."""
    try:
        if folders.lock_held(p / runstate.LOCK_NAME):
            return
        rec = runstate.read(p) if _inside(p, runstate.FILE) else None
        if rec is None and os.path.lexists(p / runstate.FILE):
            return  # a record we cannot read may point at a folder that is resumable
        named: set[str] = set()
        if rec is not None and rec.get("status") != "done":
            named = {Path(a).name for a in rec.get("argv") or [] if isinstance(a, str)}
        for f in sorted(p.iterdir()):
            m = CLIPS_UI_RESUME.fullmatch(f.name)
            if (
                m is None
                or f.is_symlink()
                or f.is_junction()
                or not f.is_dir()
                or not (f / folders.RESUME_FILE).is_file()
                or m.group(1) in named
                or folders.lock_held(p / m.group(1))
            ):
                continue
            try:
                folders._remove_leftover(f, "the clips of a stopped export that was not resumed")
            except OSError as err:
                logger.warning(
                    "Could not remove %s (%s); it is removed after a later export", f, err
                )
                continue
            logger.info("Removed %s: a later export replaced the stopped one", f)
    except OSError as err:
        logger.warning("Could not look for old resume folders in %s: %s", p, err)


def project_busy(p: Path, run: dict[str, Any] | None) -> bool:
    """A run is going on the project: another process holds its run lock
    (``.analysis.run.cutpoint-lock``, :mod:`cutpoint.runstate`; one whose process is gone does
    not count), or ``run`` (:func:`run_state`) reads ``running``."""
    if run is not None and run["status"] == "running":
        return True
    return folders.lock_held(p / runstate.LOCK_NAME)


def resume_check(p: Path) -> dict[str, Any]:
    """What ``cutpoint resume <p>`` would continue: the ``--json`` envelope of its ``--dry-run``,
    run from the projects folder.

    Raises:
        HTTPException: 409 :data:`BUSY_TEXT` while the run is still going, 404 without a run to
            resume, 409 with the CLI's message when it refuses the record, 500 or 504 when the
            check could not run.
    """
    try:
        done = subprocess.run(
            cutpoint_cmd("resume", str(p), "--dry-run", "--json"),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=job_env(),
            cwd=p.parent,
            timeout=RESUME_CHECK_S,
            check=False,
        )
    except subprocess.TimeoutExpired as err:
        raise bad("checking the run to resume took too long", 504) from err
    except OSError as err:
        raise bad(f"could not check the run to resume: {err}", 500) from err
    result = next((r for ln in reversed(done.stdout.splitlines()) if (r := parse_result(ln))), None)
    if result is None:
        raise bad(f"could not check the run to resume (exit code {done.returncode})", 500)
    err = result.get("error")
    if isinstance(err, dict):
        if err.get("code") == jsonout.FOLDER_BUSY:
            raise bad(BUSY_TEXT, 409)
        if err.get("code") == jsonout.NOT_FOUND:
            raise bad("this project has no run to resume", 404)
        raise bad(str(err.get("message") or "the run cannot be resumed"), 409)
    return result


def _skip_after(argv: list[str]) -> list[str]:
    """The steps a recorded command line gives ``--skip``."""
    steps = _values(argv, "--skip")
    return [s for s in progress.SKIPPABLE if s in steps]


def _recorded_picks(rec: dict[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    """What a recorded ``clips`` command line exports, as :func:`stitch_first` takes it: its
    ``--segments`` file (relative to the record's ``cwd``), else its ``--top``, ``--ids`` and
    ``--max-total``."""
    argv = [str(a) for a in rec.get("argv") or []]
    if seg := _values(argv, "--segments"):
        return Path(str(rec.get("cwd") or ".")) / seg[0], {}
    top, total = _values(argv, "--top"), _values(argv, "--max-total")
    picks: dict[str, Any] = {"top": None, "ids": _values(argv, "--ids") or None, "max_total": None}
    with contextlib.suppress(ValueError):
        picks["top"] = int(top[0]) if top else None
    with contextlib.suppress(ValueError):
        picks["max_total"] = float(total[0]) if total else None
    return None, picks


def resumed(p: Path, rec: dict[str, Any], outputs: dict[str, Any], stamp: str) -> None:
    """A resumed run finished: record what its own job would have (the run that stopped never
    got there). An analysis keeps its ``--skip`` (:func:`save_analysis_skip`); a clips folder
    of the project that ``clips`` or ``edit`` wrote (``outputs.clips_dir`` of the ``--json``
    envelope) becomes the latest export, with ``edit``'s final video."""
    command, argv = rec.get("command"), [str(a) for a in rec.get("argv") or []]
    if command in ("analyze", "edit") and (skip := _skip_after(argv)):
        _ui_folder(p)
        save_analysis_skip(p, skip)
    clips_dir = outputs.get("clips_dir")
    if command not in ("clips", "edit") or not isinstance(clips_dir, str):
        return
    name = Path(clips_dir).name
    if (
        not SAFE_NAME.fullmatch(name)
        or not _inside(p, name)
        or (p / name).resolve() != Path(clips_dir).resolve()
        or not (p / name / "manifest.json").is_file()
    ):
        return  # not a clips folder of this project (an --out elsewhere)
    final = outputs.get("final") if command == "edit" else None
    final_name = Path(final).name if isinstance(final, str) else None
    if (
        final_name is None
        or not FINAL_FILE.fullmatch(final_name)
        or not (p / final_name).is_file()
        or (p / final_name).resolve() != Path(str(final)).resolve()
    ):
        final_name = None
    _ui_folder(p)
    items = [e for e in load_exports(p) if e["dir"] != name]  # replaced, as edit's
    items.append(
        {"dir": name, "created": stamp, "vertical": "--vertical" in argv, "final": final_name}
    )
    cache.save_json(_own(p, f"{UI_DIR}/exports.json"), {"exports": items})


def current_export(p: Path) -> dict[str, Any] | None:
    """The latest export whose clips are still on disk, inside the project."""
    return next(
        (
            e
            for e in reversed(load_exports(p))
            if (p / e["dir"] / "manifest.json").is_file()
            and _inside(p, e["dir"])
            and _inside(p, f"{e['dir']}/manifest.json")
        ),
        None,
    )


def _final_names(p: Path) -> list[str]:
    """Finals in the project folder: the current export's first, then the newest first."""
    exp = current_export(p)
    final = (exp or {}).get("final")
    first = [final] if isinstance(final, str) and SAFE_NAME.fullmatch(final) else []
    found: list[tuple[int, str]] = []
    with contextlib.suppress(OSError), os.scandir(p) as it:
        for e in it:
            # a timeline next to its video is listed with it, not as a final of its own
            if FINAL_FILE.fullmatch(e.name) and (
                e.name.endswith(".mp4") or not (p / f"{Path(e.name).stem}.mp4").is_file()
            ):
                with contextlib.suppress(OSError):
                    found.append((e.stat().st_mtime_ns, e.name))
    found.sort(reverse=True)
    names = dict.fromkeys([*first, *(n for _, n in found)])  # in order, no duplicates
    return [n for n in names if (p / n).is_file() and _inside(p, n)]


def output_names(p: Path) -> list[str]:
    """Files of the finals inside the project: each video or timeline-only export (``assemble
    --format fcpxml|otio``), then the timelines and subtitles of the same name."""
    out: list[str] = []
    for final in _final_names(p):
        stem = Path(final).stem
        out += [
            final,
            *(
                f"{stem}{e}"
                for e in (".fcpxml", ".otio", ".srt")
                if (p / f"{stem}{e}").is_file() and _inside(p, f"{stem}{e}")
            ),
        ]
    # a timeline-only final is also its own timeline; a .fcpxml and .otio of one stem share their .srt
    return list(dict.fromkeys(out))


def chat_files(p: Path) -> list[dict[str, str]]:
    """The files of project ``p`` that AI Chat may name (:func:`cutpoint.assistant.context.project_files`):
    its analysis files that exist, its finals and the clips folder of its latest export."""
    present = [n for n in ai_context.FILE_HOLDS if (p / n).is_file() and _inside(p, n)]
    exp = current_export(p)
    return ai_context.project_files(present, output_names(p), exp["dir"] if exp else None)


def fresh_name(parent: Path, stem: str, exts: tuple[str, ...] = ("",)) -> str:
    """``stem`` or ``stem-N``: the first name no file ``name + ext`` already uses.

    A broken symlink counts as taken, so an output is never written through one.
    """
    n, name = 1, stem
    while any(os.path.lexists(parent / f"{name}{e}") for e in exts):
        n += 1
        name = f"{stem}-{n}"
    return name


def inspect_source(src: Path) -> tuple[list[Path], dict[str, dict[str, Any]], Counter[str]]:
    """Split a picked source into flat videos, Insta360 X3 clips and the suffixes of other files.

    One non-recursive listing of regular files (sub-folders are skipped); nothing is read or
    written. A folder gives its files; a file gives itself or, when it has an X3 camera name, the
    files of its clip next to it (one 360 file means that clip). Clips come from
    ``x3.group_files``; ``other`` counts the remaining suffixes, lower-cased ("no extension" when
    there is none).
    """
    f = insta360.parse_name(src) if src.is_file() else None
    if src.is_file() and f is None:
        files = [src]
    else:
        listed = sorted(p for p in (src.parent if f else src).iterdir() if safe_is_file(p))
        files = [
            p
            for p in listed
            if not f
            or (
                (g := insta360.parse_name(p))
                and (g["date"], g["time"], g["num"]) == (f["date"], f["time"], f["num"])
            )
        ]
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXTS]
    other = Counter(
        p.suffix.lower() or "no extension"
        for p in files
        if p.suffix.lower() not in VIDEO_EXTS and not insta360.NAME_RE.match(p.name)
    )
    return videos, x3.group_files(files), other


def _usable(clips: dict[str, dict[str, Any]]) -> list[str]:
    """Numbers of the X3 clips 360 analysis can use: those with a ``_00_.insv`` and a ``.lrv``."""
    return sorted(n for n, g in clips.items() if "lens00" in g and "lrv" in g)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _suffix_counts(counts: Counter[str], limit: int = 5) -> str:
    """Counts such as "12 .insv, 3 .lrv": most first, then by name; at most ``limit`` types."""
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    text = ", ".join(
        f"{n} {'without extension' if k == 'no extension' else k}" for k, n in items[:limit]
    )
    rest = len(items) - limit
    return f"{text} and {_plural(rest, 'more type')}" if rest > 0 else text


def source_problem(src: Path) -> str | None:
    """Why ``src`` cannot become a UI project, or None when it can: flat videos, or Insta360 X3
    footage with at least one clip 360 analysis can use (never both in one folder).

    Raises:
        OSError: If the source cannot be listed.
    """
    if src.is_file() and src.suffix.lower() not in VIDEO_EXTS | PANO_EXTS:
        return "source is not a video file"
    if not src.is_file() and not src.is_dir():
        return "source must be a video file or a folder containing videos"
    videos, clips, other = inspect_source(src)
    names = "Insta360 files must keep their camera names (VID_..._00_NNN.insv)."
    if not clips and not videos:
        if not other:
            return "source must be a video file or a folder containing videos"
        if src.is_file():
            return names
        hint = f". {names}" if PANO_EXTS & other.keys() else ""
        return f"no supported videos found (found {_suffix_counts(other)}){hint}"
    if not clips:
        return None
    if videos:
        flat = _suffix_counts(Counter(p.suffix.lower() for p in videos))
        return (
            f"This folder mixes 360 footage ({_plural(len(clips), 'Insta360 clip')}) and regular "
            f"videos ({flat}). Put them in separate folders and make one project for each."
        )
    if not _usable(clips):
        parts = Counter(k for g in clips.values() for k in ("lens00", "lens10", "lrv") if k in g)
        label = {"lens00": "_00_.insv", "lens10": "_10_.insv", "lrv": ".lrv"}
        found = " and ".join(f"{parts[k]} {label[k]}" for k in label if parts[k])
        missing = " or ".join(
            label[k] for k in ("lens00", "lrv") if any(k not in g for g in clips.values())
        )
        nums = sorted(clips)
        named = ", ".join(nums[:5]) + (f" and {len(nums) - 5} more" if len(nums) > 5 else "")
        return (
            f"Found {found} but no matching {missing} (clip{'s' if len(nums) > 1 else ''} "
            f"{named}): 360 analysis needs both files of each clip."
        )
    return None


def inputs_problem(srcs: list[Path], exclude: list[str], recursive: bool) -> str | None:
    """Why several inputs, or one with ``--exclude`` or ``--recursive``, cannot become a UI
    project, or None when they can: each is a video file or a folder, none holds Insta360 footage
    (a 360 project takes one folder or file and neither option, as ``360 analyze``), and
    ``analyze`` would find at least one video in them.

    Raises:
        OSError: If an input cannot be listed.
    """
    for src in srcs:
        if src.is_file() or src.is_dir():
            clips = inspect_source(src)[1]
            # a folder may hold its videos in sub-folders only (--recursive)
            problem = source_problem(src) if src.is_file() or clips else None
        else:
            clips, problem = {}, "source must be a video file or a folder containing videos"
        if problem:
            return f"{src}: {problem}" if len(srcs) > 1 else problem
        if clips:
            return (
                f"{src} holds 360 footage (Insta360), which makes a project of its own: choose "
                "just its folder or file, without sub-folders or exclude patterns."
            )
    if find_picked([str(s) for s in srcs], exclude, recursive):
        return None
    left = " that the exclude patterns leave" if exclude else ""
    deeper = "" if recursive else " (sub-folders are searched only with --recursive)"
    return f"no videos found{left} in the chosen sources{deeper}"


def _body_inputs(body: dict[str, Any]) -> list[str]:
    """``source`` (one path) or ``sources`` (1 to ``MAX_INPUTS``, blank lines left out): absolute
    paths, each once."""
    source, sources = body.get("source"), body.get("sources")
    if sources is None:
        sources = [source]
    elif source is not None:
        raise bad("send source or sources, not both")
    elif isinstance(sources, list):
        sources = [x for x in sources if not isinstance(x, str) or x.strip()]
    if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_INPUTS:
        raise bad(f"sources must list 1-{MAX_INPUTS} paths")
    sources = [_typed_path(x) for x in sources]
    for x in sources:
        if not (isinstance(x, str) and "\0" not in x and Path(x).is_absolute()):
            name = f"{x}: " if len(sources) > 1 and isinstance(x, str) else ""
            raise bad(f"{name}source must be an absolute path")
    if len({os.path.normcase(os.path.abspath(x)) for x in sources}) < len(sources):  # noqa: PTH100
        raise bad("each source can be chosen once")
    return sources


def _exclude_globs(body: dict[str, Any]) -> list[str]:
    """``exclude``: up to ``MAX_EXCLUDES`` ``analyze --exclude`` patterns, one line each."""
    v = body.get("exclude", [])
    if (
        not isinstance(v, list)
        or len(v) > MAX_EXCLUDES
        or not all(
            isinstance(g, str) and g.strip() and len(g) <= MAX_GLOB and not set(g) & set("\0\r\n")
            for g in v
        )
    ):
        raise bad(f"exclude must be up to {MAX_EXCLUDES} patterns of 1-{MAX_GLOB} characters")
    return v
