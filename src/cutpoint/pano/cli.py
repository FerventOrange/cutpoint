"""``cutpoint 360 scan | analyze | status | director | stitch | check | qa | repair``.

Optional 360-camera (Insta360 X3) module. Typical flow (rendering itself
happens in ``cutpoint clips`` / ``cutpoint extend``)::

    cutpoint 360 scan /media/x3/360 --out trip-360               # pano.json, IMU, calibration
    cutpoint 360 analyze trip-360 --style ski                     # views + analyze + events
    cutpoint clips trip-360 --max-total 150                       # 360 clips rendered flat
    cutpoint extend projects/trip-360/clips --seconds 4
    cutpoint assemble projects/trip-360/clips_ext4s --loudnorm

On Windows, paths such as ``D:\\Videos\\360`` and ``projects\\trip-360\\clips`` work too.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cutpoint import cache

logger = logging.getLogger("cutpoint")
# Event layers ``cutpoint 360 analyze`` runs after the analysis.
EVENT_LAYERS = ("audio", "voice", "visual", "crash")


def _project(arg: str) -> Path:
    from cutpoint.config import resolve_project_path  # noqa: PLC0415

    return resolve_project_path(arg)


def _clips_dir(project: Path, arg: str) -> Path:
    """A ``--from-clips`` folder: resolved like other clip-folder arguments
    (absolute, relative to the current folder, ``projects/...``), else
    ``<project>/<arg>``.

    Raises:
        FileNotFoundError: If none of these is an existing folder.
    """
    from cutpoint.config import resolve_project_path  # noqa: PLC0415

    path = resolve_project_path(arg, bare_names=False)
    if path.is_dir():
        return path
    if (project / arg).is_dir():
        return project / arg
    raise FileNotFoundError(f"clips folder not found: {path}")


def _segments_file(project: Path, arg: str) -> Path:
    """A ``--segments`` file: resolved like ``--from-clips``, else ``<project>/<arg>``."""
    from cutpoint.config import resolve_project_path  # noqa: PLC0415

    path = resolve_project_path(arg, bare_names=False)
    return path if path.is_file() or not (project / arg).is_file() else project / arg


def _no_pano(project: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"{project / 'pano.json'} not found or empty; run "
        "`cutpoint 360 scan <camera folder> --out <project>` first"
    )


def _clips(project: Path) -> list[Any]:
    from cutpoint.pano.project import load  # noqa: PLC0415

    clips = list(load(project).values())
    if not clips:
        raise _no_pano(project)
    return sorted(clips, key=lambda c: (c.start_local, c.clip_id))


def _timing(project: Path, key: str, value: Any) -> None:
    path = project / "pano_timings.json"
    data = cache.load_json(path) or {}
    data[key] = value
    cache.save_json(path, data)


# --------------------------------------------------------------------------- commands
def cmd_scan(a: argparse.Namespace) -> None:
    """Discover clips, read trailers and IMU orientation, write pano.json."""
    from cutpoint.pano import geometry as G  # noqa: PLC0415
    from cutpoint.pano import project as P  # noqa: PLC0415
    from cutpoint.pano.x3 import imu_source, scan  # noqa: PLC0415

    project = _project(a.out)
    t0 = time.perf_counter()
    clips, dirs = [], []
    for d in a.folders:
        found = scan(Path(d))
        clips += found
        dirs.append(Path(d))
    for c in clips:
        i00 = imu_source(Path(c.source))
        if i00 is None:
            continue
        o = G.orientation(Path(c.source), i00, c.fps, max(c.frames, 1), force=a.force)
        tilt = [o.tilt_deg(t) for t in o.t[:: max(1, len(o.t) // 50)]]
        c.meta["orientation"] = {
            "imu_hz": o.meta.get("imu_hz"),
            "median_tilt_deg": round(float(sorted(tilt)[len(tilt) // 2]), 1),
        }
    path = P.save(project, clips, dirs)
    el = round(time.perf_counter() - t0, 1)
    _timing(project, "scan_s", el)
    logger.info(
        "%-4s %-5s %8s %7s %9s %6s  %s",
        "clip",
        "mode",
        "duration",
        "frames",
        "insv GB",
        "tilt",
        "proxy",
    )
    for c in clips:
        logger.info(
            "%-4s %-5s %7.1fs %7d %9.2f %5.0f°  %s",
            c.clip_id,
            c.mode,
            c.duration,
            c.frames,
            c.bytes_insv / 1e9,
            c.meta.get("orientation", {}).get("median_tilt_deg") or 0,
            Path(c.lrv).name if c.lrv else "(none)",
        )
    total = sum(c.duration for c in clips)
    logger.info(
        "%d clips, %.1f min, %.1f GB of .insv; wrote %s in %.1fs",
        len(clips),
        total / 60,
        sum(c.bytes_insv for c in clips) / 1e9,
        path,
        el,
    )


def cmd_analyze(a: argparse.Namespace) -> None:
    """Proxy views, then the normal ski analysis + events on the .lrv files."""
    from cutpoint.analyze import analyze, reanalyze_if_reachable  # noqa: PLC0415
    from cutpoint.pano import views  # noqa: PLC0415
    from cutpoint.styles import get_style  # noqa: PLC0415

    project = _project(a.project)
    clips = [c for c in _clips(project) if c.lrv]
    if a.only:
        clips = [c for c in clips if c.clip_id in a.only]
    if not (a.no_events or a.views_only):  # fail on a missing extra / model before the long steps
        from cutpoint.events.pipeline import preflight  # noqa: PLC0415

        preflight(EVENT_LAYERS)
    t0 = time.perf_counter()
    st = views.run_layer(clips, force=a.force_views)
    _timing(project, "views", {**st, "total_s": round(time.perf_counter() - t0, 1)})
    if a.views_only:
        return
    t1 = time.perf_counter()
    from cutpoint.pano.project import default_style  # noqa: PLC0415

    analyze(
        [Path(c.lrv) for c in clips],
        get_style(a.style or default_style()),
        project,
        vision=False,
        model=a.model,
        language=a.language,
        use_scenes=False,
    )
    _timing(project, "analyze_s", round(time.perf_counter() - t1, 1))
    if a.no_events:
        return
    from cutpoint.events.pipeline import run_events  # noqa: PLC0415

    t2 = time.perf_counter()
    run_events(project, layers=EVENT_LAYERS)
    rescored = reanalyze_if_reachable(project)
    _timing(project, "events_s", round(time.perf_counter() - t2, 1))
    _timing(project, "analyze_total_s", round(time.perf_counter() - t0, 1))
    if rescored is None:  # footage went offline: highlights.json lacks the events
        logger.info(
            "360 analysis done in %.0fs without the events in highlights.json; "
            "rerun cutpoint 360 analyze %s once the footage is back",
            time.perf_counter() - t0,
            a.project,
        )
        return
    logger.info(
        "360 analysis done in %.0fs; next: cutpoint clips %s", time.perf_counter() - t0, a.project
    )


def _log_sdk_status() -> None:
    from cutpoint.pano.stitch import sdk_status  # noqa: PLC0415

    s = sdk_status()
    origin = {
        "env": "CUTPOINT_INSTA360_SDK",
        "config": "pano.sdk_exe",
        "auto": "auto-detected",
    }.get(s["origin"], s["origin"])
    if s["exe"] is None:
        logger.info(
            "SDK: not found (set pano.sdk_exe / CUTPOINT_INSTA360_SDK, or extract it "
            "under <CUTPOINT_HOME>/sdk/insta360/)"
        )
    elif not s["found"]:
        logger.info("SDK: %s (%s): MISSING", s["exe"], origin)
    else:
        logger.info(
            "SDK: %s (%s), version %s, %s",
            s["exe"],
            origin,
            s["version"] or "unknown",
            f"runs (-help in {s['help_seconds']:.1f}s)"
            if s["runs"]
            else f"does NOT run: {s['error']}",
        )


def cmd_status(a: argparse.Namespace) -> None:
    """The SDK, then per clip: files, cached layers and the render source that would be used."""
    from cutpoint.pano.project import find_studio_export, sdk_output  # noqa: PLC0415
    from cutpoint.pano.views import load_views  # noqa: PLC0415

    _log_sdk_status()
    if not a.project:
        return
    project = _project(a.project)
    for c in _clips(project):
        sdk = sdk_output(c)
        studio, why = find_studio_export(c)
        mode = "sdk" if sdk.is_file() else "studio" if studio else "dfisheye (preview)"
        pv = load_views(Path(c.source)) if c.lrv else None
        logger.info(
            "%s %-5s %7.1fs views:%-4s render: %-18s %s",
            c.clip_id,
            c.mode,
            c.duration,
            "yes" if pv is not None else "no",
            mode,
            studio or (sdk if sdk.is_file() else why),
        )


def _selected_segments(project: Path, a: argparse.Namespace) -> list[dict[str, Any]]:
    from cutpoint.clips import load_segments  # noqa: PLC0415
    from cutpoint.planning import select_moments  # noqa: PLC0415

    if a.segments:
        return load_segments(_segments_file(project, a.segments))
    chosen, _ = select_moments(project, top=a.top, ids=a.ids)
    return chosen


def director_params(items: list[str] | None) -> dict[str, Any] | None:
    """User director settings: ``pano.director`` in config.json, then ``--param k=v``
    (360 director only), validated; None when neither sets anything (the defaults).

    They are applied per segment after the ``kind`` tweaks (``compute_path(overrides=)``).

    Raises:
        ValueError: On an unknown field, a bad value or a non-object ``pano.director``.
    """
    from cutpoint.pano import director as D  # noqa: PLC0415

    merged = {**D.config_overrides(), **D.parse_param_args(items or [])}
    if not merged:
        return None
    p = D.with_params(D.DirectorParams(), merged)  # validate once, up front
    logger.info("director params: %s", ", ".join(f"{k}={getattr(p, k)}" for k in merged))
    return merged


def _plan_key(clip_id: str, start: float, end: float, kind: Any, vertical: bool, p: Any) -> str:
    """What a director path depends on (besides the clip's cached layers)."""
    seg = [clip_id, round(start, 3), round(end, 3), kind, vertical]
    return json.dumps([seg, p], sort_keys=True)


def _reusable(out_dir: Path) -> dict[str, tuple[str, dict[str, Any]]]:
    """Plan key -> ``(stem, path)`` of the current-version path JSONs in ``out_dir``."""
    from cutpoint.pano.director import PATH_VERSION  # noqa: PLC0415

    found = {}
    for f in sorted(out_dir.glob("[0-9][0-9][0-9]_*.json")):
        path = cache.load_json(f)
        if not isinstance(path, dict) or path.get("version") != PATH_VERSION or "vl" in path:
            continue
        seg = path.get("segment") or {}
        key = _plan_key(
            path.get("clip"),
            seg.get("start", -1),
            seg.get("end", -1),
            path.get("kind"),
            path.get("vertical"),
            path.get("params"),
        )
        found[key] = (f.stem, path)
    return found


def _without_vl(path: Any) -> Any:
    """``path`` without its ``vl`` answers (they and their timing differ on every run)."""
    return {k: v for k, v in path.items() if k != "vl"} if isinstance(path, dict) else path


def _save_path(
    out_dir: Path, stem: str, path: dict[str, Any], rendering: bool, carried: set[str]
) -> Path | None:
    """Write ``<stem>.json``; a preview of another path under that name goes unless re-rendered.
    Another path already in that file (it may be hand-edited) is first copied to the first free
    backup name (``<file>.bak``, then ``<file>.1.bak``, ...) and that name returned (None when
    there is none); the same path (``vl`` answers aside) is not backed up. Only a stem not in
    ``carried`` (its path is reused under a new stem this run) logs the backup: the caller
    drops a carried one once the run has saved every path."""
    from cutpoint.pano.render import backup_path_file  # noqa: PLC0415

    f = out_dir / f"{stem}.json"
    old, new = cache.load_json(f), json.loads(json.dumps(path))
    changed = old != new
    bak = None
    if changed and f.is_file() and _without_vl(old) != _without_vl(new):
        bak = backup_path_file(f)
        if stem not in carried:
            logger.warning("%s  re-planned; the previous path is kept as %s", stem, bak)
    if not rendering and changed:
        (out_dir / f"{stem}.mp4").unlink(missing_ok=True)
    cache.save_json(f, path, indent=1)
    return bak


def _drop_stale(out_dir: Path, stems: set[str], carried: set[str]) -> None:
    """Remove the files of stems the last director run wrote to ``out_dir`` that are not in
    ``stems``, then record ``stems`` (``director_stems.json``; no record, nothing removed).
    A path JSON (it may be hand-edited) is first copied to the first free backup name
    (``<file>.bak``, ...) unless its stem is in ``carried`` (reused under a new stem); the
    preview and its ``.cmd`` are rebuilt from it, so they just go. A stem whose backup or
    removal fails stays recorded, so the next run retries it."""
    from cutpoint.pano.render import backup_path_file  # noqa: PLC0415

    record = out_dir / "director_stems.json"
    left: set[str] = set()  # stems with a file still there
    for stem in sorted(set(cache.load_json(record) or []) - stems):
        for f in (out_dir / f"{stem}{ext}" for ext in (".json", ".cmd", ".mp4")):
            if not f.is_file():
                continue
            if f.suffix == ".json" and stem not in carried:
                try:
                    bak = backup_path_file(f)
                except OSError as err:
                    logger.warning("could not back up %s; keeping it: %s", f.name, err)
                    left.add(stem)
                    continue
                logger.warning("%s  no longer selected; the path is kept as %s", stem, bak)
            try:
                f.unlink()
                logger.info("removed stale %s", f.name)
            except OSError as err:
                logger.warning(
                    "could not remove %s (in use?); will retry next run: %s", f.name, err
                )
                left.add(stem)
    cache.save_json(record, sorted(stems | left))


def cmd_director(a: argparse.Namespace) -> None:
    """Plan (and optionally preview) director paths without the full render."""
    from cutpoint.ffmpeg_utils import output_dir  # noqa: PLC0415
    from cutpoint.pano.project import load  # noqa: PLC0415

    project = _project(a.project)
    clips = load(project)
    if not clips:  # before any folder is created
        raise _no_pano(project)
    overrides = director_params(a.param)
    name = a.out_name or ("director_vertical" if a.vertical else "director")
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"--out-name must be a plain folder name, got {name!r}")
    out_dir = project / "pano_paths" / name
    with output_dir(out_dir):  # a failed run leaves no empty folder behind
        _direct(a, project, clips, overrides, out_dir)


def _direct(
    a: argparse.Namespace,
    project: Path,
    clips: dict[str, Any],
    overrides: dict[str, Any] | None,
    out_dir: Path,
) -> None:
    """The body of :func:`cmd_director`, writing into the existing ``out_dir``."""
    from cutpoint.pano import director as D  # noqa: PLC0415
    from cutpoint.pano.project import (  # noqa: PLC0415
        RenderSource,
        clip_for,
        resolve_source,
    )
    from cutpoint.pano.render import Format360, render_segment  # noqa: PLC0415
    from cutpoint.utils import slugify  # noqa: PLC0415

    checker = None
    if getattr(a, "vl_check", False):  # absent on hand-built namespaces
        from cutpoint.pano.vlcheck import VLChecker  # noqa: PLC0415

        checker = VLChecker()
        if not checker.available():
            logger.warning("--vl-check: local vision model unavailable; skipping the check")
            checker = None
    vl_report: list[dict[str, Any]] = []
    reuse = {} if checker is not None or getattr(a, "force", True) else _reusable(out_dir)
    segs = []
    for i, m in enumerate(_selected_segments(project, a), 1):
        clip = clip_for(clips, m["source"])
        if clip is None:
            logger.info("%s: not a 360 clip; skipped", m.get("id"))
            continue
        p = D.params_for(m.get("kind"))
        p = asdict(D.with_params(p, overrides) if overrides else p)
        key = _plan_key(clip.clip_id, m["start"], m["end"], m.get("kind"), a.vertical, p)
        segs.append((f"{i:03d}_{slugify(m.get('label', ''), 40)}", m, clip, key))
    stems = {stem for stem, *_ in segs}
    # old stems whose path is reused under a new stem (known up front: an earlier segment may
    # take over such a name; its backup is redundant once the run saved the path anew)
    carried = {reuse[k][0] for stem, _, _, k in segs if k in reuse and reuse[k][0] != stem}
    redundant: list[Path] = []  # backups of carried paths, dropped after the loop
    for stem, m, clip, key in segs:
        preview = out_dir / f"{stem}.mp4"
        old_stem, path = reuse.get(key, (None, None))
        fresh = (
            old_stem == stem
            and preview.is_file()
            and preview.stat().st_mtime >= (out_dir / f"{stem}.json").stat().st_mtime
        )
        if path is not None and (not a.preview or fresh):
            if old_stem != stem:
                bak = _save_path(out_dir, stem, path, rendering=False, carried=carried)
                if bak is not None and stem in carried:
                    redundant.append(bak)
            logger.info("%s  unchanged; reused %s", stem, old_stem)
            continue
        if path is None:  # a reused path under another name only needs its preview
            path = D.compute_path(
                clip,
                m["start"],
                m["end"],
                vertical=a.vertical,
                kind=m.get("kind"),
                overrides=overrides,
                suspects=checker is not None,
            )

        def render(
            path: Any = path, clip: Any = clip, m: dict[str, Any] = m, preview: Path = preview
        ) -> None:
            """Fast low-res render from the .lrv proxy (the .insv files when there is none)."""
            src = (
                RenderSource("proxy", (clip.lrv,), "dfisheye", False, True, "lrv proxy")
                if clip.lrv
                else resolve_source(clip, "dfisheye")
            )
            fmt = Format360(360, 640) if a.vertical else Format360(640, 360)
            try:
                rec = render_segment(
                    clip,
                    path,
                    m["start"],
                    m["end"],
                    preview,
                    fmt=fmt,
                    src=src,
                    cmd_every=1,
                    work_dir=out_dir,
                )
            except BaseException:  # a partial preview must never be reused
                preview.unlink(missing_ok=True)
                raise
            logger.info("   preview %s (%.1fs)", preview, rec["seconds"])

        checked = checker is not None  # its render is the preview
        if checker is not None:  # a failed render / check loses the check only
            from cutpoint.ffmpeg_utils import FFmpegError  # noqa: PLC0415
            from cutpoint.pano.vlcheck import check_preview  # noqa: PLC0415

            try:
                render()
                _, vl = check_preview(checker, preview, path, every=a.vl_every)
                path["vl"] = {"model": checker.model, **vl}
                vl_report.append({"stem": stem, "clip": clip.clip_id, **path["vl"]})
            except (FFmpegError, OSError) as err:
                logger.warning("   vl check of %s failed: %s", stem, err)
            finally:
                if not a.preview:
                    preview.unlink(missing_ok=True)
            if checker.dead:
                logger.warning("--vl-check: the vision model stopped answering; skipping the rest")
                checker = None
        bak = _save_path(out_dir, stem, path, rendering=a.preview, carried=carried)
        if bak is not None and stem in carried:
            redundant.append(bak)
        logger.info(
            "%s  clip %s %.1f-%.1f: %s", stem, clip.clip_id, m["start"], m["end"], D.describe(path)
        )
        if path.get("quality"):
            logger.info("   %s", D.describe_quality(path))
        if path.get("vl"):
            vl = path["vl"]
            logger.info(
                "   vl check: %d/%d frame(s) bad%s (%.1fs)",
                vl["bad"],
                vl["frames"],
                "".join(
                    f"; {f['t'] - m['start']:.1f}s "
                    f"{f['where'] if f['blocked'] else 'low info'} {f['what']}"
                    for f in vl["flags"]
                ),
                vl["seconds"],
            )
        if a.preview and not checked:  # after the path JSON: a failed render keeps it
            render()
    for bak in redundant:  # every carried path is saved under its new stem by now
        bak.unlink(missing_ok=True)
    if a.out_name:  # previews of earlier selections would mislead `360 qa`
        _drop_stale(out_dir, stems, carried)
    if vl_report:
        cache.save_json(out_dir / "vl_check.json", {"segments": vl_report}, indent=1)
        secs = [r["seconds"] for r in vl_report]
        logger.info(
            "vl check: %d segment(s), %d with bad frames, %.1fs per segment (max %.1fs)",
            len(secs),
            sum(1 for r in vl_report if r["bad"]),
            sum(secs) / len(secs),
            max(secs),
        )


def cmd_stitch(a: argparse.Namespace) -> None:
    """Stitch the clips that hold chosen moments with the Insta360 Media SDK."""
    from cutpoint.pano.project import clip_for, load  # noqa: PLC0415
    from cutpoint.pano.stitch import stitch_clips  # noqa: PLC0415

    project = _project(a.project)
    clips = load(project)
    chosen: dict[str, Any] = {}
    if a.clips:
        for c in clips.values():
            if c.clip_id in a.clips:
                chosen[c.clip_id] = c
    else:
        sources: list[str] = []
        if a.from_clips:
            man = cache.load_json(_clips_dir(project, a.from_clips) / "manifest.json") or {}
            sources = [e["source"] for e in man.get("clips", [])]
        elif a.segments:
            from cutpoint.clips import load_segments  # noqa: PLC0415

            sources = [m["source"] for m in load_segments(_segments_file(project, a.segments))]
        else:
            from cutpoint.planning import select_moments  # noqa: PLC0415

            sel, _ = select_moments(project, top=a.top, max_total=a.max_total)
            sources = [m["source"] for m in sel]
        for s in sources:
            c = clip_for(clips, s)
            if c is not None:
                chosen[c.clip_id] = c
    if not chosen:
        raise ValueError(
            "No 360 clips selected (use --clips NNN ..., --from-clips DIR, --segments FILE or "
            "a project with highlights.json)"
        )
    recs = stitch_clips(
        sorted(chosen.values(), key=lambda c: c.clip_id), force=a.force, dry_run=a.dry_run
    )
    for r in recs:
        logger.info("clip %s: %s %s", r["clip"], r["status"], r["output"])


def cmd_check(a: argparse.Namespace) -> None:
    """Contact sheets of rendered clips (seam, horizon, framing, pans) for review by eye."""
    from cutpoint.config import resolve_project_path  # noqa: PLC0415
    from cutpoint.pano.check import clip_sheets  # noqa: PLC0415

    clips_dir = resolve_project_path(a.clips_dir, bare_names=False)
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"clips folder not found: {clips_dir}")
    for p in clip_sheets(clips_dir, per_clip=a.per_clip):
        logger.info("wrote %s", p)


def cmd_qa(a: argparse.Namespace) -> None:
    """Rendered-frame direction QA (occlusion, close-ups, edge cuts, pans, horizon)."""
    from cutpoint.config import resolve_project_path  # noqa: PLC0415
    from cutpoint.pano.qa import log_report, run_qa  # noqa: PLC0415

    qa_dir = resolve_project_path(a.dir, bare_names=False)
    if not qa_dir.is_dir():
        raise FileNotFoundError(f"folder not found: {qa_dir}")
    if a.paths:
        from cutpoint.pano.qa_paths import run_paths_qa  # noqa: PLC0415

        log_report(run_paths_qa(qa_dir, fps=a.fps))
        return
    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except ValueError as err:
        raise ValueError(f"--size must be WIDTHxHEIGHT, got {a.size!r}") from err
    log_report(run_qa(qa_dir, fps=a.fps, size=(w, h), json_only=a.json_only))


def cmd_repair(a: argparse.Namespace) -> None:
    """Trim, drop or replace clips with blocking direction-QA problems."""
    from cutpoint.pano.repair import log_repair, repair  # noqa: PLC0415

    project = _project(a.project)
    out = Path(a.out) if a.out else None
    rep = repair(
        project,
        _segments_file(project, a.segments),
        _segments_file(project, a.qa),
        out=None if out is None else out if out.is_absolute() else project / out,
    )
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        log_repair(rep)


# --------------------------------------------------------------------------- parser
def _positive_float(text: str) -> float:
    """argparse type: a number greater than 0."""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r}: expected a number") from exc
    if not value > 0:
        raise argparse.ArgumentTypeError(f"{text!r}: must be greater than 0")
    return value


def add_parser(sub: Any) -> None:
    """Register ``cutpoint 360 ...`` on the main CLI's subparsers."""
    from cutpoint.styles import style_names  # noqa: PLC0415

    p = sub.add_parser(
        "360", help="optional 360-camera module (Insta360 X3): scan, analyze, director, stitch"
    )
    s3 = p.add_subparsers(dest="pano_command", required=True)

    sc = s3.add_parser(
        "scan",
        help="discover X3 clips (pairs _00_/_10_/.lrv, mode, IMU, "
        "lens calibration) and write <project>/pano.json",
    )
    sc.add_argument("folders", nargs="+", metavar="folder", help="camera folder(s), read-only")
    sc.add_argument("--out", required=True, help="project (bare name = projects folder)")
    sc.add_argument("--force", action="store_true", help="recompute the IMU orientation")
    sc.set_defaults(func=cmd_scan)

    an = s3.add_parser(
        "analyze",
        help="360 proxy views (CLIP, people, beauty per "
        "direction) + cutpoint analyze + cutpoint events (audio, voice, crash) "
        "on the .lrv",
    )
    an.add_argument("project")
    an.add_argument(
        "--style",
        choices=style_names(),
        default=None,
        help="style preset (default: pano.default_style in config.json, else ski)",
    )
    an.add_argument("--model", help="whisper model (default: medium on CUDA)")
    an.add_argument("--language", help="force transcription language, e.g. en")
    an.add_argument("--only", nargs="+", metavar="NNN", help="only these clip numbers")
    an.add_argument("--views-only", action="store_true", help="only the 360 view layer")
    an.add_argument("--no-events", action="store_true", help="skip cutpoint events")
    an.add_argument("--force-views", action="store_true", help="recompute the view layer")
    an.set_defaults(func=cmd_analyze)

    stt = s3.add_parser(
        "status",
        help="Media SDK path, version and health; with a project also per clip: "
        "cached layers and render source",
    )
    stt.add_argument("project", nargs="?", help="optional: also report its clips")
    stt.set_defaults(func=cmd_status)

    di = s3.add_parser(
        "director",
        help="plan camera paths (JSON) for selected moments; "
        "--preview renders 640x360 checks from the .lrv",
    )
    di.add_argument("project")
    g = di.add_mutually_exclusive_group()
    g.add_argument("--top", type=int)
    g.add_argument("--ids", nargs="+")
    g.add_argument("--segments")
    di.add_argument("--vertical", action="store_true")
    di.add_argument("--preview", action="store_true")
    di.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a DirectorParams field (360 director only; repeatable; applied after "
        "pano.director in config.json and the moment-kind tweaks)",
    )
    di.add_argument(
        "--out-name",
        metavar="NAME",
        help="write the paths / previews to <project>/pano_paths/NAME "
        "(default director, director_vertical)",
    )
    di.add_argument(
        "--vl-check",
        action="store_true",
        help="a local vision model (Ollama) checks a few frames of each path's preview for "
        "blocked or low-information views and reports them (QA only; about 30 s a segment)",
    )
    di.add_argument(
        "--vl-every",
        type=_positive_float,
        default=3.0,
        metavar="S",
        help="seconds between baseline checked frames; suspect frames are added (--vl-check)",
    )
    di.add_argument(
        "--force",
        action="store_true",
        help="recompute every path (default: reuse a path JSON in the folder planned for the "
        "same segment, kind and settings; with --preview only if its preview is in place)",
    )
    di.set_defaults(func=cmd_director)

    sti = s3.add_parser(
        "stitch",
        help="stitch whole clips with the Insta360 Media SDK "
        "(MediaSDKTest; only clips holding chosen moments)",
    )
    sti.add_argument("project")
    g2 = sti.add_mutually_exclusive_group()
    g2.add_argument("--clips", nargs="+", metavar="NNN")
    g2.add_argument("--from-clips", metavar="DIR", help="clips folder (its manifest)")
    g2.add_argument(
        "--segments", metavar="FILE", help="segments JSON, e.g. mix_segments.json (its sources)"
    )
    sti.add_argument("--top", type=int)
    sti.add_argument("--max-total", type=float)
    sti.add_argument("--force", action="store_true")
    sti.add_argument("--dry-run", action="store_true", help="print the SDK commands only")
    sti.set_defaults(func=cmd_stitch)

    ck = s3.add_parser("check", help="contact sheets of rendered 360 clips")
    ck.add_argument("clips_dir")
    ck.add_argument("--per-clip", type=int, default=6)
    ck.set_defaults(func=cmd_check)

    qa = s3.add_parser(
        "qa",
        help="direction QA of rendered 360 clips (clips folder or director preview "
        "folder): occlusion, close-ups, edge cuts, pans, horizon; writes <dir>_qa/ "
        "(--paths: geometric, from the paths only)",
    )
    qa.add_argument("dir")
    qa.add_argument("--fps", type=float, default=2.0, help="frames checked per second")
    qa.add_argument(
        "--paths",
        action="store_true",
        help="geometric QA from the director paths and the 360 view layer, no rendering "
        "(also on pano_paths/<name>); writes <dir>_paths_qa/",
    )
    qa.add_argument("--size", default="640x360", help="decode size WIDTHxHEIGHT")
    qa.add_argument("--json-only", action="store_true", help="skip the contact sheets")
    qa.set_defaults(func=cmd_qa)

    rp = s3.add_parser(
        "repair",
        help="turn direction QA into a better mix: trim clips to their longest clean part "
        "or drop them and let cutpoint mix pick replacements (exclusions kept in "
        "mix_exclude.json)",
    )
    rp.add_argument("project")
    rp.add_argument(
        "--segments",
        default="mix_segments.json",
        help="the mix segments file, or one not written by cutpoint mix "
        "([{source, start, end, label}]: clips are only trimmed or dropped, nothing is refilled)",
    )
    rp.add_argument(
        "--qa",
        required=True,
        metavar="FILE",
        help="qa.json of the director previews or rendered clips of --segments, e.g. "
        "pano_paths/mix_qa/qa.json or clips_mix_qa/qa.json",
    )
    rp.add_argument(
        "--out",
        metavar="FILE",
        help="write the repaired list here (relative = inside the project; default: over "
        "--segments, the old one kept as the first free <name>.bak.json, <name>.1.bak.json, ...)",
    )
    rp.add_argument("--json", action="store_true", help="print the report as JSON")
    rp.set_defaults(func=cmd_repair)
