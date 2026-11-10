"""Command-line interface: ``cutpoint edit | analyze | events | crashes | profile | clips |
extend | assemble | stage | 360 | setup | paths | capabilities | style | guide | skill | ui``.

Installed as the ``cutpoint`` console script.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import urllib.error
from collections.abc import Iterator, Sequence
from dataclasses import replace
from importlib.resources import files
from pathlib import Path, PurePath
from typing import NoReturn

from cutpoint import __version__, cache, jsonout, paths, runstate, stage
from cutpoint.assemble import (
    AUDIO_TRANSITIONS,
    OUTPUT_FORMATS,
    assemble,
    assemble_project,
    check_transitions,
    read_joins,
)
from cutpoint.boundaries import SpeechSettings
from cutpoint.category import FUNNY, HYPE
from cutpoint.clips import export_clips
from cutpoint.config import (
    COLOR_MODES,
    COMPRESS_PEAKS,
    analysis_skip,
    color_default,
    compress_peaks_default,
    get_config,
    ocr_det_skip_default,
    path_key,
    projects_dir,
    resolve_project_path,
    vision_setting,
    visual_index_default,
)
from cutpoint.edit import KINDS, edit_name, resolve_kind, run_edit
from cutpoint.encode import LUT_SUFFIXES, Lut, check_lut
from cutpoint.errors import FFmpegError, FolderBusyError
from cutpoint.events import paths as event_paths
from cutpoint.extend import extend_clips, manifest_project
from cutpoint.planning import Padding
from cutpoint.progress import SKIPPABLE
from cutpoint.styles import (
    ANALYZE_STYLE,
    AUTO,
    AutoRequest,
    Style,
    auto_fallback,
    get_style,
    style_names,
)
from cutpoint.utils import (
    command_line,
    input_name,
    parse_size,
    positive_float,
    positive_int,
    unit_float,
)

logger = logging.getLogger("cutpoint")

# Parser-time constants of modules imported only when their command runs, so that every
# ``cutpoint`` start does not load the action-cam and proxy code (tests check they match).
MIX_EXCLUDE_FILE = "mix_exclude.json"  # cutpoint.mix.EXCLUDE_FILE
PROXY_PROVIDERS = ("claude-code", "codex", "antigravity")  # cutpoint.assistant.cliproxy.PROVIDERS


def _vision(a: argparse.Namespace) -> bool | None:
    """``--vision`` / ``--no-vision`` as given, else ``vision`` in config.json; None when
    neither says (rating is then off, the footage detection asks about unsure sources)."""
    return vision_setting() if a.vision is None else a.vision


def _visual_index(a: argparse.Namespace) -> bool:
    """``--visual-index`` / ``--no-visual-index`` as given, else ``visual_index`` in config.json
    (default off)."""
    return visual_index_default() if a.visual_index is None else a.visual_index


def _ocr_det_skip(a: argparse.Namespace) -> bool:
    """``--ocr-det-skip`` / ``--no-ocr-det-skip`` as given, else ``ocr_det_skip`` in config.json
    (default off)."""
    return ocr_det_skip_default() if a.ocr_det_skip is None else a.ocr_det_skip


def _skip(a: argparse.Namespace) -> tuple[str, ...]:
    """``--skip`` as given, else ``analysis.skip`` in config.json (default none)."""
    return analysis_skip() if a.skip is None else tuple(a.skip)


def _compress_peaks(a: argparse.Namespace) -> str:
    """``--compress-peaks`` as given, else ``compress_peaks`` in config.json (default off)."""
    return compress_peaks_default() if a.compress_peaks is None else a.compress_peaks


def _add_compress_peaks_argument(p: argparse.ArgumentParser, what: str) -> None:
    """``--compress-peaks off|light|strong`` (see cutpoint.encode.PEAK_PRESETS)."""
    p.add_argument(
        "--compress-peaks",
        choices=COMPRESS_PEAKS,
        default=None,
        help=f"pull loud peaks (shouts, gunshots, mic bumps) down toward speech level and "
        f"even out loudness between clips, {what}; light or strong (default: compress_peaks "
        "in config.json, else off)",
    )


RECORDED = "Recorded in clips/manifest.json"  # where --lut and --color say the grade is kept


def _add_lut_arguments(p: argparse.ArgumentParser, recorded: str = RECORDED) -> None:
    """``--lut FILE`` and ``--lut-strength 0-1`` (see cutpoint.encode.lut_filter); ``recorded``
    says where the grade is recorded."""
    p.add_argument(
        "--lut",
        metavar="FILE",
        help=f"grade every clip with this 3D LUT ({' or '.join(LUT_SUFFIXES)}), applied to "
        "BT.709 video: use the LUT made for the footage's log or flat profile (e.g. GoPro "
        f"Flat to Rec.709). {recorded}; default: no grade",
    )
    p.add_argument(
        "--lut-strength",
        type=unit_float,
        metavar="0-1",
        help="how much of the --lut grade to apply, blended with the ungraded video (default: 1)",
    )


def _add_color_arguments(p: argparse.ArgumentParser, recorded: str = RECORDED) -> None:
    """``--color off|auto|underwater`` and ``--color-strength 0-1`` (see cutpoint.color);
    ``recorded`` says where the grade is recorded."""
    p.add_argument(
        "--color",
        choices=COLOR_MODES,
        default=None,
        help="automatic color correction per clip: auto (levels, white balance, vibrance) or "
        "underwater (also restores red); clips that already look good, log/flat footage (use "
        f"--lut) and 360 clips are left as they are. {recorded}; not with "
        "--lut (default: color in config.json, never for gaming footage, else off)",
    )
    p.add_argument(
        "--color-strength",
        type=unit_float,
        metavar="0-1",
        help="how much of the --color correction to apply (default: 1)",
    )


def _color(a: argparse.Namespace, lut: Lut | None) -> tuple[str | None, float]:
    """``--color`` (None: config.json's) and ``--color-strength``, checked against ``--lut``."""
    if lut is not None and a.color not in (None, "off"):
        raise jsonout.ArgumentError(f"--color {a.color} cannot be combined with --lut; use one")
    if lut is not None and a.color_strength is not None:
        raise jsonout.ArgumentError("--color-strength does not apply to --lut (use --lut-strength)")
    if a.color_strength is not None and (a.color or color_default()) == "off":
        raise jsonout.ArgumentError("--color-strength needs --color auto or underwater")
    return a.color, 1.0 if a.color_strength is None else a.color_strength


def _lut(a: argparse.Namespace) -> Lut | None:
    """The checked ``--lut`` grade (:func:`~cutpoint.encode.check_lut`), None without --lut."""
    if a.lut is None:
        if a.lut_strength is not None:
            raise jsonout.ArgumentError("--lut-strength needs --lut")
        return None
    strength = 1.0 if a.lut_strength is None else a.lut_strength
    return check_lut(resolve_project_path(a.lut, bare_names=False), strength)


def _add_skip_argument(p: argparse.ArgumentParser) -> None:
    """``--skip STEP [STEP ...]``: analysis steps left out (see cutpoint.progress)."""
    p.add_argument(
        "--skip",
        nargs="+",
        choices=SKIPPABLE,
        metavar="STEP",
        help="leave out analysis steps: reactions (sound events such as laughter), keyframes "
        "(game HUD text or, with a CUDA GPU, what the camera sees), crashes (action-cam "
        "footage) and/or dialogue (a text model reads the transcript for funny or heated talk; "
        "gaming), given after the input paths (e.g. analyze clip.mp4 --skip crashes); default: "
        "analysis.skip in config.json, else none",
    )


def _batch_size(text: str) -> int:
    """argparse type of ``--batch-size``: a whole number, 0 or more."""
    try:
        n = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r}: expected a whole number") from exc
    if n < 0:
        raise argparse.ArgumentTypeError(f"{text!r}: must be 0 or more")
    return n


def add_batch_size_argument(p: argparse.ArgumentParser, *, vlm: bool = False) -> None:
    """``--batch-size N`` (#109): each model is closed after N sources and loaded again;
    ``vlm``: the command runs the VLM layer (``events``), so its help names the windows."""
    windows = (
        " (the vision model at the next source after N VLM windows, so not within one recording)"
        if vlm
        else ""
    )
    p.add_argument(
        "--batch-size",
        type=_batch_size,
        default=0,
        metavar="N",
        help=f"close each model after N sources{windows} and load it again for the next "
        "ones, to give back memory on long runs; the results are the same (default: 0, keep "
        "it loaded)",
    )


def _analyze_project(a: argparse.Namespace) -> Path:
    """``analyze``'s project folder: ``--out``, else one input's name or several inputs'
    common parent folder's name."""
    targets = [Path(t) for t in a.inputs]
    if a.out:
        return resolve_project_path(a.out)
    name = (
        input_name(targets[0], "project")
        if len(targets) == 1
        else (Path(os.path.commonpath([t.resolve() for t in targets])).name or "multi")
    )
    return projects_dir() / name


def _cmd_analyze(a: argparse.Namespace) -> None:
    from cutpoint.analyze import analyze  # noqa: PLC0415

    targets = [Path(t) for t in a.inputs]
    out = _analyze_project(a)
    jsonout.set_project(out)
    if a.style == AUTO:  # detected; ambiguous: the best guess's style if gameplay, else General
        style: Style | AutoRequest = AutoRequest(HYPE, must_include_crash=a.must_include_crash)
    else:
        style = get_style(a.style)
        if a.must_include_crash is not None:
            style = replace(style, must_include_crash=a.must_include_crash)
    # A failed or interrupted analyze removes the (still empty) folders it created.
    analyze(
        targets,
        style,
        out,
        vision=_vision(a),
        model=a.model,
        language=a.language,
        use_scenes=not a.no_scenes,
        max_candidates=a.max_candidates,
        vision_top=a.vision_top,
        ollama_url=a.ollama_url,
        ollama_model=a.ollama_model,
        exclude=a.exclude or [],
        recursive=a.recursive,
        recover=a.recover,
        layers=True,
        skip=_skip(a),
        batch_size=a.batch_size,
        visual_index=_visual_index(a),
        ocr_det_skip=_ocr_det_skip(a),
    )


def _cmd_detect(a: argparse.Namespace) -> None:
    """``cutpoint detect``: the footage category of the inputs and the style ``--style auto``
    would pick; talk only for sources already transcribed (by an earlier analysis). The
    vision model is asked about unsure sources, as for ``--style auto`` (``--no-vision`` or
    ``"vision": false`` skips it)."""
    from cutpoint import category  # noqa: PLC0415
    from cutpoint.analyze import find_sources  # noqa: PLC0415
    from cutpoint.sources import source_segments  # noqa: PLC0415

    found = find_sources([Path(t) for t in a.inputs], a.exclude or [], a.recursive)
    with category.footage_vision(_vision(a), a.ollama_url, a.ollama_model) as vision:
        probes = [category.probe_source(s, vision) for s in found]
    speech = {}
    for src, p in zip(found, probes, strict=True):
        segments = source_segments(src)
        if segments is not None:
            speech[p.path] = round(category.speech_coverage(segments, p.duration), 3)
    det = category.decide(probes, speech)
    hype, fallback = category.choose_style(det, HYPE)
    funny, _ = category.choose_style(det, FUNNY)
    category.warn(det, hype, fallback)
    if jsonout.active():
        rec = det.record(HYPE, hype, fallback)
        keys = ("category", "ambiguous", "reason", "decided_by", "scores", "suggest", "sources")
        for key in keys:
            jsonout.output(key, rec[key])
        jsonout.output("pending", det.pending)
        jsonout.output("styles", {"analyze": hype, "funny-moments": funny})
        return
    for s in det.sources:
        print(
            f"{Path(s.path).name}: {category.label(s.category)}"
            f"{' (ambiguous)' if s.ambiguous else ''} - {s.reason}"
        )
    note = f" (by {', '.join(det.decided_by)})" if det.decided_by else ""
    if fallback:
        note = f" (ambiguous: the {category.fallback_label(det)} style)"
    print(
        f"Footage: {category.label(det.category)}{note}; analyze --style auto uses {hype}, "
        f"edit --kind funny-moments {funny}"
    )
    if det.pending:
        print("Talk is not checked: a source has no transcript yet (analyze transcribes it)")


def _edit_project(a: argparse.Namespace) -> Path:
    """``edit``'s project folder: ``--out``, else named after the first input."""
    return resolve_project_path(a.out) if a.out else projects_dir() / edit_name(Path(a.inputs[0]))


def _cmd_edit(a: argparse.Namespace) -> None:
    targets = [Path(t) for t in a.inputs]
    project = _edit_project(a)
    jsonout.set_project(project)
    jsonout.output("kind", a.kind)
    lut = _lut(a)  # before hours of analysis
    color, color_strength = _color(a, lut)
    run_edit(
        targets,
        project,
        resolve_kind(a.kind, style=a.style, top=a.top, max_total=a.max_total),
        vertical=a.vertical,
        force=a.force,
        events=a.events,
        vision=_vision(a),
        vision_top=a.vision_top,
        ollama_url=a.ollama_url,
        ollama_model=a.ollama_model,
        skip=_skip(a),
        visual_index=_visual_index(a),
        ocr_det_skip=_ocr_det_skip(a),
        compress_peaks=_compress_peaks(a),
        lut=lut,
        color=color,
        color_strength=color_strength,
        keep_clips=a.keep_clips,
        output_format=a.output_format,
    )


def _segments_where(text: str, given: PurePath, segments: Path) -> str:
    """What a missing ``--segments`` path ``text`` (parsed as ``given``) was relative to (#127).

    A path with a drive or a root (``C:x.json``, Windows ``\\x.json``) gets no suffix: it is
    shown as typed and not resolved against the current folder or the projects folder.
    """
    if given.anchor:
        return ""
    if segments == given:
        return f" (relative to the current folder {Path.cwd()})"
    return f" ({text}, relative to the projects folder {projects_dir()})"  # projects/<name>/...


def _segments_file(a: argparse.Namespace, *, mix: bool = False) -> Path | None:
    """``--segments`` resolved (None without it), checked to be a file unless ``mix``."""
    segments = resolve_project_path(a.segments, bare_names=False) if a.segments else None
    if segments is not None and not mix and not segments.is_file():
        if segments.is_dir():
            raise IsADirectoryError(f"{segments} is a folder, not a segments file")
        where = _segments_where(a.segments, Path(a.segments), segments)
        raise FileNotFoundError(f"Segments file not found: {segments}{where}")
    return segments


def _export_options(a: argparse.Namespace) -> dict:
    """``export_clips`` arguments of the export options (:func:`_add_selection_arguments` but
    ``--segments``, ``--mix`` and ``--order``, :func:`_add_export_arguments` but
    ``--vertical``, ``--force``, ``--audio-fade`` and ``--compress-peaks``), ``--lut`` and
    ``--color`` checked."""
    lut = _lut(a)
    color, color_strength = _color(a, lut)
    return {
        "top": a.top,
        "ids": a.ids,
        "max_total": a.max_total,
        "crop": a.crop,
        "smart_tail_on": a.smart_tail,
        "tail_min": a.tail_min,
        "tail_max": a.tail_max,
        "event_tails": a.event_tails,
        "event_window": a.event_window,
        "extend": Padding.from_args(a.extend, a.extend_start, a.extend_end),
        "speech_aware": a.speech_aware,
        "speech_max_extend": a.speech_max_extend,
        "speech_max_lead": a.speech_max_lead,
        "pano_source": a.pano_source,
        "pano_cmd_every": a.pano_cmd_every,
        "size": a.size,
        "lut": lut,
        "color": color,
        "color_strength": color_strength,
    }


def _cmd_clips(a: argparse.Namespace) -> None:
    project = resolve_project_path(a.project)
    jsonout.set_project(project)
    options = _export_options(a)
    segments = _segments_file(a, mix=a.mix)
    order = a.order
    if a.mix:  # cutpoint mix's selection, played in arc order
        from cutpoint.mix import run_mix  # noqa: PLC0415 - action-cam module

        segments, order = project / "mix_segments.json", a.order or "arc"
        if not segments.is_file():
            run_mix(project, out=segments)
    export_clips(
        project,
        order=order,
        segments=segments,
        out_dir=resolve_project_path(a.out, bare_names=False)
        if a.out
        else (project / ("clips_mix_vertical" if a.vertical else "clips_mix") if a.mix else None),
        vertical=a.vertical,
        force=a.force,
        audio_fade=a.audio_fade,
        compress_peaks=_compress_peaks(a),
        **options,
    )


def _set_clips_project(clips_dir: Path) -> None:
    """For --json: the project of a clips folder (its manifest's ``project``, else the parent)."""
    if jsonout.active():
        manifest = cache.load_json(clips_dir / "manifest.json")
        jsonout.set_project(
            manifest_project(manifest if isinstance(manifest, dict) else {}, clips_dir)
        )


def _cmd_extend(a: argparse.Namespace) -> None:
    clips_dir = resolve_project_path(a.clips_dir, bare_names=False)
    _set_clips_project(clips_dir)
    extend_clips(
        clips_dir,
        Padding.from_args(a.seconds, a.start, a.end),
        out_dir=resolve_project_path(a.out, bare_names=False) if a.out else None,
        force=a.force,
        speech=SpeechSettings.from_style(
            None, enabled=a.speech_aware, max_extend=a.speech_max_extend, max_lead=a.speech_max_lead
        ),
        audio_fade=a.audio_fade,
        pano_source=a.pano_source,
        pano_cmd_every=a.pano_cmd_every,
        clip_overrides=_clip_overrides(a.clip_override),
        reuse_dir=resolve_project_path(a.reuse, bare_names=False) if a.reuse else None,
    )


def _clip_overrides(items: list[list[str]] | None) -> dict[str, dict[str, float]] | None:
    """``--clip-override GLOB KEY=S`` pairs as ``{glob: {key: seconds}}``."""
    if not items:
        return None
    out: dict[str, dict[str, float]] = {}
    for pattern, assign in items:
        key, sep, value = assign.partition("=")
        if not sep:
            raise jsonout.ArgumentError(f"--clip-override {pattern} {assign}: expected KEY=SECONDS")
        try:
            out.setdefault(pattern, {})[key.strip().replace("-", "_")] = float(value)
        except ValueError:
            raise jsonout.ArgumentError(
                f"--clip-override {pattern} {assign}: {value!r} is not a number of seconds"
            ) from None
    return out


def _cmd_events(a: argparse.Namespace) -> None:
    from cutpoint import progress  # noqa: PLC0415
    from cutpoint.events.cli import run_args  # noqa: PLC0415

    with progress.run(progress.EVENTS_STEPS, quiet=True):  # steps in progress.json only
        run_args(a)


def _cmd_stage(a: argparse.Namespace) -> None:
    if not (a.inputs or a.list or a.clear):
        raise jsonout.ArgumentError(
            "cutpoint stage: give files/folders to stage (or --list / --clear)"
        )
    stage_dir = get_config().stage_dir
    jsonout.output("stage_dir", stage_dir)
    if a.list:
        entries = stage.list_stage()
        for e in entries:
            logger.info(
                "%s  %.2f GB  read %ss  <- %s",
                e["file"],
                e["size"] / 1e9,
                e.get("read_s"),
                e.get("source"),
            )
        jsonout.output(
            "files",
            [
                {
                    "source": e.get("source"),
                    "staged": stage_dir / e["file"],
                    "bytes": e["size"],
                    "read_s": e.get("read_s"),
                }
                for e in entries
            ],
        )
        return
    if a.clear:
        removed = stage.clear_stage(
            [Path(t) for t in a.inputs], exclude=a.exclude or [], recursive=a.recursive
        )
        jsonout.output("removed", removed)
        return
    summary = stage.stage_sources(
        [Path(t) for t in a.inputs], exclude=a.exclude or [], recursive=a.recursive
    )
    jsonout.output("files", summary["items"])
    jsonout.output("copied", summary["files"])
    for key in ("skipped", "bytes", "read_s", "mb_per_s"):
        jsonout.output(key, summary[key])


def _cmd_crashes(a: argparse.Namespace) -> None:
    from cutpoint import progress  # noqa: PLC0415
    from cutpoint.activity import ACTIVITY_FILE  # noqa: PLC0415 - action-cam modules
    from cutpoint.crash import CrashParams, detect_project  # noqa: PLC0415

    params = CrashParams(
        lead_s=a.lead, max_after_s=a.max_after, max_clip_s=a.max_clip, dense_top=a.dense_top
    )
    project = resolve_project_path(a.project)
    jsonout.set_project(project)
    # steps in progress.json only; they end before the re-rank below
    with progress.run(progress.CRASHES_STEPS, quiet=True):
        out = detect_project(
            project,
            params,
            dense=a.dense,
            sheets=a.sheets,
            force=a.force,
            activity_name=a.activity,
            context=a.context,
            refresh_profile=a.refresh_profile,
            batch_size=a.batch_size,
            ollama_url=a.ollama_url,
            ollama_model=a.ollama_model,
        )
    if jsonout.active():  # recorded before reanalyze, so a failure there still reports them
        data = cache.load_json(out) or {}
        cands = data.get("crashes") or []
        jsonout.output("crashes", out)
        jsonout.output("count", len(cands))
        jsonout.output(
            "candidates",
            [
                {k: c[k] for k in _CRASH_SUMMARY_KEYS if k in c}
                for c in cands  # strongest first, as in crashes.json
            ],
        )
        jsonout.output("activity_file", project / ACTIVITY_FILE)
        jsonout.output("profile_name", (data.get("activity") or {}).get("name"))
    if a.reanalyze:
        from cutpoint.analyze import reanalyze_if_reachable  # noqa: PLC0415

        if reanalyze_if_reachable(project) is None:
            jsonout.warn(
                "highlights_not_updated",
                "a source could not be found or reached; highlights.json left unchanged, "
                "rerun once it is back",
                [project / "highlights.json"],
            )


# The per-crash keys of crashes.json that ``crashes --json`` reports.
_CRASH_SUMMARY_KEYS = (
    "id",
    "source",
    "t0",
    "t1",
    "confidence",
    "confirmed",
    "origin",
    "clip",
    "sheet",
)


def _cmd_resume(a: argparse.Namespace) -> int:
    """Replay the run recorded in ``progress.json`` (#109); see :func:`_resume_command`."""
    project = resolve_project_path(a.project).absolute()  # before the chdir below
    jsonout.set_project(project)
    rec = runstate.resumable(project)
    argv, status = rec["argv"], rec.get("status")
    counts = runstate.step_counts(rec)
    jsonout.output("resume", {"argv": argv, "status_before": status, "done_before": counts})
    jsonout.output("resumed", False)
    if status == "running":  # read() turns the record of a dead process into interrupted
        raise FolderBusyError(
            f"A cutpoint run (process {rec.get('pid')} on {rec.get('host')}) is still running "
            f"on {project}. Wait until it has finished, or stop it, and resume again."
        )
    if status == "done":
        logger.info("%s: its last run (%s) finished; nothing to resume", project, rec["command"])
        return 0
    if missing := runstate.missing_sources(rec):
        message = f"{len(missing)} source(s) of the recorded run are not there now"
        logger.warning("%s: %s", message, ", ".join(missing))
        jsonout.warn("resume_sources_changed", message, missing)
    join = command_line
    if a.dry_run:
        print(f"cutpoint {join(argv)}")
        print(f"  in {rec['cwd']}, last run {status}")
        for name, c in counts.items():
            done = f"{c['done']}/{c['total']} done" if c["total"] is not None else ""
            print(f"  {name}: {c['state']}{', ' + done if done else ''}")
        return 0
    cut = argv.index("--") if "--" in argv else len(argv)  # resume's own --json decides
    tokens = [t for t in argv[:cut] if not _json_requested([t])] + argv[cut:]
    try:
        args = _parse(build_parser(), tokens)
    except UsageError as err:
        jsonout.fail(jsonout.BAD_ARGS, f"the recorded command line: {err.message}")
        return 2
    args.json_result = a.json_result
    here = Path.cwd()
    os.chdir(rec["cwd"])
    try:
        other = _run_project(args)
        if other is not None and path_key(str(other.absolute())) != path_key(
            str(project)
        ):  # a copied project: its record names the original
            message = f"{project / runstate.FILE} records a run on {other.absolute()}"
            logger.error("%s", message)
            jsonout.fail(jsonout.ERROR, message)
            return 1
        logger.info("Resuming in %s: cutpoint %s", rec["cwd"], join(argv))
        jsonout.output("resumed", True)
        with runstate.resuming(rec):
            return _run(args, argv)  # records progress.json as the recorded command
    finally:
        os.chdir(here)


def _resume_command(args: argparse.Namespace) -> str:
    """The ``--json`` envelope's ``command``: ``resume`` reports as the command it replays,
    once its ``progress.json`` is read (also for a finished run and ``--dry-run``); else
    (none, unreadable) as ``resume``."""
    if args.func is not _cmd_resume:
        return args.json_command
    try:
        return runstate.resumable(resolve_project_path(args.project))["command"]
    except (OSError, ValueError):
        return args.json_command


def _cmd_profile(a: argparse.Namespace) -> None:
    from cutpoint.activity import ACTIVITY_FILE, record_profile  # noqa: PLC0415 - action-cam
    from cutpoint.activity import resolve as resolve_activity  # noqa: PLC0415

    project = resolve_project_path(a.project)
    jsonout.set_project(project)
    rec = resolve_activity(
        project,
        activity=a.activity,
        context=a.context,
        refresh=a.refresh_profile,
        ollama_url=a.ollama_url,
        ollama_model=a.ollama_model,
    )
    shown = {**rec, "profile": record_profile(rec)}  # what `crashes` uses, not the stored file
    if jsonout.active():
        for key, value in shown.items():  # the record's own keys, as the old --json printed them
            jsonout.output(key, value)
        jsonout.output("activity_file", project / ACTIVITY_FILE)
        return
    inc = shown["profile"]["incident"]
    print(f"activity  {rec['name']} ({rec['source']})")
    if rec.get("description"):
        print(f"footage   {rec['description']}")
    print(f"incident  {len(inc['pos'])} prompts, {len(inc['neg'])} negatives, ", end="")
    print(f"{len(inc['words'])} words")


def _cmd_profile_list(a: argparse.Namespace) -> None:
    from cutpoint import activity  # noqa: PLC0415 - numpy and the profile files

    entries = activity.list_profiles()
    for e in entries:
        if not e["valid"]:
            message = f"profile {e['name']!r} cannot be used: {'; '.join(e['problems'])}"
            jsonout.warn("invalid_profile", message, [e["path"]])
    if jsonout.active():
        jsonout.output("profiles_dir", paths.profiles_dir())
        jsonout.output("profiles", entries)
        return
    rows = [("name", "type", "source", "status", "description")]
    for e in entries:
        status = "floor (every profile inherits it)" if e["floor"] else "ok"
        if not e["valid"]:
            status = "invalid: " + "; ".join(e["problems"])
        rows.append((e["name"], e["type"], e["source"] or "-", status, e["description"] or "-"))
    _table(rows)


def _cmd_profile_show(a: argparse.Namespace) -> None:
    import json  # noqa: PLC0415

    from cutpoint import activity  # noqa: PLC0415

    info = activity.show_profile(a.name)
    if jsonout.active():
        for key, value in info.items():
            jsonout.output(key, value)
        return
    kind = "built-in profile" if info["type"] == "built-in" else "your profile"
    extra = ", the floor every profile inherits" if info["floor"] else ""
    print(f"name: {info['name']} ({kind}, source {info['source'] or '-'}{extra})")
    if info["path"]:
        print(f"file: {info['path']}")
    if info["description"]:
        print(f"description: {info['description']}")
    print(json.dumps(info["profile"], indent=2, ensure_ascii=False))


def _read_profile_document(source: str) -> object:
    """The JSON of ``--from FILE`` (``-``: standard input)."""
    import json  # noqa: PLC0415

    if source == "-":
        text, where = sys.stdin.read(), "standard input"
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--from {source}: file not found")
        text, where = path.read_text(encoding="utf-8-sig"), str(path)
    try:
        return json.loads(text)
    except ValueError as err:
        raise jsonout.ArgumentError(f"--from {where}: not valid JSON ({err})") from None


def _cmd_profile_create(a: argparse.Namespace) -> None:
    from cutpoint import activity  # noqa: PLC0415

    existed = a.name in activity.user_files()
    path, warnings = activity.create_profile(
        a.name, _read_profile_document(a.from_file), force=a.force
    )
    for message in warnings:
        logger.warning("%s: %s", path, message)
        jsonout.warn("profile_entry_ignored", message, [path])
    logger.info("%s profile %r: %s", "replaced" if existed else "saved", a.name, path)
    jsonout.output("name", a.name)
    jsonout.output("path", path)
    jsonout.output("replaced", existed)
    jsonout.output("profile", activity.show_profile(a.name))


def _cmd_profile_validate(a: argparse.Namespace) -> None:
    from cutpoint import activity  # noqa: PLC0415

    res = activity.validate_profile(a.target)
    where = res["path"] or res["name"]
    for w in res["warnings"]:
        logger.warning("%s: %s", where, w["message"])
        jsonout.warn(w["code"], w["message"], [res["path"]] if res["path"] else [])
    for key, value in res.items():
        if key != "warnings":
            jsonout.output(key, value)
    if not res["valid"]:
        raise jsonout.ArgumentError(f"{where} is not a valid profile: {'; '.join(res['problems'])}")
    for note in res["notes"]:
        logger.info("%s: %s", where, note)
    logger.info("%s is a valid profile", where)


def _cmd_profile_delete(a: argparse.Namespace) -> None:
    from cutpoint import activity  # noqa: PLC0415

    path = activity.delete_profile(a.name)
    logger.info("deleted profile %r: %s", a.name, path)
    jsonout.output("name", a.name)
    jsonout.output("path", path)


def _add_activity_arguments(p: argparse.ArgumentParser) -> None:
    """``--activity``, ``--context``, ``--refresh-profile``."""
    p.add_argument(
        "--activity",
        metavar="NAME",
        help="activity profile of your library (cutpoint profile list) or generic; "
        "default: found in your library or discovered from the footage",
    )
    p.add_argument(
        "--context",
        metavar="TEXT",
        help="what the video is, in the user's words (e.g. 'enduro race, mud on the lens'); "
        "the local LLM turns it into extra prompts and slang",
    )
    p.add_argument(
        "--refresh-profile",
        action="store_true",
        help="re-detect / re-synthesize the profile (ignore <project>/activity.json and the "
        "profile discovered earlier for this footage, which a new one replaces)",
    )


def _cmd_mix(a: argparse.Namespace) -> None:
    from cutpoint.mix import parse_quotas, read_exclusions, run_mix  # noqa: PLC0415 - action-cam

    project = resolve_project_path(a.project)
    jsonout.set_project(project)
    out, excl = Path(a.out), Path(a.exclude_file)
    out = out if out.is_absolute() else project / out
    excl = excl if excl.is_absolute() else project / excl
    quotas = parse_quotas(a.quotas)
    exclude, ignored = read_exclusions(excl)
    if ignored:
        jsonout.warn(
            "exclusions_ignored",
            f"{len(ignored)} exclusion(s) in {excl} ignored: their source is relative to the "
            "current folder of a drive",
            [excl],
        )
    report = run_mix(project, target_s=a.target, quotas=quotas, out=out, exclude=exclude)
    if jsonout.active():
        _report_mix(out, report)


def _report_mix(out: Path, report: dict) -> None:
    """``--json`` outputs of ``mix``: the files written, the moments and their durations."""
    segs = cache.load_json(out) or []
    jsonout.output("segments", out)
    jsonout.output("report", out.with_name("mix_report.json"))
    jsonout.output("sheet", report.get("sheet"))  # only one this run wrote
    jsonout.output(
        "moments",
        [
            {
                "id": s.get("id"),
                "source": s["source"],
                "start": s["start"],
                "end": s["end"],
                "duration": round(s["end"] - s["start"], 3),
                "category": s.get("category"),
            }
            for s in segs
        ],
    )
    jsonout.output("count", len(segs))
    jsonout.output("total_seconds", report.get("total_s"))
    jsonout.output("target_seconds", report.get("target_s"))
    jsonout.output(
        "categories",
        {k: {"n": v["n"], "seconds": v["seconds"]} for k, v in report["categories"].items()},
    )
    qa = report.get("content_qa") or {}
    jsonout.output("content_qa_pass", qa.get("pass"))
    failed = [k for k, v in (qa.get("checks") or {}).items() if not v.get("ok")]
    if failed:
        jsonout.warn("content_qa_failed", f"mix content QA failed: {', '.join(failed)}", [out])


# The options of assemble's "from a project" group that also apply to a clips folder
CLIPS_FOLDER_OPTIONS = ("audio_fade", "compress_peaks")


def _assemble_project(a: argparse.Namespace, given: Path) -> Path | None:
    """The project ``assemble`` exports the moments of first, or None for a clips folder:
    with ``--segments``, the positional as ``clips`` reads its project; else ``given`` (or
    the bare name in the projects folder) when it holds highlights.json.

    Raises:
        jsonout.ArgumentError: For ``--segments`` with a clips folder (it has manifest.json),
            and for a project without highlights.json (a ``clips`` or ``clips_vertical``
            folder or project.json, and no manifest.json of its own).
    """
    candidates = (given, resolve_project_path(a.clips_dir))
    if a.segments:
        project = candidates[1]
        if (project / "manifest.json").is_file():
            raise jsonout.ArgumentError(
                f"{project} is a clips folder; --segments needs the project folder"
            )
        return project
    for candidate in candidates:
        if (candidate / "highlights.json").is_file():
            return candidate
    folder = next((c for c in candidates if c.is_dir()), None)
    if folder is not None and not (folder / "manifest.json").exists():
        marks = [folder / "clips", folder / "clips_vertical"]
        if any(m.is_dir() for m in marks) or (folder / "project.json").is_file():
            raise jsonout.ArgumentError(
                f"{folder} is a project without highlights.json: run analyze, pass --segments, "
                "or give its clips folder"
            )
    return None


def _cmd_assemble(a: argparse.Namespace) -> None:
    given = resolve_project_path(a.clips_dir, bare_names=False)
    out = resolve_project_path(a.out, bare_names=False) if a.out else None
    join = {
        "transition": a.transition,
        "fade_dur": a.fade_dur,
        "loudnorm": a.loudnorm,
        "captions": a.captions,
        "caption_model": a.model,
        "audio_fade": a.audio_fade,
        "compress_peaks": _compress_peaks(a),
        "audio_transition": a.audio_transition,
        "output_format": a.output_format,
        "joins": read_joins(resolve_project_path(a.joins, bare_names=False)) if a.joins else None,
    }
    project = _assemble_project(a, given)
    if project is None:
        if used := [d for d, default in a.project_defaults.items() if getattr(a, d) != default]:
            raise jsonout.ArgumentError(
                f"--{used[0].replace('_', '-')} needs a project (a folder with highlights.json) "
                f"or --segments; {given} is not one"
            )
        _set_clips_project(given)
        assemble(given, out, **join)
        return
    jsonout.set_project(project)
    export = {**_export_options(a), "segments": _segments_file(a), "order": a.order}
    assemble_project(project, out, export=export, vertical=a.vertical, force=a.force, **join)


def _check_assemble(parser: argparse.ArgumentParser, a: argparse.Namespace) -> None:
    """Refuse an ``--audio-transition`` that does not go with ``--transition``, and options of
    the final video with a timeline ``--format`` (exit 2)."""
    if (why := check_transitions(a.transition, a.audio_transition)) is not None:
        parser.error(f"--audio-transition {a.audio_transition}: {why}")
    if a.output_format != "mp4":
        given = [f"--{n}" for n in ("loudnorm", "captions", "model") if getattr(a, n)]
        if given:
            parser.error(
                f"{given[0]} applies to the final video; --format {a.output_format} writes "
                "only the timeline (use --format mp4)"
            )


def _check_edit(parser: argparse.ArgumentParser, a: argparse.Namespace) -> None:
    """Refuse ``--no-keep-clips`` where the clips are needed or nothing is replaced (exit 2)."""
    if a.keep_clips:
        return
    if a.kind == "clips":
        parser.error("--no-keep-clips: with --kind clips the clips are the deliverable")
    if a.output_format != "mp4":
        parser.error(
            f"--no-keep-clips: the --format {a.output_format} timeline references the clips, "
            "so they must be kept"
        )
    if a.force:
        parser.error("--force replaces kept clips; with --no-keep-clips there are none")


def _cmd_setup(a: argparse.Namespace) -> int:
    from cutpoint.events.setup import run_args as run_setup  # noqa: PLC0415

    if not jsonout.active():
        return run_setup(a)
    before = {name: _setup_state(_setup_files(name)) for name in a.only}
    try:
        return run_setup(a)
    finally:  # also after a failed step: what is there by then
        _report_setup(a.only, before, getattr(a, "_whisper_result", None))


def _setup_files(component: str) -> list[Path]:
    """The files ``cutpoint setup --only <component>`` installs (none for ``pip``)."""
    files = {
        "panns": lambda: [event_paths.panns_labels(), event_paths.panns_ckpt()],
        "visual": lambda: [
            event_paths.aesthetic_head(),
            event_paths.clip_weights(),
            event_paths.detector_weights(),
        ],
        "voice": lambda: [event_paths.silero_vad()],
    }.get(component)
    return files() if files else []


def _setup_state(files: list[Path]) -> dict[Path, tuple[int, int] | None]:
    """``{file: (size, mtime_ns), or None when it is not a file}``."""
    state: dict[Path, tuple[int, int] | None] = {}
    for f in files:
        try:
            st = f.stat()
            state[f] = (st.st_size, st.st_mtime_ns) if f.is_file() else None
        except OSError:
            state[f] = None
    return state


def _report_setup(
    components: list[str], before: dict[str, dict], whisper: dict[str, str] | None = None
) -> None:
    """``--json`` outputs of ``setup``: per component the files downloaded or found; for
    ``whisper`` the result of its step (``whisper``; None: it did not finish)."""
    from cutpoint.events.setup import EVENTS_MODULES  # noqa: PLC0415
    from cutpoint.extras import LITE, install_hint  # noqa: PLC0415

    report: dict[str, dict] = {}
    for name in dict.fromkeys(components):
        if name == "pip":
            missing = [m for m in EVENTS_MODULES if importlib.util.find_spec(m) is None]
            report[name] = {"installed": not missing, "missing": missing}
            if missing:
                jsonout.warn(
                    "events_extra_missing",
                    f"Missing Python packages for cutpoint events: {', '.join(missing)}; "
                    f"cutpoint events is {LITE}. To add the events extra: "
                    f"{install_hint('events')}",
                )
            continue
        if name == "whisper":
            report[name] = whisper or {"model": None, "path": None, "status": "missing"}
            continue
        files = []
        for path, now in _setup_state(_setup_files(name)).items():
            old = before.get(name, {}).get(path)
            status = "missing" if now is None else "found" if now == old else "downloaded"
            files.append({"path": path, "bytes": now[0] if now else None, "status": status})
        report[name] = {"files": files}
    jsonout.output("components", report)


def _cmd_paths(a: argparse.Namespace) -> None:
    cfg = get_config()
    rows = [
        ("home", paths.home()),
        ("config", paths.config_file()),
        ("cache", paths.cache_dir()),
        ("models", paths.models_dir()),
        ("tools", paths.tools_dir()),
        ("stage", cfg.stage_dir),
        ("styles", paths.styles_dir()),
        ("presets", paths.presets_dir()),
        ("profiles", paths.profiles_dir()),
        ("projects", cfg.projects_dir),
        ("bundled", paths.PRESETS_DIR),
    ]
    if jsonout.active():
        for key, path in rows:
            jsonout.output(key, path)
        jsonout.output("missing", [key for key, path in rows if not path.exists()])
        return
    width = max(len(k) for k, _ in rows)
    for key, path in rows:
        state = "" if path.exists() else "  (missing)"
        print(f"{key:<{width}}  {path}{state}")


def _cmd_capabilities(a: argparse.Namespace) -> None:
    from cutpoint import capabilities  # noqa: PLC0415 - probes ffmpeg, only when asked

    data = capabilities.report()
    if not jsonout.active():
        print(capabilities.format_report(data))
        return
    for key, value in data.items():  # the report's own keys, as the old --json printed them
        jsonout.output(key, value)
    torch_info = data.get("torch") or {}
    if torch_info.get("warning"):
        jsonout.warn("torch_no_cuda", f"{torch_info['warning']} Fix: {torch_info['fix']}")
    sdk = data.get("insta360_sdk") or {}
    if sdk.get("warning"):
        jsonout.warn("sdk_needs_nvidia", sdk["warning"])


def _table(rows: list[tuple[str, ...]]) -> None:
    """Print ``rows`` (the first is the header) as left-aligned columns."""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip())


def _applies_text(applies: dict[str, list[str]]) -> str:
    items = [f"category:{c}" for c in applies["categories"]]
    items += [f"profile:{p}" for p in applies["profiles"]]
    return " ".join(items) or "-"


def _cmd_style_list(a: argparse.Namespace) -> None:
    from cutpoint import stylefiles  # noqa: PLC0415

    entries = stylefiles.list_styles()
    for e in entries:
        if not e["valid"]:
            message = f"style {e['name']!r} cannot be used: {'; '.join(e['problems'])}"
            jsonout.warn("invalid_style", message, [e["path"]])
    if jsonout.active():
        jsonout.output("styles_dir", paths.styles_dir())
        jsonout.output("styles", entries)
        return
    rows = [("name", "type", "base", "source", "applies to", "status")]
    for e in entries:
        status = "legacy" if e["legacy"] else "ok"
        if not e["valid"]:
            status = "invalid: " + "; ".join(e["problems"])
        rows.append(
            (
                e["name"],
                e["type"],
                e["base"] or "-",
                e["source"] or "-",
                _applies_text(e["applies_to"]),
                status,
            )
        )
    _table(rows)


def _cmd_style_show(a: argparse.Namespace) -> None:
    from cutpoint import stylefiles  # noqa: PLC0415

    info = stylefiles.show(a.name)
    if jsonout.active():
        for key, value in info.items():
            jsonout.output(key, value)
        return
    kind = "built-in style" if info["type"] == "built-in" else "your style"
    extra = f", source {info['source']}" if info["source"] else ""
    print(f"name: {info['name']} ({kind}{', legacy preset' if info['legacy'] else ''}{extra})")
    print(f"base: {info['base']}")
    for label, key in (("file", "path"), ("description", "description"), ("profile", "profile")):
        if info[key]:
            print(f"{label}: {info[key]}")
    print(f"applies to: {_applies_text(info['applies_to'])}")
    print(f"hash: {info['hash']}")
    if info["kind"] == "action-cam":
        print(f"mask_friends: {info['mask_friends']:g}")
        print(f"must_include_crash: {info['must_include_crash']:g}")
    print("weights:")
    rows = [("  feature", "weight", "from")]
    rows += [(f"  {w['feature']}", f"{w['weight']:g}", w["from"]) for w in info["weights"]]
    _table(rows)


def _cmd_style_create(a: argparse.Namespace) -> None:
    from cutpoint import stylefiles  # noqa: PLC0415

    existed = a.name in stylefiles.user_files()
    path, res = stylefiles.create(
        a.name,
        base=a.base,
        weights=stylefiles.parse_weights(a.weight or []),
        description=a.description,
        applies_to=stylefiles.parse_applies(a.applies_to) if a.applies_to else None,
        profile=a.profile,
        from_file=Path(a.from_file).expanduser() if a.from_file else None,
        force=a.force,
    )
    for code, message in res.warnings:
        logger.warning("%s: %s", path, message)
        jsonout.warn(code, message, [path])
    logger.info("%s style %r: %s", "replaced" if existed else "saved", a.name, path)
    jsonout.output("name", a.name)
    jsonout.output("path", path)
    jsonout.output("replaced", existed)
    jsonout.output("style", stylefiles.show(a.name))


def _cmd_style_validate(a: argparse.Namespace) -> None:
    from cutpoint import stylefiles  # noqa: PLC0415

    res = stylefiles.validate(a.target)
    where = res["path"] or res["name"]
    for w in res["warnings"]:
        logger.warning("%s: %s", where, w["message"])
        jsonout.warn(w["code"], w["message"], [res["path"]] if res["path"] else [])
    for key, value in res.items():
        if key != "warnings":
            jsonout.output(key, value)
    if not res["valid"]:
        raise jsonout.ArgumentError(f"{where} is not a valid style: {'; '.join(res['problems'])}")
    for note in res["notes"]:
        logger.info("%s: %s", where, note)
    logger.info("%s is a valid style", where)


def _cmd_style_delete(a: argparse.Namespace) -> None:
    from cutpoint import stylefiles  # noqa: PLC0415

    path = stylefiles.delete(a.name)
    logger.info("deleted style %r: %s", a.name, path)
    jsonout.output("name", a.name)
    jsonout.output("path", path)


def _cmd_style_features(a: argparse.Namespace) -> None:
    from cutpoint import stylefiles  # noqa: PLC0415

    rows = stylefiles.feature_list()
    if jsonout.active():
        jsonout.output("features", rows)
        return
    table = [("feature", "layer", "available", "needs", "what it measures")]
    for f in rows:
        avail = "yes" if f["available"] else "no"
        table.append((f["name"], f["layer"], avail, f["extra"] or "-", f["description"]))
    _table(table)


def _cmd_guide(a: argparse.Namespace) -> None:
    # The wheel ships AGENTS.md inside the package; editable installs read the repo-root file.
    packaged = files("cutpoint") / "AGENTS.md"
    if packaged.is_file():
        data = packaged.read_bytes()
    else:
        data = (Path(__file__).resolve().parents[2] / "AGENTS.md").read_bytes()
    _write_stdout_bytes(data)


def _write_stdout_bytes(data: bytes) -> None:
    """Bytes to stdout as-is, so a non-UTF-8 console encoding cannot fail on non-ASCII text."""
    sys.stdout.flush()
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _cmd_skill_install(a: argparse.Namespace) -> None:
    from cutpoint import skill  # noqa: PLC0415

    skills_dir = Path(a.dir).expanduser() if a.dir else skill.default_dir()
    target, written = skill.install(skills_dir, force=a.force)
    if written:
        logger.info("installed the Cutpoint skill: %s", target)
    else:
        logger.info("the Cutpoint skill is already installed: %s", target)


def _cmd_skill_print(a: argparse.Namespace) -> None:
    from cutpoint import skill  # noqa: PLC0415

    _write_stdout_bytes(skill.skill_bytes())


def _cmd_proxy(a: argparse.Namespace) -> None:
    logger.warning("Use at your own risk. Subscription proxy connects to third-party endpoints.")
    from cutpoint.assistant import cliproxy  # noqa: PLC0415

    try:
        cliproxy.run(port=a.port)
    except cliproxy.ProxyError as err:
        logger.error("proxy failed: %s", err)
        sys.exit(1)


def _cmd_login(a: argparse.Namespace) -> None:
    logger.warning("Use at your own risk. Subscription login connects to third-party endpoints.")
    from cutpoint.assistant import cliproxy  # noqa: PLC0415

    if a.provider == "antigravity":
        try:
            ans = (
                input(
                    "Antigravity sign-in may provision a free-tier project if your account has none. Continue? [y/N]: "
                )
                .strip()
                .lower()
            )
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            logger.info("login aborted")
            sys.exit(1)

    try:
        st = cliproxy.login_interactive(a.provider, open_browser=not a.no_browser)
    except cliproxy.LoginError as err:
        logger.error("%s", err)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
    who = f" as {st.get('email')}" if st.get("email") else ""
    logger.info("logged in to %s%s", cliproxy.TITLES.get(a.provider, a.provider), who)


def _cmd_logout(a: argparse.Namespace) -> None:
    from cutpoint.assistant import cliproxy  # noqa: PLC0415

    try:
        cliproxy.logout(a.provider)
    except cliproxy.ProxyError as err:
        logger.error("logout failed: %s", err)
        sys.exit(1)
    logger.info("logged out of %s", cliproxy.TITLES.get(a.provider, a.provider))


def _port(text: str) -> int:
    """argparse type: a TCP port, 1-65535."""
    try:
        port = int(text)
    except ValueError:
        port = 0
    if not 0 < port < 65536:
        raise argparse.ArgumentTypeError(f"{text!r}: expected a port number, 1-65535")
    return port


def _cmd_ui(a: argparse.Namespace) -> None:
    from cutpoint.ui import run  # noqa: PLC0415

    run(a.port)


def _add_selection_arguments(p: argparse.ArgumentParser, *, mix: bool) -> None:
    """Which moments ``clips`` exports (``assemble <project>`` too): ``--top``, ``--ids``,
    ``--segments`` (and ``--mix`` with ``mix``), ``--max-total`` and ``--order``."""
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--top", type=positive_int, help="top N by score (default: fits style target)")
    sel.add_argument("--ids", nargs="+", help="explicit candidate ids, e.g. h001 h004")
    sel.add_argument(
        "--segments",
        help="JSON list of {source,start,end,label} exported "
        "exactly, in the given order (no highlights.json needed; --order arc reorders "
        "a cutpoint mix file)",
    )
    if mix:
        sel.add_argument(
            "--mix",
            action="store_true",
            help="(action-cam module) export <project>/mix_segments.json (runs "
            "`cutpoint mix` first if missing) in arc order",
        )
    p.add_argument(
        "--max-total",
        type=positive_float,
        help="cap total duration (s): best-scoring moments that fit, e.g. 60",
    )
    p.add_argument(
        "--order",
        choices=["chronological", "score", "story", "arc"],
        help="playback order (default: story; chronological for commentary"
        + (
            "; arc with --mix: scenic opener, the day by stretch, crash finale, scenic outro)"
            if mix
            else ")"
        ),
    )


def _add_export_arguments(
    p: argparse.ArgumentParser,
    *,
    force: str,
    audio_fade: str | None,
    peaks: str,
    recorded: str = RECORDED,
) -> None:
    """How ``clips`` exports the moments (``assemble <project>`` too): ``--vertical``, ``--size``,
    ``--force`` (help ``force``), ``--crop``, the tails, the padding, the speech options and
    ``--audio-fade`` (help ``audio_fade``; None: the export's), the 360, peak compression
    (``peaks``: what it applies to), LUT and colour options (``recorded``: where the grade is
    recorded)."""
    p.add_argument("--vertical", action="store_true", help="9:16 1080x1920 center crop")
    p.add_argument(
        "--size",
        type=parse_size,
        metavar="WxH",
        help="output resolution, e.g. 1920x1080 (default: the largest source's, "
        "capped at 1080p); ignored with --vertical",
    )
    p.add_argument("--force", action="store_true", help=force)
    p.add_argument(
        "--crop",
        metavar="auto|none|W:H:X:Y",
        help="crop before scaling: auto = cropdetect per source (cached); "
        "default: project.json per-source setting, else none",
    )
    p.add_argument(
        "--smart-tail",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="extend speech-ended clips until the reaction settles (default: on, per style)",
    )
    p.add_argument(
        "--tail-min",
        type=float,
        metavar="S",
        help="guaranteed tail after the last word (default: style, 1.5 s)",
    )
    p.add_argument(
        "--tail-max",
        type=float,
        metavar="S",
        help="max extension after the last word (default: style, 5 s)",
    )
    p.add_argument(
        "--event-tails",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="extend clips to cover a payoff event (kill, banner, laughter, "
        "VLM moment) that starts soon after the speech (default: on when "
        "<project>/events.json exists)",
    )
    p.add_argument(
        "--event-window",
        type=float,
        metavar="S",
        help="payoff must start within S s after the speech (default: 10)",
    )
    p.add_argument(
        "--extend",
        type=float,
        metavar="S",
        help="pad both ends of every clip by S seconds, after smart/event tails "
        "(clamped to the source and its readable range; default 0)",
    )
    p.add_argument(
        "--extend-start",
        type=float,
        metavar="S",
        help="padding before the start (overrides --extend)",
    )
    p.add_argument(
        "--extend-end", type=float, metavar="S", help="padding after the end (overrides --extend)"
    )
    _add_speech_arguments(p, "on, per style", audio_fade)
    _add_pano_arguments(p, "auto")
    _add_compress_peaks_argument(p, peaks)
    _add_lut_arguments(p, recorded)
    _add_color_arguments(p, recorded)


def _add_format_argument(p: argparse.ArgumentParser) -> None:
    """``--format mp4|fcpxml|otio`` (``output_format``, see cutpoint.assemble.OUTPUT_FORMATS)."""
    p.add_argument(
        "--format",
        dest="output_format",
        choices=OUTPUT_FORMATS,
        default="mp4",
        help="mp4: join the clips into the final video, without timelines; fcpxml (Final Cut "
        "Pro, DaVinci Resolve) or otio (DaVinci Resolve 18.5+, OpenTimelineIO tools): write "
        "only that timeline, referencing the kept clips, without joining or encoding; the "
        ".srt subtitles either way (default: mp4)",
    )


def _add_speech_arguments(
    p: argparse.ArgumentParser, default_on: str, audio_fade: str | None = None
) -> None:
    """``--speech-aware``, ``--speech-max-extend``, ``--speech-max-lead``, ``--audio-fade``
    (help ``audio_fade``; None: the fade baked into every clip)."""
    p.add_argument(
        "--speech-aware",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="move in/out points that fall mid-utterance (whisper words, voice "
        "activity, laughter) to the utterance start/end and keep clips of one "
        f"source from overlapping (default: {default_on})",
    )
    p.add_argument(
        "--speech-max-extend",
        type=float,
        metavar="S",
        help="max seconds an end moves later to finish an utterance (default 8)",
    )
    p.add_argument(
        "--speech-max-lead",
        type=float,
        metavar="S",
        help="max seconds a start moves earlier to catch an utterance's start (default 5)",
    )
    p.add_argument(
        "--audio-fade",
        type=float,
        default=0.25,
        metavar="S",
        help=audio_fade or "audio fade in/out baked into every clip (default 0.25; 0 = off)",
    )


def _add_pano_arguments(p: argparse.ArgumentParser, default: str | None) -> None:
    """``--pano-source`` / ``--pano-cmd-every`` for 360 (Insta360) sources."""
    p.add_argument(
        "--pano-source",
        choices=["auto", "sdk", "studio", "dfisheye"],
        default=default,
        help="360 clips: render from the SDK stitch, a Studio equirect export or "
        "the raw dual fisheye (preview, seam); auto = best available "
        f"(default: {default or 'the mode recorded per clip, else auto'})",
    )
    p.add_argument(
        "--pano-cmd-every",
        type=int,
        metavar="N",
        help="360 clips: update the virtual camera every N frames (default 1 = smoothest; "
        "higher = faster, less smooth)",
    )


def _add_vision_arguments(p: argparse.ArgumentParser, lead: str) -> None:
    """``--vision`` / ``--vision-top``; ``lead`` starts the ``--vision`` help."""
    p.add_argument(
        "--vision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"{lead}--no-vision skips it when config.json turns it on "
        "(default: vision in config.json, else off)",
    )
    p.add_argument(
        "--vision-top",
        type=positive_int,
        default=None,
        metavar="N",
        help="top candidates sent to vision (default: 1 per 2 min of footage, min 10, max 60)",
    )


def _add_visual_index_argument(p: argparse.ArgumentParser) -> None:
    """``--visual-index`` / ``--no-visual-index``."""
    p.add_argument(
        "--visual-index",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="also write the frame embeddings the AI Chat's visual search reads (CLIP, about "
        "one frame per second; needs the events extra and a CUDA GPU) for gameplay and talk "
        "footage; action, sport and action-cam footage always has them. It changes no "
        "ranking. --no-visual-index skips it when config.json turns it on "
        "(default: visual_index in config.json, else off)",
    )


def _add_ocr_det_skip_argument(p: argparse.ArgumentParser) -> None:
    """``--ocr-det-skip`` / ``--no-ocr-det-skip``."""
    p.add_argument(
        "--ocr-det-skip",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="gameplay HUD text: skip the text detection of a HUD region that has not "
        "changed since it last ran, which makes the read faster. The read is cached apart "
        "from one without it. --no-ocr-det-skip turns it off when config.json turns it on "
        "(default: ocr_det_skip in config.json, else off)",
    )


def _ollama_url_arg(text: str) -> str:
    """argparse type of ``--ollama-url``: the value, if it has a host and a valid port."""
    from cutpoint.backends import split_ollama_url  # noqa: PLC0415

    try:
        parts, _ = split_ollama_url(text.strip(), "--ollama-url")
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from None
    if not parts.hostname:
        raise argparse.ArgumentTypeError(f"--ollama-url {text!r}: no host")
    return text


def _add_ollama_arguments(p: argparse.ArgumentParser) -> None:
    """``--ollama-url`` / ``--ollama-model`` for the vision step."""
    p.add_argument(
        "--ollama-url",
        type=_ollama_url_arg,
        help="default: vision_assistant.url in config.json (when the Vision role is an Ollama), "
        "$OLLAMA_HOST, ollama_url in config.json, or http://127.0.0.1:11434",
    )
    p.add_argument(
        "--ollama-model",
        help="default: $CUTPOINT_VISION_MODEL, vision_assistant.model in config.json (when "
        "the Vision role is an Ollama), or qwen3.5:9b",
    )


def _add_source_filter_arguments(p: argparse.ArgumentParser) -> None:
    """``--exclude`` / ``--recursive``: which files of folder inputs are used."""
    p.add_argument(
        "--exclude",
        action="append",
        metavar="GLOB",
        help="skip files whose name or full path matches (repeatable), e.g. --exclude '*DVR*'",
    )
    p.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="search sub-folders of folder inputs (default: no)",
    )


def _add_json_argument(p: argparse.ArgumentParser) -> None:
    """``--json``: the command's result as one JSON object on stdout (see cutpoint.jsonout)."""
    p.add_argument(
        "--json",
        dest="json_result",
        action="store_true",
        help="print the result as one JSON object on stdout (logs stay on stderr)",
    )
    p.set_defaults(json_command=p.prog.split(" ", 1)[1])


class UsageError(SystemExit):
    """argparse's exit 2, with the parser and message (for the --json envelope)."""

    def __init__(self, parser: argparse.ArgumentParser, message: str, code: object) -> None:
        super().__init__(code)
        self.parser = parser
        self.message = message


class _Parser(argparse.ArgumentParser):
    """ArgumentParser whose errors (usage and message on stderr, exit 2) raise UsageError."""

    def error(self, message: str) -> NoReturn:
        if sys.stderr is None:  # argparse would print the usage to stdout instead
            raise UsageError(self, message, 2)
        try:
            super().error(message)
        except SystemExit as exc:
            raise UsageError(self, message, exc.code) from None


def _add_style_parsers(sub: argparse._SubParsersAction) -> None:
    """``cutpoint style list | show | create | validate | delete | features``."""
    sy = sub.add_parser(
        "style",
        help="list, show, create, check and delete scoring styles (your styles are saved in "
        "the styles folder, see cutpoint paths)",
    )
    verbs = sy.add_subparsers(dest="style_command", required=True)
    ls = verbs.add_parser(
        "list", help="built-in and your styles, with source, base, applies_to and status"
    )
    _add_json_argument(ls)
    ls.set_defaults(func=_cmd_style_list)

    sh = verbs.add_parser(
        "show", help="a style's effective weights (its base's, then its own), and their origin"
    )
    sh.add_argument("name", help="style name")
    _add_json_argument(sh)
    sh.set_defaults(func=_cmd_style_show)

    cr = verbs.add_parser(
        "create",
        help="save a style of your own: weights over a built-in style (source: hand)",
    )
    cr.add_argument(
        "name", help="new style name: letters, digits, - and _ (not a built-in name, not auto)"
    )
    cr.add_argument(
        "--base",
        metavar="STYLE",
        help="the built-in style it builds on (required unless the --from file has one)",
    )
    cr.add_argument(
        "--weight",
        nargs="+",
        action="extend",
        metavar="FEATURE=W",
        help="a feature weight over the base's (0 or more); see cutpoint style features",
    )
    cr.add_argument("--description", metavar="TEXT", help="what the style is for")
    cr.add_argument(
        "--applies-to",
        nargs="+",
        action="extend",
        metavar="KIND:NAME",
        help="category:NAME (action-cam, gameplay, talk, general) or profile:NAME; stored "
        "for later automatic use, a style is used by --style NAME",
    )
    cr.add_argument("--profile", metavar="NAME", help="activity profile name (stored)")
    cr.add_argument(
        "--from",
        dest="from_file",
        metavar="FILE",
        help="start from this style JSON (also a legacy preset); the options above replace "
        "its values and --weight adds to its weights; its source is kept",
    )
    cr.add_argument("--force", action="store_true", help="replace your style of this name")
    _add_json_argument(cr)
    cr.set_defaults(func=_cmd_style_create)

    va = verbs.add_parser(
        "validate", help="check a style or style file and say what is wrong (exit 1 if invalid)"
    )
    va.add_argument("target", metavar="NAME|FILE", help="style name or path to a .json file")
    _add_json_argument(va)
    va.set_defaults(func=_cmd_style_validate)

    de = verbs.add_parser("delete", help="delete one of your styles (built-ins cannot be)")
    de.add_argument("name", help="style name")
    _add_json_argument(de)
    de.set_defaults(func=_cmd_style_delete)

    fe = verbs.add_parser(
        "features", help="the features style weights can name, and which this install has"
    )
    _add_json_argument(fe)
    fe.set_defaults(func=_cmd_style_features)


# ``cutpoint profile VERB ...``; any other word after ``profile`` is a project.
PROFILE_VERBS = ("list", "show", "create", "validate", "delete")


def _profile_verb(tokens: Sequence[str]) -> bool:
    """Whether ``tokens`` (the command line) run ``cutpoint profile <verb>``: the first word
    that is not an option is ``profile`` and the next one a :data:`PROFILE_VERBS` verb. A
    project named like a verb is given as ``projects/<name>``."""
    words = list(tokens)
    i = next((k for k, t in enumerate(words) if not t.startswith("-")), None)
    return (
        i is not None
        and words[i] == "profile"
        and words[i + 1 : i + 2] in ([v] for v in PROFILE_VERBS)
    )


def _add_profile_verb_parsers(sub: argparse._SubParsersAction) -> None:
    """``cutpoint profile list | show | create | validate | delete`` (the profile library)."""
    pf = sub.add_parser(
        "profile",
        help="list, show, create, check and delete activity profiles (your profiles are "
        "saved in the profiles folder, see cutpoint paths)",
    )
    verbs = pf.add_subparsers(dest="profile_command", required=True)
    ls = verbs.add_parser(
        "list", help="the built-in base (the floor) and generic, then your profiles, with source"
    )
    _add_json_argument(ls)
    ls.set_defaults(func=_cmd_profile_list)

    sh = verbs.add_parser("show", help="a profile's captions, words and wording")
    sh.add_argument("name", help="profile name")
    _add_json_argument(sh)
    sh.set_defaults(func=_cmd_profile_show)

    cr = verbs.add_parser(
        "create", help="save a profile of your own from a JSON file (source: hand)"
    )
    cr.add_argument(
        "name",
        help="profile name: letters, digits, - and _ (not base or generic); it replaces the "
        "file's own name",
    )
    cr.add_argument(
        "--from",
        dest="from_file",
        metavar="FILE",
        required=True,
        help="the profile JSON (- reads standard input); see cutpoint profile show generic",
    )
    cr.add_argument("--force", action="store_true", help="replace your profile of this name")
    _add_json_argument(cr)
    cr.set_defaults(func=_cmd_profile_create)

    va = verbs.add_parser(
        "validate",
        help="check a profile or profile file and say what is wrong (exit 1 if invalid)",
    )
    va.add_argument("target", metavar="NAME|FILE", help="profile name or path to a .json file")
    _add_json_argument(va)
    va.set_defaults(func=_cmd_profile_validate)

    de = verbs.add_parser("delete", help="delete one of your profiles (base and generic cannot be)")
    de.add_argument("name", help="profile name")
    _add_json_argument(de)
    de.set_defaults(func=_cmd_profile_delete)


def build_parser(profile_verbs: bool = False) -> argparse.ArgumentParser:
    """Build the argument parser; ``profile_verbs``: ``profile`` takes the library verbs
    (:func:`_profile_verb`) instead of a project."""
    p = _Parser(
        prog="cutpoint",
        description="Cutpoint: local AI highlight finding, clipping and assembly",
    )
    p.add_argument(
        "-v",
        "--verbose",
        "--debug",
        action="store_true",
        help="debug logging (also prints tracebacks of errors)",
    )
    p.add_argument("--version", action="version", version=f"cutpoint {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    ed = sub.add_parser(
        "edit",
        help="one command: analyze, export clips and assemble the final video with "
        "subtitles (<video>-final.mp4 and .srt in the project folder)",
    )
    ed.add_argument(
        "inputs", nargs="+", metavar="video", help="one or more video files and/or folders"
    )
    ed.add_argument(
        "--kind",
        choices=list(KINDS),
        default="highlights",
        help="what to make: highlights (up to 3 min, chronological, crossfades), montage "
        "(up to 60 s, best first, crossfades), clips (the 10 best as clips, hard cuts) or "
        "funny-moments (funny style, 8 moments, chronological, burned-in captions); "
        "all loudness-normalised (default: highlights)",
    )
    ed.add_argument(
        "--style",
        choices=[AUTO, *style_names()],
        help="analysis preset (default: funny for funny-moments, else action); auto picks "
        "one from the footage (gameplay, action cam, talk or general; ambiguous: gaming "
        "when it leans to gameplay, else the default)",
    )
    ed.add_argument(
        "--out",
        help="project dir; a bare name or projects/<name> goes under the configured projects "
        "folder. Default: <projects>/<first input's name> (a file's name without extension, "
        "or the folder's name)",
    )
    ed.add_argument("--top", type=positive_int, help="top N by score (default: per kind)")
    ed.add_argument(
        "--max-total",
        type=positive_float,
        help="cap total duration (s): best-scoring moments that fit (default: per kind)",
    )
    ed.add_argument("--vertical", action="store_true", help="9:16 1080x1920 center crop")
    ed.add_argument(
        "--force",
        action="store_true",
        help="replace the clips of an earlier run (as clips --force)",
    )
    ed.add_argument(
        "--keep-clips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep the clips in the clips folder; --no-keep-clips exports them into a "
        "temporary folder, removed after the final video is made, and leaves the clips "
        "folder alone (not with --kind clips or a timeline --format; default: keep)",
    )
    _add_format_argument(ed)
    ed.add_argument(
        "--events",
        action="store_true",
        help="also run `cutpoint events` to re-rank the moments (needs the events extra and "
        "cutpoint setup)",
    )
    _add_vision_arguments(ed, "rate top candidates with the vision model, as analyze --vision; ")
    _add_ollama_arguments(ed)
    _add_skip_argument(ed)
    _add_visual_index_argument(ed)
    _add_ocr_det_skip_argument(ed)
    _add_compress_peaks_argument(ed, "baked into the clips and so the final video")
    _add_lut_arguments(ed)
    _add_color_arguments(ed)
    _add_json_argument(ed)
    ed.set_defaults(func=_cmd_edit, usage_check=lambda a: _check_edit(ed, a))

    dt = sub.add_parser(
        "detect",
        help="detect the footage category (gameplay, action cam, talk, general) and the "
        "style --style auto picks; reads only a few frames and the container tags",
    )
    dt.add_argument(
        "inputs", nargs="+", metavar="input", help="one or more video files and/or folders"
    )
    _add_source_filter_arguments(dt)
    dt.add_argument(
        "--vision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="ask the vision model about sources the rest leaves unsure: on by default; "
        '--no-vision (or "vision": false in config.json) skips it; --vision (or "vision": true) '
        "also warns when the Vision provider lacks consent",
    )
    _add_ollama_arguments(dt)
    _add_json_argument(dt)
    dt.set_defaults(func=_cmd_detect)

    an = sub.add_parser("analyze", help="find and rank highlight candidates")
    an.add_argument(
        "inputs", nargs="+", metavar="input", help="one or more video files and/or folders"
    )
    _add_source_filter_arguments(an)
    an.add_argument(
        "--style",
        choices=[AUTO, *style_names()],
        default=ANALYZE_STYLE,
        help=f"preset (default: {ANALYZE_STYLE}); learned presets in "
        "<CUTPOINT_HOME>/styles/*.json are listed too (action-cam: telemetry-driven "
        "action-cam footage); auto picks one from the footage (gameplay, "
        f"action cam, talk or general, see cutpoint detect; ambiguous: gaming when it leans "
        f"to gameplay, else the General style, {auto_fallback(HYPE)})",
    )
    an.add_argument(
        "--out",
        help="project dir; a bare name or projects/<name> (projects\\<name> on Windows) goes under "
        "the configured projects folder (CUTPOINT_PROJECTS_DIR / config.json). "
        "Default: <projects>/<input name> (a file's name without extension, or the folder's "
        "name), or the common parent folder's name for several inputs",
    )
    _add_vision_arguments(
        an,
        "rate top candidates with the vision model (Ollama, or the vision_assistant provider in "
        "config.json); ",
    )
    an.add_argument("--model", help="whisper model (default: medium on CUDA, small on CPU)")
    an.add_argument("--language", help="force transcription language, e.g. en")
    an.add_argument("--no-scenes", action="store_true", help="skip scene detection")
    an.add_argument(
        "--max-candidates", type=positive_int, help="per source, pre-merge (default: auto)"
    )
    _add_ollama_arguments(an)
    an.add_argument(
        "--recover",
        action="store_true",
        help="if a source has a read error, resume analysis after the bad "
        "region (seeks past it; no copy of the file is made)",
    )
    an.add_argument(
        "--must-include-crash",
        type=unit_float,
        metavar="CONF",
        help="crashes from crashes.json (the crash step or cutpoint crashes) with at least this "
        "confidence, 0-1, become must-include moments with their reaction; weaker ones are "
        "ranked like any moment (default: the style's, 0.6; 0 = none forced)",
    )
    _add_skip_argument(an)
    _add_visual_index_argument(an)
    _add_ocr_det_skip_argument(an)
    add_batch_size_argument(an)
    _add_json_argument(an)
    an.set_defaults(func=_cmd_analyze)

    cl = sub.add_parser("clips", help="export selected moments as numbered clips")
    cl.add_argument(
        "project",
        help="project dir containing highlights.json (a bare name "
        "is looked up in the projects folder)",
    )
    _add_selection_arguments(cl, mix=True)
    cl.add_argument(
        "--out",
        help=(
            "clips dir (default: <project>/clips or clips_vertical; with --mix, clips_mix or "
            "clips_mix_vertical)"
        ),
    )
    _add_export_arguments(
        cl,
        force="replace clips in a non-empty dir",
        audio_fade=None,
        peaks="baked into every clip with a true-peak limiter",
    )
    _add_json_argument(cl)
    cl.set_defaults(func=_cmd_clips)

    ex = sub.add_parser(
        "extend",
        help="re-export an existing clips folder with extra "
        "padding (keeps file names and order; uses manifest.json)",
    )
    ex.add_argument("clips_dir")
    ex.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        metavar="S",
        help="padding added to both ends (default 0)",
    )
    ex.add_argument("--start", type=float, metavar="S", help="padding before (overrides)")
    ex.add_argument("--end", type=float, metavar="S", help="padding after (overrides)")
    ex.add_argument(
        "--out",
        help="output folder (default: sibling <clips_dir>_ext<S>s, _ext<A>-<B>s with "
        "--start/--end, or _speech without padding)",
    )
    ex.add_argument(
        "--force",
        action="store_true",
        help="write into a non-empty folder, or overwrite the input folder",
    )
    _add_speech_arguments(ex, "on")
    ex.add_argument(
        "--clip-override",
        nargs=2,
        action="append",
        metavar=("GLOB", "KEY=S"),
        help="per-clip setting for the clips whose file name matches GLOB: "
        "speech cap speech_max_extend=S or speech_max_lead=S (also read "
        "from manifest entries / segments), or total padding extend=S, "
        "extend_start=S or extend_end=S (replaces --seconds/--start/--end "
        "for those clips); repeatable",
    )
    ex.add_argument(
        "--reuse",
        metavar="DIR",
        help="earlier output folder: copy clips whose source, new in/out points, crop, "
        "grade, format and fade are identical there instead of re-encoding (not for "
        "--compress-peaks or 360 clips)",
    )
    _add_pano_arguments(ex, None)
    _add_json_argument(ex)
    ex.set_defaults(func=_cmd_extend)

    ev = sub.add_parser(
        "events",
        help="detect what happens: audio events, "
        "video+audio model descriptions, optional game-HUD OCR (writes "
        "events.json/events.txt)",
    )
    from cutpoint.events.cli import add_arguments as add_event_arguments  # noqa: PLC0415

    add_event_arguments(ev)
    add_batch_size_argument(ev, vlm=True)
    _add_json_argument(ev)
    ev.set_defaults(func=_cmd_events)

    st = sub.add_parser(
        "stage",
        help="copy sources on slow or network storage to a "
        "local folder once; later reads use the copy",
    )
    st.add_argument("inputs", nargs="*", metavar="input", help="video files and/or folders")
    _add_source_filter_arguments(st)
    st.add_argument("--list", action="store_true", help="list staged copies")
    st.add_argument(
        "--clear",
        action="store_true",
        help="delete staged copies: all, or those of the given inputs, selected as staging "
        "would (with --exclude and --recursive); a missing input falls back to the "
        "copies recorded under it",
    )
    _add_json_argument(st)
    st.set_defaults(func=_cmd_stage)

    cr = sub.add_parser(
        "crashes",
        help="(action-cam module) find crashes / "
        "wipeouts (telemetry + visual + "
        "audio + transcript, dense confirm) and write crashes.json",
    )
    cr.add_argument("project", help="analysed project (bare name = projects folder)")
    cr.add_argument(
        "--dense",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="confirm the best candidates on densely decoded frames (GPU; default: on)",
    )
    cr.add_argument(
        "--dense-top",
        type=int,
        default=15,
        metavar="N",
        help="candidates that get the dense confirm (default 15)",
    )
    cr.add_argument(
        "--sheets",
        type=int,
        default=15,
        metavar="N",
        help="contact sheets of the top N candidates in <project>/crash_check "
        "(default 15; 0 = none)",
    )
    cr.add_argument(
        "--lead",
        type=float,
        default=5.0,
        metavar="S",
        help="clip lead-in before the first impact (default 5)",
    )
    cr.add_argument(
        "--max-after",
        type=float,
        default=30.0,
        metavar="S",
        help="aftermath cap after the last impact (default 30)",
    )
    cr.add_argument(
        "--max-clip",
        type=float,
        default=40.0,
        metavar="S",
        help="crash clip length cap (default 40)",
    )
    cr.add_argument(
        "--force", action="store_true", help="recompute the IMU grids and the dense confirm"
    )
    cr.add_argument(
        "--reanalyze",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="re-run the (cached) analysis so must-include crashes and the "
        "wipeout feature reach highlights.json (default: on)",
    )
    cr.add_argument(
        "--batch-size",
        type=_batch_size,
        default=0,
        metavar="N",
        help="close the dense confirm's CLIP model after N dense windows and load it again "
        "for the next ones, to give back memory on long runs; the results are the same "
        "(default: 0, keep it loaded)",
    )
    _add_activity_arguments(cr)
    _add_ollama_arguments(cr)
    _add_json_argument(cr)
    cr.set_defaults(func=_cmd_crashes)

    if profile_verbs:
        _add_profile_verb_parsers(sub)
    else:
        pf = sub.add_parser(
            "profile",
            help="pick or discover the project's activity profile (crash prompts and words; "
            "found in your library or discovered from the footage, refined by --context) and "
            "write activity.json; cutpoint profile list|show|create|validate|delete manage "
            "the profile library",
            description="Pick or discover a project's activity profile and write its "
            "activity.json. The profile library has its own commands: cutpoint profile "
            "list | show NAME | create NAME --from FILE | validate NAME|FILE | delete NAME.",
        )
        pf.add_argument(
            "project",
            help="analysed project (bare name = projects folder; a project named list, show, "
            "create, validate or delete: projects/<name>)",
        )
        _add_activity_arguments(pf)
        _add_ollama_arguments(pf)
        _add_json_argument(pf)
        pf.set_defaults(func=_cmd_profile)

    rs = sub.add_parser(
        "resume",
        help="continue the interrupted run of a project (analyze, edit, events, crashes, "
        "profile, 360 analyze): runs its recorded command again, cached work is skipped",
    )
    rs.add_argument("project", help="project with a progress.json (bare name = projects folder)")
    rs.add_argument(
        "--dry-run", action="store_true", help="print the command and its step counts only"
    )
    _add_json_argument(rs)
    rs.set_defaults(func=_cmd_resume)

    mx = sub.add_parser(
        "mix",
        help="(action-cam module) pick a varied set of moments (runs, jumps, "
        "crashes, lifts, views, social) and write mix_segments.json",
    )
    mx.add_argument(
        "project", help="project analysed with --style action-cam (bare name = projects folder)"
    )
    mx.add_argument(
        "--target",
        type=positive_float,
        default=240.0,
        metavar="S",
        help="edit length (default 240)",
    )
    mx.add_argument(
        "--quotas",
        metavar="CAT=SHARE,...",
        help="category shares over the defaults run=.35,social=.2,crash=.12,"
        "scenery=.1,lift=.1,jump=.06,flex=.07 (normalised)",
    )
    mx.add_argument(
        "--out",
        default="mix_segments.json",
        help="segments file (relative = inside the project; default mix_segments.json)",
    )
    mx.add_argument(
        "--exclude-file",
        default=MIX_EXCLUDE_FILE,
        metavar="FILE",
        help="source spans to leave out, [{source, start, end, category?}, ...] (relative = "
        f"inside the project; default {MIX_EXCLUDE_FILE}, written by `cutpoint 360 repair`)",
    )
    _add_json_argument(mx)
    mx.set_defaults(func=_cmd_mix)

    asm = sub.add_parser(
        "assemble",
        help="join clips (filename order) into one video, or write their timeline; from a "
        "project, export its moments into a temporary folder first",
    )
    asm.add_argument(
        "clips_dir",
        metavar="clips_dir|project",
        help="a clips folder; or a project folder (with highlights.json, a bare name is looked "
        "up in the projects folder) or, with --segments, the project to write to: its moments "
        "are exported first (see the project options below)",
    )
    asm.add_argument(
        "--out",
        help="output file (default: <project>/final.mp4, or final_vertical.mp4 for clips_vertical "
        "or --vertical; the extension follows --format)",
    )
    _add_format_argument(asm)
    asm.add_argument(
        "--transition",
        choices=["cut", "fade"],
        default="cut",
        help="how clips join: hard cut or crossfade (default: cut)",
    )
    asm.add_argument(
        "--audio-transition",
        choices=AUDIO_TRANSITIONS,
        help="how the audio joins: cut or crossfade (default: cut with --transition cut, "
        "crossfade with fade); crossfade needs --transition fade",
    )
    asm.add_argument(
        "--fade-dur",
        type=positive_float,
        default=0.25,
        help="crossfade length in seconds with --transition fade or fade joins (default 0.25)",
    )
    asm.add_argument(
        "--joins",
        metavar="FILE",
        help='JSON file setting single joins: {"joins": [{"from": "001_a.mp4", "to": "002_b.mp4", '
        '"transition": "cut"|"fade"}]}, adjacent clip names in filename order; every other join '
        "uses --transition, and so does a pair that does not match (a WARNING). The audio "
        "follows: a cut join is an audio cut, a fade join uses --audio-transition (a crossfade "
        "with --transition cut)",
    )
    asm.add_argument(
        "--loudnorm", action="store_true", help="normalise to -14 LUFS, true peak below -1.5 dBTP"
    )
    asm.add_argument("--captions", action="store_true", help="burn in captions")
    asm.add_argument("--model", help="whisper model for captions")
    _add_json_argument(asm)
    proj = asm.add_argument_group(
        "from a project",
        "With a project instead of a clips folder, the moments are exported as `cutpoint clips` "
        "exports them, with these options, into a temporary folder in the project, assembled, "
        "and the folder is removed (after a failure or Ctrl+C too); a clips folder is never "
        "touched. With --format fcpxml or otio the timeline references the clips, so they are "
        "kept in the clips folder `cutpoint clips` writes (--force to replace its clips).",
    )
    _add_selection_arguments(proj, mix=False)
    _add_export_arguments(
        proj,
        force="--format fcpxml|otio from a project: replace the clips in a non-empty clips "
        "folder (as clips --force)",
        audio_fade="audio fade in/out: baked into the clips exported from a project; added to "
        "the clips of a clips folder that have none baked in (clips from cutpoint clips / "
        "cutpoint extend already do); default 0.25, 0 = off",
        peaks="baked into the clips exported from a project, else applied to the clips not "
        "exported with it; the true-peak limiter runs even without --loudnorm",
        recorded="Recorded in the clips' manifest.json, kept only with --format fcpxml|otio",
    )
    # the options that only apply when assemble exports the moments of a project first
    # (argparse has no public list of a group's options)
    project_options = [
        act.dest for act in proj._group_actions if act.dest not in CLIPS_FOLDER_OPTIONS
    ]
    asm.set_defaults(
        func=_cmd_assemble,
        usage_check=lambda a: _check_assemble(asm, a),
        project_defaults={d: asm.get_default(d) for d in project_options},
    )

    from cutpoint.pano.cli import add_parser as add_pano_parser  # noqa: PLC0415

    add_pano_parser(sub)

    su = sub.add_parser(
        "setup",
        help="install the models: Whisper (into the Hugging Face cache) and the events "
        "models (PANNs, CLIP, detector, Silero VAD) into CUTPOINT_HOME",
    )
    from cutpoint.events.setup import add_arguments as add_setup_arguments  # noqa: PLC0415
    from cutpoint.events.setup import check_args as check_setup_args  # noqa: PLC0415

    add_setup_arguments(su)
    _add_json_argument(su)
    su.set_defaults(func=_cmd_setup, usage_check=lambda a: check_setup_args(su, a))

    pa = sub.add_parser(
        "paths",
        help="print the resolved data folders (CUTPOINT_HOME, "
        "config, cache, models, tools, stage, styles, presets, profiles, projects, "
        "bundled)",
    )
    _add_json_argument(pa)
    pa.set_defaults(func=_cmd_paths)

    cp = sub.add_parser(
        "capabilities",
        help="report GPU vendors present, video encoders that really work, what "
        "transcription would use, whether torch reaches CUDA "
        "(with the reinstall fix if it is CPU-only on an NVIDIA machine) and the "
        "Insta360 Media SDK (needs NVIDIA)",
    )
    _add_json_argument(cp)
    cp.set_defaults(func=_cmd_capabilities)

    _add_style_parsers(sub)

    gd = sub.add_parser("guide", help="print the guide for AI agents driving Cutpoint")
    gd.set_defaults(func=_cmd_guide)

    sk = sub.add_parser(
        "skill", help="install or print the Claude Code skill for AI agents driving Cutpoint"
    )
    sks = sk.add_subparsers(dest="skill_command", required=True)
    si = sks.add_parser(
        "install",
        help="copy the skill to <dir>/cutpoint/SKILL.md (an identical file is left as is)",
    )
    si.add_argument(
        "--dir",
        metavar="DIR",
        help="skills folder, e.g. a project's .claude/skills or another agent's folder "
        "(default: ~/.claude/skills, the Claude Code user skills folder, or "
        "$CLAUDE_CONFIG_DIR/skills when CLAUDE_CONFIG_DIR is set)",
    )
    si.add_argument("--force", action="store_true", help="replace an installed skill that differs")
    si.set_defaults(func=_cmd_skill_install)
    sp = sks.add_parser("print", help="print the skill (SKILL.md) to stdout")
    sp.set_defaults(func=_cmd_skill_print)

    pr = sub.add_parser(
        "proxy",
        help="run the authenticated CLIProxyAPI daemon in the foreground for subscription logins",
    )
    pr.add_argument(
        "--port",
        type=_port,
        default=None,
        help="port on 127.0.0.1 (default: 18317 or CUTPOINT_CLIPROXY_PORT)",
    )
    pr.set_defaults(func=_cmd_proxy)

    lg = sub.add_parser(
        "login",
        help="log in to a subscription (Claude Code, Codex, Antigravity) in the browser via CLIProxyAPI",
    )
    lg.add_argument("provider", choices=PROXY_PROVIDERS)
    lg.add_argument(
        "--no-browser", action="store_true", help="only print the URL; do not open a browser"
    )
    lg.set_defaults(func=_cmd_login)
    lo = sub.add_parser("logout", help="remove a subscription login from CLIProxyAPI")
    lo.add_argument("provider", choices=PROXY_PROVIDERS)
    lo.set_defaults(func=_cmd_logout)

    ui = sub.add_parser(
        "ui",
        help="serve the local browser workspace (127.0.0.1 only)",
    )
    ui.add_argument("--port", type=_port, default=8765, help="port on 127.0.0.1 (default 8765)")
    ui.set_defaults(func=_cmd_ui)
    return p


def _raise_interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


@contextlib.contextmanager
def _stop_signal_as_interrupt() -> Iterator[None]:
    """Treat SIGTERM (POSIX) or SIGBREAK (Windows) like Ctrl+C while a command runs.

    The UI cancels a job by sending SIGTERM to its process group; by default Python then dies
    without unwinding, leaving half-written ``.part.mp4`` files behind. Raising
    KeyboardInterrupt runs the cleanup and the exit-130 path. ffmpeg children get the same
    SIGTERM from the group kill. Windows: Ctrl+Break (or CTRL_BREAK_EVENT sent to the process
    group) would otherwise end the process with 0xC000013A and no cleanup; ``taskkill /F``
    and TerminateProcess cannot be caught. Outside the main thread ``signal.signal`` raises
    ValueError, so nothing is installed there.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    signum = signal.SIGBREAK if os.name == "nt" else signal.SIGTERM
    previous = signal.signal(signum, _raise_interrupt)
    try:
        yield
    finally:
        signal.signal(signum, previous)


def _json_requested(tokens: list[str]) -> bool:
    """``--json`` or an abbreviation argparse accepts for it (``--j``, ``--js``, ``--jso``)."""
    for token in tokens:
        if token == "--":
            return False
        option = token.partition("=")[0]
        if len(option) > 2 and "--json".startswith(option):
            return True
    return False


def _usage_error(err: UsageError, command: str | None) -> NoReturn:
    """Re-raise the usage error ``err``; for ``command`` (a ``--json`` run, else None) first
    write its ``ok: false`` envelope (code ``bad_args``)."""
    if command:
        jsonout.emit_error(command, jsonout.BAD_ARGS, err.message)
    raise err


def _parse(parser: argparse.ArgumentParser, argv: Sequence[str] | None) -> argparse.Namespace:
    """``parser.parse_args(argv)``; with ``--json`` on a command that takes it, a usage error
    also writes an ``ok: false`` envelope (code ``bad_args``) before exiting 2."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    try:
        args, extra = parser.parse_known_args(tokens)
    except UsageError as err:  # from the subcommand's parser, which knows its json_command
        _usage_error(
            err, err.parser.get_default("json_command") if _json_requested(tokens) else None
        )
    command = args.json_command if getattr(args, "json_result", False) else None
    if extra:
        try:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
        except UsageError as err:
            _usage_error(err, command)
    # A command's check of option combinations argparse cannot express (exit 2, as argparse)
    if (check := getattr(args, "usage_check", None)) is not None:
        try:
            check(args)
        except UsageError as err:
            _usage_error(err, command)
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    args = _parse(build_parser(profile_verbs=_profile_verb(tokens)), argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in (
        "faster_whisper",
        "pyscenedetect",
        "urllib3",
        "httpx",
        "httpx2",
        "httpcore2",
        "huggingface_hub",
        "matplotlib",
        "numba",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if not getattr(args, "json_result", False):
        return _run(args, tokens)
    with jsonout.run(_resume_command(args)):
        code = _run(args, tokens)
        if code:
            jsonout.fail(jsonout.ERROR, f"exit code {code}")
    return code


def _recorded(args: argparse.Namespace, argv: Sequence[str]) -> contextlib.AbstractContextManager:
    """:func:`cutpoint.runstate.record` of a command that keeps ``progress.json`` and the
    project run lock (``analyze``, ``edit``, ``events``, ``crashes``, ``profile``,
    ``360 analyze``, ``clips``), else a no-op."""
    project = _run_project(args)
    if project is None:
        return contextlib.nullcontext()
    return runstate.record(args.json_command, argv, project, vars(args))


def _run_project(args: argparse.Namespace) -> Path | None:
    """The project folder of a command that keeps ``progress.json``, else None."""
    command = getattr(args, "json_command", None)
    if command not in runstate.COMMANDS:
        return None
    if command == "analyze":
        return _analyze_project(args)
    if command == "edit":
        return _edit_project(args)
    return resolve_project_path(args.project)


def _run(args: argparse.Namespace, argv: Sequence[str] = ()) -> int:
    """Run the command; errors are logged (and recorded for --json) as an exit code.
    ``argv``: the command line, for ``progress.json``."""
    try:
        with _stop_signal_as_interrupt(), _recorded(args, argv):
            code = args.func(args)
    except ModuleNotFoundError as err:
        from cutpoint.extras import missing_module_message  # noqa: PLC0415

        message = missing_module_message(err.name or "") if err.name else None
        if message is None:
            raise
        logger.debug("missing module", exc_info=True)
        logger.error("%s", message)
        jsonout.fail(jsonout.MISSING_EXTRA, message)
        return 1
    except (
        FFmpegError,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        ImportError,
        subprocess.CalledProcessError,
        shutil.Error,
        urllib.error.URLError,
    ) as err:
        # FileNotFoundError / FileExistsError are OSErrors; URLError too. TypeError is a fallback
        # for a wrongly typed setting that get_config does not check.
        logger.debug("command failed", exc_info=True)
        logger.error("%s", err)
        jsonout.fail(jsonout.error_code(err), str(err))
        return 1
    except KeyboardInterrupt:
        logger.error("interrupted")
        jsonout.fail(jsonout.INTERRUPTED, "interrupted")
        return 130
    return code if isinstance(code, int) else 0


if __name__ == "__main__":
    sys.exit(main())
