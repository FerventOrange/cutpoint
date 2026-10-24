"""Command-line interface: ``cutpoint edit | analyze | events | crashes | profile | refmatch | fit |
clips | extend | assemble | stage | 360 | setup | paths | capabilities | guide | skill | ui``.

Installed as the ``cutpoint`` console script.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import urllib.error
from collections.abc import Iterator
from dataclasses import replace
from importlib.resources import files
from pathlib import Path, PurePath

from cutpoint import __version__, paths, stage
from cutpoint.activity import resolve as resolve_activity
from cutpoint.assemble import assemble
from cutpoint.boundaries import SpeechSettings
from cutpoint.clips import export_clips
from cutpoint.config import get_config, projects_dir, resolve_project_path
from cutpoint.crash import CrashParams, detect_project
from cutpoint.edit import KINDS, edit_name, resolve_kind, run_edit
from cutpoint.events import paths as event_paths
from cutpoint.extend import extend_clips
from cutpoint.ffmpeg_utils import FFmpegError
from cutpoint.mix import EXCLUDE_FILE, load_exclusions, parse_quotas, run_mix
from cutpoint.planning import Padding
from cutpoint.refmatch import MatchParams, run_fit, run_refmatch
from cutpoint.styles import get_style, style_names
from cutpoint.utils import input_name, parse_size, positive_float, positive_int, unit_float

logger = logging.getLogger("cutpoint")


def _cmd_analyze(a: argparse.Namespace) -> None:
    from cutpoint.analyze import analyze  # noqa: PLC0415

    targets = [Path(t) for t in a.inputs]
    if a.out:
        out = resolve_project_path(a.out)
    else:  # one input: its name; several: their common parent folder's name
        name = (
            input_name(targets[0], "project")
            if len(targets) == 1
            else (Path(os.path.commonpath([t.resolve() for t in targets])).name or "multi")
        )
        out = projects_dir() / name
    style = get_style(a.style)
    if a.must_include_crash is not None:
        style = replace(style, must_include_crash=a.must_include_crash)
    # A failed or interrupted analyze removes the (still empty) folders it created.
    analyze(
        targets,
        style,
        out,
        vision=a.vision,
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
    )


def _cmd_edit(a: argparse.Namespace) -> None:
    targets = [Path(t) for t in a.inputs]
    run_edit(
        targets,
        resolve_project_path(a.out) if a.out else projects_dir() / edit_name(targets[0]),
        resolve_kind(a.kind, style=a.style, top=a.top, max_total=a.max_total),
        vertical=a.vertical,
        force=a.force,
        events=a.events,
        vision=a.vision,
        vision_top=a.vision_top,
        ollama_url=a.ollama_url,
        ollama_model=a.ollama_model,
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


def _cmd_clips(a: argparse.Namespace) -> None:
    project = resolve_project_path(a.project)
    segments = resolve_project_path(a.segments, bare_names=False) if a.segments else None
    if segments is not None and not a.mix and not segments.is_file():
        if segments.is_dir():
            raise IsADirectoryError(f"{segments} is a folder, not a segments file")
        where = _segments_where(a.segments, Path(a.segments), segments)
        raise FileNotFoundError(f"Segments file not found: {segments}{where}")
    order = a.order
    if a.mix:  # cutpoint mix's selection, played in arc order
        segments, order = project / "mix_segments.json", a.order or "arc"
        if not segments.is_file():
            run_mix(project, out=segments)
    export_clips(
        project,
        top=a.top,
        ids=a.ids,
        order=order,
        max_total=a.max_total,
        segments=segments,
        out_dir=resolve_project_path(a.out, bare_names=False)
        if a.out
        else (project / ("clips_mix_vertical" if a.vertical else "clips_mix") if a.mix else None),
        vertical=a.vertical,
        force=a.force,
        crop=a.crop,
        smart_tail_on=a.smart_tail,
        tail_min=a.tail_min,
        tail_max=a.tail_max,
        event_tails=a.event_tails,
        event_window=a.event_window,
        extend=Padding.from_args(a.extend, a.extend_start, a.extend_end),
        speech_aware=a.speech_aware,
        speech_max_extend=a.speech_max_extend,
        speech_max_lead=a.speech_max_lead,
        audio_fade=a.audio_fade,
        pano_source=a.pano_source,
        pano_cmd_every=a.pano_cmd_every,
        size=a.size,
    )


def _cmd_extend(a: argparse.Namespace) -> None:
    extend_clips(
        resolve_project_path(a.clips_dir, bare_names=False),
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
            raise ValueError(f"--clip-override {pattern} {assign}: expected KEY=SECONDS")
        try:
            out.setdefault(pattern, {})[key.strip().replace("-", "_")] = float(value)
        except ValueError:
            raise ValueError(
                f"--clip-override {pattern} {assign}: {value!r} is not a number of seconds"
            ) from None
    return out


def _cmd_events(a: argparse.Namespace) -> None:
    from cutpoint.events.cli import run_args  # noqa: PLC0415

    run_args(a)


def _cmd_stage(a: argparse.Namespace) -> None:
    if a.list:
        for e in stage.list_stage():
            logger.info(
                "%s  %.2f GB  read %ss  <- %s",
                e["file"],
                e["size"] / 1e9,
                e.get("read_s"),
                e.get("source"),
            )
        return
    if a.clear:
        stage.clear_stage(
            [Path(t) for t in a.inputs], exclude=a.exclude or [], recursive=a.recursive
        )
        return
    if not a.inputs:
        raise ValueError("cutpoint stage: give files/folders to stage (or --list / --clear)")
    stage.stage_sources([Path(t) for t in a.inputs], exclude=a.exclude or [], recursive=a.recursive)


def _cmd_refmatch(a: argparse.Namespace) -> None:
    run_refmatch(
        resolve_project_path(a.project),
        Path(a.reference),
        params=MatchParams(
            min_score=a.min_score, cut_corr=a.cut_corr, margin_score=a.margin_score, margin=a.margin
        ),
        fit=a.fit,
        name=a.name,
        check=a.check,
        taste=a.taste,
    )


def _cmd_fit(a: argparse.Namespace) -> None:
    run_fit(
        [resolve_project_path(p) for p in a.projects],
        name=a.name,
        base=a.base,
        taste=a.taste,
        do_ablate=a.ablate,
        save=not a.no_save,
    )


def _cmd_crashes(a: argparse.Namespace) -> None:
    params = CrashParams(
        lead_s=a.lead, max_after_s=a.max_after, max_clip_s=a.max_clip, dense_top=a.dense_top
    )
    project = resolve_project_path(a.project)
    detect_project(
        project,
        params,
        dense=a.dense,
        sheets=a.sheets,
        force=a.force,
        activity_name=a.activity,
        context=a.context,
        refresh_profile=a.refresh_profile,
    )
    if a.reanalyze:
        from cutpoint.analyze import reanalyze_if_reachable  # noqa: PLC0415

        reanalyze_if_reachable(project)


def _cmd_profile(a: argparse.Namespace) -> None:
    rec = resolve_activity(
        resolve_project_path(a.project),
        activity=a.activity,
        context=a.context,
        refresh=a.refresh_profile,
    )
    if a.json:
        print(json.dumps(rec, indent=2))
        return
    inc = rec["profile"]["incident"]
    print(f"activity  {rec['name']} ({rec['source']})")
    if rec.get("description"):
        print(f"footage   {rec['description']}")
    print(f"incident  {len(inc['pos'])} prompts, {len(inc['neg'])} negatives, ", end="")
    print(f"{len(inc['words'])} words")


def _add_activity_arguments(p: argparse.ArgumentParser) -> None:
    """``--activity``, ``--context``, ``--refresh-profile``."""
    p.add_argument(
        "--activity",
        metavar="NAME",
        help="activity profile (ski, snowboard, mtb, moto, skate, surf, hike, generic or a "
        "learned one in <CUTPOINT_HOME>/profiles); default: detected from the footage",
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
        help="re-detect / re-synthesize the profile (ignore <project>/activity.json)",
    )


def _cmd_mix(a: argparse.Namespace) -> None:
    project = resolve_project_path(a.project)
    out, excl = Path(a.out), Path(a.exclude_file)
    run_mix(
        project,
        target_s=a.target,
        quotas=parse_quotas(a.quotas),
        out=out if out.is_absolute() else project / out,
        exclude=load_exclusions(excl if excl.is_absolute() else project / excl),
    )


def _cmd_assemble(a: argparse.Namespace) -> None:
    assemble(
        resolve_project_path(a.clips_dir, bare_names=False),
        resolve_project_path(a.out, bare_names=False) if a.out else None,
        transition=a.transition,
        fade_dur=a.fade_dur,
        loudnorm=a.loudnorm,
        captions=a.captions,
        caption_model=a.model,
        audio_fade=a.audio_fade,
    )


def _cmd_setup(a: argparse.Namespace) -> int:
    from cutpoint.events.setup import run_args as run_setup  # noqa: PLC0415

    return run_setup(a)


def _cmd_paths(a: argparse.Namespace) -> None:
    cfg = get_config()
    rows = [
        ("home", paths.home()),
        ("config", paths.config_file()),
        ("cache", paths.cache_dir()),
        ("models", paths.models_dir()),
        ("tools", paths.tools_dir()),
        ("llama.cpp", event_paths.llama_dir()),
        ("stage", cfg.stage_dir),
        ("styles", paths.styles_dir()),
        ("presets", paths.presets_dir()),
        ("profiles", paths.profiles_dir()),
        ("projects", cfg.projects_dir),
        ("bundled", paths.PRESETS_DIR),
    ]
    width = max(len(k) for k, _ in rows)
    for key, path in rows:
        state = "" if path.exists() else "  (missing)"
        print(f"{key:<{width}}  {path}{state}")


def _cmd_capabilities(a: argparse.Namespace) -> None:
    from cutpoint import capabilities  # noqa: PLC0415 - probes ffmpeg, only when asked

    data = capabilities.report()
    print(json.dumps(data, indent=2) if a.json else capabilities.format_report(data))


def _cmd_guide(a: argparse.Namespace) -> None:
    # The wheel ships AGENTS.md inside the package; editable installs read the repo-root file.
    packaged = files("cutpoint") / "AGENTS.md"
    if packaged.is_file():
        data = packaged.read_bytes()
    else:
        data = (Path(__file__).resolve().parents[2] / "AGENTS.md").read_bytes()
    # Bytes as-is, so a non-UTF-8 console encoding cannot fail on non-ASCII text.
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

    # Bytes as-is, as for guide.
    sys.stdout.flush()
    sys.stdout.buffer.write(skill.skill_bytes())
    sys.stdout.buffer.flush()


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


def _add_speech_arguments(p: argparse.ArgumentParser, default_on: str) -> None:
    """``--speech-aware``, ``--speech-max-extend``, ``--speech-max-lead``, ``--audio-fade``."""
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
        help="audio fade in/out baked into every clip (default 0.25; 0 = off)",
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


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(
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
        choices=style_names(),
        help="analysis preset (default: funny for funny-moments, else action)",
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
        "--events",
        action="store_true",
        help="also run `cutpoint events` to re-rank the moments (needs the events extra and "
        "cutpoint setup)",
    )
    ed.add_argument(
        "--vision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="rate top candidates with Ollama, as analyze --vision (default: if reachable)",
    )
    ed.add_argument(
        "--vision-top",
        type=positive_int,
        default=None,
        metavar="N",
        help="top candidates sent to vision (default: 1 per 2 min of footage, min 10, max 60)",
    )
    ed.add_argument(
        "--ollama-url",
        help="default: $OLLAMA_HOST, ollama_url in config.json, or http://127.0.0.1:11434",
    )
    ed.add_argument("--ollama-model", help="default: qwen3-vl:8b")
    ed.set_defaults(func=_cmd_edit)

    an = sub.add_parser("analyze", help="find and rank highlight candidates")
    an.add_argument(
        "inputs", nargs="+", metavar="input", help="one or more video files and/or folders"
    )
    an.add_argument(
        "--exclude",
        action="append",
        metavar="GLOB",
        help="skip files whose name or full path matches (repeatable), e.g. --exclude '*DVR*'",
    )
    an.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="search sub-folders of folder inputs (default: no)",
    )
    an.add_argument(
        "--style",
        choices=style_names(),
        default="gaming",
        help="preset; learned presets in <CUTPOINT_HOME>/styles/*.json are listed too",
    )
    an.add_argument(
        "--out",
        help="project dir; a bare name or projects/<name> (projects\\<name> on Windows) goes under "
        "the configured projects folder (CUTPOINT_PROJECTS_DIR / config.json). "
        "Default: <projects>/<input name> (a file's name without extension, or the folder's "
        "name), or the common parent folder's name for several inputs",
    )
    an.add_argument(
        "--vision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="rate top candidates with Ollama (default: if reachable)",
    )
    an.add_argument(
        "--vision-top",
        type=positive_int,
        default=None,
        metavar="N",
        help="top candidates sent to vision (default: 1 per 2 min of footage, min 10, max 60)",
    )
    an.add_argument("--model", help="whisper model (default: medium on CUDA, small on CPU)")
    an.add_argument("--language", help="force transcription language, e.g. en")
    an.add_argument("--no-scenes", action="store_true", help="skip scene detection")
    an.add_argument(
        "--max-candidates", type=positive_int, help="per source, pre-merge (default: auto)"
    )
    an.add_argument(
        "--ollama-url",
        help="default: $OLLAMA_HOST, ollama_url in config.json, or http://127.0.0.1:11434",
    )
    an.add_argument("--ollama-model", help="default: qwen3-vl:8b")
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
        help="ski: always clip crashes from crashes.json (cutpoint crashes) with at "
        "least this confidence, 0-1 (default: style, 0.6 for ski; 0 = off)",
    )
    an.set_defaults(func=_cmd_analyze)

    cl = sub.add_parser("clips", help="export selected moments as numbered clips")
    cl.add_argument(
        "project",
        help="project dir containing highlights.json (a bare name "
        "is looked up in the projects folder)",
    )
    sel = cl.add_mutually_exclusive_group()
    sel.add_argument("--top", type=positive_int, help="top N by score (default: fits style target)")
    sel.add_argument("--ids", nargs="+", help="explicit candidate ids, e.g. h001 h004")
    sel.add_argument(
        "--segments",
        help="JSON list of {source,start,end,label} exported "
        "exactly, in the given order (no highlights.json needed; --order arc reorders "
        "a cutpoint mix file)",
    )
    sel.add_argument(
        "--mix",
        action="store_true",
        help="(action-cam/ski module) export <project>/mix_segments.json (runs "
        "`cutpoint mix` first if missing) in arc order",
    )
    cl.add_argument(
        "--max-total",
        type=positive_float,
        help="cap total duration (s): best-scoring moments that fit, e.g. 60",
    )
    cl.add_argument(
        "--order",
        choices=["chronological", "score", "story", "arc"],
        help="playback order (default: story; chronological for commentary; arc with "
        "--mix: scenic opener, the day by stretch, crash finale, scenic outro)",
    )
    cl.add_argument(
        "--out",
        help=(
            "clips dir (default: <project>/clips or clips_vertical; with --mix, clips_mix or "
            "clips_mix_vertical)"
        ),
    )
    cl.add_argument("--vertical", action="store_true", help="9:16 1080x1920 center crop")
    cl.add_argument(
        "--size",
        type=parse_size,
        metavar="WxH",
        help="output resolution, e.g. 1920x1080 (default: the largest source's, "
        "capped at 1080p); ignored with --vertical",
    )
    cl.add_argument("--force", action="store_true", help="replace clips in a non-empty dir")
    cl.add_argument(
        "--crop",
        metavar="auto|none|W:H:X:Y",
        help="crop before scaling: auto = cropdetect per source (cached); "
        "default: project.json per-source setting, else none",
    )
    cl.add_argument(
        "--smart-tail",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="extend speech-ended clips until the reaction settles (default: on, per style)",
    )
    cl.add_argument(
        "--tail-min",
        type=float,
        metavar="S",
        help="guaranteed tail after the last word (default: style, 1.5 s)",
    )
    cl.add_argument(
        "--tail-max",
        type=float,
        metavar="S",
        help="max extension after the last word (default: style, 5 s)",
    )
    cl.add_argument(
        "--event-tails",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="extend clips to cover a payoff event (kill, banner, laughter, "
        "VLM moment) that starts soon after the speech (default: on when "
        "<project>/events.json exists)",
    )
    cl.add_argument(
        "--event-window",
        type=float,
        metavar="S",
        help="payoff must start within S s after the speech (default: 10)",
    )
    cl.add_argument(
        "--extend",
        type=float,
        metavar="S",
        help="pad both ends of every clip by S seconds, after smart/event tails "
        "(clamped to the source and its readable range; default 0)",
    )
    cl.add_argument(
        "--extend-start",
        type=float,
        metavar="S",
        help="padding before the start (overrides --extend)",
    )
    cl.add_argument(
        "--extend-end", type=float, metavar="S", help="padding after the end (overrides --extend)"
    )
    _add_speech_arguments(cl, "on, per style")
    _add_pano_arguments(cl, "auto")
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
    ex.add_argument("--out", help="output folder (default: sibling <clips_dir>_ext<S>s)")
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
        help="earlier output folder: copy clips whose new in/out points, crop, "
        "format and fade are identical there instead of re-encoding",
    )
    _add_pano_arguments(ex, None)
    ex.set_defaults(func=_cmd_extend)

    ev = sub.add_parser(
        "events",
        help="detect what happens: audio events, "
        "video+audio model descriptions, optional game-HUD OCR (writes "
        "events.json/events.txt)",
    )
    from cutpoint.events.cli import add_arguments as add_event_arguments  # noqa: PLC0415

    add_event_arguments(ev)
    ev.set_defaults(func=_cmd_events)

    st = sub.add_parser(
        "stage",
        help="copy sources on slow or network storage to a "
        "local folder once; later reads use the copy",
    )
    st.add_argument("inputs", nargs="*", metavar="input", help="video files and/or folders")
    st.add_argument(
        "--exclude",
        action="append",
        metavar="GLOB",
        help="skip files whose name or full path matches (repeatable), e.g. --exclude '*DVR*'",
    )
    st.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="search sub-folders of folder inputs (default: no)",
    )
    st.add_argument("--list", action="store_true", help="list staged copies")
    st.add_argument(
        "--clear",
        action="store_true",
        help="delete staged copies: all, or those of the given inputs, selected as staging "
        "would (with --exclude and --recursive); a missing input falls back to the "
        "copies recorded under it",
    )
    st.set_defaults(func=_cmd_stage)

    rm = sub.add_parser(
        "refmatch",
        help="find a reference edit's shots in the project "
        "sources (refmatch.json) and fit style weights to them (action-cam/ski "
        "module)",
    )
    rm.add_argument("project", help="analysed project (bare name = projects folder)")
    rm.add_argument("reference", help="the finished edit to learn from")
    rm.add_argument(
        "--fit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fit weights and save <CUTPOINT_HOME>/styles/<name>.json (default: on for "
        "ski projects)",
    )
    rm.add_argument("--name", help="learned preset name (default: <base>_learned)")
    rm.add_argument(
        "--min-score",
        type=float,
        default=0.80,
        help="aligned frame correlation needed to accept a shot (default 0.80)",
    )
    rm.add_argument(
        "--margin-score",
        type=float,
        default=0.60,
        help="lower score accepted when it leads the next source/position by "
        "--margin (default 0.60)",
    )
    rm.add_argument("--margin", type=float, default=0.25)
    rm.add_argument(
        "--cut-corr",
        type=float,
        default=0.6,
        help="frame-to-frame correlation below which the edit has a cut",
    )
    rm.add_argument(
        "--check",
        type=int,
        default=6,
        help="write N side-by-side check frames to <project>/refmatch_check",
    )
    rm.add_argument(
        "--taste",
        action="store_true",
        help="also fit the personal taste head on the frame embeddings "
        "(needs `cutpoint events` visual layer); saved as <name>.taste.npz",
    )
    rm.set_defaults(func=_cmd_refmatch)

    ft = sub.add_parser(
        "fit",
        help="(action-cam/ski module) fit one learned preset "
        "on several projects' "
        "refmatch.json (cross-validated and cross-project report)",
    )
    ft.add_argument(
        "projects", nargs="+", help="projects with refmatch.json (bare name = projects folder)"
    )
    ft.add_argument("--name", help="learned preset name (default: <base>_learned)")
    ft.add_argument("--base", default="ski", help="built-in preset whose features are fitted")
    ft.add_argument(
        "--taste",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fit the taste head on frame embeddings (default: on)",
    )
    ft.add_argument(
        "--ablate",
        action="store_true",
        help="also refit on feature subsets (old features, +each new "
        "feature, all, all without VLM tags, all without taste)",
    )
    ft.add_argument(
        "--no-save",
        action="store_true",
        help="report only; do not write <CUTPOINT_HOME>/styles/<name>.json",
    )
    ft.set_defaults(func=_cmd_fit)

    cr = sub.add_parser(
        "crashes",
        help="(action-cam/ski module) find crashes / "
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
    _add_activity_arguments(cr)
    cr.set_defaults(func=_cmd_crashes)

    pf = sub.add_parser(
        "profile",
        help="pick or build the project's activity profile (crash prompts and words; "
        "detected from the footage, refined by --context) and write activity.json",
    )
    pf.add_argument("project", help="analysed project (bare name = projects folder)")
    _add_activity_arguments(pf)
    pf.add_argument("--json", action="store_true", help="print the activity.json record")
    pf.set_defaults(func=_cmd_profile)

    mx = sub.add_parser(
        "mix",
        help="(action-cam/ski module) pick a varied set of moments (runs, jumps, "
        "crashes, lifts, views, social) and write mix_segments.json",
    )
    mx.add_argument("project", help="ski-analysed project (bare name = projects folder)")
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
        default=EXCLUDE_FILE,
        metavar="FILE",
        help="source spans to leave out, [{source, start, end, category?}, ...] (relative = "
        f"inside the project; default {EXCLUDE_FILE}, written by `cutpoint 360 repair`)",
    )
    mx.set_defaults(func=_cmd_mix)

    asm = sub.add_parser("assemble", help="join clips (filename order) into one video")
    asm.add_argument("clips_dir")
    asm.add_argument(
        "--out",
        help="output file (default: <project>/final.mp4, or final_vertical.mp4 for clips_vertical)",
    )
    asm.add_argument("--transition", choices=["cut", "fade"], default="cut")
    asm.add_argument("--fade-dur", type=positive_float, default=0.25)
    asm.add_argument(
        "--loudnorm", action="store_true", help="normalise to -14 LUFS, true peak below -1.5 dBTP"
    )
    asm.add_argument("--captions", action="store_true", help="burn in captions")
    asm.add_argument("--model", help="whisper model for captions")
    asm.add_argument(
        "--audio-fade",
        type=float,
        default=0.25,
        metavar="S",
        help="audio fade in/out added to clips that have none baked in "
        "(clips from cutpoint clips / cutpoint extend already do; default "
        "0.25, 0 = off)",
    )
    asm.set_defaults(func=_cmd_assemble)

    from cutpoint.pano.cli import add_parser as add_pano_parser  # noqa: PLC0415

    add_pano_parser(sub)

    su = sub.add_parser(
        "setup",
        help="install the cutpoint events models and tools "
        "(PANNs, llama.cpp, Qwen2.5-Omni, CLIP, detector, Silero VAD) "
        "into CUTPOINT_HOME",
    )
    from cutpoint.events.setup import add_arguments as add_setup_arguments  # noqa: PLC0415

    add_setup_arguments(su)
    su.set_defaults(func=_cmd_setup)

    pa = sub.add_parser(
        "paths",
        help="print the resolved data folders (CUTPOINT_HOME, "
        "config, cache, models, tools, stage, styles, projects)",
    )
    pa.set_defaults(func=_cmd_paths)

    cp = sub.add_parser(
        "capabilities",
        help="report GPU vendors present, video encoders that really work, what "
        "transcription and llama.cpp setup would use, whether torch reaches CUDA "
        "(with the reinstall fix if it is CPU-only on an NVIDIA machine) and the "
        "Insta360 Media SDK (needs NVIDIA)",
    )
    cp.add_argument("--json", action="store_true", help="print the report as JSON")
    cp.set_defaults(func=_cmd_capabilities)

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
        "(default: ~/.claude/skills, the Claude Code user skills folder)",
    )
    si.add_argument("--force", action="store_true", help="replace an installed skill that differs")
    si.set_defaults(func=_cmd_skill_install)
    sp = sks.add_parser("print", help="print the skill (SKILL.md) to stdout")
    sp.set_defaults(func=_cmd_skill_print)

    ui = sub.add_parser(
        "ui",
        help="serve the local browser workspace for standard video highlights "
        "(127.0.0.1 only; needs the 'ui' extra)",
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


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
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
    # huggingface_hub prints through its own stderr handler and also propagates to ours
    # (e.g. the Hub's "unauthenticated requests" warning came out twice): keep only ours.
    # Its handler is added on import, so import it first.
    try:
        import huggingface_hub.utils.logging  # noqa: F401, PLC0415

        hf_logger = logging.getLogger("huggingface_hub")
        for handler in list(hf_logger.handlers):
            hf_logger.removeHandler(handler)
    except ImportError:
        pass
    try:
        with _stop_signal_as_interrupt():
            code = args.func(args)
    except ModuleNotFoundError as err:
        from cutpoint.extras import missing_module_message  # noqa: PLC0415

        message = missing_module_message(err.name or "") if err.name else None
        if message is None:
            raise
        logger.debug("missing module", exc_info=True)
        logger.error("%s", message)
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
        return 1
    except KeyboardInterrupt:
        logger.error("interrupted")
        return 130
    return code if isinstance(code, int) else 0


if __name__ == "__main__":
    sys.exit(main())
