"""``cutpoint events`` options and runner (``add_arguments`` / ``run_args``).

Examples::

    cutpoint events stream-01
    cutpoint events stream-01 --layers audio ocr
    cutpoint events stream-01 --max-windows 30 --window 15
    cutpoint events stream-01 --audio-floor Laughter=0.03
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from cutpoint import cache, jsonout
from cutpoint.events.audio_events import DEFAULT_GROUPS, AudioParams
from cutpoint.events.killfeed import OcrParams
from cutpoint.events.pipeline import DEFAULT_LAYERS, LAYERS, SKI_LAYERS, run_events
from cutpoint.events.vlm_windows import VlmParams
from cutpoint.styles import ACTION_CAM, project_kind


def _group_overrides(thr: list[str], floor: list[str]) -> tuple:
    groups = {g.name.lower(): g for g in DEFAULT_GROUPS}
    for spec, field in [(s, "abs_thr") for s in thr] + [(s, "rel_floor") for s in floor]:
        name, _, val = spec.partition("=")
        g = groups.get(name.strip().lower())
        if g is None or not val:
            raise SystemExit(
                f"bad override {spec!r}; groups: {', '.join(x.name for x in DEFAULT_GROUPS)}"
            )
        groups[g.name.lower()] = dataclasses.replace(g, **{field: float(val)})
    return tuple(groups.values())


def _forced(spec: str) -> tuple[str, float, float]:
    glob, _, rng = spec.rpartition("@")
    try:
        a, b = (float(x) for x in rng.split("-"))
    except ValueError:
        raise SystemExit(f"bad --window-at {spec!r}; use GLOB@START-END (seconds)") from None
    return glob or "*", a, b


def add_arguments(ap: argparse.ArgumentParser) -> None:
    """Add the ``cutpoint events`` options to ``ap`` (shared with cutpoint.cli)."""
    ap.add_argument("project", type=Path, help="project folder with highlights.json")
    ap.add_argument(
        "--layers",
        nargs="+",
        choices=LAYERS,
        default=None,
        help="layers to run (others still use cached results). Default: audio "
        "ocr vlm; action-cam projects: audio voice visual crash vlm (no HUD; crash = "
        "cutpoint crashes, writes crashes.json)",
    )
    ap.add_argument("--no-vlm", action="store_true", help="run the default layers without vlm")
    ap.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="GLOB",
        help="only sources whose file name matches (repeatable)",
    )
    ap.add_argument(
        "--force",
        nargs="+",
        choices=LAYERS,
        default=[],
        help="ignore cached results of these layers",
    )
    ap.add_argument(
        "--no-rescore",
        action="store_true",
        help="do not feed the events back into highlights.json scores",
    )
    ap.add_argument(
        "--sheets",
        action="store_true",
        help="write visual-layer contact sheets (top beauty shots, friends, "
        "people on lifts) to <project>/visual_check",
    )
    ap.add_argument(
        "--avoid",
        action="append",
        default=[],
        metavar="GLOB@START-END",
        help='Never decode this range (bad sectors), e.g. "Round4*@340-350"; '
        "read errors recorded in highlights.json are always avoided",
    )
    g = ap.add_argument_group("audio (PANNs)")
    g.add_argument(
        "--audio-thr",
        action="append",
        default=[],
        metavar="GROUP=P",
        help=argparse.SUPPRESS,  # developer tuning: absolute threshold per group, Laughter=0.1
    )
    g.add_argument(
        "--audio-floor",
        action="append",
        default=[],
        metavar="GROUP=P",
        help="floor of the relative (local baseline) rule, e.g. Laughter=0.03",
    )
    g = ap.add_argument_group("OCR")
    g.add_argument(
        "--game",
        default="auto",
        help="HUD preset key: auto (from the path) or any preset key; "
        "lookup: project (<project>/hud_presets), user "
        "(<CUTPOINT_HOME>/presets), then bundled",
    )
    g.add_argument(
        "--player",
        action="append",
        default=[],
        metavar="TAG",
        help="your in-game name(s), repeatable (default: player_tags in config.json)",
    )
    g.add_argument(
        "--ocr-fps", type=float, default=3.0, help="frames per second read by the OCR (default 3)"
    )
    g.add_argument("--no-hwaccel", action="store_true", help="decode on the CPU")
    # developer tuning: save region crops of OCR events here
    g.add_argument("--ocr-debug", type=Path, help=argparse.SUPPRESS)
    d = g.add_mutually_exclusive_group()
    d.add_argument(
        "--discover-regions",
        dest="discover",
        action="store_const",
        const=True,
        default=None,
        help="run HUD region discovery even if the game has a "
        "preset (writes <project>/discovery/<game>/proposal.json + overlay.jpg; "
        "an existing preset is kept). Default: only for games without one",
    )
    d.add_argument(
        "--no-discover",
        dest="discover",
        action="store_const",
        const=False,
        help="never run region discovery (games without a preset get no OCR)",
    )
    g.add_argument(
        "--discover-bursts",
        type=int,
        default=40,
        help="frame bursts sampled by discovery (4 frames each; default 40)",
    )
    g.add_argument(
        "--preset-scope",
        choices=["user", "project"],
        default="user",
        help="Where a newly discovered preset is saved: user = "
        "<CUTPOINT_HOME>/presets (default) or "
        "project = <project>/hud_presets",
    )
    g = ap.add_argument_group("VLM (vision model)")
    g.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="default 60; action-cam: one per seed + riding coverage, 60-150",
    )
    g.add_argument("--window", type=float, default=15.0, help="window length (s, default 15)")
    g.add_argument("--pre", type=float, default=3.0, help="seconds before an event (default 3)")
    g.add_argument("--post", type=float, default=12.0, help="seconds after an event (default 12)")
    g.add_argument(
        "--window-at",
        action="append",
        default=[],
        metavar="GLOB@START-END",
        help='Always describe this window, e.g. "*22.13.09*@364-400"',
    )
    g.add_argument("--frames", type=int, default=14, help="frames per window (12-16, default 14)")
    g.add_argument(
        "--no-transcript", action="store_true", help="do not send the transcript excerpt"
    )


def run_args(
    args: argparse.Namespace, *, ollama_url: str | None = None, ollama_model: str | None = None
) -> tuple[Path, Path]:
    """Run ``cutpoint events`` from parsed arguments; returns events.json / events.txt.
    ``ollama_url``, ``ollama_model``: the VLM layer's Ollama (``edit --events``; None: from
    config.json and the environment)."""
    from cutpoint.config import player_tags, resolve_project_path  # noqa: PLC0415

    args.project = resolve_project_path(args.project)
    jsonout.set_project(args.project)
    audio = AudioParams(groups=_group_overrides(args.audio_thr, args.audio_floor))
    ocr = OcrParams(
        fps=args.ocr_fps,
        game=args.game,
        hwaccel=not args.no_hwaccel,
        player_tags=tuple(args.player) or player_tags(),
    )
    vlm = VlmParams(
        window_s=args.window,
        pre_s=args.pre,
        post_s=args.post,
        max_windows=args.max_windows,
        frames=args.frames,
        transcript=not args.no_transcript,
        forced=tuple(_forced(s) for s in args.window_at),
    )
    layers = None
    if args.layers is not None or args.no_vlm:
        from cutpoint.events.common import load_highlights  # noqa: PLC0415

        ski = project_kind(load_highlights(args.project)) == ACTION_CAM
        chosen = args.layers or list(SKI_LAYERS if ski else DEFAULT_LAYERS)
        layers = tuple(x for x in chosen if not (args.no_vlm and x == "vlm"))
    jpath, tpath = run_events(
        args.project,
        layers=layers,
        only=tuple(args.only),
        audio=audio,
        ocr=ocr,
        vlm=vlm,
        force=tuple(args.force),
        ocr_debug=args.ocr_debug,
        avoid=tuple(_forced(s) for s in args.avoid),
        discover=args.discover,
        discover_bursts=args.discover_bursts,
        preset_scope=args.preset_scope,
        sheets=args.sheets,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
    )
    log = logging.getLogger(__name__)
    log.info("Wrote %s and %s", jpath, tpath)
    if jsonout.active():  # recorded before the rescore, so a failure there still reports them
        _report_events(args, jpath, tpath)
    if not args.no_rescore:
        from cutpoint.events.common import load_highlights  # noqa: PLC0415

        highlights = args.project / "highlights.json"
        backup = args.project / "highlights.pre-events.json"
        had_backup = backup.exists()
        if project_kind(load_highlights(args.project)) == ACTION_CAM:
            # Ski: events are features of the scorer, so re-run the (cached) analysis.
            from cutpoint.analyze import reanalyze_if_reachable  # noqa: PLC0415

            if reanalyze_if_reachable(args.project) is None:
                jsonout.warn(
                    "highlights_not_updated",
                    "a source could not be found or reached; highlights.json left unchanged, "
                    "rerun once it is back",
                    [highlights],
                )
            else:
                jsonout.output("highlights", highlights)
        else:
            from cutpoint.events.scoring import rescore_highlights  # noqa: PLC0415

            rescore_highlights(args.project)
            jsonout.output("highlights", highlights)
        if not had_backup and backup.exists():
            jsonout.output("highlights_backup", backup)
    return jpath, tpath


def _report_events(args: argparse.Namespace, jpath: Path, tpath: Path) -> None:
    """``--json`` outputs of ``events``: the files written and what events.json holds."""
    data = cache.load_json(jpath) or {}
    stats = data.get("stats") or {}
    layers = (data.get("settings") or {}).get("layers") or []
    jsonout.output("events", jpath)
    jsonout.output("events_txt", tpath)
    jsonout.output("layers", layers)
    jsonout.output("count", len(data.get("events") or []))
    jsonout.output("windows", len(data.get("windows") or []))
    jsonout.output("highlights", None)  # set below once the rescore wrote it
    if "crash" in layers and not args.only:  # the crash layer ranks all sources
        jsonout.output("crashes", args.project / "crashes.json")
    if stats.get("sheets"):
        jsonout.output("sheets", [Path(p) for p in stats["sheets"]])
    if stats.get("discovery"):
        jsonout.output(
            "discovery",
            {
                game: {
                    "proposal": Path(rec["proposal"]),
                    "overlay": Path(rec["overlay"]),
                    "preset": None if rec.get("preset") is None else Path(rec["preset"]),
                    "installed": rec.get("installed"),
                    "suggested": len(rec.get("suggested_regions") or []),
                }
                for game, rec in stats["discovery"].items()
            },
        )
