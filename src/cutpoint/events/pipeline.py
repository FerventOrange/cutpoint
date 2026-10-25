"""Orchestrates the three layers for a project and writes events.json / events.txt.

Stages run one after another so only one GPU model is resident at a time:
PANNs (torch) -> freed -> voice (PANNs on high-passed audio + Silero VAD) ->
visual (CLIP ViT-L/14 + Faster R-CNN, one decode per source) -> freed -> OCR
(CPU, NVDEC decode) -> crashes (CLIP, handed over by the visual layer when no
OCR runs in between) -> Ollama unloaded -> llama-server started -> windows
described -> server stopped.
"""

from __future__ import annotations

import contextlib
import fnmatch
import importlib.util
import logging
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cutpoint import cache, jsonout, progress
from cutpoint import paths as cutpoint_paths
from cutpoint.config import source_path
from cutpoint.events import paths
from cutpoint.events.audio_events import AudioParams, PannsDetector, detect_audio_events
from cutpoint.events.common import (
    VramMonitor,
    audio_slice,
    load_highlights,
    load_source,
    load_transcript_lines,
    unload_ollama,
)
from cutpoint.events.eventtypes import types_for
from cutpoint.events.killfeed import (
    OcrEngine,
    OcrParams,
    derive_streaks,
    detect_ocr_events,
    link_teamkills,
)
from cutpoint.events.regions import game_key_for_path, load_preset
from cutpoint.events.report import normalize, write_outputs
from cutpoint.events.setup import COMPONENTS, EVENTS_MODULES
from cutpoint.events.vlm_windows import (
    VlmParams,
    add_incremental,
    cached_windows,
    check_audio,
    describe_windows,
    game_prompt,
    select_windows,
    windows_to_events,
)

logger = logging.getLogger(__name__)
LAYERS = ("audio", "voice", "visual", "crash", "ocr", "vlm")
# Layers run by default; voice/visual/crash feed ski features (other styles opt in).
DEFAULT_LAYERS = ("audio", "ocr", "vlm")
SKI_LAYERS = ("audio", "voice", "visual", "crash", "vlm")
# Share (percent) of the OCR step's progress HUD discovery takes when it runs; the
# per-source OCR takes the rest. Measured first runs: discovery was about 4 of 7 min of
# OCR on a 7 min file and 7 of 14 min on two 10 min files.
DISCOVERY_PCT = 50.0


def _require_events_extra(layers: tuple[str, ...]) -> None:
    """Fail before any side effect (unloading Ollama) when the events extra is missing.

    Every layer but VLM (llama-server) loads a model from the extra.
    """
    if not set(layers) - {"vlm"}:
        return
    for module in EVENTS_MODULES:
        if importlib.util.find_spec(module) is None:
            raise ModuleNotFoundError(f"No module named {module!r}", name=module)


def _require_model_files(layers: tuple[str, ...]) -> None:
    """Fail before any side effect when a model file or tool of ``layers`` is missing.

    One error lists every missing file and the ``cutpoint setup`` command for all of them.
    """
    missing = paths.missing_files(layers)
    if not missing:
        return
    components = sorted({comp for comp, _ in missing}, key=COMPONENTS.index)
    raise FileNotFoundError(
        f"Missing model files / tools for layers {', '.join(layers)}: "
        f"{'; '.join(str(p) for _, p in missing)}. "
        f"Run: cutpoint setup --only {' '.join(components)}"
    )


def preflight(layers: tuple[str, ...]) -> None:
    """Cheap checks for ``layers`` (events extra, model files) that :func:`run_events`
    makes before it unloads Ollama or imports torch; callers that run a long step
    first (``cutpoint 360 analyze``) call it up front."""
    _require_events_extra(layers)
    _require_model_files(layers)


def run_events(
    project: Path,
    *,
    layers: tuple[str, ...] | None = None,
    only: tuple[str, ...] = (),
    audio: AudioParams | None = None,
    ocr: OcrParams | None = None,
    vlm: VlmParams | None = None,
    force: tuple[str, ...] = (),
    ocr_debug: Path | None = None,
    avoid: tuple[tuple[str, float, float], ...] = (),
    discover: bool | None = None,
    discover_bursts: int = 40,
    preset_scope: str = "user",
    sheets: bool = False,
    stage: Callable[[str], AbstractContextManager[Any]] | None = None,
    cached_ocr: bool | None = None,
    llm: bool = True,
) -> tuple[Path, Path]:
    """Run the event layers on every source of ``project``.

    Args:
        project: Project folder containing highlights.json (and transcript.txt).
        layers: Layers to run; layers left out still contribute cached results.
            None = audio, OCR and VLM; ski projects: audio, voice, visual and
            VLM (no HUD, so no OCR).
        only: Glob patterns; restrict to sources whose file name matches.
        audio, ocr, vlm: Layer parameters (defaults when None).
        force: Layers whose caches are ignored.
        ocr_debug: Save a crop for every new OCR event here.
        avoid: Extra ``(name glob, start, end)`` ranges never decoded (bad
            sectors), on top of the ``read_errors`` recorded in highlights.json.
        discover: Run HUD region discovery: True = always (the proposal is
            written; an existing preset is kept), None = only for games
            without a preset, False = never.
        discover_bursts: Frame bursts sampled by discovery (x4 frames).
        preset_scope: Where a newly discovered preset is installed:
            "user" (``<CUTPOINT_HOME>/presets``, reused by every project;
            "package" is accepted as an old alias) or "project"
            (``<project>/hud_presets``).
        sheets: Write visual-layer contact sheets to ``<project>/visual_check``.
        stage: Called with each layer of ``layers`` that runs ("audio", "voice",
            "visual", "ocr", "crash"; not "vlm"), then with "write" for the events
            collected and written; the context manager it returns wraps that part
            (``cutpoint analyze`` reports its steps with it, and may suppress a layer's
            error to go on without it).
        cached_ocr: Read cached HUD OCR events when the OCR layer does not run
            (None: unless ski, which has no HUD).
        llm: Let the crash layer's activity profile ask the local Ollama models.

    Returns:
        Paths of events.json and events.txt.
    """
    audio, ocr, vlm = audio or AudioParams(), ocr or OcrParams(), vlm or VlmParams()
    stage = stage or (lambda _layer: contextlib.nullcontext())
    hl = load_highlights(project)
    ski = hl.get("style_kind") == "ski"
    if layers is None:
        layers = SKI_LAYERS if ski else DEFAULT_LAYERS
    if cached_ocr is None:
        cached_ocr = not ski
    vlm.ski = ski
    preflight(layers)
    sources = []
    seen: set[str] = set()
    unreachable: list[tuple[str, OSError]] = []
    for s in hl["sources"]:  # virtual parts (--recover) share one path
        if s["path"] not in seen:
            seen.add(s["path"])
            try:  # its cache key needs the file's size and mtime
                sources.append(load_source(Path(s["path"]), s.get("duration")))
            except OSError as err:
                logger.warning("%s: footage not reachable (%s); skipped", s["path"], err)
                unreachable.append((s["path"], err))
    for path, err in unreachable:
        jsonout.warn(
            "footage_unreachable", f"{path}: footage not reachable ({err}); skipped", [path]
        )
    if not sources:
        raise ValueError(f"No footage of {project.name} is reachable; reconnect it and rerun")
    bad = bad_ranges(hl, sources, avoid)
    if only:
        sources = [
            s for s in sources if any(fnmatch.fnmatch(s.name.lower(), p.lower()) for p in only)
        ]
    if not sources:
        raise ValueError("No sources selected")
    by_path = {str(s.path): s for s in sources}
    transcript = load_transcript_lines(project)
    timings: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    vram: dict[str, Any] = {}
    layer1: dict[str, list[dict[str, Any]]] = {str(s.path): [] for s in sources}

    # ---- layer 1a: audio
    t0 = time.perf_counter()
    det = None
    if "audio" in layers:
        with stage("audio"):
            unload_ollama()
            with VramMonitor() as vm:
                try:
                    for i, s in enumerate(sources):
                        progress.update(progress.fraction(i, len(sources)))
                        need = "audio" in force or not detect_audio_events(s, None, audio)
                        if need and det is None:
                            det = PannsDetector()
                        evs = detect_audio_events(
                            s, det if need else None, audio, force="audio" in force
                        )
                        layer1[str(s.path)] += [{**e, "kind": "audio"} for e in evs]
                finally:
                    if det is not None:
                        stats["audio_device"] = det.device
                        det.close()
            vram["audio"] = vm.summary()
    else:
        for s in sources:
            layer1[str(s.path)] += [
                {**e, "kind": "audio"} for e in detect_audio_events(s, None, audio)
            ]
    timings["audio_s"] = round(time.perf_counter() - t0, 1)

    # ---- layer 1c/1d: wind-robust voice grids, frame embeddings + people
    if "voice" in layers:
        from cutpoint.events import voice as voice_layer  # noqa: PLC0415

        t0 = time.perf_counter()
        with stage("voice"), VramMonitor() as vm:
            stats["voice"] = voice_layer.run_layer(
                sources, voice_layer.VoiceParams(), force="voice" in force
            )
        vram["voice"] = vm.summary()
        timings["voice_s"] = round(time.perf_counter() - t0, 1)
    crash_run = "crash" in layers and not only  # project-wide ranking: all sources
    kept: list[Any] = []  # the visual layer's CLIP, reused by the crash dense confirm
    if "visual" in layers:
        from cutpoint.crash import ClipEncoder  # noqa: PLC0415
        from cutpoint.events import visual as visual_layer  # noqa: PLC0415

        # Hand CLIP over only when no OCR runs in between: one model resident at a time.
        handoff = crash_run and "ocr" not in layers
        t0 = time.perf_counter()
        with stage("visual"), VramMonitor() as vm:
            stats["visual"] = visual_layer.run_layer(
                sources,
                visual_layer.VisualParams(hwaccel=ocr.hwaccel),
                force="visual" in force,
                bad=bad,
                handoff=(lambda m: kept.append(ClipEncoder.from_visual(m))) if handoff else None,
            )
        vram["visual"] = vm.summary()
        timings["visual_s"] = round(time.perf_counter() - t0, 1)
    if sheets:
        from cutpoint.events.sheets import write_visual_sheets  # noqa: PLC0415

        stats["sheets"] = [str(p) for p in write_visual_sheets(project, sources)]

    # ---- layer 1b: OCR (with region discovery for games without a preset)
    t0 = time.perf_counter()
    ocr.project = project
    with stage("ocr") if "ocr" in layers else contextlib.nullcontext():
        base = 0.0  # progress the per-source OCR starts at (after discovery)
        if "ocr" in layers and discover is not False:
            stats["discovery"] = _discover(
                project, sources, ocr, discover, discover_bursts, preset_scope
            )
            timings["discovery_s"] = round(time.perf_counter() - t0, 1)
            if stats["discovery"]:
                base = DISCOVERY_PCT
        engine = OcrEngine() if "ocr" in layers else None
        if not ocr.player_tags and sources and ("ocr" in layers or cached_ocr):
            logger.warning(
                "No player_tags set, so kills and deaths in the HUD cannot be "
                'attributed to you. Add "player_tags": ["YourName"] to %s '
                "or pass --player NAME.",
                cutpoint_paths.config_file(),
            )
        ocr_stats = {}
        with VramMonitor() as vm:
            ocr_sources = sources if "ocr" in layers or cached_ocr else []  # ski: no HUD
            for i, s in enumerate(ocr_sources):
                if "ocr" in layers:  # a cached read only is no step of its own
                    progress.update(
                        base + (100 - base) * progress.fraction(i, len(ocr_sources)) / 100,
                        f"reading HUD text: source {i + 1}/{len(ocr_sources)}" if base else None,
                    )
                evs, st = detect_ocr_events(
                    s,
                    ocr,
                    engine,
                    force="ocr" in force,
                    debug_dir=ocr_debug,
                    avoid=bad.get(str(s.path)),
                )
                preset = load_preset(game_key_for_path(str(s.path), ocr.game, project), project)
                link_teamkills(
                    evs,
                    tags=ocr.player_tags,
                    generic_teamkills=bool(preset and preset.feed.get("teamkills")),
                )
                derive_streaks(evs)
                layer1[str(s.path)] += evs
                ocr_stats[s.name] = st
        vram["ocr"] = vm.summary()
        stats["ocr"] = ocr_stats
    timings["ocr_s"] = round(time.perf_counter() - t0, 1)

    # ---- crashes (after voice + visual, whose caches it reads): crashes.json
    if crash_run:
        from cutpoint.crash import detect_project  # noqa: PLC0415

        t0 = time.perf_counter()
        enc = kept.pop() if kept else None
        try:
            with stage("crash"), VramMonitor() as vm:
                detect_project(
                    project,
                    force="crash" in force,
                    encoder=enc,
                    warn_unreachable=False,
                    llm=llm,
                )
        finally:
            if enc is not None:
                enc.close()
        vram["crash"] = vm.summary()
        timings["crash_s"] = round(time.perf_counter() - t0, 1)

    # ---- telemetry (ski): airtime / crash seeds + coverage of the riding
    if ski:
        for src, evs in _telemetry_layer(project, sources, vlm).items():
            layer1.setdefault(src, []).extend(evs)
    if ski and vlm.max_fill_windows == 0:
        vlm.max_fill_windows = 150
    if vlm.max_windows is None:
        vlm.max_windows = _auto_max_windows(layer1) if ski else 60
        if "vlm" in layers:
            logger.info("VLM: up to %d windows", vlm.max_windows)

    # ---- layer 2: VLM windows
    t0 = time.perf_counter()
    games = {
        str(s.path): "ski" if ski else game_key_for_path(str(s.path), ocr.game, project)
        for s in sources
    }
    # Game prompt and event types (custom seeds) from each source's HUD preset; ski has none.
    presets = {} if ski else {src: load_preset(g, project) for src, g in games.items()}
    prompts = {src: game_prompt(games[src], p) for src, p in presets.items()}
    seed_types = {src: types_for(p) for src, p in presets.items()}
    windows = select_windows(
        sources, hl.get("candidates", []), layer1, vlm, seed_types, quiet="vlm" not in layers
    )
    if ski and "vlm" not in force:  # reuse described windows; only fill new footage
        prev = cached_windows(sources, vlm)
        if prev:
            n_new = len(windows)
            windows = add_incremental(prev, windows)
            logger.log(
                logging.INFO if "vlm" in layers else logging.DEBUG,
                "VLM (ski, incremental): %d cached windows reused, %d of %d new "
                "windows cover footage not described yet",
                len(prev),
                len(windows) - len(prev),
                n_new,
            )
    windows = [
        w
        for w in windows
        if not any(w.start < b + 2 and w.end > a - 2 for a, b in bad.get(w.source, []))
    ]
    results: list[dict[str, Any]] = []
    if "vlm" in layers:
        cached, vstats = describe_windows(
            None, by_path, windows, vlm, prompts, transcript, layer1, force="vlm" in force
        )
        missing = len(windows) - vstats["cached"] if "vlm" not in force else len(windows)
        if missing > 0:
            from cutpoint.events.llama_server import LlamaServer  # noqa: PLC0415

            with (
                VramMonitor() as vm,
                LlamaServer(
                    ctx=vlm.ctx,
                    image_max_tokens=vlm.image_max_tokens,
                    log_path=paths.llama_dir() / "llama-server.log",
                ) as srv,
            ):
                stats["llama_load_s"] = srv.load_s
                if vlm.audio:
                    stats["audio_check"] = _audio_check(srv, by_path, layer1)
                    if not stats["audio_check"]["accepted"]:
                        logger.warning(
                            "Audio input NOT accepted by llama-server (%s); "
                            "falling back to frames + transcript",
                            stats["audio_check"].get("error"),
                        )
                        vlm.audio = False
                results, vstats = describe_windows(
                    srv, by_path, windows, vlm, prompts, transcript, layer1, force="vlm" in force
                )
            vram["vlm"] = vm.summary()
        else:
            results = cached
        stats["vlm"] = {k: v for k, v in vstats.items() if k not in ("seconds", "prompt_tokens")}
        secs = vstats.get("seconds") or []
        if secs:
            stats["vlm"]["mean_s_per_window"] = round(sum(secs) / len(secs), 1)
            toks = [t for t in vstats.get("prompt_tokens", []) if t]
            stats["vlm"]["mean_prompt_tokens"] = round(sum(toks) / max(1, len(toks)))
    else:
        results, _ = describe_windows(None, by_path, windows, vlm, prompts, transcript, layer1)
    timings["vlm_s"] = round(time.perf_counter() - t0, 1)
    stats["vram_mib"] = vram

    with stage("write"):
        events = [
            normalize(src, e)
            for src, evs in layer1.items()
            for e in evs
            if e.get("type") not in ("coverage", "coverage_masked")
        ]
        events += [normalize(e["source"], e) for e in windows_to_events(results)]
        settings = {
            "layers": list(layers),
            "avoid": dict(bad),
            "audio": audio.key(),
            "ocr": ocr.key(None) | {"game": ocr.game},
            "vlm": asdict(vlm),
        }
        return write_outputs(
            project,
            [str(s.path) for s in sources],
            events,
            results,
            transcript,
            settings,
            timings,
            stats,
        )


def _telemetry_layer(
    project: Path, sources: list[Any], vlm: VlmParams
) -> dict[str, list[dict[str, Any]]]:
    """Layer-1 entries from ``telemetry.json`` (written by ``cutpoint analyze --style ski``).

    Airtime and crash events become window seeds and events.txt lines.
    ``coverage`` pseudo-events (low priority, never written to events.json)
    tile the riding that is not on a lift or stopped, so spare VLM windows
    describe ordinary riding too (scenic views, friends, trees).
    """
    from cutpoint.telemetry import DT, extract  # noqa: PLC0415

    data = cache.load_json(project / "telemetry.json") or {}
    out: dict[str, list[dict[str, Any]]] = {}
    for s in sources:
        rec = (data.get("sources") or {}).get(str(s.path))
        if not rec:
            continue
        evs = []
        for e in rec.get("events", []):
            if e["type"] == "airtime":
                label = (
                    f"airtime {e['air_s']:.2f} s (~{e['height_m']:.1f} m), landing "
                    f"{e['landing_g']:.1f} g [{e['confidence']}]"
                )
            else:
                label = (
                    f"impact {e['peak_g']:.1f} g then a stop "
                    f"({e.get('speed_before_kmh')} -> {e.get('speed_after_kmh')} km/h), "
                    f"head tilt {e.get('tilt_deg', 0):.0f} deg"
                )
            evs.append(
                {
                    **e,
                    "kind": "telemetry",
                    "label": label,
                    "confidence": e.get("strength", e.get("score")),
                }
            )
        tel = extract(s.path, s.duration)
        if tel is not None:
            g = tel.grid
            ride = (g["lift_mask"] == 0) & (g["still_mask"] == 0)
            step = int(vlm.window_s / DT)
            for i in range(0, tel.n - step // 2, step):
                typ = "coverage" if ride[i : i + step].mean() >= 0.6 else "coverage_masked"
                evs.append(
                    {
                        "kind": "telemetry",
                        "type": typ,
                        "t": i * DT,
                        "end": min(s.duration, (i + step) * DT),
                        "label": "riding" if typ == "coverage" else "stopped/lift",
                    }
                )
        out[str(s.path)] = evs
    return out


def _auto_max_windows(layer1: dict[str, list[dict[str, Any]]]) -> int:
    """Ski default: enough windows for every seed and most of the riding (60-150)."""
    n = sum(
        1
        for evs in layer1.values()
        for e in evs
        if (e.get("kind") == "telemetry" and e.get("type") != "coverage_masked")
        or e.get("label") in ("Laughter", "Scream", "Cheering")
    )
    return max(60, min(150, n))


def _discover(
    project: Path, sources: list[Any], ocr: OcrParams, force: bool | None, bursts: int, scope: str
) -> dict[str, Any]:
    """Run region discovery per game where needed; install new presets.

    Returns:
        ``{game: {"preset": path, "installed": bool, "regions": [...], ...}}``.
    """
    from cutpoint.events.discover import DiscoverParams, discover_regions  # noqa: PLC0415

    by_game: dict[str, list[Any]] = {}
    for s in sources:
        by_game.setdefault(game_key_for_path(str(s.path), ocr.game, project), []).append(s)
    todo = []  # the games discovery runs for, with their existing preset
    for game, srcs in by_game.items():
        existing = load_preset(game, project)
        if existing is None or force:
            todo.append((game, srcs, existing))
    out: dict[str, Any] = {}
    for gi, (game, srcs, existing) in enumerate(todo):
        name = "this game" if game == "game" else game
        first = "" if existing is not None else f"first run for {name}: "
        if existing is None:
            progress.update(detail=f"{first}discovering HUD regions")
            logger.info(
                "No HUD preset matches %s; discovering regions as %r (a few minutes). "
                "Put a preset in %s and pass --game, or --no-discover to skip",
                srcs[0].path,
                game,
                cutpoint_paths.presets_dir(),
            )

        def on_frame(done: int, total: int, gi: int = gi, first: str = first) -> None:
            # the first DISCOVERY_PCT of the step, an equal part per game
            share = (gi + progress.fraction(done, total) / 100) / len(todo)
            progress.update(DISCOVERY_PCT * share, f"{first}discovering HUD: {done}/{total} frames")

        t0 = time.perf_counter()
        proposal = discover_regions(
            srcs,
            project / "discovery" / game,
            game,
            DiscoverParams(bursts=bursts),
            on_frame=on_frame,
        )
        rec: dict[str, Any] = {
            "proposal": str(project / "discovery" / game / "proposal.json"),
            "overlay": str(project / "discovery" / game / "overlay.jpg"),
            "regions": proposal["regions"],
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "installed": False,
        }
        if existing is None:
            folder = project / "hud_presets" if scope == "project" else cutpoint_paths.presets_dir()
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{game}.json"
            cache.save_json(target, proposal)
            rec.update(installed=True, preset=str(target))
            logger.info(
                "Installed discovered HUD preset for %r: %s (edit it to calibrate)", game, target
            )
        else:
            rec["preset"] = str(existing.path or "built-in")
            logger.info(
                "Kept existing HUD preset for %r (%s); discovery proposal: %s",
                game,
                rec["preset"],
                rec["proposal"],
            )
        out[game] = rec
    return out


def bad_ranges(
    hl: dict[str, Any], sources: list[Any], extra: tuple[tuple[str, float, float], ...]
) -> dict[str, list[tuple[float, float]]]:
    """Ranges per source that must not be decoded (read errors + ``extra``)."""
    out: dict[str, list[tuple[float, float]]] = {}
    for s in hl.get("sources", []):
        for rec in s.get("read_errors") or []:
            a = float(rec.get("failure_time", 0))
            b = float(rec.get("resumed_at") or s.get("duration") or a + 30)
            out.setdefault(s["path"], []).append((a, b))
    for pattern, a, b in extra:
        for src in sources:
            if fnmatch.fnmatch(src.name.lower(), pattern.lower()):
                out.setdefault(str(src.path), []).append((a, b))
    return out


def _audio_check(
    srv: Any, sources: dict[str, Any], layer1: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Run :func:`check_audio` on the strongest laughter event (or any audio)."""
    best = None
    for src, evs in layer1.items():
        for e in evs:
            if (
                e.get("kind") == "audio"
                and e["label"] == "Laughter"
                and (best is None or e["confidence"] > best[1]["confidence"])
            ):
                best = (src, e)
    if best:
        src, e = best
        samples = audio_slice(sources[src], e["peak_t"] - 4, e["peak_t"] + 4)
        where = f"{source_path(src).name} @ {e['peak_t']:.1f}s (PANNs {e['confidence']:.2f})"
    else:
        s = next(iter(sources.values()))
        samples = audio_slice(s, 0, 8)
        where = f"{s.name} @ 0s"
    res = check_audio(srv, samples)
    res["clip"] = where
    logger.info("Audio check on %s: %s", where, res.get("reply") or res.get("error"))
    return res
