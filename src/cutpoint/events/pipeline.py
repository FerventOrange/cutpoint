"""Orchestrates the three layers for a project and writes events.json / events.txt.

Stages run one after another so only one GPU model is resident at a time:
PANNs (torch) -> freed -> voice (PANNs on high-passed audio + Silero VAD) ->
visual (CLIP ViT-L/14 + Faster R-CNN, one decode per source) -> freed -> OCR
(CPU, GPU decode) -> crashes (CLIP, handed over by the visual layer when no
OCR runs in between) -> windows described by the vision model (Ollama, unloaded
afterwards, or the hosted Vision role).
``cutpoint analyze`` may have read the OCR in the background
already (:mod:`cutpoint.events.ocr_prefetch`); its results are taken in the OCR turn.
"""

from __future__ import annotations

import contextlib
import fnmatch
import functools
import importlib.util
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

from cutpoint import cache, jsonout, progress
from cutpoint import paths as cutpoint_paths
from cutpoint.events import explosions, paths, scoring
from cutpoint.events.audio_events import (
    AudioParams,
    PannsDetector,
    cached_audio_events,
    detect_audio_events,
)
from cutpoint.events.common import (
    VramMonitor,
    free_torch,
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
from cutpoint.events.ocr_prefetch import OcrPrefetch
from cutpoint.events.regions import game_key_for_path, load_preset
from cutpoint.events.report import kept_dialogue, normalize, write_outputs
from cutpoint.events.setup import COMPONENTS, EVENTS_MODULES
from cutpoint.events.vlm_windows import (
    VlmParams,
    action_cam_wording,
    add_incremental,
    cached_windows,
    describe_windows,
    game_prompt,
    has_kills,
    select_windows,
    windows_to_events,
)
from cutpoint.sources import parts as source_parts
from cutpoint.styles import ACTION_CAM, project_kind
from cutpoint.vision import VisionRater

logger = logging.getLogger(__name__)
LAYERS = ("audio", "voice", "visual", "crash", "ocr", "vlm")
# Layers run by default; voice/visual/crash feed action-cam features (other styles opt in).
DEFAULT_LAYERS = ("audio", "ocr", "vlm")
ACTION_CAM_LAYERS = ("audio", "voice", "visual", "crash", "vlm")
# Share (percent) of the OCR step's progress HUD discovery takes when it runs; the
# per-source OCR takes the rest. Measured first runs: discovery was about 4 of 7 min of
# OCR on a 7 min file and 7 of 14 min on two 10 min files.
DISCOVERY_PCT = 50.0
_FAILURE = "_cutpoint_ocr_failure"  # the attribute :func:`ocr_failure` reads


class OcrFailure(NamedTuple):
    """Where the OCR layer failed (:func:`ocr_failure`)."""

    not_read: list[str]  # the paths of the sources it did not read, the failing one first
    discovery: bool  # in HUD region discovery: it read none
    background: bool  # the failing read ran in the background (:class:`OcrPrefetch`)
    games: int  # discovery: the games it was to discover (0: not known yet)


def _mark_ocr_failure(
    err: BaseException,
    not_read: list[Any],
    *,
    discovery: bool,
    background: bool = False,
    games: int = 0,
) -> None:
    """Note on ``err`` (re-raised unchanged) the sources the OCR layer did not read."""
    paths = [str(s.path) for s in not_read]
    setattr(err, _FAILURE, OcrFailure(paths, discovery, background, games))


def ocr_failure(err: BaseException) -> OcrFailure | None:
    """Where the OCR layer failed with ``err``: the sources it did not read (the one it
    failed on first), whether it failed in HUD region discovery (then it read none, and no
    preset was installed for at least one game), whether the failing read ran in the
    background and how many games discovery was to discover. The sources read before keep
    their events and menu spans. None: ``err`` did not come from the OCR layer's discovery
    or per-source reads."""
    return getattr(err, _FAILURE, None)


# How to avoid HUD discovery, after "Put a preset in <presets dir>" (``cutpoint events``'
# options; ``cutpoint analyze`` passes its own: :data:`cutpoint.analyze.DISCOVER_HINT`)
EVENTS_DISCOVER_HINT = "and pass --game, or --no-discover to skip"


def _require_events_extra(layers: tuple[str, ...]) -> None:
    """Fail before any side effect (unloading Ollama) when the events extra is missing.

    Every layer but VLM (the shared vision model) loads a model from the extra.
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
    first (``cutpoint 360 analyze``) call it up front. The VLM layer has no model files: its
    vision model is checked when it runs."""
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
    llm: bool | str = True,
    prefetch: OcrPrefetch | None = None,
    encoder: Any = None,
    ollama_url: str | None = None,
    ollama_model: str | None = None,
    discover_hint: str = EVENTS_DISCOVER_HINT,
    batch_size: int = 0,
    dialogue: Callable[[dict[str, list[dict[str, Any]]]], list[dict[str, Any]]] | None = None,
) -> tuple[Path, Path]:
    """Run the event layers on every source of ``project``.

    Args:
        project: Project folder containing highlights.json (and transcript.txt).
        layers: Layers to run; layers left out still contribute cached results.
            None = audio, OCR and VLM; action-cam projects: audio, voice, visual and
            VLM (no HUD, so no OCR).
        only: Glob patterns; restrict to sources whose file name matches.
        audio, ocr, vlm: Layer parameters (defaults when None).
        force: Layers whose caches are ignored.
        ocr_debug: Save a crop for every new OCR event here.
        avoid: Extra ``(name glob, start, end)`` ranges never decoded (bad
            sectors), on top of the ``read_errors`` recorded in highlights.json.
        discover: Run HUD region discovery: True = always (the proposal is
            written; an existing preset keeps its regions and only gets the fresh
            ``discovery.fit_ref``), None = only for games
            without a preset, False = never.
        discover_bursts: Frame bursts sampled by discovery (x4 frames).
        preset_scope: Where a newly discovered preset is installed:
            "user" (``<CUTPOINT_HOME>/presets``, reused by every project) or
            "project" (``<project>/hud_presets``).
        sheets: Write visual-layer contact sheets to ``<project>/visual_check``.
        stage: Called with each layer of ``layers`` that runs ("audio", "voice",
            "visual", "ocr", "crash"; not "vlm"), then with "write" for the events
            collected and written; the context manager it returns wraps that part
            (``cutpoint analyze`` reports its steps with it, and may suppress a layer's
            error to go on without it). None: each layer starts its step of the active
            progress run if it has one (``cutpoint events``, :data:`progress.EVENTS_STEPS`),
            the VLM layer too.
        cached_ocr: Read cached HUD OCR events when the OCR layer does not run
            (None: unless action-cam, which has no HUD).
        llm: Let the crash layer's activity profile ask the local Ollama models
            (:data:`cutpoint.activity.AUTO`: not asked for, see :func:`cutpoint.activity.resolve`).
        prefetch: OCR read in the background by ``cutpoint analyze``: a source's result
            is taken from it in its turn (same events, caches and log lines); what it did
            not read is read here, after it is stopped, with its engine.
        encoder: A CLIP model loaded by an earlier step (a ``ClipEncoder``, waiting on
            the CPU) for the crash layer's dense confirm, so it is not loaded again
            (``cutpoint 360 analyze``, #376); the visual layer then hands over none.
            Closed after the crash layer; the caller closes it if this raises before
            or the crash layer does not run.
        ollama_url, ollama_model: Ollama base URL and vision model for the VLM layer
            (None: from config.json and the environment, as for ``analyze --vision``).
        discover_hint: How to avoid discovery, after "Put a preset in <presets dir>" in the
            INFO line before it runs (the caller's options: ``cutpoint analyze`` has no
            ``--game`` or ``--no-discover``).
        batch_size: ``--batch-size`` (#109): a layer's model is closed after every this many
            sources and loaded again for the next ones, the vision model's (Ollama) at the
            first source boundary after at least this many VLM windows (:func:`vlm_batches`),
            the crash layer's CLIP after every this many dense windows (0: one batch). The
            results are the same.
        dialogue: ``cutpoint analyze``'s dialogue reader: called, as the ``dialogue`` stage,
            with the layer-1 events per source path (before they are normalized) after the
            VLM layer; returns events.json entries (:func:`cutpoint.events.dialogue.to_events`)
            that replace the earlier dialogue events. None: the earlier ones are kept.

    Returns:
        Paths of events.json and events.txt.
    """
    audio, ocr, vlm = audio or AudioParams(), ocr or OcrParams(), vlm or VlmParams()
    boom = explosions.ExplosionParams()
    stage = stage or _layer_step
    hl = load_highlights(project)
    is_cam = project_kind(hl) == ACTION_CAM
    if layers is None:
        layers = ACTION_CAM_LAYERS if is_cam else DEFAULT_LAYERS
    if cached_ocr is None:
        cached_ocr = not is_cam
    preflight(layers)
    try:
        rater: VisionRater | None = VisionRater(ollama_url, ollama_model)  # the shared client
    except ValueError as err:  # a bad port in OLLAMA_HOST / ollama_url: only the VLM is lost
        rater = None
        if "vlm" in layers:
            logger.warning("VLM: vision model not usable (%s); that layer is skipped", err)
            jsonout.warn("vlm_unavailable", f"VLM: vision model not usable ({err})")
        else:
            logger.info("vision model not usable (%s); feed labels are skipped", err)
    hosted = rater is not None and rater.provider_name != "ollama"
    from cutpoint.assistant.providers import AUTO_SERVER_MODEL, host_port  # noqa: PLC0415

    online: bool | None = None  # rater.available(), asked once
    if rater is None:
        vlm.model = "ollama@unavailable"  # no descriptions are cached under it
    else:
        if hosted and "vlm" in layers and rater.model == AUTO_SERVER_MODEL:
            # resolve the server's model before any cache lookup: the descriptions are keyed
            # on it. Offline it stays the placeholder, and its cached descriptions are not
            # found.
            online = rater.available()
        # the descriptions are keyed on the host too: one model id at two servers describes
        # apart (rater.host: providers.endpoint; an Ollama's host[:port] of its URL, #670)
        host = rater.host if hosted else host_port(rater.url)
        vlm.model = f"{rater.provider_name}@{host}:{rater.model}"
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
    voice_params = None
    keep = False  # hand PANNs to the voice layer instead of loading it again (#372)
    voice_needs = False
    panns_failed = False  # the audio layer could not load PANNs: voice does not retry (#629)
    if "voice" in layers:
        from cutpoint.events import voice as voice_layer  # noqa: PLC0415

        voice_params = voice_layer.VoiceParams()
        keep = voice_needs = voice_layer.needs_run(sources, voice_params, force="voice" in force)
    if "audio" in layers:
        with stage("audio"):
            unload_ollama()
            with VramMonitor() as vm:
                try:
                    for i, s in enumerate(sources):
                        if batch_size and i and not i % batch_size:  # a batch boundary
                            if det is not None:
                                det.close()
                                det = None
                            _boundary("audio", i, len(sources))
                        progress.update(progress.fraction(i, len(sources)))
                        evs = None if "audio" in force else cached_audio_events(s, audio)
                        if evs is None and not s.audio_parts:
                            logger.warning(
                                "%s: no cached 16 kHz audio; run `cutpoint analyze` first", s.name
                            )
                            evs = []  # nothing to detect: no model load
                        if evs is None:
                            if det is None:
                                try:
                                    det = PannsDetector()
                                except Exception:
                                    panns_failed = True
                                    raise
                                # the device of this run, as voice and visual record theirs:
                                # a run that only reads cached events loads no PANNs
                                stats["audio_device"] = det.device
                            evs = detect_audio_events(s, det, audio, force=True)  # miss or forced
                        layer1[str(s.path)] += [{**e, "kind": "audio"} for e in evs]
                except BaseException:
                    keep = False  # the layer failed: free the model now
                    raise
                finally:
                    if det is not None and not keep:
                        det.close()
                        det = None
            vram["audio"] = vm.summary()
    else:
        for s in sources:
            layer1[str(s.path)] += [
                {**e, "kind": "audio"} for e in detect_audio_events(s, None, audio)
            ]
    timings["audio_s"] = round(time.perf_counter() - t0, 1)

    # ---- layer 1c/1d: wind-robust voice grids, frame embeddings + people
    if "voice" in layers and panns_failed and voice_needs:
        logger.info("Voice layer skipped: PANNs did not load (see the audio layer's error)")
    elif "voice" in layers:
        t0 = time.perf_counter()
        try:
            with stage("voice"), VramMonitor() as vm:
                for k, part in enumerate(progress.batches(sources, batch_size)):
                    if k:  # a batch boundary: the PANNs of the batch before is closed
                        if det is not None:
                            det.close()
                            det = None
                        _boundary("voice", k * batch_size, len(sources))
                    st = voice_layer.run_layer(part, voice_params, force="voice" in force, det=det)
                    stats["voice"] = merge_stats(stats["voice"], st) if k else st
        finally:
            if det is not None:  # closed before the next layer loads its model
                det.close()
                det = None
        vram["voice"] = vm.summary()
        timings["voice_s"] = round(time.perf_counter() - t0, 1)
    crash_run = "crash" in layers and not only  # project-wide ranking: all sources
    kept: list[Any] = []  # the visual layer's CLIP, reused by the crash dense confirm
    # The visual layer's caption embeddings use the captions of the project's activity.json
    # (generic until the crash layer below writes one); the contact sheets are written after the
    # crash layer when it runs, so a first run's sheets use the profile it resolves.
    from cutpoint.activity import project_prompts, use_prompts  # noqa: PLC0415

    prompts = project_prompts(project)
    if "visual" in layers:
        from cutpoint.crash import ClipEncoder  # noqa: PLC0415
        from cutpoint.events import visual as visual_layer  # noqa: PLC0415

        # Hand CLIP over only when no OCR runs in between: one model resident at a time.
        handoff = crash_run and "ocr" not in layers and encoder is None
        t0 = time.perf_counter()
        with stage("visual"), VramMonitor() as vm, use_prompts(prompts):
            parts = progress.batches(sources, batch_size)
            for k, part in enumerate(parts):
                if k:  # a batch boundary: run_layer closed the models of the batch before
                    _boundary("visual", k * batch_size, len(sources))
                last = k == len(parts) - 1  # only the last batch hands CLIP over
                st = visual_layer.run_layer(
                    part,
                    visual_layer.VisualParams(hwaccel=ocr.hwaccel),
                    force="visual" in force,
                    bad=bad,
                    handoff=(
                        (lambda m: kept.append(ClipEncoder.from_visual(m)))
                        if handoff and last
                        else None
                    ),
                )
                stats["visual"] = merge_stats(stats["visual"], st) if k else st
        vram["visual"] = vm.summary()
        timings["visual_s"] = round(time.perf_counter() - t0, 1)

    def write_sheets(prompt_set: Any) -> None:
        from cutpoint.events.sheets import write_visual_sheets  # noqa: PLC0415

        with use_prompts(prompt_set):
            stats["sheets"] = [str(p) for p in write_visual_sheets(project, sources)]

    if sheets and not crash_run:
        write_sheets(prompts)

    # ---- layer 1b: OCR (with region discovery for games without a preset)
    t0 = time.perf_counter()
    ocr.project = project
    make_engine = None  # set before discovery: the stage may suppress its error
    with stage("ocr") if "ocr" in layers else contextlib.nullcontext():
        base = 0.0  # progress the per-source OCR starts at (after discovery)
        if "ocr" in layers and discover is not False:
            from cutpoint.events.feed_labels import make_labeler  # noqa: PLC0415

            try:
                stats["discovery"] = _discover(
                    project,
                    sources,
                    ocr,
                    discover,
                    discover_bursts,
                    preset_scope,
                    discover_hint,
                    make_labeler(rater) if rater is not None else None,
                )
            except Exception as err:
                if ocr_failure(err) is None:  # before _discover knew its games
                    _mark_ocr_failure(err, sources, discovery=True)
                raise
            timings["discovery_s"] = round(time.perf_counter() - t0, 1)
            if stats["discovery"]:
                base = DISCOVERY_PCT
        # the OCR model loads only when a source's footage is read (not for a replay)
        if "ocr" in layers:
            make_engine = prefetch.make_engine if prefetch else functools.cache(OcrEngine)
        if not ocr.player_tags and sources and ("ocr" in layers or cached_ocr):
            logger.warning(
                "No player_tags set, so kills and deaths in the HUD cannot be "
                'attributed to you. Add "player_tags": ["YourName"] to %s '
                "or pass --player NAME.",
                cutpoint_paths.config_file(),
            )
        ocr_stats: dict[str, Any] = {}
        stats["ocr"] = ocr_stats  # the sources read before a failure keep their menu spans
        with VramMonitor() as vm:
            ocr_sources = sources if "ocr" in layers or cached_ocr else []  # action-cam: no HUD
            shown = [base]  # the step's pct never goes back

            def on_source(done: float, n: int = len(ocr_sources)) -> None:
                pct = base + (100 - base) * progress.fraction(done, n) / 100
                if pct > shown[0]:
                    shown[0] = pct
                    progress.update(pct)

            for i, s in enumerate(ocr_sources):
                if batch_size and i and not i % batch_size and "ocr" in layers:
                    # a batch boundary: the engine is made again for the next batch (the
                    # background reader's is in use until it is stopped)
                    if prefetch is None or make_engine is not prefetch.make_engine:
                        _release_engine(make_engine)
                    _boundary("ocr", i, len(ocr_sources))
                if "ocr" in layers:  # a cached read only is no step of its own
                    shown[0] = max(
                        shown[0], base + (100 - base) * progress.fraction(i, len(ocr_sources)) / 100
                    )
                    progress.update(
                        shown[0],
                        f"reading HUD text: source {i + 1}/{len(ocr_sources)}" if base else None,
                    )
                on_progress = (lambda share, i=i: on_source(i + share)) if "ocr" in layers else None
                got = None
                background = False  # the failing read ran in the background
                try:
                    if prefetch is not None:  # the background read of these exact inputs
                        plain = "ocr" not in force and not ocr_debug and not bad.get(str(s.path))
                        if "ocr" in layers and plain:
                            background = True
                            got = prefetch.take(s, ocr, on_progress)
                            background = False
                        else:
                            prefetch.close()
                        if got is None and prefetch.stuck:  # it outlived close(): not its engine
                            prefetch.wait_for(s.path)  # nor the raw cache of the source it is on
                            if make_engine is prefetch.make_engine:
                                make_engine = functools.cache(OcrEngine)
                    evs, st = got or detect_ocr_events(
                        s,
                        ocr,
                        None,
                        force="ocr" in force,
                        debug_dir=ocr_debug,
                        avoid=bad.get(str(s.path)),
                        on_progress=on_progress,
                        make_engine=make_engine,
                    )
                    game = game_key_for_path(str(s.path), ocr.game, project)
                    preset = load_preset(game, project)
                    teamkills = bool(preset and preset.feed.get("teamkills"))
                    link_teamkills(evs, generic_teamkills=teamkills)
                    derive_streaks(evs)
                except Exception as err:
                    _mark_ocr_failure(err, ocr_sources[i:], discovery=False, background=background)
                    raise
                layer1[str(s.path)] += evs
                ocr_stats[s.name] = st
        vram["ocr"] = vm.summary()
        stats["ocr"] = ocr_stats
    timings["ocr_s"] = round(time.perf_counter() - t0, 1)
    # ---- explosions (#801): a heard boom counts once the loudness jumps with it and the
    # picture was not black just before it (a loading screen ending)
    t0 = time.perf_counter()
    booms = sum(
        explosions.confirm(
            layer1[str(s.path)],
            [(p.offset, p.cdir) for p in source_parts(s.path)],
            boom,
            luma=explosions.luma_reader(s.path),  # --recover parts share the file and its time
        )
        for s in sources
        if explosions.has_booms(layer1[str(s.path)])
    )
    timings["explosions_s"] = round(time.perf_counter() - t0, 1)
    dark = sum(
        1
        for s in sources
        for e in layer1[str(s.path)]
        if e.get("kind") == "audio" and e.get("label") == explosions.LABEL and e.get("after_black")
    )
    if booms or dark:
        logger.info(
            "%d explosion(s) confirmed by a loudness jump, %d set aside after a black screen",
            booms,
            dark,
        )
    if make_engine is not None:  # no OCR after this: the engine is not kept through the VLM
        if prefetch is not None and make_engine is prefetch.make_engine:
            prefetch.close()  # its worker uses the engine: stopped first (idempotent)
        _release_engine(make_engine)
        make_engine = None

    # ---- crashes (after voice + visual, whose caches it reads): crashes.json
    if crash_run:
        from cutpoint.crash import detect_project  # noqa: PLC0415

        t0 = time.perf_counter()
        enc = kept.pop() if kept else encoder
        try:
            with stage("crash"), VramMonitor() as vm:
                detect_project(
                    project,
                    force="crash" in force,
                    encoder=enc,
                    warn_unreachable=False,
                    llm=llm,
                    batch_size=batch_size,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                )
        finally:
            if enc is not None:
                enc.close()
        vram["crash"] = vm.summary()
        timings["crash_s"] = round(time.perf_counter() - t0, 1)
        if sheets:  # under the activity.json the crash layer just wrote
            write_sheets(project_prompts(project))

    # ---- telemetry (action-cam): airtime / crash seeds + coverage of the riding
    if is_cam:
        for src, evs in _telemetry_layer(project, sources, vlm).items():
            layer1.setdefault(src, []).extend(evs)
    if is_cam and vlm.max_fill_windows == 0:
        vlm.max_fill_windows = 150
    if vlm.max_windows is None:
        vlm.max_windows = _auto_max_windows(layer1) if is_cam else 60
        if "vlm" in layers:
            logger.info("VLM: up to %d windows", vlm.max_windows)

    # ---- layer 2: VLM windows
    # action-cam: worded for the activity.json the crash layer above may have just written
    cam_prompts = project_prompts(project) if is_cam else None
    vlm.action_cam = action_cam_wording(cam_prompts) if cam_prompts is not None else None
    if "vlm" in layers:
        progress.start("vlm")  # cutpoint events: the step of progress.json
        free_torch()  # the CUDA memory torch still caches for the layers before
    t0 = time.perf_counter()
    games = {
        str(s.path): ACTION_CAM if is_cam else game_key_for_path(str(s.path), ocr.game, project)
        for s in sources
    }
    # Game prompt and event types (custom seeds) from each source's HUD preset; action-cam has none.
    # The prompt asks about kills only for footage with kills (#596).
    presets = {} if is_cam else {src: load_preset(g, project) for src, g in games.items()}
    prompts = {
        src: game_prompt(games[src], p, has_kills(p, layer1.get(src, [])))
        for src, p in presets.items()
    }
    seed_types = {src: types_for(p) for src, p in presets.items()}
    windows = select_windows(
        sources,
        [c for c in hl.get("candidates", []) if c.get("pool") != "visual"],  # no pool seeds
        layer1,
        vlm,
        seed_types,
        quiet="vlm" not in layers,
        style=scoring.analysis_names(hl),
    )
    # a provider the user did not agree to (a hosted one, or an Ollama on another machine): no
    # request, no cached descriptions or windows either, as for analyze's vision ratings
    no_vision = rater is None or rater.consent_needed is not None
    if is_cam and "vlm" not in force and not no_vision:
        # reuse described windows; only fill new footage
        prev = cached_windows(sources, vlm)
        if prev:
            n_new = len(windows)
            # a reused window this run selects again keeps its reasons (#109: a resumed run's
            # events.json equals an uninterrupted one's)
            chosen = {(w.source, w.key): w for w in windows}
            windows = [chosen.get((w.source, w.key), w) for w in add_incremental(prev, windows)]
            logger.log(
                logging.INFO if "vlm" in layers else logging.DEBUG,
                "VLM (action-cam, incremental): %d cached windows reused, %d of %d new "
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
    if no_vision:
        cached, _ = describe_windows(None, by_path, windows, vlm, prompts, transcript, layer1)
        # the VLM layer is skipped; say why cached descriptions (or reused windows) are gone
        if rater is not None and (
            "vlm" in layers
            or cached
            or (is_cam and "vlm" not in force and cached_windows(sources, vlm))
        ):
            rater.warn_consent()
    elif "vlm" in layers:
        cached, vstats = describe_windows(
            None, by_path, windows, vlm, prompts, transcript, layer1, force="vlm" in force
        )
        missing = len(windows) - vstats["cached"] if "vlm" not in force else len(windows)
        if "vlm" not in force:
            progress.continuing("vlm", vstats["cached"], len(windows))
        if missing > 0:
            results = cached
            if online is None:
                online = rater.available()
            if online:
                try:
                    with VramMonitor() as vm:
                        results, vstats = _describe_batches(
                            rater,
                            by_path,
                            windows,
                            vlm,
                            prompts,
                            transcript,
                            layer1,
                            force="vlm" in force,
                            batch_size=batch_size,
                            unload=not hosted,
                        )
                    vram["vlm"] = vm.summary()
                finally:
                    rater.unload()
            else:
                if "vlm" in force:  # nothing to describe them with: keep the saved descriptions
                    results, vstats = describe_windows(
                        None, by_path, windows, vlm, prompts, transcript, layer1
                    )
                message = (
                    f"VLM: vision model {vlm.model} not available (see above); {missing} "
                    "window(s) not described, cached descriptions used"
                )
                logger.warning("%s", message)
                jsonout.warn("vlm_unavailable", message)
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

    talk: list[dict[str, Any]] = []
    if dialogue is not None:
        with stage("dialogue"):
            talk = dialogue(layer1)
    else:  # not read by this run: what an earlier one marked stays
        talk = kept_dialogue(project, [str(s.path) for s in sources])
    with stage("write"):
        events = [
            normalize(src, e)
            for src, evs in layer1.items()
            for e in evs
            if e.get("type") not in ("coverage", "coverage_masked")
        ]
        events += [normalize(e["source"], e) for e in windows_to_events(results)]
        events += talk
        vlm_settings = asdict(vlm)
        if cam_prompts is not None and vlm.action_cam is not None:  # the activity, not its text
            wording = vlm.action_cam.key() or "default"
            vlm_settings["action_cam"] = {"activity": cam_prompts.name, "wording": wording}
        settings = {
            "layers": list(layers),
            "avoid": dict(bad),
            "audio": audio.key(),
            "explosions": boom.key(),
            "ocr": ocr.key(None) | {"game": ocr.game},
            "vlm": vlm_settings,
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


@contextlib.contextmanager
def _layer_step(layer: str) -> Iterator[None]:
    """:func:`run_events`' default ``stage``: ``layer`` is its step of the active progress
    run while the block runs, if the run has one (``cutpoint events``)."""
    progress.start(layer)
    yield
    r = progress.current()
    if r is not None and r.current == layer:
        r.done()


def _boundary(layer: str, done: int, total: int) -> None:
    """A ``--batch-size`` boundary of ``layer`` after ``done`` of ``total`` units: the model
    is closed, so torch's cached memory is freed; logged and recorded as
    :func:`progress.continuing` does."""
    free_torch()
    progress.continuing(layer, done, total)


def merge_stats(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """The stats of two batches of a layer as one run's: counts added, lists joined, dicts
    merged, anything else (a device) from ``b``; the keys in the order one run gives them."""
    out = dict(a)
    for k, v in b.items():
        old = out.get(k)
        number = isinstance(v, int | float) and not isinstance(v, bool)
        if isinstance(v, dict) and isinstance(old, dict):
            out[k] = {**old, **v}
        elif (isinstance(v, list) and isinstance(old, list)) or (
            number and isinstance(old, int | float)
        ):
            out[k] = old + v
        else:
            out[k] = v
    return out


def _describe_batches(
    rater: VisionRater,
    by_path: dict[str, Any],
    windows: list[Any],
    vlm: VlmParams,
    prompts: dict[str, Any],
    transcript: list[dict[str, Any]],
    layer1: dict[str, list[dict[str, Any]]],
    *,
    force: bool,
    batch_size: int,
    unload: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """:func:`describe_windows` in batches (``--batch-size``, :func:`vlm_batches`), with
    ``unload`` (Ollama) its model unloaded between two; the results and stats of one call."""
    results: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    done = 0
    for k, part in enumerate(vlm_batches(windows, batch_size)):
        if k:  # a batch boundary
            if unload:
                rater.unload()
            progress.continuing("vlm", done, len(windows))
        res, st = describe_windows(
            rater, by_path, list(part), vlm, prompts, transcript, layer1, force=force
        )
        results += res
        stats = merge_stats(stats, st) if k else st
        done += len(part)
    return results, stats


def vlm_batches(windows: list[Any], size: int) -> list[list[Any]]:
    """The VLM windows in batches for ``--batch-size`` ``size``: a batch ends only where the
    windows go on with another source, once it holds at least ``size`` windows, so the vision
    model (6.6 GB for qwen3.5:9b) is not loaded again every few windows of one recording; 0:
    one batch."""
    if size <= 0:
        return [list(windows)]
    out: list[list[Any]] = []
    part: list[Any] = []
    for w in windows:
        if part and len(part) >= size and w.source != part[-1].source:
            out.append(part)
            part = []
        part.append(w)
    return [*out, part] if part or not out else out


def _release_engine(make_engine: Callable[[], OcrEngine]) -> None:
    """Drop the OCR engine ``make_engine`` (a ``functools.cache``) made, if it made one, so
    its memory is free before the next layers load their models. The next call of
    ``make_engine`` makes a new one."""
    if getattr(make_engine, "cache_info", None) is None:
        return
    make_engine.cache_clear()  # type: ignore[attr-defined]


def _telemetry_layer(
    project: Path, sources: list[Any], vlm: VlmParams
) -> dict[str, list[dict[str, Any]]]:
    """Layer-1 entries from ``telemetry.json`` (written by ``cutpoint analyze --style action-cam``).

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
    """Action-cam default: enough windows for every seed and most of the riding (60-150)."""
    n = sum(
        1
        for evs in layer1.values()
        for e in evs
        if (e.get("kind") == "telemetry" and e.get("type") != "coverage_masked")
        or e.get("label") in ("Laughter", "Scream", "Cheering")
    )
    return max(60, min(150, n))


def _discover(
    project: Path,
    sources: list[Any],
    ocr: OcrParams,
    force: bool | None,
    bursts: int,
    scope: str,
    hint: str = EVENTS_DISCOVER_HINT,
    labeler: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run region discovery per game where needed; install new presets. ``labeler`` names the
    proposals with the shared vision model (:mod:`cutpoint.events.feed_labels`).

    Sources whose key is a first guess (no ``--game``, no preset ``match`` hint) and that
    have no recorded decision (or all of them with ``force``) get the HUD fit check
    (:mod:`cutpoint.events.game_fit`) once their guess has a preset, found or discovered
    here: one whose edge text lands elsewhere on screen gets a key and a discovery of its
    own, and a preset discovered here is made again from the sources that fit. The
    decisions go to ``<project>/discovery/game_keys.json``
    (:func:`~cutpoint.events.regions.game_key_for_path` reads them).

    Returns:
        ``{game: {"preset": path, "installed": bool, "regions": [...], ...}}``; for a
        game with a preset also ``suggested_regions`` (extra notice regions and a score
        readout the preset does not have; never written into it).
    """
    from cutpoint.events import discover, game_fit  # noqa: PLC0415
    from cutpoint.events import regions as hud  # noqa: PLC0415

    params = discover.DiscoverParams(bursts=bursts)
    named = bool(ocr.game and ocr.game != "auto")
    by_game: dict[str, list[Any]] = {}
    guesses: dict[str, str] = {}  # path -> first guess, for the sources the fit check decides
    before: dict[str, str] = {}  # path -> the key recorded before (stale, or checked again)
    for s in sources:
        path = str(s.path)
        key = game_key_for_path(path, ocr.game, project)
        if not named and hud.preset_for_path(path, None, project) is None:
            entry = hud.decided_entry(path, project)
            if entry is None or force:
                key = guesses[path] = str(entry["guess"]) if entry else key
                raw = hud.load_game_keys(project).get(hud.source_id(path)) or {}
                if isinstance(raw.get("key"), str):
                    before[path] = raw["key"]
        by_game.setdefault(key, []).append(s)
    planned = sum(force or load_preset(g, project) is None for g in by_game)
    runs = {"started": 0, "total": max(1, planned)}  # discoveries (a misfit adds one)
    shown = [0.0]  # the discovery part's pct never goes back
    out: dict[str, Any] = {}

    def run(
        game: str,
        srcs: list[Any],
        existing: Any,
        keep: set[str] | None = None,
        install: bool = True,
    ) -> dict[str, Any]:
        """Discovery for ``game`` on ``srcs`` and its install, as a record in ``out``;
        ``keep``: cluster again from these sources' records only (no OCR, no "No HUD
        preset matches" line);
        ``install`` False: no preset written yet (:func:`install_preset` does it once the
        fit check has split off other games)."""
        gi = runs["started"]
        if keep is None:
            runs["started"] += 1
            runs["total"] = max(runs["total"], runs["started"])
        name = "this game" if game == "game" else game
        first = "" if existing is not None else f"first run for {name}: "
        if existing is None and keep is None:
            progress.update(detail=f"{first}discovering HUD regions")
            logger.info(
                "No HUD preset matches %s; discovering regions as %r (a few minutes). "
                "Put a preset in %s %s",
                srcs[0].path,
                game,
                cutpoint_paths.presets_dir(),
                hint,
            )

        def on_frame(done: int, total: int) -> None:
            # the first DISCOVERY_PCT of the step, an equal part per discovery
            share = (gi + progress.fraction(done, total) / 100) / runs["total"]
            shown[0] = max(shown[0], DISCOVERY_PCT * share)
            progress.update(shown[0], f"{first}discovering HUD: {done}/{total} frames")

        t0 = time.perf_counter()
        # an existing preset's regions (and big-text check) are the covered ones: extra
        # notice regions outside them are only suggested, the preset's regions stay as they are;
        # the outside regions it already has count against the extra-notice cap
        covered: dict[str, Any] = {}
        if existing is not None:
            data = cache.load_json(existing.path) if existing.path else None
            regs = data.get("regions") if isinstance(data, dict) else None
            outside = sum(isinstance(r, dict) and r.get("origin") == "outside" for r in regs or [])
            covered = {"covered": existing.all_regions(), "outside": outside}
        try:
            proposal = discover.discover_regions(
                srcs,
                project / "discovery" / game,
                game,
                params,
                on_frame=on_frame if keep is None else None,
                **covered,
                **({"labeler": labeler} if labeler else {}),
                **({"keep_src_keys": keep} if keep is not None else {}),
            )
        except Exception as err:  # the games before it have their presets
            _mark_ocr_failure(err, sources, discovery=True, games=runs["total"])
            raise
        rec: dict[str, Any] = {
            "proposal": str(project / "discovery" / game / "proposal.json"),
            "overlay": str(project / "discovery" / game / "overlay.jpg"),
            "regions": proposal["regions"],
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "installed": False,
        }
        if keep is not None and game in out:
            rec["elapsed_s"] = round(out[game]["elapsed_s"] + rec["elapsed_s"], 1)
        out[game] = rec
        if existing is None:
            if install:
                install_preset(game, proposal)
        else:
            rec["preset"] = str(existing.path)
            rec["suggested_regions"] = list(proposal.get("suggested_regions") or [])
            logger.info(
                "Kept existing HUD preset for %r (%s); discovery proposal: %s",
                game,
                rec["preset"],
                rec["proposal"],
            )
            if rec["suggested_regions"]:
                logger.info(
                    "Discovery suggests %d extra region(s) for %r the preset does not have "
                    "(%s: notices where text shows outside its regions, a score readout); to OCR "
                    'them, copy the "suggested_regions" entries of %s into the "regions" of %s',
                    len(rec["suggested_regions"]),
                    game,
                    ", ".join(r["name"] for r in rec["suggested_regions"]),
                    rec["proposal"],
                    rec["preset"],
                )
        return proposal

    def add_fit_ref(game: str, path: Path | None, ref: Any) -> None:
        """Give an existing preset the fresh grid ``fit_ref`` of a forced discovery (the HUD
        fit check); every other key stays as it is. A ref of too little evidence is not
        written, nor one for a bundled preset; a preset that cannot be written is left as it
        is (WARNING)."""
        if path is None or not game_fit.valid_ref(ref) or not ref["grid"]:
            return
        if Path(path).resolve().is_relative_to(cutpoint_paths.PRESETS_DIR.resolve()):
            return  # bundled with the package: read-only data
        if not (
            isinstance(ref.get("frames"), int)
            and isinstance(ref.get("boxes"), int)
            and ref["frames"] >= game_fit.MIN_FRAMES
            and ref["boxes"] >= game_fit.MIN_BOXES
        ):
            return
        data = cache.load_json(path)
        disc = data.get("discovery", {}) if isinstance(data, dict) else None
        if not isinstance(disc, dict) or disc.get("fit_ref") == ref:
            return
        assert isinstance(data, dict)
        data["discovery"] = disc | {"fit_ref": ref}
        try:
            cache.save_json(path, data)
        except OSError as err:
            logger.warning(
                "Could not add the HUD fit check to the preset '%s' (%s): %s", game, path, err
            )
            return
        logger.info(
            "Added the HUD fit check to the preset '%s' (its regions are unchanged): %s", game, path
        )

    def install_preset(game: str, proposal: dict[str, Any]) -> None:
        """Write a discovered preset (its record in ``out`` says where)."""
        folder = project / "hud_presets" if scope == "project" else cutpoint_paths.presets_dir()
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{game}.json"
        cache.save_json(target, proposal)
        out[game].update(installed=True, preset=str(target))
        logger.info(
            "Installed discovered HUD preset for %r: %s (edit it to calibrate)", game, target
        )

    def per_source(recs: list[dict[str, Any]], srcs: list[Any]) -> dict[str, list[dict[str, Any]]]:
        """The gameplay frames among OCR records, per source path."""
        paths = {discover.src_key(s): str(s.path) for s in srcs}
        got: dict[str, list[dict[str, Any]]] = {}
        for f in discover.gameplay_frames(recs, params):
            if discover._src(f) in paths:
                got.setdefault(paths[discover._src(f)], []).append(f)
        return got

    def fit_frames(
        game: str, proposal: dict[str, Any] | None, srcs: list[Any], tested: list[Any]
    ) -> dict[str, list[dict[str, Any]]]:
        """Gameplay frames per source: this run's discovery records, and a sample of their own
        for ``tested`` sources with too few."""
        got: dict[str, list[dict[str, Any]]] = {}
        ids = set(((proposal or {}).get("discovery") or {}).get("frames") or [])
        if ids:
            recs = discover.read_ocr_samples(project / "discovery" / game / "ocr.jsonl")
            got = per_source([f for f in recs if f["id"] in ids], srcs)
        need = [s for s in tested if len(got.get(str(s.path), [])) < game_fit.TOP_UP_FRAMES]
        if need:
            logger.debug("HUD fit check for %r: sampling %d source(s)", game, len(need))
            recs = game_fit.sample_frames(need, project / "discovery" / "_fit" / game, params)
            for path, frames in per_source(recs, need).items():
                got.setdefault(path, []).extend(frames)
        return got

    def fit_ref(preset: Any) -> Any:
        data = cache.load_json(preset.path) if preset.path else None
        return (
            ((data or {}).get("discovery") or {}).get("fit_ref") if isinstance(data, dict) else None
        )

    decided: dict[str, tuple[str, Any]] = {}  # path -> (key, Fit or None)

    def persist() -> None:
        """Save the decisions made so far (each as soon as it is made: an interrupted run
        keeps them), with the date groups of their folders."""
        entries, groups = hud.load_game_keys(project), hud.load_game_groups(project)
        for path, (key, fit) in decided.items():
            entries[hud.source_id(path)] = {
                "tag": hud.source_tag(path),
                "guess": guesses[path],
                "key": key,
                "fit": fit.record() if fit is not None else None,
            }
            folder, stem = hud.date_group(path, project)
            if stem is not None:
                groups[folder] = stem
        hud.save_game_keys(project, entries, groups)

    def made_from(key: str, s: Any) -> bool:
        """Whether the preset ``key`` was discovered from source ``s`` (its OCR records)."""
        preset = load_preset(key, project)
        data = cache.load_json(preset.path) if preset is not None and preset.path else None
        ids = (
            ((data or {}).get("discovery") or {}).get("frames") if isinstance(data, dict) else None
        )
        prefix = f"{discover.src_key(s)}@"
        return isinstance(ids, list) and any(str(i).startswith(prefix) for i in ids)

    for game, srcs in by_game.items():
        existing = load_preset(game, project)
        tested = [s for s in srcs if str(s.path) in guesses]
        # a discovery whose sources are checked against each other installs its preset only
        # after the check: an interrupted run leaves no preset of two games mixed
        split = existing is None and len(srcs) > 1 and bool(tested)
        proposal = (
            run(game, srcs, existing, install=not split) if existing is None or force else None
        )
        # a forced discovery over an existing preset: the sources are checked against its grid
        # fit_ref, or, without one, against each other like a new discovery; the preset gets
        # the fit_ref of the sources that fit (its regions stay)
        redo = existing is not None and proposal is not None
        if not tested:
            if redo:  # every source is the game (--game, a match hint): all of them are the group
                assert existing is not None and proposal is not None
                add_fit_ref(game, existing.path, (proposal.get("discovery") or {}).get("fit_ref"))
            continue
        if existing is None and len(srcs) == 1:  # nothing to compare with
            decided[str(tested[0].path)] = (game, None)
            persist()
            continue
        got = fit_frames(game, proposal, srcs, tested)
        old_ref = fit_ref(existing) if redo else None
        anchored = game_fit.valid_ref(old_ref) and bool(old_ref["grid"])
        if redo and anchored:  # the preset's grid decides; those that fit make the new one
            assert existing is not None
            profs = {
                str(s.path): game_fit.profile(got.get(str(s.path), []), params.min_score)
                for s in srcs
            }
            fits = {str(s.path): game_fit.against_ref(profs[str(s.path)], old_ref) for s in tested}
            if any(f is not None and f.ok for f in fits.values()):  # else the old ref stays
                group = game_fit.Profile()
                for k, prof in profs.items():
                    fit = fits.get(k)
                    if fit is None or fit.ok:  # decided before, inconclusive, or fits
                        group += prof
                add_fit_ref(game, existing.path, group.ref())
        elif existing is None or redo:  # the sources of this discovery against each other
            profs = {
                str(s.path): game_fit.profile(got.get(str(s.path), []), params.min_score)
                for s in srcs
            }
            fits = game_fit.split_group(
                profs,
                {str(s.path): float(s.duration) for s in srcs},
                frozenset(p for p in profs if p not in guesses),
            )
            if redo:  # the leader's group: the decided sources and those that fit
                assert existing is not None
                group = game_fit.Profile()
                for k, prof in profs.items():
                    fit = fits.get(k)
                    if k not in fits or (fit is not None and fit.ok):
                        group += prof
                add_fit_ref(game, existing.path, group.ref())
        else:  # against the preset found
            ref = fit_ref(existing)  # none, or an earlier version's: no decision, no split
            fits = {
                str(s.path): game_fit.against_ref(
                    game_fit.profile(got.get(str(s.path), []), params.min_score), ref
                )
                for s in tested
            }
        misfits = []
        for s in tested:
            fit = fits.get(str(s.path))
            if fit is None:
                logger.debug("%s: HUD fit check for %r inconclusive; keeping it", s.name, game)
            if fit is not None and not fit.ok:
                misfits.append(s)
            else:
                decided[str(s.path)] = (game, fit)
        if split:  # the preset from the sources that fit only, or as discovered
            if misfits:
                keep = {discover.src_key(s) for s in srcs if s not in misfits}
                proposal = run(game, srcs, None, keep=keep, install=False)
            assert proposal is not None
            install_preset(game, proposal)
        persist()
        # presets made for earlier misfits: (key, that misfit's profile, its name)
        made: list[tuple[str, Any, str]] = []
        for s in misfits:
            path, fit = str(s.path), fits[str(s.path)]
            prof = game_fit.profile(got.get(path, []), params.min_score)
            for key, other_prof, other in made:
                again = game_fit.against_run(prof, other_prof)
                if again is not None and again.ok:
                    logger.info("%s fits the HUD preset '%s' discovered for %s", s.name, key, other)
                    decided[path] = (key, again)
                    persist()
                    break
            else:
                taken = set(by_game) | {k for k, _ in decided.values()}
                # the key this source had before (also a stale decision) or whose preset was
                # discovered from it is its own, not another game's
                mine = before.get(path)
                new = hud.misfit_key(
                    path,
                    lambda k, taken=taken, mine=mine, s=s: (
                        k != mine
                        and (
                            k in taken
                            or (load_preset(k, project) is not None and not made_from(k, s))
                        )
                    ),
                )
                logger.info(
                    "%s does not fit the HUD preset '%s' (score %.2f); discovering it as '%s'",
                    s.name,
                    game,
                    fit.score,
                    new,
                )
                decided[path] = (new, fit)
                persist()  # before its discovery: an interrupted one keeps the key
                own = load_preset(new, project)
                if own is None or force:
                    prop = run(new, [s], own)
                    if own is not None:  # one source: it is the group
                        ref = (prop.get("discovery") or {}).get("fit_ref")
                        add_fit_ref(new, own.path, ref)
                made.append((new, prof, s.name))
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
