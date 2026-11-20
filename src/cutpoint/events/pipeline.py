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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

from cutpoint import cache, jsonout, memguard, progress, tsrepair
from cutpoint import paths as cutpoint_paths
from cutpoint.errors import LowMemoryError
from cutpoint.events import explosions, paths
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
)
from cutpoint.events.eventtypes import types_for
from cutpoint.events.killfeed import (
    OcrEngine,
    OcrParams,
    derive_streaks,
    detect_ocr_events,
    link_teamkills,
)
from cutpoint.events.layers import ACTION_CAM_LAYERS, DEFAULT_LAYERS
from cutpoint.events.menu_spans import ocr_stats_key
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
    resolve_counts,
    select_windows,
    windows_to_events,
)
from cutpoint.sources import parts as source_parts
from cutpoint.styles import ACTION_CAM, analysis_names, project_kind
from cutpoint.utils import footage_context
from cutpoint.vision import VisionRater

logger = logging.getLogger(__name__)
# Share (percent) of the OCR step's progress HUD discovery takes when it runs; the
# per-source OCR takes the rest. Measured first runs: discovery was about 4 of 7 min of
# OCR on a 7 min file and 7 of 14 min on two 10 min files.
DISCOVERY_PCT = 50.0
_FAILURE = "_cutpoint_ocr_failure"  # the attribute :func:`ocr_failure` reads


def check_game_key(key: str) -> str:
    """``key`` when it can be a HUD preset's file stem and a ``discovery`` folder name: not
    empty, not ``.`` or ``..``, no path separator or drive (``/``, ``\\``, ``:``), no NUL.

    Raises:
        jsonout.ArgumentError: For any other key (``--game ../x`` would write outside the
            presets and project folders).
    """
    if not key.strip() or key in (".", "..") or any(c in key for c in "/\\:\0"):
        raise jsonout.ArgumentError(
            f"HUD preset key {key!r} (--game, or a preset's game): a key is a name without "
            "a path (no /, \\, :, '..'; e.g. counter-strike-2)"
        )
    return key


class OcrFailure(NamedTuple):
    """Where the OCR layer failed (:func:`ocr_failure`)."""

    not_read: list[str]  # the paths of the sources it did not read, the failing one first
    discovery: bool  # in HUD region discovery: it read none
    games: int  # discovery: the games it was to discover (0: not known yet)


def _mark_ocr_failure(
    err: BaseException,
    not_read: list[Any],
    *,
    discovery: bool,
    games: int = 0,
) -> None:
    """Note on ``err`` (re-raised unchanged) the sources the OCR layer did not read."""
    paths = [str(s.path) for s in not_read]
    setattr(err, _FAILURE, OcrFailure(paths, discovery, games))


def ocr_failure(err: BaseException) -> OcrFailure | None:
    """Where the OCR layer failed with ``err``: the sources it did not read (the one it
    failed on first), whether it failed in HUD region discovery (then it read none, and no
    preset was installed for at least one game) and how many games discovery was to
    discover. The sources read before keep
    their events and menu spans. None: ``err`` did not come from the OCR layer's discovery
    or per-source reads. An out-of-memory error there is raised as a
    :class:`~cutpoint.errors.LowMemoryError` instead (it stops the run, resumable)."""
    return getattr(err, _FAILURE, None)


# How to avoid HUD discovery, after "Put a preset in <presets dir>" (``cutpoint events``'
# options; ``cutpoint analyze`` passes its own: :data:`cutpoint.analyze.DISCOVER_HINT`)
EVENTS_DISCOVER_HINT = "and pass --game, or --no-discover to skip"

# what a user sets in a preset, carried over when a stale one is discovered again (#1085)
USER_PRESET_KEYS = (
    "player_tags",
    "banner_phrases",
    "match",
    "events",
    "title",
    "view",
    "hype",
    "bigtext",
)


def _require_events_extra(layers: tuple[str, ...]) -> None:
    """Fail before any side effect (loading a model) when the events extra is missing.

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
    makes before it loads a model or imports torch; callers that run a long step
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
    context: str | None = None,
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
        context: ``analyze --context`` (#1068) for the crash layer's activity profile, as
            ``crashes --context``; None: :func:`cutpoint.activity.default_context` (the
            analysis's ``settings.context`` while the project has no profile yet). The VLM
            layer's prompts carry it, else the analysis's ``settings.context``.
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
    # copies: the model, context and wording set below are this project's, not the caller's
    audio = audio or AudioParams()
    ocr = replace(ocr, project=project) if ocr else OcrParams(project=project)
    vlm = replace(vlm) if vlm else VlmParams()
    if ocr.game != "auto":
        check_game_key(ocr.game)
    boom = explosions.ExplosionParams()
    hl = load_highlights(project)
    is_cam = project_kind(hl) == ACTION_CAM
    if layers is None:
        layers = ACTION_CAM_LAYERS if is_cam else DEFAULT_LAYERS
    if cached_ocr is None:
        cached_ocr = not is_cam
    preflight(layers)
    rater, hosted, online = _vision_client(layers, vlm, ollama_url, ollama_model)
    sources, bad = _load_sources(project, hl, only, avoid)
    run = _Run(
        project=project,
        hl=hl,
        sources=sources,
        layers=layers,
        force=force,
        stage=stage or _layer_step,
        batch_size=batch_size,
        bad=bad,
        is_cam=is_cam,
        crash_run="crash" in layers and not only,  # project-wide ranking: all sources
        layer1={str(s.path): [] for s in sources},
    )
    by_path = {str(s.path): s for s in sources}
    transcript = load_transcript_lines(project)

    # ---- layer 1a: audio
    voice_params = None
    voice_needs = False
    if "voice" in layers:
        from cutpoint.events import voice as voice_layer  # noqa: PLC0415

        voice_params = voice_layer.VoiceParams()
        voice_needs = voice_layer.needs_run(sources, voice_params, force="voice" in force)
    _audio_layer(run, audio, keep=voice_needs)
    # ---- layer 1c/1d: wind-robust voice grids, frame embeddings + people
    _voice_layer(run, voice_params, voice_needs)
    # The visual layer's caption embeddings use the captions of the project's activity.json
    # (generic until the crash layer below writes one); the contact sheets are written after the
    # crash layer when it runs, so a first run's sheets use the profile it resolves.
    from cutpoint.activity import project_prompts  # noqa: PLC0415

    prompts = project_prompts(project)
    _visual_layer(run, prompts, ocr.hwaccel, encoder)
    if sheets and not run.crash_run:
        _write_sheets(run, prompts)

    # ---- layer 1b: OCR (with region discovery for games without a preset)
    t0 = time.perf_counter()
    with run.stage("ocr") if "ocr" in layers else contextlib.nullcontext():
        base = 0.0  # progress the per-source OCR starts at (after discovery)
        if "ocr" in layers and discover is not False:
            base = _ocr_discovery(
                run,
                ocr,
                rater,
                t0,
                discover=discover,
                bursts=discover_bursts,
                scope=preset_scope,
                hint=discover_hint,
                context=context,
                prefetch=prefetch,
            )
        _ocr_layer(run, ocr, base, prefetch, cached_ocr=cached_ocr, debug=ocr_debug)
    run.timings["ocr_s"] = round(time.perf_counter() - t0, 1)
    _confirm_explosions(run, boom)
    if run.make_engine is not None:  # no OCR after this: the engine is not kept through the VLM
        if prefetch is not None and run.make_engine is prefetch.make_engine:
            prefetch.close()  # its worker uses the engine: stopped first (idempotent)
        _release_engine(run.make_engine)
        run.make_engine = None

    # ---- crashes (after voice + visual, whose caches it reads): crashes.json
    if run.crash_run:
        _crash_layer(
            run,
            encoder,
            context=context,
            llm=llm,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            sheets=sheets,
        )

    # ---- layer 2: VLM windows
    vlm, cam_prompts = _vlm_params(run, vlm, context)
    results = _vlm_layer(run, vlm, ocr.game, rater, hosted, online, by_path, transcript)
    run.stats["vram_mib"] = run.vram

    talk: list[dict[str, Any]] = []
    if dialogue is not None:
        with run.stage("dialogue"):
            talk = dialogue(run.layer1)
    else:  # not read by this run: what an earlier one marked stays
        talk = kept_dialogue(project, [str(s.path) for s in sources])
    with run.stage("write"):
        return _write_events(run, results, talk, transcript, vlm, cam_prompts, audio, boom, ocr)


@dataclass
class _Run:
    """What the layers of one :func:`run_events` share: its inputs and what they collect."""

    project: Path
    hl: dict[str, Any]
    sources: list[Any]
    layers: tuple[str, ...]
    force: tuple[str, ...]
    stage: Callable[[str], AbstractContextManager[Any]]
    batch_size: int
    bad: dict[str, list[tuple[float, float]]]  # ranges never decoded, per source path
    is_cam: bool
    crash_run: bool  # the crash layer runs (project-wide: all sources)
    layer1: dict[str, list[dict[str, Any]]]  # the layer-1 events per source path
    timings: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    vram: dict[str, Any] = field(default_factory=dict)
    # PANNs, handed from the audio layer to the voice layer instead of loading it again (#372)
    panns: Any = None
    panns_failed: bool = False  # the audio layer could not load PANNs: voice does not retry (#629)
    kept: list[Any] = field(default_factory=list)  # a CLIP handed over to the crash layer
    # the OCR engine's maker; a stage that suppresses the OCR layer's error keeps the current one
    make_engine: Callable[[], OcrEngine] | None = None


def _raise_if_all_failed(
    errors: list[Exception], total: int, before: Callable[[], None] | None = None
) -> None:
    """Raise the last of a layer's per-source ``errors`` when every one of the ``total``
    sources failed: not one bad file, the layer failed (``before`` runs first)."""
    if errors and len(errors) == total:
        if before is not None:
            before()
        raise errors[-1]


def _vision_client(
    layers: tuple[str, ...], vlm: VlmParams, ollama_url: str | None, ollama_model: str | None
) -> tuple[VisionRater | None, bool, bool | None]:
    """The shared vision client (None: not usable), whether it is hosted, and whether it is
    online when that was asked already; sets ``vlm.model``, the key of the descriptions."""
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
    return rater, hosted, online


def _load_sources(
    project: Path,
    hl: dict[str, Any],
    only: tuple[str, ...],
    avoid: tuple[tuple[str, float, float], ...],
) -> tuple[list[Any], dict[str, list[tuple[float, float]]]]:
    """The reachable sources of ``hl`` that match ``only``, and the ranges never decoded."""
    # joined MPEG-TS/PS recordings are read through their stream copy (#1052)
    tsrepair.prepare_sources(
        [Path(s["path"]) for s in hl["sources"]],
        required=[s["path"] for s in hl["sources"] if s.get("timeline")],
    )
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
    return sources, bad


def _audio_layer(run: _Run, audio: AudioParams, *, keep: bool) -> None:
    """Layer 1a: PANNs sound events of every source (cached ones read when the layer does not
    run). ``keep``: PANNs is handed to the voice layer (``run.panns``) instead of closed."""
    t0 = time.perf_counter()
    sources = run.sources
    if "audio" in run.layers:
        with run.stage("audio"):
            # no Ollama unload up front: a run with every source cached loads nothing, and
            # PANNs' memory guard (memguard.device_for) frees Ollama's models when it is short
            with VramMonitor() as vm:
                guard = SourceGuard("Audio events")
                try:
                    for i, s in enumerate(sources):
                        if run.batch_size and i and not i % run.batch_size:  # a batch boundary
                            if run.panns is not None:
                                run.panns.close()
                                run.panns = None
                            _boundary("audio", i, len(sources))
                        progress.update(progress.fraction(i, len(sources)))
                        evs = None if "audio" in run.force else cached_audio_events(s, audio)
                        if evs is None and not s.audio_parts:
                            logger.warning(
                                "%s: no cached 16 kHz audio; run `cutpoint analyze` first", s.name
                            )
                            evs = []  # nothing to detect: no model load
                        if evs is None:
                            if run.panns is None:
                                try:
                                    run.panns = PannsDetector()
                                except Exception:
                                    run.panns_failed = True
                                    raise
                                # the device of this run, as voice and visual record theirs:
                                # a run that only reads cached events loads no PANNs
                                run.stats["audio_device"] = run.panns.device
                            try:  # miss or forced
                                evs = detect_audio_events(s, run.panns, audio, force=True)
                            except Exception as err:  # noqa: BLE001 - one bad source, logged
                                guard.skip(err, s.name, s.path)
                                evs = []
                        run.layer1[str(s.path)] += [{**e, "kind": "audio"} for e in evs]
                    guard.raise_if_all_failed(len(sources))  # not one bad file: it failed
                except BaseException:
                    keep = False  # the layer failed: free the model now
                    raise
                finally:
                    if run.panns is not None and not keep:
                        run.panns.close()
                        run.panns = None
            run.vram["audio"] = vm.summary()
    else:
        for s in sources:
            run.layer1[str(s.path)] += [
                {**e, "kind": "audio"} for e in detect_audio_events(s, None, audio)
            ]
    run.timings["audio_s"] = round(time.perf_counter() - t0, 1)


def _voice_layer(run: _Run, voice_params: Any, voice_needs: bool) -> None:
    """Layer 1c: wind-robust voice grids (PANNs on high-passed audio + Silero VAD), with the
    audio layer's PANNs when it handed it over; skipped when PANNs did not load."""
    if "voice" not in run.layers:
        return
    if run.panns_failed and voice_needs:
        logger.info("Voice layer skipped: PANNs did not load (see the audio layer's error)")
        return
    from cutpoint.events import voice as voice_layer  # noqa: PLC0415

    sources = run.sources
    t0 = time.perf_counter()
    try:
        with run.stage("voice"), VramMonitor() as vm:
            voice_errors: list[Exception] = []  # of every batch: one bad file skips itself
            for k, part in enumerate(progress.batches(sources, run.batch_size)):
                if k:  # a batch boundary: the PANNs of the batch before is closed
                    if run.panns is not None:
                        run.panns.close()
                        run.panns = None
                    _boundary("voice", k * run.batch_size, len(sources))
                st = voice_layer.run_layer(
                    part,
                    voice_params,
                    force="voice" in run.force,
                    det=run.panns,
                    errors=voice_errors,
                )
                run.stats["voice"] = merge_stats(run.stats["voice"], st) if k else st
            _raise_if_all_failed(voice_errors, len(sources))
    finally:
        if run.panns is not None:  # closed before the next layer loads its model
            run.panns.close()
            run.panns = None
    run.vram["voice"] = vm.summary()
    run.timings["voice_s"] = round(time.perf_counter() - t0, 1)


def _visual_layer(run: _Run, prompts: Any, hwaccel: Any, encoder: Any) -> None:
    """Layer 1d: frame embeddings + people (``visual``), and the AI Chat's light frame index
    (``visual_light``); a CLIP either loads goes to the crash layer (``run.kept``) when no OCR
    runs in between."""
    from cutpoint.activity import use_prompts  # noqa: PLC0415

    sources = run.sources
    if "visual" in run.layers:
        from cutpoint.crash import ClipEncoder  # noqa: PLC0415
        from cutpoint.events import visual as visual_layer  # noqa: PLC0415

        # Hand CLIP over only when no OCR runs in between: one model resident at a time.
        handoff = run.crash_run and "ocr" not in run.layers and encoder is None
        t0 = time.perf_counter()
        with run.stage("visual"), VramMonitor() as vm, use_prompts(prompts):
            visual_errors: list[Exception] = []  # of every batch: one bad file skips itself
            parts = progress.batches(sources, run.batch_size)
            for k, part in enumerate(parts):
                if k:  # a batch boundary: run_layer closed the models of the batch before
                    _boundary("visual", k * run.batch_size, len(sources))
                last = k == len(parts) - 1  # only the last batch hands CLIP over
                st = visual_layer.run_layer(
                    part,
                    visual_layer.VisualParams(hwaccel=hwaccel),
                    force="visual" in run.force,
                    bad=run.bad,
                    handoff=(
                        (lambda m: run.kept.append(ClipEncoder.from_visual(m)))
                        if handoff and last
                        else None
                    ),
                    errors=visual_errors,
                )
                run.stats["visual"] = merge_stats(run.stats["visual"], st) if k else st

            def free_kept() -> None:  # a CLIP handed over by the last batch: freed with the layer
                while run.kept:
                    run.kept.pop().close()

            _raise_if_all_failed(visual_errors, len(sources), before=free_kept)
        run.vram["visual"] = vm.summary()
        run.timings["visual_s"] = round(time.perf_counter() - t0, 1)
    if "visual_light" in run.layers:  # chat only: no event, no ranking reads it (#989)
        t0 = time.perf_counter()
        with run.stage("visual_light"):
            # CLIP goes to the crash layer only when no OCR runs in between (as the visual
            # layer's): one model resident at a time
            run.stats["visual_light"] = _light_layer(
                run.hl,
                sources,
                run.kept,
                encoder,
                run.crash_run and "ocr" not in run.layers,
            )
        run.timings["visual_light_s"] = round(time.perf_counter() - t0, 1)


def _write_sheets(run: _Run, prompt_set: Any) -> None:
    """The visual layer's contact sheets, under the captions of ``prompt_set``."""
    from cutpoint.activity import use_prompts  # noqa: PLC0415
    from cutpoint.events.sheets import write_visual_sheets  # noqa: PLC0415

    with use_prompts(prompt_set):
        run.stats["sheets"] = [str(p) for p in write_visual_sheets(run.project, run.sources)]


def _ocr_discovery(
    run: _Run,
    ocr: OcrParams,
    rater: VisionRater | None,
    t0: float,
    *,
    discover: bool | None,
    bursts: int,
    scope: str,
    hint: str,
    context: str | None,
    prefetch: OcrPrefetch | None,
) -> float:
    """HUD region discovery for the OCR layer (:func:`_discover`); returns the progress the
    per-source OCR starts at."""
    from cutpoint.events.feed_labels import VisionCalls, make_labeler  # noqa: PLC0415
    from cutpoint.events.hud_map import make_mapper  # noqa: PLC0415

    # the HUD map and the feed labels share the model check, load and budget; the map's
    # prompt carries the user's words about the footage (#1068)
    calls = VisionCalls(rater) if rater is not None else None
    words = footage_context(
        context if context is not None else (run.hl.get("settings") or {}).get("context")
    )
    try:
        run.stats["discovery"] = _discover(
            run.project,
            run.sources,
            ocr,
            discover,
            bursts,
            scope,
            hint,
            make_labeler(rater, calls) if rater is not None and calls else None,
            make_mapper(rater, words, calls) if rater is not None and calls else None,
            # a top-up changes the preset the background OCR reads with: stopped first
            before_top_up=prefetch.close if prefetch is not None else None,
        )
    except LowMemoryError:
        raise
    except Exception as err:
        memguard.raise_if_out_of_memory(err, "HUD region discovery", "every source")
        if ocr_failure(err) is None:  # before _discover knew its games
            _mark_ocr_failure(err, run.sources, discovery=True)
        raise
    run.timings["discovery_s"] = round(time.perf_counter() - t0, 1)
    return DISCOVERY_PCT if run.stats["discovery"] else 0.0


def _ocr_layer(
    run: _Run,
    ocr: OcrParams,
    base: float,
    prefetch: OcrPrefetch | None,
    *,
    cached_ocr: bool,
    debug: Path | None,
) -> None:
    """Layer 1b: the HUD OCR of every source (cached reads when the layer does not run and
    ``cached_ocr``), from ``base`` percent of the step on; its engine maker is
    ``run.make_engine``."""
    layers, force, sources = run.layers, run.force, run.sources
    # the OCR model loads only when a source's footage is read (not for a replay)
    if "ocr" in layers:
        run.make_engine = prefetch.make_engine if prefetch else functools.cache(OcrEngine)
    if not ocr.player_tags and sources and ("ocr" in layers or cached_ocr):
        logger.warning(
            "No player_tags set, so kills and deaths in the HUD cannot be "
            'attributed to you. Add "player_tags": ["YourName"] to %s '
            "or pass --player NAME.",
            cutpoint_paths.config_file(),
        )
    ocr_stats: dict[str, Any] = {}
    run.stats["ocr"] = ocr_stats  # the sources read before a failure keep their menu spans
    with VramMonitor() as vm:
        ocr_sources = sources if "ocr" in layers or cached_ocr else []  # action-cam: no HUD
        shown = [base]  # the step's pct never goes back

        def on_source(done: float, n: int = len(ocr_sources)) -> None:
            pct = base + (100 - base) * progress.fraction(done, n) / 100
            if pct > shown[0]:
                shown[0] = pct
                progress.update(pct)

        for i, s in enumerate(ocr_sources):
            if run.batch_size and i and not i % run.batch_size and "ocr" in layers:
                # a batch boundary: the engine is made again for the next batch (the
                # background reader's is in use until it is stopped)
                if prefetch is None or run.make_engine is not prefetch.make_engine:
                    _release_engine(run.make_engine)
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
            try:
                if prefetch is not None:  # the background read of these exact inputs
                    plain = "ocr" not in force and not debug and not run.bad.get(str(s.path))
                    if "ocr" in layers and plain:
                        got = prefetch.take(s, ocr, on_progress)
                    else:
                        prefetch.close()
                    if got is None and prefetch.stuck:  # it outlived close(): not its engine
                        prefetch.wait_for(s.path)  # nor the raw cache of the source it is on
                        if run.make_engine is prefetch.make_engine:
                            run.make_engine = functools.cache(OcrEngine)
                evs, st = got or detect_ocr_events(
                    s,
                    ocr,
                    None,
                    force="ocr" in force,
                    debug_dir=debug,
                    avoid=run.bad.get(str(s.path)),
                    on_progress=on_progress,
                    make_engine=run.make_engine,
                )
                game = game_key_for_path(str(s.path), ocr.game, run.project)
                preset = load_preset(game, run.project)
                teamkills = bool(preset and preset.feed.get("teamkills"))
                link_teamkills(evs, generic_teamkills=teamkills)
                derive_streaks(evs)
            except LowMemoryError:
                raise
            except Exception as err:
                memguard.raise_if_out_of_memory(err, "the HUD OCR", s.name)
                _mark_ocr_failure(err, ocr_sources[i:], discovery=False)
                raise
            run.layer1[str(s.path)] += evs
            ocr_stats[ocr_stats_key(s.path, [t.path for t in sources])] = st
    run.vram["ocr"] = vm.summary()
    run.stats["ocr"] = ocr_stats


def _confirm_explosions(run: _Run, boom: explosions.ExplosionParams) -> None:
    """Explosions (#801): a heard boom counts once the loudness jumps with it and the picture
    was not black just before it (a loading screen ending)."""
    t0 = time.perf_counter()
    booms = sum(
        explosions.confirm(
            run.layer1[str(s.path)],
            [(p.offset, p.cdir) for p in source_parts(s.path)],
            boom,
            luma=explosions.luma_reader(s.path),  # --recover parts share the file and its time
        )
        for s in run.sources
        if explosions.has_booms(run.layer1[str(s.path)])
    )
    run.timings["explosions_s"] = round(time.perf_counter() - t0, 1)
    dark = sum(
        1
        for s in run.sources
        for e in run.layer1[str(s.path)]
        if e.get("kind") == "audio" and e.get("label") == explosions.LABEL and e.get("after_black")
    )
    if booms or dark:
        logger.info(
            "%d explosion(s) confirmed by a loudness jump, %d set aside after a black screen",
            booms,
            dark,
        )


def _crash_layer(
    run: _Run,
    encoder: Any,
    *,
    context: str | None,
    llm: bool | str,
    ollama_url: str | None,
    ollama_model: str | None,
    sheets: bool,
) -> None:
    """The crash layer (crashes.json), with the CLIP the visual layer handed over or
    ``encoder`` (closed after it); then the contact sheets under the activity it wrote."""
    from cutpoint.activity import default_context, project_prompts  # noqa: PLC0415
    from cutpoint.crash import detect_project  # noqa: PLC0415

    t0 = time.perf_counter()
    enc = run.kept.pop() if run.kept else encoder
    try:
        with run.stage("crash"), VramMonitor() as vm:
            detect_project(
                run.project,
                force="crash" in run.force,
                context=default_context(run.project, context=context, hl=run.hl),
                encoder=enc,
                warn_unreachable=False,
                llm=llm,
                batch_size=run.batch_size,
                ollama_url=ollama_url,
                ollama_model=ollama_model,
            )
    finally:
        if enc is not None:
            enc.close()
    run.vram["crash"] = vm.summary()
    run.timings["crash_s"] = round(time.perf_counter() - t0, 1)
    if sheets:  # under the activity.json the crash layer just wrote
        _write_sheets(run, project_prompts(run.project))


def _vlm_params(run: _Run, vlm: VlmParams, context: str | None) -> tuple[VlmParams, Any]:
    """The VLM layer's parameters for this project (telemetry seeds of action-cam footage go
    into ``run.layer1``), and the activity prompts of its action-cam wording (None: not
    action-cam)."""
    from cutpoint.activity import project_prompts  # noqa: PLC0415

    # ---- telemetry (action-cam): airtime / crash seeds + coverage of the riding
    if run.is_cam:
        for src, evs in _telemetry_layer(run.project, run.sources, vlm).items():
            run.layer1.setdefault(src, []).extend(evs)
    auto = vlm.max_windows is None
    vlm = resolve_counts(vlm, run.is_cam, run.layer1)
    if auto and "vlm" in run.layers:
        logger.info("VLM: up to %d windows", vlm.max_windows)
    # action-cam: worded for the activity.json the crash layer above may have just written
    cam_prompts = project_prompts(run.project) if run.is_cam else None
    vlm.action_cam = action_cam_wording(cam_prompts) if cam_prompts is not None else None
    # the user's words about the footage (#1068): analyze's --context, else the recorded one
    vlm.context = footage_context(
        context if context is not None else (run.hl.get("settings") or {}).get("context")
    )
    return vlm, cam_prompts


def _vlm_layer(
    run: _Run,
    vlm: VlmParams,
    ocr_game: str,
    rater: VisionRater | None,
    hosted: bool,
    online: bool | None,
    by_path: dict[str, Any],
    transcript: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Layer 2: the windows the vision model describes (cached descriptions when the layer
    does not run or cannot ask); returns the described windows."""
    layers, force = run.layers, run.force
    if "vlm" in layers:
        progress.start("vlm")  # cutpoint events: the step of progress.json
        free_torch()  # the CUDA memory torch still caches for the layers before
    t0 = time.perf_counter()
    windows, prompts = _vlm_windows(run, vlm, ocr_game)
    # a provider the user did not agree to (a hosted one, or an Ollama on another machine): no
    # request, no cached descriptions or windows either, as for analyze's vision ratings
    no_vision = rater is None or rater.consent_needed is not None
    if run.is_cam and "vlm" not in force and not no_vision:
        windows = _reuse_windows(run, vlm, windows)
    windows = [
        w
        for w in windows
        if not any(w.start < b + 2 and w.end > a - 2 for a, b in run.bad.get(w.source, []))
    ]
    layer1 = run.layer1
    results: list[dict[str, Any]] = []
    if no_vision:
        cached, _ = describe_windows(None, by_path, windows, vlm, prompts, transcript, layer1)
        # the VLM layer is skipped; say why cached descriptions (or reused windows) are gone
        if rater is not None and (
            "vlm" in layers
            or cached
            or (run.is_cam and "vlm" not in force and cached_windows(run.sources, vlm))
        ):
            rater.warn_consent()
    elif "vlm" in layers and rater is not None:  # (not no_vision: rater is not None)
        results = _describe(run, vlm, windows, prompts, rater, hosted, online, by_path, transcript)
    else:
        results, _ = describe_windows(None, by_path, windows, vlm, prompts, transcript, layer1)
    run.timings["vlm_s"] = round(time.perf_counter() - t0, 1)
    return results


def _vlm_windows(run: _Run, vlm: VlmParams, ocr_game: str) -> tuple[list[Any], dict[str, Any]]:
    """The VLM windows of this run and the game prompt per source path."""
    project = run.project
    games = {
        str(s.path): ACTION_CAM if run.is_cam else game_key_for_path(str(s.path), ocr_game, project)
        for s in run.sources
    }
    # Game prompt and event types (custom seeds) from each source's HUD preset; action-cam has none.
    # The prompt asks about kills only for footage with kills (#596).
    presets = {} if run.is_cam else {src: load_preset(g, project) for src, g in games.items()}
    prompts = {
        src: game_prompt(games[src], p, has_kills(p, run.layer1.get(src, [])))
        for src, p in presets.items()
    }
    seed_types = {src: types_for(p) for src, p in presets.items()}
    windows = select_windows(
        run.sources,
        [c for c in run.hl.get("candidates", []) if c.get("pool") != "visual"],  # no pool seeds
        run.layer1,
        vlm,
        seed_types,
        quiet="vlm" not in run.layers,
        style=analysis_names(run.hl),
    )
    return windows, prompts


def _reuse_windows(run: _Run, vlm: VlmParams, windows: list[Any]) -> list[Any]:
    """Action-cam: reuse described windows; only fill new footage."""
    prev = cached_windows(run.sources, vlm)
    if not prev:
        return windows
    n_new = len(windows)
    # a reused window this run selects again keeps its reasons (#109: a resumed run's
    # events.json equals an uninterrupted one's)
    chosen = {(w.source, w.key): w for w in windows}
    windows = [chosen.get((w.source, w.key), w) for w in add_incremental(prev, windows)]
    logger.log(
        logging.INFO if "vlm" in run.layers else logging.DEBUG,
        "VLM (action-cam, incremental): %d cached windows reused, %d of %d new "
        "windows cover footage not described yet",
        len(prev),
        len(windows) - len(prev),
        n_new,
    )
    return windows


def _describe(
    run: _Run,
    vlm: VlmParams,
    windows: list[Any],
    prompts: dict[str, Any],
    rater: VisionRater,
    hosted: bool,
    online: bool | None,
    by_path: dict[str, Any],
    transcript: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The VLM layer running: the windows not cached yet described by ``rater`` (cached
    descriptions when it is not available or stops); its stats go to ``run.stats["vlm"]``."""
    force, layer1 = "vlm" in run.force, run.layer1
    cached, vstats = describe_windows(
        None, by_path, windows, vlm, prompts, transcript, layer1, force=force
    )
    missing = len(windows) - vstats["cached"] if not force else len(windows)
    if not force:
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
                        force=force,
                        batch_size=run.batch_size,
                        unload=not hosted,
                    )
                run.vram["vlm"] = vm.summary()
            finally:
                rater.unload()
            if vstats.get("stopped"):
                left = len(windows) - vstats.get("ok", 0) - vstats.get("cached", 0)
                message = (
                    f"VLM: stopped because {vstats['stopped']}; {left} window(s) not "
                    "described, cached descriptions used (run again to describe them)"
                )
                logger.warning("%s", message)
                jsonout.warn("vlm_unavailable", message)
        else:
            if force:  # nothing to describe them with: keep the saved descriptions
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
    run.stats["vlm"] = {
        k: v for k, v in vstats.items() if k not in ("seconds", "prompt_tokens", "streak")
    }
    secs = vstats.get("seconds") or []
    if secs:
        run.stats["vlm"]["mean_s_per_window"] = round(sum(secs) / len(secs), 1)
        toks = [t for t in vstats.get("prompt_tokens", []) if t]
        run.stats["vlm"]["mean_prompt_tokens"] = round(sum(toks) / max(1, len(toks)))
    return results


def _write_events(
    run: _Run,
    results: list[dict[str, Any]],
    talk: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    vlm: VlmParams,
    cam_prompts: Any,
    audio: AudioParams,
    boom: explosions.ExplosionParams,
    ocr: OcrParams,
) -> tuple[Path, Path]:
    """events.json and events.txt of the run (:func:`write_outputs`)."""
    events = [
        normalize(src, e)
        for src, evs in run.layer1.items()
        for e in evs
        if e.get("type") not in ("coverage", "coverage_masked")
    ]
    events += [normalize(e["source"], e) for e in windows_to_events(results)]
    events += talk
    vlm_settings = asdict(vlm)
    if vlm_settings.get("context") is None:  # without --context: as before (#1068)
        vlm_settings.pop("context", None)
    if cam_prompts is not None and vlm.action_cam is not None:  # the activity, not its text
        wording = vlm.action_cam.key() or "default"
        vlm_settings["action_cam"] = {"activity": cam_prompts.name, "wording": wording}
    settings = {
        "layers": list(run.layers),
        "avoid": dict(run.bad),
        "audio": audio.key(),
        "explosions": boom.key(),
        "ocr": ocr.key(None) | {"game": ocr.game},
        "vlm": vlm_settings,
    }
    return write_outputs(
        run.project,
        [str(s.path) for s in run.sources],
        events,
        results,
        transcript,
        settings,
        run.timings,
        run.stats,
    )


def _light_layer(
    hl: dict[str, Any],
    sources: list[Any],
    kept: list[Any],
    encoder: Any,
    crash_run: bool,
    force: bool = False,
) -> dict[str, Any]:
    """The light frame index of every source that lacks frames of it (``analyze`` without a
    CUDA GPU, for the AI Chat's ``look_for``): one INFO line with the estimate, then CLIP on
    the CPU. Its CLIP (or ``encoder``, given) is handed to the crash layer through ``kept``
    when ``crash_run`` (it runs next, with no OCR in between), else closed. A source that fails
    is skipped with a WARNING. Too little memory for CLIP (its load or a failed allocation)
    does not stop the run, unlike the other layers: the layer stops with a WARNING naming the
    fix and a ``step_failed`` warning, and its CLIP is freed (the index is the chat's only)."""
    from cutpoint.crash import ClipEncoder  # noqa: PLC0415
    from cutpoint.events import visual as V  # noqa: PLC0415

    spans: dict[str, list[tuple[float, float]]] = {}
    for c in hl.get("candidates") or []:
        try:
            spans.setdefault(str(c["source"]), []).append((float(c["start"]), float(c["end"])))
        except (KeyError, TypeError, ValueError):
            continue
    todo = {}
    for s in sources:
        marks = V.light_todo(s, spans.get(str(s.path), []), force=force)
        if marks:
            todo[str(s.path)] = (s, len(marks))
    stats: dict[str, Any] = {"cached": len(sources) - len(todo), "computed": 0, "frames": 0}
    if not todo:
        return stats
    frames = sum(n for _, n in todo.values())
    logger.info(
        "No CUDA GPU: a light frame index for the AI Chat instead of keyframes "
        "(about %d min, %d frames)",
        V.light_minutes(frames, load=encoder is None),
        frames,
    )
    enc = encoder
    handed = False
    try:
        if enc is None:
            enc = ClipEncoder(device="cpu")
        guard = SourceGuard("Frame index")
        for i, (s, _) in enumerate(todo.values()):
            progress.update(progress.fraction(i, len(todo)))
            try:
                got = V.compute_light(s, enc, spans.get(str(s.path), []), force=force)
            except Exception as err:  # noqa: BLE001 - one bad source (out of memory: below)
                guard.skip(err, s.name, s.path)
                continue
            if got is not None:
                stats["computed"] += 1
                stats["frames"] += len(got.t)
        guard.raise_if_all_failed(len(todo))  # not one bad file: the layer failed
        if crash_run and encoder is None:
            kept.append(enc)  # the crash layer's dense confirm: no second load
            handed = True
    except LowMemoryError as err:  # chat only: the analysis goes on without it (#989)
        short = err.args[1] if len(err.args) > 1 else "out of memory"
        message = (
            f"Frame index for the AI Chat skipped: not enough free memory ({short}); the "
            "chat's visual search is unavailable until it is written. To add it, close other "
            "programs, then analyze the project again"
        )
        logger.warning("%s", message)
        jsonout.warn("step_failed", message)
        stats["low_memory"] = True
    finally:
        if encoder is None and enc is not None and not handed:
            enc.close()
    return stats


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
    streak = 0  # windows in a row without a usable reply, across the batches
    for k, part in enumerate(vlm_batches(windows, batch_size)):
        stopped = stats.get("stopped")
        if k and not stopped:  # a batch boundary
            if unload:
                rater.unload()
            progress.continuing("vlm", done, len(windows))
        # after a stop the later batches only take what is cached
        res, st = describe_windows(
            None if stopped else rater,
            by_path,
            list(part),
            vlm,
            prompts,
            transcript,
            layer1,
            force=force and not stopped,
            fails=streak,
        )
        streak = st.pop("streak", 0)  # merge_stats would sum it
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


class SourceGuard:
    """One bad source skips only itself (#985, #1033), the rule of every per-source loop of
    the events layers, the action-cam telemetry read, the contact sheets and the crash layer.

    In the loop, ``except Exception as err: guard.skip(err, name, path)`` (then ``continue``):
    a :class:`~cutpoint.errors.LowMemoryError` is raised again and an out-of-memory ``err``
    stops the run as one (:func:`~cutpoint.memguard.raise_if_out_of_memory`, resumable);
    any other error is logged at DEBUG with its traceback, a WARNING and a ``step_failed``
    warning name the source, and it is recorded in :attr:`failed`. After the loop,
    :meth:`raise_if_all_failed` raises the last error when every source failed: not one bad
    file, the step failed. KeyboardInterrupt is not an :class:`Exception`: it passes.

    Args:
        what: The step, in the message: ``"<name>: <what> skipped this source (<err>)"``.
        message: Another message (``str.format`` fields ``name``, ``what`` and ``err``).
        oom_what: The step in the out-of-memory error (default: ``what``).
        debug_what: The step in the DEBUG line ``"<path>: <debug_what> failed"`` (default:
            ``what``).
        log: The logger of the lines (default: this module's); the caller's keeps their
            logger name (``caplog`` and ``-v`` filters by it).
    """

    def __init__(
        self,
        what: str,
        *,
        message: str = "{name}: {what} skipped this source ({err})",
        oom_what: str | None = None,
        debug_what: str | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self.what = what
        self.log = log or logger
        self.message = message
        self.oom_what = oom_what or what
        self.debug_what = debug_what or what
        self.failed: dict[str, Exception] = {}  # str(path): its error, in the order they failed

    def skip(self, err: Exception, name: str, path: Path | str, *, warn: bool = True) -> None:
        """Skip the source ``name`` (``path``) that failed with ``err`` (see the class);
        ``warn=False``: logged at DEBUG only (the caller warned for it already).

        Raises:
            LowMemoryError: If ``err`` is one or is an out-of-memory error.
        """
        if isinstance(err, LowMemoryError):
            raise err
        memguard.raise_if_out_of_memory(err, self.oom_what, name)
        self.log.debug("%s: %s failed", path, self.debug_what, exc_info=True)
        if warn:
            message = self.message.format(name=name, what=self.what, err=err)
            self.log.warning("%s", message)
            jsonout.warn("step_failed", message, [path])
        self.failed[str(path)] = err

    def all_failed(self, total: int) -> bool:
        """Whether a source failed and as many failed as ``total`` (the sources it counts)."""
        return bool(self.failed) and len(self.failed) == total

    def raise_if_all_failed(self, total: int) -> None:
        """Raise the last source's error, unchanged (cli.py maps its type), when
        :meth:`all_failed`."""
        if self.all_failed(total):
            raise list(self.failed.values())[-1]


def _telemetry_layer(
    project: Path, sources: list[Any], vlm: VlmParams
) -> dict[str, list[dict[str, Any]]]:
    """Layer-1 entries from ``telemetry.json`` (written by ``cutpoint analyze --style action-cam``).

    Airtime and crash events become window seeds and events.txt lines.
    ``coverage`` pseudo-events (low priority, never written to events.json)
    tile the riding that is not on a lift or stopped, so spare VLM windows
    describe ordinary riding too (scenic views, friends, trees). A source whose telemetry
    fails to read keeps its events without coverage (a WARNING and a ``step_failed`` warning
    name it); when every source fails, the last error is raised.
    """
    from cutpoint.telemetry import DT, extract  # noqa: PLC0415

    data = cache.load_json(project / "telemetry.json") or {}
    out: dict[str, list[dict[str, Any]]] = {}
    guard = SourceGuard("Telemetry coverage")
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
        try:
            tel = extract(s.path, s.duration)
        except Exception as err:  # noqa: BLE001 - logged; its events stay, without coverage
            guard.skip(err, s.name, s.path)
            tel = None
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
    guard.raise_if_all_failed(len(sources))  # not one bad file: the layer failed
    return out


def _discover(
    project: Path,
    sources: list[Any],
    ocr: OcrParams,
    force: bool | None,
    bursts: int,
    scope: str,
    hint: str = EVENTS_DISCOVER_HINT,
    labeler: Callable[..., dict[str, Any]] | None = None,
    hud_mapper: Any = None,
    before_top_up: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run region discovery per game where needed; install new presets. ``labeler`` names the
    proposals with the shared vision model (:mod:`cutpoint.events.feed_labels`), after
    ``hud_mapper`` pointed at the HUD elements with it (:mod:`cutpoint.events.hud_map`).

    A game whose installed discovered preset (``<CUTPOINT_HOME>/presets``) has no
    ``discovery.hud_map`` yet is topped up once, when the vision model answers: discovery runs
    for it on this project's sources (its regions are left alone), the map's confirmed regions
    are added to the preset and a feed region OCR finds no feed lines in becomes a notice
    (:func:`cutpoint.events.hud_map.top_up`). Project presets are never topped up.
    ``before_top_up`` runs first (``cutpoint analyze`` stops its background OCR, which reads with
    the preset as it was). A top-up that gives no map (the model on the CPU alone, a timeout, too
    little memory) is recorded (:func:`cutpoint.events.hud_map.mark_tried`) and not run again
    while the preset file and the model stay the same.

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
        readout the preset does not have; never written into it) and, after a top-up,
        ``topped_up`` (the region names added, replaced and relabelled).
    """
    from cutpoint.events.discover_run import DiscoverRun  # noqa: PLC0415 - imports this module

    return DiscoverRun(
        project,
        sources,
        ocr,
        force,
        bursts,
        scope,
        hint,
        labeler,
        hud_mapper,
        before_top_up,
        load_preset=load_preset,
        game_key_for_path=game_key_for_path,
    ).run()


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
