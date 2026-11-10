"""``cutpoint edit``: one command from source videos to a finished edit.

Runs ``analyze`` -> (``events``) -> ``clips`` -> ``assemble`` in-process with the
options of a :data:`KINDS` entry, then copies the final video (or the timeline of
``--format fcpxml|otio``) and its subtitles to ``<name>-final.mp4`` / ``.srt`` in the
project folder (``<name>``: see :func:`edit_name`). The clips are kept in the clips folder
unless ``--no-keep-clips`` exports them into a temporary folder, removed after the join
(:func:`cutpoint.assemble.temporary_clips`). With ``CUTPOINT_PROGRESS=1`` it reports the
steps of :data:`cutpoint.progress.EDIT_STEPS` (analyze's, then clips and assemble).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from collections.abc import Collection
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

from cutpoint import cache, category, folders, jsonout, progress, runstate
from cutpoint.assemble import assemble, temporary_clips
from cutpoint.category import FUNNY, HYPE
from cutpoint.clips import export_clips
from cutpoint.encode import Lut
from cutpoint.planning import report_clips
from cutpoint.styles import ACTION_CAM, AUTO, AutoRequest, Style, get_style
from cutpoint.utils import file_stamp, find_videos, input_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Kind:
    """What one ``--kind`` makes.

    Attributes:
        style: Analysis style (``--style`` overrides it).
        top: Clip the N best moments (``clips --top``); None = no count limit.
        max_total: Cap the clips' total length in seconds (``clips --max-total``).
        order: Clip numbering and playback order (``clips --order``).
        transition: ``assemble --transition``.
        loudnorm: ``assemble --loudnorm``.
        captions: ``assemble --captions`` (burned in; whisper runs on the final).
        emphasis: ``--style auto``'s emphasis (:class:`~cutpoint.styles.AutoRequest`):
            ``funny`` for funny-moments, else ``hype``.
        auto: ``--style auto``: the style is detected from the footage; an ambiguous
            detection uses its best guess's style if that is gameplay, else the General
            style (:func:`~cutpoint.styles.auto_fallback`).
    """

    style: str
    top: int | None = None
    max_total: float | None = None
    order: str = "score"
    transition: str = "cut"
    loudnorm: bool = True
    captions: bool = False
    emphasis: str = HYPE
    auto: bool = False


KINDS: dict[str, Kind] = {
    "highlights": Kind("action", max_total=180, order="chronological", transition="fade"),
    "montage": Kind("action", max_total=60, order="score", transition="fade"),
    "clips": Kind("action", top=10, order="score", transition="cut"),
    "funny-moments": Kind(
        "funny", top=8, order="chronological", transition="cut", captions=True, emphasis=FUNNY
    ),
}


def resolve_kind(
    kind: str,
    *,
    style: str | None = None,
    top: int | None = None,
    max_total: float | None = None,
) -> Kind:
    """The :data:`KINDS` entry ``kind`` with the options the user gave replacing its own
    (``style`` :data:`~cutpoint.styles.AUTO` sets ``auto``)."""
    given = {"style": style, "top": top, "max_total": max_total}
    if style == AUTO:
        given = {"auto": True, "top": top, "max_total": max_total}
    return replace(KINDS[kind], **{k: v for k, v in given.items() if v is not None})


def edit_name(target: Path) -> str:
    """Name of an edit of ``target`` (project folder, ``<name>-final.mp4``): a file's stem,
    a folder's full name (``.`` and a drive root resolved), else ``edit``."""
    return input_name(target, "edit")


def _events_namespace(
    project: Path, layers: tuple[str, ...] | None, ocr_det_skip: bool = False
) -> argparse.Namespace:
    """``cutpoint events <project>`` arguments with every default, as the CLI parses them;
    ``layers`` (not None) is its ``--layers``, ``ocr_det_skip`` its ``--ocr-det-skip`` (the
    analysis's, so the OCR layer reuses its read)."""
    from cutpoint.events.cli import add_arguments  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    det_skip = "--ocr-det-skip" if ocr_det_skip else "--no-ocr-det-skip"
    return parser.parse_args([str(project), *(["--layers", *layers] if layers else []), det_skip])


def _action_cam(
    style: Style | AutoRequest,
    videos: list[Path],
    footage: category.FootageVision | None = None,
) -> bool:
    """Whether the analysis will use an action-cam style (``cutpoint events`` then runs its
    layers); for ``--style auto`` the sources are probed (:func:`cutpoint.category.detect`,
    cached for the analysis; ``footage``: the run's vision client), talk aside: it never makes
    the style an action-cam one."""
    if isinstance(style, AutoRequest):
        _probes, det = category.detect(videos, False, shared=footage)  # None: vision is off
        name, _fallback = category.choose_style(det, style.emphasis)
        style = get_style(name)
    return style.kind == ACTION_CAM


def _link_or_copy(src: Path, dst: Path) -> None:
    """Make ``dst`` a hard link to ``src`` (a copy where links are not possible), atomically."""
    with folders.atomic_output(dst) as part:
        part.unlink(missing_ok=True)  # left by a hard-killed run
        try:
            os.link(src, part)
        except OSError:  # FAT/exFAT, some network shares
            shutil.copy2(src, part)


def run_edit(
    inputs: list[Path],
    project: Path,
    kind: Kind,
    *,
    vertical: bool = False,
    force: bool = False,
    events: bool = False,
    vision: bool | None = None,
    vision_top: int | None = None,
    ollama_url: str | None = None,
    ollama_model: str | None = None,
    skip: Collection[str] = (),
    visual_index: bool | None = None,
    ocr_det_skip: bool | None = None,
    compress_peaks: str = "off",
    lut: Lut | None = None,
    color: str | None = None,
    color_strength: float = 1.0,
    keep_clips: bool = True,
    output_format: str = "mp4",
) -> Path:
    """Analyze ``inputs`` into ``project``, export clips and assemble them as ``kind`` says.

    Analysis is cached as with ``cutpoint analyze``. ``force`` has ``clips --force``
    semantics. ``events`` runs ``cutpoint events`` (default options) before the clips; its
    VLM layer reports as the ``keyframes`` step again. ``vision`` is ``analyze --vision``
    (opt-in: None or False rates nothing with Ollama); ``vision_top``, ``ollama_url``,
    ``ollama_model`` and ``skip`` are analyze's options of those names (the Ollama ones also
    for the events VLM layer; ``skip`` also leaves its steps' layers out of ``events``,
    :data:`~cutpoint.analyze.LAYER_STEP`); ``visual_index`` is ``analyze --visual-index``
    (opt-in: None or False writes the frame embeddings only for the styles that always do);
    ``ocr_det_skip`` is ``analyze --ocr-det-skip`` (None or False: off), also for ``events``.
    ``compress_peaks``
    (``clips``/``assemble --compress-peaks``) is baked into the clips, so the final keeps it
    without compressing twice. ``lut`` (:func:`~cutpoint.encode.check_lut`) grades the clips
    as ``clips --lut`` does, ``color`` and ``color_strength`` as ``clips --color`` and
    ``--color-strength`` (None: config.json's ``color``); the final keeps the grade.

    ``keep_clips`` False exports the clips into a temporary folder in ``project``, removed
    after the join (also on a failure or Ctrl+C), and leaves the clips folder alone.
    ``output_format`` is ``assemble --format``: ``fcpxml`` or ``otio`` writes that timeline
    instead of the final video; it references the clips, so they must be kept.

    Returns:
        ``<project>/<edit_name(first input)>-final.mp4`` (``-final_vertical.mp4`` with
        ``vertical``; ``.fcpxml`` / ``.otio`` for a timeline format), a hard link or copy of
        what was assembled.

    Raises:
        FileExistsError: If the clips are kept, the clips folder is not empty and ``force``
            is False (before any analysis).
        ValueError: If the project folder is a source video's folder, or would get a
            copy over a source video; if a timeline format is asked for without
            ``keep_clips``.
        ModuleNotFoundError: With ``events``, if the events extra is missing and a layer
            other than vlm is left after ``skip`` (before any analysis).
    """
    with progress.run(progress.EDIT_STEPS) as steps:
        return _run_edit(
            inputs,
            project,
            kind,
            steps,
            vertical=vertical,
            force=force,
            events=events,
            vision=vision,
            vision_top=vision_top,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            skip=skip,
            visual_index=visual_index,
            ocr_det_skip=ocr_det_skip,
            compress_peaks=compress_peaks,
            lut=lut,
            color=color,
            color_strength=color_strength,
            keep_clips=keep_clips,
            output_format=output_format,
        )


def _run_edit(
    inputs: list[Path],
    project: Path,
    kind: Kind,
    steps: progress.Run,
    *,
    vertical: bool,
    force: bool,
    events: bool,
    vision: bool | None,
    vision_top: int | None,
    ollama_url: str | None,
    ollama_model: str | None,
    skip: Collection[str],
    visual_index: bool | None,
    ocr_det_skip: bool | None,
    compress_peaks: str,
    lut: Lut | None,
    color: str | None,
    color_strength: float,
    keep_clips: bool,
    output_format: str,
) -> Path:
    """:func:`run_edit`, reporting to ``steps``."""
    from cutpoint.analyze import analyze, report_highlights  # noqa: PLC0415

    steps.start("transcribe")  # the checks below fail the first step
    if output_format != "mp4" and not keep_clips:
        raise jsonout.ArgumentError(
            f"--format {output_format} references the clips; it cannot go with --no-keep-clips"
        )
    videos = find_videos(inputs)
    if any(folders.file_keys(v.parent) & folders.file_keys(project) for v in videos):
        raise ValueError(
            f"{project} is the folder of the source videos; give --out a project name "
            "so nothing is written next to them"
        )
    clips_dir = project / ("clips_vertical" if vertical else "clips")
    final_name = "final_vertical.mp4" if vertical else "final.mp4"
    named = (project / f"{edit_name(inputs[0])}-{final_name}").with_suffix(f".{output_format}")
    named_srt = named.with_suffix(".srt")
    keys = set().union(*(folders.file_keys(v) for v in videos))
    folders.check_not_sources(project, [named.name, named_srt.name], keys)
    # The checks clips makes, before hours of analysis rather than after.
    # cutpoint resume of a run that finished its clips: they are kept (runstate.clips_kept)
    kept = keep_clips and runstate.clips_kept(project, clips_dir)
    if keep_clips and clips_dir.exists() and any(clips_dir.iterdir()) and not (force or kept):
        raise FileExistsError(f"{clips_dir} is not empty; use --force to replace its clips")
    if kind.auto:
        style: Style | AutoRequest = AutoRequest(kind.emphasis)
    else:
        style = get_style(kind.style)
    event_layers = None  # the events defaults
    # one client of the vision model for both detections (here and in analyze): a failed
    # request is not tried again; None with --no-vision / "vision": false
    with category.footage_vision(vision, ollama_url, ollama_model) as footage:
        if events:
            from cutpoint.analyze import LAYER_STEP  # noqa: PLC0415
            from cutpoint.events.pipeline import DEFAULT_LAYERS, SKI_LAYERS, preflight  # noqa: PLC0415

            ski = _action_cam(style, videos, footage)
            defaults = SKI_LAYERS if ski else DEFAULT_LAYERS
            # --skip leaves out the layers of its steps here too (keyframes: no HUD discovery).
            kept = tuple(x for x in defaults if LAYER_STEP.get(x) not in skip)
            event_layers = kept if kept != defaults else None
            preflight(kept)

        # A failed or interrupted analyze removes the (still empty) folders it created.
        analyze(
            inputs,
            style,
            project,
            vision=vision,
            vision_top=vision_top,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            layers=True,
            skip=skip,
            visual_index=visual_index,
            ocr_det_skip=bool(ocr_det_skip),
            footage=footage,
        )
    if events:
        from cutpoint.events.cli import run_args  # noqa: PLC0415

        with steps.step("keyframes", detail="describing moments with the VLM"):
            run_args(
                _events_namespace(project, event_layers, bool(ocr_det_skip)),
                ollama_url=ollama_url,
                ollama_model=ollama_model,
            )
        report_highlights(project, project / "highlights.json")  # events re-ranked it
    srt = (project / final_name).with_suffix(".srt")
    before = file_stamp(srt)  # an earlier run's final.srt is not this run's subtitles
    with temporary_clips(project) if not keep_clips else nullcontext(clips_dir) as work:
        temporary = {} if keep_clips else {"out_dir": work, "temporary": True}
        with steps.step("clips", "kept from the interrupted run" if kept else None):
            if kept:
                logger.info("%s: the clips of the interrupted run are kept", clips_dir)
                report_clips(clips_dir)
            else:
                export_clips(
                    project,
                    top=kind.top,
                    max_total=kind.max_total,
                    order=kind.order,
                    vertical=vertical,
                    force=force,
                    compress_peaks=compress_peaks,
                    lut=lut,
                    color=color,
                    color_strength=color_strength,
                    **temporary,
                )
        if keep_clips:
            runstate.clips_written(project, clips_dir)
        steps.start("assemble")
        final = assemble(
            work,
            None if keep_clips else project / final_name,
            transition=kind.transition,
            loudnorm=kind.loudnorm,
            captions=kind.captions,
            compress_peaks=compress_peaks,
            output_format=output_format,
        )
    _link_or_copy(final, named)
    written = (now := file_stamp(srt)) is not None and now != before
    if written:
        cache.save_bytes(named_srt, srt.read_bytes())
    elif named_srt.exists():
        logger.warning("%s is from an earlier run and does not match %s", named_srt, named.name)
    logger.info("%s %s", "Final video:" if output_format == "mp4" else "Timeline:   ", named)
    logger.info("Subtitles:   %s", named_srt if written else "none (no transcript)")
    logger.info("Clips:       %s", clips_dir if keep_clips else "none (temporary, removed)")
    # analyze, export_clips and assemble added their outputs and warnings to the same result.
    jsonout.output("copy", named)
    jsonout.output("copy_srt", named_srt if written else None)
    return named
