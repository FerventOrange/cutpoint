"""``cutpoint edit``: one command from source videos to a finished edit.

Runs ``analyze`` -> (``events``) -> ``clips`` -> ``assemble`` in-process with the
options of a :data:`KINDS` entry, then copies the final video and its subtitles
to ``<name>-final.mp4`` / ``.srt`` in the project folder (``<name>``: see
:func:`edit_name`). With ``CUTPOINT_PROGRESS=1`` it reports the steps of
:data:`cutpoint.progress.EDIT_STEPS` (analyze's, then clips and assemble).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from collections.abc import Collection
from dataclasses import dataclass, replace
from pathlib import Path

from cutpoint import cache, jsonout, progress
from cutpoint.assemble import assemble
from cutpoint.clips import export_clips
from cutpoint.encode import Lut
from cutpoint.ffmpeg_utils import atomic_output, check_not_sources, file_keys
from cutpoint.styles import get_style
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
    """

    style: str
    top: int | None = None
    max_total: float | None = None
    order: str = "score"
    transition: str = "cut"
    loudnorm: bool = True
    captions: bool = False


KINDS: dict[str, Kind] = {
    "highlights": Kind("action", max_total=180, order="chronological", transition="fade"),
    "montage": Kind("action", max_total=60, order="score", transition="fade"),
    "clips": Kind("action", top=10, order="score", transition="cut"),
    "funny-moments": Kind("funny", top=8, order="chronological", transition="cut", captions=True),
}


def resolve_kind(
    kind: str,
    *,
    style: str | None = None,
    top: int | None = None,
    max_total: float | None = None,
) -> Kind:
    """The :data:`KINDS` entry ``kind`` with the options the user gave replacing its own."""
    given = {"style": style, "top": top, "max_total": max_total}
    return replace(KINDS[kind], **{k: v for k, v in given.items() if v is not None})


def edit_name(target: Path) -> str:
    """Name of an edit of ``target`` (project folder, ``<name>-final.mp4``): a file's stem,
    a folder's full name (``.`` and a drive root resolved), else ``edit``."""
    return input_name(target, "edit")


def _events_namespace(project: Path) -> argparse.Namespace:
    """``cutpoint events <project>`` arguments with every default, as the CLI parses them."""
    from cutpoint.events.cli import add_arguments  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args([str(project)])


def _link_or_copy(src: Path, dst: Path) -> None:
    """Make ``dst`` a hard link to ``src`` (a copy where links are not possible), atomically."""
    with atomic_output(dst) as part:
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
    compress_peaks: str = "off",
    lut: Lut | None = None,
    color: str | None = None,
    color_strength: float = 1.0,
) -> Path:
    """Analyze ``inputs`` into ``project``, export clips and assemble them as ``kind`` says.

    Analysis is cached as with ``cutpoint analyze``. ``force`` has ``clips --force``
    semantics. ``events`` runs ``cutpoint events`` (default options) before the clips; its
    VLM layer reports as the ``keyframes`` step again. ``vision`` is ``analyze --vision``
    (opt-in: None or False rates nothing with Ollama); ``vision_top``, ``ollama_url``,
    ``ollama_model`` and ``skip`` are analyze's options of those names. ``compress_peaks``
    (``clips``/``assemble --compress-peaks``) is baked into the clips, so the final keeps it
    without compressing twice. ``lut`` (:func:`~cutpoint.encode.check_lut`) grades the clips
    as ``clips --lut`` does, ``color`` and ``color_strength`` as ``clips --color`` and
    ``--color-strength`` (None: config.json's ``color``); the final keeps the grade.

    Returns:
        ``<project>/<edit_name(first input)>-final.mp4`` (``-final_vertical.mp4`` with
        ``vertical``), a hard link or copy of the assembled video.

    Raises:
        FileExistsError: If the clips folder is not empty and ``force`` is False (before
            any analysis).
        ValueError: If the project folder is a source video's folder, or would get a
            copy over a source video.
        ModuleNotFoundError: With ``events``, if the events extra is missing (before any
            analysis).
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
            compress_peaks=compress_peaks,
            lut=lut,
            color=color,
            color_strength=color_strength,
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
    compress_peaks: str,
    lut: Lut | None,
    color: str | None,
    color_strength: float,
) -> Path:
    """:func:`run_edit`, reporting to ``steps``."""
    from cutpoint.analyze import analyze, report_highlights  # noqa: PLC0415

    steps.start("transcribe")  # the checks below fail the first step
    videos = find_videos(inputs)
    if any(file_keys(v.parent) & file_keys(project) for v in videos):
        raise ValueError(
            f"{project} is the folder of the source videos; give --out a project name "
            "so nothing is written next to them"
        )
    clips_dir = project / ("clips_vertical" if vertical else "clips")
    final_name = "final_vertical.mp4" if vertical else "final.mp4"
    named = project / f"{edit_name(inputs[0])}-{final_name}"
    named_srt = named.with_suffix(".srt")
    keys = set().union(*(file_keys(v) for v in videos))
    check_not_sources(project, [named.name, named_srt.name], keys)
    # The checks clips makes, before hours of analysis rather than after.
    if clips_dir.exists() and any(clips_dir.iterdir()) and not force:
        raise FileExistsError(f"{clips_dir} is not empty; use --force to replace its clips")
    style = get_style(kind.style)
    if events:
        from cutpoint.events.pipeline import DEFAULT_LAYERS, SKI_LAYERS, preflight  # noqa: PLC0415

        preflight(SKI_LAYERS if style.kind == "ski" else DEFAULT_LAYERS)

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
    )
    if events:
        from cutpoint.events.cli import run_args  # noqa: PLC0415

        with steps.step("keyframes", detail="describing moments with the VLM"):
            run_args(_events_namespace(project))
        report_highlights(project, project / "highlights.json")  # events re-ranked it
    with steps.step("clips"):
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
        )
    steps.start("assemble")
    srt = (project / final_name).with_suffix(".srt")
    before = file_stamp(srt)  # an earlier run's final.srt is not this run's subtitles
    final = assemble(
        clips_dir,
        transition=kind.transition,
        loudnorm=kind.loudnorm,
        captions=kind.captions,
        compress_peaks=compress_peaks,
    )
    _link_or_copy(final, named)
    written = (now := file_stamp(srt)) is not None and now != before
    if written:
        cache.save_bytes(named_srt, srt.read_bytes())
    elif named_srt.exists():
        logger.warning("%s is from an earlier run and does not match %s", named_srt, named.name)
    logger.info("Final video: %s", named)
    logger.info("Subtitles:   %s", named_srt if written else "none (no transcript)")
    logger.info("Clips:       %s", clips_dir)
    # analyze, export_clips and assemble added their outputs and warnings to the same result.
    jsonout.output("copy", named)
    jsonout.output("copy_srt", named_srt if written else None)
    return named
