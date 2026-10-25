"""``cutpoint edit``: one command from source videos to a finished edit.

Runs ``analyze`` -> (``events``) -> ``clips`` -> ``assemble`` in-process with the
options of a :data:`KINDS` entry, then copies the final video and its subtitles
to ``<name>-final.mp4`` / ``.srt`` in the project folder (``<name>``: see
:func:`edit_name`).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from cutpoint import cache, jsonout
from cutpoint.assemble import assemble
from cutpoint.clips import export_clips
from cutpoint.ffmpeg_utils import atomic_output, check_not_sources, file_keys
from cutpoint.styles import get_style
from cutpoint.utils import find_videos, input_name

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


def _identity(path: Path) -> tuple[int, int] | None:
    """``(inode, mtime ns)`` of ``path``, None if missing: did a step rewrite it?"""
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_ino, st.st_mtime_ns


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
) -> Path:
    """Analyze ``inputs`` into ``project``, export clips and assemble them as ``kind`` says.

    Analysis is cached as with ``cutpoint analyze``. ``force`` has ``clips --force``
    semantics. ``events`` runs ``cutpoint events`` (default options) before the clips.
    ``vision`` is ``analyze --vision`` (None: the style's default, else if Ollama is reachable);
    ``vision_top``, ``ollama_url`` and ``ollama_model`` are analyze's options of those names.

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
    from cutpoint.analyze import analyze, report_highlights  # noqa: PLC0415

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
    )
    if events:
        from cutpoint.events.cli import run_args  # noqa: PLC0415

        run_args(_events_namespace(project))
        report_highlights(project, project / "highlights.json")  # events re-ranked it
    export_clips(
        project,
        top=kind.top,
        max_total=kind.max_total,
        order=kind.order,
        vertical=vertical,
        force=force,
    )
    srt = (project / final_name).with_suffix(".srt")
    before = _identity(srt)  # an earlier run's final.srt is not this run's subtitles
    final = assemble(
        clips_dir,
        transition=kind.transition,
        loudnorm=kind.loudnorm,
        captions=kind.captions,
    )
    _link_or_copy(final, named)
    written = (now := _identity(srt)) is not None and now != before
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
