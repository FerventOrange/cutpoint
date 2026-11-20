"""Clip padding: ``cutpoint clips --extend`` and ``cutpoint extend <clips_dir>``.

Padding gives an editor room to trim in DaVinci Resolve. It is applied last,
after smart tails and event tails, to the final in/out points:

* ``start - pre`` and ``end + post``, clamped to 0 and the source duration;
* never into an unreadable region: a clip in a part recovered by
  ``analyze --recover`` stays inside that part, and the end stays before the
  part's readable end (the recorded read-error / bad-sector boundary);
* then the speech-aware pass (:mod:`cutpoint.boundaries`, on by
  default): an in/out point that lands mid-utterance moves to the utterance
  start/end, and clips of the same source are kept apart (overlapping padded
  ranges are split in a pause between the picks, so no footage is shown
  twice). With ``--no-speech-aware`` padded clips may overlap, as before;
* every re-exported clip gets a short audio fade in/out (``--audio-fade``) and its
  ``.srt`` subtitles for the new range (see :mod:`cutpoint.srt`).

``cutpoint extend`` re-exports an existing clips folder from its ``manifest.json``
(source, in and out points, crop, ``--lut`` or ``--color`` grade, format), keeping the exact file names and
order of the clips *still in the folder*, so renames and deletions made
during review are respected. Per-clip speech-aware caps come from the
manifest entry (``speech_max_extend`` / ``speech_max_lead``, set by a
segment) or ``--clip-override``, which can also set one clip's total padding
(``extend`` / ``extend_start`` / ``extend_end``); ``--reuse DIR`` copies clips whose result is
identical to one in an earlier output instead of re-encoding them. An exact clip (manifest
``exact``, a word splice of the AI Chat) keeps its edges and its short fade
(:func:`cutpoint.planning.exact_fade`).
"""

from __future__ import annotations

import fnmatch
import logging
import os
import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cutpoint import boundaries, cache, folders, jsonout, runstate, srt, tsrepair
from cutpoint.clipresume import ClipResume, clip_key
from cutpoint.color import COLOR_VERSION, AutoGrade
from cutpoint.config import path_key, remap_source, source_path
from cutpoint.crop import Crop, parse_spec
from cutpoint.encode import (
    Grade,
    Lut,
    OutputFormat,
    check_lut,
    export_clip_checked,
    grade_from_entry,
)
from cutpoint.ffmpeg_utils import MediaInfo, probe
from cutpoint.planning import (
    MANIFEST_GENERATOR,
    Padding,
    ReplacePlan,
    exact_fade,
    file_size,
    manifest_entries,
    manifest_parts,
    manifest_project,
    pad,
    replace_plan,
    report_clips,
    speech_pass,
    warn_kept,
    warn_pano_sources,
)
from cutpoint.sources import project_language
from cutpoint.utils import clip_files, seconds_keys

if TYPE_CHECKING:
    from cutpoint.pano.render import PanoExporter

logger = logging.getLogger(__name__)


@dataclass
class ExtendPlan:
    """One clip of the folder to re-export (see :func:`extend_clips`).

    Attributes:
        name: Clip file name (kept in the output).
        entry: Its manifest entry (plus any ``--clip-override`` speech caps).
        src: Source file actually read.
        info: Probe of ``src``.
        start: New in point (padded, then speech-adjusted).
        end: New out point.
        rec: Padding record (manifest ``extend``).
        keep: The picked moment of the entry (before any padding).
    """

    name: str
    entry: dict[str, Any]
    src: Path
    info: MediaInfo
    start: float
    end: float
    rec: dict[str, Any]
    keep: tuple[float, float]


def _seconds_tag(x: float) -> str:
    return f"{x:g}".replace(".", "p") if x != int(x) else str(int(x))


def default_out_dir(clips_dir: Path, padding: Padding) -> Path:
    """Sibling folder ``<clips_dir>_ext<S>s`` (or ``_ext<A>-<B>s`` if asymmetric)."""
    if padding.start == padding.end:
        tag = _seconds_tag(padding.start)
    else:
        tag = f"{_seconds_tag(padding.start)}-{_seconds_tag(padding.end)}"
    return clips_dir.with_name(f"{clips_dir.name}_ext{tag}s")


def _foreign_project(project: Path, clips_dir: Path) -> bool:
    """Whether ``project`` (:func:`manifest_project`) is not the clips folder's parent: a
    clips folder exported elsewhere (``clips --out``) or copied from another machine, whose
    manifest names a project this run must not write into (A663)."""
    parent = clips_dir.resolve().parent
    return path_key(str(project.resolve())) != path_key(str(parent))


def _pick(e: dict[str, Any]) -> tuple[float, float]:
    """The picked moment of a manifest entry (before any padding)."""
    return float(e["pick"]["start"]), float(e["pick"]["end"])


PEAK_KEYS = ("compress_peaks", "compress_peaks_makeup_db")
"""Manifest entry keys of ``clips --compress-peaks``; extend re-exports without them."""

PAD_KEYS = ("extend", "extend_start", "extend_end")
"""``--clip-override`` keys that set a clip's total padding (s) per side."""


def apply_overrides(
    plans: list[ExtendPlan],
    overrides: dict[str, dict[str, Any]] | None,
    padding: Padding | None = None,
) -> None:
    """Merge ``--clip-override`` keys into the planned re-exports.

    Speech caps (``speech_max_extend`` / ``speech_max_lead``) are merged into
    the manifest entry. Padding keys (``extend`` / ``extend_start`` /
    ``extend_end``: that clip's total padding per side, replacing
    ``--seconds`` / ``--start`` / ``--end``) re-pad the clip from its
    recorded in/out points; the keys are kept in the padding record
    (``extend.override``).

    Args:
        plans: Planned re-exports (``plan.entry`` is replaced by an updated
            copy; ``plan.start`` / ``end`` / ``rec`` are recomputed for
            padding keys).
        overrides: ``{file name or glob: {"speech_max_extend": S, ...}}``,
            matched case-insensitively against the clip file names.
        padding: The run's padding, the base for padding keys (default: none).

    Raises:
        ValueError: If a pattern matches no clip, or a key or value is invalid.
    """
    allowed = (*boundaries.LIMIT_KEYS, *PAD_KEYS)
    for pattern, keys in (overrides or {}).items():
        unknown = set(keys) - set(allowed)
        if unknown:
            raise ValueError(
                f"--clip-override {pattern}: unknown key(s) {sorted(unknown)}; "
                f"use {', '.join(allowed)}"
            )
        where = f"--clip-override {pattern}"
        limits = boundaries.clip_limits(keys, where)
        pads = seconds_keys(keys, PAD_KEYS, where)
        hits = [p for p in plans if fnmatch.fnmatch(p.name.lower(), pattern.lower())]
        if not hits:
            raise ValueError(f"--clip-override {pattern}: matches no clip in the folder")
        for plan in hits:
            plan.entry = plan.entry | limits
            if pads and plan.entry.get("exact") is True:
                logger.info("  %s: an exact clip (word splice) keeps its edges", plan.name)
            elif pads:
                e = plan.entry
                own = (padding or Padding()).override(pads)
                start, end, rec = pad(
                    plan.src, plan.info, float(e["start"]), float(e["end"]), own, e
                )
                plan.start, plan.end = start, end
                plan.rec = rec | {"override": (plan.rec.get("override") or {}) | pads}
            logger.info(
                "  %s: %s",
                plan.name,
                ", ".join(f"{k} {v:g}s" for k, v in (limits | pads).items()),
            )


def _fixed_tail(e: dict[str, Any]) -> bool:
    """Whether the entry's end was set by a fixed (segment) tail."""
    return str((e.get("tail") or {}).get("reason", "")).startswith("fixed tail")


def _hi_clamp(plan: ExtendPlan, hi: float) -> float:
    """Speech pass bound: stay before a recorded read error, never cut the clip."""
    if plan.entry.get("read_error"):  # a re-read may fail again: stay before the bad spot
        hi = min(hi, float(plan.entry["read_error"]["end"]))
    return max(hi, plan.end)


def _read_path(e: dict[str, Any]) -> Path:
    """File to read for a manifest entry: its source through ``source_remap``, like
    ``clips``; the recorded ``read_from`` only when that is missing and it still exists.
    An unreadable path (stale mount, EIO, EACCES) counts as missing."""
    src = remap_source(e["source"])
    old = e.get("read_from")
    # os.path.exists: any OSError counts as missing; probe then reports the remap result.
    if old and not os.path.exists(src) and os.path.exists(old):  # noqa: PTH110
        return Path(old)
    return src


def _recorded_grade(name: str, entry: dict[str, Any]) -> Grade | None:
    """The grade manifest ``entry`` of clip ``name`` records (:func:`grade_from_entry`).

    Raises:
        ValueError: If its ``color`` record cannot be read, or a newer Cutpoint wrote it
            (:data:`~cutpoint.color.COLOR_VERSION`): the clip would be re-exported ungraded.
    """
    color = entry.get("color")
    try:
        grade = grade_from_entry(entry)
        version = int(color.get("version", COLOR_VERSION)) if isinstance(color, dict) else 0
    except (TypeError, ValueError, KeyError, IndexError) as err:
        raise ValueError(f"{name}: manifest.json has an unreadable grade ({err})") from None
    if color and not isinstance(grade, AutoGrade) and not entry.get("lut"):
        raise ValueError(f"{name}: manifest.json has a color record this Cutpoint cannot read")
    if isinstance(grade, AutoGrade) and version > COLOR_VERSION:
        raise ValueError(
            f"{name}: its color grade is version {version}, newer than this Cutpoint's "
            f"{COLOR_VERSION}; update Cutpoint"
        )
    return grade


def _crop_box(entry: dict[str, Any]) -> Crop | None:
    """The crop box manifest ``entry`` records (None: no crop)."""
    box = parse_spec(entry.get("crop"))
    return box if isinstance(box, Crop) else None


def _reusable(
    old: dict[str, Any] | None, new: dict[str, Any], same_format: bool, audio_fade: float
) -> bool:
    """Whether an earlier output's entry renders exactly like the new one."""
    if (
        old is None
        or not same_format
        or old.get("read_error")
        or old.get("compress_peaks")
        or old.get("pano")
        or new.get("pano")
    ):
        return False
    return (
        old.get("source") == new.get("source")
        and abs(float(old.get("start", -1)) - new["start"]) < 1e-3
        and abs(float(old.get("end", -1)) - new["end"]) < 1e-3
        and old.get("crop") == new.get("crop")
        and grade_from_entry(old) == grade_from_entry(new)
        and float(old.get("audio_fade", -1)) == float(audio_fade)
    )


def extend_clips(
    clips_dir: Path,
    padding: Padding,
    *,
    out_dir: Path | None = None,
    force: bool = False,
    speech: boundaries.SpeechSettings | None = None,
    audio_fade: float = 0.25,
    pano_source: str | None = None,
    pano_cmd_every: int | None = None,
    clip_overrides: dict[str, dict[str, Any]] | None = None,
    reuse_dir: Path | None = None,
) -> Path:
    """Re-export the clips in ``clips_dir`` with extra padding (``cutpoint extend``).

    Args:
        clips_dir: Folder with clips and the ``manifest.json`` written by
            ``cutpoint clips``.
        padding: Seconds to add before and after each clip.
        out_dir: Output folder (default :func:`default_out_dir`, or
            ``<clips_dir>_speech`` without padding).
        force: Allow writing into a non-empty folder, including ``clips_dir``. Like
            ``clips --force``, only the clips and ``.srt`` files its manifest.json lists, and
            leftover ``.part.mp4`` files of an interrupted run, are replaced; listed ones this
            run does not write are removed.
        speech: Speech-aware boundary settings (default: on, 8 s / 5 s caps);
            entries with ``"speech_aware": false`` are left alone, and an
            entry's ``speech_max_extend`` / ``speech_max_lead`` replace the
            caps for that clip.
        audio_fade: Audio fade in/out baked into each clip (s, 0 = off).
        pano_source, pano_cmd_every: For 360 clips (manifest ``pano`` record):
            render source mode (default: the one recorded per clip, or
            ``auto`` when that is unavailable or not recorded) and camera update
            interval (default ``pano.cmd_every`` or 1, not the recorded one: older
            runs recorded 2, which leaves shake on every other frame). Their
            director path files are reused
            when they cover the new range (hand edits are kept), else re-planned.
        clip_overrides: ``{file name or glob: {"speech_max_extend": S,
            "speech_max_lead": S, "extend_start": S, "extend_end": S}}``
            per-clip speech caps, merged over the manifest entries, and
            per-clip total padding (``extend`` both sides, ``extend_start`` /
            ``extend_end`` per side) replacing ``padding`` for those clips
            (all recorded in the new manifest).
        reuse_dir: An earlier output folder. A clip whose new source, in/out
            points, crop, grade, format and fade equal its entry there is copied
            instead of re-encoded (never for 360 clips).

    Returns:
        The output folder.

    Raises:
        FileNotFoundError: If ``clips_dir`` or its manifest is missing.
        FileExistsError: If ``out_dir`` is non-empty (or is ``clips_dir``)
            and ``force`` is False, or a clip to re-export has the name of an ``.mp4``
            that ``out_dir``'s manifest.json does not list (before any export).
        ValueError: If nothing could be re-exported.
        FolderBusyError: If another run is using ``clips_dir`` or ``out_dir``.
    """
    speech = speech or boundaries.SpeechSettings()
    with ExitStack() as held:
        manifest, out_dir, in_place = _claim(held, clips_dir, out_dir, padding, speech, force)
        # As clips does: the clips are exported into a hidden staging folder and swapped in
        # only when all succeeded, so a failed or cancelled run leaves the folder as it was (a
        # new one is not made) and keeps its finished clips for the next run
        # (cutpoint.clipresume). staged_dir refuses a drive root, a mount point or a bind mount
        # before any export; only an empty one is exported into directly. A run that fails
        # leaves no empty folders behind.
        with (
            folders.output_dir(out_dir.parent),
            folders.staged_dir(out_dir, direct_if_empty=True, resume=True) as work,
        ):
            job = _plan_extend(clips_dir, manifest, padding, in_place, clip_overrides)
            grades = _recorded_grades(job.plans)
            reuse = _reuse_entries(reuse_dir, out_dir, work, job.fmt_d)
            language = project_language(job.project)
            speech_recs = speech_pass(
                job.plans,
                speech,
                job.project,
                fit_end=lambda e: not _fixed_tail(e),
                hi_clamp=_hi_clamp,
                language=language,
            )
            pano = _pano_exporter(job, clips_dir, out_dir, pano_source, pano_cmd_every)
            _warn_dropped_peaks(job.plans, clips_dir)
            replaced = _replace_set(job, out_dir, pano, pano_source)
            entries, reused = _render(
                job,
                work,
                out_dir,
                grades=grades,
                replaced=replaced,
                language=language,
                speech_recs=speech_recs,
                pano=pano,
                pano_source=pano_source,
                audio_fade=audio_fade,
                reuse=reuse,
            )
            if not entries:
                raise ValueError(f"No clips with manifest entries in {clips_dir}")
            _write_extend_manifest(
                work,
                manifest,
                clips_dir,
                entries,
                plans=job.plans,
                padding=padding,
                speech=speech,
                audio_fade=audio_fade,
                skipped=job.skipped,
                reuse=reuse,
                reused=reused,
                clip_overrides=clip_overrides,
            )
            if work != out_dir:  # staged: all or nothing; only the files named are replaced
                folders.swap_in(work, out_dir, replaced.replace)
        warn_kept(out_dir, replaced, "a source of these clips is never replaced;")
        skipped = job.skipped if not in_place else {}  # in place they stay: Kept names them
        _report_extend(clips_dir, out_dir, padding, len(entries), skipped, reused, reuse_dir)
        return out_dir


def _report_extend(
    clips_dir: Path,
    out_dir: Path,
    padding: Padding,
    count: int,
    skipped: dict[str, str],
    reused: int,
    reuse_dir: Path | None,
) -> None:
    """Report a finished :func:`extend_clips` run: ``count`` clips re-exported from
    ``clips_dir`` into ``out_dir`` with ``padding``, ``reused`` of them copied from
    ``reuse_dir``, and the clips ``skipped`` (``{file name: why}``, a ``skipped_clips``
    ``--json`` warning)."""
    if skipped:
        jsonout.warn(
            "skipped_clips",
            f"Not re-exported from {clips_dir.absolute()}: "
            + "; ".join(f"{n}: {why}" for n, why in skipped.items()),
            [clips_dir / n for n in skipped],
        )
    logger.info(
        "Re-exported %d clip(s) with +%gs / +%gs padding to %s%s%s",
        count,
        padding.start,
        padding.end,
        out_dir,
        f" ({reused} copied unchanged from {reuse_dir})" if reuse_dir else "",
        f"; skipped {len(skipped)} without a manifest entry" if skipped else "",
    )
    report_clips(out_dir)
    jsonout.output("seconds", {"start": padding.start, "end": padding.end})


@dataclass
class _ExtendJob:
    """What :func:`_plan_extend` read from a clips folder for :func:`extend_clips`.

    Attributes:
        fmt_d: The manifest's ``format`` record.
        fmt: Its output format.
        vertical: Whether the clips are vertical.
        plans: The clips to re-export, in folder order.
        skipped: ``{file name: why it has no plan}``.
        all_sources: :func:`~cutpoint.folders.source_keys` of every file the manifest's
            clips were cut from.
        project: The manifest's project (:func:`manifest_project`).
        foreign: Whether ``project`` is not the folder's parent (:func:`_foreign_project`).
    """

    fmt_d: dict[str, Any]
    fmt: OutputFormat
    vertical: bool
    plans: list[ExtendPlan]
    skipped: dict[str, str]
    all_sources: set[Any]
    project: Path
    foreign: bool


def _claim(
    held: ExitStack,
    clips_dir: Path,
    out_dir: Path | None,
    padding: Padding,
    speech: boundaries.SpeechSettings,
    force: bool,
) -> tuple[dict[str, Any], Path, bool]:
    """Take the folders' locks (:func:`~cutpoint.folders.swap_lock`) into ``held`` and check
    the run of :func:`extend_clips` before any export.

    Returns:
        ``(manifest of clips_dir, output folder, whether it is clips_dir)``.
    """
    # One run at a time on each folder: another would remove this one's staging folder.
    held.enter_context(folders.swap_lock(clips_dir))
    folders.recover_swap(clips_dir)  # a hard-killed earlier replace, before the clips are read
    manifest = cache.load_json(clips_dir / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(
            f"No manifest.json in {clips_dir}; it must come from `cutpoint clips`"
        )
    if not padding.active and not speech.enabled:
        raise jsonout.ArgumentError(
            "Nothing to do: give --seconds (or --start / --end) > 0, or leave --speech-aware on"
        )
    out_dir = out_dir or (
        default_out_dir(clips_dir, padding)
        if padding.active
        else clips_dir.with_name(f"{clips_dir.name}_speech")
    )
    in_place = out_dir.resolve() == clips_dir.resolve()
    if not in_place:
        held.enter_context(folders.swap_lock(out_dir))
        # a hard-killed earlier replace, before the folder is looked at
        folders.recover_swap(out_dir)
    non_empty = out_dir.exists() and any(out_dir.iterdir())
    if (in_place or non_empty) and not force:
        raise FileExistsError(
            f"{out_dir} {'is the input folder' if in_place else 'is not empty'}; "
            f"{runstate.force_hint()}"
        )
    return manifest, out_dir, in_place


def _plan_extend(
    clips_dir: Path,
    manifest: dict[str, Any],
    padding: Padding,
    in_place: bool,
    clip_overrides: dict[str, dict[str, Any]] | None,
) -> _ExtendJob:
    """Plan the re-export of the clips in ``clips_dir`` (padded, ``clip_overrides`` applied;
    the speech pass comes later); a file without a manifest entry is skipped, with a WARNING
    unless ``in_place``."""
    fmt_d, entries = manifest_parts(
        manifest, need_format=True, need_range=True, path=clips_dir / "manifest.json"
    )
    by_file = {e["file"]: e for e in entries}
    fmt = OutputFormat(int(fmt_d["width"]), int(fmt_d["height"]), int(fmt_d["fps"]))
    vertical = bool(fmt_d.get("vertical"))
    files = clip_files(clips_dir)
    infos: dict[str, MediaInfo] = {}
    plans: list[ExtendPlan] = []
    skipped: dict[str, str] = {}  # file name: why it has no plan
    # every file the manifest's clips were cut from, so one in the folder is named so
    all_sources = folders.source_keys(
        (e["source"], _read_path(e), []) for e in by_file.values() if e.get("source")
    )
    known = manifest_entries(clips_dir, files)  # clips renamed during review too
    project = manifest_project(manifest, clips_dir)
    foreign = _foreign_project(project, clips_dir)
    if foreign:
        logger.info(
            "%s names the project %s, not the folder's parent %s: its highlights.json and "
            "360 clip list are read there, and 360 path files are written under %s",
            clips_dir / "manifest.json",
            project,
            clips_dir.resolve().parent,
            clips_dir.resolve().parent / "pano_paths",
        )
    # joined MPEG-TS/PS recordings are read through their stream copy, the clock of the
    # analysis (#1052); one the analysis read so is never read in place
    tsrepair.prepare_sources(
        [source_path(e["source"]) for e in by_file.values() if e.get("source")],
        required=tsrepair.remuxed_sources(cache.load_json(project / "highlights.json")),
    )
    for f in files:
        e = by_file.get(f.name)
        if e is None:
            if folders.same_file(f, all_sources):
                why = "a source of these clips, not a clip"
            elif f.name in known:
                why = (
                    f"exported as {known[f.name]['file']} and renamed during review "
                    "(only clips under their manifest.json names are re-exported)"
                )
            else:
                why = "no manifest entry (added by hand?)"
            if not in_place:  # in place it stays: the Kept warning names it
                logger.warning("  %s: %s; skipped", f.name, why)
            skipped[f.name] = why
            continue
        if not isinstance(e.get("pick"), dict):
            raise ValueError(
                f"{clips_dir / 'manifest.json'}: the entry of {f.name} has no pick (clips "
                "exported by an older cutpoint); export the clips again with `cutpoint clips`"
            )
        src = _read_path(e)
        if str(src) not in infos:
            infos[str(src)] = probe(src)
        info = infos[str(src)]
        # an exact clip (a word splice) keeps its own edges: no padding
        own = Padding() if e.get("exact") is True else padding
        start, end, rec = pad(src, info, float(e["start"]), float(e["end"]), own, e)
        plans.append(ExtendPlan(f.name, e, src, info, start, end, rec, _pick(e)))

    apply_overrides(plans, clip_overrides, padding)
    return _ExtendJob(fmt_d, fmt, vertical, plans, skipped, all_sources, project, foreign)


def _recorded_grades(plans: list[ExtendPlan]) -> dict[str, Grade | None]:
    """``{clip name: grade}``: the grade each clip recorded is redone; a LUT must still load,
    a color grade be readable (before any export). A color grade keeps its recorded
    parameters: not measured again."""
    checked: dict[Lut, Lut] = {}
    grades: dict[str, Grade | None] = {}
    for plan in plans:
        grade = _recorded_grade(plan.name, plan.entry)
        if isinstance(grade, Lut):
            if grade not in checked:
                checked[grade] = check_lut(grade.path, grade.strength, f"{plan.name}: lut")
            grade = checked[grade]
        grades[plan.name] = grade
    recolored = sum(1 for g in grades.values() if isinstance(g, AutoGrade) and g.filter())
    if recolored:
        logger.info(
            "%d clip(s) are graded with the --color parameters their manifest recorded "
            "(not measured again for the new range)",
            recolored,
        )
    return grades


@dataclass(frozen=True)
class _Reuse:
    """The ``--reuse`` folder of :func:`extend_clips` (:func:`_reuse_entries`).

    Attributes:
        dir: The earlier output folder.
        by_file: ``{file name: entry}`` of its manifest.json.
        same_format: Whether its clips have the run's format.
    """

    dir: Path
    by_file: dict[str, dict[str, Any]]
    same_format: bool


def _reuse_entries(
    reuse_dir: Path | None, out_dir: Path, work: Path, fmt_d: dict[str, Any]
) -> _Reuse | None:
    """The ``--reuse`` folder ``reuse_dir`` with its manifest's entries (None without one)."""
    if reuse_dir is None:
        return None
    if reuse_dir.resolve() in (out_dir.resolve(), work.resolve()):
        raise ValueError("--reuse must be an earlier output, not the output folder")
    reuse_manifest = cache.load_json(reuse_dir / "manifest.json")
    if reuse_manifest is None:
        raise FileNotFoundError(f"No manifest.json in --reuse folder {reuse_dir}")
    reuse_fmt, reuse_entries = manifest_parts(reuse_manifest, path=reuse_dir / "manifest.json")
    return _Reuse(reuse_dir, {x["file"]: x for x in reuse_entries}, reuse_fmt == fmt_d)


def _pano_exporter(
    job: _ExtendJob,
    clips_dir: Path,
    out_dir: Path,
    pano_source: str | None,
    pano_cmd_every: int | None,
) -> PanoExporter | None:
    """The renderer of the 360 clips among ``job.plans`` (None: there are none)."""
    if not any(p.entry.get("pano") for p in job.plans):
        return None
    from cutpoint.pano.render import PATHS_DIR, Format360, PanoExporter  # noqa: PLC0415

    pano = PanoExporter(
        job.project,
        out_dir,
        job.vertical,
        pano_source or "auto",
        pano_cmd_every,
    )
    pano.fmt = Format360(job.fmt.width, job.fmt.height, job.fmt.fps)
    if job.foreign:  # the manifest's project is read, never written (A663)
        pano.paths_dir = clips_dir.resolve().parent / PATHS_DIR / out_dir.name
    return pano


def _warn_dropped_peaks(plans: list[ExtendPlan], clips_dir: Path) -> None:
    """WARNING for the clips exported with ``--compress-peaks``: re-exported without it."""
    compressed = sum(1 for p in plans if p.entry.get("compress_peaks"))
    if compressed:
        message = (
            f"{compressed} clip(s) were exported with --compress-peaks; extend re-exports "
            "them without it (assemble --compress-peaks can apply it)"
        )
        logger.warning("%s", message)
        jsonout.warn(
            "compress_peaks_dropped",
            message,
            [clips_dir / p.name for p in plans if p.entry.get("compress_peaks")],
        )


def _replace_set(
    job: _ExtendJob, out_dir: Path, pano: PanoExporter | None, pano_source: str | None
) -> ReplacePlan:
    """What the re-export replaces in ``out_dir`` (:func:`replace_plan`).

    Like clips --force: replace only the clips and .srt files out_dir's manifest.json lists
    (and leftover .part.mp4 files), never a file this run reads. Listed ones this run does not
    write (a clip not re-exported, the .srt of a clip deleted during review) are removed, as
    the new manifest.json no longer lists them; unlisted ones are never replaced. A clip
    renamed in case only (skipped as having no entry) stays.
    """
    # Only clips with a manifest entry are written; none may be a file this run reads.
    reads = []
    for plan in job.plans:
        e = plan.entry
        mode = None if pano_source else (e.get("pano") or {}).get("mode")
        pano_files = pano.inputs(e["source"], mode) if pano and e.get("pano") else []
        reads.append((e["source"], plan.src, pano_files))
    names = [p.name for p in job.plans]
    sources = folders.source_keys(reads)
    return replace_plan(out_dir, names, sources, clip_sources=job.all_sources)


def _render(
    job: _ExtendJob,
    work: Path,
    out_dir: Path,
    *,
    grades: dict[str, Grade | None],
    replaced: ReplacePlan,
    language: str | None,
    speech_recs: dict[str, dict[str, Any]],
    pano: PanoExporter | None,
    pano_source: str | None,
    audio_fade: float,
    reuse: _Reuse | None,
) -> tuple[list[dict[str, Any]], int]:
    """Re-export ``job.plans`` into ``work`` (``out_dir`` itself, or its staging folder) with
    their ``.srt`` subtitles.

    Returns:
        ``(manifest entries, clips copied from the reuse folder)``.
    """
    subs = srt.ClipSubtitles(replaced.protected_srts, language, out_dir)
    # A staged run takes over the finished clips of an interrupted one with the same plan
    # and keeps its own when interrupted (cutpoint.clipresume); 360 clips are redone.
    resume = None
    if work != out_dir:
        resume = ClipResume(
            out_dir,
            work,
            "extend",
            {
                p.name: clip_key(
                    p.src,
                    p.start,
                    p.end,
                    job.fmt,
                    job.vertical,
                    _crop_box(p.entry),
                    exact_fade(p.entry, audio_fade),
                    "",
                    grades[p.name],
                )
                for p in job.plans
                if not (pano is not None and p.entry.get("pano"))
            },
        )
    # --json: finished clips of a stopped run this one did not encode again
    jsonout.output("taken_over", len(resume.taken) if resume is not None else 0)
    entries = []
    pano_recs: list[tuple[str, dict[str, Any] | None]] = []
    reused = 0
    for plan in job.plans:
        read_err, pano_rec, copied = _export_one(
            plan,
            work,
            job,
            grade=grades[plan.name],
            resume=resume,
            pano=pano,
            pano_source=pano_source,
            audio_fade=audio_fade,
            reuse=reuse,
        )
        reused += copied
        pano_recs.append((plan.name, pano_rec))
        end = plan.end
        if read_err and read_err["end"] < end:
            end = read_err["end"]
        sub = subs.write(work / plan.name, plan.entry["source"], plan.start, end)
        entries.append(
            _clip_entry(plan, work, end, speech_recs, audio_fade, pano_rec, read_err, sub)
        )
    subs.report(len(job.plans))
    warn_pano_sources(pano_recs, out_dir)
    return entries, reused


def _export_one(
    plan: ExtendPlan,
    work: Path,
    job: _ExtendJob,
    *,
    grade: Grade | None,
    resume: ClipResume | None,
    pano: PanoExporter | None,
    pano_source: str | None,
    audio_fade: float,
    reuse: _Reuse | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
    """Write one clip of :func:`_render` into ``work``: copied unchanged from ``reuse``,
    taken over from an interrupted run (``resume``), rendered as a 360 clip, or exported.

    Returns:
        ``(read error record or None, 360 record or None, whether it was copied)``.
    """
    name, e, src, info = plan.name, plan.entry, plan.src, plan.info
    start, end = plan.start, plan.end
    pano_rec = e.get("pano")
    if (
        reuse is not None
        and (reuse.dir / name).is_file()
        and _reusable(
            reuse.by_file.get(name),
            e | {"start": start, "end": end},
            reuse.same_format,
            exact_fade(e, audio_fade),
        )
    ):
        with folders.atomic_output(work / name) as part:
            shutil.copy2(reuse.dir / name, part)
        if resume is not None:
            resume.add(name, None)
        logger.info("  %s: unchanged, copied from %s", name, reuse.dir.name)
        return None, pano_rec, True
    if resume is not None and name in resume.taken:
        return None, pano_rec, False
    if pano is not None and pano_rec:
        old = Path(pano_rec["path_file"]) if pano_rec.get("path_file") else None
        read_err, pano_rec = pano.export(
            e["source"],
            start,
            end,
            work / name,
            audio_fade=exact_fade(e, audio_fade),
            kind=e.get("kind"),
            reuse_path=old,
            source_mode=None if pano_source else pano_rec.get("mode"),
            grade=grade,
        )
        return read_err, pano_rec, False
    read_err = export_clip_checked(
        src,
        info,
        start,
        end,
        work / name,
        job.fmt,
        job.vertical,
        _crop_box(e),
        exact_fade(e, audio_fade),
        "",  # extend does not compress peaks (warned before)
        grade,
    )
    if resume is not None:
        resume.add(name, read_err)
    return read_err, pano_rec, False


def _clip_entry(
    plan: ExtendPlan,
    work: Path,
    end: float,
    speech_recs: dict[str, dict[str, Any]],
    audio_fade: float,
    pano_rec: dict[str, Any] | None,
    read_err: dict[str, Any] | None,
    sub: str | None,
) -> dict[str, Any]:
    """Log the re-exported clip and return its new manifest entry (``end``: its out point,
    before a read error)."""
    name, e, src, start, rec = plan.name, plan.entry, plan.src, plan.start, plan.rec
    moved = any((speech_recs.get(name) or {}).get(k, {}).get("moved") for k in ("start", "end"))
    logger.info(
        "  %s  [%.2f-%.2f] -> [%.2f-%.2f] (pad +%.2f / +%.2f)%s%s",
        name,
        e["start"],
        e["end"],
        start,
        end,
        rec["applied"]["start"],
        rec["applied"]["end"],
        f" clamped {rec['clamped']}" if rec.get("clamped") else "",
        " speech-adjusted" if moved else "",
    )
    p0, p1 = plan.keep
    return {k: v for k, v in e.items() if k not in ("read_from", "srt", *PEAK_KEYS)} | {
        "size": file_size(work / name),
        **({"read_from": str(src)} if str(src) != e["source"] else {}),
        "start": start,
        "end": end,
        "duration": round(end - start, 3),
        "pick": {"start": round(p0, 3), "end": round(p1, 3)},
        "extend": rec,
        **({"speech": speech_recs[name]} if name in speech_recs else {}),
        "audio_fade": exact_fade(e, audio_fade),
        **({"pano": pano_rec} if pano_rec else {}),
        **({"read_error": read_err} if read_err else {}),
        **({"srt": sub} if sub else {}),
    }


def _write_extend_manifest(
    work: Path,
    manifest: dict[str, Any],
    clips_dir: Path,
    entries: list[dict[str, Any]],
    *,
    plans: list[ExtendPlan],
    padding: Padding,
    speech: boundaries.SpeechSettings,
    audio_fade: float,
    skipped: dict[str, str],
    reuse: _Reuse | None,
    reused: int,
    clip_overrides: dict[str, dict[str, Any]] | None,
) -> None:
    """Write ``work/manifest.json``: ``clips_dir``'s ``manifest`` with the run's settings and
    the new ``entries``."""
    # Off when every clip has "speech_aware": false (the pass left them all alone).
    speech_on = speech.enabled and any(p.entry.get("speech_aware") is not False for p in plans)
    cache.save_json(
        work / "manifest.json",
        {k: v for k, v in manifest.items() if k != "compress_peaks"}
        | {
            "generator": MANIFEST_GENERATOR,
            "created": datetime.now().isoformat(timespec="seconds"),
            "extended_from": str(clips_dir.resolve()),
            "extend": {"start": padding.start, "end": padding.end},
            "speech_aware": speech.to_json() | {"enabled": speech_on},
            "audio_fade": audio_fade,
            "skipped_no_manifest": list(skipped),
            **(
                {"reused_from": str(reuse.dir.resolve()), "reused": reused}
                if reuse is not None
                else {}
            ),
            **({"clip_overrides": clip_overrides} if clip_overrides else {}),
            "clips": entries,
        },
    )
