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
(source, in and out points, crop, format), keeping the exact file names and
order of the clips *still in the folder*, so renames and deletions made
during review are respected. Per-clip speech-aware caps come from the
manifest entry (``speech_max_extend`` / ``speech_max_lead``, set by a
segment) or ``--clip-override``, which can also set one clip's total padding
(``extend`` / ``extend_start`` / ``extend_end``); ``--reuse DIR`` copies clips whose result is
identical to one in an earlier output instead of re-encoding them.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cutpoint import boundaries, cache, srt
from cutpoint.config import path_key, remap_source
from cutpoint.crop import Crop, parse_spec
from cutpoint.encode import OutputFormat, export_clip_checked
from cutpoint.ffmpeg_utils import (
    MediaInfo,
    atomic_output,
    check_not_sources,
    output_dir,
    probe,
    recover_swap,
    source_keys,
    staged_dir,
    swap_in,
)
from cutpoint.planning import Padding, pad, speech_pass
from cutpoint.utils import clip_files

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


def _manifest_project(manifest: dict[str, Any], clips_dir: Path) -> Path:
    """Project folder of a clips folder (manifest ``project``, else the parent)."""
    recorded = manifest.get("project")
    if recorded:
        try:
            if Path(recorded).is_dir():
                return Path(recorded)
        except OSError:
            pass
    return clips_dir.resolve().parent


def _pick(e: dict[str, Any]) -> tuple[float, float]:
    """The picked moment of a manifest entry (before any padding)."""
    rec = e.get("pick") or (e.get("extend") or {}).get("pre_pad") or e
    return float(rec["start"]), float(rec["end"])


PAD_KEYS = ("extend", "extend_start", "extend_end")
"""``--clip-override`` keys that set a clip's total padding (s) per side."""


def _pad_keys(keys: dict[str, Any], where: str) -> dict[str, float]:
    """Validated padding keys (``extend`` / ``extend_start`` / ``extend_end``)."""
    out: dict[str, float] = {}
    for key in PAD_KEYS:
        if key not in keys or keys[key] is None:
            continue
        try:
            val = float(keys[key])
        except (TypeError, ValueError):
            raise ValueError(f"{where}: {key} must be a number of seconds") from None
        if not val >= 0:  # also rejects NaN
            raise ValueError(f"{where}: {key} must not be negative")
        out[key] = val
    return out


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
        pads = _pad_keys(keys, where)
        hits = [p for p in plans if fnmatch.fnmatch(p.name.lower(), pattern.lower())]
        if not hits:
            raise ValueError(f"--clip-override {pattern}: matches no clip in the folder")
        for plan in hits:
            plan.entry = plan.entry | limits
            if pads:
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


def _reusable(
    old: dict[str, Any] | None, new: dict[str, Any], same_format: bool, audio_fade: float
) -> bool:
    """Whether an earlier output's entry renders exactly like the new one."""
    if (
        old is None
        or not same_format
        or old.get("read_error")
        or old.get("pano")
        or new.get("pano")
    ):
        return False
    return (
        old.get("source") == new.get("source")
        and abs(float(old["start"]) - new["start"]) < 1e-3
        and abs(float(old["end"]) - new["end"]) < 1e-3
        and old.get("crop") == new.get("crop")
        and float(old.get("audio_fade", 0.25)) == float(audio_fade)
    )


def extend_clips(
    clips_dir: Path,
    padding: Padding,
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
        force: Allow writing into a non-empty folder, including ``clips_dir``.
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
            points, crop, format and fade equal its entry there is copied
            instead of re-encoded (never for 360 clips).

    Returns:
        The output folder.

    Raises:
        FileNotFoundError: If ``clips_dir`` or its manifest is missing.
        FileExistsError: If ``out_dir`` is non-empty (or is ``clips_dir``)
            and ``force`` is False.
        ValueError: If nothing could be re-exported.
    """
    recover_swap(clips_dir)  # a hard-killed earlier replace, before the clips are read
    manifest = cache.load_json(clips_dir / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(
            f"No manifest.json in {clips_dir}; it must come from `cutpoint clips`"
        )
    speech = speech or boundaries.SpeechSettings()
    if not padding.active and not speech.enabled:
        raise ValueError(
            "Nothing to do: give --seconds (or --start / --end) > 0, or leave --speech-aware on"
        )
    out_dir = out_dir or (
        default_out_dir(clips_dir, padding)
        if padding.active
        else clips_dir.with_name(f"{clips_dir.name}_speech")
    )
    in_place = out_dir.resolve() == clips_dir.resolve()
    if not in_place:
        recover_swap(out_dir)  # a hard-killed earlier replace, before the folder is looked at
    non_empty = out_dir.exists() and any(out_dir.iterdir())
    if (in_place or non_empty) and not force:
        raise FileExistsError(
            f"{out_dir} {'is the input folder' if in_place else 'is not empty'};"
            " use --force to overwrite"
        )
    # Into a non-empty folder (in place or --out with --force), the clips are exported into a
    # hidden staging folder and swapped in only when all succeeded, so a failed or cancelled
    # run leaves the folder as it was; staged_dir refuses a drive root, a mount point or a
    # bind mount before any export. A run that fails before writing anything leaves no
    # empty folder behind.
    staged = in_place or non_empty
    with staged_dir(out_dir) if staged else output_dir(out_dir) as work:
        by_file = {e["file"]: e for e in manifest.get("clips", [])}
        fmt_d = manifest.get("format") or {}
        fmt = OutputFormat(int(fmt_d["width"]), int(fmt_d["height"]), int(fmt_d["fps"]))
        vertical = bool(fmt_d.get("vertical"))
        files = clip_files(clips_dir)
        infos: dict[str, MediaInfo] = {}
        plans: list[ExtendPlan] = []
        skipped: list[str] = []
        for f in files:
            e = by_file.get(f.name)
            if e is None:
                logger.warning("  %s: no manifest entry (added by hand?); skipped", f.name)
                skipped.append(f.name)
                continue
            src = _read_path(e)
            if str(src) not in infos:
                infos[str(src)] = probe(src)
            info = infos[str(src)]
            start, end, rec = pad(src, info, float(e["start"]), float(e["end"]), padding, e)
            plans.append(ExtendPlan(f.name, e, src, info, start, end, rec, _pick(e)))

        apply_overrides(plans, clip_overrides, padding)
        reuse_by_file: dict[str, dict[str, Any]] = {}
        reuse_same_format = False
        if reuse_dir is not None:
            if reuse_dir.resolve() in (out_dir.resolve(), work.resolve()):
                raise ValueError("--reuse must be an earlier output, not the output folder")
            reuse_manifest = cache.load_json(reuse_dir / "manifest.json")
            if reuse_manifest is None:
                raise FileNotFoundError(f"No manifest.json in --reuse folder {reuse_dir}")
            reuse_by_file = {x["file"]: x for x in reuse_manifest.get("clips", [])}
            reuse_same_format = (reuse_manifest.get("format") or {}) == fmt_d
        project = _manifest_project(manifest, clips_dir)
        speech_recs = speech_pass(
            plans, speech, project, fit_end=lambda e: not _fixed_tail(e), hi_clamp=_hi_clamp
        )
        pano = None
        if any(p.entry.get("pano") for p in plans):
            from cutpoint.pano.render import Format360, PanoExporter  # noqa: PLC0415

            pano = PanoExporter(
                project,
                out_dir,
                vertical,
                pano_source or "auto",
                pano_cmd_every,
            )
            pano.fmt = Format360(fmt.width, fmt.height, fmt.fps)
        # Only clips with a manifest entry are written; none may be a file this run reads.
        reads = []
        for plan in plans:
            e = plan.entry
            mode = None if pano_source else (e.get("pano") or {}).get("mode")
            pano_files = pano.inputs(e["source"], mode) if pano and e.get("pano") else []
            reads.append((e["source"], plan.src, pano_files))
        names = [p.name for p in plans]
        check_not_sources(out_dir, names, source_keys(reads))
        # Like clips --force: never replace a clip or .srt that out_dir's manifest.json does
        # not list; the .srt files it lists for the clips written here are replaced.
        listed = srt.listed_files(out_dir)
        srt.check_unlisted_clips(out_dir, names, listed)
        subs = srt.ClipSubtitles(srt.unlisted_srts(out_dir, listed))
        ours = {path_key(n) for n in listed or []}
        old_srts = [s for n in names if path_key(s := Path(n).with_suffix(".srt").name) in ours]
        entries = []
        reused = 0
        for plan in plans:
            name, e, src, info = plan.name, plan.entry, plan.src, plan.info
            start, end, rec = plan.start, plan.end, plan.rec
            pano_rec = e.get("pano")
            prior = reuse_by_file.get(name)
            if (
                reuse_dir is not None
                and (reuse_dir / name).is_file()
                and _reusable(
                    prior, e | {"start": start, "end": end}, reuse_same_format, audio_fade
                )
            ):
                with atomic_output(work / name) as part:
                    shutil.copy2(reuse_dir / name, part)
                reused += 1
                read_err = None
                logger.info("  %s: unchanged, copied from %s", name, reuse_dir.name)
            elif pano is not None and pano_rec:
                old = Path(pano_rec["path_file"]) if pano_rec.get("path_file") else None
                read_err, pano_rec = pano.export(
                    e["source"],
                    start,
                    end,
                    work / name,
                    audio_fade=audio_fade,
                    kind=e.get("kind"),
                    reuse_path=old,
                    source_mode=None if pano_source else pano_rec.get("mode"),
                )
            else:
                box = parse_spec(e.get("crop"))
                read_err = export_clip_checked(
                    src,
                    info,
                    start,
                    end,
                    work / name,
                    fmt,
                    vertical,
                    box if isinstance(box, Crop) else None,
                    audio_fade,
                )
            if read_err and read_err["end"] < end:
                end = read_err["end"]
            sub = subs.write(work / name, e["source"], start, end)
            moved = any(
                (speech_recs.get(name) or {}).get(k, {}).get("moved") for k in ("start", "end")
            )
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
            entries.append(
                {k: v for k, v in e.items() if k not in ("read_from", "srt")}
                | {
                    **({"read_from": str(src)} if str(src) != e["source"] else {}),
                    "start": start,
                    "end": end,
                    "duration": round(end - start, 3),
                    "pick": {"start": round(p0, 3), "end": round(p1, 3)},
                    "extend": rec,
                    **({"speech": speech_recs[name]} if name in speech_recs else {}),
                    "audio_fade": audio_fade,
                    **({"pano": pano_rec} if pano_rec else {}),
                    **({"read_error": read_err} if read_err else {}),
                    **({"srt": sub} if sub else {}),
                }
            )
        subs.report(len(plans))
        if not entries:
            raise ValueError(f"No clips with manifest entries in {clips_dir}")
        cache.save_json(
            work / "manifest.json",
            manifest
            | {
                "created": datetime.now().isoformat(timespec="seconds"),
                "extended_from": str(clips_dir.resolve()),
                "extend": {"start": padding.start, "end": padding.end},
                "speech_aware": speech.to_json(),
                "audio_fade": audio_fade,
                "skipped_no_manifest": skipped,
                **(
                    {"reused_from": str(reuse_dir.resolve()), "reused": reused}
                    if reuse_dir is not None
                    else {}
                ),
                **({"clip_overrides": clip_overrides} if clip_overrides else {}),
                "clips": entries,
            },
        )
        if staged:  # all or nothing; only the clips written here and manifest.json are replaced
            swap_in(work, out_dir, old_srts)
    if staged and not in_place:
        written = {path_key(e["file"]) for e in entries}
        kept = [p for p in clip_files(out_dir) if path_key(p.name) not in written]
        if kept:
            logger.warning(
                "Kept %s in %s: not re-exported by this run, so not in the new manifest.json; "
                "assemble includes them, so move them out if they are not part of the edit",
                ", ".join(p.name for p in kept),
                out_dir,
            )
    logger.info(
        "Re-exported %d clip(s) with +%gs / +%gs padding to %s%s%s",
        len(entries),
        padding.start,
        padding.end,
        out_dir,
        f" ({reused} copied unchanged from {reuse_dir})" if reuse_dir else "",
        f"; skipped {len(skipped)} without a manifest entry" if skipped else "",
    )
    return out_dir
