"""``cutpoint assemble``: join the clips currently in a folder into one video.

Order comes from the filenames (natural sort), never from manifest.json, so the
user can delete/rename/reorder clips after reviewing them.

Audio edge fades: ``cutpoint clips`` and ``cutpoint extend`` bake a short audio fade
in/out into every clip (manifest ``audio_fade``), so the clips, the Resolve
timeline and final.mp4 all sound the same. ``assemble`` only adds the fade
(``audio_fade``, default 0.25 s) to clips that do not carry one: older clip
folders and files added by hand. Clips are never faded twice. The video stays
a hard cut unless ``transition="fade"``.

Audio transitions (``audio_transition``) follow the video by default: ``cut`` with cuts,
``crossfade`` with fades. A fade can also take hard audio cuts (:func:`crossfade`), which get
the edge fades as on cuts. A cut cannot take an audio crossfade: the clips have no audio past
their ends to overlap without moving the later clips' audio off their video.

Peak compression (``compress_peaks``, see :data:`cutpoint.encode.PEAK_PRESETS`) works the
same way: clips exported with it (manifest ``compress_peaks``) are left alone, the others
are measured and get the compressor and leveling gain in the join, before the edge fade.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from cutpoint import jsonout, paths
from cutpoint.encode import (
    A_FMT,
    AUDIO_OUT,
    BT709_TAGS,
    PEAK_PRESETS,
    PEAK_TAG,
    SILENCE,
    TAG_BT709,
    audio_fade_filter,
    bt601_matrix,
    bt709_filter,
    color_kind,
    limiter_chain,
    plan_makeup,
    plan_peaks,
)
from cutpoint.ffmpeg_utils import (
    FFmpegError,
    MediaInfo,
    VideoEncoder,
    atomic_output,
    clip_momentary,
    encode_with_fallback,
    extract_audio,
    ffprobe_json,
    has_soxr,
    measure_loudness,
    output_dir,
    probe,
    recover_swap,
    run,
    swap_lock,
)
from cutpoint.timeline import write_timelines
from cutpoint.transcribe import Segment, Transcriber, flat_words
from cutpoint.utils import clip_files

logger = logging.getLogger(__name__)


def list_clips(clips_dir: Path, exclude: set[Path]) -> list[Path]:
    """All .mp4 files directly in ``clips_dir`` in natural filename order."""
    return [p for p in clip_files(clips_dir) if p.resolve() not in exclude]


def baked_fades(clips_dir: Path, files: list[Path]) -> list[float]:
    """Audio fade already baked into each file, per its folder's manifest.json entry (renamed
    clips included, see :func:`~cutpoint.planning.manifest_entries`; 0 if none)."""
    from cutpoint.planning import manifest_entries  # noqa: PLC0415

    by_file = manifest_entries(clips_dir, files)
    return [float((by_file.get(f.name) or {}).get("audio_fade") or 0.0) for f in files]


def _peak_tagged(path: Path) -> bool:
    """Whether ``path`` carries :data:`~cutpoint.encode.PEAK_TAG` (its container comment)."""
    try:
        comment = run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment",
                       "-of", "default=nw=1:nk=1", str(path)])  # fmt: skip
    except FFmpegError:
        return False
    return PEAK_TAG in comment


def baked_peaks(clips_dir: Path, files: list[Path]) -> list[bool]:
    """Whether each file already has peak compression baked in: its folder's manifest entry
    (renamed clips included, see :func:`~cutpoint.planning.manifest_entries`) says so, or, for a
    file without one (added during review), its own tag. The entry wins: ffmpeg copies a
    tagged source's comment into a clip exported without it."""
    from cutpoint.planning import manifest_entries  # noqa: PLC0415

    by_file = manifest_entries(clips_dir, files)
    return [
        e.get("compress_peaks") in PEAK_PRESETS
        if (e := by_file.get(f.name)) is not None
        else _peak_tagged(f)
        for f in files
    ]


def peak_filters(
    clips_dir: Path, files: list[Path], infos: list[MediaInfo], preset: str
) -> list[str]:
    """Per-clip :meth:`~cutpoint.encode.PeakPlan.filter` for ``preset`` (all "" for ``off``).

    Clips with compression baked in (:func:`baked_peaks`) count for the typical level and
    the mean loudness but get no filter and no makeup. Track 0 is measured, as
    :func:`_audio_chain` uses; the clips that get a filter again through their gain and
    compressor for the makeup (:func:`~cutpoint.encode.plan_makeup`).
    """
    if preset == "off":
        return [""] * len(files)
    baked = baked_peaks(clips_dir, files)
    if all(baked):
        logger.info("Compress peaks: every clip is already compressed (baked)")
        return [""] * len(files)
    series = [clip_momentary(f, min(i.audio_streams, 1)) for f, i in zip(files, infos, strict=True)]
    plan = plan_peaks(preset, series, [i.duration for i in infos], [not b for b in baked])
    if plan is None:
        logger.info("Compress peaks: no clip has audio above silence")
        return [""] * len(files)
    compressed = [
        clip_momentary(f, min(i.audio_streams, 1), chain=chain)
        if (chain := plan.measure_chain(n))
        else []
        for n, (f, i) in enumerate(zip(files, infos, strict=True))
    ]
    plan = plan_makeup(plan, series, compressed)
    logger.info(
        "Compress peaks (%s): compressor from %.1f dB RMS; clip gains %s; makeup %s; %d baked",
        preset,
        plan.threshold,
        " ".join("-" if g is None else f"{g:+.1f}" for g in plan.gains),
        " ".join("-" if m is None else f"{m:+.1f}" for m in plan.makeup),
        sum(baked),
    )
    return [plan.filter(i) for i in range(len(files))]


def _audio_chain(
    i: int,
    info: MediaInfo,
    fade: float,
    label: int | None = None,
    peaks: str = "",
    end: float | None = None,
) -> str:
    """``[a<label>]`` (default ``label = i``) from input ``i``: common format, ``peaks``
    (:func:`peak_filters`), edge fade, then padded/trimmed to ``end`` s if given.

    Only track 0 is used; a multi-track input whose track 0 is shorter than its longest
    track (``audio_duration``) keeps the old fade-out, after track 0 ends.
    """
    label = i if label is None else label
    if info.audio_streams:
        return (
            f"[{i}:a:0]{A_FMT},asetpts=PTS-STARTPTS{peaks}"
            f"{audio_fade_filter(info.duration, fade, info.audio_duration)}"
            f"{'' if end is None else f',apad,atrim=end={end:.6f}'}[a{label}]"
        )
    return f"{SILENCE},atrim=duration={info.vduration:.3f}[a{label}]"


def _normalize_inputs(
    infos: list[MediaInfo],
    fades: list[float] | None = None,
    peaks: list[str] | None = None,
    ends: list[float] | None = None,
) -> tuple[list[str], Fraction]:
    """Per-input filter chains producing [v<i>]/[a<i>] with identical params, the audio
    padded/trimmed to ``ends`` if given.

    Each input also gets its :func:`~cutpoint.encode.bt709_filter` (#290), as at export, so
    xfade and concat only see BT.709 tv-range frames tagged so: a clip exported before #290
    (BT.601 data, or HD without tags) no longer shifts colour next to a newer one. A clip
    with a BT.709 matrix tag only gets :data:`~cutpoint.encode.TAG_BT709` (no pixel changes):
    ffmpeg 9 writes the frames' primaries and transfer over the output tags, and xfade passes
    on the first input's, so one clip without them left the final without them.
    """
    fps = infos[0].fps
    fades = fades or [0.0] * len(infos)
    peaks = peaks or [""] * len(infos)
    chains = []
    for i, info in enumerate(infos):
        chains.append(_video_chain(i, info, infos[0]))
        end = ends[i] if ends else None
        chains.append(_audio_chain(i, info, fades[i], peaks=peaks[i], end=end))
    return chains, fps


def _video_chain(i: int, info: MediaInfo, ref: MediaInfo, trim: str = "") -> str:
    """``[v<i>]`` from input ``i``: ``ref``'s size and rate, BT.709 (see
    :func:`_normalize_inputs`); ``trim`` (a filter and ",") goes right after the rate."""
    w, h, fps = ref.width, ref.height, ref.fps
    color = bt709_filter(info) or (TAG_BT709 if info.color_space == "bt709" else "")
    return (
        f"[{i}:v:0]scale={w}:{h}:force_original_aspect_ratio=decrease:out_range=tv,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},{trim}"
        f"{color + ',' if color else ''}format=yuv420p,"
        f"settb=AVTB,setpts=PTS-STARTPTS[v{i}]"
    )


def _encode(files: list[Path], graph: str, out: Path) -> None:
    encode_with_fallback(lambda enc: _encode_with(enc, files, graph, out), "assemble")


def _encode_with(enc: VideoEncoder, files: list[Path], graph: str, out: Path) -> None:
    if enc.upload:  # the graphs end in [v]: hand the frames to the GPU encoder
        graph += f";[v]{enc.upload}[vgpu]"
    args = ["ffmpeg", "-y", "-v", "error"]
    for f in files:
        args += ["-i", str(f)]
    args += [
        "-filter_complex",
        graph,
        "-map",
        "[vgpu]" if enc.upload else "[v]",
        "-map",
        "[a]",
        *enc.args(),
        "-color_range",
        "tv",
        # Every input is BT.709 after _normalize_inputs; the tags say so, whatever the first
        # clip carried (a BT.709 clip's frames already are, so no pixel changes).
        *BT709_TAGS,
        *AUDIO_OUT,
        "-movflags",
        "+faststart",
        str(out),
    ]
    run(args)


def concat_copy(
    files: list[Path],
    out: Path,
    infos: list[MediaInfo] | None = None,
    fades: list[float] | None = None,
    peaks: list[str] | None = None,
) -> None:
    """Join with the concat demuxer (inputs must share parameters): the video is
    stream-copied, the audio (with ``infos``) rebuilt from the clips and encoded once.

    Stream-copied AAC does not join cleanly: each later clip's priming frame (1024 samples)
    plays at its cut, and the first clip's priming delays the copied video by as much. So
    each clip's audio is decoded (its edit list applied), given an edge fade where its
    ``fades`` is > 0 and the ``peaks`` filters (:func:`peak_filters`), and padded/trimmed to
    where its video ends on the join (:func:`cut_audio_ends`). ``-copyts`` keeps the first
    clip's video at 0, as it is in the clip. Without ``infos`` both streams are copied.
    """
    listing = out.with_name(f".{out.stem}.concat.txt")
    lines = ["file '" + f.resolve().as_posix().replace("'", "'\\''") + "'" for f in files]
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args = ["ffmpeg", "-y", "-v", "error"]
    if infos:
        args += ["-copyts"]
    args += ["-f", "concat", "-safe", "0", "-i", str(listing)]
    if infos:
        fades = fades or [0.0] * len(files)
        peaks = peaks or [""] * len(files)
        for f in files:
            args += ["-i", str(f)]
        ends = cut_audio_ends(infos)
        chains = [
            _audio_chain(i + 1, info, fades[i], label=i, peaks=peaks[i], end=ends[i])
            for i, info in enumerate(infos)
        ]
        graph = ";".join(chains) + ";" + "".join(f"[a{i}]" for i in range(len(files)))
        args += [
            "-filter_complex",
            f"{graph}concat=n={len(files)}:v=0:a=1[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            *AUDIO_OUT,
        ]
    else:
        args += ["-c", "copy"]
    try:
        run([*args, "-movflags", "+faststart", str(out)])
    finally:
        listing.unlink(missing_ok=True)


def concat_reencode(
    files: list[Path],
    infos: list[MediaInfo],
    out: Path,
    fades: list[float] | None = None,
    peaks: list[str] | None = None,
) -> None:
    """Join clips with differing parameters by normalising and re-encoding.

    Each clip's audio is padded/trimmed to :func:`cut_audio_ends`, as in :func:`concat_copy`:
    concat starts the next clip after the longer of its streams, and the last clip's audio
    ends with its video."""
    chains, _ = _normalize_inputs(infos, fades, peaks, cut_audio_ends(infos))
    pairs = "".join(f"[v{i}][a{i}]" for i in range(len(files)))
    _encode(files, ";".join(chains) + f";{pairs}concat=n={len(files)}:v=1:a=1[v][a]", out)


def cut_duration(info: MediaInfo) -> float:
    """How long a clip lasts on a cut join: its container ``duration``.

    The concat demuxer starts the next file there, longer than the video when the
    audio runs on. The cut offsets (:func:`clip_offsets`, so final.srt) and the
    audio rebuilt by :func:`concat_copy` use it, so they stay on the video."""
    return info.duration


def cut_audio_ends(infos: list[MediaInfo]) -> list[float]:
    """Where each clip's audio ends on a cut join: where the next clip starts
    (:func:`cut_duration`), and the last clip's at the end of its video, so the joined
    audio is as long as the video."""
    return [cut_duration(i) for i in infos[:-1]] + [infos[-1].vduration]


def clip_offsets(infos: list[MediaInfo], fade: float = 0.0) -> list[float]:
    """Where each clip starts in the joined video: back to back, each crossfade
    (``fade`` s, 0 for cuts) overlapping it with the previous clip.

    Cuts (concat demuxer or filter) start a clip where the previous file ends: its
    container ``duration``, longer than its video when the audio runs on. Crossfades
    follow the video (``vduration``)."""
    offsets, t = [], 0.0
    for info in infos:
        offsets.append(t)
        t += info.vduration - fade if fade > 0 else cut_duration(info)
    return offsets


def crossfade_duration(infos: list[MediaInfo], fade: float) -> float:
    """The crossfade :func:`crossfade` uses for ``fade``: rounded to whole output frames (the
    first clip's rate, as :func:`_normalize_inputs`), at least one frame and at most 45% of
    the shortest clip. xfade works in whole frames, so acrossfade and :func:`clip_offsets`
    (final.srt) get the same rounded value; xfade shorter than a frame cuts the video short.
    Returns 0 when one frame is more than 45% of the shortest clip: no room to fade.
    """
    room = 0.45 * min(i.vduration for i in infos)
    fps = infos[0].fps
    if fps <= 0:
        return min(fade, room)
    most = math.floor(room * fps + 1e-9)
    if most < 1:
        return 0.0
    nearest = math.floor(fade * fps + 0.5)
    frames = min(max(nearest, 1), most)
    used = float(frames / fps)
    if nearest < 1:
        logger.info("Crossfade %.3fs is shorter than one frame; using %.3fs", fade, used)
    return used


def audio_cut_point(fade: float, fps: Fraction) -> float:
    """Where in a ``fade`` s crossfade (whole frames, :func:`crossfade_duration`) the audio
    cuts with ``audio_transition="cut"``: the middle, rounded down to a whole frame, so
    final.mp4 and the timelines (:func:`~cutpoint.timeline.write_otio`) cut on the same frame."""
    if fps <= 0:
        return fade / 2
    return float(Fraction(int(fade * fps + Fraction(1, 2)) // 2) / fps)


CUT_FADE = 0.02  # s: the fade on each side of a hard audio cut inside a video crossfade


def crossfade(
    files: list[Path],
    infos: list[MediaInfo],
    out: Path,
    fade: float,
    peaks: list[str] | None = None,
    audio: str = "crossfade",
    fades: list[float] | None = None,
) -> float:
    """Join clips with xfade over ``fade`` s, as returned by :func:`crossfade_duration` (> 0),
    with the ``peaks`` filters (:func:`peak_filters`). Returns ``fade``.

    ``audio="crossfade"`` joins the audio with acrossfade over the same ``fade``.
    ``audio="cut"`` joins it with hard cuts at :func:`audio_cut_point` into each fade instead:
    the outgoing clip's audio runs that far into the fade and the incoming clip's starts that
    far in, so each clip's audio stays on its video and the audio is as long as the video. Each
    cut gets a :data:`CUT_FADE` fade on both sides against clicks, and ``fades`` are the clips'
    edge fades, as on cuts (:func:`_edge_fades`).

    Times are written to the microsecond, not the millisecond (8 frames at 24 fps are
    0.333333 s), so acrossfade matches the frame-rounded fade. xfade starts at the first
    frame at or after ``offset``, and the frames of a previous xfade can sit a microsecond
    early (rounded timestamps add up), so each offset is passed half a frame early: the fade
    still starts on the frame :func:`clip_offsets` puts it on.

    Each clip's audio is padded/trimmed to its video (``vduration``), which xfade and
    :func:`clip_offsets` follow, so audio that runs on cannot push the later clips late.

    :func:`crossfade_copy` is the faster join of the same result; this is its fallback."""
    _encode(files, ";".join(_crossfade_chains(infos, fade, peaks, audio, fades)), out)
    return fade


def _crossfade_chains(
    infos: list[MediaInfo],
    fade: float,
    peaks: list[str] | None,
    audio: str,
    fades: list[float] | None,
    *,
    video: bool = True,
    base: int = 0,
) -> list[str]:
    """The filter chains of :func:`crossfade` (``[v]`` and ``[a]``), to join with ";".

    ``video=False`` leaves out the video chains (``[a]`` only) and ``base`` shifts the input
    indexes, for :func:`crossfade_copy`, whose input 0 is the joined video: the audio is the
    same graph, so it decodes to the same samples."""
    if video:
        chains, fps = _normalize_inputs(infos, fades if audio == "cut" else None, peaks)
    else:
        fps = infos[0].fps
        edges = (fades if audio == "cut" else None) or [0.0] * len(infos)
        filters = peaks or [""] * len(infos)
        chains = [
            _audio_chain(i + base, info, edges[i], label=i, peaks=filters[i])
            for i, info in enumerate(infos)
        ]
    last = len(infos) - 1
    if audio == "cut":
        cut = audio_cut_point(fade, fps)
        for i, info in enumerate(infos):
            start = cut if i > 0 else 0.0
            end = info.vduration - fade + cut if i < last else info.vduration
            edge = min(CUT_FADE, (end - start) / 2)
            chains.append(
                f"[a{i}]apad,atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS"
                + (f",afade=t=in:d={edge:.6f}" if i > 0 else "")
                + (f",afade=t=out:st={end - start - edge:.6f}:d={edge:.6f}" if i < last else "")
                + f"[at{i}]"
            )
        pairs = "".join(f"[at{i}]" for i in range(len(infos)))
        chains.append(f"{pairs}concat=n={len(infos)}:v=0:a=1[a]")
    else:
        for i, info in enumerate(infos):
            chains.append(f"[a{i}]apad,atrim=end={info.vduration:.6f},asetpts=PTS-STARTPTS[at{i}]")
    half_frame = 0.5 / float(fps) if fps > 0 else 0.0
    vprev, aprev = "v0", "at0"
    offsets = clip_offsets(infos, fade)
    for i in range(1, len(infos)):
        offset = max(offsets[i] - half_frame, 0.0)
        vout, aout = ("v", "a") if i == last else (f"vx{i}", f"ax{i}")
        if video:
            chains.append(
                f"[{vprev}][v{i}]xfade=transition=fade:duration={fade:.6f}:offset={offset:.6f}"
                f"[{vout}]"
            )
        if audio != "cut":
            chains.append(f"[{aprev}][at{i}]acrossfade=d={fade:.6f}[{aout}]")
        vprev, aprev = vout, aout
    return chains


# --------------------------------------------------------------------------- stream-copied crossfade
COPY_MAX_SHARE = 0.6  # re-encoding more of the frames than this: crossfade() is as fast


@dataclass(frozen=True)
class VideoPackets:
    """A clip's video packets, as :func:`crossfade_copy` needs them: ``frames`` packets (one
    per frame), the frame indexes of its ``keys`` and their ``key_times`` (s from the file
    start, as ``-ss`` reads them), the stream ``time_base`` and the hash of its ``extradata``
    (the encoder settings, which every piece of a stream-copied join must share), and the
    file's ``duration`` (s), where the concat demuxer starts the next piece."""

    frames: int
    keys: tuple[int, ...]
    key_times: tuple[float, ...]
    time_base: Fraction
    extradata: str
    duration: float = 0.0


def _video_packets(path: Path, fps: Fraction) -> VideoPackets | None:
    """The :class:`VideoPackets` of ``path`` (one ffprobe run), or None when its video is not
    constant ``fps`` (timestamps one frame apart) starting on a keyframe.

    Raises:
        FFmpegError: If ffprobe fails.
    """
    data = ffprobe_json(
        path,
        streams="v:0",
        entries="packet=pts,flags:stream=time_base,extradata_hash:format=start_time,duration",
        options=("-show_data_hash", "MD5"),
        timeout=None,  # lists every packet: a long clip on slow storage takes its time
    )
    streams, packets = data.get("streams") or [], data.get("packets") or []
    if not streams or not packets or any("pts" not in p for p in packets):
        return None
    tb = Fraction(streams[0].get("time_base") or "0")
    if tb <= 0 or fps <= 0 or "K" not in packets[0].get("flags", ""):
        return None
    step = 1 / (tb * fps)  # ticks per frame
    pts = sorted(int(p["pts"]) for p in packets)
    if step.denominator != 1:
        return None
    for n, (a, b) in enumerate(pairwise(pts)):
        if b - a != step:
            logger.debug("%s: frame %d is %d ticks after the one before, not %s", path.name,
                         n + 1, b - a, step)  # fmt: skip
            return None
    start = float(data.get("format", {}).get("start_time") or 0)
    keyed = sorted(int(p["pts"]) for p in packets if "K" in p.get("flags", ""))
    if keyed[0] != pts[0]:
        return None
    return VideoPackets(
        frames=len(pts),
        keys=tuple(int((k - pts[0]) / step) for k in keyed),
        key_times=tuple(float(k * tb) - start for k in keyed),
        time_base=tb,
        extradata=str(streams[0].get("extradata_hash") or ""),
        duration=float(data.get("format", {}).get("duration") or 0),
    )


def _why_no_copy(
    infos: list[MediaInfo], packets: list[VideoPackets | None] | None = None
) -> str | None:
    """Why the clips cannot be joined by :func:`crossfade_copy`, or None if they can.

    The clips (``infos``) must be H.264 4:2:0 of the first one's size and rate, tagged BT.709
    TV range (the join's frames are, see :func:`_normalize_inputs`). With their ``packets``,
    also constant rate from a keyframe, as many frames as their video lasts, and one time base
    and set of encoder settings."""
    if len(infos) < 2 or infos[0].fps <= 0:
        return "fewer than two clips or no frame rate"
    ref = infos[0]
    for info in infos:
        name = info.path.name
        if info.vcodec != "h264" or info.pix_fmt != "yuv420p":
            return f"{name} is not H.264 yuv420p"
        if (info.width, info.height, info.fps) != (ref.width, ref.height, ref.fps):
            return f"{name} differs from {ref.path.name} in size or frame rate"
        tags = (info.color_space, info.color_primaries, info.color_transfer, info.color_range)
        if tags != ("bt709", "bt709", "bt709", "tv") or bt709_filter(info):
            return f"{name} is not tagged BT.709 TV range"
    for info, p in zip(infos, packets or [], strict=False):
        name = info.path.name
        if p is None:
            return f"{name} is not constant frame rate from a keyframe"
        if p.frames != round(info.vduration * float(ref.fps)):
            return f"{name} has {p.frames} frames for {info.vduration:.3f}s of video"
        if p.time_base != packets[0].time_base or p.time_base.numerator != 1:
            return f"{name} has another time base"
        if p.extradata != packets[0].extradata:
            return f"{name} was encoded with other settings than {ref.path.name}"
    return None


@dataclass(frozen=True)
class Body:
    """Frames [``start``, ``end``) of clip ``clip``, stream-copied (``start`` a keyframe, and
    ``end`` one or the clip's end)."""

    clip: int
    start: int
    end: int


@dataclass(frozen=True)
class Run:
    """Re-encoded crossfades from clip ``i`` frame ``s`` (a keyframe) to clip ``j`` frame
    ``e`` (exclusive; a keyframe or the clip's end): every join between them, and clips with
    no frames to copy between their fades whole."""

    i: int
    s: int
    j: int
    e: int


def run_frames(run: Run, frames: list[int], fade_frames: int) -> int:
    """How many output frames ``run`` makes: its clips' frames, less one fade per join."""
    inside = frames[run.i] - run.s + sum(frames[run.i + 1 : run.j]) + run.e
    return inside - (run.j - run.i) * fade_frames


def fade_copy_plan(packets: list[VideoPackets], fade_frames: int) -> list[Body | Run] | None:
    """The pieces of a crossfade join of clips with ``packets`` over ``fade_frames``, in order.

    Each clip but the first is copied from its first keyframe at or after the fade in, each
    but the last up to its last keyframe at or before the fade out; the fades between are
    re-encoded (:class:`Run`). A clip with no frames left to copy stays in the run. Returns
    None when more than :data:`COPY_MAX_SHARE` of the output would be re-encoded.
    """
    frames = [p.frames for p in packets]
    last = len(packets) - 1
    pieces: list[Body | Run] = []
    opened: tuple[int, int] | None = None  # (clip, frame) the current run starts at
    for k, p in enumerate(packets):
        head = 0 if k == 0 else next((f for f in p.keys if f >= fade_frames), p.frames)
        tail = p.frames if k == last else max(f for f in p.keys if f <= p.frames - fade_frames)
        if k > 0 and head >= tail and k < last:
            continue  # nothing to copy: the run goes on through this clip
        if opened is not None:
            pieces.append(Run(opened[0], opened[1], k, min(head, p.frames)))
        if head < tail:
            pieces.append(Body(k, head, tail))
        opened = (k, tail) if k < last else None
    total = sum(frames) - last * fade_frames
    encoded = sum(run_frames(r, frames, fade_frames) for r in pieces if isinstance(r, Run))
    return None if encoded > COPY_MAX_SHARE * total else pieces


def _encode_run(
    enc: VideoEncoder,
    files: list[Path],
    infos: list[MediaInfo],
    packets: list[VideoPackets],
    run_: Run,
    fade: float,
    out: Path,
) -> None:
    """Encode ``run_`` into ``out`` with ``enc``: :func:`crossfade`'s video graph over its
    clips, the first read from its keyframe ``s`` on, the last cut at frame ``e``.

    The movie time scale is the clips' too: in the default 1000 Hz, a window that is no whole
    number of milliseconds (44 frames at 60 fps) gets a rounded duration, and the concat
    demuxer starts the next piece that much early."""
    fps = infos[0].fps
    half = 0.5 / float(fps)
    offsets = clip_offsets(infos, fade)
    t0 = offsets[run_.i] + run_.s / float(fps)
    args, chains = ["ffmpeg", "-y", "-v", "error"], []
    for n, k in enumerate(range(run_.i, run_.j + 1)):
        p = packets[k]
        if k == run_.i and run_.s > 0:  # half a frame early: the keyframe is the first kept
            args += ["-ss", f"{p.key_times[p.keys.index(run_.s)] - half:.6f}"]
        args += ["-i", str(files[k])]
        trim = f"trim=end_frame={run_.e}," if k == run_.j and run_.e < p.frames else ""
        chains.append(_video_chain(n, infos[k], infos[0], trim))
    vprev, last = "v0", run_.j - run_.i
    for n in range(1, last + 1):
        offset = max(offsets[run_.i + n] - t0 - half, 0.0)
        vout = "v" if n == last else f"vx{n}"
        chains.append(
            f"[{vprev}][v{n}]xfade=transition=fade:duration={fade:.6f}:offset={offset:.6f}[{vout}]"
        )
        vprev = vout
    graph = ";".join(chains)
    if enc.upload:
        graph += f";[v]{enc.upload}[vgpu]"
    frames = [q.frames for q in packets]
    fade_frames = round(fade * float(fps))
    args += [
        "-filter_complex",
        graph,
        "-map",
        "[vgpu]" if enc.upload else "[v]",
        "-frames:v",
        str(run_frames(run_, frames, fade_frames)),
        "-an",
        *enc.args(),
        "-color_range",
        "tv",
        *BT709_TAGS,
        "-video_track_timescale",
        str(packets[0].time_base.denominator),
        "-movie_timescale",
        str(packets[0].time_base.denominator),
        str(out),
    ]
    run(args)


def _extract_body(file: Path, body: Body, p: VideoPackets, fps: Fraction, out: Path) -> None:
    """Stream-copy frames [``body.start``, ``body.end``) of ``file`` into ``out`` (video only).

    The segment muxer cuts on the keyframes at those frames (half a frame early, so the
    keyframe opens the piece) and the middle piece is kept: packet-exact, unlike the concat
    demuxer's inpoint/outpoint, which cut on decoding timestamps. The movie time scale is the
    clip's, as in :func:`_encode_run`."""
    timescale = p.time_base.denominator
    half = 0.5 / float(fps)
    cuts = [p.key_times[p.keys.index(f)] - half for f in (body.start, body.end) if 0 < f < p.frames]
    assert cuts  # every body ends at a fade or starts after one
    # Relative pattern run in out's folder: a % in the project path is no template field.
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(file.resolve()), "-map", "0:v:0", "-c", "copy",
        "-f", "segment", "-segment_times", ",".join(f"{t:.6f}" for t in cuts),
        "-reset_timestamps", "1", "-segment_format", "mp4",
        "-segment_format_options", f"video_track_timescale={timescale}:movie_timescale={timescale}",
        f"{out.stem}.%d.mp4",
    ], cwd=out.parent)  # fmt: skip
    keep = 1 if body.start > 0 else 0
    for n in range(len(cuts) + 1):
        piece = out.with_name(f"{out.stem}.{n}.mp4")
        if n == keep:
            if not piece.exists():
                raise FFmpegError(f"the segment muxer wrote no piece {n} of {file.name}")
            piece.replace(out)
        else:
            piece.unlink(missing_ok=True)


def _piece_ok(
    piece: Path, fps: Fraction, frames: int, ref: VideoPackets, timed: bool = True
) -> bool:
    """Whether ``piece`` has ``frames`` constant-rate frames from a keyframe and ``ref``'s
    time base and encoder settings, so the concat demuxer can join it to the others; if
    ``timed``, also a duration of those frames to half a tick (where the next piece starts).
    Says why not at DEBUG."""
    got = _video_packets(piece, fps)
    want = frames / float(fps)
    if got is None:
        why = "not constant frame rate from a keyframe"
    elif got.frames != frames:
        why = f"{got.frames} frames, not {frames}"
    elif (got.time_base, got.extradata) != (ref.time_base, ref.extradata):
        why = "another time base or encoder settings"
    elif timed and abs(got.duration - want) > float(ref.time_base) / 2 + 1e-6:
        why = f"lasts {got.duration:.6f}s, not {want:.6f}s"
    else:
        return True
    logger.debug("Crossfade: piece %s does not check out: %s", piece.name, why)
    return False


def _write_pieces(listing: Path, pieces: list[tuple[Path, int]], fps: Fraction) -> None:
    """Write the concat demuxer's ``listing`` of ``pieces`` (file, frames), in order.

    The demuxer starts each piece where the ones before it end, adding their durations in
    whole microseconds: a piece of no whole number of them (44 frames at 60 fps, 733333.33 us)
    would count short, and after 17 such at 60 fps the next piece starts a tick early. Each piece
    gets a ``duration`` from the frame grid, rounded so that the sums are the grid times to
    half a microsecond (``T_k`` = the frames before piece ``k`` / ``fps``)."""
    lines, before = [], 0
    for piece, frames in pieces:
        start, end = (round(Fraction(n) / fps * 10**6) for n in (before, before + frames))
        lines.append("file '" + piece.resolve().as_posix().replace("'", "'\\''") + "'\n")
        lines.append(f"duration {(end - start) // 10**6}.{(end - start) % 10**6:06d}\n")
        before += frames
    listing.write_text("".join(lines), encoding="utf-8")


def _no_copy(why: str) -> bool:
    logger.info("Crossfade: re-encoding the whole video (%s)", why)
    return False


def crossfade_copy(
    files: list[Path],
    infos: list[MediaInfo],
    out: Path,
    fade: float,
    peaks: list[str] | None = None,
    audio: str = "crossfade",
    fades: list[float] | None = None,
) -> bool:
    """:func:`crossfade` (same arguments) re-encoding only the video around the fades.

    The clips' video between the fades is stream-copied from keyframe to keyframe
    (:func:`fade_copy_plan`; ``cutpoint clips`` forces keyframes near both ends of every clip,
    :func:`~cutpoint.encode.edge_keyframes`); the windows around the fades go through
    :func:`crossfade`'s video graph (:func:`_encode_run`). The audio is :func:`crossfade`'s
    graph over the whole join, encoded once: the same samples, no seams at the pieces. The
    video has as many frames as :func:`crossfade`'s, so the timelines and final.srt match it.

    Returns:
        False, with one INFO line saying why and nothing written, when the clips cannot be
        joined so (:func:`_why_no_copy`, ``CUTPOINT_FADE_COPY=0``, too little to copy, a
        re-encoded window the copied video cannot join or an ffmpeg failure): call
        :func:`crossfade` then.
    """
    if paths.env("FADE_COPY") == "0":
        return _no_copy("CUTPOINT_FADE_COPY=0")
    if (why := _why_no_copy(infos)) is not None:
        return _no_copy(why)
    fps = infos[0].fps
    fade_frames = round(fade * float(fps))
    work = out.with_name(f".{out.stem.lstrip('.')}.fadecopy")  # out may be hidden already
    listing = work / "pieces.txt"
    try:
        packets = [_video_packets(f, fps) for f in files]
        if (why := _why_no_copy(infos, packets)) is not None:
            return _no_copy(why)
        copyable = [p for p in packets if p is not None]
        plan = fade_copy_plan(copyable, fade_frames)
        if plan is None:
            return _no_copy(f"more than {COPY_MAX_SHARE:.0%} of the frames are near a fade")
        work.mkdir(exist_ok=True)
        frames = [p.frames for p in copyable]
        ref, cpu, encoded = copyable[0], False, 0
        pieces: list[tuple[Path, int]] = []  # (file, frames)
        for n, item in enumerate(plan):
            piece = work / f"{n:04d}.mp4"
            if isinstance(item, Body):
                _extract_body(files[item.clip], item, copyable[item.clip], fps, piece)
                if not _piece_ok(piece, fps, item.end - item.start, ref):
                    return _no_copy(
                        f"the copied video of {files[item.clip].name} does not check out"
                    )
            else:
                want = run_frames(item, frames, fade_frames)

                def encode(
                    enc: VideoEncoder, item: Run = item, piece: Path = piece
                ) -> VideoEncoder:
                    _encode_run(enc, files, infos, copyable, item, fade, piece)
                    return enc

                enc = (
                    encode(VideoEncoder("cpu")) if cpu else encode_with_fallback(encode, "assemble")
                )
                ok = _piece_ok(piece, fps, want, ref)
                if not ok and enc.kind != "cpu":  # a GPU encoder's settings are not the clips'
                    cpu = True  # libx264 for this window and the rest
                    encode(VideoEncoder("cpu"))
                    ok = _piece_ok(piece, fps, want, ref)
                if not ok:
                    return _no_copy("a re-encoded fade does not check out")
                encoded += want
            pieces.append((piece, item.end - item.start if isinstance(item, Body) else want))
        _write_pieces(listing, pieces, fps)
        args = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing)]
        for f in files:
            args += ["-i", str(f)]
        graph = ";".join(_crossfade_chains(infos, fade, peaks, audio, fades, video=False, base=1))
        run([*args, "-filter_complex", graph, "-map", "0:v:0", "-map", "[a]", "-c:v", "copy",
             *AUDIO_OUT, "-movflags", "+faststart", str(out)])  # fmt: skip
        total = sum(frames) - (len(files) - 1) * fade_frames
        if not _piece_ok(out, fps, total, ref, timed=False):  # its audio may run on
            out.unlink(missing_ok=True)
            return _no_copy("the joined video does not check out")
    except (FFmpegError, OSError) as err:
        out.unlink(missing_ok=True)
        lines = [line for line in str(err).splitlines() if line.strip()]
        return _no_copy(f"ffmpeg failed: {lines[-1] if lines else err}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    logger.info(
        "Crossfade: re-encoded %d of %d frames (%.0f%%), stream-copied the rest",
        encoded,
        total,
        100 * encoded / total,
    )
    return True


# --------------------------------------------------------------------------- captions
def _ass_time(t: float) -> str:
    cs = round(max(t, 0) * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def caption_chunks(
    segments: list[Segment], max_words: int = 4, max_gap: float = 0.6
) -> list[tuple[float, float, str]]:
    """Group words into short (2-4 word) caption chunks."""
    chunks: list[tuple[float, float, str]] = []
    cur: list[dict] = []

    def flush() -> None:
        if cur:
            chunks.append((cur[0]["start"], cur[-1]["end"], " ".join(w["word"] for w in cur)))
            cur.clear()

    for w in flat_words(segments):
        if cur and (
            len(cur) >= max_words
            or w["start"] - cur[-1]["end"] > max_gap
            or cur[-1]["word"][-1:] in ".?!,"
        ):
            flush()
        cur.append(w)
    flush()
    # Hold each chunk a little longer, without overlapping the next one.
    held = []
    for i, (s, e, text) in enumerate(chunks):
        nxt = chunks[i + 1][0] if i + 1 < len(chunks) else e + 1.0
        held.append((s, min(max(e + 0.15, s + 0.5), nxt), text))
    return held


FALLBACK_CAPTION_FONT = "DejaVu Sans"


def caption_font() -> str:
    """Font family for burned-in captions: Arial on Windows, else fontconfig's sans-serif.

    Without ``fc-match`` (fontconfig) this logs a warning and returns
    :data:`FALLBACK_CAPTION_FONT`; ffmpeg's subtitle renderer needs fontconfig
    and an installed font to draw captions at all.
    """
    if sys.platform == "win32":
        return "Arial"
    exe = shutil.which("fc-match")
    if exe is None:
        logger.warning(
            "fc-match (fontconfig) not found: burned-in captions need fontconfig and a "
            "font, e.g. sudo apt install fontconfig fonts-dejavu-core; trying %r",
            FALLBACK_CAPTION_FONT,
        )
        return FALLBACK_CAPTION_FONT
    try:
        family = subprocess.run(
            [exe, "-f", "%{family}", "sans-serif"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        family = ""
    # "DejaVu Sans,DejaVu Sans Book": the first name; ',' would split the ASS style line.
    family = family.split(",")[0].strip()
    if not family:
        logger.warning(
            "fc-match found no sans-serif font: burned-in captions need one, e.g. "
            "sudo apt install fonts-dejavu-core; trying %r",
            FALLBACK_CAPTION_FONT,
        )
    return family or FALLBACK_CAPTION_FONT


def write_ass(chunks: list[tuple[float, float, str]], width: int, height: int, path: Path) -> None:
    """Bold, centred lower-third ASS subtitles (font: :func:`caption_font`)."""
    size = round(min(width, height) * 0.07)  # same visual size for 16:9 and 9:16
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
        "MarginV, Encoding\n"
        f"Style: Default,{caption_font()},{size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,{max(2, size // 12)},1,2,40,40,{round(height * 0.2)},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )
    lines = []
    for s, e, text in chunks:
        clean = text.upper().replace("\\", "").replace("{", "(").replace("}", ")")
        lines.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Default,,0,0,0,,{clean}")
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


# loudnorm alone did not guarantee the ceiling, and its linear mode only holds if
# measured true peak + gain stays below its TP target (at most 0 dBTP). A final with peaks
# near 0 dBFS that needs +8 dB never fits, and ffmpeg then silently switches to dynamic (AGC)
# mode, which re-levels the clips and undoes --compress-peaks (#523). So loudnorm only
# measures, and the gain to -14 LUFS is applied as a plain `volume`, followed by a limiter
# running at 192 kHz (4x oversampled, so inter-sample peaks are caught) that takes the
# peaks the gain pushes over the ceiling. Measured on the R6 montage: the limiter output is
# exact (-2.2 dBTP after resampling to 48 kHz), AAC at 192k then adds +1.2 dB but at 320k
# only +0.2 dB, so the normalised final is encoded at 320k. The result is measured after
# encoding; if the true peak is still above TP_CEILING, the limiter ceiling is lowered and
# the pass rerun.
LOUDNORM_I = -14.0  # LUFS target of --loudnorm
LOUDNORM_SHORT_LU = 1.0  # a final further below LOUDNORM_I than this logs a WARNING
LOUDNORM = "loudnorm=I=-14:TP=-1.0:LRA=11"  # measurement pass (and the fallback)
TP_CEILING = -1.5  # dBTP the final file must not exceed
LIMIT_START_DB = -2.3  # first limiter ceiling (dBFS at 192 kHz, before AAC)
FINAL_AUDIO_OUT = ["-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2"]
LIMIT_MARGIN_DB = 0.3  # extra headroom when a pass has to be redone
LIMIT_PASSES = 3


def loudnorm_filter(src: Path) -> str:
    """Measure ``src`` (loudnorm's first pass) and return a linear gain to LOUDNORM_I LUFS.

    The gain is a ``volume`` filter, never loudnorm's second pass, which falls back to
    dynamic mode when the peaks do not fit (#523); the limiter after it takes the peaks.
    """
    out = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(src),
            "-vn",
            "-af",
            f"{LOUDNORM}:print_format=json",
            "-f",
            "null",
            "-",
        ]
    )
    try:
        m = json.loads(out[out.rindex("{") : out.rindex("}") + 1])
        if m["input_i"] == "-inf":
            logger.info("Audio is silent; no loudness gain")
            return "anull"
        gain = LOUDNORM_I - float(m["input_i"])
        if not math.isfinite(gain):
            raise ValueError(m["input_i"])
    except (ValueError, KeyError):
        logger.warning("loudnorm measurement unparseable; using single-pass (dynamic) loudnorm")
        return LOUDNORM
    logger.info(
        "Loudness gain %+.1f dB (measured %.1f LUFS, true peak %s dBTP)",
        gain,
        float(m["input_i"]),
        m["input_tp"],
    )
    return f"volume={gain:.2f}dB"


def limiter_filter(limit_db: float) -> str:
    """Oversampled (192 kHz) lookahead limiter, back to 48 kHz afterwards.

    The downsampling uses soxr when the ffmpeg build has it, else ffmpeg's default (swr).
    """
    return limiter_chain(limit_db, has_soxr())


def post_process(
    stage: Path,
    out: Path,
    loudnorm: bool,
    captions: bool,
    model: str | None,
    limit_db: float = LIMIT_START_DB,
    ass: Path | None = None,
    dst: Path | None = None,
    limit: bool = False,
    language: str | None = None,
) -> Path | None:
    """Apply loudness normalisation (+ true-peak limiter) and/or burned-in captions.

    ``limit`` without ``loudnorm`` runs the limiter alone (compressed peaks, see
    :func:`peak_filters`). ffmpeg writes ``dst`` (default ``out``); the caption files are
    named after ``out``. ``language`` forces the captions' transcription language (None:
    detected).

    Returns:
        The caption ASS file (reuse it via ``ass`` when re-running), if any.
    """
    args = ["ffmpeg", "-y", "-v", "error", "-i", str(stage.resolve())]
    cwd, vf, tags = None, None, []
    info = probe(stage)
    if captions:
        if ass is None:
            wav = out.with_name(f".{out.stem}.captions.wav")
            extract_audio(stage, wav, info)
            tr = Transcriber(model=model, language=language)
            try:
                segments = tr.transcribe(wav)
            finally:
                tr.close()
                wav.unlink(missing_ok=True)
            ass = out.with_suffix(".ass")
            chunks = caption_chunks(segments)
            write_ass(chunks, info.width, info.height, ass)
            logger.info(
                "Captions: %d chunks (whisper %s on %s)", len(chunks), tr.model_name, tr.device
            )
        cwd = ass.parent  # relative path avoids Windows drive-colon escaping in filters
        vf = f"ass={ass.name}"
        # A BT.709 join (re-encoded, see _encode_with) keeps its tags; a stream-copied join of
        # HDR, BT.601 or full-range clips is not BT.709, so it keeps the tags ffmpeg carries.
        if not color_kind(info) and not bt601_matrix(info) and info.color_range != "pc":
            tags = ["-color_range", "tv", *BT709_TAGS]
    if loudnorm or limit:
        # loudnorm and the limiter's resampling can run the audio on past its input (about
        # 0.1 s seen, #473): it is trimmed back to where the video it was joined to ends.
        af = limiter_filter(limit_db)
        if loudnorm:
            af = f"{loudnorm_filter(stage)},{af}"
        end = info.vstart + info.vduration
        a_args = ["-af", f"{af},atrim=end={end:.6f}", *FINAL_AUDIO_OUT]
    else:
        a_args = ["-c:a", "copy"]
    tail = [*a_args, "-movflags", "+faststart", str((dst or out).resolve())]
    if vf is None:
        run([*args, "-c:v", "copy", *tail])
    else:
        encode_with_fallback(
            lambda enc: run([*args, "-vf", enc.chain(vf), *enc.args(), *tags, *tail], cwd=cwd),
            out.name,
        )
    return ass


def normalize_with_ceiling(
    stage: Path,
    out: Path,
    captions: bool,
    model: str | None,
    dst: Path | None = None,
    loudnorm: bool = True,
    language: str | None = None,
) -> tuple[float, float]:
    """Loudnorm (unless not ``loudnorm``) + limiter, re-run with a lower ceiling until true
    peak <= TP_CEILING.

    ``dst`` and ``language`` as for :func:`post_process`.

    Returns:
        Measured (integrated LUFS, true peak dBTP) of the final file.
    """
    limit_db, ass = LIMIT_START_DB, None
    for attempt in range(1, LIMIT_PASSES + 1):
        ass = post_process(
            stage,
            out,
            loudnorm,
            captions,
            model,
            limit_db,
            ass,
            dst,
            limit=not loudnorm,
            language=language,
        )
        lufs, tp = measure_loudness(dst or out)
        logger.info(
            "Loudness pass %d: %.1f LUFS, true peak %+.2f dBTP (limiter %.1f dBFS)",
            attempt,
            lufs,
            tp,
            limit_db,
        )
        if tp <= TP_CEILING:
            if loudnorm and -70 < lufs < LOUDNORM_I - LOUDNORM_SHORT_LU:
                msg = (
                    f"Final is {lufs:.1f} LUFS, {LOUDNORM_I - lufs:.1f} LU below "
                    f"{LOUDNORM_I:.0f}: the limiter takes the peaks the linear gain pushes over "
                    "the ceiling; --compress-peaks light|strong (if not used yet) leaves it "
                    "more room"
                )
                logger.warning(msg)
                jsonout.warn("loudness_below_target", msg)
            return lufs, tp
        limit_db -= tp - TP_CEILING + LIMIT_MARGIN_DB
    logger.warning(
        "True peak %+.2f dBTP is still above %.1f after %d passes", tp, TP_CEILING, LIMIT_PASSES
    )
    return lufs, tp


def _edge_fades(clips_dir: Path, files: list[Path], audio_fade: float) -> list[float]:
    """The audio edge fade to add to each clip: ``audio_fade`` where none is baked in, else 0."""
    baked = baked_fades(clips_dir, files)
    fades = [0.0 if b > 0 else max(0.0, audio_fade) for b in baked]
    n_add = sum(f > 0 for f in fades)
    logger.info(
        "Audio edge fades: %d clip(s) already faded (baked), %s",
        sum(b > 0 for b in baked),
        f"adding {audio_fade:g}s to {n_add}" if n_add else "adding none",
    )
    return fades


def _join(
    files: list[Path],
    infos: list[MediaInfo],
    clips_dir: Path,
    out: Path,
    transition: str,
    fade_dur: float,
    audio_fade: float,
    peaks: list[str] | None = None,
    audio_transition: str | None = None,
) -> float:
    """Join ``files`` into ``out`` with cuts or crossfades (see :func:`assemble`); ``peaks``
    are the per-clip :func:`peak_filters`.

    Returns:
        The crossfade duration used (0 for cuts), for :func:`clip_offsets`.
    """
    audio = audio_transition or default_audio_transition(transition)
    if transition == "fade" and len(files) > 1:
        fade_dur = crossfade_duration(infos, fade_dur)
        if fade_dur == 0:
            logger.warning(
                "One frame is more than 45% of the shortest clip: no room for a crossfade; "
                "joining with cuts"
            )
            transition = "cut"
    if len(files) == 1 or transition == "cut":
        fades = _edge_fades(clips_dir, files, audio_fade)
        uniform = len({i.concat_key() for i in infos}) == 1 and infos[0].audio_streams == 1
        if uniform:
            concat_copy(files, out, infos, fades, peaks)
        else:
            logger.info("Clip parameters differ; re-encoding instead of stream copy")
            concat_reencode(files, infos, out, fades, peaks)
        return 0.0
    if transition == "fade":
        fades = _edge_fades(clips_dir, files, audio_fade) if audio == "cut" else None
        used = fade_dur
        if not crossfade_copy(files, infos, out, fade_dur, peaks, audio, fades):
            used = crossfade(files, infos, out, fade_dur, peaks, audio, fades)
        logger.info("Crossfade duration %.3fs%s", used, ", audio cut" if audio == "cut" else "")
        return used
    raise ValueError(f"Unknown transition {transition!r}")


AUDIO_TRANSITIONS = ("cut", "crossfade")


def default_audio_transition(transition: str) -> str:
    """The audio transition that goes with the video ``transition``: ``cut`` with cuts,
    ``crossfade`` with fades."""
    return "crossfade" if transition == "fade" else "cut"


def check_transitions(transition: str, audio_transition: str | None) -> str | None:
    """Why ``audio_transition`` cannot go with ``transition``, or None if it can."""
    if audio_transition is not None and audio_transition not in AUDIO_TRANSITIONS:
        return f"unknown audio transition {audio_transition!r}"
    if transition == "cut" and audio_transition == "crossfade":
        return (
            "an audio crossfade needs a video fade (--transition fade): at a cut the clips "
            "have no audio past their ends to overlap"
        )
    return None


def assemble(
    clips_dir: Path,
    out: Path | None = None,
    *,
    transition: str = "cut",
    fade_dur: float = 0.25,
    loudnorm: bool = False,
    captions: bool = False,
    caption_model: str | None = None,
    audio_fade: float = 0.25,
    compress_peaks: str = "off",
    audio_transition: str | None = None,
) -> Path:
    """Assemble ``clips_dir`` into ``out`` and write .otio/.fcpxml timelines next to it.

    Subtitles ``<out>.srt`` are written from the clips' cached source transcripts
    (see :func:`cutpoint.srt.write_final_srt`), timed to the joined video.

    ``audio_fade``: audio fade in/out (s) added to each clip that has no fade
    baked in by ``cutpoint clips`` / ``cutpoint extend`` (0 = add none). Ignored when the
    audio crossfades (``transition="fade"`` by default), unless the join falls back to cuts
    (no room for a one-frame crossfade).

    ``audio_transition``: ``cut`` or ``crossfade`` (:data:`AUDIO_TRANSITIONS`); None follows
    ``transition`` (:func:`default_audio_transition`). ``crossfade`` needs ``transition="fade"``.

    ``compress_peaks``: ``off``, ``light`` or ``strong`` (:data:`~cutpoint.encode.PEAK_PRESETS`)
    compresses loud peaks and levels the clips that do not have it baked in; the final audio
    then always ends with the true-peak limiter (with or without ``loudnorm``).

    Like ``clips``, a replace of ``clips_dir`` killed hard is first finished or undone
    (:func:`~cutpoint.ffmpeg_utils.recover_swap`), and the folder is locked for the run.

    Raises:
        FileNotFoundError: If the folder has no .mp4 clips.
        FolderBusyError: If another run is using ``clips_dir``.
        ValueError: If ``audio_transition`` does not go with ``transition``.
    """
    if (why := check_transitions(transition, audio_transition)) is not None:
        raise ValueError(why)
    with swap_lock(clips_dir):  # one run at a time: another would remove a run's staging folder
        recover_swap(clips_dir)  # leftovers of a hard-killed clips / extend, before listing
        return _assemble(
            clips_dir,
            out,
            transition,
            fade_dur,
            loudnorm,
            captions,
            caption_model,
            audio_fade,
            compress_peaks,
            audio_transition or default_audio_transition(transition),
        )


def _assemble(
    clips_dir: Path,
    out: Path | None,
    transition: str,
    fade_dur: float,
    loudnorm: bool,
    captions: bool,
    caption_model: str | None,
    audio_fade: float,
    compress_peaks: str,
    audio_transition: str,
) -> Path:
    """:func:`assemble`, with ``clips_dir`` locked."""
    if out is None:  # clips -> final.mp4, clips_vertical -> final_vertical.mp4
        name = clips_dir.resolve().name
        stem = "final" + name[5:] if name.lower().startswith("clips") else f"final_{name}"
        out = clips_dir.resolve().parent / f"{stem}.mp4"
    out = out.resolve()
    files = list_clips(clips_dir, exclude={out})
    if not files:
        raise FileNotFoundError(f"No .mp4 clips in {clips_dir}")
    # A run that fails before writing anything leaves no empty output folder behind.
    with output_dir(out.parent):
        infos = [probe(f) for f in files]
        mode = transition
        if audio_transition != default_audio_transition(transition):
            mode += f", audio {audio_transition}"
        logger.info(
            "Assembling %d clips (%s): %s", len(files), mode, ", ".join(f.name for f in files)
        )

        peaks = peak_filters(clips_dir, files, infos, compress_peaks)
        limit = loudnorm or compress_peaks != "off"
        from cutpoint.srt import clips_language  # noqa: PLC0415 - it imports this module

        language = clips_language(clips_dir) if captions else None
        # Written to a temporary name: final.mp4 only appears once complete.
        with atomic_output(out) as part:
            stage = out.with_name(f".{out.stem}.stage.mp4") if (limit or captions) else part
            try:
                used_fade = _join(
                    files,
                    infos,
                    clips_dir,
                    stage,
                    transition,
                    fade_dur,
                    audio_fade,
                    peaks,
                    audio_transition,
                )
                if limit:
                    lufs, tp = normalize_with_ceiling(
                        stage, out, captions, caption_model, part, loudnorm, language
                    )
                    logger.info("Final audio: %.1f LUFS integrated, true peak %+.2f dBTP", lufs, tp)
                elif captions:
                    post_process(
                        stage, out, False, captions, caption_model, dst=part, language=language
                    )
            finally:
                if stage != part:
                    stage.unlink(missing_ok=True)

    fcp, oti = write_timelines(infos, out, used_fade, audio_cut=audio_transition == "cut")
    duration = probe(out).duration
    logger.info("Wrote %s (%.2fs), %s, %s", out, duration, fcp.name, oti.name)
    from cutpoint.srt import write_final_srt  # noqa: PLC0415 - it imports this module

    srt_path = write_final_srt(clips_dir, files, infos, used_fade, out)
    jsonout.output("final", out)
    jsonout.output("fcpxml", fcp)
    jsonout.output("otio", oti)
    jsonout.output("srt", srt_path)
    jsonout.output("duration", round(duration, 3))
    jsonout.output("clip_count", len(files))
    return out
