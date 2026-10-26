"""``cutpoint assemble``: join the clips currently in a folder into one video.

Order comes from the filenames (natural sort), never from manifest.json, so the
user can delete/rename/reorder clips after reviewing them.

Audio edge fades: ``cutpoint clips`` and ``cutpoint extend`` bake a short audio fade
in/out into every clip (manifest ``audio_fade``), so the clips, the Resolve
timeline and final.mp4 all sound the same. ``assemble`` only adds the fade
(``audio_fade``, default 0.25 s) to clips that do not carry one: older clip
folders and files added by hand. Clips are never faded twice. The video stays
a hard cut unless ``transition="fade"``.

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
from fractions import Fraction
from pathlib import Path

from cutpoint import jsonout
from cutpoint.encode import (
    A_FMT,
    AUDIO_OUT,
    PEAK_PRESETS,
    PEAK_TAG,
    SILENCE,
    audio_fade_filter,
    limiter_chain,
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
    has_soxr,
    measure_loudness,
    output_dir,
    probe,
    run,
)
from cutpoint.timeline import write_timelines
from cutpoint.transcribe import Segment, Transcriber, flat_words
from cutpoint.utils import clip_files

logger = logging.getLogger(__name__)


def list_clips(clips_dir: Path, exclude: set[Path]) -> list[Path]:
    """All .mp4 files directly in ``clips_dir`` in natural filename order."""
    return [p for p in clip_files(clips_dir) if p.resolve() not in exclude]


def baked_fades(clips_dir: Path, files: list[Path]) -> list[float]:
    """Audio fade already baked into each file, per its folder's manifest.json (0 if none)."""
    from cutpoint import cache  # noqa: PLC0415

    manifest = cache.load_json(clips_dir / "manifest.json") or {}
    by_file = {e.get("file"): e for e in manifest.get("clips", [])}
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
    says so, or, for a file without one (renamed or added during review), its own tag. The
    entry wins: ffmpeg copies a tagged source's comment into a clip exported without it."""
    from cutpoint import cache  # noqa: PLC0415

    manifest = cache.load_json(clips_dir / "manifest.json") or {}
    by_file = {e.get("file"): e for e in manifest.get("clips", [])}
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
    the mean loudness but get no filter. Track 0 is measured, as :func:`_audio_chain` uses.
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
    logger.info(
        "Compress peaks (%s): compressor from %.1f dB RMS; clip gains %s; %d baked",
        preset,
        plan.threshold,
        " ".join("-" if g is None else f"{g:+.1f}" for g in plan.gains),
        sum(baked),
    )
    return [plan.filter(i) for i in range(len(files))]


def _audio_chain(
    i: int, info: MediaInfo, fade: float, label: int | None = None, peaks: str = ""
) -> str:
    """``[a<label>]`` (default ``label = i``) from input ``i``: common format, ``peaks``
    (:func:`peak_filters`), edge fade.

    Only track 0 is used; a multi-track input whose track 0 is shorter than its longest
    track (``audio_duration``) keeps the old fade-out, after track 0 ends.
    """
    label = i if label is None else label
    if info.audio_streams:
        return (
            f"[{i}:a:0]{A_FMT},asetpts=PTS-STARTPTS{peaks}"
            f"{audio_fade_filter(info.duration, fade, info.audio_duration)}[a{label}]"
        )
    return f"{SILENCE},atrim=duration={info.vduration:.3f}[a{label}]"


def _normalize_inputs(
    infos: list[MediaInfo], fades: list[float] | None = None, peaks: list[str] | None = None
) -> tuple[list[str], Fraction]:
    """Per-input filter chains producing [v<i>]/[a<i>] with identical params."""
    w, h, fps = infos[0].width, infos[0].height, infos[0].fps
    fades = fades or [0.0] * len(infos)
    peaks = peaks or [""] * len(infos)
    chains = []
    for i, info in enumerate(infos):
        chains.append(
            f"[{i}:v:0]scale={w}:{h}:force_original_aspect_ratio=decrease:out_range=tv,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p,"
            f"settb=AVTB,setpts=PTS-STARTPTS[v{i}]"
        )
        chains.append(_audio_chain(i, info, fades[i], peaks=peaks[i]))
    return chains, fps


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
    """Lossless join with the concat demuxer (inputs must share parameters).

    With ``fades`` (and ``infos``), the video is still stream-copied but the
    audio is re-encoded with an edge fade on each clip whose fade is > 0, and
    with the ``peaks`` filters (:func:`peak_filters`) when any is set.
    """
    listing = out.with_name(f".{out.stem}.concat.txt")
    lines = ["file '" + f.resolve().as_posix().replace("'", "'\\''") + "'" for f in files]
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing)]
    if infos and ((fades and any(f > 0 for f in fades)) or (peaks and any(peaks))):
        fades = fades or [0.0] * len(files)
        peaks = peaks or [""] * len(files)
        for f in files:
            args += ["-i", str(f)]
        chains = []
        for i, (info, fade) in enumerate(zip(infos, fades, strict=True)):
            # Each clip's audio is padded/trimmed to where the demuxer starts the
            # next clip (cut_duration) so the stream-copied video and the
            # re-built audio cannot drift apart.
            chains.append(
                _audio_chain(i + 1, info, fade, label=i, peaks=peaks[i]).replace(
                    f"[a{i}]",
                    f",apad,atrim=end={cut_duration(info):.6f}[a{i}]"
                    if info.audio_streams
                    else f"[a{i}]",
                )
            )
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
    """Join clips with differing parameters by normalising and re-encoding."""
    chains, _ = _normalize_inputs(infos, fades, peaks)
    pairs = "".join(f"[v{i}][a{i}]" for i in range(len(files)))
    _encode(files, ";".join(chains) + f";{pairs}concat=n={len(files)}:v=1:a=1[v][a]", out)


def cut_duration(info: MediaInfo) -> float:
    """How long a clip lasts on a cut join: its container ``duration``.

    The concat demuxer starts the next file there, longer than the video when the
    audio runs on. The cut offsets (:func:`clip_offsets`, so final.srt) and the
    audio rebuilt by :func:`concat_copy` use it, so they stay on the video."""
    return info.duration


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


def crossfade(
    files: list[Path],
    infos: list[MediaInfo],
    out: Path,
    fade: float,
    peaks: list[str] | None = None,
) -> float:
    """Join clips with xfade/acrossfade over ``fade`` s, as returned by
    :func:`crossfade_duration` (> 0), with the ``peaks`` filters (:func:`peak_filters`).
    Returns ``fade``.

    Times are written to the microsecond, not the millisecond (8 frames at 24 fps are
    0.333333 s), so acrossfade matches the frame-rounded fade. xfade starts at the first
    frame at or after ``offset``, and the frames of a previous xfade can sit a microsecond
    early (rounded timestamps add up), so each offset is passed half a frame early: the fade
    still starts on the frame :func:`clip_offsets` puts it on.

    Each clip's audio is padded/trimmed to its video (``vduration``), which xfade and
    :func:`clip_offsets` follow, so audio that runs on cannot push the later clips late."""
    chains, fps = _normalize_inputs(infos, peaks=peaks)
    for i, info in enumerate(infos):
        chains.append(f"[a{i}]apad,atrim=end={info.vduration:.6f},asetpts=PTS-STARTPTS[at{i}]")
    half_frame = 0.5 / float(fps) if fps > 0 else 0.0
    vprev, aprev = "v0", "at0"
    offsets = clip_offsets(infos, fade)
    for i in range(1, len(files)):
        offset = max(offsets[i] - half_frame, 0.0)
        vout, aout = ("v", "a") if i == len(files) - 1 else (f"vx{i}", f"ax{i}")
        chains.append(
            f"[{vprev}][v{i}]xfade=transition=fade:duration={fade:.6f}:offset={offset:.6f}[{vout}]"
        )
        chains.append(f"[{aprev}][at{i}]acrossfade=d={fade:.6f}[{aout}]")
        vprev, aprev = vout, aout
    _encode(files, ";".join(chains), out)
    return fade


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


# loudnorm alone did not guarantee the ceiling: its linear mode only holds if
# the measured peak allows it, and the AAC encode afterwards adds true peak
# (one gunshot took a montage to +1.6 dBTP). So loudnorm gets a relaxed TP
# target (keeps the gain linear and close to -14 LUFS more often), followed by
# a limiter running at 192 kHz (4x oversampled, so inter-sample peaks are
# caught). Measured on the R6 montage: the limiter output is exact (-2.2 dBTP
# after resampling to 48 kHz), AAC at 192k then adds +1.2 dB but at 320k only
# +0.2 dB, so the normalised final is encoded at 320k. The result is measured
# after encoding; if the true peak is still above TP_CEILING, the limiter
# ceiling is lowered and the pass rerun.
LOUDNORM = "loudnorm=I=-14:TP=-1.0:LRA=11"
TP_CEILING = -1.5  # dBTP the final file must not exceed
LIMIT_START_DB = -2.3  # first limiter ceiling (dBFS at 192 kHz, before AAC)
FINAL_AUDIO_OUT = ["-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2"]
LIMIT_MARGIN_DB = 0.3  # extra headroom when a pass has to be redone
LIMIT_PASSES = 3


def loudnorm_filter(src: Path) -> str:
    """Two-pass loudnorm: measure ``src``, return a linear filter hitting -14 LUFS."""
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
        return (
            f"{LOUDNORM}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
            f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
            f"offset={m['target_offset']}:linear=true"
        )
    except (ValueError, KeyError):
        logger.warning("loudnorm measurement unparseable; using single-pass")
        return LOUDNORM


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
) -> Path | None:
    """Apply loudness normalisation (+ true-peak limiter) and/or burned-in captions.

    ``limit`` without ``loudnorm`` runs the limiter alone (compressed peaks, see
    :func:`peak_filters`). ffmpeg writes ``dst`` (default ``out``); the caption files are
    named after ``out``.

    Returns:
        The caption ASS file (reuse it via ``ass`` when re-running), if any.
    """
    args = ["ffmpeg", "-y", "-v", "error", "-i", str(stage.resolve())]
    cwd, vf = None, None
    if captions:
        if ass is None:
            info = probe(stage)
            wav = out.with_name(f".{out.stem}.captions.wav")
            extract_audio(stage, wav, info)
            tr = Transcriber(model=model)
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
    if loudnorm:
        a_args = ["-af", f"{loudnorm_filter(stage)},{limiter_filter(limit_db)}", *FINAL_AUDIO_OUT]
    elif limit:
        a_args = ["-af", limiter_filter(limit_db), *FINAL_AUDIO_OUT]
    else:
        a_args = ["-c:a", "copy"]
    tail = [*a_args, "-movflags", "+faststart", str((dst or out).resolve())]
    if vf is None:
        run([*args, "-c:v", "copy", *tail])
    else:
        encode_with_fallback(
            lambda enc: run([*args, "-vf", enc.chain(vf), *enc.args(), *tail], cwd=cwd), out.name
        )
    return ass


def normalize_with_ceiling(
    stage: Path,
    out: Path,
    captions: bool,
    model: str | None,
    dst: Path | None = None,
    loudnorm: bool = True,
) -> tuple[float, float]:
    """Loudnorm (unless not ``loudnorm``) + limiter, re-run with a lower ceiling until true
    peak <= TP_CEILING.

    ``dst`` as for :func:`post_process`.

    Returns:
        Measured (integrated LUFS, true peak dBTP) of the final file.
    """
    limit_db, ass = LIMIT_START_DB, None
    for attempt in range(1, LIMIT_PASSES + 1):
        ass = post_process(
            stage, out, loudnorm, captions, model, limit_db, ass, dst, limit=not loudnorm
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
            return lufs, tp
        limit_db -= tp - TP_CEILING + LIMIT_MARGIN_DB
    logger.warning(
        "True peak %+.2f dBTP is still above %.1f after %d passes", tp, TP_CEILING, LIMIT_PASSES
    )
    return lufs, tp


def _join(
    files: list[Path],
    infos: list[MediaInfo],
    clips_dir: Path,
    out: Path,
    transition: str,
    fade_dur: float,
    audio_fade: float,
    peaks: list[str] | None = None,
) -> float:
    """Join ``files`` into ``out`` with cuts or crossfades (see :func:`assemble`); ``peaks``
    are the per-clip :func:`peak_filters`.

    Returns:
        The crossfade duration used (0 for cuts), for :func:`clip_offsets`.
    """
    if transition == "fade" and len(files) > 1:
        fade_dur = crossfade_duration(infos, fade_dur)
        if fade_dur == 0:
            logger.warning(
                "One frame is more than 45% of the shortest clip: no room for a crossfade; "
                "joining with cuts"
            )
            transition = "cut"
    if len(files) == 1 or transition == "cut":
        baked = baked_fades(clips_dir, files)
        fades = [0.0 if b > 0 else max(0.0, audio_fade) for b in baked]
        n_add = sum(f > 0 for f in fades)
        logger.info(
            "Audio edge fades: %d clip(s) already faded (baked), %s",
            sum(b > 0 for b in baked),
            f"adding {audio_fade:g}s to {n_add}" if n_add else "adding none",
        )
        uniform = len({i.concat_key() for i in infos}) == 1 and infos[0].audio_streams == 1
        if uniform:
            concat_copy(files, out, infos, fades, peaks)
        else:
            logger.info("Clip parameters differ; re-encoding instead of stream copy")
            concat_reencode(files, infos, out, fades, peaks)
        return 0.0
    if transition == "fade":
        used = crossfade(files, infos, out, fade_dur, peaks)
        logger.info("Crossfade duration %.3fs", used)
        return used
    raise ValueError(f"Unknown transition {transition!r}")


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
) -> Path:
    """Assemble ``clips_dir`` into ``out`` and write .otio/.fcpxml timelines next to it.

    Subtitles ``<out>.srt`` are written from the clips' cached source transcripts
    (see :func:`cutpoint.srt.write_final_srt`), timed to the joined video.

    ``audio_fade``: audio fade in/out (s) added to each clip that has no fade
    baked in by ``cutpoint clips`` / ``cutpoint extend`` (0 = add none). Ignored for
    ``transition="fade"``, which crossfades the audio anyway, unless the join falls back to
    cuts (no room for a one-frame crossfade).

    ``compress_peaks``: ``off``, ``light`` or ``strong`` (:data:`~cutpoint.encode.PEAK_PRESETS`)
    compresses loud peaks and levels the clips that do not have it baked in; the final audio
    then always ends with the true-peak limiter (with or without ``loudnorm``).

    Raises:
        FileNotFoundError: If the folder has no .mp4 clips.
    """
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
        logger.info(
            "Assembling %d clips (%s): %s", len(files), transition, ", ".join(f.name for f in files)
        )

        peaks = peak_filters(clips_dir, files, infos, compress_peaks)
        limit = loudnorm or compress_peaks != "off"
        # Written to a temporary name: final.mp4 only appears once complete.
        with atomic_output(out) as part:
            stage = out.with_name(f".{out.stem}.stage.mp4") if (limit or captions) else part
            try:
                used_fade = _join(
                    files, infos, clips_dir, stage, transition, fade_dur, audio_fade, peaks
                )
                if limit:
                    lufs, tp = normalize_with_ceiling(
                        stage, out, captions, caption_model, part, loudnorm
                    )
                    logger.info("Final audio: %.1f LUFS integrated, true peak %+.2f dBTP", lufs, tp)
                elif captions:
                    post_process(stage, out, False, captions, caption_model, dst=part)
            finally:
                if stage != part:
                    stage.unlink(missing_ok=True)

    fcp, oti = write_timelines(infos, out, used_fade)
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
