"""Write DaVinci Resolve-importable timelines (FCPXML 1.10 and OpenTimelineIO).

``cutpoint assemble --format fcpxml|otio`` writes one of them instead of final.mp4.
Clips are laid out in the given order with absolute file references. Import in
Resolve via File > Import > Timeline.

With no fade they are hard cuts, back to back, the last clip lasting its video
(``vduration``), where final.mp4 ends. With a crossfade (``assemble --transition fade``)
each join is a cross-dissolve of ``fade`` s that starts on the edit, as in final.mp4: the incoming clip starts at :func:`cutpoint.assemble.clip_offsets`
from its first frame, and the outgoing clip is trimmed by the fade, its last frames
becoming the handle the dissolve plays over. The clips then last their video
(``vduration``), which the crossfade follows, so the timeline is as long as final.mp4.

With hard audio cuts in the crossfades (``assemble --transition fade --audio-transition
cut``) the dissolves carry no audio crossfade. The OTIO timeline cuts the audio at
:func:`cutpoint.assemble.audio_cut_point`, as in final.mp4; the FCPXML one does not mark
where, so the editor places the cut (likely at the start of the dissolve).

With mixed joins (``assemble --joins``) ``fade`` is one value per join: a dissolve at the
joins above 0 and a hard cut at the others, audio included. Every clip then lasts its
video, as in final.mp4 (:func:`cutpoint.assemble.clip_offsets`).
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from cutpoint.ffmpeg_utils import MediaInfo
from cutpoint.folders import atomic_output

logger = logging.getLogger(__name__)


def per_join(fade: float | list[float], clips: int) -> list[float]:
    """The crossfade of each join of ``clips`` clips: ``fade`` for all, or its value per join."""
    return fade if isinstance(fade, list) else [fade] * (clips - 1)


def _uri(path: Path) -> str:
    """Absolute ``file:`` URI that keeps mapped drive letters, not UNC paths."""
    return Path(os.path.abspath(path)).as_uri()  # noqa: PTH100 - normalises '..', keeps drive letters


def _frames(info: MediaInfo, fps: Fraction, fade: int = 0, last: bool = False) -> int:
    """A clip's length in frames: its container ``duration`` on cuts, its video
    (``vduration``) when crossfaded (``fade`` frames > 0), as :func:`cutpoint.assemble.clip_offsets`.
    The ``last`` clip lasts its video on cuts too: final.mp4 ends there
    (:func:`cutpoint.assemble.cut_audio_ends`)."""
    seconds = info.vduration if fade > 0 or last else info.duration
    return max(1, int(seconds * fps + Fraction(1, 2)))


def _fade_frames(fade: float, fps: Fraction) -> int:
    """``fade`` s (whole frames, see :func:`cutpoint.assemble.crossfade_duration`) in frames."""
    return max(0, int(fade * fps + Fraction(1, 2))) if fade > 0 else 0


def _join_frames(fade: float | list[float], clips: int, fps: Fraction) -> list[int]:
    """Each join's crossfade (:func:`per_join`) in frames, 0 for a cut."""
    return [_fade_frames(f, fps) for f in per_join(fade, clips)]


def _t(frames: int, fps: Fraction) -> str:
    """FCPXML rational time string for a frame count."""
    if frames == 0:
        return "0s"
    # Keep the frame timebase (e.g. 406/30s, 1001/60000s) rather than reducing it.
    return f"{frames * fps.denominator}/{fps.numerator}s"


def write_fcpxml(
    clips: list[MediaInfo],
    out: Path,
    name: str,
    fade: float | list[float] = 0.0,
    audio_cut: bool = False,
) -> Path:
    """Write an FCPXML 1.10 project referencing ``clips`` in order.

    ``fade`` > 0 (for all joins, or a list with one value per join) puts a Cross Dissolve
    ``<transition>`` on the spine at the join, starting on the edit (the incoming clip's
    offset) for ``fade`` s. The outgoing ``asset-clip`` is
    trimmed by the fade and the dissolve plays over its tail handle; the asset keeps the
    whole file. When both clips have audio the transition also carries an Audio Crossfade,
    as acrossfade does in final.mp4, unless ``audio_cut`` (hard audio cuts). With
    ``audio_cut`` only the Audio Crossfade is left out: nothing puts the audio cut at
    :func:`cutpoint.assemble.audio_cut_point` as final.mp4 does, so where the audio cuts is
    left to the editor (likely the start of the dissolve). The .otio timeline
    (:func:`write_otio`) matches final.mp4.
    """
    fps = clips[0].fps
    joins = _join_frames(fade, len(clips), fps)
    xf = max(joins, default=0)  # any dissolve: the clips last their video
    first = clips[0]
    root = ET.Element("fcpxml", version="1.10")
    resources = ET.SubElement(root, "resources")
    ET.SubElement(
        resources,
        "format",
        id="r1",
        frameDuration=_t(1, fps),
        width=str(first.width),
        height=str(first.height),
        colorSpace="1-1-1 (Rec. 709)",
    )
    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name=name)
    project = ET.SubElement(event, "project", name=name)
    last = len(clips) - 1
    total = sum(_frames(c, fps, xf, i == last) for i, c in enumerate(clips)) - sum(joins)
    sequence = ET.SubElement(
        project,
        "sequence",
        format="r1",
        duration=_t(total, fps),
        tcStart="0s",
        tcFormat="NDF",
        audioLayout="stereo",
        audioRate="48k",
    )
    spine = ET.SubElement(sequence, "spine")
    dissolve_id, crossfade_id = f"r{len(clips) + 2}", f"r{len(clips) + 3}"
    offset = 0
    for i, clip in enumerate(clips, 2):
        frames = _frames(clip, fps, xf, i - 2 == last)
        tail = joins[i - 2] if i - 2 < last else 0  # the dissolve that follows this clip
        if i > 2 and joins[i - 3] > 0:
            prev = clips[i - 3]
            transition = ET.SubElement(
                spine,
                "transition",
                name="Cross Dissolve",
                offset=_t(offset, fps),
                duration=_t(joins[i - 3], fps),
            )
            ET.SubElement(transition, "filter-video", ref=dissolve_id, name="Cross Dissolve")
            if prev.audio_streams and clip.audio_streams and not audio_cut:
                ET.SubElement(transition, "filter-audio", ref=crossfade_id, name="Audio Crossfade")
        asset_id = f"r{i}"
        asset = ET.SubElement(
            resources,
            "asset",
            id=asset_id,
            name=clip.path.stem,
            start="0s",
            duration=_t(frames, fps),
            hasVideo="1",
            format="r1",
            videoSources="1",
            hasAudio="1" if clip.audio_streams else "0",
            audioSources="1" if clip.audio_streams else "0",
            audioChannels=str(clip.channels or 2),
            audioRate=str(clip.sample_rate or 48000),
        )
        ET.SubElement(asset, "media-rep", kind="original-media", src=_uri(clip.path))
        ET.SubElement(
            spine,
            "asset-clip",
            ref=asset_id,
            name=clip.path.stem,
            offset=_t(offset, fps),
            start="0s",
            duration=_t(frames - tail, fps),
            format="r1",
            tcFormat="NDF",
        )
        offset += frames - tail
    if xf > 0:
        ET.SubElement(
            resources,
            "effect",
            id=dissolve_id,
            name="Cross Dissolve",
            uid="FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265",
        )
        if not audio_cut:
            ET.SubElement(
                resources,
                "effect",
                id=crossfade_id,
                name="Audio Crossfade",
                uid="FFAudioTransition",
            )
    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    with atomic_output(out) as part:
        part.write_text(
            f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n{body}\n',
            encoding="utf-8",
        )
    return out


def write_otio(
    clips: list[MediaInfo],
    out: Path,
    name: str,
    fade: float | list[float] = 0.0,
    audio_cut: bool = False,
) -> Path:
    """Write an OpenTimelineIO timeline with matching video and audio tracks.

    ``fade`` > 0 (for all joins, or a list with one value per join) puts an SMPTE_Dissolve
    ``Transition`` at the join on both tracks (audio only between two clips that have
    audio), starting on the edit: ``in_offset`` 0,
    ``out_offset`` the fade. The outgoing clip's ``source_range`` is trimmed by the fade and
    the dissolve plays over its tail handle (the media's ``available_range`` keeps the whole
    file). A clip without audio is a ``Gap`` on the audio track, as long as the clip is on
    the video track, so the later audio stays in line with the video.

    ``audio_cut`` (hard audio cuts in the crossfades): the audio track has no transitions;
    each join cuts half the fade (whole frames, rounded down, as
    :func:`cutpoint.assemble.audio_cut_point`) into the dissolve, where final.mp4 cuts.
    """
    import opentimelineio as otio  # noqa: PLC0415

    fps = clips[0].fps
    rate = float(fps)
    joins = _join_frames(fade, len(clips), fps)
    xf = max(joins, default=0)  # any dissolve: the clips last their video
    timeline = otio.schema.Timeline(name=name)
    timeline.global_start_time = otio.opentime.RationalTime(0, rate)
    video = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    audio = otio.schema.Track(name="A1", kind=otio.schema.TrackKind.Audio)
    zero = otio.opentime.RationalTime(0, rate)
    for i, clip in enumerate(clips):
        frames = _frames(clip, fps, xf, i == len(clips) - 1)
        head = joins[i - 1] if i > 0 else 0  # the dissolve into this clip
        tail = joins[i] if i < len(clips) - 1 else 0  # the dissolve that follows it
        rng = otio.opentime.TimeRange(zero, otio.opentime.RationalTime(frames, rate))
        used = otio.opentime.TimeRange(zero, otio.opentime.RationalTime(frames - tail, rate))
        heard = used  # the audio's part: with audio cuts, from the cut into each dissolve
        if audio_cut:
            first = head // 2
            last = frames - tail + tail // 2
            heard = otio.opentime.TimeRange(
                otio.opentime.RationalTime(first, rate),
                otio.opentime.RationalTime(last - first, rate),
            )
        if not clip.audio_streams:
            audio.append(otio.schema.Gap(source_range=heard))
        for track in (video, audio) if clip.audio_streams else (video,):
            dissolve = track is video or (clips[i - 1].audio_streams and not audio_cut)
            if head > 0 and dissolve:
                track.append(
                    otio.schema.Transition(
                        name="Cross Dissolve",
                        transition_type=otio.schema.TransitionTypes.SMPTE_Dissolve,
                        in_offset=zero,
                        out_offset=otio.opentime.RationalTime(head, rate),
                    )
                )
            ref = otio.schema.ExternalReference(target_url=_uri(clip.path), available_range=rng)
            track.append(
                otio.schema.Clip(
                    name=clip.path.stem,
                    media_reference=ref,
                    source_range=used if track is video else heard,
                )
            )
    timeline.tracks.append(video)
    timeline.tracks.append(audio)
    with atomic_output(out) as part:  # the part keeps .otio, so the adapter is the same
        otio.adapters.write_to_file(timeline, str(part))
    return out


TIMELINE_FORMATS = ("fcpxml", "otio")  # ``assemble --format`` values that write a timeline


def write_timeline(
    clips: list[MediaInfo],
    output_video: Path,
    kind: str,
    fade: float | list[float] = 0.0,
    audio_cut: bool = False,
) -> Path:
    """Write the ``kind`` timeline (:data:`TIMELINE_FORMATS`), ``<name>.fcpxml`` or
    ``<name>.otio``, for ``output_video`` (its folder and stem; the video itself need not exist).

    ``fade``: the crossfade of the join (:func:`cutpoint.assemble.crossfade_duration`, whole
    frames), 0 for hard cuts, or a list with one such value per join. ``audio_cut``: the audio joins with hard cuts in the crossfades
    (``--audio-transition cut``).
    """
    writers = {"fcpxml": write_fcpxml, "otio": write_otio}
    if kind not in writers:
        raise ValueError(f"unknown timeline format {kind!r}")
    out = writers[kind](
        clips, output_video.with_suffix(f".{kind}"), output_video.stem, fade, audio_cut
    )
    if len({c.fps for c in clips}) > 1:
        logger.warning("Clips have mixed frame rates; the timeline uses %s fps", clips[0].fps)
    return out
