"""Write DaVinci Resolve-importable timelines (FCPXML 1.10 and OpenTimelineIO).

Clips are laid out back-to-back (hard cuts) in the given order with absolute
file references. Import in Resolve via File > Import > Timeline.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from cutpoint.ffmpeg_utils import MediaInfo

logger = logging.getLogger(__name__)


def _uri(path: Path) -> str:
    """Absolute ``file:`` URI that keeps mapped drive letters, not UNC paths."""
    return Path(os.path.abspath(path)).as_uri()  # noqa: PTH100 - normalises '..', keeps drive letters


def _frames(info: MediaInfo, fps: Fraction) -> int:
    return max(1, int(info.duration * fps + Fraction(1, 2)))


def _t(frames: int, fps: Fraction) -> str:
    """FCPXML rational time string for a frame count."""
    if frames == 0:
        return "0s"
    # Keep the frame timebase (e.g. 406/30s, 1001/60000s) rather than reducing it.
    return f"{frames * fps.denominator}/{fps.numerator}s"


def write_fcpxml(clips: list[MediaInfo], out: Path, name: str) -> Path:
    """Write an FCPXML 1.10 project referencing ``clips`` in order."""
    fps = clips[0].fps
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
    total = sum(_frames(c, fps) for c in clips)
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
    offset = 0
    for i, clip in enumerate(clips, 2):
        frames = _frames(clip, fps)
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
            duration=_t(frames, fps),
            format="r1",
            tcFormat="NDF",
        )
        offset += frames
    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    out.write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n{body}\n', encoding="utf-8"
    )
    return out


def write_otio(clips: list[MediaInfo], out: Path, name: str) -> Path:
    """Write an OpenTimelineIO timeline with matching video and audio tracks."""
    import opentimelineio as otio  # noqa: PLC0415

    fps = clips[0].fps
    rate = float(fps)
    timeline = otio.schema.Timeline(name=name)
    timeline.global_start_time = otio.opentime.RationalTime(0, rate)
    video = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    audio = otio.schema.Track(name="A1", kind=otio.schema.TrackKind.Audio)
    for clip in clips:
        rng = otio.opentime.TimeRange(
            otio.opentime.RationalTime(0, rate),
            otio.opentime.RationalTime(_frames(clip, fps), rate),
        )
        for track in (video, audio) if clip.audio_streams else (video,):
            ref = otio.schema.ExternalReference(target_url=_uri(clip.path), available_range=rng)
            track.append(
                otio.schema.Clip(name=clip.path.stem, media_reference=ref, source_range=rng)
            )
    timeline.tracks.append(video)
    timeline.tracks.append(audio)
    otio.adapters.write_to_file(timeline, str(out))
    return out


def write_timelines(clips: list[MediaInfo], output_video: Path) -> tuple[Path, Path]:
    """Write ``<name>.fcpxml`` and ``<name>.otio`` next to ``output_video``."""
    name = output_video.stem
    fcp = write_fcpxml(clips, output_video.with_suffix(".fcpxml"), name)
    oti = write_otio(clips, output_video.with_suffix(".otio"), name)
    if len({c.fps for c in clips}) > 1:
        logger.warning("Clips have mixed frame rates; timelines use %s fps", clips[0].fps)
    return fcp, oti
