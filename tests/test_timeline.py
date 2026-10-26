"""FCPXML and OTIO timelines: hard cuts and start-on-edit crossfades (#246)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import pytest

from conftest import media_info
from cutpoint.assemble import clip_offsets
from cutpoint.ffmpeg_utils import MediaInfo
from cutpoint.timeline import write_fcpxml, write_otio, write_timelines

FPS = Fraction(30)
FADE = 0.5  # 15 frames
XF = 15


def _info(i: int, audio: bool = True, vdur: float = 4.0, dur: float | None = None) -> MediaInfo:
    return media_info(
        Path(f"/media/c{i}.mp4"),
        vdur + 0.1 if dur is None else dur,
        vduration=vdur,
        fps=FPS,
        audio_streams=1 if audio else 0,
        acodec="aac" if audio else None,
        sample_rate=48000 if audio else None,
        channels=2 if audio else None,
    )


def _clips(pattern: str = "111") -> list[MediaInfo]:
    # Different lengths so offsets differ per clip.
    return [_info(i, p == "1", vdur=4.0 + i) for i, p in enumerate(pattern)]


def _read_otio(path: Path):
    otio = pytest.importorskip("opentimelineio")
    return otio.adapters.read_from_file(str(path))


def _layout(track):
    """(kind, start frame, duration frames) per child, in track order."""
    out = []
    for child in track:
        kind = type(child).__name__
        if kind == "Transition":
            out.append((kind, None, child.out_offset.value))
        else:
            r = track.range_of_child(child)
            out.append((kind, int(r.start_time.value), int(r.duration.value)))
    return out


def test_otio_fade_starts_on_the_edit_and_total_is_shortened(tmp_path):
    clips = _clips()
    tl = _read_otio(write_otio(clips, tmp_path / "a.otio", "a", FADE))
    v1, a1 = tl.tracks
    starts = [round(o * 30) for o in clip_offsets(clips, FADE)]
    for track in (v1, a1):
        layout = _layout(track)
        assert [s for k, s, _ in layout if k == "Clip"] == starts
        assert [d for k, _, d in layout if k == "Transition"] == [XF, XF]
        assert sum(1 for k, _, _ in layout if k == "Transition") == 2
    total = sum(round(c.vduration * 30) for c in clips) - 2 * XF
    assert v1.duration().value == a1.duration().value == total
    # outgoing clips are trimmed by the fade, the last one is whole
    durs = [d for k, _, d in _layout(v1) if k == "Clip"]
    assert durs == [120 - XF, 150 - XF, 180]


def test_fcpxml_fade_has_cross_dissolves_at_the_cuts_and_matching_duration(tmp_path):
    clips = _clips("101")  # middle clip has no audio
    path = write_fcpxml(clips, tmp_path / "a.fcpxml", "a", FADE)
    root = ET.parse(path).getroot()
    spine = root.find(".//spine")
    assert spine is not None
    kinds = [e.tag for e in spine]
    assert kinds == ["asset-clip", "transition", "asset-clip", "transition", "asset-clip"]
    starts = [round(o * 30) for o in clip_offsets(clips, FADE)]
    for tr, start in zip(spine.findall("transition"), starts[1:], strict=True):
        assert tr.get("name") == "Cross Dissolve"
        assert tr.get("offset") == f"{start}/30s"
        assert tr.get("duration") == f"{XF}/30s"
        assert tr.find("filter-audio") is None  # a neighbour has no audio
    assert [c.get("offset") for c in spine.findall("asset-clip")] == [
        f"{s}/30s" if s else "0s" for s in starts
    ]
    total = 120 + 150 + 180 - 2 * XF
    assert root.find(".//sequence").get("duration") == f"{total}/30s"

    both = write_fcpxml(_clips("11"), tmp_path / "b.fcpxml", "b", FADE)
    tr = ET.parse(both).getroot().find(".//transition")
    assert tr is not None and tr.find("filter-audio") is not None


@pytest.mark.parametrize("fade", [0.0, FADE])
def test_otio_clip_without_audio_is_a_gap_in_line_with_video(tmp_path, fade):
    """A clip without audio was left out of A1, so later audio started early with cuts too (#246)."""
    clips = _clips("1011")
    tl = _read_otio(write_otio(clips, tmp_path / "a.otio", "a", fade))
    v1, a1 = tl.tracks
    vl, al = _layout(v1), _layout(a1)
    xf = XF if fade else 0
    v_clips = [(s, d) for k, s, d in vl if k == "Clip"]
    a_items = [(k, s, d) for k, s, d in al if k in ("Clip", "Gap")]
    assert [(s, d) for _, s, d in a_items] == v_clips
    assert [k for k, _, _ in a_items] == ["Clip", "Gap", "Clip", "Clip"]
    assert a1.duration().value == v1.duration().value
    if fade:
        starts = [round(o * 30) for o in clip_offsets(clips, fade)]
        assert [s for s, _ in v_clips] == starts
    # a dissolve only between two audio clips, so never beside the gap
    kinds = [k for k, _, _ in al]
    assert kinds.count("Transition") == (1 if xf else 0)
    for i, k in enumerate(kinds):
        if k == "Transition":
            assert kinds[i - 1] == "Clip" and kinds[i + 1] == "Clip"
            assert i - 1 > 1  # after the gap, between clips 2 and 3


def test_write_timelines_without_fade_stays_hard_cuts(tmp_path):
    clips = _clips()
    fcp, oti = write_timelines(clips, tmp_path / "final.mp4")
    root = ET.parse(fcp).getroot()
    assert root.find(".//transition") is None
    assert root.find(".//effect") is None
    # cuts follow the container duration and sit back to back
    assert [c.get("offset") for c in root.iter("asset-clip")] == [
        "0s",
        f"{round(4.1 * 30)}/30s",
        f"{round(4.1 * 30) + round(5.1 * 30)}/30s",
    ]
    assert (
        root.find(".//sequence").get("duration")
        == f"{round(4.1 * 30) + round(5.1 * 30) + round(6.1 * 30)}/30s"
    )
    tl = _read_otio(oti)
    for track in tl.tracks:
        layout = _layout(track)
        assert [k for k, _, _ in layout] == ["Clip"] * 3
        assert [s for _, s, _ in layout] == [0, 123, 276]


def test_write_timelines_passes_fade_to_both_formats(tmp_path):
    fcp, oti = write_timelines(_clips(), tmp_path / "final.mp4", FADE)
    assert len(ET.parse(fcp).getroot().findall(".//transition")) == 2
    v1, _ = _read_otio(oti).tracks
    assert sum(1 for c in v1 if type(c).__name__ == "Transition") == 2
