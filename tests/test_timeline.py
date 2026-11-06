"""FCPXML and OTIO timelines: hard cuts and start-on-edit crossfades (#246)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import pytest

from conftest import media_info
from cutpoint.assemble import clip_offsets
from cutpoint.ffmpeg_utils import MediaInfo
from cutpoint.timeline import write_fcpxml, write_otio, write_timeline

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


def _both(clips, video, fade=0.0, audio_cut=False):
    """``write_timeline`` of both formats for ``video``."""
    return tuple(write_timeline(clips, video, k, fade, audio_cut) for k in ("fcpxml", "otio"))


def test_write_timeline_without_fade_stays_hard_cuts(tmp_path):
    clips = _clips()
    fcp, oti = _both(clips, tmp_path / "final.mp4")
    assert (fcp.name, oti.name) == ("final.fcpxml", "final.otio")
    root = ET.parse(fcp).getroot()
    assert root.find(".//transition") is None
    assert root.find(".//effect") is None
    # cuts follow the container duration and sit back to back; the last clip lasts its
    # video, where final.mp4 ends (cut_audio_ends)
    assert [c.get("offset") for c in root.iter("asset-clip")] == [
        "0s",
        f"{round(4.1 * 30)}/30s",
        f"{round(4.1 * 30) + round(5.1 * 30)}/30s",
    ]
    assert (
        root.find(".//sequence").get("duration")
        == f"{round(4.1 * 30) + round(5.1 * 30) + round(6.0 * 30)}/30s"
    )
    assert list(root.iter("asset-clip"))[-1].get("duration") == f"{round(6.0 * 30)}/30s"
    tl = _read_otio(oti)
    for track in tl.tracks:
        layout = _layout(track)
        assert [k for k, _, _ in layout] == ["Clip"] * 3
        assert [s for _, s, _ in layout] == [0, 123, 276]
        assert layout[-1][2] == round(6.0 * 30)


def test_fcpxml_audio_cut_dissolves_carry_no_audio_crossfade(tmp_path):
    path = write_fcpxml(_clips("111"), tmp_path / "a.fcpxml", "a", FADE, audio_cut=True)
    text = path.read_text(encoding="utf-8")
    transitions = ET.parse(path).getroot().findall(".//transition")
    assert len(transitions) == 2
    for tr in transitions:
        assert tr.find("filter-video") is not None  # the picture still dissolves
        assert tr.find("filter-audio") is None
    assert "Audio Crossfade" not in text and "FFAudioTransition" not in text


def test_otio_audio_cut_hard_cuts_at_the_same_frame_with_equal_track_lengths(tmp_path):
    clips = _clips()
    tl = _read_otio(write_otio(clips, tmp_path / "a.otio", "a", FADE, audio_cut=True))
    v1, a1 = tl.tracks
    starts = [round(o * 30) for o in clip_offsets(clips, FADE)]
    assert sum(1 for k, _, _ in _layout(v1) if k == "Transition") == 2
    audio = _layout(a1)
    assert not any(k == "Transition" for k, _, _ in audio)  # hard cuts
    assert v1.duration().value == a1.duration().value  # audio frames == video frames
    # each incoming audio clip starts half the fade (7 whole frames of 15) into its dissolve
    cut = XF // 2
    assert [s for k, s, _ in audio if k == "Clip"] == [starts[0], starts[1] + cut, starts[2] + cut]
    heard = [c.source_range.start_time.value for c in a1 if type(c).__name__ == "Clip"]
    assert heard == [0, cut, cut]


def test_otio_audio_cut_keeps_a_clip_without_audio_as_a_gap_in_line(tmp_path):
    clips = _clips("101")
    tl = _read_otio(write_otio(clips, tmp_path / "a.otio", "a", FADE, audio_cut=True))
    v1, a1 = tl.tracks
    assert v1.duration().value == a1.duration().value
    assert [type(c).__name__ for c in a1] == ["Clip", "Gap", "Clip"]


@pytest.mark.parametrize("fmt", ["fcpxml", "otio"])
def test_timeline_write_is_atomic_when_it_fails_midway(tmp_path, monkeypatch, fmt):
    """A failed write keeps the old timeline and leaves no ``.final.part.*`` file (#600)."""
    out = tmp_path / f"final.{fmt}"
    out.write_text("OLD", encoding="utf-8")

    real_write_text = Path.write_text

    def partial_write(target, *args, **kwargs):
        real_write_text(Path(target), "PARTIAL", encoding="utf-8")
        raise OSError("disk full")

    if fmt == "fcpxml":
        monkeypatch.setattr(Path, "write_text", lambda self, *a, **k: partial_write(self))
        write = write_fcpxml
    else:
        otio = pytest.importorskip("opentimelineio")
        monkeypatch.setattr(
            otio.adapters, "write_to_file", lambda tl, name, *a, **k: partial_write(name)
        )
        write = write_otio
    with pytest.raises(OSError, match="disk full"):
        write(_clips(), out, "final", FADE)
    monkeypatch.undo()
    assert out.read_text(encoding="utf-8") == "OLD"
    assert [p.name for p in tmp_path.iterdir()] == [out.name]


# ---------------------------------------------------------------- mixed joins (#768)
MIXED = [FADE, 0.0]  # a dissolve into clip 2, a hard cut into clip 3


def test_fcpxml_mixed_joins_dissolve_only_where_asked(tmp_path):
    clips = _clips()
    starts = [round(o * 30) for o in clip_offsets(clips, MIXED)]
    assert starts == [0, 120 - XF, 120 - XF + 150]  # every clip lasts its video
    root = ET.parse(write_fcpxml(clips, tmp_path / "a.fcpxml", "a", MIXED)).getroot()
    spine = root.find(".//spine")
    assert spine is not None
    assert [e.tag for e in spine] == ["asset-clip", "transition", "asset-clip", "asset-clip"]
    tr = spine.find("transition")
    assert tr is not None
    assert (tr.get("offset"), tr.get("duration")) == (f"{starts[1]}/30s", f"{XF}/30s")
    assert tr.find("filter-audio") is not None
    assert [c.get("offset") for c in spine.findall("asset-clip")] == [
        f"{s}/30s" if s else "0s" for s in starts
    ]
    assert [c.get("duration") for c in spine.findall("asset-clip")] == [
        f"{120 - XF}/30s", "150/30s", "180/30s",
    ]  # fmt: skip
    assert root.find(".//sequence").get("duration") == f"{120 + 150 + 180 - XF}/30s"
    cut = write_fcpxml(clips, tmp_path / "b.fcpxml", "b", MIXED, audio_cut=True)
    tr = ET.parse(cut).getroot().find(".//transition")
    assert tr is not None and tr.find("filter-audio") is None


@pytest.mark.parametrize("audio_cut", [False, True])
def test_otio_mixed_joins_dissolve_only_where_asked(tmp_path, audio_cut):
    clips = _clips()
    tl = _read_otio(write_otio(clips, tmp_path / "a.otio", "a", MIXED, audio_cut=audio_cut))
    v1, a1 = tl.tracks
    video = _layout(v1)
    assert [(k, d) for k, _, d in video if k == "Transition"] == [("Transition", XF)]
    assert [(s, d) for k, s, d in video if k == "Clip"] == [(0, 105), (105, 150), (255, 180)]
    assert v1.duration().value == a1.duration().value == 435
    audio = _layout(a1)
    if audio_cut:
        assert not any(k == "Transition" for k, _, _ in audio)
        # the cut into the dissolve only at the dissolving join; the hard cut keeps its frame
        assert [(s, d) for k, s, d in audio if k == "Clip"] == [
            (0, 112),
            (105 + 7, 143),
            (255, 180),
        ]
    else:
        assert sum(1 for k, _, _ in audio if k == "Transition") == 1
