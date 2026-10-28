"""assemble: the joined audio and the container end with the video (#473)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import media_info, needs_ffmpeg
from cutpoint import assemble, sources
from cutpoint.ffmpeg_utils import probe
from test_assemble_fadecopy import SHORT, _export_clips

# -- unit -------------------------------------------------------------------------------


def test_cut_audio_ends_follow_the_cut_and_the_last_video():
    infos = [
        media_info(Path("1.mp4"), 3.02, vduration=3.0),
        media_info(Path("2.mp4"), 4.0),
        media_info(Path("3.mp4"), 5.04, vduration=5.0),
    ]
    assert assemble.cut_audio_ends(infos) == [3.02, 4.0, 5.0]  # last: video, not container


def test_audio_chain_pads_and_trims_to_the_end_only_when_given():
    info = media_info(Path("x.mp4"), 2.0)
    assert ",apad,atrim=end=" not in assemble._audio_chain(0, info, 0.25)
    chain = assemble._audio_chain(0, info, 0.25, end=1.97)
    assert chain.endswith(",apad,atrim=end=1.970000[a0]")
    silent = media_info(Path("x.mp4"), 2.0, audio_streams=0)
    assert "apad" not in assemble._audio_chain(0, silent, 0.25, end=1.97)


def test_concat_copy_trims_the_last_clip_to_its_video_and_keeps_timestamps(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(assemble, "run", lambda args, **_: calls.append(args) or "")
    files = [tmp_path / "1.mp4", tmp_path / "2.mp4"]
    infos = [media_info(files[0], 3.02, vduration=3.0), media_info(files[1], 4.03, vduration=4.0)]
    assemble.concat_copy(files, tmp_path / "o.mp4", infos, [0.0, 0.0], ["", ""])
    args = calls[-1]
    assert args.index("-copyts") < args.index("-f")
    graph = args[args.index("-filter_complex") + 1]
    assert "atrim=end=3.020000[a0]" in graph and "atrim=end=4.000000[a1]" in graph
    assert args[args.index("-c:v") + 1] == "copy"


@pytest.mark.parametrize("limit, loudnorm", [(False, True), (True, False), (True, True)])
def test_post_process_trims_the_audio_to_the_stage_video(monkeypatch, tmp_path, limit, loudnorm):
    calls: list[list[str]] = []
    monkeypatch.setattr(assemble, "run", lambda args, **_: calls.append(args) or "")
    monkeypatch.setattr(assemble, "has_soxr", lambda: True)
    monkeypatch.setattr(assemble, "loudnorm_filter", lambda _s: "loudnorm=x")
    monkeypatch.setattr(assemble, "probe", lambda p: media_info(p, 9.0, vduration=8.9))
    assemble.post_process(
        tmp_path / "s.mp4", tmp_path / "o.mp4", loudnorm, False, None, limit=limit
    )
    af = calls[-1][calls[-1].index("-af") + 1]
    assert af.endswith(",atrim=end=8.900000")
    assert ("loudnorm=x" in af) == loudnorm


def test_post_process_keeps_the_audio_up_to_a_late_video_end(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(assemble, "run", lambda args, **_: calls.append(args) or "")
    monkeypatch.setattr(assemble, "has_soxr", lambda: True)
    info = media_info(tmp_path / "s.mp4", 6.3, vduration=6.0, vstart=0.3)
    monkeypatch.setattr(assemble, "probe", lambda _p: info)
    assemble.post_process(tmp_path / "s.mp4", tmp_path / "o.mp4", False, False, None, limit=True)
    af = calls[-1][calls[-1].index("-af") + 1]
    assert af.endswith(",atrim=end=6.300000")  # where the video ends, not its length


# -- end to end -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> Path:
    return _export_clips(tmp_path_factory.mktemp("av_tail"), "30", SHORT)


def _make_clips(exported: Path, tmp_path: Path, audio_extra: float) -> Path:
    """Exported clips (BT.709, keyframes 0.5 s from both ends: fast-path eligible) whose audio
    is remuxed to run ``audio_extra`` s longer (or shorter, if negative) than the video."""
    clips = tmp_path / "clips"
    shutil.copytree(exported, clips)
    for clip in sorted(clips.glob("*.mp4")):
        tmp = clip.with_name(f".{clip.name}")
        end = probe(clip).vduration + audio_extra
        subprocess.run(
            [shutil.which("ffmpeg"), "-v", "error", "-i", str(clip), "-c:v", "copy", "-af",
             f"apad,atrim=end={end:.6f}", "-c:a", "aac", "-f", "mp4", str(tmp)],
            check=True,
        )  # fmt: skip
        tmp.replace(clip)
    return clips


def _durations(path: Path) -> tuple[float, float, float, float]:
    """(video, audio, container, first video pts) in seconds."""
    out = subprocess.run(
        [shutil.which("ffprobe"), "-v", "error", "-show_entries",
         "stream=codec_type,start_time,duration:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    doc = json.loads(out)
    by = {s["codec_type"]: s for s in doc["streams"]}
    pts = subprocess.run(
        [shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0", "-read_intervals",
         "%+#1", "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.split()  # fmt: skip
    audio = float(by["audio"]["duration"]) if "duration" in by["audio"] else -1.0
    return (
        float(by["video"]["duration"]),
        audio,
        float(doc["format"]["duration"]),
        float(pts[0]),
    )


@needs_ffmpeg
@pytest.mark.parametrize(
    "transition, loudnorm, audio_extra",
    [
        ("cut", False, -0.025),
        ("cut", False, 0.0),
        ("cut", False, 0.03),
        ("cut", True, -0.025),
        ("cut", True, 0.03),
        ("fade", True, -0.025),
        ("fade", True, 0.03),
    ],
)
def test_the_joined_audio_and_container_end_with_the_video(
    tmp_path, exported, monkeypatch, caplog, transition, loudnorm, audio_extra
):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)  # no final.srt
    clips = _make_clips(exported, tmp_path, audio_extra)
    caplog.set_level(logging.DEBUG)
    out = assemble.assemble(clips, transition=transition, fade_dur=0.25, loudnorm=loudnorm)
    video, audio, container, first_pts = _durations(out)
    assert abs(audio - video) <= 0.022
    assert abs(container - video) <= 0.022
    if transition == "cut":
        assert first_pts == pytest.approx(0.0, abs=1e-3)
        # the demuxer starts a clip after its container, so a longer audio leaves a short hold
        assert video == pytest.approx(10.5 + 2 * max(audio_extra, 0.0), abs=0.05)
    else:
        assert "stream-copied the rest" in caplog.text
    assert probe(out).vduration == pytest.approx(video, abs=1e-3)
