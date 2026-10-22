"""GPU video encoding: probing by really encoding, forced choices, capability report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import write
from cutpoint import assemble, capabilities, cli, config, encode, ffmpeg_utils, reset_state
from cutpoint.crop import Crop
from cutpoint.events import setup
from cutpoint.ffmpeg_utils import EncoderUnavailableError, FFmpegError, VideoEncoder

D128, D129 = "/dev/dri/renderD128", "/dev/dri/renderD129"


def _device(args: list[str]) -> str | None:
    """The VAAPI render node an ffmpeg command binds (``-init_hw_device vaapi=NAME:DEVICE``)."""
    if "-init_hw_device" not in args:
        return None
    return args[args.index("-init_hw_device") + 1].split(":", 1)[1]


def fake_ffmpeg(
    monkeypatch, works: set[str], devices=(D128, D129), platform="linux", stderr=None
) -> list[list[str]]:
    """Make ``ffmpeg_utils.run`` succeed only for these codecs / ``codec@device``.

    ``platform`` is what ``sys.platform`` reports; ``stderr`` maps a codec to the ffmpeg output
    its failed probe carries (default: ``"<codec> cannot be opened"``).
    """
    monkeypatch.setattr(sys, "platform", platform)  # the auto order differs on Windows
    calls: list[list[str]] = []

    def run(args, **_kw):
        calls.append(args)
        codec = args[args.index("-c:v") + 1]
        if codec in works or f"{codec}@{_device(args)}" in works:
            return ""
        text = (stderr or {}).get(codec, f"{codec} cannot be opened")
        raise FFmpegError(f"ffmpeg failed (exit 1):\n{text}")

    monkeypatch.setattr(ffmpeg_utils, "run", run)
    monkeypatch.setattr(ffmpeg_utils, "vaapi_devices", lambda: list(devices))
    return calls


@pytest.mark.parametrize("kind", ["nvenc", "amf", "vaapi"])
def test_forced_gpu_encoder_that_cannot_encode_is_an_error_not_a_fallback(monkeypatch, kind):
    fake_ffmpeg(monkeypatch, works={"libx264"})
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", kind)
    with pytest.raises(EncoderUnavailableError, match="cannot be opened"):
        ffmpeg_utils.video_encoder()


# What ffmpeg 7 appends after the real failure; none of it says why the encoder failed.
FFMPEG_TRAILERS = (
    "\n[vost#0:0/enc @ 0x2] Task finished with error code: -22 (Invalid argument)\n"
    "[vost#0:0/enc @ 0x2] Terminating thread with return code -22 (Invalid argument)\n"
    "[out#0/null @ 0x3] Nothing was written into output file, because at least one of its streams "
    "received no packets.\n"
    "Conversion failed!"
)
FFMPEG7_STDERR = (
    "[AVHWDeviceContext @ 0x1] Failed to initialise VAAPI connection: -1 (unknown libva error).\n"
    "[vost#0:0/h264_vaapi @ 0x2] Error while opening encoder - maybe incorrect parameters"
    + FFMPEG_TRAILERS
)


@pytest.mark.parametrize(
    ("kind", "codec", "platform", "stderr", "expected"),
    [
        pytest.param(
            "vaapi",
            "h264_vaapi",
            "linux",
            FFMPEG7_STDERR,
            f"{D128}: [AVHWDeviceContext @ 0x1] Failed to initialise VAAPI connection: -1 "
            "(unknown libva error).",
            id="vaapi",
        ),
        pytest.param(
            "amf",
            "h264_amf",
            "win32",
            "[AMF @ 0x1] DLL amfrt64.dll failed to open" + FFMPEG_TRAILERS,
            "h264_amf: [AMF @ 0x1] DLL amfrt64.dll failed to open",
            id="amf",
        ),
        pytest.param(
            "nvenc",
            "h264_nvenc",
            "linux",
            "[h264_nvenc @ 0x1] Cannot load libcuda.so.1" + FFMPEG_TRAILERS,
            "h264_nvenc: [h264_nvenc @ 0x1] Cannot load libcuda.so.1",
            id="nvenc",
        ),
    ],
)
def test_probe_failure_reports_the_root_cause_line_not_ffmpegs_trailers(
    monkeypatch, kind, codec, platform, stderr, expected
):
    fake_ffmpeg(
        monkeypatch, works=set(), devices=(D128,), platform=platform, stderr={codec: stderr}
    )
    enc, errors = ffmpeg_utils.find_encoder(kind)
    assert enc is None
    assert errors == [expected]


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_forced_vaapi_off_linux_says_linux_only_and_never_runs_ffmpeg(monkeypatch, platform):
    calls = fake_ffmpeg(monkeypatch, works={"h264_vaapi"}, platform=platform)
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "vaapi")
    with pytest.raises(EncoderUnavailableError, match="VAAPI is Linux-only"):
        ffmpeg_utils.video_encoder()
    assert calls == []


@pytest.mark.parametrize(
    ("platform", "order"),
    [
        ("win32", ["h264_nvenc", "h264_amf"]),
        ("linux", ["h264_nvenc", "h264_vaapi", "h264_vaapi", "h264_amf"]),  # 2 render nodes
    ],
)
def test_auto_probes_encoders_in_the_platform_order_then_falls_back_to_cpu(
    monkeypatch, platform, order
):
    calls = fake_ffmpeg(monkeypatch, works=set(), platform=platform)
    assert ffmpeg_utils.video_encoder() == VideoEncoder("cpu")
    assert [c[c.index("-c:v") + 1] for c in calls] == order


def test_forced_vaapi_without_a_render_device_says_so(monkeypatch):
    fake_ffmpeg(monkeypatch, works=set(), devices=())
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "vaapi")
    with pytest.raises(EncoderUnavailableError, match="render device"):
        ffmpeg_utils.video_encoder()


def test_config_json_selects_the_encoder_and_the_environment_overrides_it(home, monkeypatch):
    write(home / "config.json", json.dumps({"video_encoder": "cpu"}))
    reset_state()
    assert ffmpeg_utils.video_encoder() == VideoEncoder("cpu")  # forced: nothing is probed
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "bogus")
    reset_state()
    with pytest.raises(ValueError, match="auto, nvenc, amf, vaapi, cpu"):
        config.get_config()


def test_auto_keeps_nvenc_first_when_it_works(monkeypatch):
    fake_ffmpeg(monkeypatch, works={"h264_nvenc", "h264_vaapi"})
    assert ffmpeg_utils.video_encoder() == VideoEncoder("nvenc")


def test_auto_probes_every_render_node_and_uses_the_one_that_encodes(monkeypatch):
    """The first node (renderD128) can be the one that fails: pick the node that encodes."""
    calls = fake_ffmpeg(monkeypatch, works={f"h264_vaapi@{D129}"})
    assert ffmpeg_utils.video_encoder() == VideoEncoder("vaapi", D129)
    assert [_device(c) for c in calls if _device(c)] == [D128, D129]


def test_capability_report_keeps_vendor_presence_apart_from_usable_encoders(monkeypatch, capsys):
    """An AMD GPU being present does not make an encoder usable, nor transcription GPU-fast."""
    fake_ffmpeg(monkeypatch, works={"libx264"})
    monkeypatch.setattr(capabilities, "gpu_vendors", lambda: {"nvidia": False, "amd": True})
    monkeypatch.setattr(capabilities, "_compiled_encoders", lambda: {"h264_vaapi", "libx264"})
    monkeypatch.setattr("cutpoint.transcribe.cuda_available", lambda: False)
    monkeypatch.setattr(setup, "nvidia_usable", lambda: False)
    assert cli.main(["capabilities", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["gpu_vendors_present"] == {"nvidia": False, "amd": True}
    vaapi = data["video_encoders"]["vaapi"]
    assert (vaapi["listed_by_ffmpeg"], vaapi["usable"]) == (True, False)
    assert "cannot be opened" in vaapi["error"]
    assert data["video_encoders"]["cpu"]["usable"] is True
    assert data["active_video_encoder"]["kind"] == "cpu"
    assert data["transcription"]["device"] == "cpu"
    assert data["llama_cpp"]["setup_would_choose"] == ["vulkan", "cpu"]


def test_capability_report_shows_a_forced_encoder_error_instead_of_failing(monkeypatch):
    fake_ffmpeg(monkeypatch, works={"libx264"})
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "amf")
    monkeypatch.setattr(capabilities, "gpu_vendors", lambda: {"nvidia": False, "amd": True})
    monkeypatch.setattr(capabilities, "_compiled_encoders", set)
    monkeypatch.setattr("cutpoint.transcribe.cuda_available", lambda: False)
    active = capabilities.report()["active_video_encoder"]
    assert active["kind"] is None and active["requested"] == "amf"
    assert "h264_amf" in active["error"]


# -- real VAAPI encodes (skipped without a usable VAAPI device) -------------------------------

ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 320
PlayResY: 240

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Sans,60,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,5,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,HELLO
"""


@pytest.fixture
def vaapi(monkeypatch):
    """Force VAAPI for real; skip when this machine cannot encode with it."""
    if ffmpeg_utils.find_encoder("vaapi")[0] is None:
        pytest.skip("no usable VAAPI device")
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "vaapi")
    reset_state()


def _blue_source_with_red_corner(tmp_path: Path) -> Path:
    """3 s, 640x360 blue with a 320x180 red rectangle in the upper-left corner, no audio."""
    src = tmp_path / "src.mp4"
    ffmpeg_utils.run(
        [
            *("ffmpeg", "-y", "-v", "error", "-f", "lavfi"),
            *("-i", "color=c=blue:s=640x360:d=3:r=30"),
            *("-vf", "drawbox=x=0:y=0:w=320:h=180:color=red:t=fill"),
            *("-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", str(src)),
        ]
    )
    return src


def _first_frame(path: Path, w: int = 320, h: int = 240) -> np.ndarray:
    frames = ffmpeg_utils.decode_raw(
        path, size=(w, h), vf=f"scale={w}:{h}", duration=0.2, hwaccel=None
    )
    return next(iter(frames))[1].astype(int)


def _assert_red_letterboxed(frame: np.ndarray) -> None:
    """Red 320x180 picture centred in 320x240: black bars above and below, no blue."""
    assert frame[5:25].max() < 40 and frame[215:235].max() < 40
    picture = frame[45:195, 20:300]
    assert picture[..., 0].min() > 150 and picture[..., 1:].max() < 90


def test_vaapi_clip_export_and_assemble_keep_crop_scale_pad_and_captions(vaapi, tmp_path):
    src = _blue_source_with_red_corner(tmp_path)
    fmt = encode.OutputFormat(320, 240, 30)
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ("001.mp4", "002.mp4"):
        encode.export_clip(
            src, ffmpeg_utils.probe(src), 0.0, 2.0, clips / name, fmt, False, Crop(320, 180, 0, 0)
        )
    clip = ffmpeg_utils.probe(clips / "001.mp4")
    assert (clip.width, clip.height) == (320, 240) and clip.duration == pytest.approx(2.0, abs=0.2)
    _assert_red_letterboxed(_first_frame(clips / "001.mp4"))

    final = assemble.assemble(clips, tmp_path / "final.mp4", transition="fade", fade_dur=0.5)
    joined = ffmpeg_utils.probe(final)
    assert (joined.width, joined.height) == (320, 240)
    assert joined.duration == pytest.approx(3.5, abs=0.3)
    _assert_red_letterboxed(_first_frame(final))

    ass = write(tmp_path / "captions.ass", ASS)
    captioned = tmp_path / "captioned.mp4"
    assemble.post_process(final, captioned, False, True, None, ass=ass)
    cap = ffmpeg_utils.probe(captioned)
    assert (cap.width, cap.height) == (320, 240)
    assert cap.duration == pytest.approx(joined.duration, abs=0.2)
    assert np.abs(_first_frame(captioned) - _first_frame(final)).max() > 60  # the text is drawn
