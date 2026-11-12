"""Zero-config acceptance (#899): ``analyze`` and ``edit`` with no ``--style``, ``--kind`` or
profile, on one synthetic video per footage category (talk: test_zero_config_talk.py).

Each run must detect the category, succeed as on a lite install, and report the detection in
its ``--json`` envelope without warnings nobody expects. Real ffmpeg on tiny generated files
(``zero_config_fixtures``); the only stub is Whisper (``StubWhisper``: words the test writes;
the speech coverage and the talk decision are the product's). Lite is simulated (no OCR engine,
lite edition) and the model-backed analysis steps are skipped, so no model is needed. The
vision model is off, as in CI without Ollama, so footage the frames leave unsure takes the
documented fallback (``style_ambiguous``, the General style). The real-footage version is a
release check, not part of this suite.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import zero_config_fixtures as fx
from conftest import needs_ffmpeg

# name -> (builder, category, style, ambiguous, decided_by, warning codes); talk footage is in
# test_zero_config_talk.py (its 500 s recording runs on a worker of its own)
CASES = {
    "gameplay": (fx.screen_capture, "gameplay", "gaming", False, ["static HUD"], []),
    "phone": (fx.phone_video, "general", "action", False, ["camera tags"], []),
    "untagged": (fx.untagged_video, "general", "action", True, [], ["style_ambiguous"]),
    # the fake telemetry is no real GPMF: the camera is known, its samples are the stub's
    "action-cam": (fx.action_cam, "action-cam", "action-cam", False, ["telemetry"], []),
}


GPS_SCALE = [10_000_000, 10_000_000, 1000, 1000, 100, 1, 1000, 100, 1]


def _gps_group(sec: int) -> dict:
    """One GPS9 payload (a fix every second, speeding up from 10 to 30 m/s) as the parser
    reports it: scale, timestamp and the packed big-endian records."""
    vals = [47.0 + sec * 1e-4, 11.0, 2000.0, 10.0 + sec, 10.0 + sec, 9000, 3600.0 + sec, 1.5, 3]
    raw = [round(v * k) for v, k in zip(vals, GPS_SCALE, strict=True)]
    body = struct.pack(">lllllllHH", *raw)
    klv = b"GPS9?" + bytes([32]) + struct.pack(">H", 1) + body + bytes(-len(body) % 4)
    return {"GPS9": {"TimestampUs": sec * 1_000_000, "Scale": GPS_SCALE, "0x47505339": klv}}


class FakeParser:
    """The telemetry-parser library (a Rust extension that reads GoPro GPMF) for the action-cam
    fixture, whose telemetry track holds no samples: 20 s of IMU at 50 Hz, steady but for a
    violent shake from 8 s to 10 s, and a GPS fix every second."""

    camera, model = "GoPro", "HERO"

    def __init__(self, path: str) -> None:
        pass

    def telemetry(self) -> list[dict]:
        return [_gps_group(sec) for sec in range(20)]

    def normalized_imu(self) -> list[dict]:
        out = []
        for i in range(1000):
            t = i / 50
            shake = 8 <= t < 10
            sign = 1 if i % 2 else -1
            out.append(
                {
                    "timestamp_ms": t * 1000,
                    "accl": [sign * 40.0, 0.0, 9.81] if shake else [0.0, 0.0, 9.81],
                    "gyro": [0.0, 0.0, sign * 200.0] if shake else [0.0, 0.0, 1.0],
                }
            )
        return out


@pytest.fixture(autouse=True)
def lite_without_models(monkeypatch):
    monkeypatch.setitem(sys.modules, "telemetry_parser", SimpleNamespace(Parser=FakeParser))
    fx.lite_without_models(monkeypatch)


@pytest.fixture(scope="module")
def footage(tmp_path_factory) -> dict[str, Path]:
    """The synthetic videos, made once per module."""
    folder = tmp_path_factory.mktemp("zero-config")
    return {name: case[0](folder / f"{name}.mp4") for name, case in CASES.items()}


@needs_ffmpeg
@pytest.mark.slow
@pytest.mark.parametrize("name", CASES)
def test_footage_is_detected_with_no_style_kind_or_profile(footage, home, capsys, name):
    """``analyze`` and then ``edit`` of the same file, flagless (untagged footage warns
    ``style_ambiguous``, as documented without a vision model)."""
    fx.check_flagless(capsys, footage[name], CASES[name])
