"""Zero-config acceptance (#899) for talk footage: ``analyze`` and ``edit`` with no ``--style``,
``--kind`` or profile on a 500 s synthetic talk recording. Its own module so it runs on another
xdist worker than tests/test_zero_config.py (``--dist loadfile``); see that module for the
setup. Whisper is :class:`zero_config_fixtures.StubWhisper`: the speech coverage and the talk
decision are the product's.
"""

from __future__ import annotations

import pytest

import zero_config_fixtures as fx
from conftest import needs_ffmpeg

TALK = (fx.talk_recording, "talk", "commentary", False, ["speech"], [])


@pytest.fixture(autouse=True)
def lite_without_models(monkeypatch):
    fx.lite_without_models(monkeypatch)


@needs_ffmpeg
@pytest.mark.slow
def test_talk_is_detected_with_no_style_kind_or_profile(tmp_path, home, capsys):
    """``analyze`` and then ``edit`` of a talk recording, flagless."""
    fx.check_flagless(capsys, fx.talk_recording(tmp_path / "talk.mp4"), TALK)
