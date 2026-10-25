"""VLM window prompts: the gaming prompt carries the same media blocks as the ski prompt."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from cutpoint.events import vlm_windows as vw
from cutpoint.events.common import WAV_SR, AudioPart, Source


def _source(tmp_path: Path) -> Source:
    wav = tmp_path / "audio.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(WAV_SR)
        w.writeframes((np.ones(WAV_SR * 30, dtype="<i2") * 1000).tobytes())
    return Source(tmp_path / "game.mp4", 30.0, tmp_path, [AudioPart(0.0, wav, 30.0)])


def test_build_messages_uses_media_content(tmp_path: Path):
    src = _source(tmp_path)
    win = vw.Window(src.path.name, 5.0, 20.0, 1.0, ["seed"])
    frames = [b"jpg1", b"jpg2", b"jpg3"]
    lines = [{"name": src.name, "start": 6.0, "end": 8.0, "text": "nice shot"}]
    events = [{"kind": "audio", "t": 7.0, "label": "Laughter", "confidence": 0.9}]
    params = vw.VlmParams()
    content = vw.build_messages(src, win, frames, params, "r6", lines, events)[1]["content"]
    media = vw._media_content(src, win, frames, params, lines, events)
    text = "\n".join(c.get("text", "") for c in media)
    assert "input_audio" in {c["type"] for c in media}
    assert "nice shot" in text and "Laughter" in text
    assert content[1:-1] == media


def test_select_windows_quiet_logs_counts_at_debug_only(tmp_path: Path, caplog):
    src = _source(tmp_path)
    with caplog.at_level("INFO"):
        vw.select_windows([src], [], {}, vw.VlmParams(), quiet=True)
    assert "VLM windows" not in caplog.text
    with caplog.at_level("INFO"):
        vw.select_windows([src], [], {}, vw.VlmParams())
    assert "VLM windows" in caplog.text
