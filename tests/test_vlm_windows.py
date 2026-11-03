"""VLM window prompts: the gaming prompt carries the same media blocks as the ski prompt."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from cutpoint import paths
from cutpoint.events import vlm_windows as vw
from cutpoint.events.common import WAV_SR, AudioPart, Source
from cutpoint.events.regions import load_preset
from hud_fixtures import verbfeed, write_preset


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
    game = vw.game_prompt("boxfeed", None)
    msgs = vw.build_messages(src, win, frames, params, game, lines, events)
    assert [m["role"] for m in msgs] == ["user"]  # no system prompt
    content = msgs[0]["content"]
    text = "\n".join(c.get("text", "") for c in content)
    assert {c["type"] for c in content} == {"text", "image_url"}  # no audio part
    assert "nice shot" in text and "Laughter" in text


def test_select_windows_quiet_logs_counts_at_debug_only(tmp_path: Path, caplog):
    src = _source(tmp_path)
    with caplog.at_level("INFO"):
        vw.select_windows([src], [], {}, vw.VlmParams(), quiet=True)
    assert "VLM windows" not in caplog.text
    with caplog.at_level("INFO"):
        vw.select_windows([src], [], {}, vw.VlmParams())
    assert "VLM windows" in caplog.text


def test_hints_kill_says_whose_kill():
    def hint(**fields):
        ev = {"kind": "ocr", "type": "kill", "verb": "kill", "text": "a > b", "t": 5.0, **fields}
        return vw._hints([ev], 0.0, 10.0)

    assert hint(player_death=True).endswith("a > b (the PLAYER died)")
    assert hint(player_kill=True).endswith("a > b (the PLAYER's kill)")
    assert hint(player_kill=True, player_death=True).endswith(
        "a > b (the PLAYER killed themselves)"
    )
    assert hint().endswith("a > b")


# -- game_prompt: user presets are installed (trusted), project ones are custom --
def _gp(game: str, preset) -> dict:
    return vw.VlmParams().key(vw.game_prompt(game, preset))


def test_gp_user_preset_never_custom(tmp_path, home):
    write_preset(paths.presets_dir(), verbfeed())
    assert "gp" not in _gp("verbfeed", load_preset("verbfeed"))


def test_gp_project_copy_compared_to_user_preset(tmp_path, home):
    write_preset(paths.presets_dir(), verbfeed())
    proj = tmp_path / "proj"
    same = write_preset(proj / "hud_presets", verbfeed())
    assert "gp" not in _gp("verbfeed", load_preset("verbfeed", proj))
    same.unlink()
    write_preset(proj / "hud_presets", {k: v for k, v in verbfeed().items() if k != "hype"})
    assert "gp" in _gp("verbfeed", load_preset("verbfeed", proj))


def test_gp_reference_is_bundled_without_user_preset(tmp_path, home, monkeypatch):
    monkeypatch.setattr(paths, "PRESETS_DIR", tmp_path / "bundled")
    write_preset(paths.PRESETS_DIR, verbfeed())
    proj = tmp_path / "proj"
    write_preset(proj / "hud_presets", verbfeed())
    assert "gp" not in _gp("verbfeed", load_preset("verbfeed", proj))
    write_preset(proj / "hud_presets", verbfeed(view="side-on"))
    assert "gp" in _gp("verbfeed", load_preset("verbfeed", proj))


def test_action_cam_windows_cache_under_the_action_cam_key():
    key = vw.VlmParams(ski=True).key()
    assert "ski" not in key
    assert key["action_cam"] == vw.ACTION_CAM_PROMPT_VERSION
    assert "action_cam" not in vw.VlmParams().key()
