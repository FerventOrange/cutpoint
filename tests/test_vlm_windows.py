"""VLM window prompts: the gaming prompt carries the same media blocks as the ski prompt."""

from __future__ import annotations

import json
import re
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


# -- #596: the generic prompt asks about kills only for footage with kills --
COMBAT_WORDS = re.compile(
    r"\b(kill\w*|teamkill\w*|enem\w*|headshot\w*|clutch\w*|weapon\w*|gun\w*|shoot\w*|"
    r"shot|frag\w*|combat|death\w*|die[sd]?|respawn\w*|ammo|sniper\w*|fight\w*|round\w*)\b",
    re.IGNORECASE,
)


def _prompt_text(tmp_path: Path, game: vw.GamePrompt | None) -> str:
    src = _source(tmp_path)
    win = vw.Window(src.path.name, 5.0, 20.0, 1.0, [])
    msgs = vw.build_messages(src, win, [], vw.VlmParams(), game, [], [])
    return "\n".join(c.get("text", "") for c in msgs[0]["content"])


def test_neutral_prompt_has_no_combat_words(tmp_path: Path):
    for game in (None, vw.game_prompt(None, None), vw.game_prompt("peak", None)):
        text = _prompt_text(tmp_path, game)
        assert not COMBAT_WORDS.findall(text), COMBAT_WORDS.findall(text)
        assert vw.HYPE_NEUTRAL in text and vw.FUNNY_NEUTRAL in text
    assert COMBAT_WORDS.findall(_prompt_text(tmp_path, vw.game_prompt(None, None, True)))


def test_combat_prompt_keeps_the_kill_wording(tmp_path: Path):
    text = _prompt_text(tmp_path, vw.game_prompt("some-game", None, combat=True))
    assert vw.DESCRIBE_COMBAT in text and vw.HYPE_COMBAT in text and vw.FUNNY_COMBAT in text
    assert "who kills whom" in text and "a single ordinary kill" in text


def test_has_kills_from_ocr_events_or_preset_kill_words(tmp_path: Path, home):
    ocr = {"kind": "ocr", "t": 1.0, "text": "a > b"}
    assert not vw.has_kills(None, [])
    assert not vw.has_kills(None, [{"kind": "audio", "label": "Laughter", "t": 1.0}])
    assert not vw.has_kills(None, [{**ocr, "type": "banner"}])
    assert vw.has_kills(None, [{**ocr, "type": "kill"}])
    assert vw.has_kills(None, [{**ocr, "type": "multikill"}])
    write_preset(paths.presets_dir(), verbfeed())  # its feed lists kill words
    assert vw.has_kills(load_preset("verbfeed"), [])
    bare = {k: v for k, v in verbfeed(game="bare").items() if k != "feed"}
    write_preset(paths.presets_dir(), bare)
    assert not vw.has_kills(load_preset("bare"), [])


# The key main (1360d43, prompt version 5) made for the default params, before #596.
V5_KEY = {"frames": 14, "h": 606, "transcript": True, "hints": True, "prompt": 5, "model": ""}


def test_only_the_neutral_prompt_is_re_keyed():
    combat = vw.VlmParams().key(vw.game_prompt("some-game", None, combat=True))
    neutral = vw.VlmParams().key(vw.game_prompt("some-game", None))
    # combat: the text of 5, so the key of 5 byte for byte (shooter caches stay valid)
    assert json.dumps(combat) == json.dumps(V5_KEY)
    assert vw.COMBAT_PROMPT_VERSION == 5
    # neutral: new in 6, so sources without kills are described again
    assert neutral == {**V5_KEY, "prompt": 6} and neutral != combat
    assert neutral["prompt"] == vw.PROMPT_VERSION == 6
    assert neutral == vw.VlmParams().key()  # no prompt: the neutral one


def test_action_cam_keys_keep_the_generic_version_they_were_cached_with():
    params = vw.VlmParams(ski=True)
    assert params.key()["prompt"] == vw.ACTION_CAM_BASE_PROMPT == 5
    assert params.key() == {**V5_KEY, "action_cam": vw.ACTION_CAM_PROMPT_VERSION}
    assert params.key(vw.game_prompt(None, None, combat=True)) == params.key()


def test_combat_prompt_from_a_project_preset_keeps_the_custom_hash(tmp_path, home):
    write_preset(paths.presets_dir(), verbfeed())
    proj = tmp_path / "proj"
    write_preset(proj / "hud_presets", verbfeed(view="side-on"))
    gp = vw.game_prompt("verbfeed", load_preset("verbfeed", proj), combat=True)
    assert gp.custom and gp.combat and gp.view == "side-on"
    key = vw.VlmParams().key(gp)
    assert "gp" in key and "combat" not in key and key["prompt"] == 5


def test_events_txt_names_the_vision_model(tmp_path: Path):
    from cutpoint.events import report  # noqa: PLC0415

    def header(settings: dict) -> str:
        _, tpath = report.write_outputs(tmp_path, ["a.mp4"], [], [], [], settings, {}, {})
        return tpath.read_text(encoding="utf-8").splitlines()[2]

    assert header({"vlm": {"model": "qwen3.5:9b"}}) == (
        "# VLM WINDOW = qwen3.5:9b summary of a window (hl = highlight 0-10);"
    )
    assert header({"vlm": {"model": "anthropic:claude-sonnet"}}).startswith(
        "# VLM WINDOW = anthropic:claude-sonnet summary"
    )
    assert header({}) == "# VLM WINDOW = Vision model summary of a window (hl = highlight 0-10);"
    assert "Omni" not in header({})
