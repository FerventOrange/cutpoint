"""Small robustness fixes of audit #1085 (A140, A141, A183, A184, A230, A231, A232)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from conftest import media_info
from cutpoint import assemble, cli, crash, ffmpeg_utils, telemetry
from cutpoint.errors import FFmpegError
from cutpoint.events import regions
from cutpoint.telemetry import DT, TelemetryParams


def _preset(data: dict) -> regions.HudPreset:
    return regions.preset_from_dict(data, Path("p.json"))


def test_malformed_region_entries_are_skipped_with_a_warning(caplog):
    """A141: a region without ``y`` or a non-dict region no longer crashes preset loading."""
    good = {"name": "feed", "kind": "feed", "x": 10, "y": 10, "w": 100, "h": 50}
    data = {"game": "a141", "regions": [good, {"name": "bad", "x": 1}, "text", 5]}
    with caplog.at_level(logging.WARNING, logger="cutpoint"):
        preset = _preset(data)
    assert [r.name for r in preset.regions] == ["feed"]
    assert "dropped" in caplog.text


def test_string_match_and_player_tags_are_one_item_not_characters(tmp_path, monkeypatch):
    """A140: ``"player_tags": "Kip"`` was ('K', 'i', 'p'); ``"match": "siege"`` hit 'i' etc."""
    assert _preset({"game": "g", "player_tags": "Kip"}).player_tags == ("Kip",)
    presets = tmp_path / "presets"
    presets.mkdir()
    (presets / "g.json").write_text(json.dumps({"game": "g", "match": "siege"}), "utf-8")
    monkeypatch.setattr(
        regions, "preset_files", lambda project=None: sorted(presets.glob("*.json"))
    )
    assert regions.game_key_for_path("minecraft.mp4") != "g"
    assert regions.game_key_for_path("siege night.mp4") == "g"


def test_a_perfectly_still_camera_after_the_impact_is_stopped():
    """A184: a median vibration of exactly 0.0 counted as missing (``or 1``)."""
    fs = 100.0
    t = np.arange(int(20 * fs)) / fs
    am = np.ones_like(t)
    am[(t >= 10.0) & (t < 10.03)] = 6.0
    grav = np.tile([0.0, 0.0, 1.0], (len(t), 1))
    grav[t > 10.5] = [1.0, 0.0, 0.0]
    bins = np.arange(200) * DT
    vib = np.where(bins < 10.0, 0.3, 0.0)
    speed = np.full(200, np.nan)
    [event] = telemetry._crash_events(t, am, grav, speed, vib, TelemetryParams())
    assert event["type"] == "crash"


def test_decode_window_without_any_frame_returns_empty_arrays(tmp_path, monkeypatch):
    """A183: ``np.stack([])`` raised ValueError when ffmpeg gave no frame and no error."""
    monkeypatch.setattr(ffmpeg_utils, "probe", lambda p: media_info(Path(p), 10.0))
    monkeypatch.setattr(ffmpeg_utils, "decode_raw", lambda *a, **k: iter(()))
    times, frames = crash.decode_window(tmp_path / "v.mp4", 0.0, 2.0, 5.0, width=64)
    assert times.shape == (0,) and frames.shape[0] == 0
    monkeypatch.setattr(
        ffmpeg_utils, "decode_raw", lambda *a, **k: (_ for _ in ()).throw(FFmpegError("x"))
    )
    again = crash.decode_window(tmp_path / "v.mp4", 0.0, 2.0, 5.0, width=64)
    assert again[0].shape == (0,) and again[1].shape == frames.shape


@pytest.mark.parametrize("err", [KeyError("k"), IndexError("i"), AttributeError("a"), KeyError()])
def test_lookup_and_attribute_errors_are_a_one_line_error_with_a_json_envelope(
    monkeypatch, capsys, caplog, err
):
    """A230: these used to escape as a raw traceback with no ``--json`` envelope."""

    def boom(a):
        raise err

    monkeypatch.setattr(cli, "_cmd_paths", boom)
    assert cli.main(["paths", "--json"]) == 1
    out, stderr = capsys.readouterr()
    doc = json.loads(out[out.index("{") :])
    assert doc["ok"] is False and doc["error"]["code"] == "error"
    assert doc["error"]["message"].startswith(type(err).__name__)
    if not err.args:
        assert doc["error"]["message"] == "KeyError"
    assert "Traceback" not in stderr
    assert [r.levelname for r in caplog.records if r.name == "cutpoint"] == ["ERROR"]


def test_captions_filter_escapes_the_ass_file_name(tmp_path, monkeypatch):
    """A231: ``--out "best, part 2.mp4"`` put an unescaped ``,`` in the filter chain."""
    stage, out = tmp_path / "stage.mp4", tmp_path / "best, part 2.mp4"
    ass = out.with_suffix(".ass")
    ass.write_text("x", "utf-8")
    seen: list[str] = []

    class Enc:
        def chain(self, vf):
            seen.append(vf)
            return vf

        def args(self):
            return []

    monkeypatch.setattr(assemble, "probe", lambda p: media_info(Path(p), 3.0))
    monkeypatch.setattr(assemble, "run", lambda *a, **k: None)
    monkeypatch.setattr(assemble, "encode_with_fallback", lambda fn, name: fn(Enc()))
    assemble.post_process(stage, out, False, True, None, ass=ass)
    assert seen == [r"ass=best\, part 2.ass"]


@pytest.mark.parametrize(
    "argv",
    [
        ["clips", "p", "--tail-min", "-1"],
        ["clips", "p", "--tail-max", "-1"],
        ["clips", "p", "--event-window", "-1"],
        ["clips", "p", "--extend", "-1"],
        ["clips", "p", "--extend-start", "-0.5"],
        ["clips", "p", "--extend-end", "-0.5"],
        ["clips", "p", "--speech-max-extend", "-1"],
        ["clips", "p", "--speech-max-lead", "-1"],
        ["clips", "p", "--audio-fade", "-0.1"],
        ["clips", "p", "--pano-cmd-every", "0"],
        ["extend", "d", "--seconds", "-1"],
        ["extend", "d", "--start", "-1"],
        ["extend", "d", "--end", "-1"],
    ],
)
def test_negative_export_options_are_a_usage_error(capsys, argv):
    """A232: negative export options (e.g. ``--tail-min``) were accepted; all are usage errors now."""
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    assert argv[-2] in capsys.readouterr().err
