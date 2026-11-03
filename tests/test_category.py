"""Footage category detection (#513): the frame measures, the decision, the probe cache and
``cutpoint detect``."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from conftest import envelope, needs_ffmpeg
from cutpoint import category, cli, paths, styles
from cutpoint.category import ACTION_CAM, GAMEPLAY, GENERAL, TALK, SourceProbe

RNG = np.random.default_rng(7)
H, W = category.GRID[1], category.GRID[0]


def moving_frames(n: int = 16) -> np.ndarray:
    """``n`` unrelated textured frames: nothing stays still."""
    return RNG.uniform(40, 220, (n, H, W)).astype(np.float32)


@pytest.mark.parametrize(
    ("make", "hud", "still"),
    [
        ("hud", True, False),  # a static HUD box over a moving picture
        ("letterbox", False, False),  # black bars are not a HUD
        ("still", None, True),  # a still: everything static (its edges too)
    ],
)
def test_static_measures_find_a_hud_not_bars_or_a_still(make, hud, still):
    frames = moving_frames()
    if make == "hud":
        frames[:, 150:170, 20:120] = 30.0
        frames[:, 155:165, 25:115] = 230.0  # a bright bar with a dark frame: edges
    elif make == "letterbox":
        frames[:, :20], frames[:, -20:] = 0.0, 0.0
    else:
        frames[:] = frames[0]
    hud_static, static_share = category.static_measures(frames)
    assert hud is None or (hud_static > 0.01) == hud
    assert (static_share > category.STILL_SHARE) == still


def test_noise_and_shift_measures():
    flat = np.full((H, W), 128.0, np.float32)
    noisy = flat + RNG.normal(0, 3, (H, W)).astype(np.float32)
    assert category.noise_sigma(flat) == pytest.approx(0.0, abs=0.01)
    assert category.noise_sigma(noisy) == pytest.approx(3.0, rel=0.25)
    img = RNG.uniform(0, 255, (H, W)).astype(np.float32)
    assert category.shift(img, np.roll(img, (3, 4), axis=(0, 1))) == pytest.approx(5.0)


def probe(path="a.mp4", duration=600.0, **kw) -> SourceProbe:
    return SourceProbe(path, duration, kw.pop("telemetry", False), **kw)


GAME = {"handler": "VideoHandle", "frames": 16, "hud_static": 0.01, "static_share": 0.05,
        "noise_floor": 0.2, "shake": 0.0}  # fmt: skip
CAMERA = {"make": "Apple", "frames": 16, "hud_static": 0.0, "static_share": 0.0,
          "noise_floor": 3.0, "shake": 4.0}  # fmt: skip


@pytest.mark.parametrize(
    ("probes", "speech", "expect"),
    [
        ([probe(telemetry=True)], None, (ACTION_CAM, False, False)),
        ([probe(**GAME)], None, (GAMEPLAY, False, False)),
        ([probe(**CAMERA)], None, (GENERAL, False, True)),  # talk still open
        ([probe(**CAMERA)], {"a.mp4": 0.7}, (TALK, False, False)),
        ([probe(**CAMERA)], {"a.mp4": 0.45}, (TALK, True, False)),  # the band: ambiguous
        # a talky stream with a thin HUD (score 0.45): talk, but not sure
        (
            [probe(**{**CAMERA, "make": "", "hud_static": 0.0012})],
            {"a.mp4": 0.7},
            (TALK, True, False),
        ),
        ([probe(duration=60.0, **CAMERA)], None, (GENERAL, False, False)),  # too short for talk
        # screen-recorder tags without a HUD: OBS re-encodes of camera footage have them too
        ([probe(duration=60.0, **{**GAME, "hud_static": 0.0})], None, (GENERAL, True, False)),
    ],
)
def test_decide_one_source(probes, speech, expect):
    det = category.decide(probes, speech)
    assert (det.category, det.ambiguous, det.pending) == expect


def test_decide_majority_disagreement_and_the_commentary_hint():
    """Telemetry in half the footage makes it action cam; the gameplay minority (over 10%)
    is named. A talky game stays gameplay and suggests commentary."""
    mixed = [probe("cam.mp4", 300.0, telemetry=True), probe("game.mp4", 300.0, **GAME)]
    det = category.decide(mixed, {"cam.mp4": 0.1, "game.mp4": 0.1})
    assert det.category == ACTION_CAM and det.disagree == ["game.mp4"]
    assert det.scores[ACTION_CAM] == det.scores[GAMEPLAY] == 0.5
    small = [probe("game.mp4", 950.0, **GAME), probe("cam.mp4", 50.0, telemetry=True)]
    assert category.decide(small, {}).disagree == []  # a 5% minority
    det = category.decide([probe(**GAME)], {"a.mp4": 0.9})
    assert (det.category, det.suggest) == (GAMEPLAY, "commentary")


def test_an_ambiguous_detection_falls_back_and_sport_is_never_chosen():
    unsure = category.Detection(GAMEPLAY, True, "", {}, [])
    assert category.choose_style(unsure, category.HYPE, "gaming") == ("gaming", "gaming")
    chosen = {
        category.choose_style(category.Detection(c, False, "", {}, []), e, "x")[0]
        for c in category.CATEGORIES
        for e in (category.HYPE, category.FUNNY)
    }
    assert chosen == {"gaming", "funny", "action-cam", "commentary", "action"}


def test_a_learned_auto_style_is_ignored_with_a_warning(home, caplog):
    folder = paths.styles_dir()
    folder.mkdir(parents=True)
    for name in ("auto", "mine"):
        (folder / f"{name}.json").write_text(json.dumps({"base": "gaming"}), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert "auto" not in styles.style_names() and "mine" in styles.style_names()
    assert "reserved for --style auto" in caplog.text
    with pytest.raises(ValueError, match="not a style itself"):
        styles.get_style("auto")


def test_camera_tag_reason_names_the_maker():
    for key, maker in (
        ("com.apple.quicktime.software", "Apple"),
        ("com.android.version", "Android"),
    ):
        meta = category._metadata({"format": {"tags": {key: "x"}}})
        assert meta["make"] == maker.lower()
        reason = category.gameplay_score(probe(make=meta["make"]))[1]
        assert reason == [f"camera tags ({maker.lower()})"]


@pytest.fixture
def faked_probe(home, tmp_path, monkeypatch):
    """``probe_source`` of a tiny file with ffprobe, the telemetry check and the frame probe
    faked; ``state.frames`` is the number of spread frames the fake decodes."""
    from types import SimpleNamespace  # noqa: PLC0415

    from cutpoint import analyze, ffmpeg_utils  # noqa: PLC0415

    state = SimpleNamespace(src=tmp_path / "v" / "a.mp4", frames=16, probes=0)
    state.src.parent.mkdir()
    state.src.write_bytes(b"x")
    monkeypatch.setattr(
        ffmpeg_utils, "probe", lambda p: SimpleNamespace(duration=60.0, width=640, height=360)
    )
    monkeypatch.setattr(ffmpeg_utils, "ffprobe_json", lambda *a, **k: {})
    monkeypatch.setattr(analyze, "has_telemetry", lambda p: False)

    def frame_probe(*a):
        state.probes += 1
        return {"frames": state.frames}

    monkeypatch.setattr(category, "_frame_probe", frame_probe)
    return state


def test_a_cached_probe_takes_the_path_it_is_asked_for(faked_probe):
    from cutpoint import cache  # noqa: PLC0415

    src = faked_probe.src
    first = category.probe_source(src)
    cached = cache.source_dir(src) / category.CACHE_FILE
    cache.save_json(cached, {**first.to_json(), "path": "Z:/elsewhere/a.mp4"})
    again = category.probe_source(src)
    assert again.path == str(src) and faked_probe.probes == 1


def test_a_failed_frame_read_is_not_cached(faked_probe):
    faked_probe.frames = category.MIN_FRAMES - 1  # a timeout on the NAS
    category.probe_source(faked_probe.src)
    category.probe_source(faked_probe.src)
    assert faked_probe.probes == 2
    faked_probe.frames = category.SPREAD
    category.probe_source(faked_probe.src)
    category.probe_source(faked_probe.src)
    assert faked_probe.probes == 3


@needs_ffmpeg
def test_probe_source_measures_frames_once_and_detect_reports_it(video, capsys, monkeypatch):
    first = category.probe_source(video)
    assert not first.telemetry and first.frames == category.SPREAD
    assert first.hud_static is not None and first.shake is not None
    monkeypatch.setattr(category, "_frame_probe", lambda *a: pytest.fail("probed again"))
    assert category.probe_source(video) == first  # from the cache

    assert cli.main(["detect", str(video), "--json"]) == 0
    doc = envelope(capsys.readouterr().out)
    out = doc["outputs"]
    assert doc["ok"] and doc["project"] is None
    assert out["category"] in category.CATEGORIES
    assert set(out["styles"]) == {"analyze", "funny-moments"}
    assert [s["path"] for s in out["sources"]] == [str(video)]
