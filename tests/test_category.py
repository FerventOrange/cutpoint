"""Footage category detection (#513): the frame measures, the decision, the probe cache and
``cutpoint detect``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from conftest import envelope, needs_ffmpeg
from cutpoint import category, cli, paths, styles
from cutpoint.category import ACTION_CAM, GAMEPLAY, GENERAL, TALK, SourceProbe
from cutpoint.events import ocr_raw

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
    m = category.static_measures(frames)
    assert hud is None or (m["hud_static"] > 0.01) == hud
    assert (m["static_share"] > category.STILL_SHARE) == still
    assert still or "box" not in m  # a HUD bar a third of the width is no lower-third box


def lower_third(frames: np.ndarray, width: int = W) -> np.ndarray:
    """``frames`` with a news lower third: an opaque box ``width`` wide (rows 150-172) with
    two lines of fixed text (bright bars, so edges)."""
    frames[:, 150:172, :width] = 40.0
    frames[:, 154:160, 10:120] = 230.0
    frames[:, 164:169, 10:150] = 230.0
    return frames


def corner_hud(frames: np.ndarray) -> np.ndarray:
    """``frames`` with a game HUD: fixed boxes in three corners."""
    for r0, c0 in ((8, 8), (158, 8), (158, 250)):
        frames[:, r0 : r0 + 14, c0 : c0 + 60] = 30.0
        frames[:, r0 + 3 : r0 + 11, c0 + 3 : c0 + 57] = 230.0
    return frames


def test_a_lower_third_box_is_found_and_its_text_is_no_hud():
    """The box's text stays put like a HUD's: it is measured as the box's (``hud_box``) and
    the HUD evidence (:attr:`SourceProbe.hud`) leaves it out. Corner HUD boxes, also at
    the bottom, and a box half the width are no lower third."""
    m = category.static_measures(lower_third(moving_frames()))
    assert m["box"] == [pytest.approx(150 / H, abs=0.02), pytest.approx(172 / H, abs=0.02)]
    assert m["hud_static"] > category.HUD_FULL
    assert m["hud_box"] == pytest.approx(m["hud_static"])
    p = probe(**{**GAME, "handler": "", **m})
    assert p.hud == pytest.approx(0.0)
    assert "lower-third box" in category.gameplay_score(p)[1]
    hud = category.static_measures(corner_hud(moving_frames()))
    assert "box" not in hud and hud["hud_static"] > category.HUD_FULL
    assert "box" not in category.static_measures(lower_third(moving_frames(), W // 2))


def test_a_lower_third_and_a_channel_logo_are_not_sure_gameplay():
    """#701: an untagged news clip (a lower third, a channel logo) was sure gameplay from
    its static text alone. The box's text no longer counts, so it is unsure (and gets the
    text check); the same static share without a box stays sure gameplay."""
    det = category.decide([probe(**NEWS)])
    assert det.ambiguous and "lower-third box" in det.reason
    assert category.needs_text(probe(**NEWS))
    det = category.decide([probe(**{**NEWS, "box": None, "hud_box": None})])
    assert (det.category, det.ambiguous) == (GAMEPLAY, False)


def test_probes_cached_before_the_box_are_measured_again():
    assert category.CACHE_FILE == "category-v2.json"
    assert SourceProbe.from_json({**probe(**GAME).to_json(), "version": 1}) is None


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
NEWS = {**GAME, "handler": "", "hud_static": 0.0076, "box": [0.86, 0.96], "hud_box": 0.0069}


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


# --------------------------------------------------------------------------- HUD text (OCR)
def text_box(x0: float, y0: float, w: float = 0.08, h: float = 0.025) -> dict:
    return {"box": [x0, y0, x0 + w, y0 + h], "text": "AMMO 30", "score": 0.9}


def test_text_anchors_keep_boxes_that_stay_in_place():
    """A HUD number keeps its place while its width changes; a box in a few frames (a
    passing sign) is no anchor."""
    hud = [(1100.0, 650.0, 1180.0 + 6 * (i % 2), 670.0) for i in range(8)]  # 30 / 120
    sign = [(400.0, 300.0, 500.0, 330.0)] * 3 + [()] * 5
    frames = [[hud[i], *([sign[i]] if sign[i] else [])] for i in range(8)]
    assert category.text_anchors(frames) == [(0, hud[0])]


@pytest.mark.parametrize(
    ("anchor", "kept"),
    [
        (text_box(0.85, 0.9), True),  # ammo, bottom right
        ({**text_box(0.85, 0.9), "text": "2025-03-14 12:34:56"}, False),  # a camera's clock
        ({**text_box(0.02, 0.02), "text": "2025/03/14"}, False),
        ({**text_box(0.8, 0.95), "text": "54 KM/H N49.1234 W122.98"}, False),  # a dashcam
        ({**text_box(0.02, 0.02), "text": "MAR 14 2025"}, False),
        ({**text_box(0.02, 0.02), "text": "March 14"}, False),
        ({**text_box(0.02, 0.02), "text": "Sept. 3"}, False),
        ({**text_box(0.85, 0.9), "text": "Deck 3"}, True),
        ({**text_box(0.85, 0.9), "text": "Marine 2"}, True),
        ({**text_box(0.85, 0.9), "text": "Market 12"}, True),
        ({**text_box(0.85, 0.9), "text": "Octane 5"}, True),
        ({**text_box(0.02, 0.9), "text": "N49°12'34\""}, False),
        ({**text_box(0.8, 0.02), "text": "25°C 77°F"}, False),
        ({**text_box(0.5, 0.1), "text": "275°"}, True),  # a War Thunder bearing
        ({**text_box(0.4, 0.88), "text": "Watch out for that rock"}, False),  # subtitles
        ({**text_box(0.85, 0.9), "score": 0.3}, False),  # not read
        ({**text_box(0.85, 0.9), "text": " "}, False),
    ],
)
def test_hud_anchors_leave_out_camera_overlays(anchor, kept):
    assert category.hud_anchors([anchor]) == int(kept)


def test_text_in_the_lower_third_box_is_no_hud_text():
    box = [0.86, 0.96]
    line = {**text_box(0.03, 0.87, w=0.2, h=0.05), "text": "LIVE Whistler, BC"}
    assert category.hud_anchors([line]) == 1
    assert category.hud_anchors([line], box) == 0
    assert category.hud_anchors([text_box(0.85, 0.8)], box) == 1  # above the box


THIN_HUD = {**GAME, "hud_static": 0.0}  # screen recorder tags, a HUD too thin for GRID


@pytest.mark.parametrize(
    ("fields", "needs"),
    [
        (THIN_HUD, True),  # unsure general (0.4)
        ({**THIN_HUD, "handler": "", "hud_static": 0.002}, True),  # unsure gameplay (0.55)
        (GAME, False),  # sure gameplay
        ({**THIN_HUD, "make": "Apple"}, False),  # camera tags: sure general, never lifted
        ({**THIN_HUD, "static_share": 0.9}, False),  # a still (a desktop, a menu)
        ({**THIN_HUD, "telemetry": True}, False),
        ({**THIN_HUD, "hud_static": None, "frames": 2}, False),  # no frame measures
    ],
)
def test_the_text_check_runs_only_for_unsure_footage(fields, needs):
    assert category.needs_text(probe(**fields)) is needs


def test_hud_text_makes_an_unsure_screen_recording_sure_gameplay():
    """Enough HUD text lifts a thin-HUD recording to sure gameplay, also when it is talky
    (the commentary hint stays); too little keeps it unsure."""
    many, few = category.TEXT_ANCHORS, category.TEXT_ANCHORS - 1
    det = category.decide([probe(**THIN_HUD, hud_text=many)])
    assert (det.category, det.ambiguous) == (GAMEPLAY, False)
    assert f"HUD text in {many} places" in det.reason
    det = category.decide([probe(**THIN_HUD, hud_text=many)], {"a.mp4": 0.9})
    assert (det.category, det.ambiguous, det.suggest) == (GAMEPLAY, False, "commentary")
    det = category.decide([probe(**THIN_HUD, hud_text=few)])
    assert det.ambiguous and "no HUD text" in det.reason


@pytest.fixture(autouse=True)
def _reset_failed_text_checks():
    category._TEXT_FAILED.clear()
    yield
    category._TEXT_FAILED.clear()


@pytest.fixture
def text_check(faked_probe, monkeypatch):
    """``faked_probe`` of a thin-HUD screen recording, with the OCR faked:
    ``state.anchors`` is what :func:`category.text_probe` finds, ``state.texts`` counts
    its runs."""
    state = faked_probe
    state.anchors, state.texts = [text_box(0.02 + 0.2 * i, 0.03) for i in range(5)], 0
    state.frame = {"frames": 16, "hud_static": 0.0, "static_share": 0.05}

    def frame_probe(*a):
        state.probes += 1
        return dict(state.frame)

    def text_probe(*a):
        state.texts += 1
        return {"version": 1, "engine": "x", "frames": 8, "anchors": state.anchors}

    monkeypatch.setattr(category, "_frame_probe", frame_probe)
    monkeypatch.setattr(category, "text_probe", text_probe)
    monkeypatch.setattr(ocr_raw, "engine_version", lambda: "x")
    monkeypatch.setattr(category, "_metadata", lambda data: {"handler": "VideoHandle"})
    return state


def test_the_text_check_is_cached_apart_and_lifts_the_source(text_check):
    from cutpoint import cache  # noqa: PLC0415

    src = text_check.src
    first = category.probe_source(src)
    assert first.hud_text == 5 and text_check.texts == 1
    assert "hud_text" not in cache.load_json(cache.source_dir(src) / category.CACHE_FILE)
    again = category.probe_source(src)  # both records cached
    assert again == first and (text_check.probes, text_check.texts) == (1, 1)
    assert category.decide([again]).ambiguous is False


def test_a_new_ocr_engine_reads_the_text_again(text_check, monkeypatch):
    category.probe_source(text_check.src)
    monkeypatch.setattr(ocr_raw, "engine_version", lambda: "y")
    category.probe_source(text_check.src)
    assert text_check.texts == 2


def test_without_the_ocr_engine_the_check_is_skipped_quietly(text_check, monkeypatch, caplog):
    """Cutpoint lite: no check, nothing cached, the detection stays ambiguous; installing
    the engine later runs it."""
    from cutpoint import cache  # noqa: PLC0415

    monkeypatch.setattr(category, "ocr_installed", lambda: False)
    with caplog.at_level(logging.INFO):
        p = category.probe_source(text_check.src)
    assert p.hud_text is None and text_check.texts == 0 and caplog.text == ""
    assert not (cache.source_dir(text_check.src) / category.TEXT_FILE).exists()
    assert category.decide([p]).ambiguous
    monkeypatch.setattr(category, "ocr_installed", lambda: True)
    assert category.probe_source(text_check.src).hud_text == 5


def test_a_failed_text_check_keeps_the_frame_probe_and_is_not_cached(text_check, monkeypatch):
    def broken(*a):
        text_check.texts += 1
        raise RuntimeError("onnxruntime fell over")

    monkeypatch.setattr(category, "text_probe", broken)
    assert category.probe_source(text_check.src).hud_text is None
    assert category.probe_source(text_check.src).hud_text is None
    assert text_check.texts == 1  # one attempt per process (edit --style auto detects twice)


def test_a_news_clip_stays_unsure_after_the_text_check(text_check, monkeypatch):
    """The lower third's two lines and the channel logo are all text in place; only the
    logo is outside the box, too little for a HUD."""
    text_check.frame = {"frames": 16, **{k: NEWS[k] for k in ("hud_static", "static_share")}}
    text_check.frame.update(box=NEWS["box"], hud_box=NEWS["hud_box"])
    text_check.anchors = [
        {**text_box(0.86, 0.03), "text": "CBC NEWS"},
        {**text_box(0.03, 0.87, w=0.2, h=0.05), "text": "LIVE Whistler, BC"},
        {**text_box(0.03, 0.92, w=0.3, h=0.04), "text": "Record snowfall closes the mountain"},
    ]
    monkeypatch.setattr(category, "_metadata", lambda data: {})  # tags stripped
    p = category.probe_source(text_check.src)
    assert text_check.texts == 1 and p.hud_text == 1
    assert category.decide([p]).ambiguous


def test_camera_overlays_never_make_camera_footage_sure_gameplay(text_check):
    """A burned-in clock, a dashcam's speed and position, and subtitles are all text in
    place, but none is a HUD's."""
    text_check.anchors = [
        {**text_box(0.8, 0.93), "text": "2025-03-14 12:34:56"},
        {**text_box(0.02, 0.03), "text": "2025/03/14 00:00:20"},
        {**text_box(0.75, 0.95), "text": "59 KM/H N49.1234 W122.9876"},
        {**text_box(0.35, 0.85), "text": "We are heading down the ridge now"},
    ]
    p = category.probe_source(text_check.src)
    assert p.hud_text == 0 and category.decide([p]).ambiguous


@pytest.mark.parametrize(
    ("frame", "meta"),
    [
        ({"hud_static": 0.01}, {"handler": "VideoHandle"}),  # a static HUD: sure gameplay
        ({}, {"make": "Apple"}),  # camera tags: sure general
        ({"static_share": 0.9}, {"handler": "VideoHandle"}),  # a still picture
    ],
)
def test_sure_footage_and_stills_are_never_read(text_check, monkeypatch, frame, meta):
    text_check.frame.update(frame)
    monkeypatch.setattr(category, "_metadata", lambda data: meta)
    assert category.probe_source(text_check.src).hud_text is None
    assert text_check.texts == 0


def test_text_probe_reads_a_real_hud_and_leaves_out_clock_and_subtitles(monkeypatch):
    """The real OCR on synthetic frames: three fixed labels are HUD anchors, the clock and
    the bottom-centre subtitle are not (box order, scaling and recognition end to end)."""
    pytest.importorskip("rapidocr_onnxruntime")
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    w, h = 640, 360
    frames = []
    for _ in range(8):
        f = np.full((h, w, 3), 30, np.uint8) + rng.integers(0, 20, (h, w, 3), dtype=np.uint8)
        for text, org in [
            ("HEALTH 100", (20, 300)),
            ("AMMO 30", (480, 300)),
            ("SCORE 4500", (480, 60)),
            ("12:34:56", (20, 40)),
            ("Watch out for that rock", (170, 335)),
        ]:
            cv2.putText(f, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        frames.append(f)
    monkeypatch.setattr(category, "_text_frames", lambda *a: frames)
    rec = category.text_probe(Path("x"), 60.0, w, h)
    assert rec["engine"] == ocr_raw.engine_version() and rec["frames"] == 8
    assert all(0.0 <= v <= 1.0 for a in rec["anchors"] for v in a["box"])
    assert category.hud_anchors(rec["anchors"]) == 3


@needs_ffmpeg
def test_text_frames_decodes_eight_frames_at_the_ocr_width(video):
    frames = category._text_frames(video, 10.0, 160, 120)
    assert len(frames) == category.TEXT_FRAMES
    assert all(f.shape == (120, 160, 3) for f in frames)
