"""Footage category detection (#513): the frame measures, the decision, the probe cache and
``cutpoint detect``."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import fields, replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from conftest import envelope, needs_ffmpeg
from cutpoint import category, cli, paths, styles
from cutpoint.category import ACTION, ACTION_CAM, GAMEPLAY, GENERAL, TALK, SourceProbe
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
    assert hud is None or (m["hud_static"] > category.HUD_FULL) == hud
    assert (m["static_share"] > category.STILL_SHARE) == still
    assert still or "box" not in m  # a HUD bar a third of the width is no lower-third box


def test_still_edges_scattered_over_the_picture_are_no_hud():
    """#513: fixed cameras cut between in a studio line up a background pattern by chance:
    still edges spread thinly over the picture, not packed as a HUD's."""
    frames = moving_frames()
    mask = RNG.uniform(0, 1, (H, W)) < 0.03
    frames[:, mask] = 230.0
    frames[:, np.roll(mask, 1, axis=1)] = 30.0
    m = category.static_measures(frames)
    assert m["hud_static"] < category.HUD_FULL / 4


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


def test_a_talky_box_source_stays_unsure():
    det = category.decide([probe(**NEWS, duration=900)], {"a.mp4": 0.7})
    assert (det.category, det.ambiguous) == (TALK, True)


def test_probes_cached_before_the_windows_are_measured_again():
    assert category.CACHE_FILE == "category-v3.json"
    for old in (1, 2):
        assert SourceProbe.from_json({**probe(**GAME).to_json(), "version": old}) is None


@pytest.mark.parametrize(
    ("rows", "cols"),
    [
        ((30, 330), (0, 640)),  # a letterbox
        ((0, 360), (80, 560)),  # a pillarbox
        ((40, 320), (60, 580)),  # a game window inside a larger black canvas
        ((0, 360), (0, 640)),  # no bars
    ],
)
def test_black_bars_are_cropped_before_the_hud_is_measured(rows, cols):
    """#513: bars are steady and dark, so they are no HUD evidence and no picture: the frames
    are cropped to the picture inside them before they are scaled to the grid."""
    frames = np.zeros((8, 360, 640), np.float32)
    frames[:, rows[0] : rows[1], cols[0] : cols[1]] = RNG.uniform(
        40, 220, (8, rows[1] - rows[0], cols[1] - cols[0])
    )
    r, c = category._bars(frames)
    assert (r.start, r.stop, c.start, c.stop) == (*rows, *cols)
    grid = category._to_grid(frames)
    assert grid.shape == (8, H, W)
    assert float(grid[:, :2].mean()) > category.BAR_LUMA  # no bar left at the top


def test_a_hud_in_part_of_the_footage_is_found_per_window():
    """#513: a recording that is half gameplay (a HUD) and half menus or cutscenes (no
    steady pixels) lost the HUD when its frames were pooled; per window it is found, and
    dark frames (loading, fades) are left out of the windows."""
    n = 4 * category.WINDOWS
    frames = moving_frames(n)
    corner_hud(frames[: n // 2])
    frames[1] = 0.0  # a fade to black inside a HUD window
    spread = list(frames)
    pooled = category.static_measures(category._to_grid(frames))
    assert pooled["hud_static"] < category.HUD_FULL
    m = category.window_measures(spread)
    assert m["hud_static"] > category.HUD_FULL
    assert m["static_share"] < category.STILL_SHARE
    assert category.window_measures(list(moving_frames(n)))["hud_static"] < category.HUD_FULL


def test_an_all_dark_window_is_not_cropped_to_nothing():
    grid = category._to_grid(np.zeros((6, 360, 640), np.float32))
    assert grid.shape == (6, H, W)


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


@pytest.mark.parametrize(
    ("game_s", "named"),
    [(300.0, True), (120.0, True), (100.0, False), (50.0, False)],
    ids=["half", "over-10pct", "exactly-10pct", "5pct"],
)
def test_disagreeing_names_the_sources_of_a_record_as_decide_does(game_s, named):
    """``disagreeing`` reads a highlights.json ``detection`` record and names the same sources
    as :attr:`Detection.disagree`: only a minority above DISAGREE_SHARE of the footage."""
    probes = [probe("cam.mp4", 1000.0 - game_s, telemetry=True), probe("game.mp4", game_s, **GAME)]
    det = category.decide(probes, {})
    rec = json.loads(json.dumps(det.record("hype", "action-cam", None)))  # as read back
    found = category.disagreeing(rec)
    assert [s["path"] for s in found] == det.disagree == (["game.mp4"] if named else [])
    if named:
        assert found[0]["category"] == GAMEPLAY and found[0]["reason"]


def test_disagreeing_takes_a_record_without_scores_or_sources():
    assert category.disagreeing({}) == []
    assert category.disagreeing({"category": GAMEPLAY, "scores": {GAMEPLAY: 0.5}}) == []
    rec = {"category": GAMEPLAY, "scores": {GAMEPLAY: 0.5}, "sources": ["x", {"category": TALK}]}
    assert category.disagreeing(rec) == [{"category": TALK}]


def test_an_ambiguous_detection_falls_back_and_sport_is_never_chosen():
    # an unsure detection that leans to gameplay (some HUD evidence) uses gaming; any other
    # the General style (#896), not gaming
    for cat, style in ((GAMEPLAY, "gaming"), (GENERAL, "action"), (TALK, "action")):
        unsure = category.Detection(cat, True, "", {}, [])
        assert category.choose_style(unsure, category.HYPE) == (style, style)
        assert category.fallback_label(unsure) == ("Gameplay" if cat == GAMEPLAY else "General")
    assert styles.auto_fallback(category.HYPE, GAMEPLAY) == "gaming"
    assert styles.auto_fallback(category.HYPE, TALK) == "action"
    assert styles.auto_fallback(category.HYPE) == styles.resolve_auto(GENERAL, category.HYPE)
    assert styles.auto_fallback(category.HYPE) == "action"
    assert styles.auto_fallback(category.FUNNY) == "funny"
    chosen = {
        category.choose_style(category.Detection(c, False, "", {}, []), e)[0]
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


def test_the_ambiguous_warning_names_the_general_style(caplog):
    det = category.decide([probe(**{**GAME, "hud_static": 0.0})])
    with caplog.at_level(logging.WARNING):
        category.warn(det, "action", "action")
    assert "using the General style (action)" in caplog.text


def test_several_audio_tracks_are_no_screen_recorder_sign():
    """#513 miss 8: an editor's multi-track export is not pushed toward gameplay; ShadowPlay's
    own handler still is; camera tags win over both."""
    edit = probe(audio_tracks=5, frames=16, hud_static=0.0, static_share=0.0)
    assert not edit.screen_marks
    assert category.gameplay_score(edit)[0] == category.BASE_SCORE
    assert probe(handler="VideoHandle").screen_marks
    cam = probe(audio_tracks=5, make="Apple", frames=16, hud_static=0.0, static_share=0.0)
    det = category.decide([cam])
    assert (det.category, det.ambiguous, det.decided_by) == (GENERAL, False, ["camera tags"])


@pytest.mark.parametrize(
    ("size", "sure"),
    [
        ((5760, 2880), True),  # a stitched 360 export, tags stripped
        ((3840, 1920), True),
        ((2160, 1080), False),  # a 2:1 phone screen recording
        ((3840, 2160), False),
    ],
)
def test_a_2_to_1_frame_of_360_video_is_camera_footage(size, sure):
    w, h = size
    p = probe(duration=30.0, width=w, height=h, frames=16, hud_static=0.0, static_share=0.0)
    assert p.equirect is sure
    det = category.decide([p])
    assert (det.category, det.ambiguous) == (GENERAL, not sure)
    assert det.decided_by == (["360 frame"] if sure else [])
    assert ("360 frame" in det.reason) is sure


UNSURE = {"frames": 16, "hud_static": 0.0, "static_share": 0.0}  # untagged, no HUD: 0.3


@pytest.mark.parametrize("answer", ["gameplay", "camera", "screen", "other"])
@pytest.mark.parametrize("speech", [None, {"a.mp4": 0.1}, {"a.mp4": 0.7}, {"a.mp4": 0.45}])
def test_only_an_action_answer_changes_footage_the_camera_tags_decide(answer, speech):
    """#958: camera tags alone no longer keep the vision model away (an action camera without
    telemetry has a phone's tags), but re-exports and phone clips carry them too: any other
    answer leaves the decision as without vision, talk included."""
    plain = category.decide([probe(**CAMERA)], speech)
    seen = category.decide([probe(**CAMERA, vision=answer)], speech)
    assert seen == plain
    assert "vision" not in seen.reason


@pytest.mark.parametrize("answer", ["action", "gameplay", "camera", "screen", "other"])
def test_a_tagged_360_source_stays_as_without_vision(answer):
    """360 is never asked; an answer on file (an older cache) cannot make it action either."""
    tagged = {**CAMERA, "width": 5760, "height": 2880}
    plain = category.decide([probe(**tagged)])
    assert category.decide([probe(**tagged, vision=answer)]) == plain
    assert (plain.category, plain.ambiguous) == (GENERAL, False)


def test_a_partial_hud_keeps_talky_tagged_footage_unsure_whatever_the_model_saw():
    """Talk is weighed as without vision for a source asked only because of its tags."""
    tagged = {**CAMERA, "hud_static": category.HUD_FULL / 3}  # still sure general
    plain = category.decide([probe(**tagged)], {"a.mp4": 0.7})
    assert (plain.category, plain.ambiguous) == (TALK, True)
    for answer in ("action", "camera"):
        assert category.decide([probe(**tagged, vision=answer)], {"a.mp4": 0.7}) == plain


@pytest.mark.parametrize(
    ("fields", "answer", "speech", "expect"),
    [
        (UNSURE, "gameplay", None, (GAMEPLAY, False, ["vision"])),
        (UNSURE, "camera", None, (GENERAL, False, ["vision"])),
        (UNSURE, "action", None, (ACTION, False, ["vision"])),
        # a sport the model sees on footage with camera tags: vision still decides
        ({**CAMERA}, "action", None, (ACTION, False, ["vision"])),
        ({**CAMERA}, "action", {"a.mp4": 0.1}, (ACTION, False, ["vision"])),
        (UNSURE, "screen", None, (GENERAL, False, ["vision"])),
        (UNSURE, "other", None, (GENERAL, True, [])),  # the model cannot tell
        (UNSURE, None, None, (GENERAL, True, [])),  # not asked
        # unsure gameplay (0.55) the model sees as camera footage
        ({**UNSURE, "hud_static": 0.002}, "camera", None, (GENERAL, False, ["vision"])),
        # no game seen: speech decides talk, a thin HUD no longer keeps it unsure
        ({**UNSURE, "hud_static": 0.0012}, "screen", {"a.mp4": 0.7}, (TALK, False, ["speech"])),
        # a game seen: speech never makes it talk
        (UNSURE, "gameplay", {"a.mp4": 0.9}, (GAMEPLAY, False, ["vision"])),
    ],
)
def test_the_vision_answer_settles_an_unsure_source(fields, answer, speech, expect):
    det = category.decide([probe(**fields, vision=answer)], speech)
    assert (det.category, det.ambiguous, det.decided_by) == expect
    assert (f"vision: {answer}" in det.reason) is (answer is not None)
    assert det.sources[0].decided_by == (expect[2][0] if expect[2] else None)


def test_decided_by_names_the_evidence_of_most_footage():
    srcs = [
        probe("hud.mp4", 600.0, **GAME),
        probe("text.mp4", 100.0, **THIN_HUD, hud_text=category.TEXT_ANCHORS),
        probe("seen.mp4", 300.0, **UNSURE, vision="gameplay"),
    ]
    det = category.decide(srcs)
    assert det.decided_by == ["static HUD", "vision", "HUD text"]
    assert det.record("hype", "gaming", None)["decided_by"] == det.decided_by
    assert category.decide([probe(telemetry=True)]).decided_by == ["telemetry"]


@pytest.mark.parametrize(
    ("fields", "needs"),
    [
        (UNSURE, True),
        ({**UNSURE, "static_share": 0.9}, True),  # a still: a menu or a desktop
        (GAME, False),  # sure gameplay
        (CAMERA, True),  # sure general by camera tags alone: an action camera has them (#958)
        ({**CAMERA, "frames": 2, "hud_static": None}, False),  # tagged, no frame measures
        ({**UNSURE, "width": 5760, "height": 2880}, False),  # sure general by a 360 frame
        ({**CAMERA, "width": 5760, "height": 2880}, False),  # 360 with camera tags: never asked
        ({**CAMERA, "telemetry": True}, False),
        ({**UNSURE, "telemetry": True}, False),
        ({**UNSURE, "frames": 2, "hud_static": None}, False),  # no frame measures
        ({**GAME, "hud_static": 0.0, "hud_text": category.TEXT_ANCHORS}, False),  # the text
    ],
)
def test_the_vision_model_is_asked_only_about_unsure_footage(fields, needs):
    assert category.needs_vision(probe(**fields)) is needs


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
    category._ANCHORS_FAILED.clear()
    yield
    category._TEXT_FAILED.clear()
    category._ANCHORS_FAILED.clear()


@pytest.fixture
def text_check(faked_probe, monkeypatch):
    """``faked_probe`` of a thin-HUD screen recording, with the OCR faked:
    ``state.anchors`` is what :func:`category.text_probe` finds, ``state.texts`` counts
    its runs; the screen-text cue is faked too (``state.cue_anchors``, ``state.cues``)."""
    state = faked_probe
    state.anchors, state.texts = [text_box(0.02 + 0.2 * i, 0.03) for i in range(5)], 0
    state.cue_anchors, state.cues = [], 0
    state.frame = {"frames": 16, "hud_static": 0.0, "static_share": 0.05}

    def frame_probe(*a):
        state.probes += 1
        return dict(state.frame)

    def text_probe(*a):
        state.texts += 1
        return {"version": 1, "engine": "x", "frames": 8, "anchors": state.anchors}

    monkeypatch.setattr(category, "_frame_probe", frame_probe)

    def anchor_probe(*a):
        state.cues += 1
        return {"version": category.ANCHOR_VERSION, "frames": 8, "anchors": state.cue_anchors}

    monkeypatch.setattr(category, "text_probe", text_probe)
    monkeypatch.setattr(category, "anchor_probe", anchor_probe)
    monkeypatch.setattr(ocr_raw, "engine_version", lambda: "x")
    monkeypatch.setattr(category, "ocr_installed", lambda: True)  # also in Cutpoint lite
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
    """Cutpoint lite: no check, nothing cached, the screen-text cue in its place (finding
    nothing here), the detection stays ambiguous; installing the engine later runs it."""
    from cutpoint import cache  # noqa: PLC0415

    monkeypatch.setattr(category, "ocr_installed", lambda: False)
    with caplog.at_level(logging.INFO):
        p = category.probe_source(text_check.src)
    assert p.hud_text is None and text_check.texts == 0 and caplog.text == ""
    assert (p.screen_anchors, text_check.cues) == (0, 1)
    assert not (cache.source_dir(text_check.src) / category.TEXT_FILE).exists()
    assert category.decide([p]).ambiguous
    monkeypatch.setattr(category, "ocr_installed", lambda: True)
    p = category.probe_source(text_check.src)
    assert (p.hud_text, p.screen_anchors, text_check.cues) == (5, None, 1)


def test_a_failed_text_check_keeps_the_frame_probe_and_is_not_cached(text_check, monkeypatch):
    def broken(*a):
        text_check.texts += 1
        raise RuntimeError("onnxruntime fell over")

    monkeypatch.setattr(category, "text_probe", broken)
    assert category.probe_source(text_check.src).hud_text is None
    assert category.probe_source(text_check.src).hud_text is None
    assert text_check.texts == 1  # one attempt per process (edit --style auto detects twice)


def test_a_text_check_short_of_memory_stops_the_detection(text_check, monkeypatch):
    """Review F2 (#109): the OCR engine short of memory is no failed check; the detection
    does not go on without it (nor mark the check failed for the resumed run)."""
    from cutpoint.errors import LowMemoryError  # noqa: PLC0415

    def short(*a):
        raise LowMemoryError("Not enough memory for the OCR engine")

    monkeypatch.setattr(category, "text_probe", short)
    with pytest.raises(LowMemoryError):
        category.probe_source(text_check.src)
    assert not category._TEXT_FAILED


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
@pytest.mark.parametrize("ocr", [True, False])
def test_sure_footage_and_stills_are_never_read(text_check, monkeypatch, frame, meta, ocr):
    """Neither the HUD text check nor, without OCR, the screen-text cue runs on them."""
    text_check.frame.update(frame)
    monkeypatch.setattr(category, "_metadata", lambda data: meta)
    monkeypatch.setattr(category, "ocr_installed", lambda: ocr)
    p = category.probe_source(text_check.src)
    assert (p.hud_text, p.screen_anchors) == (None, None)
    assert (text_check.texts, text_check.cues) == (0, 0)


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


# --------------------------------------------------------------------------- screen-text cue (#904)
CUE_W, CUE_H = 640, 360


def word(frame: np.ndarray, x: int, y: int) -> None:
    """A word of fixed text at ``(x, y)``: bright strokes on a dark box (strong edges)."""
    frame[y - 2 : y + 14, x - 2 : x + 62] = 20
    for c in range(x, x + 60, 6):
        frame[y : y + 12, c : c + 3] = 235


def cue_frames(fixed=(), moving=(), n: int = 8) -> list[np.ndarray]:
    """``n`` BGR frames of a flat picture whose brightness changes, with words that stay in
    place (``fixed``) and words that move (``moving``: their place in the first frame)."""
    frames = []
    for i in range(n):
        f = np.full((CUE_H, CUE_W, 3), 90 + 12 * i, np.uint8)
        for x, y in fixed:
            word(f, x, y)
        for x, y in moving:
            word(f, x + 37 * i, y + 23 * i)
        frames.append(f)
    return frames


CORNERS = ((20, 20), (560, 20), (20, 320))  # top left, top right, bottom left
CENTRE, SUBTITLE = (290, 170), (290, 320)


def test_anchor_boxes_find_text_that_stays_in_place():
    """A fixed word is a box in every frame; a moving word is in none."""
    per = category.anchor_boxes(cue_frames(fixed=[CORNERS[0]], moving=[(100, 60)]))
    assert all(len(boxes) == 1 for boxes in per)
    x0, y0, x1, y1 = per[0][0]
    assert x0 <= 18 and y0 <= 18 and 82 <= x1 <= 90 and 34 <= y1 <= 40
    assert category.text_anchors(per) == [(0, per[0][0])]


def test_anchor_boxes_leave_out_black_bars_and_big_or_tall_shapes():
    frames = cue_frames(fixed=[(560, 60)])
    for f in frames:
        f[:40], f[-40:] = 0, 0  # a letterbox: its edge with the picture stays put
        f[150:250, 100:110] = 235  # a tall bar
        f[100:290, 400:450] = 235  # a block higher than a text line
    per = category.anchor_boxes(frames)
    assert all(len(boxes) == 1 and boxes[0][0] >= 550 for boxes in per)


def test_anchor_probe_counts_text_at_the_edge_of_the_picture(monkeypatch):
    """Three corner words are anchors; a word in the centre (a sign in a fixed camera's
    shot), a subtitle and a moving word are not; a lower-third box leaves its text out."""
    frames = cue_frames(fixed=[*CORNERS, CENTRE, SUBTITLE], moving=[(100, 60)])
    monkeypatch.setattr(category, "_text_frames", lambda *a: frames)
    rec = category.anchor_probe(Path("x"), 60.0, CUE_W, CUE_H)
    assert rec["version"] == category.ANCHOR_VERSION and rec["frames"] == 8
    assert len(rec["anchors"]) == 5
    assert all(0.0 <= v <= 1.0 for a in rec["anchors"] for v in a["box"])
    assert category.screen_anchors(rec["anchors"]) == 3
    assert category.screen_anchors(rec["anchors"], [0.85, 0.97]) == 2  # bottom left in the box
    monkeypatch.setattr(category, "_text_frames", lambda *a: frames[:5])
    assert category.anchor_probe(Path("x"), 60.0, CUE_W, CUE_H) is None  # too few frames


def cue_box(cx: float, cy: float) -> dict:
    return {"box": [cx - 0.04, cy - 0.01, cx + 0.04, cy + 0.01]}


@pytest.mark.parametrize(
    ("centre", "kept"),
    [
        ((0.1, 0.1), True),  # a corner
        ((0.5, 0.1), True),  # top centre (a compass, a score)
        ((0.9, 0.5), True),  # right edge
        ((0.1, 0.9), True),  # bottom left
        ((0.5, 0.5), False),  # the centre
        ((0.3, 0.7), False),  # inside the outer band
        ((0.5, 0.9), False),  # the subtitle band
    ],
)
def test_screen_anchors_keep_the_edge_of_the_picture(centre, kept):
    assert category.screen_anchors([cue_box(*centre)]) == int(kept)


BARE = {**THIN_HUD, "handler": ""}  # gameplay score 0.3 without text


@pytest.mark.parametrize(
    "fields",
    [BARE, THIN_HUD, {**BARE, "hud_static": 0.002}],  # 0.3, 0.4 and 0.55 without the cue
)
def test_screen_anchors_lift_to_unsure_gameplay_never_sure(fields):
    """Enough anchors lift by ANCHOR_WEIGHT to gameplay, never sure (0.55 + 0.2 is past
    GAMEPLAY_SURE), also when the source is talky: gameplay wins as with read text."""
    many = category.TEXT_ANCHORS
    p = probe(**fields, screen_anchors=many)
    bare = category.gameplay_score(probe(**fields))[0]
    assert category.gameplay_score(p)[0] == pytest.approx(bare + category.ANCHOR_WEIGHT)
    for speech in (None, 0.9):
        src = category._source(replace(p, duration=900.0), speech)
        assert (src.category, src.ambiguous, src.decided_by) == (GAMEPLAY, True, None)
        assert f"screen text anchors in {many} places (no OCR)" in src.reason
    assert category.decide([p]).decided_by == []


def test_few_screen_anchors_change_nothing():
    p = probe(**BARE, screen_anchors=category.TEXT_ANCHORS - 1)
    assert category.gameplay_score(p)[0] == category.gameplay_score(probe(**BARE))[0]
    det = category.decide([p])
    assert (det.category, det.ambiguous) == (GENERAL, True)
    assert "no screen text anchors (no OCR)" in det.reason


def test_screen_anchors_follow_the_text_support_guard_and_leave_needs_text_alone():
    many = category.TEXT_ANCHORS
    assert category._text_alone(probe(**BARE, screen_anchors=many))
    assert not category._text_alone(probe(**{**BARE, "hud_static": 0.002}, screen_anchors=many))
    assert category.needs_text(probe(**{**BARE, "hud_static": 0.002}, screen_anchors=9))


def test_read_hud_text_wins_over_the_screen_anchors():
    read = probe(**THIN_HUD, hud_text=category.TEXT_ANCHORS)
    assert category.decide([replace(read, screen_anchors=0)]) == category.decide([read])
    assert category.decide([read]).ambiguous is False


def test_without_ocr_the_cue_is_cached_apart_and_lifts_the_source(text_check, monkeypatch):
    from cutpoint import cache  # noqa: PLC0415

    monkeypatch.setattr(category, "ocr_installed", lambda: False)
    text_check.cue_anchors = [cue_box(0.1, 0.05), cue_box(0.9, 0.05), cue_box(0.1, 0.9)]
    src = text_check.src
    first = category.probe_source(src)
    assert (first.hud_text, first.screen_anchors, text_check.cues) == (None, 3, 1)
    cdir = cache.source_dir(src)
    assert "screen_anchors" not in cache.load_json(cdir / category.CACHE_FILE)
    assert (cdir / category.ANCHOR_FILE).exists()
    assert category.probe_source(src) == first and text_check.cues == 1  # cached
    det = category.decide([first])
    assert (det.category, det.ambiguous, det.decided_by) == (GAMEPLAY, True, [])


def test_with_ocr_the_cue_never_runs_and_the_records_stay(text_check, monkeypatch):
    """A full install: the HUD text check reads the source, the cue is never computed, the
    probe record keeps its keys and the decision is the text check's alone."""
    from cutpoint import cache  # noqa: PLC0415

    monkeypatch.setattr(category, "anchor_probe", lambda *a: pytest.fail("screen-text cue"))
    src = text_check.src
    p = category.probe_source(src)
    cdir = cache.source_dir(src)
    assert (p.hud_text, p.screen_anchors) == (5, None)
    assert not (cdir / category.ANCHOR_FILE).exists()
    record = (cdir / category.CACHE_FILE).read_bytes()
    keys = {"version"} | {f.name for f in fields(SourceProbe)}
    assert set(json.loads(record)) == keys - {"hud_text", "screen_anchors", "vision"}
    det = category.decide([p])
    assert (det.category, det.ambiguous, det.decided_by) == (GAMEPLAY, False, ["HUD text"])
    assert category.decide([replace(p, screen_anchors=9)]) == det
    assert category.probe_source(src) == p
    assert (cdir / category.CACHE_FILE).read_bytes() == record


def test_a_failed_text_check_gets_no_cue(text_check, monkeypatch):
    """A full install keeps today's result when its HUD text check fails: no cue."""
    from cutpoint import cache  # noqa: PLC0415

    def broken(*a):
        raise RuntimeError("onnxruntime fell over")

    monkeypatch.setattr(category, "text_probe", broken)
    p = category.probe_source(text_check.src)
    assert (p.hud_text, p.screen_anchors, text_check.cues) == (None, None, 0)
    assert not (cache.source_dir(text_check.src) / category.ANCHOR_FILE).exists()


def test_a_failed_cue_is_tried_once_and_not_cached(text_check, monkeypatch, caplog):
    from cutpoint import cache  # noqa: PLC0415

    monkeypatch.setattr(category, "ocr_installed", lambda: False)

    def broken(*a):
        text_check.cues += 1
        raise RuntimeError("decode fell over")

    monkeypatch.setattr(category, "anchor_probe", broken)
    with caplog.at_level(logging.WARNING):
        assert category.probe_source(text_check.src).screen_anchors is None
    assert "Screen-text cue of a.mp4 failed" in caplog.text
    assert category.probe_source(text_check.src).screen_anchors is None
    assert text_check.cues == 1
    assert not (cache.source_dir(text_check.src) / category.ANCHOR_FILE).exists()


def test_a_cue_with_too_few_frames_is_tried_once(text_check, monkeypatch):
    monkeypatch.setattr(category, "ocr_installed", lambda: False)

    def short(*a):
        text_check.cues += 1

    monkeypatch.setattr(category, "anchor_probe", short)
    assert category.probe_source(text_check.src).screen_anchors is None
    assert category.probe_source(text_check.src).screen_anchors is None
    assert text_check.cues == 1


def test_a_cue_short_of_memory_stops_the_detection(text_check, monkeypatch):
    from cutpoint.errors import LowMemoryError  # noqa: PLC0415

    monkeypatch.setattr(category, "ocr_installed", lambda: False)

    def short(*a):
        raise LowMemoryError("Not enough memory")

    monkeypatch.setattr(category, "anchor_probe", short)
    with pytest.raises(LowMemoryError):
        category.probe_source(text_check.src)
    assert not category._ANCHORS_FAILED


# --------------------------------------------------------------------------- vision (#898)
class FakeVision:
    """Stands in for :class:`category.FootageVision`: ``answer`` to every question."""

    def __init__(self, answer: str | None = "gameplay", key: str = "qwen@local") -> None:
        self.answer, self._key, self.asked = answer, key, 0

    def key(self) -> str:
        return self._key

    def ready(self) -> bool:
        return True

    def ask(self, io, duration):
        self.asked += 1
        return self.answer


def test_the_vision_answer_is_cached_per_source_and_model(faked_probe):
    from cutpoint import cache  # noqa: PLC0415

    src = faked_probe.src
    assert category.probe_source(src).vision is None  # no vision: nothing asked
    vision = FakeVision()
    first = category.probe_source(src, vision)
    assert first.vision == "gameplay" and vision.asked == 1
    assert "vision" not in cache.load_json(cache.source_dir(src) / category.CACHE_FILE)
    assert category.probe_source(src, vision) == first and vision.asked == 1  # cached
    assert category.decide([first]).decided_by == ["vision"]
    other = FakeVision("camera", key="llava@local")  # another model asks again
    assert category.probe_source(src, other).vision == "camera" and other.asked == 1
    assert category.probe_source(src, vision).vision == "gameplay"  # both kept


def test_no_answer_is_not_cached(faked_probe):
    vision = FakeVision(None)
    assert category.probe_source(faked_probe.src, vision).vision is None
    category.probe_source(faked_probe.src, vision)
    assert vision.asked == 2


def test_sure_footage_is_never_shown_to_the_vision_model(faked_probe, monkeypatch):
    """A 2:1 frame of 360 video is sure general and is not asked (camera tags are, #958)."""
    from types import SimpleNamespace  # noqa: PLC0415

    from cutpoint import ffmpeg_utils  # noqa: PLC0415

    size = SimpleNamespace(duration=60.0, width=5760, height=2880)
    monkeypatch.setattr(ffmpeg_utils, "probe", lambda p: size)
    vision = FakeVision()
    found = category.probe_source(faked_probe.src, vision)
    assert found.equirect and found.vision is None
    assert vision.asked == 0


REAL_CLIENT = category.FootageVision.client  # conftest stubs it: no host model in tests


class StubRater:
    """A :class:`cutpoint.vision.VisionRater` that answers ``reply`` (or raises it)."""

    instances: ClassVar[list[StubRater]] = []

    def __init__(self, consent=None, available=True, reply=None) -> None:
        self.model, self.host, self.consent_needed = "qwen3.5:9b", "127.0.0.1:11434", consent
        self._available, self.reply, self.asks, self.unloaded = available, reply, [], 0
        self.warned = 0

    def warn_consent(self) -> str:
        self.warned += 1
        return "needs consent"

    def available(self) -> bool:
        return self._available

    def ask_json(self, prompt, images, schema, **kw):
        self.asks.append((prompt, len(images), schema))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def unload(self) -> None:
        self.unloaded += 1


@pytest.fixture
def stub_rater(monkeypatch):
    """The real :meth:`FootageVision.client` with a :class:`StubRater` (``state.kw`` its
    arguments) and frame grabs faked; full edition."""
    from types import SimpleNamespace  # noqa: PLC0415

    from cutpoint import extras, vision  # noqa: PLC0415

    state = SimpleNamespace(kw={"reply": {"footage": "gameplay"}}, raters=[])

    def make(url=None, model=None):
        state.raters.append(StubRater(**state.kw))
        state.raters[-1].args = (url, model)
        return state.raters[-1]

    monkeypatch.setattr(category.FootageVision, "client", REAL_CLIENT)
    monkeypatch.setattr(vision, "VisionRater", make)
    monkeypatch.setattr(extras, "edition", lambda: "full")
    monkeypatch.setattr(category, "vision_frames", lambda io, d: [b"jpg"] * category.VISION_FRAMES)
    return state


def test_the_question_is_generic_and_the_model_freed(stub_rater):
    with category.footage_vision() as vision:
        assert vision.ask(Path("a.mp4"), 60.0) == "gameplay"
        assert vision.ask(Path("b.mp4"), 60.0) == "gameplay"
        assert vision.key() == "qwen3.5:9b@127.0.0.1:11434"
    [rater] = stub_rater.raters  # one client, checked once, freed once
    assert rater.unloaded == 1 and len(rater.asks) == 2
    prompt, n, schema = rater.asks[0]
    assert n == category.VISION_FRAMES and f"{n} frames" in prompt
    assert schema["properties"]["footage"]["enum"] == list(category.VISION_ANSWERS)
    with category.footage_vision(False) as vision:
        assert vision is None


@pytest.mark.parametrize(
    ("kw", "edition", "says"),
    [
        ({"consent": "vision-v2:openai@api.openai.com"}, "full", "needs your consent"),
        ({"available": False}, "full", None),  # the rater logs why (INFO)
        ({}, "lite", "not in Cutpoint lite"),
        ({"reply": category.LowMemoryError("Not enough memory")}, "full", "Not enough memory"),
    ],
)
def test_without_a_model_the_question_is_skipped_quietly(
    stub_rater, monkeypatch, caplog, kw, edition, says
):
    """#893's quiet: one INFO line, no WARNING, no consent warning, and no more tries."""
    from cutpoint import extras  # noqa: PLC0415

    stub_rater.kw = {"reply": {"footage": "gameplay"}, **kw}
    monkeypatch.setattr(extras, "edition", lambda: edition)
    with caplog.at_level(logging.INFO), category.footage_vision() as vision:
        assert vision.ask(Path("a.mp4"), 60.0) is None
        assert vision.ask(Path("b.mp4"), 60.0) is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    if says:
        [line] = [r.getMessage() for r in caplog.records]
        assert says in line
    assert sum(len(r.asks) for r in stub_rater.raters) <= 1


def test_an_explicit_vision_warns_once_of_a_missing_consent(stub_rater, caplog):
    """--vision / "vision": true asked for it: the consent warning, once; no more tries."""
    stub_rater.kw = {"reply": {"footage": "gameplay"}, "consent": "vision-v2:x@y"}
    with caplog.at_level(logging.INFO), category.footage_vision(True) as vision:
        assert vision.ask(Path("a.mp4"), 60.0) is None
        assert vision.ask(Path("b.mp4"), 60.0) is None
    [rater] = stub_rater.raters
    assert rater.warned == 1 and rater.asks == []


def test_the_run_s_ollama_options_reach_the_rater(stub_rater):
    with category.footage_vision(None, "http://h:1", "llava:7b") as vision:
        assert vision.ask(Path("a.mp4"), 60.0) == "gameplay"
    assert stub_rater.raters[0].args == ("http://h:1", "llava:7b")


def test_detect_hands_its_ollama_options_to_the_vision_client(faked_probe, stub_rater):
    src = str(faked_probe.src)
    argv = ["detect", src, "--ollama-url", "http://h:1", "--ollama-model", "llava:7b"]
    assert cli.main(argv) == 0
    assert stub_rater.raters[0].args == ("http://h:1", "llava:7b")


def test_detect_rejects_an_invalid_ollama_port(faked_probe, stub_rater):
    with pytest.raises(SystemExit) as exc:
        cli.main(["detect", str(faked_probe.src), "--ollama-url", "myhost:99999"])
    assert exc.value.code == 2
    assert stub_rater.raters == []


def test_no_vision_asks_nothing_and_the_default_asks(faked_probe, stub_rater):
    src = str(faked_probe.src)
    assert cli.main(["detect", src, "--no-vision"]) == 0
    assert stub_rater.raters == []
    assert cli.main(["detect", src]) == 0
    assert len(stub_rater.raters) == 1 and stub_rater.raters[0].asks


def test_the_config_s_vision_false_asks_nothing(faked_probe, stub_rater, home):
    from conftest import set_config  # noqa: PLC0415

    set_config(home, vision=False)
    assert cli.main(["detect", str(faked_probe.src)]) == 0
    assert stub_rater.raters == []
    assert cli.main(["detect", str(faked_probe.src), "--vision"]) == 0
    assert len(stub_rater.raters) == 1


def test_a_hosted_model_id_found_once_ready_is_looked_up_before_asking(faked_probe):
    """A cache hit under the model id that only the readiness check resolves needs no
    request (#898)."""
    from cutpoint import cache  # noqa: PLC0415

    class Hosted(FakeVision):
        def __init__(self) -> None:
            super().__init__("camera", key="@api")
            self.checked = False

        def ready(self) -> bool:
            self.checked, self._key = True, "gpt-x@api"
            return True

    cache.save_json(
        cache.source_dir(faked_probe.src) / category.VISION_FILE,
        {"version": category.VISION_VERSION, "answers": {"gpt-x@api": "gameplay"}},
    )
    vision = Hosted()
    assert category.probe_source(faked_probe.src, vision).vision == "gameplay"
    assert vision.checked and vision.asked == 0


def test_a_shared_vision_is_not_retried_by_the_second_detection(faked_probe, stub_rater):
    stub_rater.kw = {"reply": None}  # the request fails
    with category.footage_vision() as shared:
        category.detect([faked_probe.src], shared=shared)
        category.detect([faked_probe.src], shared=shared)
    assert len(stub_rater.raters) == 1 and len(stub_rater.raters[0].asks) == 1


def test_an_odd_answer_is_no_answer(stub_rater):
    stub_rater.kw = {"reply": {"footage": "a racing game"}}
    with category.footage_vision() as vision:
        assert vision.ask(Path("a.mp4"), 60.0) is None


def test_hud_text_alone_leaves_untagged_footage_unsure():
    """#513: a podcast's fixed cameras show fixed signs and labels: text anchors with no static
    HUD and no recorder tags. That alone is unsure gameplay, so the vision model is asked; a
    recorder's tags or some static HUD with the text stay sure."""
    many = category.TEXT_ANCHORS + 2
    alone = probe(**{**GAME, "handler": "", "hud_static": 0.0, "hud_text": many})
    det = category.decide([alone], {"a.mp4": 0.9})
    assert (det.category, det.ambiguous) == (GAMEPLAY, True)
    assert category.needs_vision(alone)
    det = category.decide([replace(alone, vision=category.V_CAMERA)], {"a.mp4": 0.9})
    assert (det.category, det.ambiguous) == (TALK, False)
    for sure in (replace(alone, handler="VideoHandle"), replace(alone, hud_static=0.002)):
        det = category.decide([sure])
        assert (det.category, det.ambiguous, det.decided_by) == (GAMEPLAY, False, ["HUD text"])


# ---- the Action category: a sport filmed by a riding or following camera, no telemetry


def test_a_long_narrated_source_the_model_calls_action_is_still_talk():
    """Talk is checked first: speech over the talk threshold wins over the vision answer."""
    det = category.decide([probe(**UNSURE, vision="action")], {"a.mp4": 0.7})
    assert (det.category, det.ambiguous, det.decided_by) == (TALK, False, ["speech"])
    quiet = category.decide([probe(**UNSURE, vision="action")], {"a.mp4": 0.1})
    assert (quiet.category, quiet.pending) == (ACTION, False)


def test_action_and_general_footage_of_equal_length_is_general():
    probes = [probe("a.mp4", 300.0, **UNSURE, vision="action"), probe("b.mp4", 300.0, **CAMERA)]
    det = category.decide(probes, {})
    assert det.category == GENERAL
    bigger = [probe("a.mp4", 400.0, **UNSURE, vision="action"), probe("b.mp4", 300.0, **CAMERA)]
    assert category.decide(bigger, {}).category == ACTION


@pytest.mark.parametrize(
    ("talk_s", "general_s", "action_s", "expect"),
    [
        (1000.0, 800.0, 800.0, GENERAL),  # talk is weighed against general + action pooled
        (1000.0, 300.0, 300.0, TALK),
        (500.0, 300.0, 400.0, ACTION),
        (500.0, 400.0, 400.0, GENERAL),  # action must beat general; a tie goes to general
    ],
)
def test_talk_is_weighed_against_general_and_action_pooled(talk_s, general_s, action_s, expect):
    probes = [
        probe("talk.mp4", talk_s, **CAMERA),
        probe("gen.mp4", general_s, **CAMERA),
        probe("act.mp4", action_s, **UNSURE, vision="action"),
    ]
    det = category.decide(probes, {"talk.mp4": 0.7, "gen.mp4": 0.1, "act.mp4": 0.1})
    assert det.category == expect


def test_action_and_general_footage_is_one_style_with_no_disagreement():
    probes = [probe("run.mp4", 600.0, **UNSURE, vision="action"), probe("b.mp4", 400.0, **CAMERA)]
    det = category.decide(probes, {})
    assert det.category == ACTION and det.disagree == []
    rec = json.loads(json.dumps(det.record("hype", "action", None)))
    assert category.disagreeing(rec) == []
    # the same holds when general is the project's category
    flipped = [probe("run.mp4", 400.0, **UNSURE, vision="action"), probe("b.mp4", 600.0, **CAMERA)]
    det = category.decide(flipped, {})
    assert det.category == GENERAL and det.disagree == []
    assert category.disagreeing(json.loads(json.dumps(det.record("hype", "general", None)))) == []


def test_a_gameplay_minority_beside_action_footage_is_still_named():
    probes = [
        probe("run.mp4", 800.0, **UNSURE, vision="action"),
        probe("game.mp4", 200.0, **GAME),
    ]
    det = category.decide(probes, {})
    assert det.category == ACTION and det.disagree == ["game.mp4"]
    rec = json.loads(json.dumps(det.record("hype", "action", None)))
    found = category.disagreeing(rec)
    assert [s["path"] for s in found] == ["game.mp4"] and found[0]["category"] == GAMEPLAY


def test_disagreeing_counts_a_missing_partner_score_as_zero():
    srcs = [{"path": "g.mp4", "category": GAMEPLAY}, {"path": "a.mp4", "category": ACTION}]
    rec = {"category": ACTION, "scores": {ACTION: 0.5}, "sources": srcs}  # no general score
    assert [s["path"] for s in category.disagreeing(rec)] == ["g.mp4"]
    rec["scores"] = {ACTION: 0.5, GENERAL: 0.0}
    assert [s["path"] for s in category.disagreeing(rec)] == ["g.mp4"]
    # the project's own category score missing: nothing is named
    rec["scores"] = {GENERAL: 0.5}
    assert category.disagreeing(rec) == []


def test_disagreeing_on_an_old_general_record_without_an_action_score():
    srcs = [{"path": "g.mp4", "category": GAMEPLAY}, {"path": "b.mp4", "category": GENERAL}]
    scores = {GENERAL: 0.6, GAMEPLAY: 0.4}  # version 3: no ``action`` key
    rec = {"category": GENERAL, "scores": scores, "sources": srcs}
    assert [s["path"] for s in category.disagreeing(rec)] == ["g.mp4"]


def test_telemetry_in_under_half_the_footage_is_not_pooled_into_action():
    probes = [
        probe("cam.mp4", 200.0, telemetry=True),
        probe("run.mp4", 400.0, **UNSURE, vision="action"),
    ]
    det = category.decide(probes, {})
    assert det.category == ACTION  # not action cam: telemetry is a third of the footage
    assert det.disagree == ["cam.mp4"]  # the telemetry source is the minority named
    assert det.scores[ACTION] == pytest.approx(400 / 600, abs=1e-3)
    assert det.scores[ACTION_CAM] == pytest.approx(200 / 600, abs=1e-3)


@pytest.mark.parametrize(
    ("duration", "speech", "pending"),
    [
        (600.0, None, True),  # long enough for talk, not transcribed yet
        (600.0, {"a.mp4": 0.1}, False),
        (60.0, None, False),  # too short for talk
    ],
)
def test_a_long_untranscribed_action_source_leaves_talk_pending(duration, speech, pending):
    det = category.decide([probe(duration=duration, **UNSURE, vision="action")], speech)
    assert (det.category, det.pending) == (ACTION, pending)
    assert ("talk is checked after transcription" in det.reason) is pending


def test_the_detection_record_has_its_own_version_apart_from_the_probe_cache():
    det = category.decide([probe(**UNSURE, vision="action")], {"a.mp4": 0.1})
    rec = det.record("hype", "action", None)
    assert rec["version"] == category.DETECTION_VERSION == 4
    assert category.VERSION == 3 and category.CACHE_FILE == "category-v3.json"
    assert rec["category"] == "action"


def test_the_vision_cache_is_a_new_file_so_camera_answers_are_asked_again():
    """3 (#958): the question calls fast first-person motion action; v2 answers are not read."""
    assert category.VISION_VERSION == 3
    assert category.VISION_FILE == "category-vision-v3.json"
    assert "action" in category.VISION_ANSWERS
    assert "first-person" in category.VISION_PROMPT and "FPV" in category.VISION_PROMPT


def test_a_v2_answer_is_asked_again(faked_probe):
    from cutpoint import cache  # noqa: PLC0415

    cdir = cache.source_dir(faked_probe.src)
    cache.save_json(
        cdir / "category-vision-v2.json", {"version": 2, "answers": {"qwen@local": "camera"}}
    )
    vision = FakeVision("action")
    assert category.probe_source(faked_probe.src, vision).vision == "action"
    assert vision.asked == 1


def test_the_vision_answer_action_is_cached(faked_probe):
    vision = FakeVision("action")
    first = category.probe_source(faked_probe.src, vision)
    assert first.vision == "action" and vision.asked == 1
    assert category.probe_source(faked_probe.src, vision).vision == "action" and vision.asked == 1


def test_the_vision_question_names_no_sport():
    """No hand-made per-activity presets: the question describes the kind of camera work."""
    prompt = category.VISION_PROMPT.lower()
    for word in ("ski", "bike", "biking", "cycling", "surf", "snowboard", "skate", "mtb", "climb"):
        assert not re.search(rf"\b{word}", prompt), word
    assert "action:" in prompt


def test_auto_resolves_the_action_category_to_the_action_style():
    assert styles.resolve_auto(ACTION, category.HYPE) == "action"
    assert styles.resolve_auto(ACTION, category.FUNNY) == "funny"
    assert set(styles.AUTO_STYLES) == set(category.CATEGORIES)
    assert category.label(ACTION) == "Action"


@pytest.mark.parametrize(
    ("answer", "expect"),
    [
        ("action", (ACTION, ["vision"])),
        ("camera", (GENERAL, ["camera tags"])),
        ("gameplay", (GENERAL, ["camera tags"])),
        (None, (GENERAL, ["camera tags"])),  # no answer
    ],
)
def test_a_camera_tagged_source_without_telemetry_is_shown_to_the_vision_model(
    faked_probe, monkeypatch, answer, expect
):
    """#958: a DJI or other camera original without telemetry has only its tags, as a phone
    clip or a re-export has; the vision model can still tell it is action."""
    from cutpoint import ffmpeg_utils  # noqa: PLC0415

    tags = {"streams": [{"codec_type": "video", "tags": {"handler_name": "DJI.AVC"}}]}
    monkeypatch.setattr(ffmpeg_utils, "ffprobe_json", lambda *a, **k: tags)
    vision = FakeVision(answer)
    found = category.probe_source(faked_probe.src, vision)
    assert found.camera_marks and vision.asked == 1
    det = category.decide([found], {})
    assert (det.category, det.ambiguous, det.decided_by) == (expect[0], False, expect[1])
    without = category.decide([category.probe_source(faked_probe.src)], {})
    assert (without.category, without.decided_by) == (GENERAL, ["camera tags"])


# --------------------------------------------------------------------------- parallel grabs (#946)
class _Live:
    """Counts the calls in flight; ``hit()`` holds each a moment so overlapping ones show."""

    def __init__(self) -> None:
        import threading  # noqa: PLC0415

        self.lock, self.now, self.max = threading.Lock(), 0, 0

    def hit(self) -> None:
        import time  # noqa: PLC0415

        with self.lock:
            self.now += 1
            self.max = max(self.max, self.now)
        time.sleep(0.05)
        with self.lock:
            self.now -= 1


def _both_worker_counts(monkeypatch, run, live):
    """``run()`` with one worker and with GRAB_WORKERS: ``{workers: (result, max in flight)}``."""
    out = {}
    for workers in (1, category.GRAB_WORKERS):
        monkeypatch.setattr(category, "GRAB_WORKERS", workers)
        live.now = live.max = 0
        out[workers] = (run(), live.max)
    return out


def test_frame_probe_grabs_in_parallel_with_the_sequential_result(monkeypatch):
    from cutpoint import ffmpeg_utils  # noqa: PLC0415

    live = _Live()

    def decode_raw(io, size, vf, start, **kw):
        live.hit()
        if round(start) % 7 == 3:  # some grabs fail
            raise category.FFmpegError("boom")
        yield start, np.full((size[1], size[0]), start, np.uint8)
        yield start + 1, np.full((size[1], size[0]), start + 1, np.uint8)

    seen = {}
    monkeypatch.setattr(ffmpeg_utils, "decode_raw", decode_raw)
    monkeypatch.setattr(
        category, "frame_measures", lambda s, p: seen.update(s=s, p=p) or {"s": len(s)}
    )

    def run():
        category._frame_probe(Path("v.mp4"), 100.0, 640, 360)
        return [float(f.mean()) for f in seen["s"]], [
            [float(f.mean()) for f in p] for p in seen["p"]
        ]

    runs = _both_worker_counts(monkeypatch, run, live)
    assert runs[1][1] == 1 and runs[category.GRAB_WORKERS][1] > 1
    assert runs[1][0] == runs[category.GRAB_WORKERS][0]
    assert 0 < len(runs[1][0][0]) < category.SPREAD  # the failed grabs were left out


def test_text_and_vision_frames_grab_in_parallel_with_the_sequential_result(monkeypatch):
    from cutpoint import ffmpeg_utils  # noqa: PLC0415

    live = _Live()

    def decode_raw(io, size, start, **kw):
        live.hit()
        if 25 <= start < 40:
            raise category.FFmpegError("boom")
        yield start, np.full((size[1], size[0], 3), start, np.uint8)

    def grab_jpeg(io, t, width):
        live.hit()
        if 25 <= t < 40:
            raise category.FFmpegError("boom")
        return f"{t:.2f}".encode()

    monkeypatch.setattr(ffmpeg_utils, "decode_raw", decode_raw)
    monkeypatch.setattr(ffmpeg_utils, "grab_frame_jpeg", grab_jpeg)
    for name, run, count in (
        (
            "text",
            lambda: [float(f.mean()) for f in category._text_frames(Path("v"), 100.0, 64, 48)],
            7,
        ),
        ("vision", lambda: category.vision_frames(Path("v"), 100.0), 5),
    ):
        runs = _both_worker_counts(monkeypatch, run, live)
        assert runs[1][1] == 1 and runs[category.GRAB_WORKERS][1] > 1, name
        assert runs[1][0] == runs[category.GRAB_WORKERS][0], name
        assert len(runs[1][0]) == count, name  # one failed grab each
