"""Stream chat in a discovered feed region: the chat label's thresholds, the lines it counts
and the events a chat region gives up (synthetic reads, invented names; no OCR)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cutpoint.events import killfeed, ocr_raw
from cutpoint.events.common import Source
from cutpoint.events.killfeed import (
    CHAT_LINES_PER_HOUR,
    CHAT_MIN_FOOTAGE_S,
    CHAT_UNSURE_LINES_PER_HOUR,
    FrameRec,
    OcrInterpretation,
    OcrParams,
    _chat_lines,
    chat_label,
    detect_ocr_events,
)
from cutpoint.events.regions import Region, preset_from_dict

# ------------------------------------------------------------------------ chat_label


@pytest.mark.parametrize(
    ("lines", "hours", "label"),
    [
        (6000, 1.0, "chat"),  # at the chat rate
        (5999, 1.0, "unsure"),  # just below it
        (3000, 1.0, "unsure"),  # at the unsure rate
        (2999, 1.0, "feed"),  # just below it
        (40, 1.0, "feed"),  # a kill feed
        (1500, 0.25, "chat"),  # 6000/h in 15 minutes
        (1499, 0.25, "unsure"),
        (750, 0.25, "unsure"),  # 3000/h
        (749, 0.25, "feed"),
        (1000, 599 / 3600, "short"),  # under 10 minutes: too short to tell, at a chat rate
        (100, 599 / 3600, "feed"),  # under 10 minutes, a feed rate
        (1000, 600 / 3600, "chat"),  # exactly 10 minutes counts
    ],
)
def test_chat_label_thresholds_and_footage_floor(lines: int, hours: float, label: str) -> None:
    assert chat_label(lines, hours)[0] == label


def test_chat_label_rate_and_no_footage() -> None:
    assert chat_label(12000, 2.0) == ("chat", 6000.0)
    assert chat_label(500, 0.0) == ("feed", 0.0)
    assert CHAT_MIN_FOOTAGE_S == 600.0


# ---------------------------------------------------------------------- _chat_lines


def test_chat_lines_collapse_variants_and_ignore_weak_items() -> None:
    box = [[0, 0], [10, 0], [10, 10]]
    items = [
        (box, "Mango Fan: GG, nice!", 0.9),
        (box, "mango fan gg nice", 0.8),  # case and punctuation variants of the first
        (box, "MANGO FAN... GG NICE!!", 0.7),
        (box, "Hi", 0.95),  # fewer than 3 letters or digits
        (box, "?!.", 0.95),  # no letters at all
        (box, "Totally different line", 0.59),  # not sure enough
        (box, "Another line", 0.6),  # at the confidence floor
        (box, "ok1", 0.9),  # three characters count
    ]
    assert _chat_lines(items) == {"mango fan gg nice", "another line", "ok1"}


def test_chat_lines_count_non_latin_text() -> None:
    box = [[0, 0], [10, 0], [10, 10]]
    items = [
        (box, "안녕하세요 여러분", 0.9),
        (box, "こんにちは みなさん", 0.9),
        (box, "大家好今天玩什么", 0.9),
    ]
    assert len(_chat_lines(items)) == 3


# ------------------------------------------------------ OcrInterpretation drops chat

FPS = 1.0
SEC = 1200  # 20 min of 1 fps frames: 1/3 h
HOURS = SEC / FPS / 3600
KILL_AT = (100, 400, 700)
CHAT_BOX = Region("chat_box", 0, 0, 400, 400, "feed")
GAME_FEED = Region("game_feed", 1500, 0, 400, 400, "feed")
BOX = [[0, 0], [100, 0], [100, 20], [0, 20]]


def _kill(n: int) -> list[tuple[Any, str, float]]:
    return [(BOX, f"Pilot{n} destroyed Target{n}", 0.95)]


def _preset(tmp_path: Path, source: str | None) -> Any:
    data: dict[str, Any] = {"game": "tilefall", "regions": []}
    if source is not None:
        data["source"] = source
    path = tmp_path / "tilefall.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return preset_from_dict(data, path)


def _run(
    tmp_path: Path,
    rates: dict[str, int],
    source: str | None = "discovered",
    sec: int = SEC,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """20 minutes of frames: region ``r`` reads ``rates[r]`` distinct invented chat lines per
    frame, and every region reads one kill at each of KILL_AT."""
    regions = tuple(r for r in (CHAT_BOX, GAME_FEED) if r.name in rates)
    stats: dict[str, Any] = {}
    interp = OcrInterpretation(
        "rec.mp4",
        OcrParams(game="tilefall", fps=FPS),
        _preset(tmp_path, source),
        regions,
        (1920, 1080),
        stats,
    )
    for n in range(sec):
        recs = {}
        for r in regions:
            items = [(BOX, f"viewer{n}x{k} says hello{n}x{k}", 0.9) for k in range(rates[r.name])]
            recs[r.name] = {"g": "read", "items": (_kill(n) if n in KILL_AT else []) + items}
        interp.add(FrameRec(n=n, t=n / FPS, regions=recs))
    return interp.finish(), stats


def _kills(events: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in events:
        if e["type"] == "kill":
            out[e["region"]] = out.get(e["region"], 0) + 1
    return out


def test_chat_region_gives_no_events_and_other_regions_keep_theirs(tmp_path: Path) -> None:
    # 3 distinct lines per frame over 1/3 h: about 10800/h; the game feed reads only its kills
    events, stats = _run(tmp_path, {"chat_box": 3, "game_feed": 0})
    assert _kills(events) == {"game_feed": 3}
    info = stats["feed_regions"]
    assert set(info) == {"chat_box", "game_feed"}
    assert set(info["chat_box"]) == {"label", "lines_per_hour", "lines", "hours", "dropped"}
    assert info["chat_box"]["label"] == "chat"
    assert info["chat_box"]["dropped"] == {"kill": 3}
    assert info["chat_box"]["lines"] >= 3 * SEC
    assert info["chat_box"]["lines_per_hour"] >= CHAT_LINES_PER_HOUR
    assert info["chat_box"]["hours"] == round(HOURS, 3)
    assert info["game_feed"]["label"] == "feed"
    assert info["game_feed"]["dropped"] == {}


def test_unsure_region_stays_a_feed_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="cutpoint.events.killfeed")
    # one line per frame: about 3600/h, between the two rates
    events, stats = _run(tmp_path, {"chat_box": 1, "game_feed": 0})
    info = stats["feed_regions"]["chat_box"]
    assert CHAT_UNSURE_LINES_PER_HOUR <= info["lines_per_hour"] < CHAT_LINES_PER_HOUR
    assert info["label"] == "feed"
    assert info["unsure"] is True
    assert "chat_like" not in info
    assert info["dropped"] == {}
    assert _kills(events) == {"chat_box": 3, "game_feed": 3}
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("may be stream chat" in m for m in warned)


@pytest.mark.parametrize("source", [None, "user"])
def test_hand_made_preset_is_never_relabelled_chat(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, source: str | None
) -> None:
    caplog.set_level(logging.INFO, logger="cutpoint.events.killfeed")
    events, stats = _run(tmp_path, {"chat_box": 3, "game_feed": 0}, source=source)
    info = stats["feed_regions"]["chat_box"]
    assert info["label"] == "feed"
    assert info["chat_like"] is True
    assert info["dropped"] == {}
    assert _kills(events) == {"chat_box": 3, "game_feed": 3}
    assert any(
        "not discovered" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    )


def test_real_feed_with_few_distinct_lines_keeps_its_kills(tmp_path: Path) -> None:
    # a game log that shows the same few lines over and over is a handful of distinct lines
    events, stats = _run(tmp_path, {"game_feed": 0})
    assert _kills(events) == {"game_feed": 3}
    info = stats["feed_regions"]["game_feed"]
    assert info["label"] == "feed"
    assert "unsure" not in info
    assert "chat_like" not in info
    assert info["lines"] == 3  # only the three kill lines


def test_short_footage_keeps_the_feed_with_a_warning_and_its_events(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="cutpoint.events.killfeed")
    # 500 s at a chat rate: under the 10 minute floor
    events, stats = _run(tmp_path, {"chat_box": 3, "game_feed": 0}, sec=500)
    info = stats["feed_regions"]["chat_box"]
    assert info["label"] == "feed"
    assert info["too_short"] is True
    assert info["dropped"] == {}
    assert info["lines_per_hour"] >= CHAT_LINES_PER_HOUR
    assert _kills(events) == {"chat_box": 2, "game_feed": 2}
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("too short to tell" in m for m in warned)


# ------------------------------------------- "det" gate: order against the later passes


def _det_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Like :func:`_run` with "det"-gated regions; spies on what ``drop_labels`` and
    ``menu_rules`` (which makes ``menu_spans``) are given."""
    chat = Region("chat_box", 0, 0, 400, 400, "feed", gate="det")
    game = Region("game_feed", 1500, 0, 400, 400, "feed", gate="det")
    seen: dict[str, Any] = {}
    real_drop = killfeed.Interpreter.drop_labels
    real_rules = killfeed.menu_rules

    def drop_labels(self: Any) -> None:
        seen["labels"] = {e.get("region") for e in self.tracker.events}
        real_drop(self)

    def menu_rules(frames: Any, **kw: Any) -> Any:
        seen["feed_times"] = set(kw["feed_times"] or ())
        return real_rules(frames, **kw)

    monkeypatch.setattr(killfeed.Interpreter, "drop_labels", drop_labels)
    monkeypatch.setattr(killfeed, "menu_rules", menu_rules)
    stats: dict[str, Any] = {}
    interp = OcrInterpretation(
        "rec.mp4",
        OcrParams(game="tilefall", fps=FPS),
        _preset(tmp_path, "discovered"),
        (chat, game),
        (1920, 1080),
        stats,
    )
    for n in range(SEC):
        items = [(BOX, f"viewer{n}x{k} says hello{n}x{k}", 0.9) for k in range(3)]
        chat_rec = {"g": "read", "items": (_kill(n) if n in KILL_AT else []) + items}
        game_rec = {"g": "read", "items": _kill(n + 1) if n == 300 else []}
        interp.add(FrameRec(n=n, t=n / FPS, regions={"chat_box": chat_rec, "game_feed": game_rec}))
    return interp.finish(), stats, seen


def test_det_gated_chat_is_dropped_before_labels_and_menu_spans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, stats, seen = _det_run(tmp_path, monkeypatch)
    assert stats["feed_regions"]["chat_box"]["label"] == "chat"
    assert stats["feed_regions"]["chat_box"]["dropped"] == {"kill": 3}
    assert _kills(events) == {"game_feed": 1}
    assert seen["labels"] == {"game_feed"}  # drop_labels never saw a chat event
    assert seen["feed_times"] == {300.0}  # nor did the menu test (chat kills at 100, 400, 700)
    assert "menu_spans" in stats


# --------------------------------------------- detect_ocr_events: replay, resume, avoid

DUR = 1200.0
CHAT_RECT = (0, 0, 400, 400)
HUD_RECT = (1000, 800, 500, 100)


class _Stop(Exception):
    pass


class _Engine:
    """Fake OCR: 4 invented chat lines per second in the chat region (two of them with a
    kill line), a notice now and then in the HUD region."""

    def __init__(self) -> None:
        self.t = 0.0

    def _rows(self, crop: Any) -> list[tuple[str, list[list[int]]]]:
        h, w = crop.shape[:2]
        if (w, h) == CHAT_RECT[2:]:
            n = round(self.t)
            out = []
            for k in range(4):
                text = f"viewer{n}x{k} says hello{n}x{k}"
                if n in (100, 700) and k == 0:
                    text = f"Alder{n} destroyed Kip{n}"
                x1 = 300 + (n % 7) * 12 if k == 3 else 300
                y = 20 + 30 * k
                out.append((text, [[5, y], [x1, y], [x1, y + 24], [5, y + 24]]))
            return out
        if int(self.t) % 50 < 5:
            return [("Hold the line", [[5, 5], [200, 5], [200, 29], [5, 29]])]
        return []

    def detect(self, crop: Any, scale: float = 1.0) -> list[Any]:
        return [b for _, b in self._rows(crop)]

    def read_rows(self, crop: Any, rows: Any, min_score: float, pad: float = 6.0) -> list[Any]:
        return [(b, t, 0.95) for t, b in self._rows(crop)]

    def __call__(self, crop: Any, m: Any) -> list[Any]:
        return [(b, t, 0.9) for t, b in self._rows(crop)]


def _detect(
    cdir: Path,
    mp: pytest.MonkeyPatch,
    engine: _Engine | None,
    stop: int | None = None,
    avoid: list[tuple[float, float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def region(name: str, r: tuple[int, ...], kind: str) -> dict[str, Any]:
        x, y, w, h = r
        return {"name": name, "x": x, "y": y, "w": w, "h": h, "kind": kind, "gate": "det"}

    data = {
        "game": "tilefall",
        "source": "discovered",
        "bigtext": False,
        "banner_phrases": ["Hold the line"],
        "regions": [region("chat", CHAT_RECT, "feed"), region("hud", HUD_RECT, "notice")],
    }
    path = cdir / "tilefall.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    mp.setattr(killfeed, "load_preset", lambda *_a, **_k: preset_from_dict(data, path))
    mp.setattr(killfeed, "probe", lambda _p: SimpleNamespace(width=1920, height=1080, duration=DUR))
    mp.setattr(killfeed, "decoder_for", lambda _p: None)
    made = [0]

    def frames(
        path: Any,
        fps: float,
        rect: tuple[int, int, int, int],
        hw: Any,
        errors: Any,
        start: float = 0.0,
        dur: float | None = None,
        cancel: Any = None,
        exit_codes: Any = None,
    ) -> Any:
        _, _, w, h = rect
        end = DUR if dur is None else start + dur
        k = 0
        while start + k / fps < end - 1e-6:
            if stop is not None and made[0] >= stop:
                raise _Stop
            made[0] += 1
            t = start + k / fps
            if engine is not None:
                engine.t = t
            yield t, np.full((h, w, 3), (int(t) * 53) % 256, np.uint8)
            k += 1

    mp.setattr(killfeed, "iter_frames", frames)
    src = Source(cdir / "rec.mp4", DUR, cdir, [])
    return detect_ocr_events(src, OcrParams(game="tilefall", fps=FPS), engine, avoid=avoid)


def _stable(stats: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in stats.items() if k not in ("elapsed_s", "replayed")}


def test_chat_replay_and_resume_match_the_live_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live, resumed = (tmp_path / n for n in ("live", "resumed"))
    for d in (live, resumed):
        d.mkdir()
    events, stats = _detect(live, monkeypatch, _Engine())
    assert stats["feed_regions"]["chat"]["label"] == "chat"
    assert events
    assert not [e for e in events if e.get("region") == "chat"]
    assert stats["feed_regions"]["chat"]["dropped"]
    assert stats["ocr_calls"]["chat"] > 0

    # interpretation changes: the cached raw read is interpreted again, no OCR (no engine)
    monkeypatch.setattr(killfeed, "GENERIC_VERSION", killfeed.GENERIC_VERSION + 1)
    events2, stats2 = _detect(live, monkeypatch, None)
    assert stats2["replayed"]
    assert events2 == events
    assert stats2["feed_regions"] == stats["feed_regions"]
    assert _stable(stats2) == _stable(stats)

    # a read interrupted in its second chunk, then resumed
    stop = ocr_raw.CHUNK_FRAMES + 400
    with pytest.raises(_Stop):
        _detect(resumed, monkeypatch, _Engine(), stop=stop)
    events3, stats3 = _detect(resumed, monkeypatch, _Engine())
    assert events3 == events
    assert stats3["feed_regions"] == stats["feed_regions"]


def test_avoid_rates_chat_on_decoded_hours_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 300 s skipped: 900 s decoded, 0.25 h, still over the 10 minute floor
    _, stats = _detect(tmp_path, monkeypatch, _Engine(), avoid=[(300.0, 600.0)])
    info = stats["feed_regions"]["chat"]
    assert info["hours"] == pytest.approx(0.25, abs=0.002), stats["frames"]
    assert info["label"] == "chat"
    assert info["lines_per_hour"] >= CHAT_LINES_PER_HOUR
    # 900 s skipped: 300 s decoded, too short to tell, however many lines
    short = tmp_path / "short"
    short.mkdir()
    _, stats = _detect(short, monkeypatch, _Engine(), avoid=[(0.0, 900.0)])
    info = stats["feed_regions"]["chat"]
    assert info["hours"] == round(300 / 3600, 3)
    assert info["label"] == "feed"
    assert info["too_short"] is True
    assert info["dropped"] == {}
