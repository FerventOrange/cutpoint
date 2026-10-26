"""Raw OCR cache (#315): interpretation changes replay the cached read without OCR, an
interrupted read resumes (fake decoder and OCR engine, invented HUD texts)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cutpoint.events import killfeed, ocr_raw, pipeline
from cutpoint.events.common import Source
from cutpoint.events.killfeed import OcrParams, detect_ocr_events
from cutpoint.events.regions import preset_from_dict

NOTICE = (300, 200, 1320, 700)  # a centred notice region ("det" gate at half size)
FEED = (1400, 40, 500, 400)  # a feed with coloured names ("det" gate)
BANNER = (600, 940, 700, 120)  # a banner region ("mask" gate, every 2nd frame)
ORIGIN = (300, 40)  # top left of the decoded HUD area
FPS = 3.0
RED, BLUE, WHITE = (0, 0, 255), (255, 0, 0), (255, 255, 255)


def _preset(**notice: Any) -> dict[str, Any]:
    def region(name: str, rect: tuple[int, ...], kind: str, **extra: Any) -> dict[str, Any]:
        x, y, w, h = rect
        return {"name": name, "x": x, "y": y, "w": w, "h": h, "kind": kind, **extra}

    return {
        "game": "zonewar",
        "regions": [
            region("notice", NOTICE, "notice", gate="det", det_scale=0.5, **notice),
            region("feed", FEED, "feed", gate="det"),
            region("banner", BANNER, "banner", every=2),
        ],
        "banner_phrases": ["Squad wiped", "Hull breached", "Zone held"],
        "feed": {"entry": "one_line", "colors": {"red": "enemy", "blue": "ally"}},
        "bigtext": False,
    }


def _screen(t: float) -> list[tuple[str, float, float, tuple[int, int, int] | None]]:
    """(text, x, y, colour) of what the screen shows at ``t``: a still menu at 8-20 s,
    coloured feed names from 24 s and 40 s, banners."""
    out: list[tuple[str, float, float, tuple[int, int, int] | None]] = []
    if 8 <= t < 20:
        out += [("Hull breached", 320, 220, None)]
        out += [(f"Tab {chr(65 + i)}", 320 + 180 * i, 860, None) for i in range(7)]
    else:
        out.append(("Hold", 900, 820, None))  # still on screen: the notice is kept alive
    if 4 <= t < 7 or 44 <= t < 47:
        out.append(("Squad wiped", 860, 400, None))
    if t >= 24:
        out += [("Orrin", 1410, 60, RED), ("Pell", 1640, 60, BLUE)]
    if t >= 40:
        out += [("Ash", 1410, 90, BLUE), ("Kip", 1640, 90, RED)]
    if 50 <= t < 53:
        out.append(("Zone held", 700, 960, WHITE))
    return out


def _width(text: str) -> int:
    return min(200, 12 * len(text))


def _paint(t: float, frame: np.ndarray) -> None:
    """Draw the coloured lines of the screen at ``t`` (decoded HUD area pixels)."""
    for text, x, y, colour in _screen(t):
        if colour is not None:
            x0, y0 = x - ORIGIN[0], y - ORIGIN[1]
            frame[y0 : y0 + 24, x0 : x0 + _width(text)] = colour


class _Engine:
    """Detection and recognition of the screen at the decoder's current time; a crop
    is told apart by its size. ``calls`` counts every OCR call."""

    def __init__(self) -> None:
        self.t = 0.0
        self.calls = 0

    def _boxes(self, crop: np.ndarray) -> list[tuple[str, list[list[float]]]]:
        self.calls += 1
        h, w = crop.shape[:2]
        x0, y0, _, _ = next(r for r in (NOTICE, FEED, BANNER) if (r[2], r[3]) == (w, h))
        return [
            (text, [[x - x0, y - y0], [x - x0 + bw, y - y0], [x - x0 + bw, y - y0 + 24]])
            for text, x, y, _ in _screen(self.t)
            if x0 <= x < x0 + w and y0 <= y < y0 + h
            for bw in [_width(text)]
        ]

    def detect(self, crop: np.ndarray, scale: float = 1.0) -> list[list[list[float]]]:
        return [b for _, b in self._boxes(crop)]

    def read_rows(self, crop, rows, min_score, pad=6.0) -> list[tuple[list, str, float]]:
        boxes = self._boxes(crop)
        out = []
        for y0, x0, y1, x1 in rows:
            for text, b in boxes:
                if y0 <= (b[0][1] + b[2][1]) / 2 <= y1 and x0 <= b[0][0] <= x1:
                    out.append((b, text, 0.95))
        return out

    def __call__(self, crop: np.ndarray, min_score: float) -> list[tuple[list, str, float]]:
        return [(b, text, 0.9) for text, b in self._boxes(crop)]


class _Stop(Exception):
    """The read is interrupted."""


def _run(
    cdir: Path,
    monkeypatch,
    engine: _Engine | None,
    preset: dict[str, Any] | None = None,
    fps: float = FPS,
    avoid: list[tuple[float, float]] | None = None,
    stop_after: int | None = None,
    starts: list[float] | None = None,
    fail_at: float | None = None,
    **kwargs: Any,
) -> tuple[list[dict], dict]:
    """Detect the OCR events of a 60 s source whose frames are moving gameplay (a new
    grey level every frame) except for the still menu at 8-20 s. The decoder starts
    at ``start`` (logged in ``starts``) and raises :class:`_Stop` after ``stop_after``
    frames; a decode starting at ``fail_at`` gives no frames."""
    data = preset or _preset()
    monkeypatch.setattr(killfeed, "load_preset", lambda *_a, **_k: preset_from_dict(data))
    monkeypatch.setattr(
        killfeed, "probe", lambda _p: SimpleNamespace(width=1920, height=1080, duration=60.0)
    )
    made = [0]

    def frames(path, fps, rect, hw, errors, start=0.0, dur=None):
        _, _, w, h = rect
        if starts is not None:
            starts.append(start)
        if start == fail_at:
            raise RuntimeError("no frames")
        end = 60.0 if dur is None else start + dur
        k = 0
        while start + k / fps < end - 1e-6:
            if stop_after is not None and made[0] >= stop_after:
                raise _Stop
            made[0] += 1
            t = start + k / fps  # as ffmpeg_utils.decode_raw stamps it
            tc = round(t * fps) / fps  # the screen's time, however t was summed up
            made_by = kwargs.get("make_engine")
            current = engine or (made_by.engines[-1] if made_by and made_by.engines else None)
            if current:
                current.t = tc
            frame = np.full((h, w, 3), 90 if 8 <= tc < 20 else round(t * fps) * 53 % 256, np.uint8)
            _paint(tc, frame)
            yield t, frame
            k += 1

    monkeypatch.setattr(killfeed, "iter_frames", frames)
    src = Source(cdir / "rec.mp4", 60.0, cdir, [])
    params = OcrParams(game="zonewar", fps=fps)
    return detect_ocr_events(src, params, engine, avoid=avoid, **kwargs)


def _untimed(stats: dict[str, Any]) -> dict[str, Any]:
    """``stats`` without timing and the replay flag."""
    return {k: v for k, v in stats.items() if k not in ("elapsed_s", "replayed")}


def _dirs(tmp_path: Path, *names: str) -> list[Path]:
    out = [tmp_path / n for n in names]
    for d in out:
        d.mkdir()
    return out


def _bump(monkeypatch) -> None:
    """An interpretation change."""
    monkeypatch.setattr(killfeed, "GENERIC_VERSION", killfeed.GENERIC_VERSION + 1)


def test_replay_equals_the_live_read(tmp_path: Path, monkeypatch) -> None:
    engine = _Engine()
    events, stats = _run(tmp_path, monkeypatch, engine)
    assert engine.calls > 0
    # what the replay must cover: colours read from the feed's pixels, the kept-alive
    # notice, the mask gate every 2nd frame, the static menu
    kills = [
        (e["t"], e["killer"], e["victim"], e["killer_team"], e["victim_team"])
        for e in events
        if e["type"] == "kill"
    ]
    assert kills == [
        (24.0, "Orrin", "Pell", "enemy", "ally"),
        (40.0, "Ash", "Kip", "ally", "enemy"),
    ]
    # read when the notice's layout changed, kept alive on the other frames
    hold = [(e["t"], e["end"], e["sightings"]) for e in events if e.get("text") == "Hold"]
    assert hold == [(0.0, 7.67, 3), (20.0, 59.67, 3)]
    assert [(e["text"], e["region"]) for e in events if e["type"] == "banner"][-1] == (
        "Zone held",
        "banner",
    )
    assert stats["static_frames"] == 36 and stats["det_calls"]["notice"] == 180
    assert 0 < stats["ocr_calls"]["banner"] <= 90
    raw = list(tmp_path.glob("ocr-raw-*"))
    assert len(raw) == 1 and json.loads((raw[0] / "meta.json").read_text())["complete"]

    _bump(monkeypatch)
    replay, replay_stats = _run(tmp_path, monkeypatch, None)  # no engine: no OCR
    assert replay == events
    assert replay_stats["replayed"] and "replayed" not in stats
    assert _untimed(replay_stats) == _untimed(stats)
    # and once more from the new result's own cache
    assert _run(tmp_path, monkeypatch, None) == (replay, replay_stats)


def test_interpretation_change_makes_no_ocr_call(tmp_path: Path, monkeypatch) -> None:
    ref, _ = _run(tmp_path, monkeypatch, _Engine())
    _bump(monkeypatch)
    engine = _Engine()
    assert _run(tmp_path, monkeypatch, engine)[0] == ref
    assert engine.calls == 0


@pytest.mark.parametrize("change", ["fps", "region", "raw_version"])
def test_read_changes_read_again(tmp_path: Path, monkeypatch, change: str) -> None:
    _run(tmp_path, monkeypatch, _Engine())
    preset, fps = _preset(), FPS
    if change == "fps":
        fps = 2.0
    elif change == "region":
        preset = _preset(min_text_h=10)
    else:
        monkeypatch.setattr(ocr_raw, "RAW_VERSION", ocr_raw.RAW_VERSION + 1)
    engine = _Engine()
    _run(tmp_path, monkeypatch, engine, preset, fps=fps)
    assert engine.calls > 0
    assert len(list(tmp_path.glob("ocr-raw-*"))) == 2


def test_force_and_debug_read_again(tmp_path: Path, monkeypatch) -> None:
    ref, _ = _run(tmp_path, monkeypatch, _Engine())
    _bump(monkeypatch)
    engine = _Engine()
    debug = tmp_path / "crops"
    assert _run(tmp_path, monkeypatch, engine, debug_dir=debug)[0] == ref
    assert engine.calls > 0 and any(debug.glob("rec_*_feed.png"))
    engine = _Engine()
    assert _run(tmp_path, monkeypatch, engine, force=True)[0] == ref
    assert engine.calls > 0


class _Factory:
    """``make_engine`` for :func:`detect_ocr_events`: counts the engines it makes."""

    def __init__(self) -> None:
        self.engines: list[_Engine] = []

    def __call__(self) -> _Engine:
        self.engines.append(_Engine())
        return self.engines[-1]


def test_lazy_engine_is_made_only_for_a_live_read(tmp_path: Path, monkeypatch, caplog) -> None:
    make = _Factory()
    ref, _ = _run(tmp_path, monkeypatch, None, make_engine=make)
    assert len(make.engines) == 1 and make.engines[0].calls > 0
    _bump(monkeypatch)
    caplog.set_level(logging.INFO, logger="cutpoint.events.killfeed")
    events, stats = _run(tmp_path, monkeypatch, None, make_engine=make)
    assert events == ref and stats["replayed"]
    assert len(make.engines) == 1  # a replay makes no engine
    (line,) = [r.getMessage() for r in caplog.records if " OCR events, " in r.getMessage()]
    assert "replayed from the raw OCR cache: 0 live OCR calls (the cached read made {" in line
    assert _run(tmp_path, monkeypatch, None, make_engine=make) == (events, stats)  # cache hit
    assert len(make.engines) == 1
    # force reads again: the engine is made then
    _run(tmp_path, monkeypatch, None, make_engine=make, force=True)
    assert len(make.engines) == 2 and make.engines[1].calls > 0


def test_pipeline_loads_no_ocr_model_for_a_replay(tmp_path: Path, monkeypatch) -> None:
    srcs = [SimpleNamespace(path=tmp_path / f"{n}.mp4", name=f"{n}.mp4") for n in "ab"]
    monkeypatch.setattr(
        pipeline, "load_highlights", lambda p: {"sources": [{"path": str(s.path)} for s in srcs]}
    )
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "load_source", lambda path, dur: srcs["ab".index(path.stem)])
    monkeypatch.setattr(pipeline, "bad_ranges", lambda *a: {})
    monkeypatch.setattr(pipeline, "load_transcript_lines", lambda project: [])
    monkeypatch.setattr(pipeline, "detect_audio_events", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "load_preset", lambda *a: None)
    made: list[object] = []
    monkeypatch.setattr(pipeline, "OcrEngine", lambda: made.append(object()) or made[-1])
    engines = []

    def ocr_events(s, params, engine, *, make_engine=None, **_k):
        assert engine is None
        if s.name == "b.mp4":  # a live read; a.mp4 is a replay (no engine wanted)
            engines.extend([make_engine(), make_engine()])
            raise _Stop  # the rest of the run is not needed
        return [], {}

    monkeypatch.setattr(pipeline, "detect_ocr_events", ocr_events)
    with pytest.raises(_Stop):
        pipeline.run_events(tmp_path, layers=("ocr",), discover=False)
    assert len(made) == 1 and engines == [made[0], made[0]]  # made once, for b.mp4


def test_force_without_engine_keeps_the_raw_read(tmp_path: Path, monkeypatch) -> None:
    _run(tmp_path, monkeypatch, _Engine())
    # the OCR layer does not run (no engine): nothing would read the footage again
    _run(tmp_path, monkeypatch, None, force=True)
    assert len(list(tmp_path.glob("ocr-raw-*/meta.json"))) == 1


def test_segment_without_frames_leaves_the_read_incomplete(tmp_path: Path, monkeypatch) -> None:
    avoid = [(25.0, 27.0)]  # segments 0-23 s and 29-60 s
    events, _ = _run(tmp_path, monkeypatch, _Engine(), avoid=avoid, fail_at=29.0)
    assert [e["t"] for e in events if e["type"] == "kill"] == []  # the feed shows from 24 s
    (meta,) = tmp_path.glob("ocr-raw-*/meta.json")
    status = json.loads(meta.read_text())
    assert not status["complete"] and status["failed"]
    _bump(monkeypatch)
    assert _run(tmp_path, monkeypatch, None, avoid=avoid) == ([], {"skipped": "not cached"})
    # the next read starts over and reads the part that failed
    starts: list[float] = []
    events, _ = _run(tmp_path, monkeypatch, _Engine(), avoid=avoid, starts=starts)
    assert starts == [0.0, 29.0]
    assert [e["t"] for e in events if e["type"] == "kill"] == [29.0, 40.0]
    assert json.loads(meta.read_text())["complete"]


def test_failed_segment_survives_an_interrupt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ocr_raw, "CHUNK_FRAMES", 50)
    avoid = [(25.0, 27.0)]
    with pytest.raises(_Stop):
        _run(tmp_path, monkeypatch, _Engine(), avoid=avoid, fail_at=0.0, stop_after=60)
    (meta,) = tmp_path.glob("ocr-raw-*/meta.json")
    status = json.loads(meta.read_text())
    assert status["chunks"] == 1 and not status["complete"] and not status.get("failed")
    # the resumed read keeps the failure from its first part
    _run(tmp_path, monkeypatch, _Engine(), avoid=avoid)
    status = json.loads(meta.read_text())
    assert status["failed"] and not status["complete"]
    # after a bump the read starts over instead of replaying the gap
    _bump(monkeypatch)
    _run(tmp_path, monkeypatch, _Engine(), avoid=avoid)
    status = json.loads(meta.read_text())
    assert status["complete"] and not status.get("failed")


def test_interrupted_read_resumes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ocr_raw, "CHUNK_FRAMES", 50)
    whole, part = _dirs(tmp_path, "whole", "part")
    # 23-29 s are skipped: frames 0-68 in 0-23 s, then 93 from 29 s
    avoid = [(25.0, 27.0)]
    ref, ref_stats = _run(whole, monkeypatch, _Engine(), avoid=avoid)
    with pytest.raises(_Stop):
        _run(part, monkeypatch, _Engine(), avoid=avoid, stop_after=130)
    (folder,) = part.glob("ocr-raw-*")
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["chunks"] == 2 and not meta["complete"]
    assert not list(part.glob("events-ocr-*.json"))  # no result of a part read
    # interrupted before the first chunk: no raw folder is left
    early = tmp_path / "early"
    early.mkdir()
    with pytest.raises(_Stop):
        _run(early, monkeypatch, _Engine(), avoid=avoid, stop_after=30)
    assert not list(early.iterdir())

    starts: list[float] = []
    engine = _Engine()
    events, stats = _run(part, monkeypatch, engine, avoid=avoid, starts=starts)
    # frame 100 is frame 31 of the second segment
    assert starts == [pytest.approx(29 + 31 / 3)]
    assert events == ref
    assert _untimed(stats) == _untimed(ref_stats)
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["chunks"] == 4 and meta["complete"]  # 162 frames
    assert not [p for p in tmp_path.rglob("*") if p.name.endswith(".tmp")]
    # the resumed read replays like an uninterrupted one
    _bump(monkeypatch)
    assert _run(part, monkeypatch, None, avoid=avoid)[0] == ref


def test_cleanup_keeps_the_most_recently_used(tmp_path: Path) -> None:
    def raw(name: str, complete: bool, age: float) -> Path:
        folder = tmp_path / f"ocr-raw-{name}"
        folder.mkdir()
        meta = folder / "meta.json"
        meta.write_text(json.dumps({"complete": complete}))
        t = time.time() - age
        os.utime(meta, (t, t))
        return folder

    current = raw("current", True, 10 * 86400)
    raw("new", True, 100)
    raw("old", True, 200)
    raw("part", False, 3600)  # an interrupted read, recent: may resume
    raw("stale", False, 8 * 86400)
    ocr_raw.cleanup(tmp_path, current)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "ocr-raw-current",
        "ocr-raw-new",
        "ocr-raw-part",
    ]


def test_complete_reads_are_cleaned_up(tmp_path: Path, monkeypatch) -> None:
    for fps in (3.0, 2.0, 1.5):
        _run(tmp_path, monkeypatch, _Engine(), fps=fps)
    assert len(list(tmp_path.glob("ocr-raw-*"))) == ocr_raw.KEEP_RAW


def test_raw_cache_is_never_written_in_the_checkout(tmp_path: Path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(ocr_raw, "_checkout", lambda: tmp_path)
    cdir = tmp_path / "cache"
    cdir.mkdir()
    events, _ = _run(cdir, monkeypatch, _Engine())
    assert [e["t"] for e in events if e["type"] == "kill"] == [24.0, 40.0]
    assert not list(cdir.glob("ocr-raw-*"))
    assert any("not writing the raw OCR cache" in r.getMessage() for r in caplog.records)


def test_only_cutpoints_own_checkout_is_guarded(tmp_path: Path) -> None:
    home, repo = tmp_path / "home", tmp_path / "repo"
    (home / ".git").mkdir(parents=True)  # a home folder kept in git
    (repo / ".git").mkdir(parents=True)
    module = Path("cutpoint", "events", "ocr_raw.py")
    assert ocr_raw._checkout(home / ".venv" / "lib" / "site-packages" / module) is None
    assert ocr_raw._checkout(repo / "src" / module) == repo.resolve()


def test_checkout_is_found() -> None:
    root = ocr_raw._checkout()
    if root is None:
        pytest.skip("not run from a git checkout")
    assert ocr_raw.in_checkout(Path(ocr_raw.__file__).parent)
    assert not ocr_raw.in_checkout(root.parent)


@pytest.mark.slow
def test_raw_cache_is_git_ignored() -> None:
    root, git = ocr_raw._checkout(), shutil.which("git")
    if root is None or git is None:
        pytest.skip("needs git and a git checkout")
    for path in ("cache/x/ocr-raw-y/meta.json", "projects/p/ocr-raw-y/chunk-00000.zip"):
        done = subprocess.run([git, "-C", str(root), "check-ignore", "-q", path], check=False)
        assert done.returncode == 0, path
