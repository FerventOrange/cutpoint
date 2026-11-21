"""#1085 hud-events refactors (A338, A339, A340, A341, A694, A704, A705, A707) give the same
results as the code before them; the old code is kept here as the reference."""

from __future__ import annotations

import bisect
import json
import os
import random
import threading

import numpy as np
import pytest

from cutpoint import cache, category
from cutpoint.events import killfeed as kf
from cutpoint.events import regions
from cutpoint.events.killfeed import MENU
from cutpoint.events.text import notice_letters


def _old_rows(items, tol):
    rows = []
    for it in sorted(items, key=lambda i: np.mean([p[1] for p in i[0]])):
        cy = np.mean([p[1] for p in it[0]])
        if rows and abs(np.mean([p[1] for p in rows[-1][0][0]]) - cy) <= tol:
            rows[-1].append(it)
        else:
            rows.append([it])
    return [sorted(r, key=lambda i: min(p[0] for p in i[0])) for r in rows]


def _old_group_rows(boxes, min_h, tol):
    rows = []
    for b in sorted(boxes, key=lambda b: sum(p[1] for p in b)):
        y0, y1 = min(p[1] for p in b), max(p[1] for p in b)
        if min_h and y1 - y0 < min_h:
            continue
        cy, x0, x1 = (y0 + y1) / 2, min(p[0] for p in b), max(p[0] for p in b)
        if rows and abs(rows[-1][0] - cy) <= tol:
            r = rows[-1]
            r[1], r[2], r[3], r[4] = min(r[1], x0), max(r[2], x1), min(r[3], y0), max(r[4], y1)
        else:
            rows.append([cy, x0, x1, y0, y1])
    return [tuple(r) for r in rows]


def _quad(rng: random.Random, ints: bool) -> list[list[float]]:
    x, y = rng.uniform(0, 600), rng.uniform(0, 300)
    w, h = rng.uniform(5, 200), rng.uniform(4, 40)
    tilt = rng.choice([0.0, 0.0, rng.uniform(-3, 3)])  # rotated quads have mean != mid
    pts = [[x, y], [x + w, y + tilt], [x + w, y + h + tilt], [x, y + h]]
    return [[round(a), round(b)] for a, b in pts] if ints else pts


@pytest.mark.parametrize("seed", range(40))
def test_row_groupers_identical(seed: int) -> None:
    rng = random.Random(seed)
    ints = seed % 2 == 0
    boxes = [_quad(rng, ints) for _ in range(rng.randint(0, 25))]
    items = [(b, f"t{i}", rng.random()) for i, b in enumerate(boxes)]
    for tol in (0.0, 6.0, 12.0, 30.0):
        assert kf._rows(items, tol) == _old_rows(items, tol)
        for min_h in (0, 10.0):
            assert kf.group_rows(boxes, min_h, tol) == _old_group_rows(boxes, min_h, tol)


def _old_notice(line: str) -> bool:
    n = sum(ch.isalpha() for ch in line)
    return not (n < 4 or n < 0.6 * len(line.replace(" ", "")))


@pytest.mark.parametrize(
    "line", ["", "abc", "abcd", "ab 12", "Kip has left", "1234 abcd", "x" * 6 + "1" * 4, "é ü ß ø"]
)
def test_notice_letters_identical(line: str) -> None:
    assert notice_letters(line) == _old_notice(line)


def test_ref_factors_and_box() -> None:
    for w, h in ((1920, 1080), (1280, 720), (1440, 1080), (2560, 1080), (853, 480)):
        assert regions.ref_factors(w, h) == (1920 / w, 1080 / h)
    box = [[np.float32(3.5), 7], [40, np.float32(7.25)], [40, 30], [3.5, 30]]
    ref = (100, 50, 1920 / 1280, 1080 / 720)
    rx, ry, sx, sy = ref
    old = [
        kf._plain((rx + min(p[0] for p in box)) * sx),
        kf._plain((ry + min(p[1] for p in box)) * sy),
        kf._plain((rx + max(p[0] for p in box)) * sx),
        kf._plain((ry + max(p[1] for p in box)) * sy),
    ]
    assert kf.ref_box(box, ref) == old
    assert all(type(a) is type(b) for a, b in zip(kf.ref_box(box, ref), old, strict=True))


def _old_staying_lines(frames, text_spans, text_keys, max_gap):
    spans = kf.staying_spans(frames, text_spans, text_keys, max_gap)
    starts, ends = sorted(a for a, _ in spans), sorted(b for _, b in spans)
    out = []
    for f in frames:
        t = f["t"]
        shown = sorted(a for a, b in spans if a <= t <= b)
        together = max(
            (
                bisect.bisect_right(shown, a + MENU.static_feed_guard_s) - i
                for i, a in enumerate(shown)
            ),
            default=0,
        )
        out.append((bisect.bisect_right(starts, t) - bisect.bisect_left(ends, t), together))
    return out


@pytest.mark.parametrize("seed", range(25))
def test_staying_lines_identical(seed: int) -> None:
    rng = random.Random(seed)
    times = sorted(round(rng.uniform(0, 300) * 3) / 3 for _ in range(rng.randint(0, 200)))
    if seed % 5 == 0:
        rng.shuffle(times)  # not in time order: still per frame
    frames = [{"t": t, "id": i} for i, t in enumerate(times)]
    spans = []
    for _ in range(rng.randint(0, 80)):
        a = round(rng.uniform(0, 300) * 3) / 3
        spans.append((a, a + rng.choice([0.0, 1.0, MENU.text_stay_s, rng.uniform(0, 40)])))
    keys = [rng.choice("abcde") for _ in spans] if seed % 2 else None
    got = kf.staying_lines(frames, spans, keys, MENU.static_max_gap_s)
    assert got == _old_staying_lines(frames, spans, keys, MENU.static_max_gap_s)


def test_preset_parse_cache_follows_edits(tmp_path, monkeypatch) -> None:
    home = tmp_path / "presets"
    home.mkdir()
    monkeypatch.setattr(regions.paths, "presets_dir", lambda: home)
    p = home / "g.json"
    p.write_text(json.dumps({"game": "g", "match": ["alpha"]}), encoding="utf-8")
    assert regions.preset_for_path("x/alpha.mp4") == "g"
    assert regions._read_json(p) is regions._read_json(p)  # parsed once
    # same size, same mtime: the file id or the edit's mtime tells it apart; force mtime
    st = p.stat()
    p.write_text(json.dumps({"game": "g", "match": ["bravo"]}), encoding="utf-8")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000))
    assert regions.preset_for_path("x/alpha.mp4") is None
    assert regions.preset_for_path("x/bravo.mp4") == "g"
    p.unlink()
    assert regions.preset_for_path("x/bravo.mp4") is None


def test_preset_cache_reads_held_files(tmp_path) -> None:
    project = tmp_path / "proj"
    path = regions.game_keys_path(project)
    regions.save_game_keys(project, {"s": {"key": "a"}})
    assert regions.load_game_keys(project) == {"s": {"key": "a"}}
    with cache.hold_outputs(project, (path.relative_to(project).parts[0],)):
        regions.save_game_keys(project, {"s": {"key": "b"}})  # in memory only
        assert regions.load_game_keys(project) == {"s": {"key": "b"}}
    assert regions.load_game_keys(project) == {"s": {"key": "a"}}


def test_shared_ocr_builds_once_and_releases(monkeypatch) -> None:
    built = []

    class Fake:
        def __init__(self, **kw):
            built.append(kw)

    monkeypatch.setattr(kf, "OcrEngine", Fake)
    make = kf.SharedOcr(lanes=1, rec_batch=1)
    got = []
    threads = [threading.Thread(target=lambda: got.append(make())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 1 and built[0] == {"lanes": 1, "rec_batch": 1}
    assert all(g is got[0] for g in got)
    make.release()
    assert make() is not got[0] and len(built) == 2


def test_probe_sources_shares_one_text_engine(monkeypatch, tmp_path) -> None:
    made = []

    class Fake:
        def __init__(self, **kw):
            made.append(self)

    monkeypatch.setattr(kf, "OcrEngine", Fake)
    monkeypatch.setattr(category, "ocr_installed", lambda: True)
    seen = []

    def probe_source(src, vision=None):
        make = category._TEXT_OCR.get()
        seen.append(make() if make is not None else None)
        return src

    monkeypatch.setattr(category, "probe_source", probe_source)
    srcs = [tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"]
    assert category.probe_sources(srcs) == srcs
    assert len(made) == 1 and all(s is made[0] for s in seen)
    assert category._TEXT_OCR.get() is None  # freed after the detection
