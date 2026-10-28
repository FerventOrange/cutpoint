"""Frame ids of discovery's OCR bursts (#444): bursts overlap on a short source, and a
cached re-run must give the same frames as the first run (fake decode and OCR)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint.events import discover, game_profile, killfeed


@pytest.fixture
def fake_ocr(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Each decoded frame shows its time as a text; returns the burst starts decoded."""
    decoded: list[float] = []

    def burst_frames(path, t, n, step, width):
        decoded.append(t)
        return [
            np.full((72, 128, 3), round((t + k * step) * 100) % 256, np.uint8) for k in range(n)
        ]

    class Engine:
        def __init__(self, det_side=960, **_kwargs):
            pass

        def __call__(self, img, min_score):
            return [([(0, 0), (40, 0), (40, 10), (0, 10)], f"text {img[0, 0, 0]}", 0.9)]

    monkeypatch.setattr(discover, "_burst_frames", burst_frames)
    monkeypatch.setattr(killfeed, "OcrEngine", Engine)
    return decoded


def _key(frames: list[dict]) -> list[tuple]:
    return [(f["id"], f["source"], f["t"], f["burst"], f["k"], f["boxes"]) for f in frames]


def _run(tmp_path: Path, bursts) -> list[dict]:
    params = discover.DiscoverParams()
    return discover._ocr_frames(bursts, params, tmp_path / "frames", tmp_path / "ocr.jsonl")


def test_overlapping_bursts_get_unique_ids_and_replay_the_same(tmp_path, fake_ocr):
    src = SimpleNamespace(name="short.mp4", io_path=tmp_path / "short.mp4")
    # 0.04 s apart (shared ids at 0.1 s), 1.5 s apart (3 of 4 times shared) and one
    # start twice (the plan's clamp on a short source)
    bursts = [(src, 0.0), (src, 0.04), (src, 1.5), (src, 14.57), (src, 14.57)]
    first = _run(tmp_path, bursts)
    ids = [f["id"] for f in first]
    assert len(ids) == len(set(ids)) == 5 * 4
    assert ids[:4] == [f"short.mp4@{t:.1f}" for t in (0.0, 1.5, 3.0, 4.5)]  # first keeps its id
    assert [(f["burst"], f["k"]) for f in first] == [(b, k) for b in range(5) for k in range(4)]

    fake_ocr.clear()
    again = _run(tmp_path, bursts)
    assert fake_ocr == []  # every frame from the cache
    assert _key(again) == _key(first)


def test_old_cache_is_reused_except_for_a_shared_id(tmp_path, fake_ocr):
    """A cache written with shared ids (the last burst's frame under the id): the frames
    no other burst shares come from the cache, a frame stored for another time is OCR'd."""
    src = SimpleNamespace(name="short.mp4", io_path=tmp_path / "short.mp4")
    bursts = [(src, 0.0), (src, 0.04), (src, 60.0)]
    first = _run(tmp_path, bursts)
    old = []
    for f in first:  # the old ids: source and time, a later burst's record replacing the first's
        old.append(f | {"id": f["id"].split("#")[0]})
    (tmp_path / "ocr.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in old), encoding="utf-8"
    )

    fake_ocr.clear()
    again = _run(tmp_path, bursts)
    assert fake_ocr == [0.0, 0.04]  # the 60 s burst is cached; the two that shared ids are not
    assert [f["boxes"] for f in again] == [f["boxes"] for f in first]
    assert [(f["burst"], f["k"], f["t"]) for f in again] == [
        (f["burst"], f["k"], f["t"]) for f in first
    ]

    fake_ocr.clear()
    assert _key(_run(tmp_path, bursts)) == _key(again)
    assert fake_ocr == []


def test_mined_records_keep_only_the_last_per_id_after_an_upgrade(tmp_path, fake_ocr, monkeypatch):
    """ocr.jsonl keeps the old shared-id records next to the new ones; the phrase miner
    sees exactly the run's frames, the last record of each id."""
    src = SimpleNamespace(name="short.mp4", io_path=tmp_path / "short.mp4")
    bursts = [(src, 0.0), (src, 0.04), (src, 60.0)]
    first = _run(tmp_path, bursts)
    old = [f | {"id": f["id"].split("#")[0]} for f in first]
    (tmp_path / "ocr.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in old), encoding="utf-8"
    )
    again = _run(tmp_path, bursts)  # appends the re-OCR'd frames to the old records

    records = discover.read_ocr_samples(tmp_path / "ocr.jsonl")
    assert len(records) > len(again)  # the stale shared-id records are still in the file

    seen: list[list[dict]] = []

    def capture(frames, params):
        seen.append(frames)
        return []

    monkeypatch.setattr(game_profile, "gameplay_frames", capture)
    preset = {"discovery": {"frames": [f["id"] for f in again]}}
    game_profile.mine_phrases(preset, records)

    used = seen[0]
    assert len(used) == len(again)
    assert sorted(f["id"] for f in used) == sorted(f["id"] for f in again)
    by_id = {f["id"]: f for f in again}
    assert all(
        f["boxes"] == by_id[f["id"]]["boxes"] and f["t"] == by_id[f["id"]]["t"] for f in used
    )
