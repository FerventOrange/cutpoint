"""Frame ids of discovery's OCR bursts (#444): bursts overlap on a short source, and a
cached re-run must give the same frames as the first run (fake decode and OCR)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint.events import discover, killfeed


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
        def __init__(self, **_kwargs):
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
    src = SimpleNamespace(
        name="short.mp4", path=tmp_path / "short.mp4", io_path=tmp_path / "short.mp4"
    )
    # 0.04 s apart (shared ids at 0.1 s), 1.5 s apart (3 of 4 times shared) and one
    # start twice (the plan's clamp on a short source)
    bursts = [(src, 0.0), (src, 0.04), (src, 1.5), (src, 14.57), (src, 14.57)]
    first = _run(tmp_path, bursts)
    ids = [f["id"] for f in first]
    assert len(ids) == len(set(ids)) == 5 * 4
    tag = discover._source_tag(src)
    assert ids[:4] == [
        f"short.mp4~{tag}@{t:.1f}" for t in (0.0, 1.5, 3.0, 4.5)
    ]  # first keeps its id
    assert [(f["burst"], f["k"]) for f in first] == [(b, k) for b in range(5) for k in range(4)]

    fake_ocr.clear()
    again = _run(tmp_path, bursts)
    assert fake_ocr == []  # every frame from the cache
    assert _key(again) == _key(first)


def test_old_cache_is_reused_except_for_a_shared_id(tmp_path, fake_ocr):
    """A cache written with shared ids (the last burst's frame under the id): the frames
    no other burst shares come from the cache, a frame stored for another time is OCR'd."""
    src = SimpleNamespace(
        name="short.mp4", path=tmp_path / "short.mp4", io_path=tmp_path / "short.mp4"
    )
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


def test_a_run_interrupted_mid_burst_resumes_with_only_the_unfinished_bursts(
    tmp_path, fake_ocr, monkeypatch
):
    """Ctrl+C during the OCR: frames already OCR'd are not OCR'd again; only unfinished
    bursts are decoded again. The resumed run gives the same records as an uninterrupted one."""
    src = SimpleNamespace(
        name="long.mp4", path=tmp_path / "long.mp4", io_path=tmp_path / "long.mp4"
    )
    bursts = [(src, 0.0), (src, 60.0), (src, 120.0)]
    reference = _run(tmp_path / "reference", bursts)
    frames = discover.DiscoverParams().burst_frames

    real_engine = killfeed.OcrEngine
    calls = {"n": 0}

    class InterruptedEngine(real_engine):
        def __call__(self, img, min_score):
            calls["n"] += 1
            if calls["n"] == frames + 2:  # the second frame of the second burst
                raise KeyboardInterrupt
            return super().__call__(img, min_score)

    monkeypatch.setattr(killfeed, "OcrEngine", InterruptedEngine)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, bursts)

    resumed_calls = {"n": 0}

    class CountingEngine(real_engine):
        def __call__(self, img, min_score):
            resumed_calls["n"] += 1
            return super().__call__(img, min_score)

    monkeypatch.setattr(killfeed, "OcrEngine", CountingEngine)

    fake_ocr.clear()
    resumed = _run(tmp_path, bursts)

    assert fake_ocr == [60.0, 120.0]  # the finished first burst comes from the cache
    # frame k=0 of the second burst was saved before the interrupt
    assert resumed_calls["n"] == 2 * frames - 1
    assert _key(resumed) == _key(reference)


def test_a_new_burst_plan_does_not_overwrite_the_pictures_of_cached_frames(tmp_path, fake_ocr):
    """The picture of a cached record stays its own when a later plan has other bursts (the
    names came from the burst number, which another plan gives to other frames)."""
    src = SimpleNamespace(
        name="long.mp4", path=tmp_path / "long.mp4", io_path=tmp_path / "long.mp4"
    )
    first = _run(tmp_path, [(src, 0.0), (src, 60.0)])
    pics = {f["id"]: (tmp_path / "frames" / f["frame"]).read_bytes() for f in first}
    assert len({f["frame"] for f in first}) == len(first)

    second = _run(tmp_path, [(src, 30.0), (src, 0.0), (src, 90.0)])  # burst 0 is another time
    fake_ocr.clear()
    kept = [f for f in second if f["id"] in pics]
    assert {f["id"] for f in kept} == {f["id"] for f in first if f["t"] < 6}
    for f in kept:  # the cached frames of 0.0 s still show their own picture
        assert (tmp_path / "frames" / f["frame"]).read_bytes() == pics[f["id"]]
    assert len({f["frame"] for f in second}) == len(second)


def test_pictures_of_frames_no_plan_uses_are_removed_and_their_records_redone(tmp_path, fake_ocr):
    src = SimpleNamespace(
        name="long.mp4", path=tmp_path / "long.mp4", io_path=tmp_path / "long.mp4"
    )
    first = _run(tmp_path, [(src, 0.0), (src, 60.0)])
    second = _run(tmp_path, [(src, 0.0)])
    pics = {p.name for p in (tmp_path / "frames").glob("*.jpg")}
    assert pics == {f["frame"] for f in second}  # bounded: the 60 s burst's pictures went
    fake_ocr.clear()
    again = _run(tmp_path, [(src, 0.0), (src, 60.0)])
    assert fake_ocr == [60.0]  # records whose picture is gone are OCR'd again
    assert _key(again) == _key(first)


def test_a_legacy_picture_name_two_records_share_is_not_trusted(tmp_path, fake_ocr):
    """Older caches named a picture b<burst>_<k>.jpg: two records naming one file means a
    later plan replaced the first's picture, so neither is reused."""
    src = SimpleNamespace(
        name="long.mp4", path=tmp_path / "long.mp4", io_path=tmp_path / "long.mp4"
    )
    bursts = [(src, 0.0), (src, 60.0)]
    first = _run(tmp_path, bursts)
    legacy = []
    for f in first:
        name = f"b{f['burst']:03d}_{f['k']}.jpg"
        (tmp_path / "frames" / f["frame"]).rename(tmp_path / "frames" / name)
        # burst 1's records claim burst 0's files (as after another plan wrote over them)
        legacy.append(f | {"frame": f"b000_{f['k']}.jpg"})
    (tmp_path / "ocr.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in legacy), encoding="utf-8"
    )
    fake_ocr.clear()
    again = _run(tmp_path, bursts)
    assert fake_ocr == [0.0, 60.0]
    assert len({f["frame"] for f in again}) == len(again)


def test_a_legacy_cache_with_unique_picture_names_is_reused(tmp_path, fake_ocr):
    src = SimpleNamespace(name="long.mp4", path=tmp_path / "long.mp4", io_path=tmp_path / "x.mp4")
    bursts = [(src, 0.0), (src, 60.0)]
    first = _run(tmp_path, bursts)
    legacy = []
    for f in first:
        name = f"b{f['burst']:03d}_{f['k']}.jpg"
        (tmp_path / "frames" / f["frame"]).rename(tmp_path / "frames" / name)
        legacy.append(f | {"frame": name})
    (tmp_path / "ocr.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in legacy), encoding="utf-8"
    )
    fake_ocr.clear()
    again = _run(tmp_path, bursts)
    assert fake_ocr == []
    assert {p.name for p in (tmp_path / "frames").glob("*.jpg")} == {f["frame"] for f in again}


def test_same_named_sources_in_other_folders_get_their_own_pictures(tmp_path, fake_ocr):
    a = SimpleNamespace(name="clip.mp4", path=tmp_path / "a" / "clip.mp4", io_path=tmp_path / "a")
    b = SimpleNamespace(name="clip.mp4", path=tmp_path / "b" / "clip.mp4", io_path=tmp_path / "b")
    frames = _run(tmp_path, [(a, 0.0), (b, 0.0)])
    assert len({f["frame"] for f in frames}) == len(frames)


def test_pictures_round_trip_in_a_non_ascii_folder(tmp_path):
    folder = tmp_path / "prøjekt ünï"
    folder.mkdir()
    img = np.full((20, 30, 3), 90, np.uint8)
    assert discover.write_jpeg(folder / "f.jpg", img, 80)
    back = discover.read_image(folder / "f.jpg")
    assert back is not None
    assert back.shape == img.shape
    assert discover.read_image(folder / "missing.jpg") is None


def test_a_picture_that_cannot_be_removed_does_not_stop_discovery(tmp_path, fake_ocr, monkeypatch):
    src = SimpleNamespace(name="long.mp4", path=tmp_path / "long.mp4", io_path=tmp_path / "x.mp4")
    _run(tmp_path, [(src, 0.0), (src, 60.0)])

    def locked(self, missing_ok=False):
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "unlink", locked)
    second = _run(tmp_path, [(src, 0.0)])
    assert len(second) == discover.DiscoverParams().burst_frames


def test_same_named_sources_in_other_folders_get_distinct_ids_and_both_replay(tmp_path, fake_ocr):
    """#794: the file name alone was the id, so the second source's records replaced the first's."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = SimpleNamespace(name="clip.mp4", path=tmp_path / "a" / "clip.mp4", io_path=tmp_path / "a")
    b = SimpleNamespace(name="clip.mp4", path=tmp_path / "b" / "clip.mp4", io_path=tmp_path / "b")
    bursts = [(a, 0.0), (b, 0.0)]
    first = _run(tmp_path, bursts)
    # each alone (another plan) gives the ids it has in the shared run
    assert {f["id"] for f in _run(tmp_path / "solo", [(a, 0.0)])}.isdisjoint(
        {f["id"] for f in _run(tmp_path / "solo2", [(b, 0.0)])}
    )
    assert len({f["id"] for f in first}) == len(first)
    assert len({f["frame"] for f in first}) == len(first)
    assert all((tmp_path / "frames" / f["frame"]).is_file() for f in first)

    fake_ocr.clear()
    again = _run(tmp_path, bursts)
    assert fake_ocr == []  # neither source's frames were overwritten or re-OCR'd
    assert _key(again) == _key(first)


def test_a_record_under_the_old_file_name_id_is_adopted_when_its_picture_matches(
    tmp_path, fake_ocr
):
    """#794: older caches keyed a frame by file name and time; the same file's frames keep
    their OCR (no re-read), and a record whose picture is another path's does not."""
    src = SimpleNamespace(
        name="clip.mp4", path=tmp_path / "clip.mp4", io_path=tmp_path / "clip.mp4"
    )
    bursts = [(src, 0.0)]
    first = _run(tmp_path, bursts)
    old = [f | {"id": f"clip.mp4@{f['t']:.1f}"} for f in first]
    (tmp_path / "ocr.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in old), encoding="utf-8"
    )
    fake_ocr.clear()
    again = _run(tmp_path, bursts)
    assert fake_ocr == []
    assert _key(again) == _key(first)  # taken over under the new ids

    # the same old ids with pictures of another path (a same-named file elsewhere): redone
    foreign = [r | {"frame": "0123456789abcdef.jpg"} for r in old]
    for r in foreign:
        (tmp_path / "frames" / r["frame"]).write_bytes(b"x")
    (tmp_path / "ocr.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in foreign), encoding="utf-8"
    )
    fake_ocr.clear()
    _run(tmp_path, bursts)
    assert fake_ocr == [0.0]


def test_ocr_records_of_other_plans_are_pruned_after_a_complete_run(tmp_path, fake_ocr):
    """#795: ocr.jsonl kept every record ever written; a complete run keeps the current ids."""
    src = SimpleNamespace(
        name="clip.mp4", path=tmp_path / "clip.mp4", io_path=tmp_path / "clip.mp4"
    )
    _run(tmp_path, [(src, 0.0), (src, 60.0)])
    second = _run(tmp_path, [(src, 0.0)])
    ids = [
        json.loads(line)["id"] for line in (tmp_path / "ocr.jsonl").read_text("utf-8").splitlines()
    ]
    assert ids == [f["id"] for f in second]
    # a stale record from a file that no longer exists is gone too, and the run still replays
    with (tmp_path / "ocr.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "gone.mp4@1.0", "t": 1.0, "frame": "x.jpg", "boxes": []}) + "\n")
    fake_ocr.clear()
    again = _run(tmp_path, [(src, 0.0)])
    assert fake_ocr == []
    assert "gone.mp4@1.0" not in (tmp_path / "ocr.jsonl").read_text("utf-8")
    assert _key(again) == _key(second)


def test_an_interrupted_run_leaves_the_ocr_cache_unpruned(tmp_path, fake_ocr):
    """#795: only a complete run prunes ocr.jsonl; an interrupted one keeps every record."""
    src = SimpleNamespace(
        name="clip.mp4", path=tmp_path / "clip.mp4", io_path=tmp_path / "clip.mp4"
    )
    cache = tmp_path / "ocr.jsonl"
    cache.write_text(
        json.dumps({"id": "other.mp4@1.0", "t": 1.0, "frame": "x.jpg", "boxes": []}) + "\n",
        encoding="utf-8",
    )
    calls = []

    def stop(done, total):
        calls.append(done)
        if len(calls) > 5:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        discover._ocr_frames(
            [(src, 0.0), (src, 60.0)],
            discover.DiscoverParams(),
            tmp_path / "frames",
            cache,
            on_frame=stop,
        )
    assert "other.mp4@1.0" in cache.read_text("utf-8")
