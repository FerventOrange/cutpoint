"""The HUD fit check (#822): edge text in the same screen cells fits, text in another
corner does not, too little text or no grid reference decides nothing, a merged two-game
discovery splits, and the key of a source that does not fit (invented boxes and paths)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from cutpoint.events import game_fit
from cutpoint.events.game_fit import Profile, against_ref, profile, split_group
from cutpoint.events.regions import Region, misfit_key

MIN_SCORE = 0.5
TOP_LEFT = Region("feed", 0, 0, 400, 200, "feed")
BOTTOM_RIGHT = Region("corner", 1600, 900, 320, 180, "notice")
# box centres per layout: top-left cells 0,0 1,0 1,1; bottom-right cells 14,10 15,11
AT = {
    "a": [(60, 40), (120, 80), (200, 150)],
    "b": [(1700, 960), (1800, 1000), (1880, 1050)],
    "c": [(60, 40)],  # one box a frame: too little text to tell
}


def _box(cx: int, cy: int, text: str = "ab12", score: float = 0.9) -> dict:
    return {"x0": cx - 20, "y0": cy - 8, "x1": cx + 20, "y1": cy + 8, "text": text, "score": score}


def _frames(corner: str, n: int = 8, per: int = 3) -> list[dict]:
    """``n`` frames with ``per`` boxes in the top-left (``"a"``) or bottom-right (``"b"``)."""
    return [{"id": f"{corner}{i}", "boxes": [_box(*p) for p in AT[corner][:per]]} for i in range(n)]


def test_edge_boxes_need_score_letters_and_an_edge_outside_the_big_text() -> None:
    frame = {
        "boxes": [
            _box(60, 40),  # kept
            _box(60, 40, score=0.2),  # low score
            _box(60, 40, text="+"),  # fewer than two letters or digits
            _box(960, 540),  # centre of the screen
            _box(960, 150),  # top edge row, but in the big-text region
        ]
    }
    assert game_fit.edge_boxes(frame, MIN_SCORE) == [_box(60, 40)]


def test_the_grid_counts_each_cell_once_per_frame() -> None:
    frames = [{"boxes": [_box(60, 40), _box(70, 50), _box(1880, 1050)]}, {"boxes": [_box(60, 40)]}]
    p = profile(frames, MIN_SCORE)
    assert p.frames == 2 and p.boxes == 4
    assert p.grid() == {"0,0": 1.0, "15,11": 0.5}
    assert p.ref() == {
        "cells": [16, 12],
        "grid": {"0,0": 1.0, "15,11": 0.5},
        "frames": 2,
        "boxes": 4,
    }


def test_text_in_the_same_cells_fits() -> None:
    ref = profile(_frames("a", 10), MIN_SCORE).ref()
    fit = against_ref(profile(_frames("a"), MIN_SCORE), ref)
    assert fit is not None and fit.ok
    assert fit.ref == "preset" and fit.score == pytest.approx(1.0)
    assert fit.record() == {"score": 1.0, "frames": 8, "ref": "preset"}


def test_text_in_another_corner_does_not_fit() -> None:
    ref = profile(_frames("a", 10), MIN_SCORE).ref()
    fit = against_ref(profile(_frames("b"), MIN_SCORE), ref)
    assert fit is not None and not fit.ok and fit.score == 0.0


def test_a_layout_sharing_one_cell_does_not_fit() -> None:
    """One cell of three in common, the rest elsewhere: below the threshold."""
    ref = profile(_frames("a", 10), MIN_SCORE).ref()
    other = [(60, 40), (1700, 960), (1800, 1000), (1880, 500)]
    frames = [{"boxes": [_box(*p) for p in other]} for _ in range(8)]
    fit = against_ref(profile(frames, MIN_SCORE), ref)
    assert fit is not None and not fit.ok
    assert fit.score == pytest.approx(1 / (3**0.5 * 4**0.5))
    assert fit.score < game_fit.TAU_GRID


@pytest.mark.parametrize("frames", [_frames("b", n=3), _frames("b", n=8, per=1)])
def test_too_little_text_decides_nothing(frames: list[dict]) -> None:
    """Fewer than 4 gameplay frames, or fewer than 15 edge boxes."""
    ref = profile(_frames("a", 10), MIN_SCORE).ref()
    assert against_ref(profile(frames, MIN_SCORE), ref) is None


@pytest.mark.parametrize(
    "ref",
    [
        None,  # a preset of an earlier version, or written by hand
        {"inside": 1.0, "occ": {"feed": 1.0}},  # the per-region fit_ref of an earlier version
        {"cells": [8, 6], "grid": {"0,0": 1.0}},  # another grid
        {"cells": [16, 12], "grid": "bad"},
        {"cells": [16, 12], "grid": {"0,0": "x"}},
        {"cells": [16, 12], "grid": {}},  # all zero: nothing to compare with
        "bad",
        {"cells": [16, 12], "grid": {"0,0": float("nan")}},
        {"cells": [16, 12], "grid": {"0,0": -0.5}},
        {"cells": [16, 12], "grid": {"junk": 5}},
        {"cells": [16, 12], "grid": {"0,0": 1.5}},
        {"cells": [16, 12], "grid": {"16,0": 1.0}},  # outside the grid
    ],
)
def test_a_preset_without_a_grid_fit_ref_decides_nothing(ref) -> None:
    """No split until a discovery writes a grid fit_ref."""
    assert against_ref(profile(_frames("b"), MIN_SCORE), ref) is None


def test_a_merged_discovery_of_two_games_splits_greedily() -> None:
    """One discovery over two games; the source with the most frames leads, the others
    join when their grid looks like the group's."""
    profs = {
        "a1": profile(_frames("a", 12), MIN_SCORE),
        "a2": profile(_frames("a", 8), MIN_SCORE),
        "b1": profile(_frames("b", 8), MIN_SCORE),
        "b2": profile(_frames("b", 2), MIN_SCORE),  # too little to tell
    }
    fits = split_group(profs, {"a1": 300.0, "a2": 600.0, "b1": 900.0, "b2": 60.0})
    assert fits["a1"] is not None and fits["a1"].ok and fits["a1"].ref == "run"
    assert fits["a2"] is not None and fits["a2"].ok
    assert fits["b1"] is not None and not fits["b1"].ok and fits["b1"].score == 0.0
    assert fits["b2"] is None


def test_the_largest_group_leads_not_the_longest_source() -> None:
    """The Windows case in numbers: one source of game B has the most frames (52), but the
    two of game A have more together (50 + 42): A's group leads, B does not fit it."""
    profs = {
        "a1": profile(_frames("a", 50), MIN_SCORE),
        "a2": profile(_frames("a", 42), MIN_SCORE),
        "b": profile(_frames("b", 52), MIN_SCORE),
    }
    fits = split_group(profs, {"a1": 600.0, "a2": 500.0, "b": 700.0})
    assert fits["a1"] is not None and fits["a1"].ok and fits["a1"].score == pytest.approx(1.0)
    assert fits["a2"] is not None and fits["a2"].ok
    assert fits["b"] is not None and not fits["b"].ok and fits["b"].score == 0.0


def test_groups_of_equal_frames_tie_on_sources_then_seed_then_order() -> None:
    # 10 + 10 frames of A against 20 of B: the group of more sources leads
    profs = {
        "a1": profile(_frames("a", 10), MIN_SCORE),
        "a2": profile(_frames("a", 10), MIN_SCORE),
        "b": profile(_frames("b", 20), MIN_SCORE),
    }
    fits = split_group(profs, {"b": 900.0})
    assert fits["a1"].ok and fits["a2"].ok and not fits["b"].ok
    # one source each, as many frames: the first in the order (the longest) leads
    profs = {"a": profile(_frames("a", 12), MIN_SCORE), "b": profile(_frames("b", 12), MIN_SCORE)}
    fits = split_group(profs, {"a": 300.0, "b": 600.0})
    assert fits["b"].ok and not fits["a"].ok
    fits = split_group(profs, {"a": 600.0, "b": 300.0})
    assert fits["a"].ok and not fits["b"].ok


def test_a_group_without_enough_evidence_keeps_every_guess() -> None:
    profs = {k: profile(_frames("a", 2), MIN_SCORE) for k in ("x", "y")}
    assert split_group(profs, {}) == {"x": None, "y": None}


def test_fixed_sources_join_the_group_untested() -> None:
    profs = {
        "b0": profile(_frames("b", 20), MIN_SCORE),  # decided before: leads, stays
        "a1": profile(_frames("a", 8), MIN_SCORE),
    }
    fits = split_group(profs, {}, frozenset({"b0"}))
    assert set(fits) == {"a1"}
    assert fits["a1"] is not None and not fits["a1"].ok


def _digest(path: str) -> str:
    norm = os.path.normcase(Path(path).absolute())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:6]


def test_a_misfit_is_keyed_by_its_file_name() -> None:
    def free(_key: str) -> bool:
        return False

    assert misfit_key("/rec/Neon Drift/Box Stack.mp4", free) == "box-stack"
    name = "/rec/Neon Drift/Box Stack 2024.11.14 - 19.01.53.09.DVR.mp4"
    assert misfit_key(name, free) == "box-stack"


@pytest.mark.parametrize(
    "path", ["/rec/VODs/2026-11-08 19-01-53.mkv", "/rec/VODs/a1.mp4", "/rec/VODs/videos.mp4"]
)
def test_a_misfit_with_a_generic_name_gets_the_folder_key_and_a_hash(path: str) -> None:
    assert misfit_key(path, lambda _k: False) == f"vods-{_digest(path)}"


def test_a_misfit_whose_name_is_taken_gets_the_folder_key_and_a_hash() -> None:
    path = "/rec/Neon Drift/Box Stack.mp4"
    assert misfit_key(path, {"box-stack"}.__contains__) == f"neon-drift-{_digest(path)}"


def test_profiles_pool() -> None:
    a = Profile(4, 10, (("0,0", 2),))
    b = Profile(6, 20, (("0,0", 3), ("15,11", 6)))
    assert a + b == Profile(10, 30, (("0,0", 5), ("15,11", 6)))
    assert (a + b).grid() == {"0,0": 0.5, "15,11": 0.6}


# --------------------------------------------------------------------------- pipeline


class _Fake:
    """Fake discovery and sampler for ``pipeline._discover``: each source's text sits in the
    top-left (``"a"``) or bottom-right (``"b"``) corner, as ``corner`` of its file name says;
    discovery proposes a region in each corner its kept sources use."""

    def __init__(self, corner: dict[str, str]) -> None:
        self.corner = corner
        self.runs: list[tuple[str, list[str], bool]] = []  # (game, file names, re-cluster)
        self.sampled: list[str] = []
        self.fail: set[tuple[str, bool]] = set()  # (game, re-cluster) runs that are interrupted
        self.ref: object = None  # when set, the fit_ref every discovery proposes
        self.frames: dict[str, int] = {}  # per file name: frames a discovery reads (else 12/8)
        self.kept: list[set[str]] = []  # the kept src_keys of each re-cluster

    def records(self, src, n: int) -> list[dict]:
        from cutpoint.events import discover  # noqa: PLC0415

        key = discover.src_key(src)
        out = _frames(self.corner[src.name], n)
        return [
            f | {"id": f"{key}@{i}", "source": src.name, "src_key": key} for i, f in enumerate(out)
        ]

    def discover_regions(self, srcs, out_dir, game, params, on_frame=None, keep_src_keys=None, **_):
        import json  # noqa: PLC0415
        from dataclasses import asdict  # noqa: PLC0415

        from cutpoint.events import discover  # noqa: PLC0415

        self.runs.append((game, [s.name for s in srcs], keep_src_keys is not None))
        if (game, keep_src_keys is not None) in self.fail:
            raise KeyboardInterrupt
        recs = [
            r
            for i, s in enumerate(srcs)
            for r in self.records(s, self.frames.get(s.name, 12 if i == 0 else 8))
        ]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ocr.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8"
        )
        if keep_src_keys is not None:
            self.kept.append(set(keep_src_keys))
            recs = [r for r in recs if discover._src(r) in keep_src_keys]
        used = {self.corner[r["source"]] for r in recs}
        regs = [r for c, r in (("a", TOP_LEFT), ("b", BOTTOM_RIGHT)) if c in used]
        return {
            "game": game,
            "source": "discovered",
            "regions": [asdict(r) for r in regs],
            "discovery": {
                "frames": [r["id"] for r in recs],
                "fit_ref": self.ref if self.ref is not None else profile(recs, MIN_SCORE).ref(),
            },
        }

    def sample_frames(self, sources, out_dir, params) -> list[dict]:
        self.sampled += [s.name for s in sources]
        return [r for s in sources for r in self.records(s, 8)]


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """Sources in the game folder ``Neon Drift`` (one first guess, ``neon-drift``)."""
    from types import SimpleNamespace  # noqa: PLC0415

    from cutpoint.events import discover, pipeline  # noqa: PLC0415
    from cutpoint.events.killfeed import OcrParams  # noqa: PLC0415

    state = SimpleNamespace(project=tmp_path / "proj", fake=None)

    def sources(corner: dict[str, str], where: str = "Neon Drift") -> list:
        state.fake = fake = _Fake(corner)
        monkeypatch.setattr(discover, "discover_regions", fake.discover_regions)
        monkeypatch.setattr(game_fit, "sample_frames", fake.sample_frames)
        out = []
        for name in corner:
            path = tmp_path / where / name
            path.parent.mkdir(exist_ok=True)
            if not path.exists():  # rewritten, it would be another file
                path.write_bytes(b"x")
            out.append(SimpleNamespace(path=path, name=name, duration=600.0, io_path=path))
        return out

    def run(srcs, force=None) -> dict:
        return pipeline._discover(state.project, srcs, OcrParams(), force, 4, "user")

    state.sources, state.run = sources, run
    return state


def _keys(project) -> dict[str, str]:
    from cutpoint.events import regions  # noqa: PLC0415

    # Entry names are case-folded where the file system ignores case (Windows): compare
    # with ``os.path.normcase(name)`` for names with capitals.
    return {Path(k).name: v["key"] for k, v in regions.load_game_keys(project).items()}


def test_two_games_in_one_folder_get_two_presets(folder, caplog) -> None:
    from cutpoint import paths  # noqa: PLC0415
    from cutpoint.events import regions  # noqa: PLC0415

    srcs = folder.sources({"alpha one.mp4": "a", "alpha two.mp4": "a", "beta one.mp4": "b"})
    with caplog.at_level("INFO"):
        folder.run(srcs)
    assert folder.fake.runs == [
        ("neon-drift", ["alpha one.mp4", "alpha two.mp4", "beta one.mp4"], False),
        ("neon-drift", ["alpha one.mp4", "alpha two.mp4", "beta one.mp4"], True),  # again
        ("beta-one", ["beta one.mp4"], False),
    ]
    assert folder.fake.sampled == []  # discovery read enough frames of each
    assert (
        "beta one.mp4 does not fit the HUD preset 'neon-drift' (score 0.00); "
        "discovering it as 'beta-one'"
    ) in caplog.messages
    assert _keys(folder.project) == {
        "alpha one.mp4": "neon-drift",
        "alpha two.mp4": "neon-drift",
        "beta one.mp4": "beta-one",
    }
    entries = regions.load_game_keys(folder.project)
    beta = entries[regions.source_id(srcs[2].path)]
    assert beta["guess"] == "neon-drift" and beta["tag"] == regions.source_tag(srcs[2].path)
    # its text lands in cells the folder's game never uses
    assert beta["fit"] == {"score": 0.0, "frames": 8, "ref": "run"}
    # the folder's preset was made again from the sources that fit: no region of the other game
    names = [r.name for r in regions.load_preset("neon-drift").regions]
    assert names == ["feed"]
    assert (paths.presets_dir() / "beta-one.json").is_file()
    for s, key in zip(srcs, ("neon-drift", "neon-drift", "beta-one"), strict=True):
        assert regions.game_key_for_path(str(s.path), None, folder.project) == key

    # a second run reads the decisions: no discovery, no sampling, no INFO line
    folder.fake.runs.clear()
    caplog.clear()
    with caplog.at_level("INFO"):
        assert folder.run(srcs) == {}
    assert folder.fake.runs == [] and folder.fake.sampled == []
    assert not [m for m in caplog.messages if "does not fit" in m]


def test_one_game_in_one_folder_keeps_one_discovery(folder, caplog) -> None:
    srcs = folder.sources({"alpha one.mp4": "a", "alpha two.mp4": "a"})
    with caplog.at_level("INFO"):
        folder.run(srcs)
    assert folder.fake.runs == [("neon-drift", ["alpha one.mp4", "alpha two.mp4"], False)]
    assert _keys(folder.project) == {"alpha one.mp4": "neon-drift", "alpha two.mp4": "neon-drift"}
    assert not [m for m in caplog.messages if "does not fit" in m]


def test_out_of_memory_in_the_fit_check_names_the_step_and_the_sources(folder, monkeypatch) -> None:
    from cutpoint.errors import LowMemoryError, is_out_of_memory  # noqa: PLC0415

    folder.run(folder.sources({"alpha one.mp4": "a", "alpha two.mp4": "a"}))
    srcs = folder.sources({"alpha one.mp4": "a", "alpha two.mp4": "a", "alpha three.mp4": "a"})
    err = RuntimeError("CUDA out of memory. Tried to allocate 64.00 MiB (GPU 0)")

    def sample(sources, out_dir, params):
        raise err

    monkeypatch.setattr(game_fit, "sample_frames", sample)
    with pytest.raises(LowMemoryError) as got:
        folder.run(srcs)
    assert str(got.value).startswith(
        "Out of memory in the HUD fit check (neon-drift: alpha three.mp4); close other programs"
    )
    assert got.value.__cause__ is err and is_out_of_memory(err)


def test_a_new_source_is_checked_against_the_preset_found(folder, caplog) -> None:
    from cutpoint.events import regions  # noqa: PLC0415

    folder.run(folder.sources({"alpha one.mp4": "a", "alpha two.mp4": "a"}))
    srcs = folder.sources(
        {"alpha one.mp4": "a", "alpha two.mp4": "a", "alpha three.mp4": "a", "beta two.mp4": "b"}
    )
    with caplog.at_level("INFO"):
        folder.run(srcs)
    assert folder.fake.sampled == ["alpha three.mp4", "beta two.mp4"]  # the undecided ones
    assert folder.fake.runs == [("beta-two", ["beta two.mp4"], False)]
    assert _keys(folder.project)["alpha three.mp4"] == "neon-drift"
    assert _keys(folder.project)["beta two.mp4"] == "beta-two"
    fit = regions.load_game_keys(folder.project)[regions.source_id(srcs[2].path)]["fit"]
    assert fit["ref"] == "preset" and fit["score"] == 1.0
    assert (
        "beta two.mp4 does not fit the HUD preset 'neon-drift' (score 0.00); "
        "discovering it as 'beta-two'"
    ) in caplog.messages


def test_a_second_misfit_joins_the_first_misfits_preset(folder, caplog) -> None:
    srcs = folder.sources(
        {"alpha one.mp4": "a", "alpha two.mp4": "a", "beta one.mp4": "b", "beta two.mp4": "b"}
    )
    with caplog.at_level("INFO"):
        folder.run(srcs)
    assert "beta two.mp4 fits the HUD preset 'beta-one' discovered for beta one.mp4" in (
        caplog.messages
    )
    # one discovery for both of the other game; the second fits the first one's preset
    assert [g for g, _, again in folder.fake.runs if not again] == ["neon-drift", "beta-one"]
    assert _keys(folder.project)["beta two.mp4"] == "beta-one"


def test_game_and_force(folder) -> None:
    """``--game`` skips the check; ``--discover-regions`` checks every source again."""
    from cutpoint.events import pipeline  # noqa: PLC0415
    from cutpoint.events.killfeed import OcrParams  # noqa: PLC0415

    srcs = folder.sources({"alpha one.mp4": "a", "beta one.mp4": "b"})
    pipeline._discover(folder.project, srcs, OcrParams(game="zone-war"), None, 4, "user")
    assert folder.fake.runs == [("zone-war", ["alpha one.mp4", "beta one.mp4"], False)]
    assert _keys(folder.project) == {}
    folder.run(srcs)
    folder.fake.runs.clear()
    folder.run(srcs, force=True)  # presets kept, discovery runs, the check again
    assert [g for g, _, _ in folder.fake.runs] == ["neon-drift", "beta-one"]
    assert folder.fake.sampled == []
    assert _keys(folder.project)["beta one.mp4"] == "beta-one"


def test_date_named_videos_of_two_games_in_a_generic_folder(folder, caplog) -> None:
    """One group keyed by the first file's name; the misfit's own name is generic: the
    folder key and a hash of its path."""
    srcs = folder.sources(
        {
            "2026-11-08 19-01-53.mkv": "a",
            "2026-11-08 21-30-00.mkv": "a",
            "2026-11-09 18-00-00.mkv": "b",
        },
        where="VODs",
    )
    with caplog.at_level("INFO"):
        folder.run(srcs)
    from cutpoint.events import regions  # noqa: PLC0415

    folder_id = regions.source_id(srcs[0].path.parent)
    assert regions.load_game_groups(folder.project) == {folder_id: "2026-11-08 19-01-53"}
    misfit = f"vods-{_digest(str(srcs[2].path))}"
    assert [g for g, _, again in folder.fake.runs if not again] == ["2026-11-08-19-01-53", misfit]
    assert _keys(folder.project) == {
        "2026-11-08 19-01-53.mkv": "2026-11-08-19-01-53",
        "2026-11-08 21-30-00.mkv": "2026-11-08-19-01-53",
        "2026-11-09 18-00-00.mkv": misfit,
    }
    assert (
        "2026-11-09 18-00-00.mkv does not fit the HUD preset '2026-11-08-19-01-53' (score 0.00); "
        f"discovering it as '{misfit}'"
    ) in caplog.messages


_R1, _R2, _R3 = (f"Replay 2026-11-08 {h}-01-53.mp4" for h in (19, 20, 21))


@pytest.mark.parametrize(
    ("corner", "frames"),
    [
        # the group's seed (most gameplay frames of its game) is also the first by name
        ({_R1: "a", _R2: "a", _R3: "b"}, {_R1: 14, _R2: 8, _R3: 8}),
        # the other game is first by name (it names the key), the seed comes second
        ({_R1: "b", _R2: "a", _R3: "a"}, {_R1: 8, _R2: 14, _R3: 8}),
        # the Windows case: the other game's source has the most frames of any one source,
        # but the game of two sources has more in total (50 + 42 > 52)
        ({_R1: "a", _R2: "a", _R3: "b"}, {_R1: 50, _R2: 42, _R3: 52}),
    ],
)
def test_a_mixed_date_group_keeps_its_key_on_the_largest_group(folder, caplog, corner, frames):
    """Two sources of one game and one of another in a date group of a generic folder: the
    game of the largest group (most gameplay frames in total) keeps the date group's key,
    its preset's regions and fit_ref come from that game's frames only, and the other game
    gets the misfit key."""
    from cutpoint.events import discover, regions  # noqa: PLC0415

    srcs = folder.sources(corner, where="clips")
    folder.fake.frames = frames
    with caplog.at_level("INFO"):
        folder.run(srcs)
    key = "replay-2026-11-08-19-01-53"
    a = [s for s in srcs if corner[s.name] == "a"]
    (b,) = [s for s in srcs if corner[s.name] == "b"]
    misfit = misfit_key(str(b.path), {key}.__contains__)  # its own name, unless that is the key
    assert _keys(folder.project) == {os.path.normcase(s.name): key for s in a} | {
        os.path.normcase(b.name): misfit
    }
    entries = regions.load_game_keys(folder.project)
    leader = max(a, key=lambda s: folder.fake.frames[s.name])
    fit = entries[regions.source_id(leader.path)]["fit"]
    # the seed fits its own preset
    assert fit == {"score": 1.0, "frames": frames[leader.name], "ref": "run"}
    assert entries[regions.source_id(b.path)]["fit"]["score"] == 0.0
    # one OCR pass over all three, a re-cluster from the leader's game only, the misfit's own
    assert folder.fake.runs == [
        (key, list(corner), False),
        (key, list(corner), True),
        (misfit, [b.name], False),
    ]
    assert folder.fake.kept == [{discover.src_key(s) for s in a}]
    preset = regions.load_preset(key)
    assert [r.name for r in preset.regions] == ["feed"]  # no region of the other game
    data = json.loads(Path(preset.path).read_text(encoding="utf-8"))
    a_recs = [r for s in a for r in folder.fake.records(s, folder.fake.frames[s.name])]
    assert data["discovery"]["fit_ref"] == profile(a_recs, MIN_SCORE).ref()
    assert game_fit.against_ref(
        profile(folder.fake.records(leader, frames[leader.name]), MIN_SCORE),
        data["discovery"]["fit_ref"],
    ).ok
    assert [r.name for r in regions.load_preset(misfit).regions] == ["corner"]
    assert (
        f"{b.name} does not fit the HUD preset '{key}' (score 0.00); discovering it as '{misfit}'"
    ) in caplog.messages


def test_sampling_loads_no_ocr_model_without_frames_to_read(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace  # noqa: PLC0415

    from cutpoint.events import discover, killfeed  # noqa: PLC0415

    def engine(**_kwargs):
        raise AssertionError("OCR model loaded")

    monkeypatch.setattr(killfeed, "OcrEngine", engine)
    monkeypatch.setattr(discover, "_burst_frames", lambda *a: [])  # nothing decodes
    path = tmp_path / "gone.mp4"
    src = SimpleNamespace(path=path, name=path.name, duration=600.0, io_path=path)
    assert game_fit.sample_frames([src], tmp_path / "_fit", discover.DiscoverParams()) == []
    assert not (tmp_path / "_fit").exists()  # no empty folder left behind


def test_an_interrupted_misfit_discovery_keeps_the_split(folder) -> None:
    """The folder's preset is installed only from the sources that fit, each decision is
    saved when made: a run interrupted in the misfit's discovery leaves no mixed preset, and
    the next run discovers the misfit under the key it was given (no hash key)."""
    from cutpoint.events import regions  # noqa: PLC0415

    srcs = folder.sources({"alpha one.mp4": "a", "alpha two.mp4": "a", "beta one.mp4": "b"})
    folder.fake.fail = {("beta-one", False)}
    with pytest.raises(KeyboardInterrupt):
        folder.run(srcs)
    assert [r.name for r in regions.load_preset("neon-drift").regions] == ["feed"]
    assert regions.load_preset("beta-one") is None
    assert _keys(folder.project) == {
        "alpha one.mp4": "neon-drift",
        "alpha two.mp4": "neon-drift",
        "beta one.mp4": "beta-one",
    }
    folder.fake.fail.clear()
    folder.fake.runs.clear()
    folder.run(srcs)
    assert folder.fake.runs == [("beta-one", ["beta one.mp4"], False)]
    assert folder.fake.sampled == []
    assert regions.load_preset("beta-one") is not None


def test_an_interrupted_split_installs_no_preset(folder) -> None:
    from cutpoint.events import regions  # noqa: PLC0415

    srcs = folder.sources({"alpha one.mp4": "a", "beta one.mp4": "b", "alpha two.mp4": "a"})
    folder.fake.fail = {("neon-drift", True)}  # in the re-cluster from the sources that fit
    with pytest.raises(KeyboardInterrupt):
        folder.run(srcs)
    assert regions.load_preset("neon-drift") is None  # not the mix of both games
    assert _keys(folder.project) == {}


def test_a_lost_decision_finds_the_preset_made_from_its_source(folder, caplog) -> None:
    """The decisions file is gone (or another project): the misfit's own earlier preset is
    not "taken", so it keeps its key and is not discovered again."""
    from cutpoint.events import regions  # noqa: PLC0415

    srcs = folder.sources({"alpha one.mp4": "a", "alpha two.mp4": "a", "beta one.mp4": "b"})
    folder.run(srcs)
    regions.game_keys_path(folder.project).unlink()
    folder.fake.runs.clear()
    with caplog.at_level("INFO"):
        folder.run(srcs)
    assert folder.fake.runs == []  # both presets exist
    assert _keys(folder.project)["beta one.mp4"] == "beta-one"
    assert (
        "beta one.mp4 does not fit the HUD preset 'neon-drift' (score 0.00); "
        "discovering it as 'beta-one'"
    ) in caplog.messages


def test_a_preset_with_an_old_fit_ref_splits_nothing(folder, caplog) -> None:
    """A preset of an earlier version (per-region fit_ref, or none) decides nothing: every
    source keeps the guess, without a misfit discovery."""
    from cutpoint import paths  # noqa: PLC0415
    from cutpoint.events import regions  # noqa: PLC0415

    srcs = folder.sources({"alpha one.mp4": "a", "beta one.mp4": "b"})
    for ref in ({"inside": 1.0, "occ": {"feed": 1.0}}, None):
        target = paths.presets_dir() / "neon-drift.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        disc = {"fit_ref": ref} if ref is not None else {}
        target.write_text(
            json.dumps({"game": "neon-drift", "regions": [asdict(TOP_LEFT)], "discovery": disc}),
            encoding="utf-8",
        )
        regions.game_keys_path(folder.project).unlink(missing_ok=True)
        folder.fake.runs.clear()
        with caplog.at_level("INFO"):
            folder.run(srcs)
        assert folder.fake.runs == []
        assert _keys(folder.project) == {
            "alpha one.mp4": "neon-drift",
            "beta one.mp4": "neon-drift",
        }
        assert not [m for m in caplog.messages if "does not fit" in m]
        entries = regions.load_game_keys(folder.project)
        assert all(e["fit"] is None for e in entries.values())


def test_discover_regions_clusters_only_the_kept_sources_and_writes_fit_ref(
    tmp_path, monkeypatch, caplog
) -> None:
    from types import SimpleNamespace  # noqa: PLC0415

    from cutpoint.events import discover  # noqa: PLC0415

    keys = {"a.mp4": "a.mp4~0000aaaa", "b.mp4": "b.mp4~0000bbbb"}
    names = ["Kip", "Zorbo", "Mellan", "Quaff", "Dross", "Tavik", "Brinn", "Oskel", "Pyra"]
    names += ["Velt", "Hunn"]  # invented

    def ocr_frames(bursts, params, frame_dir, cache_path, on_frame=None):
        out = []
        for bi, (src, _t) in enumerate(bursts):
            for k in range(params.burst_frames):
                boxes = [  # a changing stack at the top right, a label at the bottom left
                    _box(
                        1700,
                        60 + 30 * row,
                        text=f"{names[(bi * 4 + k + row) % 11]} hit {names[(bi * 3 + k * 2 + row * 5) % 11]}",
                    )
                    for row in range(3)
                ] + [_box(100, 1000, text="AMMO")]
                key = keys[src.name]
                out.append(
                    {
                        "id": f"{key}@{bi}.{k}",
                        "source": src.name,
                        "src_key": key,
                        "t": bi + k,
                        "burst": bi,
                        "k": k,
                        "frame": f"{bi}_{k}.jpg",
                        "boxes": boxes,
                    }
                )
        return out

    monkeypatch.setattr(discover, "_ocr_frames", ocr_frames)
    monkeypatch.setattr(discover, "probe", lambda path: None)
    monkeypatch.setattr(discover, "write_overlay", lambda *a: None)
    monkeypatch.setattr(discover, "src_key", lambda src: keys[src.name])
    srcs = [
        SimpleNamespace(name=n, path=tmp_path / n, io_path=tmp_path / n, duration=600.0)
        for n in keys
    ]
    params = discover.DiscoverParams(bursts=12)
    with caplog.at_level("INFO"):
        got = discover.discover_regions(
            srcs, tmp_path / "out", "zone-war", params, keep_src_keys={keys["a.mp4"]}
        )
    # the log counts the kept source and its bursts, not every source of the run
    assert (
        "Region discovery for zone-war again: 6 bursts x 4 frames over 1 source(s), "
        "from the frames already read"
    ) in caplog.messages
    frames = got["discovery"]["frames"]
    assert frames and all(f.startswith("a.mp4~0000aaaa@") for f in frames)
    ref = got["discovery"]["fit_ref"]
    assert ref["cells"] == [16, 12] and ref["frames"] == len(frames)
    # the stack at the top right and the label at the bottom left, in every frame
    assert ref["grid"] == {"14,0": 1.0, "14,1": 1.0, "0,11": 1.0}
    assert ref["boxes"] == 4 * len(frames)


def test_a_stale_decision_keeps_its_key_while_another_source_holds_it(folder) -> None:
    """beta one's file changed (stale decision) while beta two keeps "beta-one": beta one
    gets its old key back, no hash key and no new discovery."""
    import os  # noqa: PLC0415

    srcs = folder.sources(
        {"alpha one.mp4": "a", "alpha two.mp4": "a", "beta one.mp4": "b", "beta two.mp4": "b"}
    )
    folder.run(srcs)
    assert _keys(folder.project)["beta one.mp4"] == "beta-one"
    assert _keys(folder.project)["beta two.mp4"] == "beta-one"
    st = srcs[2].path.stat()
    os.utime(srcs[2].path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    folder.fake.runs.clear()
    folder.run(srcs)
    assert folder.fake.runs == []
    assert folder.fake.sampled == ["beta one.mp4"]
    assert _keys(folder.project)["beta one.mp4"] == "beta-one"


def _old_preset(discovery: dict | None) -> tuple[Path, bytes]:
    """A preset ``neon-drift`` of an earlier version: its file and bytes."""
    from cutpoint import cache, paths  # noqa: PLC0415

    data = {"game": "neon-drift", "regions": [asdict(TOP_LEFT)], "banner_phrases": ["ACE"]}
    if discovery is not None:
        data["discovery"] = discovery
    target = paths.presets_dir() / "neon-drift.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    cache.save_json(target, data)
    return target, target.read_bytes()


@pytest.mark.parametrize(
    "discovery",
    [
        None,
        {"elapsed_s": 12.5},
        {"elapsed_s": 12.5, "fit_ref": {"inside": 1.0, "occ": {"feed": 1}}},
    ],
)
def test_a_forced_discovery_gives_an_old_preset_the_fit_ref_only(folder, caplog, discovery) -> None:
    """``--discover-regions`` over a preset without a grid fit_ref adds (or replaces) only
    ``discovery.fit_ref``: regions and every other key stay."""
    srcs = folder.sources({"alpha one.mp4": "a", "alpha two.mp4": "a"})
    target, before = _old_preset(discovery)
    old = json.loads(before)
    with caplog.at_level("INFO"):
        folder.run(srcs, force=True)
    got = json.loads(target.read_text(encoding="utf-8"))
    ref = got["discovery"].pop("fit_ref")
    assert ref["cells"] == [16, 12] and set(ref["grid"]) == {"0,0", "1,0", "1,1"}
    old.setdefault("discovery", {}).pop("fit_ref", None)
    assert got == old
    assert list(got) == list(old)  # key order kept
    assert (
        f"Added the HUD fit check to the preset 'neon-drift' (its regions are unchanged): {target}"
    ) in caplog.messages
    assert _keys(folder.project) == {"alpha one.mp4": "neon-drift", "alpha two.mp4": "neon-drift"}


def test_a_forced_discovery_with_too_little_text_leaves_the_preset(folder, caplog) -> None:
    srcs = folder.sources({"alpha one.mp4": "c", "alpha two.mp4": "c"})
    target, before = _old_preset({"elapsed_s": 12.5})
    with caplog.at_level("INFO"):
        folder.run(srcs, force=True)
    assert target.read_bytes() == before
    assert not [m for m in caplog.messages if "Added the HUD fit check" in m]


@pytest.mark.parametrize(
    ("ref", "written"),
    [
        (None, True),  # the fake's own: every source's frames
        ({"cells": [16, 12], "grid": {"0,0": 1.0}, "frames": 2, "boxes": 6}, False),
        ({"cells": [16, 12], "grid": {"0,0": 1.0}, "frames": 8, "boxes": 8}, False),
        ({"cells": [16, 12], "grid": {}, "frames": 8, "boxes": 0}, False),
        ({"inside": 1.0, "occ": {"feed": 1.0}}, False),  # not a grid
    ],
)
def test_a_forced_discovery_of_a_named_game_writes_the_discovery_ref(folder, ref, written) -> None:
    """With ``--game`` nothing is checked: every source is the game, discovery's fit_ref of
    them all is written when it is usable."""
    from cutpoint.events import pipeline  # noqa: PLC0415
    from cutpoint.events.killfeed import OcrParams  # noqa: PLC0415

    srcs = folder.sources({"alpha one.mp4": "a", "alpha two.mp4": "a"})
    folder.fake.ref = ref
    target, before = _old_preset({"elapsed_s": 12.5})
    pipeline._discover(folder.project, srcs, OcrParams(game="neon-drift"), True, 4, "user")
    if written:
        got = json.loads(target.read_text(encoding="utf-8"))
        assert set(got["discovery"]["fit_ref"]["grid"]) == {"0,0", "1,0", "1,1"}
    else:
        assert target.read_bytes() == before


def test_a_forced_discovery_over_a_preset_splits_like_a_new_one(folder, caplog) -> None:
    """Two games in the folder, an older preset: the sources are checked against each other,
    the preset gets the grid of the leader's group only, the other game is split off."""
    srcs = folder.sources({"alpha one.mp4": "a", "alpha two.mp4": "a", "beta one.mp4": "b"})
    target, before = _old_preset({"elapsed_s": 12.5})
    with caplog.at_level("INFO"):
        folder.run(srcs, force=True)
    got = json.loads(target.read_text(encoding="utf-8"))
    leader_group = profile(_frames("a", 12) + _frames("a", 8), MIN_SCORE).ref()
    assert got["discovery"].pop("fit_ref") == leader_group
    assert got == json.loads(before)  # regions and every other key as they were
    assert (
        "beta one.mp4 does not fit the HUD preset 'neon-drift' (score 0.00); "
        "discovering it as 'beta-one'"
    ) in caplog.messages
    assert [g for g, _, _ in folder.fake.runs] == ["neon-drift", "beta-one"]
    assert _keys(folder.project) == {
        "alpha one.mp4": "neon-drift",
        "alpha two.mp4": "neon-drift",
        "beta one.mp4": "beta-one",
    }


def test_a_forced_discovery_anchors_on_the_presets_grid(folder, caplog) -> None:
    """The preset's game is the minority of the folder: its grid fit_ref decides, so its own
    source stays and the new grid is made from it; the other game is split off."""
    srcs = folder.sources({"beta one.mp4": "b", "beta two.mp4": "b", "alpha one.mp4": "a"})
    target, before = _old_preset(
        {"elapsed_s": 12.5, "fit_ref": profile(_frames("a", 10), MIN_SCORE).ref()}
    )
    with caplog.at_level("INFO"):
        folder.run(srcs, force=True)
    got = json.loads(target.read_text(encoding="utf-8"))
    # alpha one's 8 discovery frames (the third source), not the folder's majority
    assert got["discovery"].pop("fit_ref") == profile(_frames("a", 8), MIN_SCORE).ref()
    old = json.loads(before)
    old["discovery"].pop("fit_ref")
    assert got == old
    assert _keys(folder.project) == {
        "alpha one.mp4": "neon-drift",
        "beta one.mp4": "beta-one",
        "beta two.mp4": "beta-one",
    }
    assert "beta two.mp4 fits the HUD preset 'beta-one' discovered for beta one.mp4" in (
        caplog.messages
    )


def test_a_forced_discovery_where_no_source_fits_keeps_the_presets_grid(folder, caplog) -> None:
    srcs = folder.sources({"beta one.mp4": "b", "beta two.mp4": "b"})
    target, before = _old_preset(
        {"elapsed_s": 12.5, "fit_ref": profile(_frames("a", 10), MIN_SCORE).ref()}
    )
    with caplog.at_level("INFO"):
        folder.run(srcs, force=True)
    assert target.read_bytes() == before
    assert not [m for m in caplog.messages if "Added the HUD fit check" in m]
    assert _keys(folder.project) == {"beta one.mp4": "beta-one", "beta two.mp4": "beta-one"}


def test_a_preset_that_cannot_be_written_does_not_stop_the_run(folder, caplog, monkeypatch) -> None:
    from cutpoint import cache  # noqa: PLC0415

    srcs = folder.sources({"alpha one.mp4": "a", "alpha two.mp4": "a"})
    target, before = _old_preset(None)
    save = cache.save_json

    def locked(path, data, **kw):
        if Path(path) == target:
            raise PermissionError(13, "Permission denied", str(path))
        save(path, data, **kw)

    monkeypatch.setattr(cache, "save_json", locked)
    with caplog.at_level("INFO"):
        folder.run(srcs, force=True)
    assert target.read_bytes() == before
    warned = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "Could not add the HUD fit check" in r.message and str(target) in r.message for r in warned
    )
    assert _keys(folder.project) == {"alpha one.mp4": "neon-drift", "alpha two.mp4": "neon-drift"}


def test_a_bundled_preset_is_not_written(folder, monkeypatch) -> None:
    from cutpoint import paths  # noqa: PLC0415

    srcs = folder.sources({"alpha one.mp4": "a", "alpha two.mp4": "a"})
    target, before = _old_preset(None)
    monkeypatch.setattr(paths, "PRESETS_DIR", target.parent)  # as if it shipped with the package
    folder.run(srcs, force=True)
    assert target.read_bytes() == before


def test_a_run_without_force_leaves_an_old_preset(folder, caplog) -> None:
    srcs = folder.sources({"alpha one.mp4": "a", "beta one.mp4": "b"})
    target, before = _old_preset(None)
    with caplog.at_level("INFO"):
        folder.run(srcs)
    assert folder.fake.runs == []
    assert target.read_bytes() == before
    assert not [m for m in caplog.messages if "Added the HUD fit check" in m]
