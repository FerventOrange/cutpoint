"""The HUD fit check (#822): text boxes that land in a preset's regions fit, boxes in
another corner do not, too little text decides nothing, a merged two-game discovery splits,
and the key of a source that does not fit (invented boxes, regions and paths)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from cutpoint.events import game_fit
from cutpoint.events.game_fit import Profile, against_ref, profile, split_group
from cutpoint.events.regions import Region, misfit_key

MIN_SCORE = 0.5
TOP_LEFT = Region("feed", 0, 0, 400, 200, "feed")
BOTTOM_RIGHT = Region("corner", 1600, 900, 320, 180, "notice")


def _box(cx: int, cy: int, text: str = "ab12", score: float = 0.9) -> dict:
    return {"x0": cx - 20, "y0": cy - 8, "x1": cx + 20, "y1": cy + 8, "text": text, "score": score}


def _frames(corner: str, n: int = 8, per: int = 3) -> list[dict]:
    """``n`` frames with ``per`` boxes in the top-left (``"a"``) or bottom-right (``"b"``)."""
    at = {"a": [(60, 40), (120, 80), (200, 150)], "b": [(1700, 960), (1800, 1000), (1880, 1050)]}
    return [{"id": f"{corner}{i}", "boxes": [_box(*p) for p in at[corner][:per]]} for i in range(n)]


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


def test_text_in_the_same_regions_fits() -> None:
    ref = profile(_frames("a", 10), [TOP_LEFT], MIN_SCORE).ref()
    fit = against_ref(profile(_frames("a"), [TOP_LEFT], MIN_SCORE), ref)
    assert fit is not None and fit.ok
    assert fit.ref == "preset" and fit.score == pytest.approx(1.0)
    assert fit.record() == {"score": 1.0, "inside": 1.0, "frames": 8, "ref": "preset"}


def test_text_in_another_corner_does_not_fit() -> None:
    ref = profile(_frames("a", 10), [TOP_LEFT], MIN_SCORE).ref()
    fit = against_ref(profile(_frames("b"), [TOP_LEFT], MIN_SCORE), ref)
    assert fit is not None and not fit.ok
    assert fit.score == 0.0 and fit.inside == 0.0


@pytest.mark.parametrize("frames", [_frames("b", n=3), _frames("b", n=8, per=1)])
def test_too_little_text_decides_nothing(frames: list[dict]) -> None:
    """Fewer than 4 gameplay frames, or fewer than 15 edge boxes."""
    ref = profile(_frames("a", 10), [TOP_LEFT], MIN_SCORE).ref()
    assert against_ref(profile(frames, [TOP_LEFT], MIN_SCORE), ref) is None
    assert against_ref(profile(_frames("b"), [], MIN_SCORE), ref) is None  # no regions


def test_a_merged_discovery_of_two_games_splits_greedily() -> None:
    """One discovery over two games found regions of both; the source with the most frames
    leads, the others join when they look like the group."""
    regions = [TOP_LEFT, BOTTOM_RIGHT]
    profs = {
        "a1": profile(_frames("a", 12), regions, MIN_SCORE),
        "a2": profile(_frames("a", 8), regions, MIN_SCORE),
        "b1": profile(_frames("b", 8), regions, MIN_SCORE),
        "b2": profile(_frames("b", 2), regions, MIN_SCORE),  # too little to tell
    }
    fits = split_group(profs, {"a1": 300.0, "a2": 600.0, "b1": 900.0, "b2": 60.0})
    assert fits["a1"] is not None and fits["a1"].ok and fits["a1"].ref == "run"
    assert fits["a2"] is not None and fits["a2"].ok
    assert fits["b1"] is not None and not fits["b1"].ok
    assert fits["b2"] is None


def test_a_group_without_enough_evidence_keeps_every_guess() -> None:
    profs = {k: profile(_frames("a", 2), [TOP_LEFT], MIN_SCORE) for k in ("x", "y")}
    assert split_group(profs, {}) == {"x": None, "y": None}


def test_fixed_sources_join_the_group_untested() -> None:
    regions = [TOP_LEFT, BOTTOM_RIGHT]
    profs = {
        "b0": profile(_frames("b", 20), regions, MIN_SCORE),  # decided before: leads, stays
        "a1": profile(_frames("a", 8), regions, MIN_SCORE),
    }
    fits = split_group(profs, {}, frozenset({"b0"}))
    assert set(fits) == {"a1"}
    assert fits["a1"] is not None and not fits["a1"].ok


def test_a_preset_without_fit_ref_uses_the_absolute_rule() -> None:
    fit = against_ref(profile(_frames("a"), [TOP_LEFT], MIN_SCORE), None)
    assert fit is not None and fit.ok and fit.ref == "absolute"
    fit = against_ref(profile(_frames("b"), [TOP_LEFT], MIN_SCORE), {"occ": "bad"})
    assert fit is not None and not fit.ok and fit.ref == "absolute"
    # all text inside, but only in a banner region: no feed or notice hit
    banner = Region("banner", 0, 0, 400, 200, "banner")
    fit = against_ref(profile(_frames("a"), [banner], MIN_SCORE), None)
    assert fit is not None and not fit.ok


def test_a_fit_ref_needs_the_inside_share_too() -> None:
    """Same regions hit, but most of the text lands outside them: no fit."""
    ref = profile(_frames("a", 10), [TOP_LEFT], MIN_SCORE).ref()
    frames = [{"boxes": [_box(60, 40)] + [_box(1800, 1000)] * 4} for _ in range(8)]  # 1 of 5 inside
    fit = against_ref(profile(frames, [TOP_LEFT], MIN_SCORE), ref)
    assert fit is not None and fit.score == pytest.approx(1.0) and not fit.ok


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
    a = Profile(4, 10, 5, 1, (("feed", 2),))
    b = Profile(6, 20, 15, 2, (("feed", 3),))
    assert a + b == Profile(10, 30, 20, 3, (("feed", 5),))
    assert (a + b).inside_share == pytest.approx(2 / 3)
    assert (a + b).occ() == {"feed": 0.5}


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
        recs = [r for i, s in enumerate(srcs) for r in self.records(s, 12 if i == 0 else 8)]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ocr.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8"
        )
        if keep_src_keys is not None:
            recs = [r for r in recs if discover._src(r) in keep_src_keys]
        used = {self.corner[r["source"]] for r in recs}
        regs = [r for c, r in (("a", TOP_LEFT), ("b", BOTTOM_RIGHT)) if c in used]
        return {
            "game": game,
            "source": "discovered",
            "regions": [asdict(r) for r in regs],
            "discovery": {
                "frames": [r["id"] for r in recs],
                "fit_ref": profile(recs, regs, MIN_SCORE).ref(),
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
    # all its text is in the merged preset's regions, but not in those the folder's game uses
    assert beta["fit"] == {"score": 0.0, "inside": 1.0, "frames": 8, "ref": "run"}
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


def test_a_fit_ref_of_regions_the_preset_no_longer_has_uses_the_absolute_rule() -> None:
    ref = {"inside": 1.0, "occ": {"renamed": 1.0}}
    fit = against_ref(profile(_frames("a"), [TOP_LEFT], MIN_SCORE), ref)
    assert fit is not None and fit.ok and fit.ref == "absolute"


def test_discover_regions_clusters_only_the_kept_sources_and_writes_fit_ref(
    tmp_path, monkeypatch
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
    srcs = [
        SimpleNamespace(name=n, path=tmp_path / n, io_path=tmp_path / n, duration=600.0)
        for n in keys
    ]
    params = discover.DiscoverParams(bursts=12)
    got = discover.discover_regions(
        srcs, tmp_path / "out", "zone-war", params, keep_src_keys={keys["a.mp4"]}
    )
    frames = got["discovery"]["frames"]
    assert frames and all(f.startswith("a.mp4~0000aaaa@") for f in frames)
    ref = got["discovery"]["fit_ref"]
    assert set(ref) == {"inside", "occ"}
    assert got["regions"] and set(ref["occ"]) == {r["name"] for r in got["regions"]}


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
