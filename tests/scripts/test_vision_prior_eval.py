"""``scripts/vision_prior_eval.py``, the vision prior measurement (#948): the style group, the
ratings taken, the cells and the rule, and the command line, on synthetic ``highlights.json``
files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import load_script, write
from cutpoint import paths, vision

pytestmark = pytest.mark.script
vp = load_script("vision_prior_eval")


def rated(i: int, rating: float, kind: str = "frames", **extra) -> dict:
    """Candidate ``i`` rated ``rating`` with the ``kind`` prompt."""
    return {
        "id": f"h{i:03d}", "source": "a.mp4", "start": i * 10.0, "end": i * 10.0 + 5,
        "score": 50.0, "label": "x", "signals": {"vision": rating},
        "vision_rated": {"start": i * 10.0, "end": i * 10.0 + 5, "base": 50.0, "blended": 51.0,
                         "kind": kind, "centre": 5.0},
        **extra,
    }  # fmt: skip


def project(root: Path, name: str, style: str, cands: list[dict], **top) -> Path:
    out = root / name
    out.mkdir(parents=True)
    data = {"version": 1, "style": style, "settings": {}, "candidates": cands, **top}
    (out / "highlights.json").write_text(json.dumps(data), encoding="utf-8")
    return out


def test_only_current_ratings_with_a_kind_count(tmp_path):
    cands = [
        rated(0, 7.0),
        rated(1, 3.0, "speech"),
        rated(2, 9.0, vision_stale={"score": 9.0, "label": None, "window": [0, 1]}),  # stale
        {**rated(3, 8.0), "signals": {"vision": None}},  # stale, as sync_vision leaves it
        {"id": "h004", "signals": {"vision": 6.0}},  # rated before #909: no kind
        {**rated(5, 6.0), "vision_rated": {**rated(5, 6.0)["vision_rated"], "kind": None}},
        {"id": "h006", "signals": {}},  # never rated
    ]
    assert vp.ratings({"candidates": cands}) == [("frames", 7.0), ("speech", 3.0)]


@pytest.mark.parametrize(
    ("hl", "base", "resolved", "known"),
    [
        ({"style": "sport"}, "sport", True, True),
        ({"style": "gaming"}, "gaming", True, True),
        ({"style": "mystery"}, "mystery", False, False),  # not in the library: unknown
        ({"style": "mystery", "style_kind": "generic"}, "mystery", False, False),
        ({"style": "mystery", "style_kind": "action-cam"}, "action-cam", False, True),
    ],
)
def test_the_style_base(hl, base, resolved, known):
    assert vp.style_base(hl) == (base, resolved, known)


def test_a_user_style_resolves_to_its_base_from_the_library():
    write(paths.styles_dir() / "trailride.json", json.dumps({"base": "sport", "weights": {}}))
    assert vp.style_base({"style": "trailride"}) == ("sport", True, True)


def test_cells_per_group_and_kind_with_the_rule(tmp_path, monkeypatch):
    monkeypatch.setitem(vision.RATING_PRIOR, ("action", "frames"), 5.6)
    monkeypatch.setitem(vision.RATING_PRIOR, ("other", "frames"), 5.6)
    monkeypatch.setitem(vision.RATING_PRIOR, ("other", "speech"), 4.1)
    write(paths.styles_dir() / "trailride.json", json.dumps({"base": "sport", "weights": {}}))
    # action frames: 15 ratings of 7.0 over two projects -> used (n 15, delta 1.4)
    a = project(tmp_path, "ride", "action", [rated(i, 7.0) for i in range(10)])
    b = project(tmp_path, "trail", "trailride", [rated(i, 7.0) for i in range(5)])
    # other frames: 14 ratings of 3.0 -> not used (n 14); other speech: 20 of 4.3 -> not (0.2)
    c = project(
        tmp_path,
        "game",
        "gaming",
        [rated(i, 3.0) for i in range(14)] + [rated(20 + i, 4.3, "speech") for i in range(20)],
    )
    d = project(tmp_path, "novision", "funny", [{"id": "h001", "signals": {}}])
    data = vp.evaluate([vp.load_project(p) for p in (a, b, c, d)])
    cells = {(x["group"], x["kind"]): x for x in data["cells"]}
    assert set(cells) == {(g, k) for g in ("action", "other") for k in ("frames", "speech")}
    assert cells["action", "frames"] == {
        "group": "action", "kind": "frames", "n": 15, "mean": 7.0, "sd": 0.0, "prior": 5.6,
        "delta": 1.4, "use": True,
    }  # fmt: skip
    assert cells["other", "frames"]["n"] == 14 and cells["other", "frames"]["use"] is False
    assert cells["other", "frames"]["delta"] == -2.6
    assert cells["other", "speech"]["n"] == 20 and cells["other", "speech"]["use"] is False
    assert cells["action", "speech"] == {
        "group": "action", "kind": "speech", "n": 0, "mean": None, "sd": None,
        "prior": vision.RATING_PRIOR["action", "speech"], "delta": None, "use": False,
    }  # fmt: skip
    assert [(p["style"], p["group"], p["by_kind"]) for p in data["projects"]] == [
        ("action", "action", {"frames": 10}),
        ("trailride", "action", {"frames": 5}),
        ("gaming", "other", {"frames": 14, "speech": 20}),
    ]
    assert data["skipped"] == [str(d)]


def test_an_unresolved_user_style_stays_out_of_the_cells(tmp_path, capsys):
    a = project(
        tmp_path, "gone", "mystery", [rated(i, 7.0) for i in range(4)], style_kind="generic"
    )
    data = vp.evaluate([vp.load_project(a)])
    cells = {(x["group"], x["kind"]): x for x in data["cells"]}
    assert all(c["n"] == 0 for c in cells.values())
    assert data["unresolved"] == [str(a)] and data["projects"] == [] and data["skipped"] == []
    vp.print_results(data)
    assert "Unresolved" in capsys.readouterr().out


def test_sd_is_the_sample_sd():
    got = vp.cell([2.0, 4.0, 6.0], 4.0)
    assert got["mean"] == 4.0 and got["sd"] == 2.0 and got["delta"] == 0.0
    assert vp.cell([5.0], 4.0)["sd"] is None


def test_a_parent_folder_lists_its_projects_and_a_bare_folder_fails(tmp_path):
    a = project(tmp_path / "all", "b", "gaming", [])
    b = project(tmp_path / "all", "a", "gaming", [])
    (tmp_path / "all" / "notes").mkdir()
    assert vp.project_dirs([tmp_path / "all"]) == [b, a]
    assert vp.project_dirs([a]) == [a]
    with pytest.raises(FileNotFoundError):
        vp.project_dirs([tmp_path / "all" / "notes"])


def test_command_line_prints_and_writes_json(tmp_path, capsys):
    root = tmp_path / "projects"
    project(root, "ride", "action-cam", [rated(i, 6.5) for i in range(3)])
    project(root, "empty", "gaming", [])
    out = tmp_path / "res" / "prior.json"
    assert vp.main([str(root), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "action   frames     3   6.50   0.00" in text
    assert "ride: action-cam -> action: frames 3" in text
    assert "Skipped (no current vision rating):" in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == 1 and data["rule"] == {"min_n": 15, "min_delta": 0.5}
    assert data["skipped"] == [str(root / "empty")]
    before = sorted(p.name for p in root.rglob("*"))
    assert vp.main([str(root / "ride")]) == 0  # read-only: nothing new in the projects
    assert sorted(p.name for p in root.rglob("*")) == before


def test_command_line_fails_on_a_missing_folder(tmp_path):
    assert vp.main([str(tmp_path / "nope")]) == 1
