"""``scripts/style_auto_eval.py``, the #513 ``--style auto`` measurement harness: the confusion
counts, the sure-wrong list, the minute weighting, the project majority and style, ``--compare``
and the flip-ready line. ``probe_source`` and ``decide`` are patched to canned results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import load_script
from cutpoint import category
from cutpoint.category import Detection, SourceDecision, SourceProbe

pytestmark = pytest.mark.script
se = load_script("style_auto_eval")

# name -> (true label, duration s, decided category, ambiguous)
CANNED = {
    "a1.mp4": ("action-cam", 600, "action-cam", False),
    "a2.mp4": ("action-cam", 600, "action-cam", False),
    "g1.mp4": ("gameplay", 600, "gameplay", False),
    "g2.mp4": ("gameplay", 3000, "general", True),
    "g3.mp4": ("gameplay", 600, "talk", True),
    "t1.mp4": ("talk", 1200, "gameplay", False),  # sure-wrong
    "x1.mp4": ("general", 600, "general", True),
    "x2.mp4": ("general", 60, "action-cam", False),  # sure-wrong
}


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``run(canned, entries)`` -> ``(rows, projects)`` with the detection patched."""
    for name in CANNED:
        (tmp_path / name).write_bytes(b"")

    def probe(src: Path) -> SourceProbe:
        return SourceProbe(str(src), float(CANNED[Path(src).name][1]), False)

    def decide(probes, speech=None) -> Detection:
        srcs = [
            SourceDecision(p.path, p.duration, CANNED[Path(p.path).name][2],
                           CANNED[Path(p.path).name][3], "why")
            for p in probes
        ]  # fmt: skip
        weight: dict[str, float] = {}
        for s in srcs:
            weight[s.category] = weight.get(s.category, 0) + s.duration
        cat = max(weight, key=lambda k: weight[k])
        unsure = sum(s.duration for s in srcs if s.category == cat and s.ambiguous)
        return Detection(cat, unsure >= weight[cat] / 2, "why", {}, srcs)

    monkeypatch.setattr(category, "probe_source", probe)
    monkeypatch.setattr(category, "decide", decide)
    monkeypatch.setattr(se, "source_segments", lambda src: None)

    def go(entries):
        return se.evaluate(entries)

    return go


def entries(tmp_path: Path) -> list[dict]:
    def e(project, label, *names):
        return {"project": project, "label": label, "inputs": [str(tmp_path / n) for n in names]}

    return [
        e("Cam", "action-cam", "a1.mp4", "a2.mp4"),
        e("Game", "gameplay", "g1.mp4", "g2.mp4", "g3.mp4"),
        e("Talk", "talk", "t1.mp4"),
        e("Other", "general", "x1.mp4"),
        e("Other", "general", "x2.mp4"),  # same project name: one project
    ]


def test_confusion_counts_and_minutes(run, tmp_path):
    rows, _ = run(entries(tmp_path))
    c = se.confusion(rows)
    assert c["action-cam"] == {"action-cam sure": {"n": 2, "minutes": 20.0}}
    assert c["gameplay"] == {
        "gameplay sure": {"n": 1, "minutes": 10.0},
        "general ambiguous": {"n": 1, "minutes": 50.0},
        "talk ambiguous": {"n": 1, "minutes": 10.0},
    }
    assert c["talk"] == {"gameplay sure": {"n": 1, "minutes": 20.0}}
    assert c["general"]["general ambiguous"] == {"n": 1, "minutes": 10.0}
    assert list(c) == ["action-cam", "gameplay", "talk", "general"]


def test_sure_wrong_lists_only_sure_mismatches(run, tmp_path):
    rows, _ = run(entries(tmp_path))
    wrong = se.sure_wrong(rows)
    assert [(Path(w["path"]).name, w["label"], w["category"]) for w in wrong] == [
        ("t1.mp4", "talk", "gameplay"),
        ("x2.mp4", "general", "action-cam"),
    ]  # g2 and g3 are wrong but ambiguous: they fall back, so they are not listed


def test_ambiguous_rates_by_count_and_by_minutes(run, tmp_path):
    rows, _ = run(entries(tmp_path))
    a = se.ambiguous_rates(rows)
    g = a["gameplay"]
    assert (g["n"], g["ambiguous"], g["rate"]) == (3, 2, 0.6667)
    assert (g["minutes"], g["ambiguous_minutes"], g["minutes_rate"]) == (70.0, 60.0, 0.8571)
    assert a["action-cam"]["rate"] == 0.0
    assert a["all"]["n"] == 8 and a["all"]["ambiguous"] == 3


def test_project_majority_style_and_fallback(run, tmp_path):
    rows, projects = run(entries(tmp_path))
    by = {p["project"]: p for p in projects}
    assert len(projects) == 4 and by["Other"]["sources"] == 2
    cam = by["Cam"]
    assert (cam["style"], cam["fallback"], cam["right"]) == ("action-cam", False, True)
    game = by["Game"]  # general (ambiguous) wins by duration: the default style, which is right
    assert (game["category"], game["ambiguous"]) == ("general", True)
    assert (game["style"], game["fallback"], game["right_style"], game["right"]) == (
        "gaming", True, "gaming", True,
    )  # fmt: skip
    assert by["Talk"]["style"] == "gaming" and by["Talk"]["right_style"] == "commentary"
    assert not by["Talk"]["right"]
    # the label of a project is that of most of its footage: x1 is 10 min, x2 1 min, both general
    assert by["Other"]["label"] == "general"
    mixed = [r for r in rows if r.project == "Game"]
    mixed[0].label = "talk"  # g1 (10 min) against g2 + g3 (60 min)
    assert se.project_label(mixed) == "gameplay"


def test_flip_ready_lists_every_failed_criterion(run, tmp_path):
    rows, projects = run(entries(tmp_path))
    wrong = se.sure_wrong(rows)
    res = se.flip_ready(rows, wrong, projects, 0.25)
    assert not res["ready"] and res["line"].startswith("Flip ready: NO")
    assert "sure-wrong: 2 source(s) > 0" in res["failed"]
    assert "gameplay: 2/3 ambiguous (67%) > 25%" in res["failed"]  # per label, by count
    assert "gameplay: 60/70 min ambiguous (86%) > 25%" in res["failed"]  # by minutes
    assert "general: 1/2 ambiguous (50%) > 25%" in res["failed"]
    assert any(
        f.startswith("project Talk (talk): style gaming (detected), right is commentary")
        for f in res["failed"]
    )
    assert "  FAIL general: 1/2 ambiguous (50%) > 25%" in res["line"]


def test_flip_ready_needs_every_label_by_count_and_by_minutes(run, tmp_path):
    rows, projects = run(entries(tmp_path))
    ok_projects = [dict(p, right=True) for p in projects]
    clean = [r for r in rows if r.label in ("action-cam", "gameplay")]
    for r in clean:  # gameplay: 1 of 3 by count (33%), but 10 of 70 minutes
        r.ambiguous = r.path.endswith("g3.mp4")
    res = se.flip_ready(clean, [], ok_projects, 0.25)
    assert res["failed"] == ["gameplay: 1/3 ambiguous (33%) > 25%"]  # minutes 14% pass
    next(r for r in clean if r.path.endswith("g2.mp4")).ambiguous = True  # 60 of 70 min
    next(r for r in clean if r.path.endswith("g3.mp4")).ambiguous = False  # 1/3 by count
    res = se.flip_ready(clean, [], ok_projects, 0.34)
    assert res["failed"] == ["gameplay: 50/70 min ambiguous (71%) > 34%"]  # count passes
    ok = se.flip_ready(clean, [], ok_projects, 0.75)
    assert ok["ready"] and ok["failed"] == []
    assert ok["line"].startswith("Flip ready: YES") and "all 4 projects right" in ok["line"]


def test_flip_ready_needs_every_project_right(run, tmp_path):
    rows, projects = run(entries(tmp_path))
    clean = [r for r in rows if r.label == "action-cam"]
    res = se.flip_ready(clean, [], projects, 0.25)
    assert [f.split(":")[0] for f in res["failed"]] == [
        "project Talk (talk)",
        "project Other (general)",
    ]
    assert not res["ready"]


def test_timing_counts_ocr_only_where_it_ran(run, tmp_path):
    rows, _ = run(entries(tmp_path))
    rows[0].probe_s, rows[1].probe_s, rows[1].ocr_s = 1.0, 5.0, 4.0
    t = se.timing(rows)
    assert t["probe_s"]["n"] == 8 and t["probe_s"]["max"] == 5.0
    assert t["ocr_s"] == {"n": 1, "total": 4.0, "mean": 4.0, "median": 4.0, "max": 4.0}


def test_timed_ocr_records_text_probe_calls(monkeypatch):
    monkeypatch.setattr(category, "text_probe", lambda *a, **k: "rec")
    spent: list[float] = []
    with se.timed_ocr(spent):
        assert category.text_probe("x") == "rec"
    assert len(spent) == 1
    assert category.text_probe("x") == "rec" and len(spent) == 1  # restored


def test_compare_shows_changes(run, tmp_path):
    rows, projects = run(entries(tmp_path))
    old = se.summarize(rows, projects, 0.25)
    rows2, projects2 = run(entries(tmp_path))
    g2 = next(r for r in rows2 if r.path.endswith("g2.mp4"))
    g2.category, g2.ambiguous = "gameplay", False  # the OCR made it sure
    next(p for p in projects2 if p["project"] == "Game").update(style="gaming-x")
    new = se.summarize(rows2, projects2, 0.25)
    diff = se.compare(old, new)
    assert diff["changed"] == [
        {
            "path": g2.path,
            "label": "gameplay",
            "old": "general ambiguous",
            "new": "gameplay sure",
        }
    ]
    assert diff["styles"] == [{"project": "Game", "old": "gaming", "new": "gaming-x"}]
    assert diff["sure_wrong"] == [2, 2]
    assert diff["ambiguous_rate"] == [0.375, 0.25]
    assert diff["added"] == [] and diff["removed"] == []


def test_load_labels_checks_the_file(tmp_path):
    f = tmp_path / "labels.json"
    f.write_text(json.dumps({"entries": [{"label": "sport", "inputs": ["a.mp4"]}]}))
    with pytest.raises(ValueError, match="entry 1: label must be one of"):
        se.load_labels(f)
    f.write_text(json.dumps({"entries": [{"label": "talk", "inputs": []}]}))
    with pytest.raises(ValueError, match="inputs"):
        se.load_labels(f)
    f.write_text(json.dumps({"entries": [{"label": "talk", "inputs": ["a/b.mp4"]}]}))
    assert se.entry_project(se.load_labels(f)[0]) == "b"


def test_main_writes_out_and_compares(run, tmp_path, capsys):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"entries": entries(tmp_path)}))
    out, other = tmp_path / "out" / "new.json", tmp_path / "out" / "old.json"
    assert se.main([str(labels), "--out", str(other)]) == 0
    assert se.main([str(labels), "--out", str(out), "--compare", str(other)]) == 0
    text = capsys.readouterr().out
    assert "SURE-WRONG: 2" in text and "no source changed its decision" in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["flip_ready"]["ready"] is False and data["flip_ready"]["failed"]
    assert data["compare"]["changed"] == []
    assert data["confusion"]["action-cam"]["action-cam sure"]["n"] == 2


def test_main_refuses_out_in_a_source_folder(run, tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"entries": entries(tmp_path)}))
    with pytest.raises(SystemExit):
        se.main([str(labels), "--out", str(tmp_path / "result.json")])
    assert not (tmp_path / "result.json").exists()
