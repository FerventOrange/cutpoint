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
from cutpoint.errors import FFmpegError

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

    def probe(src: Path, vision=None) -> SourceProbe:
        if Path(src).name.startswith("bad"):
            raise FFmpegError("cannot read")
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
        return se.evaluate(entries)[:2]

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
    game = by["Game"]  # general (ambiguous) wins by duration: the General style (#896), wrong
    assert (game["category"], game["ambiguous"]) == ("general", True)
    assert (game["style"], game["fallback"], game["right_style"], game["right"]) == (
        "action", True, "gaming", False,
    )  # fmt: skip
    other = by["Other"]  # ambiguous general footage: the General fallback is right
    assert (other["style"], other["fallback"], other["right"]) == ("action", True, True)
    assert by["Talk"]["style"] == "gaming" and by["Talk"]["right_style"] == "commentary"
    assert not by["Talk"]["right"]
    # the label of a project is that of most of its footage: x1 is 10 min, x2 1 min, both general
    assert by["Other"]["label"] == "general"
    mixed = [r for r in rows if r.project == "Game"]
    mixed[0].label = "talk"  # g1 (10 min) against g2 + g3 (60 min)
    assert se.project_label(mixed) == "gameplay"


def row(label, cat, ambiguous, minutes, project="P", **kw):
    path = f"{label}-{cat}-{minutes}.mp4"
    return se.Row(path, project, label, cat, ambiguous, minutes, "", 0.1, None, **kw)


def good_rows():
    """One clean source per label."""
    return [row(lab, lab, False, 10.0) for lab in se.CATEGORIES]


def test_flip_ready_yes_when_everything_passes():
    projects = [
        {
            "project": "P",
            "label": "talk",
            "style": "commentary",
            "right_style": "commentary",
            "fallback": False,
            "right": True,
        }
    ]
    res = se.flip_ready(good_rows(), [], projects, 0.25)
    assert res["ready"] and res["failed"] == []
    assert res["line"].startswith("Flip ready: YES") and "all 1 projects right" in res["line"]


def test_flip_ready_needs_every_label_by_count_and_by_minutes():
    rows = good_rows()  # gameplay: 1 sure source of 10 min
    rows.append(row("gameplay", "general", True, 1.0))  # 1 of 2 by count, 1 of 11 min
    res = se.flip_ready(rows, [], [], 0.25)
    assert res["failed"] == ["gameplay: 1/2 ambiguous (50%) > 25%"]  # minutes 9% pass
    rows = [r for r in good_rows() if r.label != "gameplay"]
    rows += [row("gameplay", "gameplay", False, 1.0 + i / 10) for i in range(4)]
    rows.append(row("gameplay", "general", True, 40.0))  # 1 of 5 by count, 40 of 45 min
    res = se.flip_ready(rows, [], [], 0.25)
    assert res["failed"] == ["gameplay: 40/45 min ambiguous (90%) > 25%"]


def test_flip_ready_fails_for_a_label_without_sources():
    rows = [r for r in good_rows() if r.label != "talk"]  # a partial labels file
    res = se.flip_ready(rows, [], [], 0.25)
    assert res["failed"] == ["talk: no sources labelled (nothing measured)"]
    assert not res["ready"]


def test_flip_ready_fails_for_untranscribed_long_general_sources():
    rows = good_rows()
    rows.append(row("general", "general", False, 20.0, untranscribed=True))
    res = se.flip_ready(rows, [], [], 0.25)
    assert len(res["failed"]) == 1 and res["failed"][0].startswith("untranscribed: 1 general")


def test_flip_ready_fails_for_unreadable_sources_and_missing_ocr():
    errors = [{"path": "x", "error": "e"}]
    res = se.flip_ready(good_rows(), [], [], 0.25, errors=errors, ocr=False)
    assert res["failed"][0] == "unreadable: 1 source(s) failed to probe > 0"
    assert res["failed"][1].startswith("HUD text check unavailable (lite?)")
    assert not res["ready"]


def test_flip_ready_lists_every_failed_criterion(run, tmp_path):
    rows, projects = run(entries(tmp_path))
    wrong = se.sure_wrong(rows)
    res = se.flip_ready(rows, wrong, projects, 0.25)
    assert not res["ready"] and res["line"].startswith("Flip ready: NO")
    assert "sure-wrong: 2 source(s) > 0" in res["failed"]
    assert "gameplay: 2/3 ambiguous (67%) > 25%" in res["failed"]  # per label, by count
    assert "gameplay: 60/70 min ambiguous (86%) > 25%" in res["failed"]  # by minutes
    assert "general: 1/2 ambiguous (50%) > 25%" in res["failed"]
    assert any(f.startswith("untranscribed: 2 general") for f in res["failed"])  # x1, g2
    assert any(f.startswith("project Talk (talk): style gaming (detected)") for f in res["failed"])
    assert "  FAIL general: 1/2 ambiguous (50%) > 25%" in res["line"]


def test_flip_ready_needs_every_project_right(run, tmp_path):
    _rows, projects = run(entries(tmp_path))
    res = se.flip_ready(good_rows(), [], projects, 0.25)
    assert [f.split(":")[0] for f in res["failed"]] == [
        "project Game (gameplay)",
        "project Talk (talk)",
    ]
    assert not res["ready"]


def test_evaluate_marks_untranscribed_and_records_unreadable(run, tmp_path):
    (tmp_path / "bad1.mp4").write_bytes(b"")
    bad = {"project": "Bad", "label": "general", "inputs": [str(tmp_path / "bad1.mp4")]}
    rows, projects, errors = se.evaluate([*entries(tmp_path), bad])
    by = {Path(r.path).name: r for r in rows}
    assert by["x1.mp4"].untranscribed and by["g2.mp4"].untranscribed  # general, 8+ min
    assert not by["g1.mp4"].untranscribed and not by["x2.mp4"].untranscribed
    assert [(e["project"], e["error"]) for e in errors] == [("Bad", "cannot read")]
    assert "Bad" not in {p["project"] for p in projects}  # nothing left to decide
    data = se.summarize(rows, projects, 0.25, errors, True)
    assert data["errors"] == errors
    assert any(f.startswith("unreadable: 1") for f in data["flip_ready"]["failed"])


def test_evaluate_refuses_out_before_probing(run, tmp_path, monkeypatch):
    monkeypatch.setattr(category, "probe_source", lambda src, vision=None: pytest.fail("probed"))
    with pytest.raises(ValueError, match="source's folder"):
        se.evaluate(entries(tmp_path), tmp_path / "result.json")
    inside = [{"label": "talk", "inputs": [str(tmp_path)]}]
    with pytest.raises(ValueError, match="source's folder"):  # inside an input folder
        se.evaluate(inside, tmp_path / "deep" / "r.json")


def test_evaluate_refuses_out_next_to_a_remapped_source(run, tmp_path, monkeypatch):
    other = tmp_path / "mirror"
    other.mkdir()
    monkeypatch.setattr(se, "remap_source", lambda p: other / Path(p).name)
    monkeypatch.setattr(category, "probe_source", lambda src, vision=None: pytest.fail("probed"))
    with pytest.raises(ValueError, match="source's folder"):
        se.evaluate(entries(tmp_path), other / "r.json")


def test_real_decide_on_canned_probes(monkeypatch):
    """The real ``category.decide``: telemetry is sure action cam; screen-recorder tags with
    no HUD is general and ambiguous; a long untranscribed general source is open for talk."""
    cam = SourceProbe("cam.mp4", 600.0, True)
    obs = SourceProbe(
        "obs.mp4",
        900.0,
        False,
        handler="VideoHandle",
        audio_tracks=2,
        frames=32,
        hud_static=0.0,
        static_share=0.0,
    )
    probes = {p.path: p for p in (cam, obs)}
    monkeypatch.setattr(category, "probe_source", lambda src, vision=None: probes[str(src)])
    monkeypatch.setattr(se, "find_sources", lambda inputs, exclude, recursive: list(inputs))
    monkeypatch.setattr(se, "remap_source", lambda p: p)
    monkeypatch.setattr(se, "source_segments", lambda src: None)
    ents = [
        {"project": "C", "label": "action-cam", "inputs": ["cam.mp4"]},
        {"project": "O", "label": "gameplay", "inputs": ["obs.mp4"]},
    ]
    rows, projects, errors = se.evaluate(ents)
    cam_row, obs_row = rows
    assert (cam_row.category, cam_row.ambiguous) == ("action-cam", False)
    assert (obs_row.category, obs_row.ambiguous) == ("general", True)
    assert obs_row.untranscribed and not cam_row.untranscribed
    assert [(p["project"], p["style"], p["fallback"], p["right"]) for p in projects] == [
        ("C", "action-cam", False, True),
        ("O", "action", True, False),  # unsure gameplay gets the General style (#896)
    ]
    assert se.sure_wrong(rows) == [] and errors == []


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
    assert diff["styles"] == [{"project": "Game", "old": "action", "new": "gaming-x"}]
    assert diff["sure_wrong"] == [2, 2]
    assert diff["ambiguous_rate"] == [0.375, 0.25]
    assert diff["added"] == [] and diff["removed"] == []
    assert diff["projects_added"] == [] and diff["projects_removed"] == []


def test_compare_reports_added_and_removed_sources_and_projects(run, tmp_path):
    rows, projects = run(entries(tmp_path))
    old = se.summarize(rows, projects, 0.25)
    kept = [r for r in rows if r.project != "Talk" and not r.path.endswith("a2.mp4")]
    new = se.summarize(
        [*kept, row("talk", "talk", False, 5.0, project="New")],
        [*[p for p in projects if p["project"] != "Talk"], dict(projects[0], project="New")],
        0.25,
    )
    diff = se.compare(old, new)
    assert diff["removed"] == sorted([str(tmp_path / "a2.mp4"), str(tmp_path / "t1.mp4")])
    assert diff["added"] == ["talk-talk-5.0.mp4"]
    assert diff["projects_removed"] == ["Talk"] and diff["projects_added"] == ["New"]
    se.print_compare(diff, "old.json")


def test_load_old_checks_the_file(tmp_path):
    f = tmp_path / "old.json"
    f.write_text(json.dumps({"hello": 1}))
    with pytest.raises(ValueError, match="not a style_auto_eval --out file"):
        se.load_old(f)
    f.write_text("[1, 2]")
    with pytest.raises(ValueError, match="not a style_auto_eval --out file"):
        se.load_old(f)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"entries": [{"label": "talk", "inputs": ["nope.mp4"]}]}))
    assert se.main([str(labels), "--compare", str(f)]) == 1  # refused before any probing


def test_load_labels_checks_the_file(tmp_path):
    f = tmp_path / "labels.json"
    f.write_text(json.dumps({"entries": [{"label": "sport", "inputs": ["a.mp4"]}]}))
    with pytest.raises(ValueError, match="entry 1: label 'sport' is not one of"):
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
    assert se.main([str(labels), "--out", str(tmp_path / "result.json")]) == 1
    assert not (tmp_path / "result.json").exists()
