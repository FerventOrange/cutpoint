"""Style files (#514 phase 2): the schema-1 check, old files failing plainly, ``style_hash`` and
``cutpoint style list|show|create|validate|delete|features``. Invented styles and weights only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import run_json, warning_codes, write
from cutpoint import analyze, cache, stylefiles
from cutpoint.styles import STYLES, content_hash, get_style, load_learned, style_hash


@pytest.fixture
def styles_dir(home, monkeypatch) -> Path:
    folder = home / "my-styles"
    monkeypatch.setenv("CUTPOINT_STYLES_DIR", str(folder))
    return folder


def good(**over) -> dict:
    doc = {
        "schema": 1,
        "name": "mine",
        "base": "action",
        "source": "hand",
        "weights": {"scene": 0.4},
    }
    return {**doc, **over}


def put(folder: Path, name: str, doc: dict | str) -> Path:
    text = doc if isinstance(doc, str) else json.dumps(doc)
    return write(folder / f"{name}.json", text)


# ------------------------------------------------------------------ check()

_ZERO = dict.fromkeys(STYLES["action"].weights, 0)

PROBLEMS = [
    ({"schema": 2}, "schema"),
    ({"schema": True}, "schema"),
    ({"schema": "1"}, "schema"),
    ({"name": "other"}, "must match"),
    ({"base": "ski"}, "not a built-in"),
    ({"base": "nope"}, "not a built-in"),
    ({"base": "usr"}, "your own styles"),
    ({"base": None}, "missing"),
    ({"weights": {"bogus": 1}}, "unknown feature"),
    ({"weights": {"scene": -0.1}}, "negative"),
    ({"weights": {"scene": "high"}}, "not a number"),
    ({"weights": {"scene": True}}, "not a number"),
    ({"weights": _ZERO}, "every weight is 0"),
    ({"source": "robot"}, '"source"'),
    ({"source": None}, '"source" is missing'),
    ({"applies_to": {"categories": ["nope"]}}, "unknown categor"),
    ({"applies_to": {"zones": []}}, "unknown key"),
    ({"applies_to": {"categories": "talk"}}, "must be a list"),
    ({"mask_friends": "hard"}, "mask_friends"),
    ({"must_include_crash": 1.5}, "must_include_crash"),
    ({"description": 5}, "description"),
]


@pytest.mark.parametrize(("change", "text"), PROBLEMS, ids=[str(c) for c, _ in PROBLEMS])
def test_check_names_each_problem(styles_dir, change, text):
    put(styles_dir, "usr", {"name": "usr", "base": "action"})  # a user style, not a valid base
    doc = {k: v for k, v in good().items() if k not in change or change[k] is not None}
    doc.update({k: v for k, v in change.items() if v is not None})
    res = stylefiles.check(doc, "mine")
    assert not res.ok and any(text in p for p in res.problems), res.problems


def test_check_accepts_a_good_file_and_collects_every_problem(styles_dir):
    assert stylefiles.check(good(), "mine").ok
    bad = good(schema=3, name="x", base="nope", weights={"bogus": 1, "scene": -1})
    res = stylefiles.check(bad, "mine", [])
    assert len(res.problems) >= 4


def test_unknown_profile_is_only_a_warning():
    res = stylefiles.check(good(profile="no-such"), "mine", {"bmx"})
    assert res.ok and len(res.warnings) == 1
    assert stylefiles.check(good(profile="bmx"), "mine", {"bmx"}).warnings == []


# ------------------------------------------------------------------ loading and hash


def _no_traceback(caplog) -> None:
    assert all(r.exc_info is None for r in caplog.records if r.levelname == "ERROR")


def test_file_without_schema_fails_plainly_and_is_never_rewritten(styles_dir):
    path = put(styles_dir, "old", {"name": "old", "base": "action", "weights": {"scene": 0.3}})
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    with pytest.raises(ValueError, match=r'old\.json.*"schema" is missing.*add "schema": 1') as e:
        get_style("old")
    assert '"source" is missing' in str(e.value)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


OLD_WEIGHTS = [
    ("action-cam", "laughter", 'write "voice_laughter" instead'),
    ("action", "taste", "delete its weight"),
    ("action-cam", "taste", "delete its weight"),
]


@pytest.mark.parametrize(("base", "key", "hint"), OLD_WEIGHTS)
def test_old_weight_name_fails_with_the_file_key_and_fix(
    styles_dir, capsys, caplog, base, key, hint
):
    path = put(styles_dir, "old", good(name="old", base=base, weights={key: 0.4, "speech": 0.2}))
    before = path.read_bytes()
    problem = stylefiles.check(json.loads(before), "old").problems
    assert problem == [problem[0]] and f'"{key}" in "weights"' in problem[0] and hint in problem[0]
    with pytest.raises(ValueError, match=r"old\.json"):
        load_learned(path)
    for verb in ("show", "validate"):
        code, doc = style(capsys, verb, "old")
        assert code == 1 and doc["error"]["code"] == "bad_args", verb
        message = doc["error"]["message"]
        assert str(path) in message and f'"{key}"' in message and hint in message, message
    code, doc = style(capsys, "validate", str(path))
    assert code == 1 and hint in doc["error"]["message"]
    _no_traceback(caplog)
    assert path.read_bytes() == before


def test_laughter_is_still_a_weight_of_the_generic_styles(styles_dir):
    assert stylefiles.check(good(weights={"laughter": 0.4}), "mine").ok


def test_analyze_with_an_old_style_file_stops_plainly(styles_dir, tmp_path, capsys, caplog):
    put(styles_dir, "old", {"name": "old", "base": "action", "weights": {"scene": 0.3}})
    video = write(tmp_path / "v.mp4", "x")
    code, doc = run_json(capsys, ["analyze", str(video), "--style", "old"])
    assert code == 1 and doc["error"]["code"] == "error"  # as for any unusable style file
    assert (
        "old.json" in doc["error"]["message"] and '"schema" is missing' in doc["error"]["message"]
    )
    _no_traceback(caplog)


MASKS = [(True, True), (False, False), (7, True), (0.5, True), (0, False), (-1, False)]


@pytest.mark.parametrize(("value", "hard"), MASKS, ids=[repr(v) for v, _ in MASKS])
def test_mask_friends_takes_booleans_and_numbers(styles_dir, capsys, value, hard):
    put(styles_dir, "cam", good(name="cam", base="action-cam", weights={}, mask_friends=value))
    assert get_style("cam").mask_friends is hard
    code, doc = style(capsys, "show", "cam")
    assert code == 0 and doc["outputs"]["mask_friends"] is hard
    assert STYLES["action-cam"].mask_friends is True


def test_invalid_schema_file_names_itself_and_valid_one_loads(styles_dir):
    path = put(styles_dir, "mine", good(weights={"bogus": 1}))
    with pytest.raises(ValueError, match=r"mine\.json.*bogus"):
        get_style("mine")
    doc = good(weights={"scene": 0.7})
    put(styles_dir, "mine", doc)
    style = get_style("mine")
    assert style.weights["scene"] == 0.7 and style.file_hash == content_hash(doc)
    assert path.is_file()


def test_style_hash_changes_with_any_edit_and_builtins_are_stable(styles_dir):
    path = put(styles_dir, "mine", good())
    first = style_hash(get_style("mine"))
    assert len(first) == 12 and first == style_hash(get_style("mine"))
    put(styles_dir, "mine", good(description="edited"))
    assert style_hash(get_style("mine")) != first
    assert path.is_file()
    assert style_hash(STYLES["funny"]) == style_hash(get_style("funny"))
    assert style_hash(STYLES["funny"]) != style_hash(STYLES["action"])


def test_analyze_records_style_hash_and_rerank_works_without_it(home, tmp_path, monkeypatch):
    from test_analyze import _stub_probe_and_transcriber  # noqa: PLC0415

    write(tmp_path / "a.mp4")
    _stub_probe_and_transcriber(monkeypatch)
    proj = tmp_path / "proj"
    style = get_style("funny")
    analyze.analyze(tmp_path / "a.mp4", style, proj, vision=False, use_scenes=False)
    hl = cache.load_json(proj / "highlights.json")
    assert hl["settings"]["style_hash"] == style_hash(style)
    del hl["settings"]["style_hash"]  # a project analysed before the field existed
    cache.save_json(proj / "highlights.json", hl)
    analyze.reanalyze(proj)
    assert cache.load_json(proj / "highlights.json")["settings"]["style_hash"] == style_hash(style)


# ------------------------------------------------------------------ CLI


def style(capsys, *argv):
    return run_json(capsys, ["style", *argv])


def test_create_writes_ordered_keys_and_show_marks_origin(styles_dir, capsys):
    code, doc = style(capsys, "create", "mine", "--base", "action", "--weight", "scene=0.4",
                      "--description", "d", "--applies-to", "category:action-cam")  # fmt: skip
    assert code == 0 and doc["command"] == "style create" and doc["outputs"]["replaced"] is False
    on_disk = json.loads((styles_dir / "mine.json").read_text(encoding="utf-8"))
    assert list(on_disk) == [
        "schema",
        "name",
        "base",
        "source",
        "description",
        "applies_to",
        "weights",
    ]
    assert on_disk["source"] == "hand"
    code, doc = style(capsys, "show", "mine")
    rows = {w["feature"]: w["from"] for w in doc["outputs"]["weights"]}
    assert code == 0 and rows["scene"] == "style" and rows["loudness"] == "base"
    assert doc["outputs"]["hash"] == content_hash(on_disk)


@pytest.mark.parametrize("name", ["funny", "auto", "AUTO", "../x", "a b", "con"])
def test_create_refuses_unusable_names(styles_dir, capsys, name):
    code, doc = style(capsys, "create", name, "--base", "action")
    assert code == 1 and doc["error"]["code"] == "bad_args"
    assert not styles_dir.exists() or not list(styles_dir.glob("*.json"))


def test_create_existing_needs_force(styles_dir, capsys):
    put(styles_dir, "mine", good(description="old"))
    code, doc = style(capsys, "create", "mine", "--base", "action")
    assert code == 1 and doc["error"]["code"] == "output_not_empty"
    assert json.loads((styles_dir / "mine.json").read_text())["description"] == "old"
    code, doc = style(capsys, "create", "mine", "--base", "action", "--force")
    assert code == 0 and doc["outputs"]["replaced"] is True
    assert "description" not in json.loads((styles_dir / "mine.json").read_text())


def test_create_from_file_keeps_extra_keys_and_weight_overrides(styles_dir, tmp_path, capsys):
    src = write(tmp_path / "seed.json", json.dumps(
        {"schema": 1, "name": "seed", "base": "action", "source": "fitted",
         "fit": {"score": 0.9}, "weights": {"scene": 0.1, "speed": 0.2}}))  # fmt: skip
    code, _ = style(capsys, "create", "copy", "--from", str(src), "--weight", "scene=0.6")
    assert code == 0
    got = json.loads((styles_dir / "copy.json").read_text(encoding="utf-8"))
    assert got["name"] == "copy" and got["fit"] == {"score": 0.9} and got["source"] == "fitted"
    assert got["weights"] == {"scene": 0.6, "speed": 0.2}


def test_create_without_base_or_with_missing_from(styles_dir, tmp_path, capsys):
    code, doc = style(capsys, "create", "mine")
    assert code == 1 and doc["error"]["code"] == "bad_args"
    code, doc = style(capsys, "create", "mine", "--from", str(tmp_path / "gone.json"))
    assert code == 1 and doc["error"]["code"] == "not_found"
    code, doc = style(capsys, "create", "mine", "--base", "action", "--weight", "bogus=1")
    assert code == 1 and doc["error"]["code"] == "bad_args"
    assert not styles_dir.exists() or not list(styles_dir.glob("*.json"))


def test_create_with_unknown_profile_warns_but_saves(styles_dir, capsys):
    code, doc = style(capsys, "create", "mine", "--base", "action", "--profile", "no-such-profile")
    assert code == 0 and warning_codes(doc) == ["unknown_profile"]
    assert (styles_dir / "mine.json").is_file()


def test_weight_of_a_feature_the_base_never_measures_warns_but_saves(styles_dir, capsys):
    code, doc = style(capsys, "create", "mine", "--base", "gaming", "--weight", "airtime=0.3")
    assert code == 0 and warning_codes(doc) == ["feature_not_computed"]
    assert "'airtime' is never measured for base 'gaming'" in doc["warnings"][0]["message"]
    assert (styles_dir / "mine.json").is_file()
    code, doc = style(capsys, "validate", "mine")
    assert code == 0 and doc["outputs"]["valid"] is True
    assert warning_codes(doc) == ["feature_not_computed"]


def test_computed_features_follow_the_base_scorer():
    assert (
        stylefiles.check(good(base="action-cam", weights={"airtime": 0.3}), "mine").warnings == []
    )
    assert stylefiles.check(good(base="gaming", weights={"airtime": 0}), "mine").warnings == []
    res = stylefiles.check(good(base="action-cam", weights={"loudness": 0.3}), "mine")
    assert res.ok and len(res.warnings) == 1


def test_validate_old_file_and_invalid_fail(styles_dir, capsys, caplog):
    path = put(styles_dir, "old", {"name": "old", "base": "action", "weights": {"scene": 0.3}})
    code, doc = style(capsys, "validate", "old")
    assert code == 1 and doc["error"]["code"] == "bad_args" and doc["outputs"]["valid"] is False
    assert str(path) in doc["error"]["message"] and "legacy" not in doc["outputs"]
    assert any('"schema" is missing' in p for p in doc["outputs"]["problems"])
    _no_traceback(caplog)
    put(styles_dir, "mine", good(schema=2, weights={"scene": -1}))
    code, doc = style(capsys, "validate", "mine")
    assert code == 1 and doc["error"]["code"] == "bad_args"
    assert len(doc["outputs"]["problems"]) == 2
    code, doc = style(capsys, "validate", "funny")
    assert code == 0 and doc["outputs"]["type"] == "built-in"
    code, doc = style(capsys, "validate", "nobody")
    assert code == 1 and doc["error"]["code"] == "not_found"


def test_list_reports_invalid_and_unreadable_files_without_crashing(styles_dir, capsys):
    put(styles_dir, "good", good(name="good"))
    put(styles_dir, "old", {"name": "old", "base": "action", "weights": {}})
    put(styles_dir, "bad", good(name="bad", base="nope"))
    put(styles_dir, "broken", "{not json")
    code, doc = style(capsys, "list")
    assert code == 0 and doc["command"] == "style list"
    by = {e["name"]: e for e in doc["outputs"]["styles"]}
    assert by["funny"]["type"] == "built-in" and by["good"]["valid"] is True
    assert not by["old"]["valid"] and by["old"]["source"] is None and "legacy" not in by["old"]
    assert any('"schema" is missing' in p for p in by["old"]["problems"])
    assert not by["bad"]["valid"] and not by["broken"]["valid"]
    assert warning_codes(doc).count("invalid_style") == 3
    assert doc["outputs"]["styles_dir"] == str(styles_dir)


def test_show_unknown_and_delete(styles_dir, capsys):
    code, doc = style(capsys, "show", "nobody")
    assert code == 1 and doc["error"]["code"] == "not_found"
    code, doc = style(capsys, "show", "funny")
    assert code == 0 and {w["from"] for w in doc["outputs"]["weights"]} == {"base"}
    code, doc = style(capsys, "delete", "funny")
    assert code == 1 and doc["error"]["code"] == "bad_args"
    code, doc = style(capsys, "delete", "nobody")
    assert code == 1 and doc["error"]["code"] == "not_found"
    path = put(styles_dir, "mine", good())
    code, doc = style(capsys, "delete", "mine")
    assert code == 0 and doc["command"] == "style delete" and not path.exists()


def test_features_lists_the_grid_with_availability(home, capsys):
    code, doc = style(capsys, "features")
    assert code == 0 and doc["command"] == "style features"
    rows = {f["name"]: f for f in doc["outputs"]["features"]}
    assert {"scene", "speed", "voice_laughter"} <= set(rows)
    assert all(isinstance(f["available"], bool) for f in rows.values())


def test_force_replaces_a_file_without_schema_with_a_schema_file(styles_dir, capsys):
    put(styles_dir, "old", {"name": "old", "base": "action", "weights": {"scene": 0.3}})
    code, doc = style(capsys, "create", "old", "--base", "action", "--force")
    assert code == 0 and doc["outputs"]["replaced"] is True
    assert json.loads((styles_dir / "old.json").read_text(encoding="utf-8"))["schema"] == 1


def test_create_from_a_file_without_schema_or_source_writes_hand(styles_dir, tmp_path, capsys):
    src = write(tmp_path / "seed.json", json.dumps({"name": "seed", "base": "action"}))
    code, _ = style(capsys, "create", "copy", "--from", str(src))
    assert code == 0
    assert json.loads((styles_dir / "copy.json").read_text(encoding="utf-8"))["source"] == "hand"


def test_schema_file_without_weights_is_valid_and_uses_the_base(styles_dir):
    doc = {k: v for k, v in good().items() if k != "weights"}
    assert stylefiles.check(doc, "mine").ok
    put(styles_dir, "mine", doc)
    assert get_style("mine").weights == STYLES["action"].weights


def test_name_differing_only_in_case_is_refused(styles_dir, capsys):
    put(styles_dir, "mine", good())
    code, doc = style(capsys, "create", "MINE", "--base", "action")
    assert code == 1 and doc["error"]["code"] == "bad_args"
    assert [p.name for p in styles_dir.glob("*.json")] == ["mine.json"]


def test_create_from_a_file_with_a_removed_feature_is_refused(styles_dir, tmp_path, capsys):
    src = write(
        tmp_path / "seed.json",
        json.dumps({"name": "seed", "base": "action", "weights": {"taste": 0.3, "scene": 0.2}}),
    )
    code, doc = style(capsys, "create", "copy", "--from", str(src))
    assert code == 1 and doc["error"]["code"] == "bad_args"
    assert '"taste"' in doc["error"]["message"] and "delete its weight" in doc["error"]["message"]
    assert not (styles_dir / "copy.json").exists()
