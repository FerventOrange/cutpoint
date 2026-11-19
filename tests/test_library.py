"""``cutpoint.library``: the name rule, case-clash check, lookup, delete and target parsing the
style files and the activity profiles share. Messages are checked in both callers."""

from __future__ import annotations

from pathlib import Path

import pytest

from cutpoint import activity, library, stylefiles


def test_format_problem_names_the_kind_and_accepts_safe_names() -> None:
    assert library.format_problem("bmx_2-a", "style") is None
    for bad in ("", "-x", "a b", "../x", "x" * 65, None, 3):
        msg = library.format_problem(bad, "profile")
        assert msg is not None and msg.startswith(f"{bad!r} is not a usable profile name")
    assert library.format_problem("x" * 64, "style") is None


def test_reserved_problem_ignores_case() -> None:
    assert library.reserved_problem("NUL") == (
        "'NUL' cannot be a file name on Windows; choose another name"
    )
    assert library.reserved_problem("lpt9") is not None
    assert library.reserved_problem("lpt10") is None
    assert library.reserved_problem("console") is None


def test_both_name_rules_share_the_reserved_stems_and_keep_their_own_words() -> None:
    assert stylefiles.name_problem("com1") == library.reserved_problem("com1")
    assert activity.name_problem("com1") == library.reserved_problem("com1")
    assert "built-in style name" in (stylefiles.name_problem("action") or "")
    assert "built-in profile name" in (activity.name_problem("generic") or "")
    assert "usable style name" in (stylefiles.name_problem("a b") or "")
    assert "usable profile name" in (activity.name_problem("a b") or "")


def test_case_clash_problem_finds_a_file_that_differs_only_in_case() -> None:
    files = {"Bmx": Path("Bmx.json"), "other": Path("other.json")}
    assert library.case_clash_problem(files, "bmx", "style") == (
        "a style named 'Bmx' exists; names that differ only in case clash on Windows and macOS"
    )
    assert library.case_clash_problem(files, "Bmx", "style") is None  # the same name: --force's
    assert library.case_clash_problem(files, "new", "profile") is None


def test_find_returns_the_file_or_raises_the_given_message() -> None:
    files = {"a": Path("a.json")}
    assert library.find(files, "a", "missing") == Path("a.json")
    with pytest.raises(FileNotFoundError, match=r"^no a$"):
        library.find(files, "b", "no a")


def test_delete_unlinks_and_returns_the_file(tmp_path: Path) -> None:
    f = tmp_path / "a.json"
    f.write_text("{}", encoding="utf-8")
    assert library.delete({"a": f}, "a", "missing") == f
    assert not f.exists()
    with pytest.raises(FileNotFoundError, match="missing"):
        library.delete({"a": f}, "b", "missing")


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("a.json", True),
        ("dir/a", True),
        ("dir" + chr(92) + "a", True),
        ("bmx", False),
        ("jsonx", False),
    ],
)
def test_looks_like_path(target: str, expected: bool) -> None:
    assert library.looks_like_path(target) is expected


def test_file_target_needs_an_existing_file(tmp_path: Path) -> None:
    f = tmp_path / "x.json"
    f.write_text("{}", encoding="utf-8")
    assert library.file_target(str(f)) == f
    with pytest.raises(FileNotFoundError, match=r"nope\.json: file not found"):
        library.file_target("nope.json")
    with pytest.raises(FileNotFoundError, match="file not found"):
        library.file_target(str(tmp_path))  # a folder is no file
