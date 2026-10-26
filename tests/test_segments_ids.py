"""Segment ids of a hand-written segments file (issue #120) and the missing-file error (#127)."""

from __future__ import annotations

import json
import sys
from pathlib import Path, PureWindowsPath

import pytest

from cutpoint import cli, clips, mix
from cutpoint.clips import load_segments, segment_ids, segment_source


def _segs(*ids: object) -> list[dict[str, object]]:
    """Entries with the given ids; ``None`` leaves the entry without one."""
    return [{"id": i} if i is not None else {} for i in ids]


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        pytest.param((None, None, None), ["s001", "s002", "s003"], id="no-collision"),
        pytest.param(("s002", None), ["s002", "s002b"], id="generated-id-is-taken"),
        pytest.param(("s002", "s002b", None), ["s002", "s002b", "s003"], id="other-slot"),
        pytest.param(("s002", None, "s003", None), ["s002", "s002b", "s003", "s004"], id="chained"),
        pytest.param((None, "s001", "s001b"), ["s001c", "s001", "s001b"], id="next-letter"),
        pytest.param(("x", "s002b", "s002", None), ["x", "s002b", "s002", "s004"], id="plain"),
    ],
)
def test_segment_ids(given: tuple[object, ...], expected: list[object]) -> None:
    assert segment_ids(_segs(*given)) == expected


def test_duplicate_ids_name_both_segments_and_the_id() -> None:
    with pytest.raises(ValueError, match=r"Segments 1 and 3 have the same id 'a'"):
        segment_ids(_segs("a", "b", "a"))


@pytest.mark.parametrize("bad", [["a"], {"k": 1}], ids=["list", "dict"])
def test_unhashable_id_is_a_value_error(bad: object) -> None:
    with pytest.raises(ValueError, match="Segment 2: id must be"):
        segment_ids(_segs("a", bad))


def _write_segments(tmp_path: Path, entries: list[dict[str, object]]) -> Path:
    (tmp_path / "a.mp4").write_bytes(b"")
    full = [{"source": "a.mp4", "start": 1.0, "end": 2.0, **e} for e in entries]
    f = tmp_path / "segments.json"
    f.write_text(json.dumps(full), encoding="utf-8")
    return f


def test_load_segments_keeps_plain_ids_without_collision(tmp_path: Path) -> None:
    f = _write_segments(tmp_path, [{}, {"id": "intro"}, {}])
    assert [m["id"] for m in load_segments(f)] == ["s001", "intro", "s003"]


def test_load_segments_null_tail_is_unset(tmp_path: Path) -> None:
    f = _write_segments(tmp_path, [{"tail": None}, {"tail": 1.5}])
    first, second = load_segments(f)
    assert "tail" not in first
    assert second["tail"] == 1.5


def test_load_segments_rejects_a_nan_bound(tmp_path: Path) -> None:
    f = _write_segments(tmp_path, [{"max_end": float("nan")}])  # json.dumps writes NaN
    with pytest.raises(ValueError, match=r"Segment 1: max_end must be a number of seconds"):
        load_segments(f)


@pytest.mark.parametrize("kind", ["relative", "projects", "absolute"])
def test_missing_segments_file_is_one_error_line(
    kind: str,
    tmp_path: Path,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Issue #127: the error says what a relative path was relative to; no traceback."""
    monkeypatch.chdir(tmp_path)
    (home / "projects" / "proj").mkdir(parents=True)
    given, tail = {
        "relative": ("nope.json", f"(relative to the current folder {Path.cwd()})"),
        "projects": (
            "projects/proj/x.json",
            f"(projects/proj/x.json, relative to the projects folder {home / 'projects'})",
        ),
        "absolute": (str(tmp_path / "nowhere" / "abs.json"), ""),
    }[kind]
    assert cli.main(["clips", "proj", "--segments", given]) == 1
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1
    msg = errors[0].getMessage()
    assert msg.startswith("Segments file not found: ")
    if tail:
        assert msg.endswith(tail)
    else:
        assert msg.endswith(given)
    assert errors[0].exc_info is None
    assert "Traceback" not in capsys.readouterr().err + caplog.text


@pytest.mark.parametrize("command", ["clips", "repair"])
def test_segments_folder_is_one_error_line(
    tmp_path: Path,
    home: Path,
    caplog: pytest.LogCaptureFixture,
    command: str,
) -> None:
    """Issue #150: ``--segments`` given a folder says so instead of "not found"."""
    (home / "projects" / "proj").mkdir(parents=True)
    folder = tmp_path / "segs"
    folder.mkdir()
    if command == "clips":
        argv = ["clips", "proj", "--segments", str(folder)]
    else:
        argv = ["360", "repair", str(tmp_path), "--segments", str(folder), "--qa", "qa.json"]
    assert cli.main(argv) == 1
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1
    assert "is a folder, not a segments file" in errors[0].getMessage()


@pytest.mark.parametrize("case", ["qa-folder", "qa-utf16", "qa-broken-json", "qa-not-object"])
def test_repair_input_errors_name_the_file_and_the_problem(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, case: str
) -> None:
    """360 repair input errors name the file and the problem (follow-up to #150/#156)."""
    segs = tmp_path / "segs.json"
    segs.write_text(json.dumps([{"source": "a.insv", "start": 1, "end": 2}]), encoding="utf-8")
    qa = tmp_path / "qa.json"
    qa.write_text("{", encoding="utf-8")
    target, phrase = qa, "is not valid JSON: "
    if case == "qa-folder":
        qa.unlink()
        qa.mkdir()
        phrase = "is a folder, not a QA report"
    elif case == "qa-utf16":
        qa.write_text("{}", encoding="utf-16")
        phrase = "not UTF-8 text (save it as UTF-8)"
    elif case == "qa-not-object":
        qa.write_text("[]", encoding="utf-8")
        phrase = ": not a QA report"
    argv = ["360", "repair", str(tmp_path), "--segments", str(segs), "--qa", str(qa)]
    assert cli.main(argv) == 1
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1
    assert str(target) in errors[0].getMessage()
    assert phrase in errors[0].getMessage()


def test_segments_file_with_a_bom_loads_and_utf16_is_not_valid_json(
    tmp_path: Path, home: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Issue #150: PowerShell 5.1 writes a BOM (loads) or UTF-16 (one clear error line)."""
    f = _write_segments(tmp_path, [{}])
    text = f.read_text(encoding="utf-8")
    f.write_text(text, encoding="utf-8-sig")
    assert load_segments(f)[0]["start"] == 1.0
    f.write_text(text, encoding="utf-16")
    (home / "projects" / "proj").mkdir(parents=True)
    assert cli.main(["clips", "proj", "--segments", str(f)]) == 1
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1
    assert f"{f} is not valid JSON" in errors[0].getMessage()
    assert "not UTF-8 text" in errors[0].getMessage()


@pytest.mark.parametrize("source", ["D:\\Videos\\a.insv", "D:/Videos/a.insv"])
def test_windows_source_keeps_its_drive_and_is_not_joined_to_the_folder(
    tmp_path: Path, source: str
) -> None:
    """Review F3: on any OS a drive-letter source is absolute; its identity is the Windows
    form a project written on Windows records."""
    identity, io_path = segment_source(source, tmp_path)
    assert identity == "D:\\Videos\\a.insv"
    assert str(tmp_path) not in str(io_path)
    assert str(io_path).replace("\\", "/") == "D:/Videos/a.insv"


@pytest.mark.parametrize("given", ["/nope.json", "\\nope.json", "C:nope.json", "C:/nope.json"])
def test_missing_segments_file_rooted_or_drive_path_has_no_suffix(given: str) -> None:
    """Follow-up to #127: Windows ``\\nope.json`` is relative to the current drive's root."""
    win = PureWindowsPath(given)
    assert cli._segments_where(given, win, Path("C:/nope.json")) == ""


@pytest.mark.parametrize(
    "on_windows",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="D:foo is drive-relative on Windows"
            ),
        ),
    ],
    ids=["windows", "elsewhere"],
)
def test_load_segments_refuses_a_drive_relative_source_only_on_windows(
    tmp_path: Path, windows, on_windows: bool
) -> None:
    """Issue #175: ``D:foo.mp4`` is relative to drive D's current folder on Windows (refused,
    as ``360 repair`` does); on other systems it is a file name next to the segments file."""
    windows(on_windows)
    f = tmp_path / "segs.json"
    f.write_text(json.dumps([{"source": "D:foo.mp4", "start": 1, "end": 3}]), encoding="utf-8")
    if on_windows:
        with pytest.raises(ValueError, match=r"Segment 1: source 'D:foo.mp4' is relative to the "):
            load_segments(f)
    else:
        (tmp_path / "D:foo.mp4").write_bytes(b"")  # POSIX only: a plain file name
        assert load_segments(f)[0]["source"].endswith("D:foo.mp4")


def _mapped_drive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remap: bool
) -> tuple[Path, Path]:
    """``old`` stands for a disconnected mapped drive ``Z:\\`` and ``unc`` for its
    ``\\\\server\\share`` form (``clips.unc_path`` stubbed, it is a no-op off Windows);
    with ``remap`` a copy of the footage under ``new`` is reachable through ``source_remap``."""

    base = tmp_path.resolve()
    old, unc, new = base / "Z", base / "server_share", base / "copy"
    if remap:
        (new / "F").mkdir(parents=True)
        (new / "F" / "a.mp4").write_bytes(b"")
        (base / "home").mkdir(exist_ok=True)
        (base / "home" / "config.json").write_text(
            json.dumps({"source_remap": {str(old): str(new)}}), encoding="utf-8"
        )

    def fake_unc(path: str) -> str:
        return str(unc) + path[len(str(old)) :] if path.startswith(str(old)) else path

    monkeypatch.setattr(clips, "unc_path", fake_unc, raising=False)
    return old, unc


def test_segment_source_records_a_remapped_mapped_drive_source_in_unc_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #181: with ``source_remap`` applying while the mapped drive is disconnected,
    the identity is the UNC form ``analyze`` records, not the drive letter; the read path
    is the copy."""
    old, unc = _mapped_drive(tmp_path, monkeypatch, remap=True)
    identity, io_path = segment_source(old / "F" / "a.mp4", tmp_path)
    assert identity == str(unc / "F" / "a.mp4")
    assert io_path == tmp_path.resolve() / "copy" / "F" / "a.mp4"


def test_segment_source_without_a_remap_keeps_the_resolved_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #181: no remap -> unchanged (the resolved path; UNC form is resolve()'s job)."""
    old, _ = _mapped_drive(tmp_path, monkeypatch, remap=False)
    assert segment_source(old / "a.mp4", tmp_path)[0] == str(old / "a.mp4")


@pytest.mark.parametrize("remap", [True, False])
def test_apply_exclusions_matches_a_drive_letter_exclusion_to_a_unc_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remap: bool
) -> None:
    """Issue #181: an exclusion recorded as ``Z:\\F\\a.mp4`` (drive disconnected, with or
    without a ``source_remap`` copy) cuts a candidate recorded as
    ``\\\\server\\share\\F\\a.mp4``."""
    old, unc = _mapped_drive(tmp_path, monkeypatch, remap)
    cand = mix.Cand("run", str(unc / "F" / "a.mp4"), 10.0, 20.0, 15.0, 0.9, "run", "run")
    exclusion = [{"source": str(old / "F" / "a.mp4"), "start": 0, "end": 100}]
    assert mix.apply_exclusions([cand], exclusion) == []
