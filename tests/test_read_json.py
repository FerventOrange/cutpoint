"""A corrupt highlights.json gives a clear ValueError, not a JSON traceback (#228)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cutpoint import analyze, cli
from cutpoint.events.common import load_highlights
from cutpoint.planning import select_moments

READERS = {
    "select_moments": select_moments,
    "reanalyze": analyze.reanalyze,
    "load_highlights": load_highlights,
}
HINT = "cutpoint analyze"


@pytest.mark.parametrize("reader", READERS)
def test_truncated_highlights_names_file_and_how_to_fix(tmp_path, reader):
    (tmp_path / "highlights.json").write_bytes(b'{"candidates": [')
    with pytest.raises(ValueError, match="is not valid JSON") as err:
        READERS[reader](tmp_path)
    assert str(tmp_path / "highlights.json") in str(err.value)
    assert HINT in str(err.value)
    assert "not found" not in str(err.value)


# reanalyze has its own non-object check (tests/test_analyze.py)
@pytest.mark.parametrize("reader", ["select_moments", "load_highlights"])
@pytest.mark.parametrize("content", ["null", "[]", "{}", '{"sources": null}'])
def test_non_object_highlights_is_a_clear_error(tmp_path, reader, content):
    (tmp_path / "highlights.json").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=r"has no (candidates|sources) list") as err:
        READERS[reader](tmp_path)
    assert str(tmp_path / "highlights.json") in str(err.value)
    assert HINT in str(err.value)
    assert "not found" not in str(err.value)


@pytest.mark.parametrize("content", ['{"sources": []}', '{"candidates": null}'])
def test_select_moments_without_a_candidates_list_is_a_clear_error(tmp_path, content):
    (tmp_path / "highlights.json").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="has no candidates list"):
        select_moments(tmp_path)


def test_cli_clips_reports_corrupt_highlights(tmp_path: Path, caplog):
    (tmp_path / "highlights.json").write_text("{", encoding="utf-8")
    with caplog.at_level(logging.ERROR):
        assert cli.main(["clips", str(tmp_path)]) == 1
    assert "is not valid JSON" in caplog.text
