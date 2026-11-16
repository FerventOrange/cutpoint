"""The persisted shapes and orders of the file stamps built from ``utils.file_stamp``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cutpoint import clipresume, runstate, ui
from cutpoint.events import hud_map
from cutpoint.pano import candidates, cli
from cutpoint.utils import file_stamp

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def existing(tmp_path: Path) -> Path:
    f = tmp_path / "a.json"
    f.write_bytes(b"12345")
    return f


def test_file_stamp_order(existing: Path) -> None:
    st = existing.stat()
    assert file_stamp(existing) == (st.st_mtime_ns, st.st_size, st.st_ino)


def test_clipresume_stamp(existing: Path) -> None:
    st = existing.stat()
    assert clipresume._stamp(existing) == [st.st_size, st.st_mtime_ns]


def test_pano_cli_stamp(existing: Path) -> None:
    st = existing.stat()
    assert cli._stamp(existing) == (st.st_ino, st.st_mtime_ns, st.st_size)


def test_pano_candidates_file_id(existing: Path) -> None:
    st = existing.stat()
    assert candidates._file_id(existing) == ["a.json", st.st_size, st.st_mtime_ns]


def test_ui_file_ident(existing: Path) -> None:
    st = existing.stat()
    assert ui._file_ident(existing) == ("a.json", st.st_size, st.st_mtime_ns)


def test_runstate_stamp(existing: Path) -> None:
    st = existing.stat()
    assert runstate._stamp(existing) == [st.st_mtime_ns, st.st_size, st.st_ino]


def test_hud_map_stamp(existing: Path) -> None:
    st = existing.stat()
    assert hud_map._stamp(existing) == [st.st_size, st.st_mtime_ns]
    assert hud_map._stamp(str(existing)) == [st.st_size, st.st_mtime_ns]


def test_directories_are_stamped(tmp_path: Path) -> None:
    assert clipresume._stamp(tmp_path) is not None
    assert cli._stamp(tmp_path) is not None
    assert runstate._stamp(tmp_path) is not None


@pytest.mark.parametrize(
    "fn",
    [
        clipresume._stamp,
        hud_map._stamp,
        cli._stamp,
        candidates._file_id,
        ui._file_ident,
        runstate._stamp,
        file_stamp,
    ],
)
def test_missing_file_is_none(tmp_path: Path, fn) -> None:
    assert fn(tmp_path / "missing.json") is None
