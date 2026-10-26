"""Insta360 X3 file naming: grouping camera files into clips by name only."""

from __future__ import annotations

from pathlib import Path

from cutpoint.pano import x3


def test_group_files_pairs_files_by_clip_number_and_ignores_other_names() -> None:
    names = [
        "VID_20260101_100000_00_001.insv",  # 5.7K clip: both lenses and the proxy
        "VID_20260101_100000_10_001.insv",
        "LRV_20260101_100000_11_001.lrv",
        "VID_20260101_110000_00_002.insv",  # 4K clip: one lens and the proxy
        "LRV_20260101_110000_11_002.lrv",
        "LRV_20260101_120000_11_003.lrv",  # a proxy without its _00_.insv
        "VID_20260101_130000_00_004.INSV",  # upper-case extension
        "notes.txt",
        "VID_20260101_100000_00_001.mp4",
        "IMG_0001.insv",  # not a camera name
    ]
    got = x3.group_files(Path(n) for n in names)
    assert sorted(got) == ["001", "002", "003", "004"]
    assert {n: sorted(k for k in g if k.startswith(("lens", "lrv"))) for n, g in got.items()} == {
        "001": ["lens00", "lens10", "lrv"],
        "002": ["lens00", "lrv"],
        "003": ["lrv"],
        "004": ["lens00"],
    }
    assert got["001"]["date"] == "20260101" and got["001"]["time"] == "100000"
    assert got["004"]["lens00"] == Path("VID_20260101_130000_00_004.INSV")
