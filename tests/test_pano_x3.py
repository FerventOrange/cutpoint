"""Insta360 X3 file naming: grouping camera files into clips by name only."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutpoint import insta360
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


def test_parse_name_normalises_kind_and_extension() -> None:
    assert insta360.parse_name("vid_20260101_100000_10_007.INSV") == {
        "kind": "VID", "date": "20260101", "time": "100000", "lens": "10", "num": "007",
        "ext": "insv",
    }  # fmt: skip
    assert insta360.parse_name("VID_20260101_100000_00_001.mp4") is None


def test_imu_source_finds_the_00_file_of_any_file_of_the_clip(tmp_path: Path) -> None:
    for name in ("VID_20260101_100000_00_001.insv", "LRV_20260101_100000_11_001.lrv"):
        (tmp_path / name).write_bytes(b"x")
    assert insta360.imu_source(tmp_path / "LRV_20260101_100000_11_001.lrv") == (
        tmp_path / "VID_20260101_100000_00_001.insv"
    )
    assert insta360.imu_source(tmp_path / "LRV_20260101_120000_11_002.lrv") is None  # no _00_
    assert insta360.imu_source(tmp_path / "clip.mp4") is None


@pytest.fixture
def camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A card with a 5.7K clip, a 4K clip and a proxy without its lens file; ffprobe and
    the trailer reader replaced. Returns the folder and the paths probed / read."""
    folder = tmp_path / "card"
    folder.mkdir()
    sizes = {
        "VID_20260101_100000_00_001.insv": 300, "VID_20260101_100000_10_001.insv": 200,
        "LRV_20260101_100000_11_001.lrv": 10, "VID_20260101_110000_00_002.insv": 400,
        "LRV_20260101_110000_11_002.lrv": 10, "LRV_20260101_120000_11_003.lrv": 10,
    }  # fmt: skip
    for name, size in sizes.items():
        (folder / name).write_bytes(b"x" * size)
    probed: list[str] = []
    read: list[str] = []

    def probe(path: Path) -> dict:
        probed.append(Path(path).name)
        if Path(path).suffix == ".lrv":
            return {"width": 736, "height": 368, "nb_frames": "300", "avg_frame_rate": "30000/1001",
                    "duration": "10.01"}  # fmt: skip
        return {"width": 2880, "height": 2880, "nb_frames": "301", "avg_frame_rate": "30000/1001"}

    def trailer(insv00: Path) -> dict:
        read.append(insv00.name)
        return {"camera": "Insta360 X3"}

    monkeypatch.setattr(x3, "probe_video_stream", probe)
    monkeypatch.setattr(x3, "read_trailer_meta", trailer)
    return folder, probed, read


def test_scan_pairs_and_probes_each_clip_and_caches_the_results(home, camera, caplog) -> None:
    folder, probed, read = camera

    clips = x3.scan(folder)

    assert [c.clip_id for c in clips] == ["001", "002"]  # 003 has no _00_.insv
    assert "clip 003: no _00_.insv" in caplog.text
    first, second = clips
    assert first.mode == "5.7k" and second.mode == "4k"
    assert [Path(p).name for p in first.insv] == [
        "VID_20260101_100000_00_001.insv", "VID_20260101_100000_10_001.insv",
    ]  # fmt: skip
    assert first.source == str(folder / "LRV_20260101_100000_11_001.lrv")
    assert first.start_local == "2026-01-01T10:00:00"
    assert (first.width, first.height) == (2880, 2880)  # the lens file's size
    assert (first.frames, first.duration) == (300, 10.01)  # the proxy's timing
    assert first.fps == pytest.approx(30000 / 1001)
    assert first.bytes_insv == 500
    assert first.meta == {"camera": "Insta360 X3"}

    probed.clear()
    read.clear()
    again = x3.scan(folder)
    assert probed == [] and read == []  # both cached per clip
    assert [c.to_json() for c in again] == [c.to_json() for c in clips]


def test_scan_does_not_cache_a_failed_probe(home, camera, monkeypatch) -> None:
    folder, probed, _ = camera
    monkeypatch.setattr(x3, "probe_video_stream", lambda path: probed.append(path) or {})
    [clip, _] = x3.scan(folder, with_meta=False)
    assert (clip.width, clip.frames, clip.fps) == (0, 0, pytest.approx(30000 / 1001))
    probed.clear()
    x3.scan(folder, with_meta=False)
    assert probed  # retried on the next run


def test_scan_records_an_unreadable_trailer(home, camera, monkeypatch, caplog) -> None:
    folder, _, _ = camera

    def broken(insv00: Path) -> dict:
        raise Exception("bad trailer")

    monkeypatch.setattr(x3, "read_trailer_meta", broken)
    clips = x3.scan(folder)
    assert clips[0].meta == {"error": "bad trailer"}
    assert "trailer unreadable (bad trailer)" in caplog.text


def test_scan_refuses_a_missing_or_foreign_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Not a folder"):
        x3.scan(tmp_path / "missing")
    (tmp_path / "clip.mp4").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="No Insta360"):
        x3.scan(tmp_path)


def test_read_trailer_meta_reads_the_two_lens_calibration(monkeypatch) -> None:
    offset = [2, 1450.0, 2960.0, 2970.0, 0.1, 0.2, 0.3, 1451.0, 8930.0, 2975.0, 180.1, 0.4, 0.5,
              11904, 5952]  # fmt: skip
    metadata = {"offset": offset, "serial_number": "IAQEB1234", "fw_version": "v1.0.40"}

    class Parser:
        camera, model = "Insta360", "X3"

        def __init__(self, path: str) -> None:
            self.path = path

        def telemetry(self) -> list[dict]:
            return [{"Default": {"Metadata": metadata}}]

    monkeypatch.setitem(sys.modules, "telemetry_parser", SimpleNamespace(Parser=Parser))

    meta = x3.read_trailer_meta(Path("VID_20260101_100000_00_001.insv"))

    assert meta["camera"] == "Insta360 X3"
    assert (meta["serial"], meta["firmware"]) == ("IAQEB1234", "v1.0.40")
    lenses = meta["calibration"]["lenses"]
    assert lenses[0] == {"radius": 1450.0, "cx": 2960.0, "cy": 2970.0, "yaw": 0.1,
                         "pitch": 0.2, "roll": 0.3}  # fmt: skip
    assert lenses[1]["yaw"] == 180.1
    assert meta["calibration"]["sensor"] == [11904, 5952]


def test_probe_video_stream_is_empty_when_ffprobe_fails(monkeypatch) -> None:
    from cutpoint import errors, ffmpeg_utils  # noqa: PLC0415

    def fail(*a, **k):
        raise errors.FFmpegError("ffprobe failed")

    monkeypatch.setattr(ffmpeg_utils, "ffprobe_json", fail)
    assert x3.probe_video_stream(Path("broken.insv")) == {}
