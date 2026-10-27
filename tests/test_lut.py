"""``--lut`` / ``--lut-strength`` (#290 stage 1): the filter string, validation, the manifest,
``extend`` and real grades with generated LUTs."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pytest

from conftest import media_info, needs_ffmpeg, run_json, write
from cutpoint import cli, encode, extend, ffmpeg_utils, jsonout, planning
from cutpoint.boundaries import SpeechSettings
from cutpoint.clips import ClipPlan, RenderResult, _entry
from cutpoint.encode import Lut
from cutpoint.ffmpeg_utils import VideoEncoder
from cutpoint.planning import Padding
from cutpoint.tail import TailResult

TO_RGB = "scale=in_color_matrix=bt709:in_range=tv:out_range=pc"
TO_YUV = "scale=out_color_matrix=bt709:out_range=tv"
FMT = {"width": 1920, "height": 1080, "fps": 30, "vertical": False}


def cube(path: Path, fn, n: int = 9) -> Path:
    """A .cube LUT of ``n`` points per axis mapping RGB (0-1) through ``fn``."""
    rows = [f"LUT_3D_SIZE {n}"]
    for b in range(n):  # red changes fastest
        for g in range(n):
            for r in range(n):
                rows.append(" ".join(f"{v:.6f}" for v in fn(r / (n - 1), g / (n - 1), b / (n - 1))))
    return write(path, "\n".join(rows) + "\n")


def identity(r, g, b):
    return r, g, b


def swap(r, g, b):  # red and blue exchanged
    return b, g, r


# -- the filter string ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "escaped"),
    [
        (PureWindowsPath(r"C:\My LUTs\flat.cube"), r"C\\:/My LUTs/flat.cube"),
        (PureWindowsPath(r"D:\it's [v2]; a,b.cube"), r"D\\:/it\\\'s \[v2\]\; a\,b.cube"),
        (PurePosixPath("/luts/back\\slash.cube"), r"/luts/back\\\\slash.cube"),
    ],
    ids=["drive-and-space", "quote-brackets-separators", "posix-backslash"],
)
def test_filter_path_escapes_for_an_option_inside_a_filtergraph(path, escaped):
    assert encode.filter_path(path) == escaped


def test_full_strength_grades_in_bt709_rgb_without_a_blend():
    vf = encode.lut_filter(PureWindowsPath(r"C:\L\a.cube"))
    assert vf == f"{TO_RGB},lut3d=file=C\\\\:/L/a.cube:interp=tetrahedral,{TO_YUV}"


def test_partial_strength_blends_the_grade_over_the_original():
    vf = encode.lut_filter(PurePosixPath("/l/a.cube"), 0.35)
    assert vf == (
        f"split=2[lut_o][lut_i];[lut_i]{TO_RGB},lut3d=file=/l/a.cube:interp=tetrahedral,"
        f"{TO_YUV}[lut_g];[lut_g][lut_o]blend=all_mode=normal:all_opacity=0.35"
    )


def test_zero_strength_is_no_grade():
    assert encode.lut_filter(PurePosixPath("/l/a.cube"), 0.0) is None
    assert Lut(Path("/l/a.cube"), 0.0).filter() == ""


def _export_args(monkeypatch, info, lut: Lut | None = None, **kw) -> list[str]:
    calls: list[list[str]] = []
    monkeypatch.setattr(encode, "run", lambda args, **_kw: calls.append(args) or "")
    fmt = encode.OutputFormat(1920, 1080, 30)
    extra = {"lut": lut} if lut is not None or kw.get("pass_none") else {}
    encode._encode(VideoEncoder("cpu", None), info.path, info, 1.0, 5.0, Path("o.mp4"), fmt,
                   False, None, 0.0, **extra)  # fmt: skip
    return calls[0]


def _graph(args: list[str]) -> str:
    return args[args.index("-filter_complex") + 1]


def test_without_lut_the_export_command_is_unchanged(monkeypatch):
    for info in (media_info(color_space="bt709"), media_info(), media_info(color_space="bt470bg")):
        assert _export_args(monkeypatch, info, pass_none=True) == _export_args(monkeypatch, info)
        assert "lut3d" not in _graph(_export_args(monkeypatch, info))


@pytest.mark.parametrize(
    ("tags", "before"),
    [
        ({"color_space": "bt709"}, "fps=30,"),
        ({"color_space": "smpte170m"}, "range=tv:fast=1,"),
        ({}, encode.TAG_BT709 + ","),
    ],
    ids=["bt709", "bt601", "untagged-hd"],
)
def test_the_lut_follows_the_bt709_conversion_and_precedes_the_8_bit_format(
    monkeypatch, tags, before
):
    lut = Lut(Path("/l/a.cube"), 1.0)
    plain = _export_args(monkeypatch, media_info(**tags))
    graded = _export_args(monkeypatch, media_info(**tags), lut)
    assert f"{before}{lut.filter()},format=yuv420p[v];" in _graph(graded)
    assert _graph(graded) == _graph(plain).replace(
        "format=yuv420p[v]", f"{lut.filter()},format=yuv420p[v]"
    )
    # The output tags are those of the ungraded clip: the LUT does not change the matrix.
    assert [a for a in graded if a != _graph(graded)] == [a for a in plain if a != _graph(plain)]


def test_a_partial_grade_with_a_gpu_upload_stays_one_chain_to_v(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(encode, "run", lambda args, **_kw: calls.append(args) or "")
    fmt = encode.OutputFormat(1920, 1080, 30)
    lut = Lut(Path("/l/a.cube"), 0.5)
    info = media_info(color_space="bt709")
    encode._encode(VideoEncoder("vaapi", "/dev/dri/renderD128"), info.path, info, 1.0, 5.0,
                   Path("o.mp4"), fmt, False, None, 0.0, lut=lut)  # fmt: skip
    graph = _graph(calls[0])
    assert (
        graph.startswith("[0:v:0]scale=") and "all_opacity=0.5,format=yuv420p,format=nv12" in graph
    )
    assert graph.split(";[", 3)[2].endswith("hwupload[v]")


# -- validation ----------------------------------------------------------------------------------


@pytest.fixture
def loads(monkeypatch) -> list[list[str]]:
    """Stub the ffmpeg load check of check_lut; returns its command lines."""
    calls: list[list[str]] = []
    monkeypatch.setattr(encode, "run", lambda args, **_kw: calls.append(args) or "")
    return calls


def test_check_lut_returns_the_absolute_path_and_tries_loading_it(tmp_path, loads, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "Look.CUBE")
    lut = encode.check_lut("Look.CUBE", 0.25)
    assert lut == Lut(tmp_path / "Look.CUBE", 0.25)
    assert f"lut3d=file={encode.filter_path(tmp_path / 'Look.CUBE')}" in loads[0][6]
    assert encode.check_lut(write(tmp_path / "x.3dl")).path.suffix == ".3dl"


@pytest.mark.parametrize(
    ("name", "strength", "error", "message"),
    [
        ("missing.cube", 1.0, FileNotFoundError, "--lut file not found: "),
        ("look.png", 1.0, jsonout.ArgumentError, "not a LUT file (use .cube or .3dl)"),
        ("look.cube", 1.5, jsonout.ArgumentError, "--lut strength 1.5: must be from 0 to 1"),
        ("folder.cube", 1.0, jsonout.ArgumentError, "cannot be read"),
    ],
)
def test_check_lut_refuses_before_any_work(tmp_path, loads, name, strength, error, message):
    write(tmp_path / "look.png")
    write(tmp_path / "look.cube")
    (tmp_path / "folder.cube").mkdir()
    with pytest.raises(error, match=message.replace("(", r"\(").replace(")", r"\)")):
        encode.check_lut(tmp_path / name, strength)
    assert loads == []


def test_check_lut_reports_a_lut_ffmpeg_cannot_load(tmp_path, monkeypatch):
    def fail(args, **_kw):
        raise ffmpeg_utils.FFmpegError("Invalid data found")

    monkeypatch.setattr(encode, "run", fail)
    with pytest.raises(jsonout.ArgumentError, match="ffmpeg cannot load it"):
        encode.check_lut(write(tmp_path / "bad.cube"))


@pytest.mark.parametrize("command", [["clips", "p"], ["edit", "nosuch.mp4"]])
def test_cli_missing_lut_is_not_found_before_any_work(tmp_path, capsys, command):
    code, doc = run_json(capsys, [*command, "--lut", str(tmp_path / "gone.cube")])
    assert code == 1 and doc["error"]["code"] == "not_found"
    assert doc["error"]["message"] == f"--lut file not found: {tmp_path / 'gone.cube'}"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--lut", "look.txt"], "not a LUT file"),
        (["--lut-strength", "0.5"], "--lut-strength needs --lut"),
    ],
)
def test_cli_bad_lut_options_are_bad_args(tmp_path, capsys, monkeypatch, extra, message):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "look.txt")
    code, doc = run_json(capsys, ["clips", "p", *extra])
    assert code == 1 and doc["error"]["code"] == "bad_args"
    assert message in doc["error"]["message"]


@pytest.mark.parametrize("command", ["clips", "edit"])
def test_cli_lut_strength_out_of_range_exits_2(capsys, command):
    with pytest.raises(SystemExit) as exc:
        cli.main([command, "p", "--lut", "a.cube", "--lut-strength", "1.5", "--json"])
    assert exc.value.code == 2
    out, _err = capsys.readouterr()
    assert json.loads(out)["error"]["code"] == "bad_args"


def test_cli_passes_the_checked_lut_to_clips_and_edit(tmp_path, monkeypatch, loads):
    got = {}
    monkeypatch.setattr(cli, "export_clips", lambda project, **kw: got.update(clips=kw["lut"]))
    monkeypatch.setattr(cli, "run_edit", lambda *a, **kw: got.update(edit=kw["lut"]))
    lut = write(tmp_path / "look.cube")
    assert cli.main(["clips", "p", "--lut", str(lut), "--lut-strength", "0.4"]) == 0
    assert cli.main(["edit", "v.mp4", "--lut", str(lut)]) == 0
    assert got == {"clips": Lut(lut, 0.4), "edit": Lut(lut, 1.0)}
    assert cli.main(["clips", "p"]) == 0
    assert got["clips"] is None


# -- manifest and extend -------------------------------------------------------------------------


def test_manifest_entry_records_the_grade_and_reads_it_back(tmp_path):
    m = {"id": "h001", "source": str(tmp_path / "a.mp4"), "end": 5.0, "score": 50}
    tail = TailResult(5.0, 0.0, "none")
    plan = ClipPlan("001_a.mp4", m, tmp_path / "a.mp4", media_info(), 1.0, 5.0, (1.0, 5.0), tail)
    lut = Lut(tmp_path / "look.cube", 0.5)
    entry = json.loads(
        json.dumps(_entry(plan, RenderResult(tmp_path / "001_a.mp4", 5.0), 0.25, lut=lut))
    )
    assert (entry["lut"], entry["lut_strength"]) == (str(tmp_path / "look.cube"), 0.5)
    assert Lut.from_entry(entry) == lut
    plain = _entry(plan, RenderResult(tmp_path / "001_a.mp4", 5.0), 0.25)
    assert "lut" not in plain and Lut.from_entry(plain) is None


@pytest.fixture
def exports(monkeypatch) -> list[tuple]:
    """Stub probe and the encoder of extend (no ffmpeg); returns the export arguments."""
    calls: list[tuple] = []

    def fake_export(src, info, start, end, out, *args):
        calls.append(args)
        write(out, "encoded")

    monkeypatch.setattr(extend, "probe", media_info)
    monkeypatch.setattr(extend, "export_clip_checked", fake_export)
    monkeypatch.setattr(planning.sources, "readable_range", lambda src, t, d: (0.0, d))
    return calls


def _graded_clips(tmp_path: Path, lut: Path | None, **grade) -> Path:
    clips = tmp_path / "proj" / "clips"
    write(clips / "001_a.mp4")
    e = {"file": "001_a.mp4", "source": str(write(tmp_path / "a.mp4")), "start": 10.0, "end": 20.0}
    e |= ({"lut": str(lut)} if lut else {}) | grade
    write(clips / "manifest.json", json.dumps({"format": FMT, "clips": [e]}))
    return clips


def test_extend_reproduces_the_recorded_grade(tmp_path, exports, loads):
    lut = write(tmp_path / "look.cube")
    clips = _graded_clips(tmp_path, lut, lut_strength=0.6)
    out = extend.extend_clips(clips, Padding(1.0, 1.0), speech=SpeechSettings(enabled=False))
    assert exports[0][-1] == Lut(lut, 0.6)
    entry = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["clips"][0]
    assert (entry["lut"], entry["lut_strength"]) == (str(lut), 0.6)


def test_extend_of_ungraded_clips_passes_no_grade(tmp_path, exports, loads):
    extend.extend_clips(
        _graded_clips(tmp_path, None), Padding(1.0, 1.0), speech=SpeechSettings(enabled=False)
    )
    assert exports[0][-1] is None and loads == []


def test_extend_stops_before_any_export_when_the_recorded_lut_is_gone(tmp_path, exports, loads):
    clips = _graded_clips(tmp_path, tmp_path / "moved.cube")
    with pytest.raises(FileNotFoundError, match=r"001_a\.mp4: lut file not found"):
        extend.extend_clips(clips, Padding(1.0, 1.0), speech=SpeechSettings(enabled=False))
    assert exports == []


def test_extend_reuse_needs_the_same_grade():
    old = {"source": "a.mp4", "start": 1.0, "end": 2.0, "lut": "/l/a.cube", "lut_strength": 1.0}
    assert extend._reusable(old, dict(old), True, 0.25)
    assert not extend._reusable(old, old | {"lut_strength": 0.5}, True, 0.25)
    assert not extend._reusable({k: v for k, v in old.items() if k[:3] != "lut"}, old, True, 0.25)


# -- real grades ---------------------------------------------------------------------------------


@pytest.fixture
def lut_dir(tmp_path: Path) -> Path:
    """A folder whose path has a space, a quote and brackets (and a drive letter on Windows)."""
    d = tmp_path / "my luts [it's]"
    d.mkdir()
    return d


def _source(path: Path) -> Path:
    """2 s of colour bars, 320x180 at 30 fps, BT.709 tv range, lossless (FFV1) with audio."""
    tags = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    ffmpeg_utils.run(
        [
            *("ffmpeg", "-y", "-v", "error", "-f", "lavfi"),
            *("-i", "smptehdbars=s=320x180:d=2:r=30,format=rgb24"),
            *("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2"),
            *("-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"),
            *("-c:v", "ffv1", "-color_range", "tv", *tags, "-c:a", "aac", "-shortest", str(path)),
        ]
    )  # fmt: skip
    return path


def _export(src: Path, out: Path, lut: Lut | None) -> np.ndarray:
    """Export 0.5-1.5 s of ``src`` (graded with ``lut``) and return its mean RGB frame."""
    info = ffmpeg_utils.probe(src)
    encode.export_clip(
        src, info, 0.5, 1.5, out, encode.choose_format([info], False), False, lut=lut
    )
    vf = (
        "scale=flags=accurate_rnd+full_chroma_int:in_color_matrix=bt709:in_range=tv:"
        "out_range=pc,format=rgb24"
    )
    frames = ffmpeg_utils.decode_raw(out, size=(320, 180), vf=vf, start=0.5, hwaccel=None)
    return next(iter(frames))[1].astype(np.float32)


@pytest.fixture
def graded(tmp_path, monkeypatch):
    """``graded(lut)``: mean RGB frame of the test source exported with ``lut``."""
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    src = _source(tmp_path / "bars.mkv")
    n = iter(range(100))
    return lambda lut: _export(src, tmp_path / f"out{next(n)}.mp4", lut)


@needs_ffmpeg
def test_real_identity_lut_leaves_the_picture_as_it_was(graded, lut_dir):
    lut = encode.check_lut(cube(lut_dir / "identity.cube", identity))
    plain, same = graded(None), graded(lut)
    assert np.abs(same - plain).mean() < 1.5


@needs_ffmpeg
def test_real_swap_lut_exchanges_red_and_blue_and_half_strength_is_midway(graded, lut_dir):
    path = cube(lut_dir / "swap.cube", swap)
    plain, full, half = graded(None), graded(Lut(path, 1.0)), graded(Lut(path, 0.5))
    assert np.abs(plain[..., 0] - plain[..., 2]).mean() > 20  # bars: red and blue differ
    assert np.abs(full[..., 0] - plain[..., 2]).mean() < 3
    assert np.abs(full[..., 2] - plain[..., 0]).mean() < 3
    assert np.abs(full[..., 1] - plain[..., 1]).mean() < 3
    assert np.abs(full - plain).mean() > 10
    assert np.abs(half - (plain + full) / 2).mean() < 2.5


@needs_ffmpeg
def test_real_3dl_lut_loads(graded, lut_dir):
    """An Autodesk .3dl: a header of 17 input levels, then 12-bit RGB rows, blue fastest."""
    levels = [round(i * 1023 / 16) for i in range(17)]
    rows = [" ".join(map(str, levels))]
    for r in range(17):
        for g in range(17):
            for b in range(17):
                rows.append(" ".join(str(round(v * 4095 / 16)) for v in (r, g, b)))
    lut = encode.check_lut(write(lut_dir / "identity.3dl", "\n".join(rows) + "\n"))
    assert np.abs(graded(lut) - graded(None)).mean() < 1.5


@needs_ffmpeg
def test_real_malformed_lut_is_refused_up_front(lut_dir):
    with pytest.raises(jsonout.ArgumentError, match="ffmpeg cannot load it"):
        encode.check_lut(write(lut_dir / "bad.cube", "LUT_3D_SIZE 4\n0 0 0\n"))


@needs_ffmpeg
def test_real_graded_clip_is_tagged_bt709_so_assemble_does_not_convert_it(
    tmp_path, graded, lut_dir
):
    graded(Lut(cube(lut_dir / "swap.cube", swap), 0.5))
    info = ffmpeg_utils.probe(tmp_path / "out0.mp4")
    assert info.color_space == "bt709"  # an SD clip: untagged, assemble would read it as BT.601
    assert encode.bt709_filter(info) == ""


@needs_ffmpeg
def test_real_clips_record_the_grade_and_extend_redoes_it(
    video, home, tmp_path, lut_dir, capsys, monkeypatch
):
    lut = cube(lut_dir / "swap.cube", swap)
    seg = [{"source": str(video), "start": 1.0, "end": 3.0, "label": "a"}]
    segments = write(tmp_path / "seg.json", json.dumps(seg))
    argv = [
        "clips",
        "lutp",
        "--segments",
        str(segments),
        "--lut",
        str(lut),
        "--lut-strength",
        "0.5",
    ]
    code, doc = run_json(capsys, argv)
    assert code == 0, doc["error"]
    clips_dir = home / "projects" / "lutp" / "clips"
    entry = json.loads((clips_dir / "manifest.json").read_text(encoding="utf-8"))["clips"][0]
    assert (entry["lut"], entry["lut_strength"]) == (str(lut), 0.5)

    seen = []

    def spy(*args):
        seen.append(args[-1])
        return encode.export_clip_checked(*args)

    monkeypatch.setattr(extend, "export_clip_checked", spy)
    out = extend.extend_clips(clips_dir, Padding(0.5, 0.5), speech=SpeechSettings(enabled=False))
    assert seen == [Lut(lut, 0.5)]
    entry = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["clips"][0]
    assert (entry["lut"], entry["lut_strength"]) == (str(lut), 0.5)
    assert (out / "001_a.mp4").stat().st_size > 0
