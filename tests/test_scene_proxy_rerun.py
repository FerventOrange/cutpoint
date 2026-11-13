"""#1031: a GPU scene proxy whose frames stayed on the GPU (``hwdownload``) and that ffmpeg
reported dropped frames for is made again without keeping its frames on the GPU. ffmpeg is faked."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from conftest import media_info
from cutpoint import analyze, ffmpeg_utils
from cutpoint.errors import FFmpegError
from cutpoint.ffmpeg_utils import ProgressResult

CUDA = ffmpeg_utils.HwDecoder("cuda")


@pytest.fixture(autouse=True)
def plain(monkeypatch):
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, 60.0))
    monkeypatch.setattr(analyze, "decoder_for", lambda src: CUDA)


class Calls(list):
    """The recorded commands; ``results`` is what each run reports (an int, or an error)."""

    def __init__(self):
        super().__init__()
        self.results: list = []


@pytest.fixture
def runs(monkeypatch):
    """Fake run_progress: records each command; ``dropped`` is the list of per-run results
    (default: nothing dropped); a FFmpegError in it is raised."""
    calls = Calls()
    results = calls.results

    def run_progress(args, total, label, **kw):
        calls.append(args)
        res = results[len(calls) - 1] if len(calls) <= len(results) else 0
        if isinstance(res, Exception):
            raise res
        return ProgressResult(stderr="", out_time=total, dropped=res)

    monkeypatch.setattr(analyze, "run_progress", run_progress)
    return calls


def make(tmp_path, name="a.mp4"):
    proxy = tmp_path / "p.mp4"
    proxy.write_bytes(b"first")
    analyze._make_proxy(tmp_path / name, proxy, analyze._proxy_args(60.0), 60.0)
    return proxy


def has_download(cmd):
    return any("hwdownload" in a for a in cmd)


# ------------------------------------------------------------- run_progress parses drop_frames
@pytest.mark.parametrize(
    ("lines", "want"),
    [
        ("out_time_us=1000000\ndrop_frames=0\nout_time_us=2000000\ndrop_frames=7\nprogress=end\n", 7),
        ("out_time_us=1000000\nprogress=end\n", 0),
        ("drop_frames=N/A\nprogress=end\n", 0),
    ],
    ids=["reported", "absent", "garbage"],
)  # fmt: skip
def test_run_progress_reports_the_dropped_frames(monkeypatch, lines, want):
    script = "import sys; sys.stdout.write(sys.argv[1]); sys.stdout.flush()"
    argv = [sys.executable, "-c", script, lines]
    monkeypatch.setattr(ffmpeg_utils, "_command", lambda tool, args: argv)
    res = ffmpeg_utils.run_progress(["ffmpeg", "out.mp4"], 10.0, "t")
    assert res.dropped == want


# ------------------------------------------------------------------ the helper's decision
def test_dropped_frames_after_a_gpu_chain_run_it_again_without_keeping_frames_on_the_gpu(
    runs, caplog, tmp_path
):
    runs.results.append(12)
    with caplog.at_level("INFO", logger="cutpoint.analyze"):
        proxy = make(tmp_path)
    assert len(runs) == 2
    assert has_download(runs[0]) and not has_download(runs[1])
    assert "cuda" in " ".join(runs[1])  # still hardware decoding, only the download differs
    assert "hwaccel_output_format" not in " ".join(runs[1])
    infos = [
        r for r in caplog.records if r.levelname == "INFO" and "12 frames dropped" in r.message
    ]
    assert len(infos) == 1
    assert not proxy.exists()  # the first proxy was removed before the re-run (fake wrote none)


def test_the_first_proxy_is_removed_before_the_rerun(monkeypatch, tmp_path):
    proxy = tmp_path / "p.mp4"
    seen = []

    def run_progress(args, total, label, **kw):
        seen.append(proxy.exists())
        proxy.write_bytes(b"x")
        return ProgressResult("", total, 3 if len(seen) == 1 else 0)

    monkeypatch.setattr(analyze, "run_progress", run_progress)
    analyze._make_proxy(tmp_path / "a.mp4", proxy, analyze._proxy_args(60.0), 60.0)
    assert seen == [False, False]


def test_nothing_dropped_runs_once(runs, caplog, tmp_path):
    with caplog.at_level("INFO", logger="cutpoint.analyze"):
        make(tmp_path)
    assert len(runs) == 1 and has_download(runs[0])
    assert not [r for r in caplog.records if "dropped" in r.message]


def test_dropped_frames_without_a_gpu_download_chain_run_once(runs, tmp_path):
    runs.results.append(5)
    make(tmp_path, "a.ts")  # MPEG-TS: decoded without hwdownload
    assert len(runs) == 1 and not has_download(runs[0])


def test_dropped_frames_on_the_cpu_run_once(runs, monkeypatch, tmp_path):
    monkeypatch.setattr(analyze, "decoder_for", lambda src: None)
    runs.results.append(5)
    make(tmp_path)
    assert len(runs) == 1


def test_a_failing_rerun_still_reaches_the_cpu_fallback(runs, tmp_path):
    runs.results.extend([4, FFmpegError("ffmpeg failed during scene proxy (exit 1):\n\nboom")])
    make(tmp_path)
    assert len(runs) == 3
    assert has_download(runs[0]) and not has_download(runs[1])
    assert "cuda" not in " ".join(runs[2])  # the CPU command


def test_the_gpu_half_of_a_split_pass_runs_again_without_keeping_frames_on_the_gpu(
    monkeypatch, tmp_path
):
    calls = []

    def run_progress(args, total, label, **kw):
        calls.append(list(args))
        first_gpu = "-hwaccel" in args and any("hwdownload" in a for a in args)
        return ProgressResult(stderr="", out_time=total, dropped=5 if first_gpu else 0)

    monkeypatch.setattr(analyze, "run_progress", run_progress)
    monkeypatch.setattr(analyze, "_scan", lambda proxy, cancel: [])
    cdir = tmp_path / "cdir"
    cdir.mkdir()
    cuts = analyze._hybrid_cuts(tmp_path / "a.mp4", cdir, 60.0, CUDA, 30.0, 2)
    assert cuts == []
    gpu = [c for c in calls if "-hwaccel" in c]
    cpu = [c for c in calls if "-hwaccel" not in c]
    assert len(gpu) == 2 and len(cpu) == 1
    assert has_download(gpu[0]) and not has_download(gpu[1])
    assert "hwaccel_output_format" not in " ".join(gpu[1])
    assert not list(cdir.glob("proxy-*"))


# --------------------------------------------------------------------- _scene_decode download
def test_scene_decode_download_keeps_no_frame_on_the_gpu_for_cuda():
    src = Path("a.mp4")
    kept, kept_chain = analyze._scene_decode(CUDA, "fps=2,scale=320:-2", src)
    assert "hwdownload" in kept_chain and "cuda" in kept  # the default keeps them on the GPU
    args, chain = analyze._scene_decode(CUDA, "fps=2,scale=320:-2", src, download=True)
    assert "hwdownload" not in chain
    assert "hwaccel_output_format" not in args and "-hwaccel" in args
