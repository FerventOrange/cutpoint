"""NVDEC download chains of the raw-frame decoders (#870).

``scale_cuda`` without ``format=`` keeps a 10-bit surface at p010, so a following
``hwdownload,format=nv12`` fails on 10-bit sources. Each chain converts to nv12 on the GPU
first, so one download format works for 8-bit, 10-bit, 4:4:4 and 12-bit NVDEC output alike.

ffmpeg runs on fakes: there is no GPU here, and a CPU-only ffmpeg cannot negotiate CUDA frames,
so the real 10-bit decode is verified on an NVIDIA machine.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import media_info
from cutpoint import crash
from cutpoint import ffmpeg_utils as fu
from cutpoint.events import visual
from cutpoint.pano import qa

CUDA = fu.HwDecoder("cuda")
SRC = Path(__file__).resolve().parents[1] / "src" / "cutpoint"
BARE = re.compile(r"scale_cuda=[^,\"'`:]+:[^,\"'`:]+,hwdownload")  # no format= before it


class _Proc:
    """A finished ``subprocess.Popen`` decode process."""

    def __init__(self, stdout: bytes, stderr: bytes = b"") -> None:
        self.stdout, self.stderr = io.BytesIO(stdout), io.BytesIO(stderr)
        self.returncode = 0

    def poll(self) -> int:
        return self.returncode

    def kill(self) -> None:
        pass

    def wait(self, timeout=None) -> int:
        return self.returncode


@pytest.fixture
def ffmpeg(monkeypatch):
    """Fake decode passes. ``hw`` is the probed decoder; with ``gpu_frames`` false the GPU pass
    yields nothing (as on a failed download), so the CPU pass follows. Every argv is in ``cmds``."""
    state = SimpleNamespace(hw=CUDA, gpu_frames=True, pix_fmt="yuv420p", size=(8, 4), cmds=[])
    showinfo = b"[Parsed_showinfo_9 @ 0] n:0 pts:0 pts_time:0.5 duration:1\n"

    def popen(cmd, **_kw):
        state.cmds.append(cmd)
        w, h = state.size
        gpu = "-hwaccel" in cmd
        frame = bytes(w * h * 3) if (state.gpu_frames or not gpu) else b""
        return _Proc(frame, showinfo if frame else b"")

    monkeypatch.setattr(fu, "find_tool", lambda name: name)
    monkeypatch.setattr(visual, "find_tool", lambda name: name)
    monkeypatch.setattr(fu, "decoder_for", lambda path: state.hw)
    monkeypatch.setattr(
        fu, "probe", lambda p: media_info(Path(p), 10.0, width=16, height=8, pix_fmt=state.pix_fmt)
    )
    monkeypatch.setattr(fu.subprocess, "Popen", popen)
    return state


def _vf(cmd):
    return cmd[cmd.index("-vf") + 1]


def _crash(ffmpeg):
    ffmpeg.size = (8, 4)  # 16x8 source at width 8
    _, frames = crash.decode_window(Path("in.mp4"), 0.0, 1.0, 5.0, width=8)
    assert frames.shape == (1, 4, 8, 3)
    return "scale_cuda=8:4:format=nv12,hwdownload,format=nv12,fps=5,format=rgb24"


def _visual(ffmpeg):
    ffmpeg.size = (8, 4)
    assert len(list(visual.iter_frames(Path("in.mp4"), 8, 4, False, True))) == 1
    return "fps=1,scale_cuda=8:4:format=nv12,hwdownload,format=nv12,format=rgb24,showinfo"


def _qa(ffmpeg):
    ffmpeg.size = (8, 4)
    _, frames = qa.decode(Path("in.mp4"), 2.0, (8, 4))
    assert frames.shape == (1, 4, 8, 3)
    return "fps=2:round=up,scale_cuda=8:4:format=nv12,hwdownload,format=nv12,format=rgb24,showinfo"


CPU_CHAINS = {
    _crash: "scale=8:4:flags=area,fps=5,format=rgb24",
    _visual: "fps=1,scale=8:4:flags=area,format=rgb24,showinfo",
    _qa: "fps=2:round=up,scale=8:4:flags=area,format=rgb24,showinfo",
}
CHAINS = pytest.mark.parametrize("chain", list(CPU_CHAINS), ids=["crash", "visual", "pano_qa"])


@CHAINS
def test_the_cuda_pass_converts_to_nv12_on_the_gpu_before_the_download(ffmpeg, chain):
    want = chain(ffmpeg)
    (gpu,) = ffmpeg.cmds
    assert gpu[gpu.index("-hwaccel") : gpu.index("-hwaccel") + 4] == [
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
    ]  # fmt: skip
    assert _vf(gpu) == want


@CHAINS
def test_a_failed_cuda_pass_falls_back_to_the_unchanged_cpu_chain(ffmpeg, chain):
    ffmpeg.gpu_frames = False
    chain(ffmpeg)
    gpu, cpu = ffmpeg.cmds
    assert "scale_cuda" in _vf(gpu)
    assert "-hwaccel" not in cpu and "-hwaccel_output_format" not in cpu
    assert _vf(cpu) == CPU_CHAINS[chain]


@CHAINS
def test_without_a_gpu_decoder_only_the_cpu_chain_runs(ffmpeg, chain):
    ffmpeg.hw = None
    chain(ffmpeg)
    (cpu,) = ffmpeg.cmds
    assert "-hwaccel" not in cpu
    assert _vf(cpu) == CPU_CHAINS[chain]


def test_no_chain_downloads_a_scale_cuda_surface_without_a_format():
    """A ``scale_cuda`` followed by ``hwdownload`` names its output format (#870)."""
    hits = [
        f"{path.relative_to(SRC)}:{n}"
        for path in sorted(SRC.rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if BARE.search(line)
    ]
    assert hits == []


def test_the_guard_pattern_catches_the_old_chain():
    assert BARE.search('f"scale_cuda={w}:{h},hwdownload,format=nv12"')
    assert not BARE.search('f"scale_cuda={w}:{h}:format=nv12,hwdownload,format=nv12"')
