"""``visual.iter_frames``: a stale stderr reader must not pollute the CPU fallback pass.

Regression test for "bind loop variables in visual reader thread". No ffmpeg, no GPU:
``subprocess.Popen`` is faked and the pass order is forced with Events. The decode
loop and its reader thread live in :func:`cutpoint.ffmpeg_utils.decode_raw`, so the
fakes are installed there.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from cutpoint import ffmpeg_utils
from cutpoint.events import visual

W, H = 2, 2
FRAME = bytes(W * H * 3)
TIMEOUT = 10.0  # only reached if the test is broken; normal runs never wait on it


def _showinfo(n: int, t: float) -> bytes:
    return f"[Parsed_showinfo_3 @ 0] n:{n} pts:{n} pts_time:{t}\n".encode()


class _Reader(threading.Thread):
    """Reader thread whose join() returns at once, like the 5 s join timing out."""

    started: ClassVar[list[_Reader]] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _Reader.started.append(self)

    def join(self, timeout=None):
        super().join(0)

    def wait_finished(self) -> None:
        super().join(TIMEOUT)
        assert not self.is_alive()


def test_stale_hw_reader_does_not_pollute_cpu_pass(monkeypatch):
    _Reader.started = []
    release_stale = threading.Event()

    def hw_stderr():
        # Outlives the hw pass: blocks until the CPU pass is reading, then writes a
        # timestamp (and, on exit, the end marker) that belongs to the hw pass.
        assert release_stale.wait(TIMEOUT)
        yield _showinfo(0, 999.0)

    def cpu_stderr():
        release_stale.set()
        _Reader.started[0].wait_finished()  # stale writes land before ours
        yield _showinfo(0, 0.5)
        yield _showinfo(1, 1.5)

    passes = [
        (io.BytesIO(b""), hw_stderr()),  # hw pass: no frames
        (io.BytesIO(FRAME * 2), cpu_stderr()),  # CPU pass: two frames
    ]
    cmds: list[list[str]] = []

    def fake_popen(cmd, **_kw):
        cmds.append(cmd)
        stdout, stderr = passes[len(cmds) - 1]
        return SimpleNamespace(
            stdout=stdout, stderr=stderr, wait=lambda: 0, poll=lambda: 0, kill=lambda: None
        )

    monkeypatch.setattr(ffmpeg_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ffmpeg_utils, "find_tool", lambda name: name)
    monkeypatch.setattr(visual, "find_tool", lambda name: name)
    monkeypatch.setattr(ffmpeg_utils, "stage_inputs", lambda cmd: cmd)
    monkeypatch.setattr(
        ffmpeg_utils,
        "threading",
        SimpleNamespace(
            Thread=_Reader,
            Condition=threading.Condition,
            Event=threading.Event,
            Timer=threading.Timer,
        ),
    )

    out = list(visual.iter_frames(Path("x.mp4"), W, H, keyframes=False, hwaccel=True))

    assert len(cmds) == 2
    assert "cuda" in cmds[0] and "cuda" not in cmds[1]
    assert [t for t, _ in out] == [0.5, 1.5]
    assert all(f.shape == (H, W, 3) for _, f in out)
