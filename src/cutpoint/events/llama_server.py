"""Start, health-check and stop a local ``llama-server`` (llama.cpp) process.

The server is loaded with the model + mmproj from ``models/llm`` (``-m`` /
``--mmproj``). If that fails, e.g. the 400 some builds return with a manually
loaded mmproj, :class:`LlamaServer` can retry with ``-hf`` loading, using
``LLAMA_CACHE=models/llama-cache`` so any download stays under ``models/``.

Before starting, any Ollama models are unloaded so the two never hold VRAM at
the same time.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from cutpoint.events import paths
from cutpoint.events.common import unload_ollama
from cutpoint.ffmpeg_utils import exit_code
from cutpoint.winjob import kill_with_parent

logger = logging.getLogger(__name__)


def free_port(preferred: int = 8089) -> int:
    """``preferred`` if it is free, else any free local port."""
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def _pdeathsig_hook() -> Callable[[], None] | None:
    """Linux ``preexec_fn`` that makes the child get SIGTERM when Python exits.

    The Linux counterpart of :func:`cutpoint.winjob.kill_with_parent`: without either, a
    crashed or killed driver leaves llama-server holding ~8 GB of VRAM. libc is loaded here,
    in the parent, so the child only calls ``prctl`` between fork and exec.
    None on other platforms or if libc is unavailable.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes  # noqa: PLC0415

        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError) as err:
        logger.debug("prctl not available: %s", err)
        return None
    parent = os.getpid()

    def hook() -> None:
        try:
            prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
            if os.getppid() != parent:  # parent already gone
                os.kill(os.getpid(), signal.SIGTERM)
        except Exception:  # noqa: BLE001 - best effort, must not break the launch
            pass

    return hook


class LlamaServerError(RuntimeError):
    """Raised when llama-server cannot be started or answers with an error."""


class LlamaServer:
    """Context manager owning one llama-server process.

    Args:
        model: GGUF model path (``-m``). Ignored when ``hf_repo`` is used.
        mmproj: Multimodal projector path (``--mmproj``).
        hf_repo: ``-hf`` spec (e.g. ``ggml-org/Qwen2.5-Omni-7B-GGUF:Q4_K_M``)
            used instead of ``model``/``mmproj``.
        ctx: Context size in tokens.
        image_max_tokens: Cap per image (keeps 12-16 frames inside the context).
        extra_args: More command-line arguments.
        log_path: Where server output goes (default: next to the exe).
    """

    def __init__(
        self,
        model: Path | None = None,
        mmproj: Path | None = None,
        *,
        hf_repo: str | None = None,
        ctx: int = 16384,
        image_max_tokens: int | None = None,
        port: int = 8089,
        extra_args: list[str] | None = None,
        log_path: Path | None = None,
        startup_timeout_s: float = 300.0,
    ) -> None:
        # None = the Qwen2.5-Omni files installed by ``cutpoint setup``.
        self.model = model or paths.omni_model()
        self.mmproj = mmproj or paths.omni_mmproj()
        self.hf_repo = hf_repo
        self.ctx = ctx
        self.image_max_tokens = image_max_tokens
        self.port = free_port(port)
        self.extra_args = extra_args or []
        self.log_path = log_path or paths.llama_dir() / "llama-server.log"
        self.startup_timeout_s = startup_timeout_s
        self.proc: subprocess.Popen[bytes] | None = None
        self.url = f"http://127.0.0.1:{self.port}"
        self.load_s: float | None = None

    def command(self) -> list[str]:
        """The llama-server command line."""
        exe = paths.llama_server_exe()
        if not exe.exists():
            raise LlamaServerError(f"{exe} not found; run cutpoint setup --only llama")
        cmd = [
            str(exe),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-c",
            str(self.ctx),
            "-ngl",
            "99",
            "-fa",
            "on",
            "-np",
            "1",
            "--no-webui",
            "--jinja",
            # Host-RAM prompt cache grows ~650 MB per multimodal request
            # (13 GB after 20 windows); every window is new, so disable it.
            "--cache-ram",
            "0",
        ]
        if self.hf_repo:
            cmd += ["-hf", self.hf_repo]
        else:
            cmd += ["-m", str(self.model)]
            if self.mmproj:
                cmd += ["--mmproj", str(self.mmproj)]
        if self.image_max_tokens:
            cmd += ["--image-max-tokens", str(self.image_max_tokens)]
        return cmd + self.extra_args

    def start(self) -> LlamaServer:
        """Launch the server and wait until ``/health`` reports ok."""
        unload_ollama()
        cmd = self.command()
        env = dict(os.environ, LLAMA_CACHE=str(paths.llama_cache()))
        exe_dir = str(paths.llama_server_exe().parent)
        if sys.platform.startswith("linux"):  # the Linux builds ship .so files next to the exe
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                p for p in (exe_dir, env.get("LD_LIBRARY_PATH")) if p
            )
        logger.info("Starting llama-server: %s", subprocess.list2cmdline(cmd))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with self.log_path.open("wb") as log:  # the child keeps its own copy of the descriptor
            self.proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=exe_dir,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                preexec_fn=_pdeathsig_hook(),
            )
        kill_with_parent(self.proc)
        deadline = time.time() + self.startup_timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise LlamaServerError(
                    f"llama-server exited ({exit_code(self.proc.returncode)}); "
                    f"see {self.log_path}:\n{self.log_tail()}"
                )
            try:
                r = requests.get(f"{self.url}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    self.load_s = round(time.perf_counter() - t0, 1)
                    logger.info("llama-server ready on %s in %.1fs", self.url, self.load_s)
                    return self
            except (requests.RequestException, ValueError):
                pass
            time.sleep(1.0)
        self.stop()
        raise LlamaServerError(f"llama-server not healthy after {self.startup_timeout_s}s")

    def log_tail(self, n: int = 20) -> str:
        """Last ``n`` lines of the server log."""
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except OSError:
            return ""

    def stop(self) -> None:
        """Terminate the server (waits for VRAM to be released)."""
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
            logger.info("llama-server stopped")
        self.proc = None

    def __enter__(self) -> LlamaServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 800,
        temperature: float = 0.2,
        json_schema: dict[str, Any] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        """POST ``/v1/chat/completions``; returns the parsed response JSON.

        Raises:
            LlamaServerError: On an HTTP error (the body is included).
        """
        body: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "cache_prompt": False,
        }
        if json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": json_schema},
            }
        r = requests.post(f"{self.url}/v1/chat/completions", json=body, timeout=timeout_s)
        if r.status_code != 200:
            raise LlamaServerError(f"HTTP {r.status_code}: {r.text[:500]}")
        return r.json()
