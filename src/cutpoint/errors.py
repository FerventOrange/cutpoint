"""Exceptions shared across cutpoint (stdlib only, so any module can import them cheaply)."""

from __future__ import annotations


class FFmpegError(RuntimeError):
    """Raised when ffmpeg/ffprobe is missing or a command fails."""


class FFmpegNotFoundError(FFmpegError):
    """Raised when the ffmpeg or ffprobe executable cannot be found."""


class EncoderUnavailableError(FFmpegError):
    """A video encoder the user asked for cannot encode on this machine."""


class FolderBusyError(RuntimeError):
    """Another cutpoint run holds the lock of the folder (see
    :func:`cutpoint.folders.swap_lock`)."""


class LowMemoryError(RuntimeError):
    """Not enough free memory to load a model, also after freeing cached ones
    (:mod:`cutpoint.memguard`); ``cutpoint resume`` continues the run. ``args[1]``, when
    given: what is short (``"1.2 GB RAM free, 3.5 GB needed"``); ``str()`` is the message."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class LowVramError(LowMemoryError):
    """:class:`LowMemoryError` where the GPU's memory alone is short: Whisper, PANNs and the
    crash check's CLIP run on the CPU instead (:func:`cutpoint.memguard.device_for`); the
    visual models stop the run."""


class NeedsAnalyzeError(RuntimeError):
    """A re-rank (:mod:`cutpoint.rerank`) needs features the project's caches do not hold, so
    ``analyze`` has to run with the style. ``missing``: what is missing (each with ``feature``,
    ``detail`` and ``source``); ``command``: the command that analyzes the project with the
    style."""

    def __init__(self, message: str, missing: list[dict[str, str | None]], command: str) -> None:
        super().__init__(message)
        self.missing = missing
        self.command = command
