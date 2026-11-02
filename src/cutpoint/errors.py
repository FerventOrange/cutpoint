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
