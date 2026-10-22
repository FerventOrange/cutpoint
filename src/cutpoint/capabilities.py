"""What this machine can use: GPU vendors present, working video encoders, transcription.

``cutpoint capabilities`` prints it. Three facts are kept apart on purpose: a vendor's
GPU being *present*, an encoder being *usable* (ffmpeg really encoded a short clip with
it) and what *transcription* runs on. Nothing here installs or downloads anything.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from cutpoint.config import get_config
from cutpoint.ffmpeg_utils import (
    VIDEO_CODECS,
    EncoderUnavailableError,
    FFmpegError,
    find_encoder,
    run,
    video_encoder,
)

_VENDORS = {
    "nvidia": ("0x10de", re.compile(r"nvidia", re.I)),
    "amd": ("0x1002", re.compile(r"amd|radeon", re.I)),
}


def _adapters() -> list[str]:
    """Display adapters as ``vendor ids`` (Linux ``0x1002``) or driver names (Windows)."""
    if sys.platform.startswith("linux"):
        drm = Path("/sys/class/drm")
        return [p.read_text().strip().lower() for p in drm.glob("card*/device/vendor")]
    if sys.platform != "win32":
        return []
    import winreg  # noqa: PLC0415

    key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    names = []
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as cls:
        for i in range(winreg.QueryInfoKey(cls)[0]):
            try:
                with winreg.OpenKey(cls, winreg.EnumKey(cls, i)) as sub:
                    names.append(str(winreg.QueryValueEx(sub, "DriverDesc")[0]))
            except OSError:  # "Configuration" / "Properties" folders have no adapter
                continue
    return names


def gpu_vendors() -> dict[str, bool]:
    """Which GPU vendors have a display adapter here (hardware present, not a usable runtime)."""
    try:
        adapters = _adapters()
    except OSError:
        adapters = []
    return {
        name: any(vendor_id == a or pattern.search(a) for a in adapters)
        for name, (vendor_id, pattern) in _VENDORS.items()
    }


def _compiled_encoders() -> set[str]:
    """H.264 encoders this ffmpeg build lists (listed is not working)."""
    try:
        out = run(["ffmpeg", "-hide_banner", "-encoders"])
    except FFmpegError:
        return set()
    return {c for c in VIDEO_CODECS.values() if re.search(rf"\s{c}\s", out)}


def _transcription() -> dict[str, Any]:
    from cutpoint.transcribe import cuda_available  # noqa: PLC0415 - may import ctranslate2

    cuda = cuda_available()
    return {
        "engine": "faster-whisper",
        "device": "cuda" if cuda else "cpu",
        "note": "NVIDIA CUDA is used when its libraries load; AMD and other GPUs are not used "
        "for transcription, which runs on the CPU.",
    }


def _llama() -> dict[str, Any]:
    from cutpoint.events import setup  # noqa: PLC0415

    return {
        "requested": setup.llama_backend_choice(),
        "setup_would_choose": list(setup.llama_backends()),
        "nvidia_driver_usable": setup.nvidia_usable(),
    }


def report() -> dict[str, Any]:
    """The capability report ``cutpoint capabilities`` prints (JSON-serialisable)."""
    compiled = _compiled_encoders()
    encoders: dict[str, Any] = {}
    for kind, codec in VIDEO_CODECS.items():
        enc, errors = find_encoder(kind)
        encoders[kind] = {
            "codec": codec,
            "listed_by_ffmpeg": codec in compiled,
            "usable": enc is not None,
            "device": enc.device if enc else None,
            "error": "; ".join(errors) or None,
        }
    requested = get_config().video_encoder
    active: dict[str, Any] = {"requested": requested}
    try:
        enc = video_encoder()
        active |= {"kind": enc.kind, "codec": enc.codec, "device": enc.device, "error": None}
    except EncoderUnavailableError as err:
        active |= {"kind": None, "codec": None, "device": None, "error": str(err)}
    return {
        "platform": sys.platform,
        "gpu_vendors_present": gpu_vendors(),
        "video_encoders": encoders,
        "active_video_encoder": active,
        "transcription": _transcription(),
        "llama_cpp": _llama(),
    }


def format_report(data: dict[str, Any]) -> str:
    """The report as text for people."""
    v = data["gpu_vendors_present"]
    lines = [
        f"GPU vendors present: NVIDIA {'yes' if v['nvidia'] else 'no'}, "
        f"AMD {'yes' if v['amd'] else 'no'} (hardware only; says nothing about usable runtimes)",
        "Video encoders (a short clip was really encoded with each):",
    ]
    for kind, e in data["video_encoders"].items():
        state = "usable" if e["usable"] else f"not usable ({e['error'] or 'no device'})"
        dev = f" on {e['device']}" if e["device"] else ""
        lines.append(f"  {kind:<6} {e['codec']:<11} {state}{dev}")
    a = data["active_video_encoder"]
    chosen = a["error"]
    if a["kind"]:
        chosen = a["codec"] + (f" on {a['device']}" if a["device"] else "")
    lines.append(f"Active video encoder: {chosen} (requested: {a['requested']})")
    t = data["transcription"]
    lines.append(f"Transcription: {t['engine']} on {t['device']}. {t['note']}")
    ll = data["llama_cpp"]
    lines.append(
        f"llama.cpp setup backend: {' > '.join(ll['setup_would_choose'])} "
        f"(requested: {ll['requested']}; usable NVIDIA driver: "
        f"{'yes' if ll['nvidia_driver_usable'] else 'no'})"
    )
    return "\n".join(lines)
