"""What this machine can use: GPU vendors present, working video encoders, transcription,
torch and the Insta360 Media SDK.

``cutpoint capabilities`` prints it. Three facts are kept apart on purpose: a vendor's
GPU being *present*, an encoder being *usable* (ffmpeg really encoded a short clip with
it) and what *transcription* runs on. Nothing here installs or downloads anything.
"""

from __future__ import annotations

import importlib.util
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

# Said in ``capabilities`` and before ``cutpoint 360 stitch`` when no NVIDIA GPU is found.
SDK_NEEDS_NVIDIA = (
    "The Insta360 Media SDK (cutpoint 360 stitch) needs an NVIDIA GPU and none was found "
    "here; stitching will likely fail. Use Insta360 Studio exports or the dual-fisheye "
    "preview instead."
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


def _nvidia_driver_usable() -> bool:
    from cutpoint.events import setup  # noqa: PLC0415

    return setup.nvidia_usable()


def nvidia_present() -> bool:
    """True when an NVIDIA GPU is here: its display adapter or a driver that answers."""
    return gpu_vendors()["nvidia"] or _nvidia_driver_usable()


def _llama(nvidia_driver: bool) -> dict[str, Any]:
    from cutpoint.events import setup  # noqa: PLC0415

    return {
        "requested": setup.llama_backend_choice(),
        "setup_would_choose": list(setup.llama_backends()),
        "nvidia_driver_usable": nvidia_driver,
    }


def _torch(nvidia_driver: bool) -> dict[str, Any]:
    """Whether torch (``events`` extra) is installed, its CUDA build and whether it reaches the
    GPU; ``warning`` and ``fix`` when a usable NVIDIA driver is here but torch cannot use it."""
    from cutpoint import extras  # noqa: PLC0415

    info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_build": None,
        "cuda_available": False,
        "error": None,
        "warning": None,
        "fix": None,
    }
    if importlib.util.find_spec("torch") is None:
        info["fix"] = f"it comes with the 'events' extra: {extras.install_hint('events')}"
        return info
    try:
        import torch  # noqa: PLC0415 - optional and slow; only for this report

        info |= {
            "installed": True,
            "version": str(torch.__version__),
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except (ImportError, OSError, RuntimeError) as err:  # e.g. a DLL that fails to load
        info |= {"installed": True, "error": f"{type(err).__name__}: {err}"}
        return info
    if info["cuda_available"] or not nvidia_driver:
        return info
    build = info["cuda_build"]
    if build is None:
        info["warning"] = (
            f"torch {info['version']} is a CPU-only build, but this machine has a usable "
            "NVIDIA GPU: events, crash detection and 360 views run on the CPU, much slower."
        )
        info["fix"] = extras.torch_cuda_hint()
    else:
        info["warning"] = (
            f"torch {info['version']} is a CUDA {build} build but cannot use the NVIDIA GPU; "
            f"the driver is probably too old for CUDA {build}."
        )
        info["fix"] = "update the NVIDIA driver" + (
            ""
            if build == "12.6"
            else f", or install the CUDA 12.6 build: {extras.torch_cuda_hint()}"
        )
    return info


def _insta360_sdk(nvidia: bool) -> dict[str, Any]:
    """Where the Insta360 Media SDK is, and whether the NVIDIA GPU it needs is here."""
    from cutpoint.pano.project import sdk_location, sdk_version  # noqa: PLC0415

    info: dict[str, Any] = {
        "found": False,
        "path": None,
        "origin": "none",
        "version": None,
        "error": None,
        "nvidia_gpu_present": nvidia,
        "warning": None if nvidia else SDK_NEEDS_NVIDIA,
    }
    try:
        exe, origin = sdk_location()
    except ValueError as err:  # a malformed ``pano`` config section
        info["error"] = str(err)
        return info
    if exe is not None:
        info |= {
            "found": exe.is_file(),
            "path": str(exe),
            "origin": origin,
            "version": sdk_version(exe),
        }
    return info


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
    vendors = gpu_vendors()
    nvidia_driver = _nvidia_driver_usable()
    # torch must load its own CUDA/cuDNN DLLs before _transcription() loads ctranslate2's (the
    # gpu extra's newer cuDNN): on Windows, the other order makes `import torch` fail.
    torch_info = _torch(nvidia_driver)
    return {
        "platform": sys.platform,
        "gpu_vendors_present": vendors,
        "video_encoders": encoders,
        "active_video_encoder": active,
        "transcription": _transcription(),
        "llama_cpp": _llama(nvidia_driver),
        "torch": torch_info,
        "insta360_sdk": _insta360_sdk(vendors["nvidia"] or nvidia_driver),
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
    tc = data["torch"]
    if not tc["installed"]:
        lines.append(f"torch: not installed ({tc['fix']})")
    elif tc["error"]:
        lines.append(f"torch: installed but does not load ({tc['error']})")
    else:
        build = f"CUDA {tc['cuda_build']} build" if tc["cuda_build"] else "CPU-only build"
        gpu = "uses the GPU" if tc["cuda_available"] else "runs on the CPU"
        lines.append(f"torch: {tc['version']}, {build}, {gpu}")
    if tc["warning"]:
        lines += [f"  WARNING: {tc['warning']}", f"  Fix: {tc['fix']}"]
    sdk = data["insta360_sdk"]
    if sdk["error"]:
        state = f"not checked ({sdk['error']})"
    elif sdk["found"]:
        state = f"found at {sdk['path']} ({sdk['origin']})"
    elif sdk["path"]:
        state = f"not found: {sdk['path']} ({sdk['origin']}) does not exist"
    else:
        state = "not found"
    gpu = "yes" if sdk["nvidia_gpu_present"] else "no"
    lines.append(f"Insta360 Media SDK: {state}; NVIDIA GPU (needed): {gpu}")
    if sdk["warning"]:
        lines.append(f"  WARNING: {sdk['warning']}")
    return "\n".join(lines)
