"""Memory guard (#109): a model is loaded only when the memory it needs is free.

Before a model loads (Whisper, PANNs, the visual models, CLIP, the OCR engine; the vision model
before the request that loads it into Ollama, again after each unload), :func:`check` compares
the free memory with the model's need (:data:`NEEDS`, estimates) plus the ``memory_guard``
headroom of ``config.json`` (GB of RAM left free, default
:data:`cutpoint.config.MEMORY_GUARD_GB`; 0 turns the guard off). A model on the CPU needs its RAM; on a GPU its VRAM, with the headroom alone left in RAM.
Ollama's vision model needs the size Ollama reports for it (:attr:`VisionRater.size
<cutpoint.vision.VisionRater>`), in RAM and VRAM together (Ollama splits a model between them),
and nothing when Ollama has it loaded already.

When memory is short, :func:`cutpoint.analyze.free_models` and
:func:`cutpoint.events.common.unload_ollama` free what they can and the guard checks again;
still short, it raises :class:`~cutpoint.errors.LowMemoryError` (``--json`` ``error.code``
``low_memory``, exit 1; ``progress.json`` status ``low_memory``) with an ERROR text that names
``cutpoint resume <project>``. Shrinking ``--batch-size`` does not help: the need is per model.

Free RAM: ``MemAvailable`` of ``/proc/meminfo`` (Linux), ``GlobalMemoryStatusEx`` (Windows);
elsewhere (macOS) it is unknown and the guard is off. Free VRAM: torch's
``cuda.mem_get_info`` when torch already uses CUDA, else ``nvidia-smi`` of the GPU the models
use (:func:`visible_gpu`: the first entry of ``CUDA_VISIBLE_DEVICES``, an index or GPU UUID, is
torch's ``cuda:0``; GPU 0 when it is unset); unknown: no VRAM check, also when
``CUDA_VISIBLE_DEVICES`` hides every GPU (empty, ``-1``) or names one nvidia-smi does not list. A vision server Cutpoint cannot unload (a custom server on this machine, Q6) gets the
RAM headroom check alone and a WARNING; one on another machine uses no memory here.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from cutpoint import state
from cutpoint.errors import LowMemoryError, LowVramError
from cutpoint.utils import command_line

if TYPE_CHECKING:
    from cutpoint.vision import VisionRater

logger = logging.getLogger(__name__)

GIB = 1024**3
# (RAM GB on the CPU, VRAM GB on a GPU) a loaded model takes; estimates, kept low so the guard
# stops only a run that would run out of memory
NEEDS: dict[str, tuple[float, float]] = {
    "whisper-tiny": (0.5, 0.5),
    "whisper-base": (0.5, 0.5),
    "whisper-small": (1.0, 1.0),
    "whisper-medium": (2.0, 2.0),
    "whisper-turbo": (2.0, 2.0),
    "whisper-large": (3.5, 4.0),
    "panns": (1.0, 0.5),
    "clip": (2.0, 1.0),  # CLIP ViT-L/14 image tower (crash dense check)
    "visual": (2.5, 1.5),  # CLIP ViT-L/14 + Faster R-CNN v2
    "ocr": (1.0, 0.0),  # RapidOCR, CPU only
    "vlm": (7.0, 7.0),  # Ollama reported no size: qwen3.5:9b takes about 6.6 GB
}
LABELS = {
    "panns": "PANNs",
    "clip": "CLIP",
    "visual": "the visual models (CLIP and Faster R-CNN)",
    "ocr": "the OCR engine",
}
# vision servers warned about, once per process; emptied by reset_state
_WARNED: set[str] = state.register_container(__name__, "_WARNED", set())


def whisper_need(name: str) -> str:
    """The :data:`NEEDS` key of the Whisper model ``name`` (unknown names: medium's)."""
    n = name.lower().removesuffix(".en")
    if "turbo" in n or n.startswith("distil-large"):
        return "whisper-turbo"
    if n.startswith("large"):
        return "whisper-large"
    n = n.removeprefix("distil-")
    return f"whisper-{n}" if f"whisper-{n}" in NEEDS else "whisper-medium"


def headroom_gb() -> float:
    """``memory_guard`` of config.json: GB of RAM left free (0: the guard is off)."""
    from cutpoint.config import get_config  # noqa: PLC0415

    return get_config().memory_guard


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def ram_free() -> int | None:
    """Free RAM in bytes (Linux ``MemAvailable``, Windows ``ullAvailPhys``); None: unknown
    (macOS, or the query failed), so the guard is off."""
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
        return None
    if sys.platform == "win32":
        stat = _MemoryStatus()
        stat.dwLength = ctypes.sizeof(_MemoryStatus)
        try:
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            return None
        return int(stat.ullAvailPhys) if ok else None
    return None


def visible_gpu() -> str | None:
    """The GPU torch's ``cuda:0`` is: the first entry of ``CUDA_VISIBLE_DEVICES`` (an index
    or a GPU UUID), ``"0"`` when it is unset; None when it hides every GPU (empty, ``-1``)."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is None:
        return "0"
    first = env.split(",")[0].strip()
    return None if not first or first.startswith("-") else first


def _smi_rows(fields: str) -> list[list[str]]:
    """``index, uuid, <fields>`` of every GPU nvidia-smi lists ([] if unavailable)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu=index,uuid,{fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [[f.strip() for f in line.split(",")] for line in out.splitlines() if line.strip()]


def gpu_mib(*fields: str) -> tuple[int, ...] | None:
    """nvidia-smi's ``memory.<field>`` values in MiB (``"free"``, ``"total"``, ...) of
    :func:`visible_gpu`: the row of that index, or of the UUID it starts (as CUDA, a unique
    prefix names a GPU); None if unknown (also for a prefix of several UUIDs)."""
    sel = visible_gpu()
    if sel is None:
        return None
    rows = [
        r for r in _smi_rows(",".join(f"memory.{f}" for f in fields)) if len(r) == 2 + len(fields)
    ]
    if sel.isdigit():
        found = [r for r in rows if r[0] == str(int(sel))]
    else:
        found = [r for r in rows if r[1].lower().startswith(sel.lower())]
    if len(found) != 1:
        return None
    try:
        return tuple(int(v) for v in found[0][2:])
    except ValueError:  # [N/A]
        return None


def gpu_free_mib() -> int | None:
    """Free memory in MiB of :func:`visible_gpu` via nvidia-smi (:func:`gpu_mib`); None if
    unknown."""
    got = gpu_mib("free")
    return got[0] if got is not None else None


def vram_free() -> int | None:
    """Free memory of the GPU in bytes: torch's ``cuda.mem_get_info`` when torch already uses
    CUDA (no new CUDA context for the check), else nvidia-smi's of :func:`visible_gpu`;
    None: unknown."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            return int(torch.cuda.mem_get_info()[0])
    except Exception:
        logger.debug("torch.cuda.mem_get_info failed", exc_info=True)
    mib = gpu_free_mib()
    return mib * 2**20 if mib is not None else None


def _gb(n: float) -> str:
    return f"{n:.1f} GB"


def _shortfall(
    head: float, ram_gb: float, vram_gb: float | None, pooled: bool
) -> tuple[str, bool] | None:
    """What is short (``"1.2 GB RAM free, 3.5 GB needed"``) and whether it is the VRAM alone,
    or None: enough, or unknown. ``pooled``: RAM and VRAM together hold ``ram_gb`` (Ollama)."""
    ram = ram_free()
    if ram is None:
        return None
    vram = vram_free() if pooled or vram_gb else None
    if pooled:
        free, need = (ram + (vram or 0)) / GIB, head + ram_gb
        if free < need:
            where = "RAM and VRAM" if vram is not None else "RAM"
            return f"{_gb(free)} {where} free, {_gb(need)} needed", False
        return None
    if ram / GIB < head + ram_gb:
        return f"{_gb(ram / GIB)} RAM free, {_gb(head + ram_gb)} needed", False
    if vram_gb and vram is not None and vram / GIB < vram_gb:
        return f"{_gb(vram / GIB)} VRAM free, {_gb(vram_gb)} needed", True
    return None


def _free(url: str | None = None) -> None:
    """Free cached models: torch's memory and the loaded models of the Ollama at ``url``
    (None: the configured one)."""
    from cutpoint import analyze  # noqa: PLC0415
    from cutpoint.events import common  # noqa: PLC0415

    analyze.free_models()
    common.unload_ollama(url)


def _resume_hint() -> str:
    from cutpoint import runstate  # noqa: PLC0415

    project = runstate.resume_project()
    if project is None:
        return "then run the command again"
    return f"then run `{command_line(['cutpoint', 'resume', str(project)])}` to continue"


def _guard(
    what: str,
    ram_gb: float,
    vram_gb: float | None,
    *,
    pooled: bool = False,
    loaded: Callable[[], bool] | None = None,
    url: str | None = None,
) -> None:
    """Stop the run unless ``what`` fits (see the module doc); ``loaded``: whether it is in
    memory already (then nothing is short), asked only when memory looks short; ``url``: the
    Ollama to unload (None: the configured one)."""
    head = headroom_gb()
    if head <= 0:
        return
    found = _shortfall(head, ram_gb, vram_gb, pooled)
    if found is None or (loaded is not None and loaded()):
        return
    logger.info("Memory low for %s (%s): freeing cached models", what, found[0])
    _free(url)
    found = _shortfall(head, ram_gb, vram_gb, pooled)
    if found is None:
        logger.info("Memory for %s is free now", what)
        return
    short, vram_only = found
    error = LowVramError if vram_only else LowMemoryError
    # a VRAM shortage is not about the RAM memory_guard keeps free
    keeps = "" if vram_only else f" (memory_guard keeps {head:g} GB of RAM free)"
    raise error(
        f"Not enough memory for {what}: {short}{keeps}; "
        f"close other programs, {_resume_hint()} (memory_guard 0 in config.json turns this "
        "check off)",
        short,
    )


def check(model: str, *, gpu: bool = False, label: str | None = None) -> None:
    """Before ``model`` (a :data:`NEEDS` key) loads, on the GPU (``gpu``) or the CPU; see the
    module doc. ``label``: its name in the messages (default :data:`LABELS`).

    Raises:
        LowVramError: On the GPU, if its VRAM is what is short (a model that runs on the CPU
            too uses :func:`device_for`, which catches it).
        LowMemoryError: If the memory it needs is not free, also after freeing cached models.
    """
    ram_gb, vram_gb = NEEDS[model]
    what = label or LABELS.get(model, model)
    if gpu:
        _guard(what, 0.0, vram_gb)
    else:
        _guard(what, ram_gb, None)


def device_for(
    model: str,
    device: str,
    *,
    label: str | None = None,
    cpu_model: str | None = None,
    cpu_label: str | None = None,
) -> str:
    """:func:`check` of ``model`` (a :data:`NEEDS` key) on ``device``, for a model that runs on
    the CPU too (Whisper, PANNs, the crash check's CLIP): ``device``, or ``"cpu"`` with a
    WARNING when the GPU's memory is what is short (a game or OBS holds it), checked for RAM.
    ``cpu_model`` and ``cpu_label``: the :data:`NEEDS` key and name of what runs on the CPU
    instead (Whisper without ``--model``: the CPU default), named in the WARNING (default
    ``model``).

    Raises:
        LowMemoryError: If the RAM it needs is not free, also after freeing cached models.
    """
    try:
        check(model, gpu=device == "cuda", label=label)
    except LowVramError as err:
        what = label or LABELS.get(model, model)
        on_cpu = f" with {cpu_label}" if cpu_label else ""
        logger.warning("%s: %s; running on the CPU%s", what, err.args[1], on_cpu)
        check(cpu_model or model, label=cpu_label or label)
        return "cpu"
    return device


def _ollama_loaded(url: str, model: str) -> bool:
    """Ollama at ``url`` has ``model`` in memory (``/api/ps``); False when it does not answer."""
    import requests  # noqa: PLC0415

    from cutpoint.vision import refuse_redirect  # noqa: PLC0415

    try:
        resp = requests.get(f"{url}/api/ps", timeout=3, allow_redirects=False)
        refuse_redirect(resp)
        loaded = resp.json().get("models", [])
    except (requests.RequestException, ValueError, AttributeError):
        return False
    names = {m.get("name") or m.get("model") for m in loaded if isinstance(m, dict)}
    return model in names or f"{model}:latest" in names


def check_vision(rater: VisionRater) -> None:
    """Before a request that may load ``rater``'s vision model; see the module doc.

    Raises:
        LowMemoryError: If the memory it needs is not free, also after freeing cached models.
    """
    from cutpoint.assistant.providers import host_port, is_consent_exempt  # noqa: PLC0415

    if not rater.url or not is_consent_exempt(rater.url):
        return  # on another machine, a hosted provider or the subscription proxy: no memory here
    if rater.via is not None:  # a server on this machine Cutpoint cannot unload (Q6)
        host = host_port(rater.url)
        if host not in _WARNED and headroom_gb() > 0 and ram_free() is not None:
            _WARNED.add(host)
            logger.warning(
                "Vision server %s (%s) cannot be unloaded by Cutpoint: the memory guard only "
                "checks that %g GB of RAM stay free; close it between runs to give its memory back",
                rater.provider_name,
                host,
                headroom_gb(),
            )
        _guard(f"the vision server at {host}", 0.0, None)
        return
    size = rater.size / GIB if rater.size else NEEDS["vlm"][0]
    _guard(
        f"the vision model {rater.model}",
        size,
        None,
        pooled=True,
        loaded=lambda: _ollama_loaded(rater.url, rater.model),
        url=rater.url,
    )
