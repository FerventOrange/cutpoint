"""faster-whisper transcription with CUDA library setup and CPU fallback."""

from __future__ import annotations

import ctypes
import gc
import importlib.util
import logging
import os
import re
import sys
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cutpoint.extras import install_hint
from cutpoint.ffmpeg_utils import pcm_float

logger = logging.getLogger(__name__)

Segment = dict[str, Any]  # {"start", "end", "text", "words": [{"start","end","word"}], ...}

_dll_dirs_added = False
_cpu_notice_logged = False
_torch_loaded = False

# Libraries ctranslate2 needs for CUDA inference (the 'gpu' extra's wheels or a CUDA
# 12 / cuDNN 9 install). cuDNN's front library alone is not enough: it loads the
# ops/cnn parts lazily, so probe those.
_CUDA_LIBS = {
    "win32": ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"),
    "linux": ("libcublas.so.12", "libcublasLt.so.12", "libcudnn_ops.so.9", "libcudnn_cnn.so.9"),
}


def _load_torch_first() -> None:
    """Windows: import torch, if installed, before ctranslate2 or the CUDA libraries load (once).

    torch bundles its own cuDNN; once the gpu extra's newer cuDNN is loaded, ``import torch``
    fails with WinError 127 (#215). The other order works, and whisper then runs on torch's
    cuDNN. Other platforms skip it (no clash; importing torch is slow). A broken torch must
    not break transcription.
    """
    global _torch_loaded
    if _torch_loaded:
        return
    _torch_loaded = True
    if sys.platform != "win32" or importlib.util.find_spec("torch") is None:
        return
    try:
        import torch  # noqa: F401, PLC0415 - optional and slow; only before the CUDA probe
    except Exception as err:  # noqa: BLE001 - any torch failure: transcribe without it
        logger.debug("Importing torch before ctranslate2 failed: %s", err)


def _add_cuda_dll_dirs() -> None:
    """Expose pip-installed cuBLAS/cuDNN libraries (nvidia-*-cu12 wheels) to ctranslate2."""
    global _dll_dirs_added
    if _dll_dirs_added:
        return
    _dll_dirs_added = True
    if sys.platform.startswith("linux"):
        _preload_cuda_libs()
    if sys.platform != "win32":
        return
    for base in map(Path, sys.path):
        nvidia = base / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            # ctranslate2 loads some DLLs lazily via LoadLibrary, which uses PATH.
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def _preload_cuda_libs() -> None:
    """Linux: load the wheels' cuBLAS/cuDNN globally (best effort, silent).

    ctranslate2 opens them by soname, and ``site-packages/nvidia/*/lib`` is not
    on the loader path; once loaded with ``RTLD_GLOBAL`` they resolve by soname.
    """
    libs = [
        lib
        for base in map(Path, sys.path)
        for pattern in ("*/lib/libcublas*.so.12", "*/lib/libcudnn*.so.9")
        for lib in sorted((base / "nvidia").glob(pattern))
    ]
    # Some depend on others (cuBLAS on cuBLASLt): retry until a pass loads nothing new.
    while libs:
        failed = []
        for lib in libs:
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(lib)
        if len(failed) == len(libs):
            break
        libs = failed


def _missing_cuda_libs() -> list[str]:
    """CUDA libraries for whisper that cannot be loaded (searched like ctranslate2 does)."""
    missing = []
    for name in _CUDA_LIBS.get("linux" if sys.platform.startswith("linux") else sys.platform, ()):
        try:
            if sys.platform == "win32":
                # winmode=0: plain LoadLibrary search (incl. PATH), as ctranslate2 uses.
                ctypes.WinDLL(name, winmode=0)
            else:
                ctypes.CDLL(name)
        except OSError:
            missing.append(name)
    return missing


def cuda_available() -> bool:
    """True if ctranslate2 sees a CUDA device and the CUDA libraries for it load."""
    global _cpu_notice_logged
    _load_torch_first()
    _add_cuda_dll_dirs()
    try:
        import ctranslate2  # noqa: PLC0415 - after the DLL dirs are added

        if ctranslate2.get_cuda_device_count() == 0:
            return False
    except Exception:  # noqa: BLE001 - any import/driver issue means "no CUDA"
        return False
    missing = _missing_cuda_libs()
    if missing:
        if not _cpu_notice_logged:
            _cpu_notice_logged = True
            logger.info(
                "NVIDIA GPU found, but the CUDA libraries for whisper are not installed "
                "(%s); transcribing on the CPU. %s",
                ", ".join(missing),
                _cuda_libs_hint(),
            )
        return False
    return True


def load_wav(path: Path) -> Any:
    """Load a 16 kHz mono 16-bit WAV as float32 (bypasses PyAV decoding)."""
    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError(f"{path} must be 16 kHz mono 16-bit PCM")
        pcm = wf.readframes(wf.getnframes())
    return pcm_float(pcm)


def _cuda_libs_hint() -> str:
    """How to add the CUDA 12 cuBLAS / cuDNN 9 libraries (the 'gpu' extra)."""
    return f"For GPU transcription add the 'gpu' extra: {install_hint('gpu')}"


class Transcriber:
    """Lazily-loaded WhisperModel that falls back to CPU int8 on CUDA failure."""

    def __init__(self, model: str | None = None, language: str | None = None) -> None:
        self.device = "cuda" if cuda_available() else "cpu"
        self.model_name = model or ("medium" if self.device == "cuda" else "small")
        self.language = language
        self._model: Any = None

    def _load(self) -> Any:
        from faster_whisper import WhisperModel  # noqa: PLC0415

        if self._model is None:
            compute = "float16" if self.device == "cuda" else "int8"
            logger.info("Loading whisper %s on %s (%s)", self.model_name, self.device, compute)
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=compute)
        return self._model

    def _fallback_to_cpu(self, err: Exception) -> None:
        text = str(err).lower()
        if any(k in text for k in ("cublas", "cudnn", "cudart", ".dll", ".so")):
            logger.warning(
                "CUDA libraries for whisper failed to load (%s); falling back to "
                "CPU int8 (much slower). Make sure the 'gpu' extra is installed (%s) "
                "and the NVIDIA driver supports CUDA 12",
                err,
                install_hint("gpu"),
            )
        else:
            logger.warning("CUDA transcription failed (%s); falling back to CPU int8", err)
        self.device = "cpu"
        self.close()

    def transcribe(
        self, wav: Path, on_progress: Callable[[float], None] | None = None
    ) -> list[Segment]:
        """Transcribe ``wav`` with word timestamps; ``on_progress`` gets the share of the
        audio transcribed so far (0-1) as each segment is decoded."""
        try:
            return self._run(wav, on_progress)
        except (RuntimeError, OSError) as err:
            if self.device != "cuda":
                raise
            self._fallback_to_cpu(err)
            return self._run(wav, on_progress)

    def _run(self, wav: Path, on_progress: Callable[[float], None] | None = None) -> list[Segment]:
        model = self._load()
        segments, info = model.transcribe(
            load_wav(wav),
            language=self.language,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
            beam_size=5,
        )
        result: list[Segment] = []
        total = float(getattr(info, "duration", 0) or 0)
        t0, next_pct = time.perf_counter(), 10
        logger.info(
            "Transcribing %s (%.0fs audio, whisper %s/%s)",
            wav.name,
            total,
            self.model_name,
            self.device,
        )
        for seg in segments:  # generator: decoding happens here
            if total >= 600 and seg.end / total * 100 >= next_pct:
                pct = int(seg.end / total * 100)
                logger.info("  transcribe: %d%% (%.0fs elapsed)", pct, time.perf_counter() - t0)
                next_pct = (pct // 10 + 1) * 10
            if on_progress is not None and total > 0:
                on_progress(min(1.0, seg.end / total))
            result.append(
                {
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                    "avg_logprob": round(float(seg.avg_logprob), 3),
                    "no_speech_prob": round(float(seg.no_speech_prob), 3),
                    "compression_ratio": round(float(seg.compression_ratio), 3),
                    "words": [
                        {
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "word": w.word.strip(),
                            "probability": round(float(w.probability), 3),
                        }
                        for w in (seg.words or [])
                    ],
                }
            )
        return result

    def close(self) -> None:
        """Release the model (frees VRAM before e.g. the vision model loads)."""
        self._model = None
        gc.collect()


def transcript_name(model: str, language: str | None = None) -> str:
    """Cache file name of a transcript: ``transcript-<model>.json`` when Whisper detected the
    language, ``transcript-<model>-<language>.json`` when ``language`` forced it (#371)."""
    return f"transcript-{model}-{language.lower()}.json" if language else f"transcript-{model}.json"


def flat_words(segments: list[Segment]) -> list[dict[str, Any]]:
    """Flatten segments into one chronological word list."""
    return [w for s in segments for w in s.get("words", [])]


# Thresholds for dropping likely hallucinations (see filter_hallucinations).
MIN_AVG_LOGPROB = -1.2
NO_SPEECH_PROB = 0.6
NO_SPEECH_LOGPROB = -0.7
MAX_COMPRESSION = 2.4
_SHORT_PHRASE_WORDS = 3


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).strip()


def _is_repeat_loop(text: str) -> bool:
    """True for text that is one short phrase repeated, e.g. "Yeah. Yeah. Yeah."."""
    toks = _norm_text(text).split()
    for unit in range(1, _SHORT_PHRASE_WORDS + 1):
        reps = len(toks) // unit
        if reps >= 4 and len(toks) % unit == 0 and toks == toks[:unit] * reps:
            return True
    return False


def filter_hallucinations(segments: list[Segment]) -> tuple[list[Segment], int]:
    """Drop transcript segments that look like whisper hallucinations.

    A segment is dropped when it:
      * is one short phrase (<= 3 words) repeated 4+ times inside the segment;
      * repeats the previous segment's short text for the 3rd+ time in a row
        (the first two of such a run are kept);
      * has ``avg_logprob`` below ``MIN_AVG_LOGPROB``, or ``no_speech_prob``
        above ``NO_SPEECH_PROB`` together with ``avg_logprob`` below
        ``NO_SPEECH_LOGPROB``, or ``compression_ratio`` above
        ``MAX_COMPRESSION`` (only for transcripts that stored these fields;
        older cached transcripts get the repetition checks only).

    Returns:
        The kept segments and the number dropped.
    """
    kept: list[Segment] = []
    run_text, run_len = "", 0
    for seg in segments:
        text = _norm_text(seg.get("text", ""))
        short = 0 < len(text.split()) <= _SHORT_PHRASE_WORDS
        run_len = run_len + 1 if short and text == run_text else 1
        run_text = text
        logprob = seg.get("avg_logprob")
        no_speech = seg.get("no_speech_prob")
        compression = seg.get("compression_ratio")
        if _is_repeat_loop(seg.get("text", "")) or (short and run_len >= 3):
            continue
        if logprob is not None and (
            logprob < MIN_AVG_LOGPROB
            or (
                no_speech is not None and no_speech > NO_SPEECH_PROB and logprob < NO_SPEECH_LOGPROB
            )
        ):
            continue
        if compression is not None and compression > MAX_COMPRESSION:
            continue
        kept.append(seg)
    return kept, len(segments) - len(kept)
