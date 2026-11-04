"""The memory guard (#109 phase 5): a model loads only when its memory is free; else cached
models are freed and, still short, the run stops with ``low_memory`` (exit 1), resumable."""

from __future__ import annotations

import ctypes
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import cutpoint.analyze
from conftest import set_config
from cutpoint import config, memguard, progress
from cutpoint.errors import LowMemoryError, LowVramError
from cutpoint.events import common
from test_resume import _Resp

GB = memguard.GIB
_RAM_FREE = memguard.ram_free  # conftest turns it off (no_host_probes); ``mem`` fakes it


# -- reading the free memory ------------------------------------------------------------------


def test_linux_reads_memavailable(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:  16000000 kB\nMemAvailable:   3145728 kB\n", encoding="utf-8")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(memguard, "Path", lambda p: meminfo if p == "/proc/meminfo" else Path(p))
    assert _RAM_FREE() == 3 * GB
    meminfo.write_text("garbage\n", encoding="utf-8")
    assert _RAM_FREE() is None


def test_windows_asks_globalmemorystatusex(monkeypatch):
    def status(ref) -> int:
        ref._obj.ullAvailPhys = 5 * GB
        assert ref._obj.dwLength == ctypes.sizeof(memguard._MemoryStatus)
        return 1

    fake = SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=status))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", fake, raising=False)
    assert _RAM_FREE() == 5 * GB


@pytest.mark.parametrize("platform", ["darwin", "freebsd14"])
def test_an_unknown_platform_turns_the_guard_off(home, monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(memguard, "ram_free", _RAM_FREE)
    assert _RAM_FREE() is None
    set_config(home, memory_guard=1000)
    memguard.check("whisper-large")  # never blocks
    memguard.check("visual", gpu=True)


def test_vram_from_nvidia_smi_without_torch_cuda(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(common, "gpu_memory_free_mib", lambda: 3072)
    assert memguard.vram_free() == 3 * GB
    monkeypatch.setattr(common, "gpu_memory_free_mib", lambda: None)
    assert memguard.vram_free() is None


def test_vram_from_torch_once_it_uses_cuda(monkeypatch):
    cuda = SimpleNamespace(is_initialized=lambda: True, mem_get_info=lambda: (2 * GB, 8 * GB))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(common, "gpu_memory_free_mib", lambda: pytest.fail("nvidia-smi asked"))
    assert memguard.vram_free() == 2 * GB


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("small", "whisper-small"),
        ("small.en", "whisper-small"),
        ("medium", "whisper-medium"),
        ("large-v3", "whisper-large"),
        ("large-v3-turbo", "whisper-turbo"),
        ("distil-large-v3", "whisper-turbo"),
        ("distil-small.en", "whisper-small"),
        (str(Path("models") / "my-whisper"), "whisper-medium"),
    ],
)
def test_whisper_needs(name, key):
    assert memguard.whisper_need(name) == key


@pytest.mark.parametrize(("value", "want"), [(None, 2.0), (0, 0.0), (3, 3.0), (1.5, 1.5)], ids=str)
def test_memory_guard_setting(home, value, want):
    set_config(home, **({} if value is None else {"memory_guard": value}))
    assert config.get_config().memory_guard == want


@pytest.mark.parametrize("value", [-1, "2", True])
def test_a_bad_memory_guard_setting_is_an_error(home, value):
    set_config(home, memory_guard=value)
    with pytest.raises(ValueError, match="memory_guard"):
        config.get_config()


# -- the check --------------------------------------------------------------------------------


def test_enough_memory_frees_nothing(mem):
    mem.ram = 2.0 + 1.0  # the default headroom and PANNs' need on the CPU
    memguard.check("panns")
    assert mem.freed == mem.unloaded == 0


def test_short_ram_frees_unloads_then_stops_with_low_memory(mem, caplog):
    mem.ram = 2.5
    with caplog.at_level(logging.INFO), pytest.raises(LowMemoryError) as exc:
        memguard.check("whisper-medium", label="Whisper medium")
    assert mem.freed == mem.unloaded == 1
    text = str(exc.value)
    assert "Whisper medium" in text and "2.5 GB RAM free, 4.0 GB needed" in text
    assert "run the command again" in text  # no progress.json to resume yet
    assert any("freeing cached models" in m for m in caplog.messages)


def test_enough_after_freeing_goes_on(mem, caplog):
    mem.ram, mem.after = 1.0, 8.0
    with caplog.at_level(logging.INFO):
        memguard.check("clip")
    assert mem.freed == mem.unloaded == 1
    assert "Memory for CLIP is free now" in caplog.messages


def test_short_vram_on_the_gpu(mem):
    mem.ram, mem.vram = 2.5, 0.4  # RAM: the headroom alone on the GPU
    with pytest.raises(LowMemoryError, match=re.escape("0.4 GB VRAM free, 1.5 GB needed")):
        memguard.check("visual", gpu=True)
    assert mem.freed == mem.unloaded == 1
    mem.vram = None  # no VRAM known: no VRAM check
    memguard.check("visual", gpu=True)


def test_memory_guard_zero_checks_nothing(home, monkeypatch):
    set_config(home, memory_guard=0)
    monkeypatch.setattr(memguard, "ram_free", lambda: pytest.fail("memory read"))
    monkeypatch.setattr(memguard, "vram_free", lambda: pytest.fail("memory read"))
    memguard.check("whisper-large")
    memguard.check("visual", gpu=True)


def test_the_headroom_is_the_setting(home, mem):
    set_config(home, memory_guard=6)
    mem.ram = 6.5
    with pytest.raises(LowMemoryError, match="memory_guard keeps 6 GB"):
        memguard.check("ocr")


# -- the vision model -------------------------------------------------------------------------


def _rater(url: str, *, via: bool = False, size: int | None = None) -> Any:
    return SimpleNamespace(
        url=url,
        via=object() if via else None,
        model="qwen3.5:9b",
        size=size,
        provider_name="openai" if via else "ollama",
    )


def test_ollama_needs_the_size_it_reports_in_ram_and_vram(mem, monkeypatch):
    monkeypatch.setattr(memguard, "_ollama_loaded", lambda url, model: False)
    mem.ram, mem.vram = 3.0, 4.0
    memguard.check_vision(_rater("http://127.0.0.1:11434", size=4 * GB))  # 7 >= 2 + 4
    with pytest.raises(LowMemoryError, match="RAM and VRAM free"):
        memguard.check_vision(_rater("http://127.0.0.1:11434", size=6 * GB))
    mem.vram = None
    with pytest.raises(LowMemoryError, match=re.escape("3.0 GB RAM free, 9.0 GB needed")):
        memguard.check_vision(_rater("http://127.0.0.1:11434"))  # no size: 7 GB


def test_a_loaded_ollama_model_needs_nothing(mem, monkeypatch):
    asked = []
    monkeypatch.setattr(memguard, "_ollama_loaded", lambda url, model: asked.append(model) or True)
    mem.ram = 1.0
    memguard.check_vision(_rater("http://localhost:11434"))
    assert asked == ["qwen3.5:9b"] and mem.freed == 0
    mem.ram = 64.0
    asked.clear()
    memguard.check_vision(_rater("http://localhost:11434"))
    assert asked == []  # Ollama is asked only when memory looks short


def test_ollama_loaded_reads_api_ps(monkeypatch):
    import requests  # noqa: PLC0415

    monkeypatch.setattr(
        requests, "get", lambda url, **kw: _Resp({"models": [{"name": "qwen3.5:9b"}]})
    )
    assert memguard._ollama_loaded("http://127.0.0.1:11434", "qwen3.5:9b")
    assert not memguard._ollama_loaded("http://127.0.0.1:11434", "llava")


def test_ollama_loaded_does_not_follow_a_redirect(monkeypatch):
    import requests  # noqa: PLC0415

    sent = []

    def moved(url, **kw):
        sent.append((url, kw.get("allow_redirects", True)))
        resp = requests.Response()
        resp.status_code = 307
        resp.headers["Location"] = "http://elsewhere.example.com/api/ps"
        return resp

    monkeypatch.setattr(requests, "get", moved)
    assert not memguard._ollama_loaded("http://127.0.0.1:11434", "qwen3.5:9b")
    assert sent == [("http://127.0.0.1:11434/api/ps", False)]


@pytest.mark.parametrize("via", [False, True], ids=["ollama", "hosted"])
def test_a_server_on_another_machine_is_not_checked(mem, via):
    mem.ram = 0.1
    memguard.check_vision(_rater("http://192.168.1.5:11434", via=via))
    memguard.check_vision(_rater("https://api.openai.com/v1", via=via))
    assert mem.freed == 0


def test_the_subscription_proxy_is_not_a_server_here(mem, monkeypatch, caplog):
    """Review F6: Cutpoint's subscription proxy forwards to hosted models."""
    from cutpoint.assistant import providers  # noqa: PLC0415

    monkeypatch.setattr(providers, "is_proxy", lambda url: True)
    mem.ram = 0.1
    with caplog.at_level(logging.WARNING):
        memguard.check_vision(_rater("http://127.0.0.1:8765", via=True))
    assert mem.freed == 0 and caplog.messages == []


def test_the_raters_ollama_is_the_one_unloaded(mem, monkeypatch):
    urls = []
    monkeypatch.setattr(common, "unload_ollama", lambda url=None: urls.append(url) or [])
    monkeypatch.setattr(memguard, "_ollama_loaded", lambda url, model: False)
    mem.ram = 1.0
    with pytest.raises(LowMemoryError):
        memguard.check_vision(_rater("http://127.0.0.1:11500"))
    with pytest.raises(LowMemoryError):
        memguard.check("ocr")
    assert urls == ["http://127.0.0.1:11500", None]


def test_a_local_custom_server_gets_the_ram_floor_and_a_warning_once(mem, caplog):
    mem.ram = 2.5
    rater = _rater("http://127.0.0.1:8080/v1", via=True)
    with caplog.at_level(logging.WARNING):
        memguard.check_vision(rater)
        memguard.check_vision(rater)
    warned = [m for m in caplog.messages if "cannot be unloaded" in m]
    assert len(warned) == 1 and "127.0.0.1:8080" in warned[0]
    mem.ram = 1.0
    with pytest.raises(LowMemoryError, match=re.escape("1.0 GB RAM free, 2.0 GB needed")):
        memguard.check_vision(rater)


def test_the_rater_checks_before_its_first_request_after_a_load(monkeypatch):
    from cutpoint import vision  # noqa: PLC0415

    checked = []
    monkeypatch.setattr(memguard, "check_vision", lambda r: checked.append(r))
    rater = vision.VisionRater("http://127.0.0.1:11434", "qwen3.5:9b")
    rater.check_memory()
    rater.used = True  # a request went out: the model is loaded
    rater.check_memory()
    assert checked == [rater]


def test_available_records_the_size_ollama_reports(monkeypatch):
    from cutpoint import vision  # noqa: PLC0415

    tags = {"models": [{"name": "qwen3.5:9b", "digest": "d1", "size": 6_600_000_000}]}
    monkeypatch.setattr(vision.requests, "get", lambda url, **kw: _Resp(tags))
    rater = vision.VisionRater("http://127.0.0.1:11434", "qwen3.5:9b")
    assert rater.available() and rater.size == 6_600_000_000


# -- where the models load --------------------------------------------------------------------


def test_whisper_short_of_vram_transcribes_on_the_cpu(mem, monkeypatch, caplog):
    """Review F1: VRAM held by another program (a game, OBS) moves Whisper to the CPU with a
    WARNING, as its CUDA failure fallback does; the run goes on."""
    from cutpoint import transcribe  # noqa: PLC0415

    loads = []
    monkeypatch.setattr(transcribe, "load_model", lambda *a: loads.append(a) or object())
    mem.ram, mem.vram = 64.0, 0.5
    tr = transcribe.Transcriber(model="large-v3")
    tr.device = tr._want = "cuda"
    with caplog.at_level(logging.WARNING):
        tr._load()
    assert tr.device == "cpu" and loads == [("large-v3", "cpu", "int8")]
    assert mem.freed == mem.unloaded == 1  # freed first: still short
    assert any("0.5 GB VRAM free, 4.0 GB needed; running on the CPU" in m for m in caplog.messages)


def test_whisper_goes_back_to_the_gpu_once_its_vram_is_free(mem, monkeypatch):
    """Review F10: a VRAM shortage moves one load to the CPU; the next load (a --batch-size
    boundary) is on the GPU again when its memory is free; a CUDA failure stays on the CPU."""
    from cutpoint import transcribe  # noqa: PLC0415

    monkeypatch.setattr(transcribe, "load_model", lambda *a: SimpleNamespace(device=a[1]))
    monkeypatch.setattr(transcribe.Transcriber, "close", lambda self: setattr(self, "_model", None))
    tr = transcribe.Transcriber(model="small")
    tr.device = tr._want = "cuda"
    mem.vram = 0.1
    assert tr._load().device == "cpu"
    tr.close()
    mem.vram = 8.0
    assert tr._load().device == "cuda" and tr.device == "cuda"
    tr._fallback_to_cpu(RuntimeError("CUDA failed"))
    assert tr._load().device == "cpu"


def test_whisper_short_of_ram_stops_without_a_cpu_fallback(mem, monkeypatch):
    from cutpoint import transcribe  # noqa: PLC0415

    monkeypatch.setattr(transcribe, "load_model", lambda *a: pytest.fail("loaded"))
    mem.ram, mem.vram = 1.0, 0.5
    tr = transcribe.Transcriber(model="large-v3")
    tr.device = tr._want = "cuda"
    with pytest.raises(LowMemoryError, match="RAM free") as exc:
        tr.transcribe(Path("a.wav"))
    assert not isinstance(exc.value, LowVramError)
    assert tr.device == "cuda"  # not the CUDA failure fallback either
    mem.ram, mem.vram = 3.0, 0.5  # VRAM short, then RAM short on the CPU (2 + 3.5)
    with pytest.raises(LowMemoryError, match="RAM free") as exc:
        tr.transcribe(Path("a.wav"))
    assert not isinstance(exc.value, LowVramError)  # the run stops: no model loaded


def test_short_vram_of_a_model_without_cpu_fallback_stops(mem):
    mem.ram, mem.vram = 64.0, 0.1
    with pytest.raises(LowVramError):
        memguard.check("visual", gpu=True)


@pytest.mark.parametrize("model", ["panns", "clip"])
def test_a_model_short_of_vram_runs_on_the_cpu(mem, model, caplog):
    """Review F8: PANNs and the crash check's CLIP run on the CPU too: VRAM held by a game or
    OBS moves them there with a WARNING; RAM short still stops the run."""
    mem.ram, mem.vram = 64.0, 0.1
    with caplog.at_level(logging.WARNING):
        assert memguard.device_for(model, "cuda") == "cpu"
    assert any("0.1 GB VRAM free" in m and "running on the CPU" in m for m in caplog.messages)
    mem.vram = 8.0
    assert memguard.device_for(model, "cuda") == "cuda"
    mem.ram, mem.vram = 1.0, 0.1
    with pytest.raises(LowMemoryError) as exc:
        memguard.device_for(model, "cuda")
    assert not isinstance(exc.value, LowVramError)
    with pytest.raises(LowMemoryError):
        memguard.device_for(model, "cpu")


def test_an_analyze_layer_step_lets_low_memory_through():
    with progress.run(progress.ANALYZE_STEPS, quiet=True) as steps:
        hook = cutpoint.analyze._StepHook(steps)
        with pytest.raises(LowMemoryError), hook("audio"):
            raise LowMemoryError("short")
        assert "reactions" not in hook.failed


# -- #712 review: the resume hint, the CPU default without --model, VRAM-only wording -------


@pytest.mark.parametrize("nt", [False, True], ids=["posix", "windows"])
def test_the_resume_hint_quotes_a_path_with_a_space(monkeypatch, tmp_path, nt):
    from cutpoint import runstate, utils  # noqa: PLC0415

    project = tmp_path / "my projects" / "p"
    monkeypatch.setattr(runstate, "resume_project", lambda: project)
    monkeypatch.setattr(utils, "os", SimpleNamespace(name="nt" if nt else "posix"))
    hint = memguard._resume_hint()
    quoted = f'"{project}"' if nt else f"'{project}'"
    assert hint == f"then run `cutpoint resume {quoted}` to continue"


def test_a_vram_stop_does_not_name_the_ram_headroom(mem):
    mem.ram, mem.vram = 64.0, 0.4
    with pytest.raises(LowVramError) as exc:
        memguard.check("visual", gpu=True)
    assert "VRAM free" in str(exc.value) and "GB of RAM free" not in str(exc.value)
    mem.ram = 2.5
    with pytest.raises(LowMemoryError, match="memory_guard keeps 2 GB of RAM free"):
        memguard.check("visual")


def _transcriber(monkeypatch, model: str | None, small_cached: bool = True) -> Any:
    from cutpoint import transcribe  # noqa: PLC0415

    cached = {"small": "hub/small"} if small_cached else {}
    monkeypatch.setattr(transcribe, "cached_model", cached.get)
    monkeypatch.setattr(transcribe, "cuda_available", lambda: True)
    monkeypatch.setattr(transcribe, "load_model", lambda *a: SimpleNamespace(args=a))
    monkeypatch.setattr(transcribe.Transcriber, "close", lambda self: setattr(self, "_model", None))
    return transcribe.Transcriber(model=model)


@pytest.mark.parametrize(
    ("model", "on_gpu", "on_cpu"),
    [(None, "medium", "small"), ("large-v3", "large-v3", "large-v3")],
    ids=["no-model", "explicit-model"],
)
def test_whisper_short_of_vram_runs_the_cpu_default_unless_a_model_is_given(
    mem, monkeypatch, caplog, model, on_gpu, on_cpu
):
    """Owner decision (#712): without --model, the CPU fallback runs small; --model is kept."""
    tr = _transcriber(monkeypatch, model)
    mem.vram = 0.1
    with caplog.at_level(logging.WARNING):
        assert tr._load().args == (on_cpu, "cpu", "int8")
    assert tr.model_name == on_cpu
    warning = next(m for m in caplog.messages if "running on the CPU" in m)
    assert warning.startswith(f"Whisper {on_gpu}:")
    assert ("the CPU default without --model" in warning) is (model is None)
    tr.close()
    mem.vram = 8.0  # free again: back on the GPU with the GPU model
    assert tr._load().args == (on_gpu, "cuda", "float16")
    assert tr.model_name == on_gpu


@pytest.mark.parametrize(
    ("model", "on_cpu"), [(None, "small"), ("large-v3", "large-v3")], ids=["no-model", "explicit"]
)
def test_a_cuda_failure_falls_back_to_the_cpu_default_unless_a_model_is_given(
    mem, monkeypatch, caplog, model, on_cpu
):
    tr = _transcriber(monkeypatch, model)
    with caplog.at_level(logging.WARNING):
        tr._fallback_to_cpu(RuntimeError("CUDA failed"))
    assert tr.model_name == on_cpu and tr._load().args == (on_cpu, "cpu", "int8")
    warning = next(m for m in caplog.messages if "falling back to CPU" in m)
    assert ("with Whisper small (the CPU default without --model)" in warning) is (model is None)


@pytest.mark.parametrize("cause", ["vram", "cuda"])
def test_without_small_downloaded_the_cpu_fallback_keeps_medium(mem, monkeypatch, caplog, cause):
    """Review F2: no download mid-run; the WARNING names the setup command for small."""
    tr = _transcriber(monkeypatch, None, small_cached=False)
    mem.vram = 0.1 if cause == "vram" else 8.0
    with caplog.at_level(logging.WARNING):
        if cause == "cuda":
            tr._fallback_to_cpu(RuntimeError("CUDA failed"))
        assert tr._load().args == ("medium", "cpu", "int8")
    assert tr.model_name == "medium"
    kept = [m for m in caplog.messages if "is not downloaded" in m]
    assert len(kept) == 1 and "--whisper-model small" in kept[0]
    assert not any("the CPU default without --model" in m for m in caplog.messages)


def test_a_cpu_fallback_transcript_is_found_by_the_next_run(mem, monkeypatch, tmp_path):
    """Review F3: the small transcript of a CPU fallback is not transcribed again."""
    from cutpoint import analyze, transcribe  # noqa: PLC0415

    runs = []

    def run(self, wav, on_progress=None):
        self._load()
        runs.append(self.model_name)
        return [{"start": 0.0, "end": 1.0, "text": "hi"}]

    monkeypatch.setattr(transcribe.Transcriber, "_run", run)
    cdir = tmp_path / "cache"
    cdir.mkdir()
    mem.vram = 0.1
    first = _transcriber(monkeypatch, None)
    assert analyze._transcript(tmp_path / "a.wav", cdir, first) == run(first, None)
    assert runs == ["small", "small"] and (cdir / "transcript-small.json").is_file()
    mem.vram = 8.0  # the GPU free again: a new run without --model keeps the small one
    later = _transcriber(monkeypatch, None)
    assert later.model_name == "medium"
    assert analyze._transcript(tmp_path / "a.wav", cdir, later)[0]["text"] == "hi"
    assert len(runs) == 2
    monkeypatch.setattr(analyze.cache, "source_dir", lambda _src: cdir)
    assert analyze._cached_transcripts([tmp_path / "a.wav"], later) == 1
    explicit = _transcriber(monkeypatch, "medium")
    assert analyze._cached_transcripts([tmp_path / "a.wav"], explicit) == 0


def test_after_a_fallback_a_cached_medium_transcript_is_still_used(mem, monkeypatch, tmp_path):
    """Review F10: a CPU fallback mid-run (small) still finds the medium transcripts."""
    from cutpoint import analyze, transcribe  # noqa: PLC0415

    runs = []

    def run(self, wav, on_progress=None):
        self._load()
        runs.append(wav.name)
        return [{"start": 0.0, "end": 1.0, "text": wav.name}]

    monkeypatch.setattr(transcribe.Transcriber, "_run", run)
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    (second / "transcript-medium.json").write_text(
        '[{"start": 0.0, "end": 1.0, "text": "cached"}]', encoding="utf-8"
    )
    mem.vram = 0.1
    tr = _transcriber(monkeypatch, None)
    analyze._transcript(tmp_path / "a.wav", first, tr)
    assert tr.model_name == "small" and runs == ["a.wav"]
    assert analyze._transcript(tmp_path / "b.wav", second, tr)[0]["text"] == "cached"
    assert runs == ["a.wav"]


@pytest.mark.parametrize("marked", [True, False], ids=["fallback", "explicit-small"])
def test_a_gpu_run_reuses_only_a_small_transcript_of_a_fallback(
    mem, monkeypatch, tmp_path, caplog, marked
):
    """Windows F3: a no-model GPU run reuses transcript-small.json only when a CPU fallback
    of its default made it (marked), with an INFO; not an explicit --model small one."""
    from cutpoint import analyze, transcribe  # noqa: PLC0415

    runs = []

    def run(self, wav, on_progress=None):
        self._load()
        runs.append(self.model_name)
        return [{"start": 0.0, "end": 1.0, "text": self.model_name}]

    monkeypatch.setattr(transcribe.Transcriber, "_run", run)
    cdir = tmp_path / "cache"
    cdir.mkdir()
    mem.vram = 0.1 if marked else 8.0
    first = _transcriber(monkeypatch, None if marked else "small")
    analyze._transcript(tmp_path / "a.wav", cdir, first)
    assert runs == ["small"]
    assert (cdir / "fallback-transcript-small.json").is_file() is marked
    mem.vram = 8.0
    later = _transcriber(monkeypatch, None)
    with caplog.at_level(logging.INFO):
        text = analyze._transcript(tmp_path / "a.wav", cdir, later)[0]["text"]
    if marked:
        assert text == "small" and runs == ["small"]
        assert any("Reusing transcript-small.json" in m for m in caplog.messages)
    else:
        assert text == "medium" and runs == ["small", "medium"]
