"""Whisper loads from local files first: no Hub request with the model cached (#651).

Fakes only: a fake ``faster_whisper`` (no ctranslate2, no weights) over the real
huggingface_hub cache lookup, with every socket call patched to raise.
"""

from __future__ import annotations

import logging
import socket
import sys
import types
from pathlib import Path

import pytest
from huggingface_hub import constants as hf_constants
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

from conftest import envelope, messages, write
from cutpoint import cli, transcribe
from cutpoint.events import setup

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def repo_id(name: str) -> str:
    return name if "/" in name else f"Systran/faster-whisper-{name}"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any connection attempt fails the test."""

    def refuse(*_a, **_k):
        raise AssertionError("network access")

    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture
def hf_cache(tmp_path, monkeypatch) -> Path:
    cache = tmp_path / "hf"
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(cache))
    return cache


def cache_model(cache: Path, name: str) -> Path:
    """A cached snapshot of ``name`` as huggingface_hub lays it out."""
    folder = cache / f"models--{repo_id(name).replace('/', '--')}"
    write(folder / "refs" / "main", COMMIT)
    snap = folder / "snapshots" / COMMIT
    for f in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
        write(snap / f)
    return snap


class Fakes:
    """A fake faster_whisper: WhisperModel resolves the model as the real one does (a folder
    as is, else the Hub cache); a download (``local_files_only`` False) is only recorded."""

    def __init__(self, tmp_path: Path) -> None:
        self.loads: list[dict] = []  # WhisperModel(...) calls
        self.downloads: list[str] = []  # names fetched with the Hub allowed
        self.download_error: Exception | None = None
        self.load_error: Exception | None = None
        self.fail_on: set[str] = set()  # devices whose transcribe() raises a CUDA error
        self.root = tmp_path / "downloaded"

    def download_model(self, name: str, local_files_only: bool = False, **_kw) -> str:
        if local_files_only:  # the real cache lookup, which never touches the network
            return snapshot_download(repo_id(name), local_files_only=True)
        self.downloads.append(name)
        if self.download_error is not None:
            raise self.download_error
        return str(self.root / name)

    def module(self) -> types.ModuleType:
        fakes = self

        class WhisperModel:
            def __init__(
                self, name, device="auto", compute_type="default", local_files_only=False, **kw
            ):
                call = {"name": name, "device": device, "local_files_only": local_files_only}
                fakes.loads.append(call)
                if fakes.load_error is not None:
                    raise fakes.load_error
                path = Path(name)
                call["path"] = (
                    str(path)
                    if path.is_dir()
                    else fakes.download_model(name, local_files_only=local_files_only)
                )
                self.device = device

            def transcribe(self, _audio, **_kw):
                if self.device in fakes.fail_on:
                    raise RuntimeError("Library cublas64_12.dll is not found")
                return iter(()), types.SimpleNamespace(duration=1.0)

        mod = types.ModuleType("faster_whisper")
        mod.WhisperModel = WhisperModel
        utils = types.ModuleType("faster_whisper.utils")
        utils.download_model = self.download_model
        mod.utils = utils
        return mod


@pytest.fixture
def fakes(tmp_path, monkeypatch, hf_cache) -> Fakes:
    f = Fakes(tmp_path)
    mod = f.module()
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", mod.utils)
    monkeypatch.setattr(transcribe, "cuda_available", lambda: False)
    monkeypatch.setattr(transcribe, "load_wav", lambda _p: None)
    return f


def info(caplog) -> list[str]:
    return messages(caplog, "INFO", transcribe.logger.name)


# ------------------------------------------------------------------------------ loads
def test_cached_model_loads_local_only_without_network(fakes, hf_cache, caplog):
    snap = cache_model(hf_cache, "small")
    with caplog.at_level(logging.INFO, logger=transcribe.logger.name):
        transcribe.Transcriber()._load()
    assert fakes.loads == [
        {"name": "small", "device": "cpu", "local_files_only": True, "path": str(snap)}
    ]
    assert fakes.downloads == []
    assert not any("Downloading" in m for m in info(caplog))


def test_missing_model_is_downloaded_once_with_an_info_line(fakes, caplog):
    with caplog.at_level(logging.INFO, logger=transcribe.logger.name):
        transcribe.Transcriber()._load()
    assert [c["local_files_only"] for c in fakes.loads] == [True, False]
    assert fakes.downloads == ["small"]
    assert (
        "Downloading Whisper small (~480 MB) on first use "
        "(cutpoint setup --only whisper fetches it ahead of time)"
    ) in info(caplog)


def test_partial_cache_is_completed_by_one_download(fakes, hf_cache, monkeypatch):
    def incomplete(*_a, **_k):  # what huggingface_hub raises for a partly downloaded snapshot
        from huggingface_hub.errors import IncompleteSnapshotError  # noqa: PLC0415

        raise IncompleteSnapshotError("1 file(s) are missing (model.bin)", snapshot_path="x")

    monkeypatch.setattr(sys.modules[__name__], "snapshot_download", incomplete)
    transcribe.Transcriber()._load()
    assert [c["local_files_only"] for c in fakes.loads] == [True, False]
    assert fakes.downloads == ["small"]


def test_other_model_name_names_its_setup_command(fakes, caplog):
    with caplog.at_level(logging.INFO, logger=transcribe.logger.name):
        transcribe.Transcriber(model="org/custom-ct2")._load()
    assert fakes.downloads == ["org/custom-ct2"]
    assert (
        "Downloading Whisper org/custom-ct2 on first use (cutpoint setup --only whisper "
        "--whisper-model org/custom-ct2 fetches it ahead of time)"
    ) in info(caplog)


def test_failed_download_is_raised_not_retried(fakes):
    fakes.download_error = LocalEntryNotFoundError("offline")
    with pytest.raises(LocalEntryNotFoundError):
        transcribe.Transcriber()._load()
    assert len(fakes.loads) == 2 and fakes.downloads == ["small"]


def test_corrupt_model_is_raised_without_a_download(fakes, hf_cache):
    cache_model(hf_cache, "small")
    fakes.load_error = RuntimeError("Unable to open file 'model.bin'")
    with pytest.raises(RuntimeError, match=r"model\.bin"):
        transcribe.Transcriber()._load()
    assert len(fakes.loads) == 1 and fakes.downloads == []


def test_model_folder_loads_as_given(fakes, tmp_path):
    folder = write(tmp_path / "my-model" / "model.bin").parent
    transcribe.Transcriber(model=str(folder))._load()
    assert fakes.loads == [
        {"name": str(folder), "device": "cpu", "local_files_only": True, "path": str(folder)}
    ]
    assert fakes.downloads == []


# ------------------------------------------------------------------------------ CPU fallback
def test_cpu_fallback_loads_local_only(fakes, hf_cache, monkeypatch, tmp_path):
    monkeypatch.setattr(transcribe, "cuda_available", lambda: True)
    cache_model(hf_cache, "medium")
    cache_model(hf_cache, "small")  # without --model the CPU fallback runs small (#712)
    fakes.fail_on = {"cuda"}
    tr = transcribe.Transcriber()
    assert tr.transcribe(tmp_path / "a.wav") == []
    assert [(c["name"], c["device"], c["local_files_only"]) for c in fakes.loads] == [
        ("medium", "cuda", True),
        ("small", "cpu", True),
    ]
    assert fakes.downloads == [] and tr.device == "cpu"


def test_hub_failure_on_cuda_does_not_retry_on_the_cpu(fakes, monkeypatch, tmp_path):
    monkeypatch.setattr(transcribe, "cuda_available", lambda: True)
    fakes.download_error = LocalEntryNotFoundError("offline")
    with pytest.raises(LocalEntryNotFoundError):
        transcribe.Transcriber().transcribe(tmp_path / "a.wav")
    assert [c["device"] for c in fakes.loads] == ["cuda", "cuda"]
    assert fakes.downloads == ["medium"]


# ------------------------------------------------------------------------------ setup
@pytest.mark.parametrize(("cuda", "want"), [(True, "medium"), (False, "small")])
def test_setup_whisper_downloads_the_device_default(fakes, monkeypatch, cuda, want):
    monkeypatch.setattr(transcribe, "cuda_available", lambda: cuda)
    assert cli.main(["setup", "--only", "whisper"]) == 0
    assert fakes.downloads == [want]


def test_setup_whisper_model_option(fakes):
    assert cli.main(["setup", "--only", "whisper", "--whisper-model", "large-v3"]) == 0
    assert fakes.downloads == ["large-v3"]


def test_setup_default_includes_whisper():
    assert "whisper" in setup.COMPONENTS
    ap = cli.build_parser()
    assert "whisper" in ap.parse_args(["setup"]).only


def test_setup_whisper_needs_no_torch(fakes, monkeypatch):
    """Cutpoint lite: no torch or other events modules."""
    for name in setup.EVENTS_MODULES:
        monkeypatch.setitem(sys.modules, name, None)  # import raises ImportError
    assert cli.main(["setup", "--only", "whisper"]) == 0
    assert fakes.downloads == ["small"]


@pytest.mark.parametrize("cached", [True, False], ids=["found", "downloaded"])
def test_setup_json_reports_the_whisper_model(fakes, hf_cache, capsys, cached):
    if cached:
        cache_model(hf_cache, "small")
    assert cli.main(["setup", "--only", "whisper", "--json"]) == 0
    doc = envelope(capsys.readouterr().out)
    assert doc["ok"] is True
    # the fake download returns its own folder: setup reports the path fetch_model gave
    assert doc["outputs"]["components"] == {
        "whisper": {
            "model": "small",
            "path": str(fakes.root / "small"),
            "status": "found" if cached else "downloaded",
        }
    }
    assert fakes.downloads == ["small"]  # setup may refresh a cached model


def test_setup_json_failed_whisper_step_is_missing(fakes, capsys):
    fakes.download_error = LocalEntryNotFoundError("offline")
    assert cli.main(["setup", "--only", "whisper", "--json"]) == 1
    doc = envelope(capsys.readouterr().out)
    assert doc["ok"] is False
    assert doc["outputs"]["components"]["whisper"] == {
        "model": None,
        "path": None,
        "status": "missing",
    }


def test_setup_whisper_model_without_whisper_component_is_bad_args(fakes, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["setup", "--only", "voice", "--whisper-model", "large-v3", "--json"])
    assert exc.value.code == 2
    doc = envelope(capsys.readouterr().out)
    assert doc["error"]["code"] == "bad_args"
    assert "add whisper to --only" in doc["error"]["message"]
    assert fakes.downloads == []
