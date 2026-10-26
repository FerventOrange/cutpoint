"""torch / CUDA and Insta360 SDK status in ``cutpoint capabilities``, the CUDA reinstall hint
per install type, and the ``360 stitch`` no-NVIDIA warning. Everything external is stubbed."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutpoint import capabilities, extras
from cutpoint.pano import cli as pano_cli
from cutpoint.pano import project, stitch

# --------------------------------------------------------------------------- torch


@pytest.fixture
def torch_stub(monkeypatch):
    """Install a fake ``torch``; call ``make(version, cuda_build, available)``."""
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda n, *a: object() if n == "torch" else real(n, *a)
    )
    monkeypatch.setattr(extras, "torch_cuda_hint", lambda: "CUDA-HINT")

    def make(version="2.6.0", build=None, available=False):
        fake = SimpleNamespace(
            __version__=version,
            version=SimpleNamespace(cuda=build),
            cuda=SimpleNamespace(is_available=lambda: available),
        )
        monkeypatch.setitem(sys.modules, "torch", fake)

    return make


def test_torch_not_installed(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None)
    monkeypatch.setattr(extras, "install_hint", lambda extra: f"HINT[{extra}]")
    info = capabilities._torch(True)
    assert info["installed"] is False
    assert info["fix"].endswith("HINT[events]")


def test_torch_import_error_is_captured(torch_stub, monkeypatch):
    torch_stub()
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` raises ImportError
    info = capabilities._torch(True)
    assert info["installed"] is True and "torch" in info["error"]
    assert info["warning"] is None


def test_torch_cpu_build_with_nvidia_warns(torch_stub):
    torch_stub(build=None)
    info = capabilities._torch(True)
    assert "CPU-only" in info["warning"]
    assert info["fix"] == "CUDA-HINT"


@pytest.mark.parametrize(
    ("build", "fix_has_hint"), [("12.6", False), ("13.0", True)], ids=["cu126", "other"]
)
def test_torch_cuda_build_without_gpu_advises_driver(torch_stub, build, fix_has_hint):
    torch_stub(build=build)
    info = capabilities._torch(True)
    assert f"CUDA {build} build" in info["warning"]
    assert info["fix"].startswith("update the NVIDIA driver")
    assert ("CUDA-HINT" in info["fix"]) is fix_has_hint


@pytest.mark.parametrize(
    ("nvidia", "build", "available"),
    [(False, None, False), (True, "12.6", True)],
    ids=["no-nvidia", "works"],
)
def test_torch_no_warning(torch_stub, nvidia, build, available):
    torch_stub(build=build, available=available)
    info = capabilities._torch(nvidia)
    assert info["installed"] and info["warning"] is None and info["fix"] is None


def test_report_probes_torch_before_transcription(monkeypatch):
    # On Windows the gpu extra's cuDNN, once loaded by the transcription probe, breaks `import torch`.
    calls = []
    for name in ("_torch", "_transcription", "_llama", "_insta360_sdk"):
        monkeypatch.setattr(capabilities, name, lambda *a, n=name: calls.append(n) or {})
    monkeypatch.setattr(capabilities, "_compiled_encoders", lambda: set())
    monkeypatch.setattr(capabilities, "find_encoder", lambda kind: (None, []))
    monkeypatch.setattr(
        capabilities, "video_encoder", lambda: SimpleNamespace(kind="cpu", codec="x", device=None)
    )
    monkeypatch.setattr(capabilities, "gpu_vendors", lambda: {"nvidia": False, "amd": False})
    monkeypatch.setattr(capabilities, "_nvidia_driver_usable", lambda: False)
    capabilities.report()
    assert calls.index("_torch") < calls.index("_transcription")


# --------------------------------------------------------------------------- Insta360 SDK


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    monkeypatch.delenv("CUTPOINT_INSTA360_SDK", raising=False)
    monkeypatch.setattr(project, "find_sdk", lambda: None)
    monkeypatch.setattr(project, "pano_config", dict)
    return tmp_path


def test_sdk_found(sdk, monkeypatch):
    exe = sdk / "MediaSDK-3.1.5-x" / "stitcher"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    monkeypatch.setenv("CUTPOINT_INSTA360_SDK", str(exe))
    info = capabilities._insta360_sdk(True)
    assert info["found"] is True and info["origin"] == "env"
    assert info["version"] == "3.1.5" and info["warning"] is None


def test_sdk_configured_path_missing(sdk, monkeypatch):
    monkeypatch.setattr(project, "pano_config", lambda: {"sdk_exe": str(sdk / "nope")})
    info = capabilities._insta360_sdk(True)
    assert info["found"] is False and info["origin"] == "config" and info["path"]


def test_sdk_malformed_pano_config(sdk, monkeypatch):
    def bad():
        raise ValueError("'pano' must be an object")

    monkeypatch.setattr(project, "pano_config", bad)
    info = capabilities._insta360_sdk(True)
    assert info["found"] is False and "must be an object" in info["error"]


def test_nvidia_present_from_adapter_or_driver(monkeypatch):
    monkeypatch.setattr(capabilities, "gpu_vendors", lambda: {"nvidia": False})
    monkeypatch.setattr(capabilities, "_nvidia_driver_usable", lambda: False)
    assert capabilities.nvidia_present() is False
    monkeypatch.setattr(capabilities, "_nvidia_driver_usable", lambda: True)
    assert capabilities.nvidia_present() is True


# --------------------------------------------------------------------------- format_report


def _data(torch_info, sdk_info):
    ok = {"kind": None, "error": "x", "codec": None, "device": None, "requested": "auto"}
    return {
        "platform": "linux",
        "gpu_vendors_present": {"nvidia": False, "amd": False, "intel": False},
        "video_encoders": {},
        "active_video_encoder": ok,
        "transcription": {"engine": "faster-whisper", "device": "cpu", "note": "n"},
        "llama_cpp": {"requested": "auto", "setup_would_choose": [], "nvidia_driver_usable": False},
        "torch": torch_info,
        "insta360_sdk": sdk_info,
    }


def test_format_report_has_torch_and_sdk_lines(sdk, torch_stub):
    torch_stub(version="2.6.0+cpu", build=None)
    text = capabilities.format_report(
        _data(capabilities._torch(True), capabilities._insta360_sdk(False))
    )
    torch_line = next(ln for ln in text.splitlines() if ln.startswith("torch:"))
    assert (
        "2.6.0+cpu" in torch_line and "CPU-only" in torch_line and "runs on the CPU" in torch_line
    )
    assert "Fix: CUDA-HINT" in text
    sdk_line = next(ln for ln in text.splitlines() if ln.startswith("Insta360 Media SDK:"))
    assert "not found" in sdk_line and sdk_line.endswith("no")
    with_gpu = capabilities.format_report(
        _data(capabilities._torch(True), capabilities._insta360_sdk(True))
    )
    sdk_line = next(ln for ln in with_gpu.splitlines() if ln.startswith("Insta360 Media SDK:"))
    assert sdk_line.endswith("yes") and capabilities.SDK_NEEDS_NVIDIA not in with_gpu
    assert capabilities.SDK_NEEDS_NVIDIA in text


# --------------------------------------------------------------------------- extras


class _Dist:
    def __init__(self, provides, requires):
        self.metadata = SimpleNamespace(get_all=lambda k: provides)
        self.requires = requires


def _fake_metadata(monkeypatch, dist, installed):
    def distribution(name):
        if name == "cutpoint":
            if dist is None:
                raise extras.importlib.metadata.PackageNotFoundError(name)
            return dist
        if name in installed:
            return object()
        raise extras.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(extras.importlib.metadata, "distribution", distribution)


def test_extras_in_use(monkeypatch):
    dist = _Dist(
        ["events", "gpu", "win", "all"],
        [
            'torch>=2; extra == "events"',
            'numpy; extra == "events"',
            'nvidia-cublas; extra == "gpu"',
            'pywin32; sys_platform == "win32" and extra == "win"',
            'rich; extra == "all"',
            "requests",
        ],
    )
    monkeypatch.setattr(extras.sys, "platform", "linux")
    _fake_metadata(monkeypatch, dist, {"torch", "numpy", "rich"})
    # gpu is missing a package; win has no requirement on linux; all is never listed
    assert extras.extras_in_use() == ["events"]
    _fake_metadata(monkeypatch, None, set())
    assert extras.extras_in_use() == []


def _receipt(req=None, **options):
    return {"tool": {"requirements": [{"name": "cutpoint", **(req or {})}], "options": options}}


def test_with_torch_index_added_for_pypi_and_wheel():
    for req in ({}, {"url": "https://x/cutpoint-1.0-py3-none-any.whl"}):
        out = extras._with_torch_index(_receipt(req, index=[{"url": "https://mirror/simple"}]))
        opts = out["tool"]["options"]
        assert opts["index"][-1] == {"url": extras.TORCH_INDEX}
        assert opts["index"][0] == {"url": "https://mirror/simple"}
        assert opts["index-strategy"] == "unsafe-best-match"


@pytest.mark.parametrize(
    "receipt",
    [
        _receipt({"editable": "/src/cutpoint"}),
        _receipt({"directory": "/src/cutpoint"}),
        _receipt({"git": "https://github.com/x/cutpoint"}),
        _receipt(index=[{"url": "https://download.pytorch.org/whl/cu128"}]),
    ],
    ids=["editable", "directory", "git", "cuda-index"],
)
def test_with_torch_index_left_alone(receipt):
    assert extras._with_torch_index(receipt) is receipt


def test_with_torch_index_explicit_pytorch_index_does_not_count():
    r = _receipt(index=[{"url": "https://download.pytorch.org/whl/cu126", "explicit": True}])
    assert "index-strategy" in extras._with_torch_index(r)["tool"]["options"]


def _install(monkeypatch, receipt=None, direct_url=None, uv=False, in_use=()):
    monkeypatch.setattr(extras, "_tool_receipt", lambda: receipt)
    monkeypatch.setattr(extras, "_direct_url", lambda: direct_url)
    monkeypatch.setattr(extras, "uv_managed", lambda: uv)
    monkeypatch.setattr(extras, "extras_in_use", lambda: list(in_use))
    monkeypatch.setattr(extras.sys, "executable", "/py/python")


def test_hint_tool_pypi_adds_cuda_index(monkeypatch):
    _install(monkeypatch, receipt=_receipt({"extras": ["gpu"]}))
    hint = extras.torch_cuda_hint()
    assert hint.startswith('uv tool install "cutpoint[events,gpu]" --reinstall')
    assert f"--index {extras.TORCH_INDEX}" in hint and "unsafe-best-match" in hint


def test_hint_clone_uv_sync_keeps_extras(monkeypatch, tmp_path):
    clone = tmp_path / "clone"
    monkeypatch.setattr(extras.sys, "prefix", str(clone / ".venv"))
    info = {"url": clone.as_uri(), "dir_info": {"editable": True}}
    _install(monkeypatch, direct_url=info, uv=True, in_use=["gpu"])
    assert extras.torch_cuda_hint() == "uv sync --extra events --extra gpu"


def test_hint_clone_other_env_uses_uv_pip(monkeypatch, tmp_path):
    monkeypatch.setattr(extras.sys, "prefix", str(tmp_path / "other"))
    clone = tmp_path / "clone"
    info = {"url": clone.as_uri(), "dir_info": {"editable": True}}
    _install(monkeypatch, direct_url=info, uv=True)
    hint = extras.torch_cuda_hint()
    assert hint.startswith("uv pip install --python") and "[events]" in hint
    assert "--reinstall-package torch" in hint


@pytest.mark.parametrize("pypi", [True, False], ids=["pypi", "clone"])
def test_hint_pip(monkeypatch, pypi):
    _install(monkeypatch, direct_url={} if pypi else None, uv=False, in_use=["gpu"])
    monkeypatch.setattr(extras, "_python", lambda: ("/py/python", ""))
    hint = extras.torch_cuda_hint()
    if pypi:
        version = importlib.metadata.version("cutpoint")
        assert hint.startswith(f'/py/python -m pip install "cutpoint[events,gpu]=={version}"')
        assert f"--extra-index-url {extras.TORCH_INDEX} --force-reinstall" in hint
    else:
        assert hint.startswith('/py/python -m pip install -e ".[events,gpu]"')
        assert "--force-reinstall" in hint


def test_hint_uv_pypi_install(monkeypatch):
    _install(monkeypatch, direct_url={}, uv=True)
    hint = extras.torch_cuda_hint()
    assert hint.startswith('uv pip install --python "/py/python" "cutpoint[events]"')
    assert "--index-strategy unsafe-best-match" in hint
    assert "--reinstall-package torch" in hint


# --------------------------------------------------------------------------- 360 stitch


@pytest.mark.parametrize("nvidia", [False, True])
def test_stitch_warns_only_without_nvidia(monkeypatch, caplog, tmp_path, nvidia):
    clip = SimpleNamespace(clip_id="001", source="a.insv")
    monkeypatch.setattr(pano_cli, "_project", lambda _p: Path(tmp_path))
    monkeypatch.setattr(project, "load", lambda _p: {"001": clip})
    monkeypatch.setattr(stitch, "stitch_clips", lambda clips, **_kw: [])
    monkeypatch.setattr(capabilities, "nvidia_present", lambda: nvidia)
    args = argparse.Namespace(
        project="p",
        clips=["001"],
        from_clips=None,
        segments=None,
        top=None,
        max_total=None,
        force=False,
        dry_run=True,
    )
    with caplog.at_level(logging.WARNING):
        pano_cli.cmd_stitch(args)
    warned = [r for r in caplog.records if r.getMessage() == capabilities.SDK_NEEDS_NVIDIA]
    assert bool(warned) is (not nvidia)
    if warned:
        assert warned[0].levelno == logging.WARNING
