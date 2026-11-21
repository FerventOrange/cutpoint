"""Tests for optional-dependency messages (extras.py) and how cli.main reports them."""

from __future__ import annotations

import argparse
import json
import logging
from types import SimpleNamespace

import pytest

from cutpoint import cli, extras


def test_missing_module_message():
    # core dependencies, not extras
    for core in ("telemetry_parser", "fastapi", "uvicorn", "keyring"):
        assert extras.missing_module_message(core) is None
    msg = extras.missing_module_message("torch.nn.functional")  # top-level package decides
    assert msg is not None and msg.startswith("torch is not installed")
    assert "not available in Cutpoint lite" in msg and "'events' extra" in msg
    assert extras.missing_module_message("definitely_not_an_extra") is None


# --- edition ------------------------------------------------------------------------------


def test_edition_is_full_only_with_every_events_module(monkeypatch):
    present = set(extras.MODULE_EXTRA)
    monkeypatch.setattr(
        extras.importlib.util, "find_spec", lambda name: object() if name in present else None
    )
    assert extras.edition() == "full"
    present.discard("torch")
    assert extras.edition() == "lite"
    present.clear()
    assert extras.edition() == "lite"


def test_uv_managed_under_uv_run_or_when_pip_is_missing(monkeypatch):
    monkeypatch.setenv("UV", "1")
    assert extras.uv_managed() is True
    monkeypatch.delenv("UV")
    monkeypatch.setattr(extras.importlib.util, "find_spec", lambda name: object())
    assert extras.uv_managed() is False
    monkeypatch.setattr(extras.importlib.util, "find_spec", lambda name: None)
    assert extras.uv_managed() is True


def _patch_paths_command(monkeypatch, exc: BaseException) -> None:
    def handler(args: argparse.Namespace) -> int:
        raise exc

    monkeypatch.setattr(cli, "_cmd_paths", handler)


def test_main_returns_1_for_missing_known_optional_module(monkeypatch, caplog):
    _patch_paths_command(monkeypatch, ModuleNotFoundError("No module named 'torch'", name="torch"))
    with caplog.at_level(logging.ERROR):
        assert cli.main(["paths"]) == 1
    assert any(
        "torch is not installed" in r.getMessage() and "'events' extra" in r.getMessage()
        for r in caplog.records
    )


def test_main_reraises_for_unknown_missing_module(monkeypatch):
    _patch_paths_command(
        monkeypatch, ModuleNotFoundError("No module named 'bogus_pkg'", name="bogus_pkg")
    )
    with pytest.raises(ModuleNotFoundError, match="bogus_pkg"):
        cli.main(["paths"])


# --- install_hint: one row per install kind ------------------------------------------------

RECEIPT = """\
[tool]
requirements = [{ name = "cutpoint", extras = ["all"] }]
[tool.options]
index = [{ url = "https://download.pytorch.org/whl/cu126", explicit = false, default = false, format = "simple", authenticate = "auto" }]
index-strategy = "unsafe-best-match"
"""
_NOT_INSTALLED = object()  # cutpoint has no distribution: running from a source tree
_FORK = "https://github.com/someone/cutpoint"
_UPSTREAM = "https://github.com/FerventOrange/cutpoint"
_UPSTREAM_CASED = "https://GitHub.com/ferventorange/Cutpoint"  # still the upstream repository
_RERUN = f"{extras.INSTALL_SH} (Linux, macOS) or {extras.INSTALL_PS1} (Windows PowerShell)"
_NO_LITE = " without --lite / -Lite and with CUTPOINT_LITE unset"
_REINSTALL = "--reinstall-package torch --reinstall-package torchvision"
_CUDA_TORCH = "2.14.1+cu126"
_CPU_TORCH = "2.14.1+cpu"
_TAIL = "--python 3.12 --reinstall --no-sources"


class _Dist:
    def __init__(self, direct_url: str | None):
        self._direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return self._direct_url


def _receipt_with(source: str) -> str:
    return f'[tool]\nrequirements = [{{ name = "cutpoint", {source} }}]\n'


def _rerun_with(sh_src: str, ps_src: str) -> str:
    return (
        f"re-run the installer: {extras.INSTALL_SH} -s -- --source {sh_src} (Linux, macOS) or "
        f"$env:CUTPOINT_SOURCE = {ps_src}; {extras.INSTALL_PS1} (Windows PowerShell)"
    )


def _set_install(
    monkeypatch,
    tmp_path,
    *,
    receipt=None,
    direct_url=None,
    uv=False,
    platform="linux",
    torch=None,
    edition="full",
    exe="/py/python",
    os_name="posix",
    in_clone_venv=False,
):
    """Fake the environment at its boundaries: sys.prefix with an optional tool receipt,
    cutpoint's direct_url.json (``<uri>`` is ``tmp_path`` as a file URI), the installed torch
    version, uv detection, platform, interpreter path and edition."""
    prefix = tmp_path / "clone" / ".venv" if in_clone_venv else tmp_path / "env"
    prefix.mkdir(parents=True)
    monkeypatch.setattr(extras.sys, "prefix", str(prefix))
    if receipt is not None:
        (prefix / "uv-receipt.toml").write_text(receipt, encoding="utf-8")

    def distribution(name):
        if direct_url is _NOT_INSTALLED:
            raise extras.importlib.metadata.PackageNotFoundError(name)
        uri = tmp_path.as_uri()
        return _Dist(direct_url.replace("<uri>", uri) if direct_url else direct_url)

    def from_name(name):
        if torch is None:
            raise extras.importlib.metadata.PackageNotFoundError(name)
        return SimpleNamespace(version=torch)

    monkeypatch.setattr(extras.importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(extras.importlib.metadata.Distribution, "from_name", from_name)
    monkeypatch.setattr(extras, "uv_managed", lambda: uv)
    monkeypatch.setattr(extras, "edition", lambda: edition)
    monkeypatch.setattr(extras.sys, "platform", platform)
    monkeypatch.setattr(extras.sys, "executable", exe)
    monkeypatch.setattr(extras, "os", SimpleNamespace(name=os_name))


_DIR = json.dumps({"url": "<uri>/src", "dir_info": {}})
_EDITABLE = json.dumps({"url": "<uri>/clone", "dir_info": {"editable": True}})
_GIT = json.dumps(
    {"url": "https://github.com/x/cutpoint", "vcs_info": {"vcs": "git", "commit_id": "abc123"}}
)
_ARCHIVE = json.dumps({"url": "https://example.com/cutpoint.whl", "archive_info": {}})
_PIP = '/py/python -m pip install "cutpoint[gpu]"'
_UV_PIP = 'uv pip install --python "/py/python"'


@pytest.mark.parametrize(
    ("extra", "env", "want"),
    [
        # tool install: the installer picks the torch backend for this machine (uv tool install
        # without --torch-backend pulls CUDA torch on a CPU-only machine)
        pytest.param(
            "gpu", {"receipt": RECEIPT, "edition": "lite"}, f"re-run the installer{_NO_LITE}: "
            f"{_RERUN}", id="tool-lite",
        ),
        pytest.param(  # an unreadable receipt still marks a tool install
            "gpu", {"receipt": "[tool\nnot toml {{{"}, f"re-run the installer: {_RERUN}",
            id="tool-malformed-receipt",
        ),
        # the bare installer installs the PyPI release, which would replace a git or local
        # install; sh takes options after `-s --`, PowerShell needs $env:
        pytest.param(
            "events",
            {"receipt": _receipt_with(f'git = "{_UPSTREAM}?rev=main"')},
            _rerun_with("git@main", "'git@main'"),
            id="tool-git-rev",
        ),
        pytest.param(
            "events",
            {"receipt": _receipt_with(f'git = "{_UPSTREAM_CASED}", branch = "dev"')},
            _rerun_with("git@dev", "'git@dev'"),
            id="tool-git-branch-key",
        ),
        pytest.param(  # PowerShell expands $ in double quotes: single-quoted, ' doubled
            "events",
            {"receipt": _receipt_with('directory = "//nas/share$x/Bob\'s cutpoint"')},
            _rerun_with(
                "'//nas/share$x/Bob'\"'\"'s cutpoint'", "'//nas/share$x/Bob''s cutpoint'"
            ),
            id="tool-directory-quoted",
        ),
        pytest.param(
            "events",
            {"receipt": _receipt_with('editable = "C:/src/cutpoint"')},
            _rerun_with("C:/src/cutpoint", "'C:/src/cutpoint'")
            + " (the installer does not make an editable install)",
            id="tool-editable",
        ),
        # the installers only install from github.com/FerventOrange/cutpoint, so a fork or a
        # wheel URL keeps its source with an explicit --torch-backend; the gone ui and keyring
        # extras of an old receipt are dropped
        pytest.param(
            "events",
            {
                "receipt": _receipt_with(
                    f'git = "{_FORK}#subdirectory=pkg", extras = ["gpu", "ui", "keyring"]'
                ),
                "torch": _CUDA_TORCH,
            },
            "the installer only installs from github.com/FerventOrange/cutpoint; for this git "
            f'install run: uv tool install "cutpoint[events,gpu] @ git+{_FORK}#subdirectory=pkg" '
            f"--torch-backend cu126 {_TAIL}",
            id="tool-fork-cuda",
        ),
        pytest.param(
            "events",
            {"receipt": _receipt_with('url = "https://example.com/c-1.0-py3-none-any.whl"')},
            "the installer does not install from a URL; for this URL install run: uv tool "
            'install "cutpoint[events] @ https://example.com/c-1.0-py3-none-any.whl" '
            f"--torch-backend cpu {_TAIL} (cu126 instead of cpu with an NVIDIA GPU)",
            id="tool-wheel-url-no-torch",
        ),
        # regular (non-editable) install: reinstall from where it came from
        pytest.param("gpu", {"direct_url": "not json"}, _PIP, id="pip-registry"),
        pytest.param("gpu", {"direct_url": None, "uv": True}, f'{_UV_PIP} "cutpoint[gpu]"',
                     id="uv-pip-registry"),
        pytest.param(
            "gpu",
            {"exe": "C:/Program Files/Python/python.exe", "os_name": "nt"},
            '& "C:/Program Files/Python/python.exe" -m pip install "cutpoint[gpu]" '
            "(PowerShell: in cmd drop the leading &)",
            id="pip-python-path-with-space-windows",
        ),
        pytest.param(
            "gpu", {"exe": "/opt/my python/bin/python"},
            '"/opt/my python/bin/python" -m pip install "cutpoint[gpu]"',
            id="pip-python-path-with-space-posix",
        ),
        pytest.param("gpu", {"direct_url": _DIR, "uv": True}, f'{_UV_PIP} "<dir>/src[gpu]"',
                     id="uv-pip-local-dir"),
        pytest.param(  # a uv git install applies [tool.uv.sources] too: no CUDA note
            "events", {"direct_url": _GIT, "uv": True, "platform": "win32"},
            f'{_UV_PIP} "cutpoint[events] @ git+https://github.com/x/cutpoint@abc123"',
            id="windows-uv-pip-git-commit",
        ),
        # PyPI torch is CPU-only on Windows; uv pip applies cutpoint's [tool.uv.sources]
        # (cu126) to a directory or git install (#71) but keeps an installed CPU torch (#50)
        pytest.param(
            "events",
            {"direct_url": _DIR, "uv": True, "platform": "win32", "torch": _CUDA_TORCH},
            f'{_UV_PIP} "<dir>/src[events]"',
            id="windows-uv-dir-cuda-torch",
        ),
        pytest.param(
            "events",
            {"direct_url": _DIR, "uv": True, "platform": "win32", "torch": _CPU_TORCH},
            f'{_UV_PIP} "<dir>/src[events]" (with an NVIDIA GPU add {_REINSTALL} so torch uses '
            "CUDA)",
            id="windows-uv-dir-cpu-torch",
        ),
        pytest.param(
            "events",
            {"direct_url": _ARCHIVE, "uv": True, "platform": "win32"},
            f'{_UV_PIP} "cutpoint[events] @ https://example.com/cutpoint.whl" (with an NVIDIA '
            f"GPU add --index {extras.TORCH_INDEX} --index-strategy unsafe-best-match so torch "
            "uses CUDA)",
            id="windows-uv-archive",
        ),
        pytest.param(
            "events",
            {"direct_url": _DIR, "platform": "win32", "torch": _CPU_TORCH},
            '/py/python -m pip install "<dir>/src[events]" (with an NVIDIA GPU add '
            f"--extra-index-url {extras.TORCH_INDEX} --force-reinstall so torch uses CUDA)",
            id="windows-pip-dir-cpu-torch",
        ),
        # editable install or source tree
        pytest.param("events", {"direct_url": _NOT_INSTALLED, "uv": True}, extras.UV_SYNC,
                     id="not-installed"),
        pytest.param(  # uv sync would change the clone's .venv; gpu needs no torch
            "gpu", {"direct_url": _EDITABLE, "uv": True, "platform": "win32", "torch": _CPU_TORCH},
            f'{_UV_PIP} -e "<dir>/clone[gpu]"',
            id="uv-editable-outside-clone-venv",
        ),
        pytest.param(  # uv pip keeps an installed CPU torch, so it says how to get CUDA (#96)
            "events",
            {"direct_url": _EDITABLE, "uv": True, "platform": "win32", "torch": _CPU_TORCH},
            f'{_UV_PIP} -e "<dir>/clone[events]" (with an NVIDIA GPU add {_REINSTALL} so torch '
            "uses CUDA)",
            id="windows-uv-editable-cpu-torch",
        ),
        pytest.param(  # uv sync in the clone's own .venv installs the locked CUDA build
            "events",
            {"direct_url": _EDITABLE, "uv": True, "platform": "win32", "torch": _CPU_TORCH,
             "in_clone_venv": True},
            extras.UV_SYNC,
            id="uv-editable-in-clone-venv",
        ),
        pytest.param(  # torch is a CUDA build already: no note
            "events",
            {"direct_url": _EDITABLE, "uv": True, "platform": "win32", "torch": _CUDA_TORCH},
            f'{_UV_PIP} -e "<dir>/clone[events]"',
            id="windows-uv-editable-cuda-torch",
        ),
        pytest.param(  # PyPI torch is CPU-only on Windows only: no note elsewhere
            "events", {"direct_url": _EDITABLE, "uv": True, "torch": _CPU_TORCH},
            f'{_UV_PIP} -e "<dir>/clone[events]"',
            id="linux-uv-editable-cpu-torch",
        ),
        pytest.param(  # pip ignores [tool.uv.sources]: CPU torch on Windows (#104)
            "events", {"direct_url": _EDITABLE, "platform": "win32", "torch": _CUDA_TORCH},
            '/py/python -m pip install -e ".[events]" (from the repository; with an NVIDIA GPU '
            f"add --extra-index-url {extras.TORCH_INDEX} so torch uses CUDA)",
            id="windows-pip-editable",
        ),
        pytest.param(  # gpu pulls in no torch: no note
            "gpu", {"direct_url": _EDITABLE, "platform": "win32", "torch": _CPU_TORCH},
            '/py/python -m pip install -e ".[gpu]" (from the repository)',
            id="windows-pip-editable-gpu",
        ),
        pytest.param(
            "events", {"direct_url": _EDITABLE},
            '/py/python -m pip install -e ".[events]" (from the repository)',
            id="pip-editable",
        ),
    ],
)  # fmt: skip
def test_install_hint(monkeypatch, tmp_path, extra, env, want):
    _set_install(monkeypatch, tmp_path, **env)
    assert extras.install_hint(extra) == want.replace("<dir>", tmp_path.as_posix())


@pytest.mark.parametrize(
    ("source", "want"),
    [
        pytest.param(
            'editable = "C:/src/cutpoint"',
            _rerun_with("C:/src/cutpoint", "'C:/src/cutpoint'")
            + " (the installer does not make an editable install; it installs CUDA torch when "
            "it finds an NVIDIA GPU)",
            id="installer",
        ),
        pytest.param(  # its purpose is CUDA, also without an installed torch
            'url = "https://example.com/c.whl"',
            "the installer does not install from a URL; for this URL install run: uv tool "
            f'install "cutpoint[events] @ https://example.com/c.whl" --torch-backend cu126 {_TAIL}',
            id="url",
        ),
    ],
)  # fmt: skip
def test_torch_cuda_hint_tool_install(monkeypatch, tmp_path, source, want):
    _set_install(monkeypatch, tmp_path, receipt=_receipt_with(source), uv=True)
    assert extras.torch_cuda_hint() == want
