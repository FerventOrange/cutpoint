"""Tests for optional-dependency messages (extras.py) and how cli.main reports them."""

from __future__ import annotations

import argparse
import json
import logging
from types import SimpleNamespace

import pytest

from cutpoint import cli, extras


def test_missing_module_message():
    # core dependencies since the ui, keyring and ski extras became empty aliases
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


def _editable_url(clone) -> str:
    return json.dumps({"url": clone.as_uri(), "dir_info": {"editable": True}})


class _Dist:
    def __init__(self, direct_url: str | None):
        self._direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return self._direct_url


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


# --- install_hint variants ---------------------------------------------------------------

RECEIPT = """\
[tool]
requirements = [{ name = "cutpoint", extras = ["all"] }]
[tool.options]
index = [{ url = "https://download.pytorch.org/whl/cu126", explicit = false, default = false, format = "simple", authenticate = "auto" }]
index-strategy = "unsafe-best-match"
"""


def _set_install(monkeypatch, tmp_path, *, receipt=None, direct_url=None, uv=False):
    """Fake the environment: sys.prefix, tool receipt, dist metadata and uv detection."""
    monkeypatch.setattr(extras.sys, "prefix", str(tmp_path))
    if receipt is not None:
        (tmp_path / "uv-receipt.toml").write_text(receipt, encoding="utf-8")
    monkeypatch.setattr(extras.importlib.metadata, "distribution", lambda name: _Dist(direct_url))
    monkeypatch.setattr(extras, "uv_managed", lambda: uv)


_RERUN = f"{extras.INSTALL_SH} (Linux, macOS) or {extras.INSTALL_PS1} (Windows PowerShell)"


@pytest.mark.parametrize(
    "receipt",
    [
        RECEIPT,
        "[tool\nnot toml {{{",  # an unreadable receipt still marks a tool install
    ],
    ids=["pypi", "malformed"],
)
@pytest.mark.parametrize(
    ("edition", "lite"),
    [("lite", " without --lite / -Lite and with CUTPOINT_LITE unset"), ("full", "")],
)
@pytest.mark.parametrize("extra", ["events", "gpu"])
def test_install_hint_uv_tool_reruns_the_installer(
    monkeypatch, tmp_path, receipt, edition, lite, extra
):
    # uv tool install without --torch-backend pulls CUDA torch on a CPU-only machine; the
    # installer picks the backend for this machine
    _set_install(monkeypatch, tmp_path, receipt=receipt, uv=True)
    monkeypatch.setattr(extras, "edition", lambda: edition)
    assert extras.install_hint(extra) == f"re-run the installer{lite}: {_RERUN}"


def _receipt_with(source: str) -> str:
    return f'[tool]\nrequirements = [{{ name = "cutpoint", {source} }}]\n'


@pytest.mark.parametrize(
    ("source", "notes"),
    [
        (
            'git = "https://github.com/FerventOrange/cutpoint?rev=main"',
            "add --source git@main, or set CUTPOINT_SOURCE=git@main for irm | iex",
        ),
        (
            'git = "https://github.com/FerventOrange/cutpoint?tag=v1#subdirectory=pkg"',
            "add --source git@v1, or set CUTPOINT_SOURCE=git@v1 for irm | iex",
        ),
        (
            'git = "https://github.com/FerventOrange/cutpoint", branch = "dev"',
            "add --source git@dev, or set CUTPOINT_SOURCE=git@dev for irm | iex",
        ),
        (
            'git = "https://github.com/FerventOrange/cutpoint"',
            "add --source git, or set CUTPOINT_SOURCE=git for irm | iex",
        ),
        (
            'directory = "C:/src/cutpoint"',
            'add --source "C:/src/cutpoint", or set CUTPOINT_SOURCE="C:/src/cutpoint" for irm | iex',
        ),
        (
            'path = "C:/dl/cutpoint-1.0-py3-none-any.whl"',
            'add --source "C:/dl/cutpoint-1.0-py3-none-any.whl", or set '
            'CUTPOINT_SOURCE="C:/dl/cutpoint-1.0-py3-none-any.whl" for irm | iex',
        ),
        (
            'editable = "C:/src/cutpoint"',
            'add --source "C:/src/cutpoint", or set CUTPOINT_SOURCE="C:/src/cutpoint" for irm | '
            "iex; the installer does not make an editable install",
        ),
        ('extras = ["all"]', None),  # PyPI: the installer's default source
    ],
    ids=["git-rev", "git-tag", "git-branch-key", "git", "directory", "path", "editable", "pypi"],
)
def test_installer_hint_keeps_the_tool_source(monkeypatch, tmp_path, source, notes):
    # the bare installer installs the PyPI release, which would replace a git or local install
    _set_install(monkeypatch, tmp_path, receipt=_receipt_with(source), uv=True)
    monkeypatch.setattr(extras, "edition", lambda: "full")
    want = f"re-run the installer: {_RERUN}" + (f" ({notes})" if notes else "")
    assert extras.install_hint("events") == want
    cuda = "; ".join(
        [*([notes] if notes else []), "it installs CUDA torch when it finds an NVIDIA GPU"]
    )
    assert extras.torch_cuda_hint() == f"re-run the installer: {_RERUN} ({cuda})"


def test_installer_one_liners():
    raw = "https://raw.githubusercontent.com/FerventOrange/cutpoint/main"
    assert f"curl -LsSf {raw}/install.sh | sh" == extras.INSTALL_SH
    assert f"irm {raw}/install.ps1 | iex" == extras.INSTALL_PS1


def test_install_hint_editable_uv_outside_clone_venv(monkeypatch, tmp_path):
    clone = tmp_path / "clone"
    _set_install(monkeypatch, tmp_path, direct_url=_editable_url(clone), uv=True)
    monkeypatch.setattr(extras.sys, "prefix", str(tmp_path / "other-venv"))
    monkeypatch.setattr(extras.sys, "executable", "/the/python")
    hint = extras.install_hint("gpu")
    assert hint == f'uv pip install --python "/the/python" -e "{clone.as_posix()}[gpu]"'


def test_install_hint_not_installed_counts_as_source_tree(monkeypatch, tmp_path):
    _set_install(monkeypatch, tmp_path, uv=True)

    def missing(name):
        raise extras.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(extras.importlib.metadata, "distribution", missing)
    assert extras.install_hint("events") == extras.UV_SYNC


@pytest.mark.parametrize("direct_url", [None, '{"dir_info": {}}', "not json"])
def test_install_hint_regular_install(monkeypatch, tmp_path, direct_url):
    _set_install(monkeypatch, tmp_path, direct_url=direct_url, uv=False)
    hint = extras.install_hint("gpu")
    assert "-m pip install" in hint and 'pip install "cutpoint[gpu]"' in hint
    assert " -e " not in hint
    _set_install(monkeypatch, tmp_path, direct_url=direct_url, uv=True)
    hint = extras.install_hint("gpu")
    assert hint.startswith("uv pip install --python") and hint.endswith('"cutpoint[gpu]"')


# --- _python: shell prefix for sys.executable ---------------------------------------------


@pytest.mark.parametrize(
    ("exe", "os_name", "expected"),
    [
        ("/usr/bin/python3", "posix", ("/usr/bin/python3", "")),
        ("C:/Python/python.exe", "nt", ("C:/Python/python.exe", "")),
        (
            "C:/Program Files/Python/python.exe",
            "nt",
            (
                '& "C:/Program Files/Python/python.exe"',
                "PowerShell: in cmd drop the leading &",
            ),
        ),
        ("/opt/my python/bin/python", "posix", ('"/opt/my python/bin/python"', "")),
    ],
)
def test_python_prefix(monkeypatch, tmp_path, exe, os_name, expected):
    monkeypatch.setattr(extras.sys, "executable", exe)
    monkeypatch.setattr(extras, "os", SimpleNamespace(name=os_name))
    assert extras._python() == expected
    prefix, note = expected  # and the pip hint carries it, with the shell note
    _set_install(monkeypatch, tmp_path, direct_url=None, uv=False)
    want = f'{prefix} -m pip install "cutpoint[gpu]"' + (f" ({note})" if note else "")
    assert extras.install_hint("gpu") == want


# --- install_hint: non-editable installs (#34) --------------------------------------------


@pytest.mark.parametrize("kind", ["file", "git"])
def test_install_hint_non_editable_reinstalls_from_origin(monkeypatch, tmp_path, kind):
    if kind == "file":
        src = tmp_path / "src"
        direct_url = {"url": src.as_uri(), "dir_info": {}}
        expected = f'"{src.as_posix()}[gpu]"'
    else:
        direct_url = {
            "url": "https://github.com/x/cutpoint",
            "vcs_info": {"vcs": "git", "commit_id": "abc123"},
        }
        expected = '"cutpoint[gpu] @ git+https://github.com/x/cutpoint@abc123"'
    _set_install(monkeypatch, tmp_path, direct_url=json.dumps(direct_url), uv=True)
    assert extras.install_hint("gpu").endswith(f" {expected}")


_UV_INDEX = f"--index {extras.TORCH_INDEX} --index-strategy unsafe-best-match"
_DIR = {"url": "file:///x/cutpoint", "dir_info": {}}
_GIT = {"url": "https://github.com/x/cutpoint", "vcs_info": {"vcs": "git", "commit_id": "abc"}}
_ARCHIVE = {"url": "https://example.com/cutpoint.whl", "archive_info": {}}


@pytest.mark.parametrize(
    ("uv", "direct_url", "present", "absent"),
    [
        # uv pip applies cutpoint's [tool.uv.sources] (cu126) to a directory or git install (#71)
        (True, _DIR, [], ["NVIDIA", "--index "]),
        (True, _GIT, [], ["NVIDIA", "--index "]),
        (True, _ARCHIVE, [_UV_INDEX], []),
        (False, _DIR, [f"--extra-index-url {extras.TORCH_INDEX}"], []),
    ],
)
def test_install_hint_non_editable_events_on_windows_cuda_note(
    monkeypatch, tmp_path, uv, direct_url, present, absent
):
    monkeypatch.setattr(extras.sys, "platform", "win32")
    monkeypatch.setattr(extras, "_cpu_torch", lambda: False)  # the CPU-torch notes: see below
    _set_install(monkeypatch, tmp_path, direct_url=json.dumps(direct_url), uv=uv)
    hint = extras.install_hint("events")
    assert all(p in hint for p in present)
    assert not any(a in hint for a in absent)


# --- install_hint: installed CPU torch (#50) ----------------------------------------------


@pytest.mark.parametrize(
    ("uv", "extra_flags"),
    [
        (True, " --reinstall-package torch --reinstall-package torchvision"),
        (False, " --force-reinstall"),
    ],
)
def test_install_hint_reinstalls_installed_cpu_torch(monkeypatch, tmp_path, uv, extra_flags):
    monkeypatch.setattr(extras.sys, "platform", "win32")
    _set_install(monkeypatch, tmp_path, direct_url='{"dir_info": {}}', uv=uv)
    monkeypatch.setattr(extras, "_cpu_torch", lambda: True)
    assert extra_flags in extras.install_hint("events")
    monkeypatch.setattr(extras, "_cpu_torch", lambda: False)
    assert extra_flags not in extras.install_hint("events")


@pytest.mark.parametrize("in_clone_venv", [True, False])
def test_install_hint_editable_cpu_torch(monkeypatch, tmp_path, in_clone_venv):
    # uv pip keeps an installed CPU torch, so the -e form says how to get CUDA (#96); uv sync
    # in the clone's own .venv installs the locked CUDA build
    clone = tmp_path / "clone"
    _set_install(monkeypatch, tmp_path, direct_url=_editable_url(clone), uv=True)
    monkeypatch.setattr(extras.sys, "platform", "win32")
    monkeypatch.setattr(extras, "_cpu_torch", lambda: True)
    monkeypatch.setattr(extras.sys, "prefix", str(clone / ".venv" if in_clone_venv else tmp_path))
    hint = extras.install_hint("events")
    if in_clone_venv:
        assert hint == extras.UV_SYNC
    else:
        note = " (with an NVIDIA GPU add --reinstall-package torch --reinstall-package torchvision"
        assert hint.count(note) == 1 and hint.endswith("so torch uses CUDA)")


@pytest.mark.parametrize(
    ("platform", "cpu_torch", "extra"),
    [("win32", False, "events"), ("win32", True, "gpu"), ("linux", True, "events")],
)
def test_install_hint_editable_no_reinstall_note(monkeypatch, tmp_path, platform, cpu_torch, extra):
    _set_install(monkeypatch, tmp_path, direct_url=_editable_url(tmp_path / "clone"), uv=True)
    monkeypatch.setattr(extras.sys, "platform", platform)
    monkeypatch.setattr(extras, "_cpu_torch", lambda: cpu_torch)
    assert "NVIDIA" not in extras.install_hint(extra)


@pytest.mark.parametrize(
    ("platform", "extra", "cpu_torch", "note"),
    [
        (
            "win32",
            "events",
            False,
            f"; with an NVIDIA GPU add --extra-index-url {extras.TORCH_INDEX}",
        ),
        (
            "win32",
            "events",
            True,
            f"; with an NVIDIA GPU add --extra-index-url {extras.TORCH_INDEX} --force-reinstall",
        ),
        ("win32", "gpu", True, ""),
        ("linux", "events", True, ""),
    ],
)
def test_install_hint_pip_editable_points_at_cuda_torch_index(
    monkeypatch, tmp_path, platform, extra, cpu_torch, note
):
    # pip ignores [tool.uv.sources], so a plain pip -e install gets CPU torch on Windows (#104)
    monkeypatch.setattr(extras.sys, "executable", "C:/py/python.exe")
    monkeypatch.setattr(extras, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(extras.sys, "platform", platform)
    monkeypatch.setattr(extras, "_cpu_torch", lambda: cpu_torch)
    _set_install(monkeypatch, tmp_path, direct_url=_editable_url(tmp_path / "clone"), uv=False)
    end = " so torch uses CUDA" if note else ""
    assert extras.install_hint(extra) == (
        f'C:/py/python.exe -m pip install -e ".[{extra}]" (from the repository{note}{end})'
    )
