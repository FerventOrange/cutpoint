"""Tests for optional-dependency messages (extras.py) and how cli.main reports them."""

from __future__ import annotations

import argparse
import logging
from types import SimpleNamespace

import pytest

from cutpoint import cli, extras


def test_missing_module_message():
    msg = extras.missing_module_message("telemetry_parser")
    assert msg is not None and msg.startswith("telemetry_parser is not installed")
    assert "'ski' extra" in msg
    msg = extras.missing_module_message("torch.nn.functional")  # top-level package decides
    assert msg is not None and msg.startswith("torch is not installed")
    assert "'events' extra" in msg
    assert extras.missing_module_message("definitely_not_an_extra") is None


class _Dist:
    def __init__(self, direct_url: str | None):
        self._direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return self._direct_url


def test_install_hint_uv_and_pip(monkeypatch, tmp_path):
    monkeypatch.setattr(extras.sys, "prefix", str(tmp_path))  # no tool receipt
    editable = _Dist('{"url": "file:///x", "dir_info": {"editable": true}}')
    monkeypatch.setattr(extras.importlib.metadata, "distribution", lambda name: editable)
    monkeypatch.setenv("UV", "1")
    assert extras.install_hint("events").startswith("uv sync --extra events")
    monkeypatch.delenv("UV")
    monkeypatch.setattr(extras.importlib.util, "find_spec", lambda name: object())
    assert "-m pip install" in extras.install_hint("events")


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


def test_install_hint_uv_tool_keeps_extras_and_index(monkeypatch, tmp_path):
    _set_install(monkeypatch, tmp_path, receipt=RECEIPT, uv=True)
    hint = extras.install_hint("ski")
    assert hint == (
        'uv tool install "cutpoint[all,ski]" --reinstall '
        "--index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match"
    )
    assert "NVIDIA" not in hint  # index already carried over


@pytest.mark.parametrize(
    ("platform", "note"),
    [
        ("win32", "(with an NVIDIA GPU add --index " + extras.TORCH_INDEX),
        ("linux", "(with an NVIDIA driver too old for CUDA 13, add --index " + extras.TORCH_INDEX),
        ("darwin", None),
    ],
)
def test_install_hint_uv_tool_gpu_note_by_platform(monkeypatch, tmp_path, platform, note):
    receipt = '[tool]\nrequirements = [{ name = "cutpoint" }]\n'
    _set_install(monkeypatch, tmp_path, receipt=receipt, uv=True)
    monkeypatch.setattr(extras.sys, "platform", platform)
    hint = extras.install_hint("events")
    assert hint.startswith('uv tool install "cutpoint[events]" --reinstall')
    if note is None:
        assert "NVIDIA" not in hint
    else:
        assert note in hint and "--index-strategy unsafe-best-match" in hint
    assert "NVIDIA" not in extras.install_hint("ski")  # not a torch extra


def test_install_hint_uv_tool_malformed_receipt_falls_back(monkeypatch, tmp_path):
    _set_install(monkeypatch, tmp_path, receipt="[tool\nnot toml {{{", uv=True)
    assert extras.install_hint("ski").startswith('uv tool install "cutpoint[ski]" --reinstall')


def test_install_hint_editable(monkeypatch, tmp_path):
    direct = '{"url": "file:///x", "dir_info": {"editable": true}}'
    _set_install(monkeypatch, tmp_path, direct_url=direct, uv=True)
    assert extras.install_hint("ski").startswith("uv sync --extra ski")
    _set_install(monkeypatch, tmp_path, direct_url=direct, uv=False)
    hint = extras.install_hint("ski")
    assert "-m pip install -e" in hint and ".[ski]" in hint


def test_install_hint_not_installed_counts_as_source_tree(monkeypatch, tmp_path):
    _set_install(monkeypatch, tmp_path, uv=True)

    def missing(name):
        raise extras.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(extras.importlib.metadata, "distribution", missing)
    assert extras.install_hint("ski").startswith("uv sync --extra ski")


@pytest.mark.parametrize("direct_url", [None, '{"dir_info": {}}', "not json"])
def test_install_hint_regular_install(monkeypatch, tmp_path, direct_url):
    _set_install(monkeypatch, tmp_path, direct_url=direct_url, uv=False)
    hint = extras.install_hint("ski")
    assert "-m pip install" in hint and 'pip install "cutpoint[ski]"' in hint
    assert " -e " not in hint
    _set_install(monkeypatch, tmp_path, direct_url=direct_url, uv=True)
    hint = extras.install_hint("ski")
    assert hint.startswith("uv pip install --python") and hint.endswith('"cutpoint[ski]"')


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
                " (PowerShell; in cmd drop the leading &)",
            ),
        ),
        ("/opt/my python/bin/python", "posix", ('"/opt/my python/bin/python"', "")),
    ],
)
def test_python_prefix(monkeypatch, exe, os_name, expected):
    monkeypatch.setattr(extras.sys, "executable", exe)
    monkeypatch.setattr(extras, "os", SimpleNamespace(name=os_name))
    assert extras._python() == expected


def test_pip_hints_use_python_prefix(monkeypatch, tmp_path):
    exe = "C:/Program Files/Python/python.exe"
    note = " (PowerShell; in cmd drop the leading &)"
    monkeypatch.setattr(extras.sys, "executable", exe)
    monkeypatch.setattr(extras, "os", SimpleNamespace(name="nt"))
    _set_install(monkeypatch, tmp_path, direct_url='{"dir_info": {"editable": true}}', uv=False)
    hint = extras.install_hint("ski")
    assert hint.startswith(f'& "{exe}" -m pip install -e') and hint.endswith(note)
    _set_install(monkeypatch, tmp_path, direct_url=None, uv=False)
    hint = extras.install_hint("ski")
    assert hint == f'& "{exe}" -m pip install "cutpoint[ski]"{note}'


# --- uv tool receipts: reinstall from the original source ---------------------------------


def _receipt_with(source: str) -> str:
    return f'[tool]\nrequirements = [{{ name = "cutpoint", {source} }}]\n'


@pytest.mark.parametrize(
    ("source", "expected_target"),
    [
        (
            'extras = ["ski"], editable = "C:/src/cutpoint"',
            '--editable "C:/src/cutpoint[events,ski]"',
        ),
        ('directory = "C:/src/cutpoint"', '"C:/src/cutpoint[events]"'),
        (
            'git = "https://github.com/FerventOrange/cutpoint?rev=main"',
            '"cutpoint[events] @ git+https://github.com/FerventOrange/cutpoint@main"',
        ),
        (
            'git = "https://github.com/FerventOrange/cutpoint?tag=v1#subdirectory=pkg"',
            '"cutpoint[events] @ git+https://github.com/FerventOrange/cutpoint@v1#subdirectory=pkg"',
        ),
        (
            'git = "https://github.com/FerventOrange/cutpoint"',
            '"cutpoint[events] @ git+https://github.com/FerventOrange/cutpoint"',
        ),
        (
            'url = "https://example.com/cutpoint-0.1.0-py3-none-any.whl"',
            '"cutpoint[events] @ https://example.com/cutpoint-0.1.0-py3-none-any.whl"',
        ),
    ],
)
def test_install_hint_uv_tool_source(monkeypatch, tmp_path, source, expected_target):
    _set_install(monkeypatch, tmp_path, receipt=_receipt_with(source), uv=True)
    hint = extras.install_hint("events")
    assert hint.startswith(f"uv tool install {expected_target} --reinstall")
