"""Tests for optional-dependency messages (extras.py) and how cli.main reports them."""

from __future__ import annotations

import argparse
import json
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


def _editable_url(clone) -> str:
    return json.dumps({"url": clone.as_uri(), "dir_info": {"editable": True}})


class _Dist:
    def __init__(self, direct_url: str | None):
        self._direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return self._direct_url


def test_install_hint_uv_and_pip(monkeypatch, tmp_path):
    monkeypatch.setattr(extras.sys, "prefix", str(tmp_path / ".venv"))  # no tool receipt
    editable = _Dist(_editable_url(tmp_path))
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
    direct = _editable_url(tmp_path)
    _set_install(monkeypatch, tmp_path, direct_url=direct, uv=True)
    monkeypatch.setattr(extras.sys, "prefix", str(tmp_path / ".venv"))
    assert extras.install_hint("ski").startswith("uv sync --extra ski")
    _set_install(monkeypatch, tmp_path, direct_url=direct, uv=False)
    hint = extras.install_hint("ski")
    assert "-m pip install -e" in hint and ".[ski]" in hint


def test_install_hint_editable_uv_outside_clone_venv(monkeypatch, tmp_path):
    clone = tmp_path / "clone"
    _set_install(monkeypatch, tmp_path, direct_url=_editable_url(clone), uv=True)
    monkeypatch.setattr(extras.sys, "prefix", str(tmp_path / "other-venv"))
    monkeypatch.setattr(extras.sys, "executable", "/the/python")
    hint = extras.install_hint("ski")
    assert hint == f'uv pip install --python "/the/python" -e "{clone.as_posix()}[ski]"'


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
                "PowerShell: in cmd drop the leading &",
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
    note = "PowerShell: in cmd drop the leading &"
    monkeypatch.setattr(extras.sys, "executable", exe)
    monkeypatch.setattr(extras, "os", SimpleNamespace(name="nt"))
    _set_install(monkeypatch, tmp_path, direct_url='{"dir_info": {"editable": true}}', uv=False)
    hint = extras.install_hint("ski")
    assert hint.startswith(f'& "{exe}" -m pip install -e') and hint.endswith(f"; {note})")
    _set_install(monkeypatch, tmp_path, direct_url=None, uv=False)
    hint = extras.install_hint("ski")
    assert hint == f'& "{exe}" -m pip install "cutpoint[ski]" ({note})'


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


# --- install_hint: non-editable installs and tool receipts (#34) ---------------------------


@pytest.mark.parametrize("kind", ["file", "git"])
def test_install_hint_non_editable_reinstalls_from_origin(monkeypatch, tmp_path, kind):
    if kind == "file":
        src = tmp_path / "src"
        direct_url = {"url": src.as_uri(), "dir_info": {}}
        expected = f'"{src.as_posix()}[ski]"'
    else:
        direct_url = {
            "url": "https://github.com/x/cutpoint",
            "vcs_info": {"vcs": "git", "commit_id": "abc123"},
        }
        expected = '"cutpoint[ski] @ git+https://github.com/x/cutpoint@abc123"'
    _set_install(monkeypatch, tmp_path, direct_url=json.dumps(direct_url), uv=True)
    assert extras.install_hint("ski").endswith(f" {expected}")


_UV_INDEX = f"--index {extras.TORCH_INDEX} --index-strategy unsafe-best-match"
_DIR = {"url": "file:///x/cutpoint", "dir_info": {}}
_GIT = {"url": "https://github.com/x/cutpoint", "vcs_info": {"vcs": "git", "commit_id": "abc"}}
_ARCHIVE = {"url": "https://example.com/cutpoint.whl", "archive_info": {}}


@pytest.mark.parametrize(
    ("uv", "direct_url", "cpu_torch", "present", "absent"),
    [
        # uv pip applies cutpoint's [tool.uv.sources] (cu126) to a directory or git install (#71)
        (True, _DIR, False, [], ["NVIDIA", "--index "]),
        (True, _GIT, False, [], ["NVIDIA", "--index "]),
        (True, _DIR, True, ["--reinstall-package torch"], ["--index "]),
        (True, _ARCHIVE, False, [_UV_INDEX], []),
        (False, _DIR, False, [f"--extra-index-url {extras.TORCH_INDEX}"], []),
    ],
)
def test_install_hint_non_editable_events_on_windows_cuda_note(
    monkeypatch, tmp_path, uv, direct_url, cpu_torch, present, absent
):
    monkeypatch.setattr(extras.sys, "platform", "win32")
    monkeypatch.setattr(extras, "_cpu_torch", lambda: cpu_torch)
    _set_install(monkeypatch, tmp_path, direct_url=json.dumps(direct_url), uv=uv)
    hint = extras.install_hint("events")
    assert all(p in hint for p in present)
    assert not any(a in hint for a in absent)


def test_install_hint_tool_receipt_keeps_default_index_and_python(monkeypatch, tmp_path):
    receipt = (
        '[tool]\nrequirements = [{ name = "cutpoint" }]\npython = "C:/Py/python.exe"\n'
        '[tool.options]\nindex = [{ url = "https://example.com/simple", default = true }]\n'
    )
    _set_install(monkeypatch, tmp_path, receipt=receipt, uv=True)
    hint = extras.install_hint("ski")
    assert '--python "C:/Py/python.exe"' in hint
    assert "--default-index https://example.com/simple" in hint
    assert "--index https://example.com/simple" not in hint


# --- install_hint: explicit indexes, CUDA note and installed CPU torch (#50) --------------

_TOOL = '[tool]\nrequirements = [{ name = "cutpoint" }]\n[tool.options]\n'


def test_install_hint_tool_receipt_explicit_index_is_left_out(monkeypatch, tmp_path):
    receipt = _TOOL + (
        'index = [{ name = "pytorch", url = "https://example.com/cu126", explicit = true }]\n'
    )
    _set_install(monkeypatch, tmp_path, receipt=receipt, uv=True)
    hint = extras.install_hint("ski")
    assert "https://example.com/cu126" not in hint
    assert "(explicit index pytorch left out" in hint


@pytest.mark.parametrize(
    ("options", "note_has_strategy"),
    [
        ('index = [{ url = "https://mirror.example/simple", default = true }]', True),
        ('index = [{ url = "https://download.pytorch.org/whl/cpu" }]', True),
        (
            'index = [{ url = "https://mirror.example/simple", default = true }]\n'
            'index-strategy = "unsafe-best-match"',
            False,
        ),
    ],
)
def test_install_hint_tool_cuda_note_kept_without_cuda_index(
    monkeypatch, tmp_path, options, note_has_strategy
):
    monkeypatch.setattr(extras.sys, "platform", "win32")
    _set_install(monkeypatch, tmp_path, receipt=_TOOL + options + "\n", uv=True)
    hint = extras.install_hint("events")
    note = hint[hint.index("(with an NVIDIA GPU") :]
    assert f"add --index {extras.TORCH_INDEX}" in note
    assert ("--index-strategy unsafe-best-match" in note) is note_has_strategy
    # the receipt's own strategy is never repeated in the note
    assert hint.count("--index-strategy") == 1


@pytest.mark.parametrize(
    ("source", "note"),
    [
        ('directory = "C:/src/cutpoint"', False),
        ('editable = "C:/src/cutpoint"', False),
        ('git = "https://github.com/FerventOrange/cutpoint"', False),
        ('path = "C:/dl/cutpoint-1.0-py3-none-any.whl"', True),
    ],
)
def test_install_hint_gpu_note_skipped_for_project_sources(monkeypatch, tmp_path, source, note):
    # a directory/editable/git install resolves with cutpoint's own [tool.uv.sources] (#61)
    monkeypatch.setattr(extras.sys, "platform", "win32")
    _set_install(monkeypatch, tmp_path, receipt=_receipt_with(source), uv=True)
    assert ("(with an NVIDIA GPU" in extras.install_hint("events")) is note


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
def test_install_hint_editable_cpu_torch_adds_reinstall_note(monkeypatch, tmp_path, in_clone_venv):
    # uv pip keeps an installed CPU torch, so the -e forms say how to get CUDA (#96)
    clone = tmp_path / "clone"
    _set_install(monkeypatch, tmp_path, direct_url=_editable_url(clone), uv=True)
    monkeypatch.setattr(extras.sys, "platform", "win32")
    monkeypatch.setattr(extras, "_cpu_torch", lambda: True)
    monkeypatch.setattr(extras.sys, "prefix", str(clone / ".venv" if in_clone_venv else tmp_path))
    note = " (with an NVIDIA GPU add --reinstall-package torch --reinstall-package torchvision"
    hint = extras.install_hint("events")
    assert hint.count(note) == 1 and hint.endswith("so torch uses CUDA)")
    assert hint.startswith("uv sync --extra events") == in_clone_venv


@pytest.mark.parametrize(
    ("platform", "cpu_torch", "extra"),
    [("win32", False, "events"), ("win32", True, "ski"), ("linux", True, "events")],
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
        ("win32", "ski", True, ""),
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
