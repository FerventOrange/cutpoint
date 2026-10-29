"""Optional dependency groups (``pyproject.toml`` extras), the edition they make (full or lite)
and how to install them.

Used to turn a missing optional import into an actionable message such as
"torch is not installed: this feature is not available in Cutpoint lite. ...".
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import re
import shlex
import sys
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Top-level import name -> extra that provides it. fastapi, uvicorn, keyring and telemetry_parser
# are core dependencies (the ui, keyring and ski extras are empty aliases), so they are not here.
MODULE_EXTRA: dict[str, str] = {
    "torch": "events",
    "torchvision": "events",
    "open_clip": "events",
    "panns_inference": "events",
    "torchlibrosa": "events",
    "rapidocr_onnxruntime": "events",
    "onnxruntime": "events",
    "scipy": "events",
    "sklearn": "events",
}
# Extras that pull in torch (the CUDA build needs the PyTorch index).
TORCH_EXTRAS = ("events", "all")
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
# The one-step installers. They pick the torch build for the machine, so re-running one is also
# how a tool install adds the full edition or upgrades (`uv tool upgrade` drops the torch backend).
_RAW = "https://raw.githubusercontent.com/FerventOrange/cutpoint/main"
INSTALL_SH = f"curl -LsSf {_RAW}/install.sh | sh"
INSTALL_PS1 = f"irm {_RAW}/install.ps1 | iex"
# What a clone runs: its default `full` group installs every extra.
UV_SYNC = (
    "uv sync (without --no-group full or UV_NO_GROUP=full: a clone installs every extra by "
    "default, a large download with torch and CUDA)"
)
LITE = "not available in Cutpoint lite"


def edition() -> str:
    """``"full"`` when the full-only packages (torch and the rest of the ``events`` extra) can
    be imported, else ``"lite"``."""
    full_only = [m for m, extra in MODULE_EXTRA.items() if extra == "events"]
    return "full" if all(importlib.util.find_spec(m) for m in full_only) else "lite"


def _notes(*notes: str) -> str:
    """The non-empty ``notes`` as one parenthetical to append to a hint: `` (a; b)`` or ``""``."""
    kept = [n for n in notes if n]
    return f" ({'; '.join(kept)})" if kept else ""


def uv_managed() -> bool:
    """True when this environment should be changed with uv rather than pip.

    That is the case under ``uv run`` (uv sets ``UV``) or when the environment
    has no pip at all (``uv venv`` / ``uv sync`` do not install it).
    """
    return bool(os.environ.get("UV")) or importlib.util.find_spec("pip") is None


def _tool_receipt() -> dict[str, Any] | None:
    """The ``uv tool install`` receipt of this environment, or None if it is not a tool.

    An unreadable receipt still marks a tool install (as an empty receipt).
    """
    receipt = Path(sys.prefix) / "uv-receipt.toml"
    if not receipt.is_file():
        return None
    try:
        return tomllib.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return {}


def _direct_url() -> dict[str, Any] | None:
    """cutpoint's PEP 610 ``direct_url.json``.

    None when cutpoint is not installed (running from a source tree); empty for a registry
    (PyPI) install or an unreadable file.
    """
    try:
        raw = importlib.metadata.distribution("cutpoint").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        info = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return info if isinstance(info, dict) else {}


def _editable(info: dict[str, Any] | None) -> bool:
    """True when cutpoint is installed from a clone in editable mode (or not installed)."""
    if info is None:
        return True  # running from a source tree
    try:
        return bool(info.get("dir_info", {}).get("editable"))
    except AttributeError:
        return False


def _pip_source(info: dict[str, Any], extras: str) -> str:
    """The ``pip install`` target that reinstalls a non-editable cutpoint from where it came
    from: a local path, a VCS URL at the installed commit, an archive URL, or PyPI."""
    url = info.get("url")
    if not isinstance(url, str) or not url:
        return f'"cutpoint[{extras}]"'
    vcs = info.get("vcs_info")
    if isinstance(vcs, dict):
        ref = f"{vcs.get('vcs') or 'git'}+{url}"
        if vcs.get("commit_id"):
            ref += f"@{vcs['commit_id']}"
        if info.get("subdirectory"):
            ref += f"#subdirectory={info['subdirectory']}"
        return f'"cutpoint[{extras}] @ {ref}"'
    path = _file_path(url)
    if path is not None:
        return f'"{path.as_posix()}[{extras}]"'
    return f'"cutpoint[{extras}] @ {url}"'


def _file_path(url: Any) -> Path | None:
    """The local path of a ``file:`` URL, or None for any other URL."""
    if not isinstance(url, str):
        return None
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "file":
        return None
    # Keep the host of a UNC URL (file://server/share/...).
    raw = parts.path if parts.netloc in ("", "localhost") else f"//{parts.netloc}{parts.path}"
    return Path(urllib.request.url2pathname(raw))


def _git_ref(git: str, req: dict[str, Any]) -> str | None:
    """The rev, tag or branch of a receipt's ``git`` source.

    uv writes it as a ``?rev=``/``?tag=``/``?branch=`` query; separate keys are a fallback.
    """
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(git).query)
    for key in ("rev", "tag", "branch"):
        if query.get(key):
            return query[key][0]
        if req.get(key):
            return str(req[key])
    return None


def _git_url(git: str, ref: str | None) -> str:
    """PEP 508 ``git+`` URL of a receipt's ``git`` source at ``ref``, keeping a
    ``#subdirectory=`` fragment or ``?subdirectory=`` query."""
    parts = urllib.parse.urlsplit(git)
    sub = (
        urllib.parse.parse_qs(parts.fragment).get("subdirectory")
        or urllib.parse.parse_qs(parts.query).get("subdirectory")
        or [None]
    )[0]
    scheme = parts.scheme if parts.scheme.startswith("git+") else f"git+{parts.scheme}"
    url = urllib.parse.urlunsplit((scheme, parts.netloc, parts.path, "", ""))
    if ref:
        url += f"@{ref}"
    if sub:
        url += f"#subdirectory={sub}"
    return url


def _upstream(git: str) -> bool:
    """True for cutpoint's own repository, the only git source the installers install from."""
    parts = urllib.parse.urlsplit(git.removeprefix("git+"))
    path = parts.path.rstrip("/").lower().removesuffix(".git")
    return (parts.hostname or "").lower() == "github.com" and path == "/ferventorange/cutpoint"


def _installed_torch_backend() -> str | None:
    """``cu126`` for an installed CUDA torch, ``cpu`` for a CPU one, None without torch."""
    try:
        version = importlib.metadata.Distribution.from_name("torch").version
    except importlib.metadata.PackageNotFoundError:
        return None
    return "cu126" if "+cu" in version else "cpu"


def _fork_hint(req: dict[str, Any], extra: str, backend: str | None) -> str:
    """``uv tool install`` from a git repository the installers do not install from (a fork),
    with an explicit ``--torch-backend`` (``backend``, else that of the installed torch)."""
    extras = {extra, *(str(e) for e in req.get("extras", []) if isinstance(e, str))}
    url = _git_url(str(req["git"]), _git_ref(str(req["git"]), req))
    backend = backend or _installed_torch_backend()
    note = "" if backend else " (cu126 instead of cpu with an NVIDIA GPU)"
    return (
        "the installer only installs from github.com/FerventOrange/cutpoint; for this git "
        f'install run: uv tool install "cutpoint[{",".join(sorted(extras))}] @ {url}" '
        f"--torch-backend {backend or 'cpu'} --python 3.12 --reinstall --no-sources{note}"
    )


def _installer_hint(
    receipt: dict[str, Any], extra: str, *notes: str, backend: str | None = None
) -> str:
    """What a tool install runs: the installer again (not lite in the lite edition), with the
    receipt's git, directory, path or editable source (it installs the PyPI release by
    default); a fork of the repository gets a ``uv tool install`` command instead.

    The installer picks the torch backend for this machine; a ``uv tool install`` without
    ``--torch-backend`` would pull CUDA torch on a machine without an NVIDIA GPU.
    """
    try:
        req = next(
            (
                r
                for r in receipt.get("tool", {}).get("requirements", [])
                if str(r.get("name", "")).lower() == "cutpoint"
            ),
            {},
        )
        if req.get("git") and not _upstream(str(req["git"])):
            return _fork_hint(req, extra, backend)
        if req.get("git"):
            ref = _git_ref(str(req["git"]), req)
            source: str | None = f"git@{ref}" if ref else "git"
        else:
            key = next((k for k in ("editable", "directory", "path") if req.get(k)), None)
            source = str(req[key]) if key else None
        editable = bool(req.get("editable"))
    except (AttributeError, TypeError):
        source, editable = None, False
    sh, ps1 = INSTALL_SH, INSTALL_PS1
    if source is not None:
        sh = f"{INSTALL_SH} -s -- --source {shlex.quote(source)}"
        ps1 = f'$env:CUTPOINT_SOURCE = "{source}"; {INSTALL_PS1}'
    lite = " without --lite / -Lite and with CUTPOINT_LITE unset" if edition() == "lite" else ""
    if editable:
        notes = ("the installer does not make an editable install", *notes)
    return f"re-run the installer{lite}: {sh} (Linux, macOS) or {ps1} (Windows PowerShell)" + (
        _notes(*notes)
    )


def _cpu_torch() -> bool:
    """True when an installed torch is not a CUDA build (no ``+cu`` local version)."""
    try:
        return "+cu" not in importlib.metadata.Distribution.from_name("torch").version
    except importlib.metadata.PackageNotFoundError:
        return False


# uv keeps an installed CPU torch that satisfies torch==x.y.z unless told to reinstall it.
_REINSTALL_TORCH = "--reinstall-package torch --reinstall-package torchvision"
_REINSTALL_TORCH_NOTE = f"with an NVIDIA GPU add {_REINSTALL_TORCH} so torch uses CUDA"


def _python() -> tuple[str, str]:
    """``sys.executable`` as a shell command prefix, and a note for the hint (or ``""``).

    A path without whitespace works bare in cmd, PowerShell and sh. A quoted path needs
    PowerShell's call operator, which cmd does not understand.
    """
    exe = sys.executable
    if not any(c.isspace() for c in exe):
        return exe, ""
    if os.name == "nt":
        return f'& "{exe}"', "PowerShell: in cmd drop the leading &"
    return f'"{exe}"', ""


def _torch_index_note(uv: bool) -> str:
    """Note text that points ``uv pip`` (``uv``) or pip at the CUDA torch index."""
    flags = (
        f"--index {TORCH_INDEX} --index-strategy unsafe-best-match"
        if uv
        else f"--extra-index-url {TORCH_INDEX}"
    )
    if _cpu_torch():
        # The installed CPU torch satisfies torch==x.y.z, so it would stay without
        # a reinstall (torchvision is likewise a CPU build then).
        flags += f" {_REINSTALL_TORCH}" if uv else " --force-reinstall"
    return f"with an NVIDIA GPU add {flags} so torch uses CUDA"


def install_hint(extra: str) -> str:
    """The command that adds ``extra`` to the running environment: the installer for a tool
    install, ``uv sync`` in a clone, else ``uv pip`` or pip."""
    receipt = _tool_receipt()
    if receipt is not None:
        return _installer_hint(receipt, extra)
    info = _direct_url()
    if not _editable(info):
        source = _pip_source(info or {}, extra)
        uv = uv_managed()
        # PyPI torch is CPU-only on Windows.
        gpu = ""
        # uv installs of a directory or git source apply cutpoint's own [tool.uv.sources],
        # which pin torch to the cu126 index; an archive or pip does not.
        project_sources = (
            uv
            and isinstance(info, dict)
            and ("dir_info" in info or isinstance(info.get("vcs_info"), dict))
        )
        if extra in TORCH_EXTRAS and sys.platform == "win32" and project_sources:
            if _cpu_torch():
                gpu = _REINSTALL_TORCH_NOTE
        elif extra in TORCH_EXTRAS and sys.platform == "win32":
            gpu = _torch_index_note(uv)
        if uv:
            return f'uv pip install --python "{sys.executable}" {source}{_notes(gpu)}'
        exe, note = _python()
        return f"{exe} -m pip install {source}{_notes(gpu, note)}"
    if uv_managed():
        clone = _file_path(info.get("url")) if info else None
        if clone is not None and Path(sys.prefix).resolve() != (clone / ".venv").resolve():
            # uv sync would change the clone's .venv, not the environment that is running.
            # uv pip applies the clone's [tool.uv.sources] (cu126 torch) but keeps an installed
            # CPU torch.
            gpu = (
                _REINSTALL_TORCH_NOTE
                if extra in TORCH_EXTRAS and sys.platform == "win32" and _cpu_torch()
                else ""
            )
            return (
                f'uv pip install --python "{sys.executable}" '
                f'-e "{clone.as_posix()}[{extra}]"{_notes(gpu)}'
            )
        # uv sync installs the locked +cu126 build, replacing a +cpu one.
        return UV_SYNC
    # pip ignores [tool.uv.sources], so it installs PyPI torch (CPU-only on Windows).
    gpu = _torch_index_note(False) if extra in TORCH_EXTRAS and sys.platform == "win32" else ""
    exe, note = _python()
    return f'{exe} -m pip install -e ".[{extra}]"{_notes("from the repository", gpu, note)}'


def missing_module_message(module: str) -> str | None:
    """Message for a missing optional module, or None if it is not an extra's."""
    top = module.split(".", 1)[0]
    extra = MODULE_EXTRA.get(top)
    if extra is None:
        return None
    return (
        f"{top} is not installed: this feature is {LITE} (it needs the '{extra}' extra). "
        f"To add it: {install_hint(extra)}"
    )


_EXTRA_MARKER = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")
_PLATFORM_MARKER = re.compile(r"""sys_platform\s*==\s*["']([^"']+)["']""")
_REQ_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _installed(name: str) -> bool:
    try:
        importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def extras_in_use() -> list[str]:
    """cutpoint's extras whose requirements for this platform are all installed.

    ``all`` is left out (it is the others together), as are the empty alias extras (``ui``,
    ``keyring``, ``ski``: their packages are core dependencies); empty when cutpoint is not
    installed.
    """
    try:
        dist = importlib.metadata.distribution("cutpoint")
    except importlib.metadata.PackageNotFoundError:
        return []
    needs: dict[str, list[str]] = {
        e: [] for e in dist.metadata.get_all("Provides-Extra") or [] if e != "all"
    }
    for req in dist.requires or []:
        extra = _EXTRA_MARKER.search(req)
        platform = _PLATFORM_MARKER.search(req)
        name = _REQ_NAME.match(req)
        if extra is None or extra.group(1) not in needs or name is None:
            continue
        if platform is not None and platform.group(1) != sys.platform:
            continue
        needs[extra.group(1)].append(name.group(0))
    return sorted(e for e, names in needs.items() if names and all(map(_installed, names)))


def torch_cuda_hint() -> str:
    """The command that replaces an installed torch that cannot use CUDA with the CUDA 12.6
    build, for this install type, keeping the extras in use."""
    receipt = _tool_receipt()
    if receipt is not None:
        return _installer_hint(
            receipt, "events", "it installs CUDA torch when it finds an NVIDIA GPU", backend="cu126"
        )
    in_use = sorted({"events", *extras_in_use()})
    extras = ",".join(in_use)
    uv = uv_managed()
    uv_flags = f"--index {TORCH_INDEX} --index-strategy unsafe-best-match {_REINSTALL_TORCH}"
    pip_flags = f"--extra-index-url {TORCH_INDEX} --force-reinstall"
    info = _direct_url()
    if not _editable(info):
        source = _pip_source(info or {}, extras)
        if uv:
            return f'uv pip install --python "{sys.executable}" {source} {uv_flags}'
        if not (info or {}).get("url"):
            # --force-reinstall would pick the newest cutpoint from PyPI; keep this version.
            source = f'"cutpoint[{extras}]=={importlib.metadata.version("cutpoint")}"'
        exe, note = _python()
        return f"{exe} -m pip install {source} {pip_flags}{_notes(note)}"
    if uv:
        clone = _file_path(info.get("url")) if info else None
        if clone is not None and Path(sys.prefix).resolve() != (clone / ".venv").resolve():
            # uv pip applies the clone's [tool.uv.sources] (cu126 torch).
            return (
                f'uv pip install --python "{sys.executable}" '
                f'-e "{clone.as_posix()}[{extras}]" {_REINSTALL_TORCH}'
            )
        # uv sync installs the locked +cu126 build, replacing a +cpu one.
        return UV_SYNC
    exe, note = _python()
    return f'{exe} -m pip install -e ".[{extras}]" {pip_flags}{_notes("from the repository", note)}'
