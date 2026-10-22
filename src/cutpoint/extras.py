"""Optional dependency groups (``pyproject.toml`` extras) and how to install them.

Used to turn a missing optional import into an actionable message such as
"telemetry_parser is not installed; it comes with the 'ski' extra: ...".
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import sys
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Top-level import name -> extra that provides it.
MODULE_EXTRA: dict[str, str] = {
    "telemetry_parser": "ski",
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
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "file":
        # Keep the host of a UNC URL (file://server/share/...).
        raw = parts.path if parts.netloc in ("", "localhost") else f"//{parts.netloc}{parts.path}"
        path = Path(urllib.request.url2pathname(raw)).as_posix()
        return f'"{path}[{extras}]"'
    return f'"cutpoint[{extras}] @ {url}"'


def _git_url(git: str, req: dict[str, Any]) -> str:
    """PEP 508 ``git+`` URL for a receipt's ``git`` source.

    uv writes the ref as a ``?rev=``/``?tag=``/``?branch=`` query and the subdirectory as a
    ``#subdirectory=`` fragment or ``?subdirectory=`` query; separate keys are a fallback.
    """
    parts = urllib.parse.urlsplit(git)
    query = urllib.parse.parse_qs(parts.query)
    fragment = urllib.parse.parse_qs(parts.fragment)

    def first(key: str, *sources: dict[str, list[str]]) -> str | None:
        for src in sources:
            if src.get(key):
                return src[key][0]
        return str(req[key]) if req.get(key) else None

    ref = next((r for k in ("rev", "tag", "branch") if (r := first(k, query))), None)
    subdirectory = first("subdirectory", fragment, query)
    scheme = parts.scheme if parts.scheme.startswith("git+") else f"git+{parts.scheme}"
    url = urllib.parse.urlunsplit((scheme, parts.netloc, parts.path, "", ""))
    if ref:
        url += f"@{ref}"
    if subdirectory:
        url += f"#subdirectory={subdirectory}"
    return url


def _tool_source(req: dict[str, Any], extras: str) -> str:
    """The ``uv tool install`` target that reinstalls ``req`` from where it came from.

    ``req`` is the receipt's requirement without name and extras; none of the source keys
    means a registry (PyPI) install.
    """
    for key in ("editable", "directory", "path"):
        if req.get(key):
            target = f'"{req[key]}[{extras}]"'
            return f"--editable {target}" if key == "editable" else target
    if req.get("git"):
        return f'"cutpoint[{extras}] @ {_git_url(str(req["git"]), req)}"'
    if req.get("url"):
        return f'"cutpoint[{extras}] @ {req["url"]}"'
    return f'"cutpoint[{extras}]"'


def _tool_hint(receipt: dict[str, Any], extra: str) -> str:
    """``uv tool install`` command that keeps the receipt's source, extras and indexes and adds
    ``extra``."""
    tool = receipt.get("tool", {})
    extras = {extra}
    source: dict[str, Any] = {}
    try:
        for req in tool.get("requirements", []):
            if str(req.get("name", "")).lower() == "cutpoint":
                extras.update(str(e) for e in req.get("extras", []))
                source = {k: req[k] for k in req if k not in ("name", "extras")}
        options = tool.get("options", {})
        # An explicit index only serves packages pinned to it in tool.uv.sources, which the
        # CLI cannot express; as --index it would serve every package. It comes from uv
        # config (no flag sets it), which the reinstall reads again.
        entries = [i for i in options.get("index", []) if i.get("url")]
        indexes = [
            ("--default-index" if i.get("default") else "--index", str(i["url"]))
            for i in entries
            if not i.get("explicit")
        ]
        explicit = [str(i.get("name") or i["url"]) for i in entries if i.get("explicit")]
        strategy = options.get("index-strategy")
        python = tool.get("python")
    except (AttributeError, TypeError, KeyError):
        extras, source, indexes, explicit, strategy, python = {extra}, {}, [], [], None, None
    cmd = f"uv tool install {_tool_source(source, ','.join(sorted(extras)))} --reinstall"
    if python:
        cmd += f' --python "{python}"'
    cmd += "".join(f" {flag} {url}" for flag, url in indexes)
    if strategy:
        cmd += f" --index-strategy {strategy}"
    if explicit:
        cmd += (
            f" (explicit index {', '.join(explicit)} left out: it applies through your uv "
            "config, so keep that config in place)"
        )
    # A PyTorch CUDA index (download.pytorch.org/whl/cu126, .../nightly/cu128) already
    # serves the CUDA wheels; a mirror or the CPU index does not.
    has_torch_index = any("download.pytorch.org/whl/" in url and "/cu" in url for _, url in indexes)
    if not has_torch_index and extra in TORCH_EXTRAS:
        # PyPI torch is CPU-only on Windows; on Linux it is already a CUDA 13 build.
        # --reinstall already replaces an installed CPU torch (the tool is resolved anew).
        flags = f"--index {TORCH_INDEX}"
        if not strategy:
            flags += " --index-strategy unsafe-best-match"
        elif strategy != "unsafe-best-match":
            flags += " and change --index-strategy to unsafe-best-match"
        if sys.platform == "win32":
            cmd += f" (with an NVIDIA GPU add {flags} so torch uses CUDA)"
        elif sys.platform == "linux":
            cmd += f" (with an NVIDIA driver too old for CUDA 13, add {flags})"
    return cmd


def _cpu_torch() -> bool:
    """True when an installed torch is not a CUDA build (no ``+cu`` local version)."""
    try:
        return "+cu" not in importlib.metadata.Distribution.from_name("torch").version
    except importlib.metadata.PackageNotFoundError:
        return False


def _python() -> tuple[str, str]:
    """``sys.executable`` as a shell command prefix, and a note to append to the hint.

    A path without whitespace works bare in cmd, PowerShell and sh. A quoted path needs
    PowerShell's call operator, which cmd does not understand.
    """
    exe = sys.executable
    if not any(c.isspace() for c in exe):
        return exe, ""
    if os.name == "nt":
        return f'& "{exe}"', " (PowerShell; in cmd drop the leading &)"
    return f'"{exe}"', ""


def install_hint(extra: str) -> str:
    """The command that adds ``extra`` to the running environment."""
    receipt = _tool_receipt()
    if receipt is not None:
        return _tool_hint(receipt, extra)
    info = _direct_url()
    if not _editable(info):
        source = _pip_source(info or {}, extra)
        uv = uv_managed()
        # PyPI torch is CPU-only on Windows (as in _tool_hint).
        gpu = ""
        if extra in TORCH_EXTRAS and sys.platform == "win32":
            flags = (
                f"--index {TORCH_INDEX} --index-strategy unsafe-best-match"
                if uv
                else f"--extra-index-url {TORCH_INDEX}"
            )
            if _cpu_torch():
                # The installed CPU torch satisfies torch==x.y.z, so it would stay without
                # a reinstall (torchvision is likewise a CPU build then).
                flags += (
                    " --reinstall-package torch --reinstall-package torchvision"
                    if uv
                    else " --force-reinstall"
                )
            gpu = f" (with an NVIDIA GPU add {flags} so torch uses CUDA)"
        if uv:
            return f'uv pip install --python "{sys.executable}" {source}{gpu}'
        exe, note = _python()
        return f"{exe} -m pip install {source}{gpu}{note}"
    if uv_managed():
        return (
            f"uv sync --extra {extra} (list every other extra you use too: uv sync "
            f"removes packages of extras you leave out), or: uv pip install --python "
            f'"{sys.executable}" -e ".[{extra}]"'
        )
    exe, note = _python()
    return f'{exe} -m pip install -e ".[{extra}]" (from the repository){note}'


def missing_module_message(module: str) -> str | None:
    """Message for a missing optional module, or None if it is not an extra's."""
    top = module.split(".", 1)[0]
    extra = MODULE_EXTRA.get(top)
    if extra is None:
        return None
    return f"{top} is not installed; it comes with the '{extra}' extra: {install_hint(extra)}"
