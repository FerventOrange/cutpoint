"""``cutpoint skill``: the Claude Code skill for agents driving Cutpoint.

``SKILL.md`` in this package is the skill; it ships in the wheel as package data. ``install``
copies it to ``<skills dir>/cutpoint/SKILL.md`` (default: the Claude Code user skills folder,
:func:`default_dir`).
"""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

from cutpoint.cache import save_bytes

NAME = "cutpoint"


def default_dir() -> Path:
    """The Claude Code user skills folder: ``$CLAUDE_CONFIG_DIR/skills`` when that variable is
    set (Claude Code's config-folder override), else ``~/.claude/skills``."""
    config = os.environ.get("CLAUDE_CONFIG_DIR")
    base = Path(config).expanduser() if config else Path.home() / ".claude"
    return base / "skills"


def skill_bytes() -> bytes:
    """The packaged ``SKILL.md``, as stored."""
    return (files(__name__) / "SKILL.md").read_bytes()


def install(skills_dir: Path, *, force: bool = False) -> tuple[Path, bool]:
    """Copy the skill to ``skills_dir/cutpoint/SKILL.md`` (atomic write).

    Returns:
        ``(target, written)``; ``written`` is False when the target already had this content.

    Raises:
        FileExistsError: If the target exists with other content and ``force`` is False.
    """
    target = skills_dir / NAME / "SKILL.md"
    data = skill_bytes()
    if target.is_file() and target.read_bytes() == data:
        return target, False
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists with different content; use --force to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    save_bytes(target, data)
    return target, True
