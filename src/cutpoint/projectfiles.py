"""Names of the files Cutpoint writes into a project folder, and the ``highlights.json`` loader.

Stdlib only at import time, so the CLI and the light commands may import it; the loader
imports :mod:`cutpoint.cache` when it is called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

HIGHLIGHTS = "highlights.json"  # the ranked candidates of the analysis
PRE_EVENTS = "highlights.pre-events.json"  # the ranking before the events (deleted once ranked)
RERANK_BACKUP = "highlights.before-rerank.json"  # the highlights.json a rerank replaced (Undo)
UI_DIR = "ui"  # per-project state folder of cutpoint ui
REWRITE_HINT = "; re-run `cutpoint analyze` to rewrite it"


def load_highlights(
    project: Path,
    *,
    strict: bool = False,
    need: str = "sources",
    missing: str = "{path} not found",
    invalid: str = "{path} has no {need} list" + REWRITE_HINT,
) -> Any:
    """The ``highlights.json`` of ``project``.

    Args:
        project: Project folder.
        strict: Raise when the file is missing or unusable; otherwise return None.
        need: Key of the list a strict read requires.
        missing: Message of the missing file (``{path}``, ``{project}``).
        invalid: Message of a file without the ``need`` list (``{path}``, ``{project}``,
            ``{need}``).

    Returns:
        The file's object; lenient: None when it is missing, unreadable or not an object.

    Raises:
        FileNotFoundError: If ``strict`` and the file is missing (``missing``).
        ValueError: If ``strict`` and it is unreadable, not valid JSON (the message of
            :func:`cutpoint.cache.read_json` with :data:`REWRITE_HINT`) or has no ``need``
            list (``invalid``) or, for ``need="candidates"``, an entry that is not an object.
    """
    from cutpoint import cache  # noqa: PLC0415  (numpy: keep this module light)

    path = project / HIGHLIGHTS
    if not strict:
        data = cache.load_json(path)
        return data if isinstance(data, dict) else None
    try:
        data = cache.read_json(path, hint=REWRITE_HINT)
    except FileNotFoundError:
        raise FileNotFoundError(missing.format(path=path, project=project)) from None
    if not isinstance(data, dict) or not isinstance(data.get(need), list):  # e.g. null
        raise ValueError(invalid.format(path=path, project=project, need=need))
    if need == "candidates":
        for n, cand in enumerate(data[need], 1):
            if not isinstance(cand, dict):
                raise ValueError(f"{path}: candidate {n} is not an object" + REWRITE_HINT)
    return data
