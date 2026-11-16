"""The one rule set for a review's segment list (``{id, source, start, end, label, exact?}``).

The review UI (``cutpoint.ui.validate_segments``) and the AI Chat's ops
(``cutpoint.assistant.ops``) both check a sequence with :func:`check_segments`, so they cannot
disagree about what a valid sequence is. Each caller words the errors itself: ``fail`` turns the
broken rule into its own exception (an HTTP 400 for the UI, an ``OpsError`` for the chat).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from cutpoint.clips import too_short

MAX_SEGMENTS = 500  # segments in one review
LABEL_MAX = 200  # characters in a segment's label

# fail(rule, i, row, **info) -> the exception to raise. ``i`` counts from 1; ``row`` is the
# segment (None for the list rules). The rules, in the order they are checked, and their info:
#   list, count, object, id, source, number (key), range (start, end), unreadable,
#   past (end, duration), label, exact
Fail = Callable[..., Exception]


def check_segments(
    raw: Any,
    *,
    has_source: Callable[[str], bool],
    duration: Callable[[str], float | None],
    fail: Fail,
) -> list[dict[str, Any]]:
    """``raw`` cleaned (``exact`` kept only when true); else ``fail(rule, ...)`` is raised.

    Args:
        raw: The segment list.
        has_source: Whether a source path is part of the project.
        duration: Seconds of a project source, asked only once a segment's times pass (None:
            cannot be read).
        fail: Builds the exception for a broken rule.
    """
    if not isinstance(raw, list):
        raise fail("list", 0, None)
    if len(raw) > MAX_SEGMENTS:
        raise fail("count", 0, None)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise fail("object", i, item)
        sid, label = item.get("id"), item.get("label", "segment")
        if not isinstance(sid, str) or not 0 < len(sid) <= 64 or sid in seen:
            raise fail("id", i, item)
        seen.add(sid)
        source = item.get("source")
        if not isinstance(source, str) or not has_source(source):
            raise fail("source", i, item)
        times: list[float] = []
        for key in ("start", "end"):
            v = item.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise fail("number", i, item, key=key)
            times.append(float(v))
        start, end = times
        if start < 0 or too_short(start, end):
            raise fail("range", i, item, start=start, end=end)
        length = duration(source)
        if length is None:
            raise fail("unreadable", i, item)
        if end > length + 0.01:
            raise fail("past", i, item, end=end, duration=length)
        if not isinstance(label, str) or len(label) > LABEL_MAX:
            raise fail("label", i, item)
        exact = item.get("exact", False)
        if not isinstance(exact, bool):
            raise fail("exact", i, item)
        out.append(
            {
                "id": sid,
                "source": source,
                "start": round(start, 3),
                "end": round(end, 3),
                "label": "".join(c for c in label if c.isprintable()).strip() or "segment",
                # a word splice: exported at exactly its edges and joined with hard cuts
                **({"exact": True} if exact else {}),
            }
        )
    return out
