"""The visual pool's mark on a candidate (:mod:`cutpoint.visual_pool`): its ``pool`` value.

Stdlib only, so :mod:`cutpoint.scorer` and the event scoring import it without a cycle through
:mod:`cutpoint.actioncam`, which :mod:`cutpoint.visual_pool` imports and which imports the
scorer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

POOL = "visual"  # the ``pool`` value of the visual pool's candidates


def is_pool(c: Mapping[str, Any]) -> bool:
    """Whether the candidate ``c`` is a pool moment."""
    return c.get("pool") == POOL
