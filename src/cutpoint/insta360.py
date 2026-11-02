"""Insta360 X3 file names and the ``_00_.insv`` that carries a clip's IMU trailer.

They sit below the 360 package (:mod:`cutpoint.pano`) so the core (telemetry, analyze) reads
X3 telemetry without importing it; :mod:`cutpoint.pano` imports them from here too.
File naming: see :mod:`cutpoint.pano.x3`.
"""

from __future__ import annotations

import re
from pathlib import Path

NAME_RE = re.compile(
    r"^(?P<kind>VID|LRV)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<lens>\d{2})_"
    r"(?P<num>\d{3})\.(?P<ext>insv|lrv)$",
    re.IGNORECASE,
)


def parse_name(path: str | Path) -> dict[str, str] | None:
    """Fields of an X3 file name (``kind``, ``date``, ``time``, ``lens``, ``num``, ``ext``)."""
    m = NAME_RE.match(Path(path).name)
    return (
        {
            k: v.upper() if k in ("kind",) else v.lower() if k == "ext" else v
            for k, v in m.groupdict().items()
        }
        if m
        else None
    )


def imu_source(path: str | Path) -> Path | None:
    """The ``_00_.insv`` that carries the IMU trailer for any file of a clip.

    telemetry-parser rejects ``.lrv`` files and the ``_10_`` lens has no
    trailer, so every X3 telemetry read goes through the ``_00_`` file. It
    only reads the trailer at the end of the file (a 7.8 GB clip parses in
    ~2 s even over a network share), so no staging is needed.

    The lookup runs next to the file read through ``source_remap``, so moved
    footage is found at its new location.

    Returns:
        The path (which exists), or None for a non-X3 file or a missing ``_00_``.
    """
    from cutpoint.config import remap_source  # noqa: PLC0415

    path = remap_source(path)
    f = parse_name(path)
    if f is None:
        return None
    cand = Path(path).with_name(f"VID_{f['date']}_{f['time']}_00_{f['num']}.insv")
    # On a case-sensitive file system an existing ``cand`` is the real name (unless its
    # case-swapped twin also exists). Otherwise list the folder: a case-insensitive one
    # (Windows, macOS) finds ``cand`` under any case, and the on-disk name is wanted.
    if cand.exists() and not cand.with_name(cand.name.swapcase()).exists():
        return cand
    # Match names case-insensitively, so .INSV / vid_ also work on case-sensitive file systems.
    # The parent stays as passed (no resolve()); only the file name comes from the listing.
    try:
        siblings = list(Path(path).parent.iterdir())
    except OSError:
        return None
    exact = [p for p in siblings if p.name.casefold() == cand.name.casefold()]
    exact.sort(key=lambda p: p.name != cand.name)  # the built spelling first, if on disk
    if exact:
        return exact[0]
    loose = re.compile(rf"VID_.*_00_{f['num']}\.insv", re.IGNORECASE)
    hits = [p for p in siblings if loose.fullmatch(p.name)]
    return hits[0] if len(hits) == 1 else None
