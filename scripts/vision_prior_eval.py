"""Development measurement: the vision rating prior per style group and prompt kind (#948).
Not part of the package; it only reads ``highlights.json`` of analyzed projects, calls no model
and writes nothing except ``--json``.

Usage::

    uv run --offline python scripts/vision_prior_eval.py PROJECT [PROJECT ...]
    uv run --offline python scripts/vision_prior_eval.py PROJECTS_DIR
    uv run --offline python scripts/vision_prior_eval.py PROJECTS_DIR --json prior.json

Each argument is a project folder (it has a ``highlights.json``) or a folder of projects (every
subfolder with a ``highlights.json``). Analyze the footage with vision on first (``--vision``);
the more moments rated (``--vision-top``), the better the measurement.

A project's style group is :func:`cutpoint.analyze.prior_group` of its style's built-in base:
``action`` for action, sport and action-cam, ``other`` for the rest. The base of a user style
is read from this machine's style library (``cutpoint style show NAME``); a style not found
there takes its group from ``style_kind`` when that says action-cam, and is otherwise
``unresolved``: its ratings are left out of the cells (they could be action or other) and the
project is listed under ``unresolved``.

Per candidate with a current vision rating: the model's raw rating ``signals.vision`` and the
prompt kind ``vision_rated.kind`` (``frames`` or ``speech``). A stale rating (``vision_stale``:
the span changed since it was rated) or one without a recorded kind (a project rated before
#909) is left out.

Per (group, kind): ``n``, ``mean``, ``sd`` (sample), today's prior
(:data:`cutpoint.vision.RATING_PRIOR`), ``delta`` (mean - prior) and ``use``: whether the rule
says the cell gets the measured mean as its own prior (``n >= 15`` and ``|delta| >= 0.5``).
Also the projects skipped (no current vision rating), the ones unresolved and the ratings per
project.

``--json FILE`` writes the results.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = 1
MIN_N = 15
MIN_DELTA = 0.5
KINDS = ("frames", "speech")


def project_dirs(paths: Iterable[Path]) -> list[Path]:
    """The project folders of ``paths``: a folder with a ``highlights.json`` itself, else its
    subfolders that have one (by name). A folder with neither is an error."""
    out: list[Path] = []
    for p in paths:
        if (p / "highlights.json").is_file():
            out.append(p)
            continue
        if not p.is_dir():
            raise FileNotFoundError(f"not a folder: {p}")
        subs = sorted(d for d in p.iterdir() if (d / "highlights.json").is_file())
        if not subs:
            raise FileNotFoundError(f"no highlights.json in {p} or its subfolders")
        out.extend(subs)
    return out


def style_base(hl: dict[str, Any]) -> tuple[str, bool, bool]:
    """``(base, resolved, known)`` of a project's style: a built-in style is its own base, a
    user style's base comes from the style library (both resolved); otherwise action-cam when
    ``style_kind`` says so (known, not resolved) or the name as given (neither)."""
    from cutpoint import styles  # noqa: PLC0415

    name = str(hl.get("style") or "")
    if name in styles.STYLES:
        return name, True, True
    try:
        return styles.get_style(name).base, True, True
    except (ValueError, OSError):
        if hl.get("style_kind") == styles.ACTION_CAM:
            return styles.ACTION_CAM, False, True
        return name, False, False


def ratings(hl: dict[str, Any]) -> list[tuple[str, float]]:
    """``(kind, raw rating)`` of each candidate with a current vision rating and its kind."""
    out: list[tuple[str, float]] = []
    for c in hl.get("candidates") or []:
        rated = c.get("vision_rated")
        r = (c.get("signals") or {}).get("vision")
        if not isinstance(rated, dict) or r is None or "vision_stale" in c:
            continue
        kind = rated.get("kind")
        if kind in KINDS:
            out.append((kind, float(r)))
    return out


def load_project(project: Path) -> dict[str, Any]:
    """A project's style, group and ratings."""
    from cutpoint.analyze import prior_group  # noqa: PLC0415

    hl = json.loads((project / "highlights.json").read_text(encoding="utf-8"))
    base, resolved, known = style_base(hl)
    got = ratings(hl)
    return {
        "project": str(project),
        "style": hl.get("style"),
        "base": base,
        "resolved": resolved,
        "known": known,
        "group": prior_group(base),
        "ratings": got,
        "by_kind": {k: n for k in KINDS if (n := sum(kind == k for kind, _ in got))},
    }


def cell(values: list[float], prior: float) -> dict[str, Any]:
    """``n``, ``mean``, ``sd`` (sample; None under 2), ``prior``, ``delta`` and ``use`` of one
    (group, kind)."""
    n = len(values)
    mean = sum(values) / n if n else None
    sd = (
        math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        if n >= 2 and mean is not None
        else None
    )
    delta = mean - prior if mean is not None else None
    return {
        "n": n,
        "mean": None if mean is None else round(mean, 2),
        "sd": None if sd is None else round(sd, 2),
        "prior": prior,
        "delta": None if delta is None else round(delta, 2),
        "use": delta is not None and n >= MIN_N and abs(delta) >= MIN_DELTA,
    }


def evaluate(projects: list[dict[str, Any]]) -> dict[str, Any]:
    """The cells per (group, kind) over ``projects``, the projects used, those skipped and those
    unresolved."""
    from cutpoint.vision import PRIOR_GROUPS, RATING_PRIOR  # noqa: PLC0415

    values: dict[tuple[str, str], list[float]] = {}
    for p in projects:
        if not p["known"]:
            continue
        for kind, r in p["ratings"]:
            values.setdefault((p["group"], kind), []).append(r)
    cells = [
        {"group": g, "kind": k, **cell(values.get((g, k), []), RATING_PRIOR[g, k])}
        for g in PRIOR_GROUPS
        for k in KINDS
    ]
    used = [p for p in projects if p["ratings"] and p["known"]]
    return {
        "schema": SCHEMA,
        "rule": {"min_n": MIN_N, "min_delta": MIN_DELTA},
        "cells": cells,
        "projects": [
            {k: p[k] for k in ("project", "style", "base", "resolved", "group", "by_kind")}
            for p in used
        ],
        "skipped": [p["project"] for p in projects if not p["ratings"]],
        "unresolved": [p["project"] for p in projects if p["ratings"] and not p["known"]],
    }


def _num(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def print_results(data: dict[str, Any]) -> None:
    print(f"{'group':<8} {'kind':<7} {'n':>4} {'mean':>6} {'sd':>6} {'prior':>6} {'delta':>6}  use")
    for c in data["cells"]:
        print(
            f"{c['group']:<8} {c['kind']:<7} {c['n']:>4} {_num(c['mean']):>6} {_num(c['sd']):>6} "
            f"{c['prior']:>6.2f} {_num(c['delta']):>6}  {'yes' if c['use'] else 'no'}"
        )
    rule = data["rule"]
    print(f"(use: n >= {rule['min_n']} and |delta| >= {rule['min_delta']})")
    print()
    print("Projects:")
    for p in data["projects"]:
        counts = ", ".join(f"{k} {n}" for k, n in p["by_kind"].items())
        note = "" if p["resolved"] else " (unresolved)"
        print(f"  {p['project']}: {p['style']} -> {p['group']}{note}: {counts}")
    if data["skipped"]:
        print("Skipped (no current vision rating):")
        for s in data["skipped"]:
            print(f"  {s}")
    if data["unresolved"]:
        print("Unresolved (style not in this machine's library; left out of the cells):")
        for s in data["unresolved"]:
            print(f"  {s}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", type=Path, nargs="+", help="project folders, or folders of projects")
    ap.add_argument("--json", type=Path, metavar="FILE", help="write the results as JSON here")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug output")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        projects = [load_project(p) for p in project_dirs(args.paths)]
    except (OSError, ValueError, KeyError, TypeError) as err:  # JSONDecodeError is a ValueError
        logger.error("%s: %s", type(err).__name__, err)
        return 1
    data = evaluate(projects)
    print_results(data)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
