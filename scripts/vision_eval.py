"""Development benchmark: what the vision rating costs and how much of its effect a cheaper
variant keeps (#909). Not part of the package; it only reads ``highlights.json`` of analyzed
projects, calls no model and writes nothing except ``--json``.

Usage::

    uv run --offline python scripts/vision_eval.py REFERENCE VARIANT [VARIANT ...]
    uv run --offline python scripts/vision_eval.py REFERENCE VARIANT --novision NOVISION --json out.json

``REFERENCE`` is the project analyzed with the full vision rating, each ``VARIANT`` the same
footage analyzed another way (for example a cheaper rating). ``--novision`` is the same footage
analyzed with vision off. All are project folders.

Per variant:

- ``cost``: ``settings.vision.vision_s`` divided by ``pre_vision_s`` (the analyze time before the
  vision step), and the same for the reference. Needs a project analyzed by a version that writes
  them; otherwise "-". The ratio is comparable only between cold-cache runs: a re-rank
  (``rerank`` in ``settings.vision``) or a second analysis of the same footage has a much
  smaller ``pre_vision_s``. The row shows each variant's own ``vision_s`` too.
- ``edit``: the default edit is :func:`cutpoint.default_fit.fit_target` of the scored moments
  (not the visual pool) against ``settings.target_total``, what ``clips`` takes without options.
  The share of the reference's picks that the variant also picks. Two moments are the same when
  they come from the same file name and overlap by at least 50% of the shorter one, as in
  ``moment_eval.py`` and ``default_fit.CLASH_SHARE``.
- ``top5``, ``top10``: the share of the reference's 5 (10) best moments (by score) that are among
  the variant's 5 (10) best, by the same rule.
- ``tau20``: Kendall tau-b between the reference's and the variant's scores over the reference's
  20 best moments that the variant also has (``n`` of them). 1 is the same order, -1 the opposite.
- ``recovered`` (with ``--novision``): the reference's changes against the no-vision project are
  the picks it added and the picks it dropped in the default edit. The share the variant makes
  too.

``--json FILE`` writes the results.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Sequence
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = 1
MIN_OVERLAP = 0.5
TOPS = (5, 10)
TAU_TOP = 20

Moment = dict[str, Any]


def source_name(source: str) -> str:
    """The file name of a source path, whatever separator it was written with."""
    return (PureWindowsPath(source).name if "\\" in source else PurePath(source).name).lower()


def load_project(project: Path) -> dict[str, Any]:
    """The scored moments (best score first, ties keep the file order), the default edit and the
    vision settings of a project's ``highlights.json``."""
    from cutpoint import default_fit  # noqa: PLC0415

    path = project / "highlights.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    settings = data.get("settings") or {}
    cands = [
        c
        for c in data.get("candidates", [])
        if c.get("pool") != "visual" and all(k in c for k in ("source", "start", "end", "score"))
    ]
    if not cands:
        raise ValueError(f"{path}: no scored candidates")
    ranked = sorted(cands, key=lambda c: -c["score"])
    target = settings.get("target_total")
    picks = (
        default_fit.fit_target(ranked, target)
        if target
        else ranked[: data.get("suggested_top") or 5]
    )
    vision = settings.get("vision")
    return {
        "project": str(project),
        "ranked": ranked,
        "edit": picks,
        "vision": vision if isinstance(vision, dict) else {},
    }


def same(a: Moment, b: Moment, min_overlap: float = MIN_OVERLAP) -> bool:
    """Whether two moments are the same: one file name, overlap at least ``min_overlap`` of the
    shorter."""
    if source_name(str(a["source"])) != source_name(str(b["source"])):
        return False
    over = min(a["end"], b["end"]) - max(a["start"], b["start"])
    shorter = min(a["end"] - a["start"], b["end"] - b["start"])
    return over > 0 and over >= min_overlap * shorter


def contains(group: Sequence[Moment], m: Moment) -> bool:
    return any(same(m, g) for g in group)


def share(have: int, of: int) -> float | None:
    return have / of if of else None


def edit_overlap(ref: Sequence[Moment], var: Sequence[Moment]) -> float | None:
    """The share of ``ref``'s picks that ``var`` also has."""
    return share(sum(contains(var, m) for m in ref), len(ref))


def top_overlap(ref: Sequence[Moment], var: Sequence[Moment], n: int) -> float | None:
    """The share of ``ref``'s ``n`` best that are among ``var``'s ``n`` best."""
    top = ref[:n]
    return share(sum(contains(var[:n], m) for m in top), len(top))


def kendall_tau(pairs: Sequence[tuple[float, float]]) -> float | None:
    """Kendall tau-b of (x, y) pairs; None for fewer than two pairs or no spread in x or y."""
    conc = disc = tie_x = tie_y = 0
    for i, (x1, y1) in enumerate(pairs):
        for x2, y2 in pairs[i + 1 :]:
            dx, dy = x1 - x2, y1 - y2
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tie_x) * (conc + disc + tie_y))
    return (conc - disc) / denom if denom else None


def tau_top(
    ref: Sequence[Moment], var: Sequence[Moment], n: int = TAU_TOP
) -> tuple[float | None, int]:
    """(tau, moments used): ``ref``'s ``n`` best that ``var`` has too, their scores in both."""
    pairs = []
    for m in ref[:n]:
        match = next((v for v in var if same(m, v)), None)
        if match is not None:
            pairs.append((float(m["score"]), float(match["score"])))
    return kendall_tau(pairs), len(pairs)


def recovered(
    ref: Sequence[Moment], novision: Sequence[Moment], var: Sequence[Moment]
) -> dict[str, Any]:
    """The reference's changes against ``novision`` in the default edit: ``added`` (in ``ref``,
    not in ``novision``) and ``dropped`` (the reverse). ``var`` recovers an added pick by having
    it and a dropped one by not having it."""
    added = [m for m in ref if not contains(novision, m)]
    dropped = [m for m in novision if not contains(ref, m)]
    got = sum(contains(var, m) for m in added) + sum(not contains(var, m) for m in dropped)
    total = len(added) + len(dropped)
    return {"added": len(added), "dropped": len(dropped), "share": share(got, total)}


def cost(vision: dict[str, Any]) -> dict[str, Any]:
    """vision_s, pre_vision_s and their ratio from ``settings.vision`` (None when absent)."""
    v, pre = vision.get("vision_s"), vision.get("pre_vision_s")
    ok = isinstance(v, int | float) and isinstance(pre, int | float)
    return {
        "vision_s": v if ok else None,
        "pre_vision_s": pre if ok else None,
        "ratio": v / pre if ok and pre > 0 else None,
    }


def evaluate(
    ref: dict[str, Any], var: dict[str, Any], novision: dict[str, Any] | None
) -> dict[str, Any]:
    tau, n = tau_top(ref["ranked"], var["ranked"])
    out: dict[str, Any] = {
        "project": var["project"],
        "cost": cost(var["vision"]),
        "edit": edit_overlap(ref["edit"], var["edit"]),
        "edit_clips": len(var["edit"]),
        **{f"top{k}": top_overlap(ref["ranked"], var["ranked"], k) for k in TOPS},
        "tau20": tau,
        "tau20_n": n,
    }
    if novision is not None:
        out["recovered"] = recovered(ref["edit"], novision["edit"], var["edit"])
    return out


# -- output ----------------------------------------------------------------------------------


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.0%}"


def _num(v: float | None, fmt: str) -> str:
    return "-" if v is None else format(v, fmt)


def print_results(data: dict[str, Any]) -> None:
    ref = data["reference"]
    c = ref["cost"]
    print(
        f"vision benchmark: overlap >= {data['min_overlap']:.0%} of the shorter moment\n"
        f"reference {ref['project']}: {len(ref['edit'])} edit clips, "
        f"vision {_num(c['vision_s'], '.0f')} s, before vision {_num(c['pre_vision_s'], '.0f')} s, "
        f"cost {_num(c['ratio'], '.2f')}x"
    )
    has_nv = any("recovered" in v for v in data["variants"])
    head = f"  {'vis_s':>6} {'cost':>6} {'edit':>6} {'top5':>6} {'top10':>6} {'tau20':>7} {'n':>3}"
    print(head + (f" {'recovered':>10}" if has_nv else "") + "  project")
    for v in data["variants"]:
        row = (
            f"  {_num(v['cost']['vision_s'], '.0f'):>6} {_num(v['cost']['ratio'], '.2f'):>6} {_pct(v['edit']):>6} {_pct(v['top5']):>6} "
            f"{_pct(v['top10']):>6} {_num(v['tau20'], '.2f'):>7} {v['tau20_n']:>3}"
        )
        if has_nv:
            r = v.get("recovered")
            row += f" {_pct(r['share']) if r else '-':>10}"
        print(f"{row}  {v['project']}")
    if data.get("novision"):
        nv = data["novision"]
        print(
            f"no-vision {nv['project']}: the reference added {nv['added']} and dropped "
            f"{nv['dropped']} picks of its default edit"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reference", type=Path, help="project analyzed with the full vision rating")
    ap.add_argument("variants", type=Path, nargs="+", help="projects to compare with it")
    ap.add_argument("--novision", type=Path, metavar="DIR", help="the same footage, vision off")
    ap.add_argument("--json", type=Path, metavar="FILE", help="write the results as JSON here")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug output")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        ref = load_project(args.reference)
        novision = load_project(args.novision) if args.novision else None
        variants = [evaluate(ref, load_project(p), novision) for p in args.variants]
    except (OSError, ValueError, KeyError) as err:  # JSONDecodeError is a ValueError
        logger.error("%s: %s", type(err).__name__, err)
        return 1
    data: dict[str, Any] = {
        "schema": SCHEMA,
        "min_overlap": MIN_OVERLAP,
        "reference": {
            "project": ref["project"],
            "cost": cost(ref["vision"]),
            "edit": [[m["source"], m["start"], m["end"]] for m in ref["edit"]],
        },
        "variants": variants,
    }
    if novision is not None:
        data["novision"] = {
            "project": novision["project"],
            **recovered(ref["edit"], novision["edit"], ref["edit"]),
        }
    print_results(data)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
