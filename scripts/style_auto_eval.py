"""Development harness for #513: how well ``--style auto`` (the footage category detection of
:mod:`cutpoint.category`) does on footage whose category is known, as the evidence for the
owner's decision to make auto the default. Not part of the package; nothing here changes the
default. It runs :func:`cutpoint.category.probe_source` and :func:`cutpoint.category.decide`
as ``cutpoint detect`` does (the cached probes, the HUD text check for unsure sources, the
cached transcripts for talk), never parses logs, and only reads the sources: it writes to
Cutpoint's caches and to ``--out``.

Usage::

    uv run --offline python scripts/style_auto_eval.py labels.json --out auto.json
    uv run --offline python scripts/style_auto_eval.py labels.json --compare auto-old.json
    uv run --offline python scripts/style_auto_eval.py labels.json --home H --cache-dir C

Labels are a JSON file (keep real ones out of the repository)::

    {"entries": [
      {"project": "Day-1", "label": "action-cam", "inputs": ["Z:/Videos/Whistler/Day-1"]},
      {"project": "Cuts", "label": "general", "inputs": ["cut1.mp4"], "recursive": false,
       "exclude": []}
    ]}

``label`` is the true category (``action-cam``, ``gameplay``, ``talk`` or ``general``);
``inputs`` are files or folders, expanded as ``cutpoint detect`` does. ``project`` groups the
sources that one ``analyze`` would take together (entries of the same name are one project;
without it an entry is a project named after its first input). A project's true label is the
label of most of its footage (by duration).

``--home`` and ``--cache-dir`` set ``CUTPOINT_HOME`` and ``CUTPOINT_CACHE_DIR`` for the run, so
a scratch home keeps the user's config, models and caches out of it; a re-run with the same
cache reads the probes, the HUD text results and the transcripts instead of measuring again.

Reported (printed, and as ``--out`` JSON):

- ``confusion``: per true label, the decided category, sure or ambiguous, by count and minutes;
- ``sure_wrong``: sources decided sure as another category than their label (these would get
  the wrong style with no fallback);
- ``ambiguous``: per label, the sources that fall back to the default style, by count and by
  minutes;
- ``projects``: the style ``--style auto`` uses for each project (detected, or the fallback
  when ambiguous) against the style of the project's label;
- ``timing``: seconds per source for the probe (``probe_s``, which includes the HUD text
  check) and for the HUD text check alone (``ocr_s``, only when it ran: a cached one costs
  nothing);
- ``flip_ready``: the criteria for making auto the default, all required: no sure-wrong
  source; every label's ambiguous share at most ``--max-ambiguous`` (default 0.25), by count
  and by minutes; every project's style right. Each failed criterion is printed with its
  numbers. The decision stays with the owner.

``--compare OLD.json`` shows what changed against the JSON of an earlier run: sources whose
decision changed, projects whose style changed, and the totals.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cutpoint import category
from cutpoint.analyze import find_sources
from cutpoint.category import CATEGORIES, HYPE
from cutpoint.sources import source_segments
from cutpoint.styles import resolve_auto

logger = logging.getLogger(__name__)

ANALYZE_STYLE = "gaming"  # cutpoint.cli.ANALYZE_STYLE: analyze's default, auto's fallback
DEFAULT_MAX_AMBIGUOUS = 0.25
SCHEMA = 1
SURE, UNSURE = "sure", "ambiguous"


@dataclass
class Row:
    """One source: its label and what the detection made of it."""

    path: str
    project: str
    label: str
    category: str
    ambiguous: bool
    minutes: float
    reason: str
    probe_s: float
    ocr_s: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "project": self.project,
            "label": self.label,
            "category": self.category,
            "ambiguous": self.ambiguous,
            "minutes": round(self.minutes, 2),
            "reason": self.reason,
            "probe_s": round(self.probe_s, 2),
            "ocr_s": None if self.ocr_s is None else round(self.ocr_s, 2),
        }


# --------------------------------------------------------------------------- labels
def load_labels(path: Path) -> list[dict[str, Any]]:
    """The entries of a labels file, checked.

    Raises:
        ValueError: Naming the entry, for a missing key or an unknown label.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected {{'entries': [...]}} with at least one entry")
    for i, e in enumerate(entries, 1):
        if not isinstance(e, dict):
            raise ValueError(f"{path}: entry {i} is not an object")
        if e.get("label") not in CATEGORIES:
            raise ValueError(f"{path}: entry {i}: label must be one of {', '.join(CATEGORIES)}")
        inputs = e.get("inputs")
        if (
            not isinstance(inputs, list)
            or not inputs
            or not all(isinstance(x, str) for x in inputs)
        ):
            raise ValueError(f"{path}: entry {i}: inputs must be a list of paths")
    return entries


def entry_project(entry: dict[str, Any]) -> str:
    """Project name of an entry: ``project``, else the name of its first input."""
    name = entry.get("project")
    if name:
        return str(name)
    first = Path(entry["inputs"][0])
    return first.stem if first.suffix else first.name


# --------------------------------------------------------------------------- running
@contextlib.contextmanager
def timed_ocr(spent: list[float]) -> Iterator[None]:
    """While active, the seconds of every :func:`cutpoint.category.text_probe` call are
    appended to ``spent`` (the HUD text check, when it actually ran)."""
    orig = category.text_probe

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            spent.append(time.perf_counter() - t0)

    category.text_probe = wrapper  # type: ignore[assignment]
    try:
        yield
    finally:
        category.text_probe = orig  # type: ignore[assignment]


def project_label(rows: list[Row]) -> str:
    """The label of most of the footage (by duration; a tie goes to the first seen)."""
    weight: dict[str, float] = defaultdict(float)
    for r in rows:
        weight[r.label] += max(r.minutes, 1e-6)
    return max(weight, key=lambda k: weight[k])


def evaluate(entries: list[dict[str, Any]]) -> tuple[list[Row], list[dict[str, Any]]]:
    """Probe and decide every project of ``entries``: ``(rows, projects)``.

    Per project, the speech coverage of the cached transcripts is given to
    :func:`cutpoint.category.decide` as ``cutpoint detect`` does.
    """
    groups: dict[str, list[tuple[Path, str]]] = {}
    for e in entries:
        found = find_sources(
            [Path(p) for p in e["inputs"]], e.get("exclude") or [], bool(e.get("recursive"))
        )
        groups.setdefault(entry_project(e), []).extend((s, e["label"]) for s in found)
    rows: list[Row] = []
    projects: list[dict[str, Any]] = []
    for name, items in groups.items():
        probes, seconds, ocr = [], [], []
        for src, _ in items:
            spent: list[float] = []
            t0 = time.perf_counter()
            with timed_ocr(spent):
                probes.append(category.probe_source(src))
            seconds.append(time.perf_counter() - t0)
            ocr.append(sum(spent) if spent else None)
        speech: dict[str, float] = {}
        for (src, _), p in zip(items, probes, strict=True):
            segments = source_segments(src)
            if segments is not None:
                speech[p.path] = round(category.speech_coverage(segments, p.duration), 3)
        det = category.decide(probes, speech)
        mine = []
        for (src, lab), s, secs, o in zip(items, det.sources, seconds, ocr, strict=True):
            mine.append(
                Row(
                    str(src), name, lab, s.category, s.ambiguous, s.duration / 60, s.reason, secs, o
                )
            )
        rows.extend(mine)
        label = project_label(mine)
        style, fallback = category.choose_style(det, HYPE, ANALYZE_STYLE)
        right = resolve_auto(label, HYPE)
        projects.append(
            {
                "project": name,
                "sources": len(mine),
                "minutes": round(sum(r.minutes for r in mine), 1),
                "label": label,
                "category": det.category,
                "ambiguous": det.ambiguous,
                "style": style,
                "fallback": fallback is not None,
                "right_style": right,
                "right": style == right,
            }
        )
    return rows, projects


# --------------------------------------------------------------------------- summaries
def _cell(rows: list[Row]) -> dict[str, Any]:
    return {"n": len(rows), "minutes": round(sum(r.minutes for r in rows), 1)}


def confusion(rows: list[Row]) -> dict[str, dict[str, dict[str, Any]]]:
    """``{label: {"<category> sure|ambiguous": {n, minutes}}}``, labels in :data:`CATEGORIES`
    order and only those present."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for lab in CATEGORIES:
        mine = [r for r in rows if r.label == lab]
        if not mine:
            continue
        cells: dict[str, list[Row]] = defaultdict(list)
        for r in mine:
            cells[f"{r.category} {UNSURE if r.ambiguous else SURE}"].append(r)
        out[lab] = {k: _cell(v) for k, v in sorted(cells.items())}
    return out


def sure_wrong(rows: list[Row]) -> list[dict[str, Any]]:
    """Sources decided sure as another category than their label."""
    return [
        {
            "path": r.path,
            "project": r.project,
            "label": r.label,
            "category": r.category,
            "reason": r.reason,
        }
        for r in rows
        if not r.ambiguous and r.category != r.label
    ]


def ambiguous_rates(rows: list[Row]) -> dict[str, dict[str, Any]]:
    """Per label (and ``all``): sources and minutes, how many are ambiguous, and the rates."""
    out: dict[str, dict[str, Any]] = {}
    for lab, mine in [
        *((lab, [r for r in rows if r.label == lab]) for lab in CATEGORIES),
        ("all", rows),
    ]:
        if not mine:
            continue
        unsure = [r for r in mine if r.ambiguous]
        minutes = sum(r.minutes for r in mine)
        unsure_min = sum(r.minutes for r in unsure)
        out[lab] = {
            "n": len(mine),
            "ambiguous": len(unsure),
            "rate": round(len(unsure) / len(mine), 4),
            "minutes": round(minutes, 1),
            "ambiguous_minutes": round(unsure_min, 1),
            "minutes_rate": round(unsure_min / minutes, 4) if minutes else 0.0,
        }
    return out


def timing(rows: list[Row]) -> dict[str, Any]:
    """Seconds for the probe of every source, and for the HUD text check where it ran."""

    def stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        return {
            "n": len(values),
            "total": round(sum(values), 1),
            "mean": round(statistics.fmean(values), 2),
            "median": round(statistics.median(values), 2),
            "max": round(max(values), 2),
        }

    return {
        "probe_s": stats([r.probe_s for r in rows]),
        "ocr_s": stats([r.ocr_s for r in rows if r.ocr_s is not None]),
    }


def flip_ready(
    rows: list[Row],
    wrong: list[dict[str, Any]],
    projects: list[dict[str, Any]],
    max_ambiguous: float,
) -> dict[str, Any]:
    """The criteria, all required: no sure-wrong source; every label's ambiguous share at most
    ``max_ambiguous``, by count and by minutes; every project's style right. Returns the
    verdict, the failed criteria with their numbers and the text to print."""
    limit = f"{100 * max_ambiguous:.0f}%"
    failed: list[str] = []
    if wrong:
        failed.append(f"sure-wrong: {len(wrong)} source(s) > 0")
    for lab, a in ambiguous_rates(rows).items():
        if lab == "all":
            continue
        if a["rate"] > max_ambiguous:
            failed.append(
                f"{lab}: {a['ambiguous']}/{a['n']} ambiguous ({100 * a['rate']:.0f}%) > {limit}"
            )
        if a["minutes_rate"] > max_ambiguous:
            failed.append(
                f"{lab}: {a['ambiguous_minutes']:.0f}/{a['minutes']:.0f} min ambiguous "
                f"({100 * a['minutes_rate']:.0f}%) > {limit}"
            )
    for p in projects:
        if not p["right"]:
            how = "fallback" if p["fallback"] else "detected"
            failed.append(
                f"project {p['project']} ({p['label']}): style {p['style']} ({how}), "
                f"right is {p['right_style']}"
            )
    ok = not failed
    line = f"Flip ready: {'YES' if ok else 'NO'} (the decision is the owner's)"
    if ok:
        line += (
            f" - sure-wrong 0, every label at most {limit} ambiguous by count and minutes, "
            f"all {len(projects)} projects right"
        )
    else:
        line += "\n" + "\n".join(f"  FAIL {f}" for f in failed)
    return {"ready": ok, "max_ambiguous": max_ambiguous, "failed": failed, "line": line}


def summarize(
    rows: list[Row], projects: list[dict[str, Any]], max_ambiguous: float
) -> dict[str, Any]:
    wrong = sure_wrong(rows)
    return {
        "schema": SCHEMA,
        "created": datetime.now().isoformat(timespec="seconds"),
        "sources": [r.to_json() for r in rows],
        "confusion": confusion(rows),
        "sure_wrong": wrong,
        "ambiguous": ambiguous_rates(rows),
        "projects": projects,
        "timing": timing(rows),
        "flip_ready": flip_ready(rows, wrong, projects, max_ambiguous),
    }


# --------------------------------------------------------------------------- printing
def print_summary(data: dict[str, Any]) -> None:
    print("\nCONFUSION (true label -> decided category; sources, minutes)")
    for lab, cells in data["confusion"].items():
        for key, c in cells.items():
            print(f"  {lab:11s} -> {key:20s} {c['n']:4d} {c['minutes']:8.1f} min")
    wrong = data["sure_wrong"]
    print(f"\nSURE-WRONG: {len(wrong)}")
    for w in wrong:
        print(f"  {w['label']} decided {w['category']}: {w['path']} ({w['reason']})")
    print("\nAMBIGUOUS (the default style is used)")
    print(f"  {'label':11s} {'ambiguous/n':>12s} {'rate':>6s} {'minutes':>16s} {'rate':>6s}")
    for lab, a in data["ambiguous"].items():
        print(
            f"  {lab:11s} {a['ambiguous']:5d}/{a['n']:<6d} {100 * a['rate']:5.1f}% "
            f"{a['ambiguous_minutes']:7.1f}/{a['minutes']:<8.1f} {100 * a['minutes_rate']:5.1f}%"
        )
    print("\nPROJECTS (style --style auto uses, against the style of the right category)")
    for p in data["projects"]:
        how = "fallback" if p["fallback"] else "detected"
        verdict = "ok" if p["right"] else "WRONG"
        print(
            f"  {p['project'][:24]:24s} {p['label']:10s} -> {p['category']:10s} "
            f"{p['style']:11s} ({how}) vs {p['right_style']:11s} {verdict}"
        )
    n_right = sum(1 for p in data["projects"] if p["right"])
    print(f"  right style: {n_right}/{len(data['projects'])} projects")
    t = data["timing"]
    print("\nTIME PER SOURCE (probe includes the HUD text check)")
    for key, name in (("probe_s", "probe"), ("ocr_s", "HUD text check")):
        s = t[key]
        if s["n"]:
            print(
                f"  {name:15s} n={s['n']} mean {s['mean']}s median {s['median']}s "
                f"max {s['max']}s total {s['total']}s"
            )
        else:
            print(f"  {name:15s} not run")
    print(f"\n{data['flip_ready']['line']}")


def _key(s: dict[str, Any]) -> str:
    return f"{s['category']} {UNSURE if s['ambiguous'] else SURE}"


def compare(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """What changed from ``old`` to ``new`` (both ``--out`` JSON): sources whose decision
    changed, sources added or removed, projects whose style changed, and the totals."""
    olds = {s["path"]: s for s in old["sources"]}
    news = {s["path"]: s for s in new["sources"]}
    changed = [
        {
            "path": p,
            "label": news[p]["label"],
            "old": _key(olds[p]),
            "new": _key(news[p]),
        }
        for p in news
        if p in olds and _key(olds[p]) != _key(news[p])
    ]
    oldp = {p["project"]: p for p in old["projects"]}
    styles = [
        {"project": p["project"], "old": oldp[p["project"]]["style"], "new": p["style"]}
        for p in new["projects"]
        if p["project"] in oldp and oldp[p["project"]]["style"] != p["style"]
    ]
    return {
        "changed": changed,
        "added": sorted(set(news) - set(olds)),
        "removed": sorted(set(olds) - set(news)),
        "styles": styles,
        "sure_wrong": [len(old["sure_wrong"]), len(new["sure_wrong"])],
        "ambiguous_rate": [old["ambiguous"]["all"]["rate"], new["ambiguous"]["all"]["rate"]],
        "flip_ready": [old["flip_ready"]["ready"], new["flip_ready"]["ready"]],
    }


def print_compare(diff: dict[str, Any], old_name: str) -> None:
    print(f"\nCOMPARE with {old_name}")
    moves = Counter((c["label"], c["old"], c["new"]) for c in diff["changed"])
    for (lab, a, b), n in sorted(moves.items()):
        print(f"  {n:4d} {lab} sources: {a} -> {b}")
    if not diff["changed"]:
        print("  no source changed its decision")
    for c in diff["changed"]:
        print(f"    {c['path']}: {c['old']} -> {c['new']}")
    for p in diff["added"]:
        print(f"  added   {p}")
    for p in diff["removed"]:
        print(f"  removed {p}")
    for s in diff["styles"]:
        print(f"  project {s['project']}: style {s['old']} -> {s['new']}")
    (w0, w1), (a0, a1), (f0, f1) = diff["sure_wrong"], diff["ambiguous_rate"], diff["flip_ready"]
    print(f"  sure-wrong {w0} -> {w1}; ambiguous {100 * a0:.1f}% -> {100 * a1:.1f}%")
    print(f"  flip ready {'yes' if f0 else 'no'} -> {'yes' if f1 else 'no'}")


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("labels", type=Path, help="labels JSON (entries of label and inputs)")
    ap.add_argument("--out", type=Path, help="write the results as JSON here")
    ap.add_argument("--compare", type=Path, metavar="OLD", help="show changes against this --out")
    ap.add_argument(
        "--max-ambiguous",
        type=float,
        default=DEFAULT_MAX_AMBIGUOUS,
        help=f"flip-ready: most ambiguous sources, 0-1 (default: {DEFAULT_MAX_AMBIGUOUS})",
    )
    ap.add_argument("--home", type=Path, help="CUTPOINT_HOME for this run (a scratch home)")
    ap.add_argument("--cache-dir", type=Path, help="CUTPOINT_CACHE_DIR for this run")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug output")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not 0 <= args.max_ambiguous <= 1:
        ap.error("--max-ambiguous must be between 0 and 1")
    if args.home:
        os.environ["CUTPOINT_HOME"] = str(args.home)
    if args.cache_dir:
        os.environ["CUTPOINT_CACHE_DIR"] = str(args.cache_dir)
    try:
        entries = load_labels(args.labels)
        old = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare else None
        rows, projects = evaluate(entries)
    except (OSError, ValueError, FileNotFoundError) as err:
        logger.error("%s", err)
        return 1
    if args.out:  # never next to a source: the sources' folders are read-only
        out_dir = args.out.resolve().parent
        if any(out_dir == Path(r.path).resolve().parent for r in rows):
            ap.error(f"--out {args.out} is in a source's folder; write elsewhere")
    data = summarize(rows, projects, args.max_ambiguous)
    print_summary(data)
    if old is not None:
        diff = compare(old, data)
        data["compare"] = {"with": str(args.compare), **diff}
        print_compare(diff, str(args.compare))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
