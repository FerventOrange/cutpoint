"""Development benchmark: how well a project's ranked moments match the owner's labels. Every
change to the ranking (scores, weights, events, dialogue, the candidate list) should be measured
with it. Not part of the package; it only reads ``highlights.json`` (and runs
:func:`cutpoint.planning.select_moments` on it, as ``clips`` does) and the label files, and writes
nothing except ``--json``.

Usage::

    uv run --offline python scripts/moment_eval.py --project P1 --labels L1.json --json run.json
    uv run --offline python scripts/moment_eval.py --manifest pairs.json
    uv run --offline python scripts/moment_eval.py --compare before.json after.json

``--project DIR`` (an analyzed project folder) and ``--labels FILE`` are repeated and paired in
order. ``--manifest`` is a JSON file listing pairs (paths relative to the manifest)::

    {"pairs": [{"project": "projects/a", "labels": "labels/a.json"}]}

Labels are one JSON file per video. Keep them out of the repository: they describe private
footage. All times are seconds in the source video::

    {"video": "stream.mp4", "duration": 3600.0,
     "keep": [[100, 110]],
     "skip": [[400, 406]],
     "reference": [[1200, 1260]],
     "holdout": false, "note": "free text"}

- ``video``: the source file name. Candidates match it by name (case-insensitive, folders in
  the candidate's ``source`` ignored).
- ``keep`` / ``skip``: moments the owner judged good / not good. Either may be empty.
- ``reference``: ranges of a hand edit made of the best moments. It may leave good moments out
  on purpose, so it is a softer measure than ``keep``.
- ``holdout``: true for a video that must not be tuned on. It is reported apart and left out of
  the totals.
- ``duration`` (optional): the video length, used when the project does not list the source.
  The length for the budgets is the project's duration of the matching source(s); a label
  duration that differs from it by more than 1 s is an error (labels for another file).
- ``note`` (optional): free text, not used.

Metrics, per video. Candidates are ranked best score first (ties keep the file order); visual
pool candidates (``pool: "visual"``) are not ranked, as in :func:`select_moments`.

- Overlap rule: a candidate hits a label range when the overlap is at least ``--min-overlap``
  (default 0.5) of the shorter of the two, as ``planning.CLASH_SHARE`` does. So a short
  candidate inside a long range hits it, a long candidate that holds a short range hits it, and
  a candidate that only grazes a range does not.
- ``keep_top10``, ``keep_top20``, ``skip_top10``, ``skip_top20``: how many keep (skip) ranges a
  candidate among the 10 (20) best hits. More is better for keeps; for skips fewer is better,
  and they are never flagged. Top-N favours long candidates (a 200 s stretch holds many
  ranges); the budget metrics below charge for length, so read both.
- ``keep_15``, ``keep_30``, ``ref_15``, ``ref_30``: take the ranked candidates best first,
  adding one only if the union of the taken ones stays within 15% of the video length (one that
  does not fit is skipped and the next is tried). The 30% set extends the 15% set: the fill
  goes on to 30%, retrying only the candidates not yet taken, in rank order, so a 30% value is
  never below the 15% one. A keep or reference range is hit when a taken candidate hits it by
  the overlap rule, or when the union covers at least ``--min-overlap`` of it. ``refsec_15``, ``refsec_30`` are the share of the reference seconds the union covers.
- ``lift``: the value divided by the control's mean (a multiple of chance).
- Control: the same candidates in random order, ``--runs`` times (default 1000, ``--seed``
  default 7), for every metric above. The report gives the mean and the 95th percentile. A
  keep or reference metric "beats noise" only when its value is above the 95th percentile.
- Default edit: ``select_moments(project)``, what ``clips`` takes without options (the
  best moments that fit the style's ``target_total``; ``--edit-total`` gives it a ``max_total``). Edit highlights:
  ``select_moments(project, max_total=180)``, what ``edit --kind highlights`` takes. For both:
  which keep, skip and reference ranges the picks of this video hit by the same overlap rule,
  and the clips and seconds of this video (``clips_all`` counts the picks of every source).
  There is no random control for the edits.

``--json FILE`` writes the results. ``--compare A.json B.json`` shows, per video and metric, the
change from A to B, and whether "beats noise" changed, with rows for what the edits contain. It
warns when the two runs differ in schema, seed, runs or ``--min-overlap``.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import random
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = 2
DEFAULT_RUNS = 1000
DEFAULT_SEED = 7
DEFAULT_MIN_OVERLAP = 0.5
TOPS = (10, 20)
BUDGETS = (0.15, 0.30)
HIGHLIGHTS_TOTAL = 180.0  # ``edit --kind highlights``
DURATION_TOLERANCE_S = 1.0

Range = tuple[float, float]

#: Metric name -> True when more is better (flagged against the control).
METRICS: dict[str, bool] = {
    "keep_top10": True,
    "keep_top20": True,
    "skip_top10": False,
    "skip_top20": False,
    "keep_15": True,
    "keep_30": True,
    "ref_15": True,
    "ref_30": True,
    "refsec_15": True,
    "refsec_30": True,
}

# -- labels and projects ---------------------------------------------------------------------


def _ranges(raw: Any, field: str, where: str) -> list[Range]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{where}: {field} must be a list of [start, end] pairs")
    out: list[Range] = []
    for item in raw:
        ok = isinstance(item, list | tuple) and len(item) == 2
        if not ok or not all(isinstance(x, int | float) and not isinstance(x, bool) for x in item):
            raise ValueError(f"{where}: {field} entry {item!r} is not [start, end] in seconds")
        a, b = float(item[0]), float(item[1])
        if a < 0 or b <= a:
            raise ValueError(f"{where}: {field} entry {item!r} needs 0 <= start < end")
        out.append((a, b))
    return out


def load_labels(path: Path) -> dict[str, Any]:
    """Read and check one labels file."""
    where = str(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a JSON object")
    video = data.get("video")
    if not isinstance(video, str) or not video:
        raise ValueError(f"{where}: 'video' (the source file name) is required")
    holdout = data.get("holdout", False)
    if not isinstance(holdout, bool):
        raise ValueError(f"{where}: 'holdout' must be true or false")
    duration = data.get("duration")
    if duration is not None and (not isinstance(duration, int | float) or duration <= 0):
        raise ValueError(f"{where}: 'duration' must be a positive number of seconds")
    labels = {
        "video": video,
        "duration": float(duration) if duration else None,
        "keep": _ranges(data.get("keep"), "keep", where),
        "skip": _ranges(data.get("skip"), "skip", where),
        "reference": _ranges(data.get("reference"), "reference", where),
        "holdout": holdout,
        "note": str(data.get("note") or ""),
    }
    if not (labels["keep"] or labels["skip"] or labels["reference"]):
        raise ValueError(f"{where}: no keep, skip or reference ranges")
    return labels


def load_manifest(path: Path) -> list[tuple[Path, Path]]:
    """The (project, labels) pairs of a manifest; relative paths are relative to it."""
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = data.get("pairs") if isinstance(data, dict) else None
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{path}: expected {{'pairs': [{{'project': ..., 'labels': ...}}]}}")
    out = []
    for p in pairs:
        if not isinstance(p, dict) or not p.get("project") or not p.get("labels"):
            raise ValueError(f"{path}: every pair needs 'project' and 'labels'")
        out.append((path.parent / p["project"], path.parent / p["labels"]))
    return out


def source_name(source: str) -> str:
    """The file name of a source path, whatever separator it was written with."""
    return PureWindowsPath(source).name if "\\" in source else PurePath(source).name


def load_project(project: Path, video: str) -> dict[str, Any]:
    """Ranked candidates of ``video`` in the project's ``highlights.json``, with its settings."""
    path = project / "highlights.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    want = video.lower()
    indexes = {
        i for i, s in enumerate(data.get("sources", [])) if source_name(s["path"]).lower() == want
    }
    cands = [
        c
        for c in data.get("candidates", [])
        if c.get("pool") != "visual" and source_name(c["source"]).lower() == want
    ]
    if not cands:
        raise ValueError(f"{project}: no candidates from a source named {video!r}")
    cands.sort(key=lambda c: -c["score"])  # stable: ties keep the file order
    duration = sum(float(s["duration"]) for i, s in enumerate(data["sources"]) if i in indexes)
    return {
        "candidates": [(float(c["start"]), float(c["end"])) for c in cands],
        "duration": duration or None,
    }


# -- metrics -----------------------------------------------------------------------------------


def overlap(a: Range, b: Range) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def hit(cand: Range, label: Range, min_overlap: float) -> bool:
    """Whether the overlap is at least ``min_overlap`` of the shorter of ``cand`` and the label
    range."""
    shorter = min(cand[1] - cand[0], label[1] - label[0])
    return overlap(cand, label) >= min_overlap * shorter


def top_hits(cands: Sequence[Range], labels: Sequence[Range], n: int, min_overlap: float) -> int:
    top = cands[:n]
    return sum(any(hit(c, lab, min_overlap) for c in top) for lab in labels)


class Union:
    """Disjoint sorted intervals with their total length."""

    def __init__(self) -> None:
        self.starts: list[float] = []
        self.ends: list[float] = []
        self.total = 0.0

    def _span(self, a: float, b: float) -> tuple[int, int]:
        """Indexes [i, j) of the intervals that touch [a, b]."""
        i = bisect.bisect_left(self.ends, a)
        j = bisect.bisect_right(self.starts, b)
        return i, max(i, j)

    def added(self, a: float, b: float) -> float:
        """The seconds [a, b] would add."""
        i, j = self._span(a, b)
        return (b - a) - sum(
            max(0.0, min(b, self.ends[k]) - max(a, self.starts[k])) for k in range(i, j)
        )

    def add(self, a: float, b: float) -> None:
        i, j = self._span(a, b)
        if i < j:
            a, b = min(a, self.starts[i]), max(b, self.ends[j - 1])
        self.total += (b - a) - sum(self.ends[k] - self.starts[k] for k in range(i, j))
        self.starts[i:j] = [a]
        self.ends[i:j] = [b]

    def covered(self, a: float, b: float) -> float:
        i, j = self._span(a, b)
        return sum(max(0.0, min(b, self.ends[k]) - max(a, self.starts[k])) for k in range(i, j))


def budget_fills(
    cands: Sequence[Range], budgets: Sequence[float]
) -> Iterator[tuple[Union, list[Range]]]:
    """Fill one union best first for each of the ``budgets`` (ascending), yielding the union and
    the candidates taken after each. A candidate that would push the union past the budget is
    skipped; the next budget retries only the candidates not yet taken, in rank order, so each
    set extends the one before. The union is the same object each time: read it before
    advancing."""
    u = Union()
    taken: list[Range] = []
    left = list(cands)
    for budget in budgets:
        rest = []
        for a, b in left:
            if u.total + u.added(a, b) <= budget:
                u.add(a, b)
                taken.append((a, b))
            else:
                rest.append((a, b))
        left = rest
        yield u, taken


def budget_union(cands: Sequence[Range], budget: float) -> tuple[Union, list[Range]]:
    """The union and the candidates taken for the one ``budget`` (see :func:`budget_fills`)."""
    return next(iter(budget_fills(cands, [budget])))


def budget_cover(
    u: Union, taken: Sequence[Range], labels: Sequence[Range], min_overlap: float
) -> tuple[int, float]:
    """(ranges hit, share of the range seconds covered). A range is hit when a taken candidate
    hits it by the overlap rule, or when the union covers ``min_overlap`` of it."""
    hits = sum(
        any(hit(c, lab, min_overlap) for c in taken)
        or u.covered(*lab) >= min_overlap * (lab[1] - lab[0])
        for lab in labels
    )
    secs = sum(u.covered(a, b) for a, b in labels) / sum(b - a for a, b in labels)
    return hits, secs


def measure(
    cands: Sequence[Range],
    labels: dict[str, Any],
    duration: float,
    min_overlap: float,
) -> dict[str, float]:
    """Every metric that has labels, for the candidates in this order."""
    out: dict[str, float] = {}
    for kind in ("keep", "skip"):
        if labels[kind]:
            for n in TOPS:
                out[f"{kind}_top{n}"] = top_hits(cands, labels[kind], n, min_overlap)
    if labels["keep"] or labels["reference"]:
        fills = budget_fills(cands, [frac * duration for frac in BUDGETS])
        for frac, (u, taken) in zip(BUDGETS, fills, strict=True):
            pct = round(frac * 100)
            if labels["keep"]:
                out[f"keep_{pct}"] = budget_cover(u, taken, labels["keep"], min_overlap)[0]
            if labels["reference"]:
                hits, secs = budget_cover(u, taken, labels["reference"], min_overlap)
                out[f"ref_{pct}"] = hits
                out[f"refsec_{pct}"] = secs
    return out


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q`` (0-1) percentile, nearest rank."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def control(
    cands: Sequence[Range],
    labels: dict[str, Any],
    duration: float,
    min_overlap: float,
    runs: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Mean and 95th percentile of every metric over ``runs`` random orders of ``cands``."""
    rng = random.Random(seed)
    order = list(cands)
    samples: dict[str, list[float]] = {}
    for _ in range(runs):
        rng.shuffle(order)
        for name, v in measure(order, labels, duration, min_overlap).items():
            samples.setdefault(name, []).append(v)
    return {
        name: {"mean": sum(v) / len(v), "p95": percentile(v, 0.95)} for name, v in samples.items()
    }


def default_edit(
    project: Path, video: str, labels: dict[str, Any], total: float | None, min_overlap: float
) -> dict[str, Any]:
    """What ``select_moments(project, max_total=total)`` picks: the clips and seconds of this
    video and the labels they hit."""
    from cutpoint.planning import select_moments  # noqa: PLC0415

    try:
        picks, _ = select_moments(project, max_total=total)
    except ValueError:
        picks = []
    want = video.lower()
    own = [p for p in picks if source_name(p["source"]).lower() == want]
    mine = [(float(p["start"]), float(p["end"])) for p in own]
    out: dict[str, Any] = {
        "max_total": total,
        "clips": len(own),
        "clips_all": len(picks),
        "seconds": round(sum(float(p["duration"]) for p in own), 1),
    }
    for kind in ("keep", "skip", "reference"):
        contained = [
            list(lab) for lab in labels[kind] if any(hit(c, lab, min_overlap) for c in mine)
        ]
        out[kind] = {"contained": contained, "total": len(labels[kind])}
    return out


def evaluate_video(
    project: Path,
    labels: dict[str, Any],
    runs: int,
    seed: int,
    min_overlap: float,
    edit_total: float | None,
) -> dict[str, Any]:
    info = load_project(project, labels["video"])
    known, stated = info["duration"], labels["duration"]
    if known and stated and abs(known - stated) > DURATION_TOLERANCE_S:
        raise ValueError(
            f"{project}: the labels of {labels['video']!r} say {stated:.1f} s but the project's "
            f"source is {known:.1f} s (labels for another file?)"
        )
    duration = known or stated or max(e for _, e in info["candidates"])
    cands = info["candidates"]
    values = measure(cands, labels, duration, min_overlap)
    ctl = control(cands, labels, duration, min_overlap, runs, seed)
    metrics: dict[str, Any] = {}
    for name, value in values.items():
        m = {"value": value, "control_mean": ctl[name]["mean"], "control_p95": ctl[name]["p95"]}
        m["lift"] = value / m["control_mean"] if m["control_mean"] else None
        m["beats_noise"] = bool(METRICS[name] and value > m["control_p95"])
        metrics[name] = m
    totals = {
        "keep": len(labels["keep"]),
        "skip": len(labels["skip"]),
        "reference": len(labels["reference"]),
    }
    return {
        "video": labels["video"],
        "project": str(project),
        "holdout": labels["holdout"],
        "duration_s": round(duration, 1),
        "candidates": len(cands),
        "labels": totals,
        "metrics": metrics,
        "edit": default_edit(project, labels["video"], labels, edit_total, min_overlap),
        "edit_highlights": default_edit(
            project, labels["video"], labels, HIGHLIGHTS_TOTAL, min_overlap
        ),
    }


# -- output ----------------------------------------------------------------------------------


def _fmt(name: str, v: float) -> str:
    return f"{v:.0%}" if name.startswith("refsec") else f"{v:.1f}".rstrip("0").rstrip(".")


def _count(name: str, v: float, labels: dict[str, int]) -> str:
    if name.startswith("refsec"):
        return f"{v:.0%}"
    kind = "reference" if name.startswith("ref_") else name.split("_")[0]
    return f"{_fmt(name, v)}/{labels[kind]}"


def print_video(v: dict[str, Any]) -> None:
    mins = v["duration_s"] / 60
    print(f"\n{v['video']}  ({mins:.0f} min, {v['candidates']} candidates)")
    print(f"  {'metric':<11} {'value':>7} {'random':>8} {'p95':>7} {'x chance':>9}  beats noise")
    for name, m in v["metrics"].items():
        lift = f"{m['lift']:.1f}x" if m["lift"] is not None else "-"
        flag = ("yes" if m["beats_noise"] else "no") if METRICS[name] else "(fewer is better)"
        print(
            f"  {name:<11} {_count(name, m['value'], v['labels']):>7} "
            f"{_fmt(name, m['control_mean']):>8} {_fmt(name, m['control_p95']):>7} "
            f"{lift:>9}  {flag}"
        )
    for key, title in (("edit", "default edit"), ("edit_highlights", "edit highlights")):
        e = v[key]
        cap = f"max {e['max_total']:.0f} s" if e["max_total"] else "suggested top"
        parts = [
            f"{k} {len(e[k]['contained'])}/{e[k]['total']}"
            for k in ("keep", "skip", "reference")
            if e[k]["total"]
        ]
        print(
            f"  {title} ({cap}): {e['clips']} clips ({e['clips_all']} in all), "
            f"{e['seconds']:.0f} s; contains " + ", ".join(parts)
        )


def print_results(data: dict[str, Any]) -> None:
    tuned = [v for v in data["videos"] if not v["holdout"]]
    held = [v for v in data["videos"] if v["holdout"]]
    print(
        f"moment benchmark: seed {data['seed']}, {data['runs']} random orders, "
        f"overlap >= {data['min_overlap']:.0%} of the shorter of candidate and label range"
    )
    for v in tuned:
        print_video(v)
    if len(tuned) > 1:
        print("\ntotals (videos not held out)")
        for name in METRICS:
            have = [v for v in tuned if name in v["metrics"] and not name.startswith("refsec")]
            if have:
                got = sum(v["metrics"][name]["value"] for v in have)
                kind = "reference" if name.startswith("ref_") else name.split("_")[0]
                of = sum(v["labels"][kind] for v in have)
                beats = sum(v["metrics"][name]["beats_noise"] for v in have)
                extra = f", beats noise in {beats}/{len(have)}" if METRICS[name] else ""
                print(f"  {name:<11} {got:g}/{of}{extra}")
    if held:
        print("\nHOLDOUT videos (reported for checking only; do not tune on them)")
        for v in held:
            print_video(v)


def edit_values(v: dict[str, Any]) -> dict[str, float]:
    """What the edits of a video contain, as metric-like rows."""
    out: dict[str, float] = {}
    for key in ("edit", "edit_highlights"):
        e = v.get(key)
        if not e:
            continue
        out[f"{key}_clips"] = e["clips"]
        out[f"{key}_secs"] = e["seconds"]
        for kind in ("keep", "skip", "reference"):
            if e[kind]["total"]:
                out[f"{key}_{kind[:4]}"] = len(e[kind]["contained"])
    return out


def compare_warnings(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Settings that differ between two runs, which makes the control and the rows incomparable."""
    return [
        f"{key} differs: {old.get(key)!r} -> {new.get(key)!r}"
        for key in ("schema", "seed", "runs", "min_overlap")
        if old.get(key) != new.get(key)
    ]


def compare(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    """Per video in both runs, the change in each metric value and in "beats noise", and in what
    the edits contain."""
    old_by = {v["video"].lower(): v for v in old["videos"]}
    out = []
    for v in new["videos"]:
        o = old_by.get(v["video"].lower())
        if o is None:
            continue
        rows = []
        for name, m in v["metrics"].items():
            if name not in o["metrics"]:
                continue
            om = o["metrics"][name]
            rows.append(
                {
                    "metric": name,
                    "old": om["value"],
                    "new": m["value"],
                    "delta": m["value"] - om["value"],
                    "old_beats_noise": om["beats_noise"],
                    "new_beats_noise": m["beats_noise"],
                }
            )
        old_edit, new_edit = edit_values(o), edit_values(v)
        for name, value in new_edit.items():
            if name in old_edit:
                rows.append(
                    {
                        "metric": name,
                        "old": old_edit[name],
                        "new": value,
                        "delta": value - old_edit[name],
                        "old_beats_noise": False,
                        "new_beats_noise": False,
                    }
                )
        out.append({"video": v["video"], "holdout": v["holdout"], "metrics": rows})
    return out


def print_compare(diff: list[dict[str, Any]], a: str, b: str, warnings: Sequence[str] = ()) -> None:
    print(f"\ncompare {a} -> {b}")
    for w in warnings:
        print(f"  WARNING: {w}; the runs are not comparable")
    if not diff:
        print("  no video in both runs")
    for v in diff:
        tag = "  [HOLDOUT]" if v["holdout"] else ""
        print(f"\n{v['video']}{tag}")
        for r in v["metrics"]:
            note = ""
            if r["old_beats_noise"] != r["new_beats_noise"]:
                note = "  now beats noise" if r["new_beats_noise"] else "  no longer beats noise"
            print(
                f"  {r['metric']:<20} {_fmt(r['metric'], r['old']):>6} -> "
                f"{_fmt(r['metric'], r['new']):>6}  ({_signed(r['metric'], r['delta'])}){note}"
            )


def _signed(name: str, d: float) -> str:
    if name.startswith("refsec"):
        return f"{d * 100:+.1f} points"
    return f"{d:+g}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", type=Path, action="append", default=[], metavar="DIR")
    ap.add_argument("--labels", type=Path, action="append", default=[], metavar="FILE")
    ap.add_argument("--manifest", type=Path, help="JSON listing project and labels pairs")
    ap.add_argument("--json", type=Path, metavar="FILE", help="write the results as JSON here")
    ap.add_argument("--compare", type=Path, nargs=2, metavar=("A", "B"), help="two --json files")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="random orders (default 1000)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="control seed (default 7)")
    ap.add_argument(
        "--min-overlap",
        type=float,
        default=DEFAULT_MIN_OVERLAP,
        help="share of the shorter of candidate and label range their overlap must reach to hit (default 0.5)",
    )
    ap.add_argument(
        "--edit-total",
        type=float,
        help="max_total for the default edit (default: none, the suggested top)",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="debug output")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.compare:
        if args.project or args.labels or args.manifest:
            ap.error("--compare takes two result files and no projects")
        try:
            a, b = (json.loads(p.read_text(encoding="utf-8")) for p in args.compare)
        except (OSError, ValueError) as err:
            logger.error("%s", err)
            return 1
        print_compare(
            compare(a, b), str(args.compare[0]), str(args.compare[1]), compare_warnings(a, b)
        )
        return 0
    if len(args.project) != len(args.labels):
        ap.error("--project and --labels must be given the same number of times")
    if args.runs < 1:
        ap.error("--runs must be at least 1")
    if not 0 < args.min_overlap <= 1:
        ap.error("--min-overlap must be above 0 and at most 1")
    pairs = list(zip(args.project, args.labels, strict=True))
    try:
        if args.manifest:
            pairs += load_manifest(args.manifest)
        if not pairs:
            ap.error("give --project and --labels, --manifest or --compare")
        videos = [
            evaluate_video(
                project,
                load_labels(labels),
                args.runs,
                args.seed,
                args.min_overlap,
                args.edit_total,
            )
            for project, labels in pairs
        ]
    except (OSError, ValueError, KeyError) as err:  # JSONDecodeError is a ValueError
        logger.error("%s: %s", type(err).__name__, err)
        return 1
    data = {
        "schema": SCHEMA,
        "seed": args.seed,
        "runs": args.runs,
        "min_overlap": args.min_overlap,
        "videos": videos,
    }
    print_results(data)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
