"""Development harness for ``--color`` (#290 stage 2): how each mode and levels variant grades
chosen windows of real footage. Not part of the package; it writes nothing to Cutpoint's
cache or to any project.

For every window it samples the frames an export would measure (:func:`cutpoint.color.sample_stats`,
timed: all-intra sources decode every frame), runs :func:`cutpoint.color.estimate` for each mode
(``auto``, ``underwater``) and levels variant (``luma``, the production one, and the
experimental per-channel ``channel``) and prints one table per window:

- mode, levels, skip reason (``-`` when graded; ``good``, ``log``, ``veto_*``)
- gains (r g b), black and white point(s), red-from-green mix, vibrance
- M3 colourfulness and gray-world residual, before -> predicted
- predicted clipped share against the share the levels intend (the veto allows +0.5 points),
  by the variant's own rule and by the per-channel rule (``int/ch``), so the variants can be
  compared on one rule
- median R/G before -> after (scuba target 0.8-0.9)
- mean CIE76 delta E between the sampled frames and their simulated grade

Vetoed grades are simulated anyway (without the veto), so the numbers and sheets show what
the veto prevented.

Usage::

    uv run python scripts/color_eval.py D:/Videos/GX010801.MP4@120-135 clip.mp4@0:42-0:50
    uv run python scripts/color_eval.py --windows windows.json --sheets out/ --json

A window is ``SOURCE@START-END`` (seconds or ``[h:]m:s``), or a JSON list of
``{"source", "start", "end", "label"}`` objects (``--windows``). ``--sheets DIR`` writes
contact sheets (PNG, 4x2 frames of the window's inner 80 %, rendered by ffmpeg through the
export chain with :meth:`cutpoint.color.AutoGrade.filter`): ``<label>_before.png`` and
``<label>_<mode>_<levels>.png``. ``--json`` prints the results as JSON instead of tables.
``--strength``, ``--vertical`` and ``--crop W:H:X:Y`` are as on ``cutpoint clips``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import color
from cutpoint.crop import Crop, parse_spec
from cutpoint.encode import bt709_filter, choose_format
from cutpoint.ffmpeg_utils import probe, run
from cutpoint.utils import slugify

SHEET_TILE = (4, 2)
SHEET_WIDTH = 480  # px per frame

# BT.709 / sRGB primaries, D65 (the sampled frames are BT.709 RGB).
_RGB_TO_XYZ = np.array(
    [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32
)
_D65 = np.array([0.95047, 1.0, 1.08883], np.float32)


def _seconds(text: str) -> float:
    """``12.5``, ``1:02.5`` or ``1:00:02.5`` as seconds."""
    total = 0.0
    for part in text.strip().split(":"):
        total = total * 60 + float(part)
    return total


def parse_window(spec: str) -> dict[str, Any]:
    """``SOURCE@START-END`` as a window dict."""
    m = re.fullmatch(r"(.+)@([0-9:.]+)-([0-9:.]+)", spec)
    if not m:
        raise argparse.ArgumentTypeError(f"{spec!r}: expected SOURCE@START-END")
    return {"source": m[1], "start": _seconds(m[2]), "end": _seconds(m[3])}


def _label(w: dict[str, Any]) -> str:
    stem = Path(w["source"]).stem
    return w.get("label") or slugify(f"{stem}_{w['start']:.0f}-{w['end']:.0f}s", 60)


def _lab(rgb: np.ndarray) -> np.ndarray:
    """CIE Lab of float 0-1 RGB (sRGB transfer as an approximation of BT.709's)."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    t = (lin @ _RGB_TO_XYZ.T) / _D65
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def _rg(rgb: np.ndarray) -> float:
    """Median over the frames of mean red / mean green."""
    means = rgb.reshape(len(rgb), -1, 3).mean(axis=1)
    return float(np.median(means[:, 0] / np.maximum(means[:, 1], 1e-4)))


def _as_channel(g: color.AutoGrade) -> color.AutoGrade:
    """``g`` with its points per channel (the same values), for the per-channel intended
    clipping rule."""
    b, w = color._rgb(g.black), color._rgb(g.white)
    return replace(g, black=b, white=w)


def evaluate(frames: np.ndarray, grade: color.AutoGrade) -> dict[str, Any]:
    """The numbers of one grade on the sampled ``frames`` (a vetoed grade is simulated as if
    it were not)."""
    before = color._float(frames)
    rec: dict[str, Any] = {
        "skip": grade.skip,
        "params": grade.params(),
        "before": color.metrics(frames).to_json(),
        "rg_before": round(_rg(before), 3),
    }
    if grade.skip in (color.SKIP_LOG, color.SKIP_NO_FRAMES, color.SKIP_360):
        return rec
    sim = replace(grade, skip=None)
    after = color.simulate(frames, sim)
    rec |= {
        "predicted": color.metrics(after).to_json(),
        "intended_clipped": color._intended_clipping(before, sim),
        # The per-channel rule for every variant, so luma and channel compare on one rule.
        "intended_clipped_channel": color._intended_clipping(before, _as_channel(sim)),
        "rg_after": round(_rg(after), 3),
        "delta_e": round(float(np.linalg.norm(_lab(after) - _lab(before), axis=-1).mean()), 2),
        "filter": sim.filter(),
    }
    return rec


def _sheet(
    src: Path, start: float, end: float, chain: str, out: Path, vertical: bool, crop: Crop | None
) -> None:
    """A contact sheet of ``[start, end]`` (inner 80 %) through the export chain with
    ``chain`` as its colour filters."""
    info = probe(src)
    fmt = choose_format([info], vertical)
    dur = end - start
    inner = dur * color.SAMPLE_INNER
    n = SHEET_TILE[0] * SHEET_TILE[1]
    vf = (
        f"{fmt.video_filter(vertical, crop, chain)},fps={n / inner:.6f},"
        f"scale={SHEET_WIDTH}:-2:in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24,"
        f"tile={SHEET_TILE[0]}x{SHEET_TILE[1]}"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start + (dur - inner) / 2:.3f}",
            "-t",
            f"{inner:.3f}",
            "-i",
            str(src),
            "-vf",
            vf,
            "-frames:v",
            "1",
            str(out),
        ]
    )


def run_window(w: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Sample one window and evaluate every mode and levels variant on it."""
    src = Path(w["source"])
    info = probe(src)
    fmt = choose_format([info], args.vertical)
    crop = args.crop
    t0 = time.perf_counter()
    stats = color.sample_stats(src, info, w["start"], w["end"], fmt, args.vertical, crop)
    took = time.perf_counter() - t0
    label = _label(w)
    result: dict[str, Any] = {
        "label": label,
        "source": str(src),
        "start": w["start"],
        "end": w["end"],
        "frames": len(stats.frames),
        "sample_s": round(took, 2),
        "high_bit": stats.high_bit,
        "looks_underwater": color.looks_underwater(stats),
        "variants": [],
    }
    if args.sheets:
        args.sheets.mkdir(parents=True, exist_ok=True)
        base = bt709_filter(info)
        _sheet(
            src,
            w["start"],
            w["end"],
            base,
            args.sheets / f"{label}_before.png",
            args.vertical,
            crop,
        )
    for mode in args.modes:
        for levels in args.levels:
            grade = color.estimate(stats, mode, args.strength, levels)
            rec = {"mode": mode, "levels": levels, **evaluate(stats.frames, grade)}
            if args.sheets and rec.get("filter"):
                chain = ",".join(f for f in (bt709_filter(info), rec["filter"]) if f)
                out = args.sheets / f"{label}_{mode}_{levels}.png"
                _sheet(src, w["start"], w["end"], chain, out, args.vertical, crop)
                rec["sheet"] = str(out)
            result["variants"].append(rec)
    return result


def _point(v: Any) -> str:
    return "/".join(f"{x:.3f}" for x in v) if isinstance(v, list) else f"{v:.3f}"


def print_table(r: dict[str, Any]) -> None:
    """One window's results as a text table."""
    print(
        f"\n== {r['label']}  {r['source']} {r['start']:g}-{r['end']:g} s: {r['frames']} frames "
        f"sampled in {r['sample_s']:.2f} s{', 10-bit' if r['high_bit'] else ''}"
        f"{', looks underwater' if r['looks_underwater'] else ''}"
    )
    head = (
        f"{'mode':<10} {'levels':<7} {'skip':<12} {'gains r/g/b':<17} {'black':<17} "
        f"{'white':<17} {'mix':>5} {'vib':>5}  {'M3 bef->pred':<15} {'resid bef->pred':<15} "
        f"{'clip% pred/int/ch':<18} {'R/G bef->aft':<12} {'dE':>5}"
    )
    print(head)
    print("-" * len(head))
    for v in r["variants"]:
        p, b, q = v["params"], v["before"], v.get("predicted")
        gains = "/".join(f"{g:.3f}" for g in p["gains"])
        m3 = f"{b['m3']:.1f}->{q['m3']:.1f}" if q else f"{b['m3']:.1f}"
        res = f"{b['residual']:.4f}->{q['residual']:.4f}" if q else f"{b['residual']:.4f}"
        clip = (
            f"{100 * q['clipped']:.2f}/{100 * v['intended_clipped']:.2f}"
            f"/{100 * v['intended_clipped_channel']:.2f}"
            if q
            else "-"
        )
        rg = f"{v['rg_before']:.3f}->{v['rg_after']:.3f}" if q else f"{v['rg_before']:.3f}"
        de = f"{v['delta_e']:.2f}" if q else "-"
        print(
            f"{v['mode']:<10} {v['levels']:<7} {v['skip'] or '-':<12} {gains:<17} "
            f"{_point(p['black']):<17} {_point(p['white']):<17} {p['mix']:>5.3f} "
            f"{p['vibrance']:>5.3f}  {m3:<15} {res:<15} {clip:<18} {rg:<12} {de:>5}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("windows", nargs="*", type=parse_window, help="SOURCE@START-END")
    ap.add_argument("--windows", dest="windows_file", type=Path, help="JSON list of windows")
    ap.add_argument("--modes", nargs="+", choices=color.MODES, default=list(color.MODES))
    ap.add_argument("--levels", nargs="+", choices=color.LEVELS, default=list(color.LEVELS))
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--vertical", action="store_true")
    ap.add_argument("--crop", type=parse_spec, help="W:H:X:Y crop, as clips --crop")
    ap.add_argument("--sheets", type=Path, metavar="DIR", help="write contact sheets here")
    ap.add_argument("--json", action="store_true", help="print JSON instead of tables")
    args = ap.parse_args(argv)
    if args.crop is not None and not isinstance(args.crop, Crop):
        ap.error("--crop: give W:H:X:Y")
    windows = list(args.windows)
    if args.windows_file:
        windows += json.loads(args.windows_file.read_text(encoding="utf-8"))
    if not windows:
        ap.error("give at least one window (SOURCE@START-END or --windows FILE)")
    results = [run_window(w, args) for w in windows]
    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    else:
        for r in results:
            print_table(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
