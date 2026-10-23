"""Contact sheets of rendered 360 clips (``cutpoint 360 check``).

One row per clip, ``per_clip`` frames evenly spread, each captioned with the
clip time and the director target (friend / selfie / forward / view), yaw and
pitch at that moment, so seam placement, horizon level, framing and pans can
be checked by eye. Sheets go to ``<clips_dir>_check/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from cutpoint import cache
from cutpoint.utils import clip_files

logger = logging.getLogger(__name__)


def clip_sheets(
    clips_dir: Path, per_clip: int = 6, clips_per_sheet: int = 6, width: int = 320
) -> list[Path]:
    """Write contact sheets for the ``.mp4`` files in ``clips_dir``; returns their paths."""
    from cutpoint.events.sheets import contact_sheet  # noqa: PLC0415
    from cutpoint.ffmpeg_utils import probe  # noqa: PLC0415
    from cutpoint.pano.director import evaluate  # noqa: PLC0415

    manifest = cache.load_json(clips_dir / "manifest.json") or {}
    by_file = {e["file"]: e for e in manifest.get("clips", [])}
    files = clip_files(clips_dir)
    out_dir = clips_dir.with_name(clips_dir.name + "_check")
    sheets: list[Path] = []
    for k in range(0, len(files), clips_per_sheet):
        items = []
        for f in files[k : k + clips_per_sheet]:
            dur = probe(f).duration
            e = by_file.get(f.name, {})
            path = None
            pf = (e.get("pano") or {}).get("path_file")
            if pf and Path(pf).is_file():
                path = json.loads(Path(pf).read_text(encoding="utf-8"))
            for t in np.linspace(0.3, max(0.4, dur - 0.3), per_clip):
                cap = f"{f.stem[:22]} {t:.1f}s"
                if path is not None:
                    src_t = float(e["start"]) + t
                    y, p, _ = evaluate(path, np.array([src_t]))
                    key = min(path["keys"], key=lambda q: abs(q["t"] - src_t))
                    cap += f" {key['target']} y{y[0]:.0f} p{p[0]:.0f}"
                items.append({"path": str(f), "t": float(t), "caption": cap})
        out = out_dir / f"sheet_{k // clips_per_sheet + 1:02d}.jpg"
        if contact_sheet(items, out, cols=per_clip, width=width) is not None:
            sheets.append(out)
    return sheets
