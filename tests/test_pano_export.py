"""``PanoExporter.export`` (the 360 clip hand-off) and ``frame_people`` (per-frame people
features). Synthetic clips and detections only; the ffmpeg render itself is replaced."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cutpoint.pano import geometry as G
from cutpoint.pano import project as P
from cutpoint.pano import render as R
from cutpoint.pano import views as VW
from cutpoint.pano.x3 import X3Clip


# --------------------------------------------------------------------------- export
@pytest.fixture
def exporter(tmp_path: Path, monkeypatch):
    """An exporter over one synthetic clip with the path planning and the render replaced by
    recorders. Returns ``(exporter, clip, calls)``."""
    clip = X3Clip("001", "20240101_100000", "5.7k", str(tmp_path / "VID_001.lrv"), ["a_00_.insv"])
    calls: dict[str, Any] = {"render": [], "path_for": [], "mode": "dfisheye", "preview": True}

    def resolve(c, prefer="auto"):
        return P.RenderSource(
            calls["mode"], ("f",), "dfisheye", False, calls["preview"], calls["mode"]
        )

    def render_segment(c, path, start, end, out, **kw):
        calls["render"].append(
            {"clip": c, "path": path, "start": start, "end": end, "out": out, **kw}
        )
        return {"seconds": 2.0, "realtime": 1.5, "read_error": calls.get("read_error"), "extra": 7}

    monkeypatch.setattr(R, "load", lambda p: {P.norm(clip.source): clip})
    monkeypatch.setattr(R, "resolve_source", resolve)
    monkeypatch.setattr(R, "render_segment", render_segment)
    ex = R.PanoExporter(tmp_path, tmp_path / "clips")
    path = {"keys": [], "planned": {"start": 1.0, "end": 5.0}}

    def path_for(c, name, start, end, kind, reuse=None, selected=None):
        calls["path_for"].append((c.clip_id, name, start, end, kind, reuse, selected))
        return path, tmp_path / "pano_paths" / "clips" / "001_a.json"

    monkeypatch.setattr(ex, "path_for", path_for)
    calls["path"] = path
    return ex, clip, calls


def test_export_plans_a_path_renders_it_and_returns_the_manifest_record(
    tmp_path: Path, exporter
) -> None:
    ex, clip, calls = exporter
    out = tmp_path / "clips" / "001_a.mp4"
    err, rec = ex.export(
        clip.source, 1.0, 5.0, out, audio_fade=0.3, kind="laughter", selected=(1.5, 4.5)
    )
    assert err is None
    assert calls["path_for"] == [("001", "001_a.mp4", 1.0, 5.0, "laughter", None, (1.5, 4.5))]
    (r,) = calls["render"]
    assert (r["clip"], r["path"], r["start"], r["end"], r["out"]) == (
        clip,
        calls["path"],
        1.0,
        5.0,
        out,
    )
    assert r["audio_fade"] == 0.3
    assert r["fmt"] == R.Format360() and r["work_dir"] == ex.paths_dir
    assert r["src"].mode == "dfisheye" and r["grade"] is None
    # the render's record plus the clip id, the path file and the format; read_error moves out
    assert rec == {
        "seconds": 2.0,
        "realtime": 1.5,
        "extra": 7,
        "clip": "001",
        "path_file": str(tmp_path / "pano_paths" / "clips" / "001_a.json"),
        "format": [1920, 1080],
    }


def test_export_returns_the_read_error_of_the_render_apart_from_the_record(
    tmp_path: Path, exporter
) -> None:
    ex, clip, calls = exporter
    calls["read_error"] = {"at": 3.2, "error": "bad packet"}
    err, rec = ex.export(clip.source, 1.0, 5.0, tmp_path / "001_a.mp4", audio_fade=0.0)
    assert err == {"at": 3.2, "error": "bad packet"}
    assert "read_error" not in rec


def test_a_vertical_exporter_renders_9_16(tmp_path: Path, exporter) -> None:
    ex, clip, calls = exporter
    ex.fmt = R.Format360(1080, 1920)
    _, rec = ex.export(clip.source, 1.0, 5.0, tmp_path / "001_a.mp4", audio_fade=0.0)
    assert rec["format"] == [1080, 1920]
    assert calls["render"][0]["fmt"].height == 1920


def test_a_source_that_is_not_an_x3_clip_of_the_project_is_refused(
    tmp_path: Path, exporter
) -> None:
    ex, _clip, calls = exporter
    with pytest.raises(ValueError, match="not an X3 clip"):
        ex.export(str(tmp_path / "plain.mp4"), 0.0, 3.0, tmp_path / "x.mp4", audio_fade=0.0)
    assert calls["render"] == [] and calls["path_for"] == []


def test_the_preview_quality_warning_is_logged_once_per_exporter(
    tmp_path: Path, exporter, caplog
) -> None:
    ex, clip, calls = exporter
    with caplog.at_level(logging.WARNING, logger=R.logger.name):
        for k in range(2):
            ex.export(clip.source, 1.0, 5.0, tmp_path / f"00{k}_a.mp4", audio_fade=0.0)
    assert caplog.text.count("PREVIEW quality") == 1
    caplog.clear()
    calls["preview"] = False
    ex._warned = False
    with caplog.at_level(logging.WARNING, logger=R.logger.name):
        ex.export(clip.source, 1.0, 5.0, tmp_path / "003_a.mp4", audio_fade=0.0)
    assert caplog.text == ""  # a stitched source is no preview


# --------------------------------------------------------------------------- frame_people
def det(frame: int, label: int, score: float, height: float, ray) -> list[float]:
    """A ``DET_COLS`` row: frame, label, score, yaw, pitch, width, height, wearer deg, ray."""
    return [frame, label, score, 0.0, 0.0, height / 2, height, 0.0, *ray]


AWAY = (1.0, 0.0, 0.0)  # ~87 deg from the wearer's direction: another person
WEARER = tuple(G.WEARER_V)


def views(rows: list[list[float]], n: int = 3) -> VW.PanoViews:
    d = np.array(rows, np.float32).reshape(-1, 11)
    return VW.PanoViews(
        t=np.arange(n, dtype=np.float32),
        emb=np.zeros((n, 9, 768), np.float16),
        det=d,
        box_emb=np.zeros((0, 768), np.float16),
        box_idx=np.zeros(0, np.int64),
    )


def test_frame_people_counts_others_and_measures_the_nearest_and_sees_the_wearer() -> None:
    close_deg = VW.PeopleRules360().close_height_deg
    pv = views(
        [
            det(0, 1, 0.9, 20.0, AWAY),
            det(0, 1, 0.9, 40.0, AWAY),  # the nearest of two others
            det(1, 1, 0.9, 40.0, WEARER),  # the wearer, not another person
            det(2, 1, 0.3, 30.0, AWAY),  # too unsure
            det(2, 1, 0.9, 3.0, AWAY),  # too small
            det(2, 2, 0.9, 30.0, AWAY),  # not a person
        ]
    )
    out = VW.frame_people(pv)
    assert out["count"].tolist() == [2.0, 0.0, 0.0]
    assert out["close"] == pytest.approx([40.0 / close_deg, 0.0, 0.0])
    assert out["self"].tolist() == [0.0, 1.0, 0.0]
    assert all(v.dtype == np.float32 for v in out.values())


def test_a_person_taller_than_the_close_limit_has_close_one() -> None:
    out = VW.frame_people(views([det(0, 1, 0.9, 60.0, AWAY)], n=1))
    assert out["close"].tolist() == [1.0]


def test_the_people_rules_decide_who_counts() -> None:
    pv = views([det(0, 1, 0.3, 30.0, AWAY)], n=1)
    assert VW.frame_people(pv)["count"].tolist() == [0.0]
    assert VW.frame_people(pv, VW.PeopleRules360(min_score=0.2))["count"].tolist() == [1.0]


def test_no_detections_give_all_zero_features() -> None:
    out = VW.frame_people(views([], n=4))
    assert set(out) == {"count", "close", "self"}
    assert all(v.shape == (4,) and not v.any() for v in out.values())
