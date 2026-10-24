"""Geometric direction QA (``cutpoint 360 qa --paths``), ``project_box`` and the
director plumbing (``--param``, ``pano.director``, ``--out-name``, ``quality``).

No models or footage: view-layer detections and director paths are synthetic.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import set_config
from cutpoint import paths
from cutpoint.pano import cli
from cutpoint.pano import director as D
from cutpoint.pano import geometry as G
from cutpoint.pano import project as P
from cutpoint.pano import qa_paths as QP
from cutpoint.pano import views as VW
from cutpoint.pano.qa import QAThresholds
from pano_helpers import make_views

W, H = 1920, 1080
TH = QAThresholds()
FWD_V = [0.0, -1.0, 0.0]  # camera ray far from the wearer cone
WEAR_V = list(G.WEARER_V)


# --------------------------------------------------------------------------- project_box
def test_project_box_centre_lands_at_frame_centre() -> None:
    b = G.project_box(30.0, -10.0, 86.0, W, H, 30.0, -10.0, 4.0, 4.0)
    assert b is not None
    assert (b[0] + b[2]) / 2 == pytest.approx(0.5, abs=1e-6)
    assert (b[1] + b[3]) / 2 == pytest.approx(0.5, abs=1e-3)


def test_project_box_directions_follow_display_frame() -> None:
    right = G.project_box(0.0, 0.0, 86.0, W, H, 20.0, 0.0, 2.0, 2.0)
    up = G.project_box(0.0, 0.0, 86.0, W, H, 0.0, 15.0, 2.0, 2.0)
    assert right is not None and up is not None
    assert right[0] > 0.5  # + yaw = right
    assert up[3] < 0.5  # + pitch = up (y down)
    assert G.project_box(0.0, 0.0, 86.0, W, H, 180.0, 0.0, 10.0, 10.0) is None


def test_project_box_across_the_yaw_seam_mirrors_the_plain_case() -> None:
    # 179 is 3 deg left of -178, as -1 is of 2
    a = G.project_box(-178.0, -8.0, 86.0, W, H, 179.0, -10.0, 20.0, 30.0)
    b = G.project_box(2.0, -8.0, 86.0, W, H, -1.0, -10.0, 20.0, 30.0)
    assert a is not None and b is not None
    assert a == pytest.approx(b, abs=1e-6)


def test_project_box_cut_by_the_frame_edge_keeps_its_visible_height() -> None:
    # F1: the box centre (yaw 60) is outside an 86-deg view; the far part must not stretch y
    b = G.project_box(0.0, 0.0, 86.0, W, H, 60.0, 0.0, 80.0, 20.0)
    assert b is not None
    assert 0.25 <= b[1] <= 0.30 and 0.70 <= b[3] <= 0.75
    assert 0.65 < b[0] < 0.75 and b[2] > 1.0  # left edge: yaw 20
    vis = G.project_box_visible(0.0, 0.0, 86.0, W, H, 60.0, 0.0, 80.0, 20.0)
    # ~70 % of its yaw span is outside and the far part is stretched: visible share ~0.13,
    # which the QA drops
    # (keeping such slivers, MIN_VISIBLE 0.10-0.15, tracked the rendered QA worse)
    assert vis is not None and 0.0 < vis[1] < QP.MIN_VISIBLE
    inside = G.project_box_visible(0.0, 0.0, 86.0, W, H, 30.0, 0.0, 40.0, 20.0)
    assert inside is not None and inside[1] >= QP.MIN_VISIBLE


def test_project_box_near_the_pole_keeps_a_width() -> None:
    # F2: the yaw line is kept off the pole, where every yaw is the same point
    b = G.project_box(0.0, -70.0, 86.0, W, H, 0.0, -85.0, 20.0, 20.0)
    assert b is not None
    assert b[2] - b[0] > 0.01


def test_project_box_fov_edges_match_flat_fov() -> None:
    # a box exactly as wide as the hfov spans the frame; one at vfov/2 above touches the top
    b = G.project_box(0.0, 0.0, 90.0, W, H, 0.0, 0.0, 90.0, 0.0)
    assert b is not None
    assert b[0] == pytest.approx(0.0, abs=1e-6)
    assert b[2] == pytest.approx(1.0, abs=1e-6)
    v = G.flat_fov(90.0, W, H)
    top = G.project_box(0.0, 0.0, 90.0, W, H, 0.0, v / 2, 0.01, 0.0)
    assert top is not None
    assert top[1] == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("u,v", [(0.8, 0.3), (0.1, 0.9), (0.55, 0.05)])
def test_project_box_inverts_the_renderer_rays(u: float, v: float) -> None:
    """A ray through output pixel (u, v) (``flat_rays`` + ``view_matrix``, as v360) projects back there."""
    vy, vp, hfov = 35.0, -12.0, 86.0
    vf = G.flat_fov(hfov, W, H)
    d = np.array(
        [
            (2 * u - 1) * math.tan(math.radians(hfov) / 2),
            (2 * v - 1) * math.tan(math.radians(vf) / 2),
            1.0,
        ]
    )
    world = G.view_matrix(vy, vp) @ (d / np.linalg.norm(d))
    by, bp = G.ray_yaw_pitch(world)
    b = G.project_box(vy, vp, hfov, W, H, float(by), float(bp), 1e-3, 1e-3)
    assert b is not None
    assert (b[0] + b[2]) / 2 == pytest.approx(u, abs=1e-4)
    assert (b[1] + b[3]) / 2 == pytest.approx(v, abs=1e-4)


# --------------------------------------------------------------------------- synthetic views
def fixed_path(yaw: float = 0.0, pitch: float = -8.0, fov: float = 86.0, end: float = 12.0) -> dict:
    keys = [
        {"t": float(t), "yaw": yaw, "pitch": pitch, "fov": fov, "target": "friend"}
        for t in np.arange(0.0, end + 1e-6, 0.5)
    ]
    return {"keys": keys, "cuts": [], "segment": {"start": 0.0, "end": end}, "source": ""}


def flags_of(pv: VW.PanoViews, path: dict, t: float = 2.4) -> dict:
    times = np.array([t])
    return QP.geo_frames(pv, path, times, QP.view_frame_index(pv.t.astype(float), times), TH)[0]


FRIEND = (0.99, 0.0, -8.0, 12.0, 28.0, FWD_V)  # ~0.5 of a 86-deg frame tall


def test_well_framed_friend_has_no_flags() -> None:
    r = flags_of(make_views({2: [FRIEND]}, source="a.lrv"), fixed_path())
    assert r["flags"] == []
    assert r["subj"] is not None
    assert TH.size_min_h <= r["subj"][3] - r["subj"][1] <= TH.size_max_h
    assert r["safe"] is True


def test_friend_at_the_side_is_edge_cut() -> None:
    # the view turned so the friend straddles the left edge
    r = flags_of(make_views({2: [FRIEND]}, source="b.lrv"), fixed_path(yaw=43.0))
    assert "edge_cut" in r["flags"]
    assert r["subj"][0] == 0.0


def test_close_person_is_too_close_and_wearer_is_occluded() -> None:
    close = (0.99, 0.0, -15.0, 70.0, 70.0, FWD_V)
    assert "too_close" in flags_of(make_views({2: [close]}, source="c.lrv"), fixed_path())["flags"]
    wearer = (0.99, 0.0, -20.0, 60.0, 60.0, WEAR_V)  # in the wearer cone: a self box
    r = flags_of(make_views({2: [wearer]}, source="d.lrv"), fixed_path())
    assert "occluded" in r["flags"]
    assert r["subj"] is None
    assert len(r["self"]) == 1


def test_mostly_outside_boxes_dropped_and_duplicates_suppressed() -> None:
    sliver = (0.99, 70.0, -8.0, 60.0, 28.0, FWD_V)  # centre outside the 86-deg view
    dup = (0.80, 0.5, -8.0, 12.5, 28.0, FWD_V)
    r = flags_of(make_views({2: [sliver, FRIEND, dup]}, source="e.lrv"), fixed_path())
    assert len(r["boxes"]) == 1
    assert r["boxes"][0][0] == pytest.approx(0.99)  # the higher score survives NMS


def test_close_person_mostly_outside_the_frame_is_an_edge_cut() -> None:
    # inches from the lens: a small share of its box, a big part of the frame at the edge
    close = (0.99, 70.0, -10.0, 60.0, 60.0, FWD_V)
    r = flags_of(make_views({2: [close]}, source="k.lrv"), fixed_path())
    assert r["boxes"] == []  # dropped as a sliver (MIN_VISIBLE) ...
    assert r["flags"] == ["edge_cut"]  # ... but still an edge cut
    far = (0.99, 70.0, -8.0, 60.0, 28.0, FWD_V)  # a thin sliver: nothing
    assert flags_of(make_views({2: [far]}, source="l.lrv"), fixed_path())["flags"] == []


def test_tall_box_outside_the_wearer_cone_is_a_close_friend_not_self() -> None:
    tall = (0.99, 0.0, -20.0, 80.0, 80.0, FWD_V)  # >70 deg tall: views.is_wearer would say wearer
    r = flags_of(make_views({2: [tall]}, source="h.lrv"), fixed_path())
    assert r["self"] == []
    assert r["subj"] is not None
    assert "too_close" in r["flags"]
    assert "occluded" not in r["flags"]


def test_frames_far_from_any_view_frame_have_no_detections() -> None:
    pv = make_views({2: [FRIEND]}, n=3, source="f.lrv")
    idx = QP.view_frame_index(pv.t.astype(float), np.array([2.4, 9.0]))
    assert idx.tolist() == [2, -1]


def test_run_paths_qa_on_a_paths_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = tmp_path / "pano_paths" / "trial"
    d.mkdir(parents=True)
    good = {**fixed_path(), "source": "g.lrv", "kind": "laughter"}
    bad = {**fixed_path(yaw=43.0), "source": "g.lrv"}
    (d / "001_a.json").write_text(json.dumps(good))
    (d / "002_b.json").write_text(json.dumps(bad))
    (d / "notes.json").write_text(json.dumps({"hello": 1}))  # not a path
    (d / "003_c.json").write_text(json.dumps({**fixed_path(), "source": "other.lrv"}))
    pv = make_views({f: [FRIEND] for f in range(10)}, source="g.lrv")
    monkeypatch.setattr(
        VW, "load_views", lambda src, params=None: pv if str(src) == "g.lrv" else None
    )
    rep = QP.run_paths_qa(d)
    assert rep["mode"] == "paths"
    assert Path(rep["out_dir"]) == d.with_name("trial_paths_qa")
    saved = json.loads((Path(rep["out_dir"]) / "qa.json").read_text(encoding="utf-8"))
    assert saved["mode"] == "paths"
    assert [c["file"] for c in saved["clips"]] == ["001_a.json", "002_b.json"]
    assert saved["skipped"] == [{"file": "003_c.json", "error": "no 360 view cache"}]
    a, b = saved["clips"]
    assert a["pass"] is True and a["score"] == 100.0
    assert a["share"]["subject_present"] == 1.0
    assert len(a["frames"]) == 24  # 12 s at 2 fps
    assert a["frames"][0]["occl_zs"] is None
    assert b["pass"] is False and b["share"]["edge_cut"] > 0.5
    assert a["pan"]["speed_max"] == 0.0
    assert saved["edit"]["clips"] == 2


def test_find_path_inputs_from_a_manifest(tmp_path: Path) -> None:
    d = tmp_path / "clips"
    d.mkdir()
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps(fixed_path()))
    man = {
        "clips": [
            {
                "file": "001_x.mp4",
                "start": 5.0,
                "end": 9.0,
                "duration": 4.0,
                "pano": {"path_file": str(pf)},
            },
            {
                "file": "002_y.mp4",
                "start": 1.0,
                "end": 2.0,
                "pano": {"path_file": str(tmp_path / "gone.json")},
            },
        ]
    }
    (d / "manifest.json").write_text(json.dumps(man))
    items = QP.find_path_inputs(d)
    assert [(i["file"], i["offset"], i["duration"]) for i in items] == [
        ("001_x.mp4", 5.0, 4.0),
        ("002_y.mp4", 1.0, 1.0),
    ]
    assert items[0]["path"]["keys"]
    assert items[1]["path"] is None and items[1]["path_missing"] is True


# --------------------------------------------------------------------------- quality block
def test_compute_path_writes_quality_block(monkeypatch: pytest.MonkeyPatch) -> None:
    n, nb = 25, 36
    times = np.arange(n) * 0.5
    ys = np.zeros((n, nb))
    ys[:, 0] = 1.0  # always look at yaw 0
    monkeypatch.setattr(
        D,
        "score_grid",
        lambda clip, t0, t1, p, pv, orient=None, vertical=False: {
            "times": times,
            "centres": np.arange(nb) * 10.0,
            "yaw_score": ys,
            "forward_yaw": 0.0,
            "selfie": np.zeros(n),
            "person_pitch": np.full((n, nb), -8.0),
            "people": np.where(ys > 0, 0.5, 0.0),
            "talk": np.zeros(n),
        },
    )
    clip = SimpleNamespace(source="q.lrv", duration=12.0, clip_id="001")
    orient = G.Orientation(
        t=np.arange(400) / 30.0,
        up=np.tile([0.0, 0.0, -1.0], (400, 1)),
        heading=np.zeros(400),
        heading_smooth=np.zeros(400),
        meta={"fps": 30.0},
    )
    pv = make_views({f: [FRIEND] for f in range(10)}, source="q.lrv")
    path = D.compute_path(clip, 2.0, 8.0, orient=orient, pv=pv, margin=0.0)
    sus = D.compute_path(clip, 2.0, 8.0, orient=orient, pv=pv, margin=0.0, suspects=True)
    assert sus.pop("vl_suspect") == []  # the framed friend is no suspect
    assert sus == path
    q = path["quality"]
    assert q["mode"] == "paths"
    assert q["frames"] == 12
    assert q["subject_present"] == 1.0
    assert q["edge_cut"] == 0.0 and q["too_close"] == 0.0 and q["occluded"] == 0.0
    assert q["pass"] is True
    assert "quality 100 ok" in D.describe_quality(path)
    assert D.describe_quality({"keys": []}) == ""
    # user overrides win over the kind tweaks (laughter sets w_selfie_talk 0.8)
    tweaked = D.compute_path(clip, 2.0, 8.0, kind="laughter", orient=orient, pv=pv, margin=0.0)
    assert tweaked["params"]["w_selfie_talk"] == 0.8
    over = D.compute_path(
        clip,
        2.0,
        8.0,
        kind="laughter",
        overrides={"w_selfie_talk": "0.1", "max_speed": 30},
        orient=orient,
        pv=pv,
        margin=0.0,
    )
    assert over["params"]["w_selfie_talk"] == 0.1
    assert over["params"]["max_speed"] == 30.0


# --------------------------------------------------------------------------- params
def test_param_args_and_coercion() -> None:
    kv = D.parse_param_args(["max_speed=35", "n_bins = 24", "min_hold_s=3"])
    p = D.with_params(D.DirectorParams(), kv)
    assert p.max_speed == 35.0 and isinstance(p.max_speed, float)
    assert p.n_bins == 24 and isinstance(p.n_bins, int)
    assert p.min_hold_s == 3.0
    assert D.with_params(p, {"n_bins": 12.0}).n_bins == 12  # JSON number


@pytest.mark.parametrize(
    "items,match",
    [
        (["max_speed"], "KEY=VALUE"),
        (["nope=1"], "unknown director param"),
        (["max_speed=fast"], "not a valid float"),
        (["n_bins=2.5"], "not a valid int"),
        (["max_speed=-5"], "must be > 0"),
    ],
)
def test_param_validation(items: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        D.with_params(D.DirectorParams(), D.parse_param_args(items))


def _config(data: dict) -> None:
    set_config(paths.home(), **data)


def test_director_params_config_then_cli() -> None:
    assert cli.director_params([]) is None  # nothing set: the defaults
    _config({"pano": {"director": {"max_speed": 30, "min_hold_s": 4.0}}})
    o = cli.director_params(["max_speed=50"])
    assert o == {"max_speed": "50", "min_hold_s": 4.0}
    p = D.with_params(D.DirectorParams(), o)
    assert p.max_speed == 50.0 and p.min_hold_s == 4.0
    _config({"pano": {"director": {"dt": 0}}})
    with pytest.raises(ValueError, match="must be > 0"):
        cli.director_params([])
    _config({"pano": {"director": {"bogus": 1}}})
    with pytest.raises(ValueError, match="unknown director param"):
        cli.director_params([])


# --------------------------------------------------------------------------- cmd_director routing
def _run_director(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clip: SimpleNamespace | None = None,
    segments: list[dict] | None = None,
    **kw,
) -> tuple[Path, list]:
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    seen: list = []
    fake_clip = clip or SimpleNamespace(clip_id="001", lrv="x.lrv")
    monkeypatch.setattr(cli, "_project", lambda arg: proj)
    monkeypatch.setattr(P, "load", lambda project: {"x.lrv": fake_clip})
    monkeypatch.setattr(P, "clip_for", lambda clips, src: fake_clip)
    monkeypatch.setattr(
        cli,
        "_selected_segments",
        lambda project, a: (
            segments or [{"source": "x.lrv", "start": 1.0, "end": 4.0, "label": "laugh"}]
        ),
    )

    def fake_compute(clip, start, end, *, vertical=False, kind=None, overrides=None, **_):
        seen.append(overrides)
        return {
            "keys": [],
            "cuts": [],
            "quality": {
                **dict.fromkeys(
                    (
                        "occluded",
                        "too_close",
                        "edge_cut",
                        "subject_present",
                        "size_ok",
                        "safe_area",
                    ),
                    0.0,
                ),
                "score": 100.0,
                "pass": True,
                "fails": [],
            },
        }

    monkeypatch.setattr(D, "compute_path", fake_compute)
    a = argparse.Namespace(project="proj", vertical=False, preview=False, param=[], out_name=None)
    for k, v in kw.items():
        setattr(a, k, v)
    cli.cmd_director(a)
    return proj, seen


def test_cmd_director_default_and_out_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proj, seen = _run_director(tmp_path, monkeypatch)
    assert (proj / "pano_paths" / "director" / "001_laugh.json").is_file()
    assert seen == [None]  # no params: unchanged behaviour
    proj, seen = _run_director(tmp_path, monkeypatch, out_name="trial_a", param=["max_speed=35"])
    assert (proj / "pano_paths" / "trial_a" / "001_laugh.json").is_file()
    assert seen == [{"max_speed": "35"}]


def test_cmd_director_out_name_removes_only_stale_recorded_stems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    out = tmp_path / "proj" / "pano_paths" / "mix"
    out.mkdir(parents=True)
    for f in ("002_run.mp4", "002_run.cmd", "003_mine.json", "notes.txt"):
        (out / f).write_text("x")
    (out / "002_run.json").write_text('{"keys": ["hand-edited"]}')
    (out / "director_stems.json").write_text('["001_laugh", "002_run"]')  # an earlier run
    _run_director(tmp_path, monkeypatch, out_name="mix")
    left = sorted(f.name for f in out.iterdir())
    assert left == [
        "001_laugh.json",
        "002_run.json.bak",
        "003_mine.json",
        "director_stems.json",
        "notes.txt",
    ]
    assert (out / "002_run.json.bak").read_text() == '{"keys": ["hand-edited"]}'
    assert "no longer selected" in caplog.text and "002_run.json.bak" in caplog.text


@pytest.mark.parametrize("failing", ["backup", "unlink"])
def test_cmd_director_keeps_stale_stem_until_its_files_are_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    out = tmp_path / "proj" / "pano_paths" / "mix"
    out.mkdir(parents=True)
    (out / "002_run.mp4").write_text("x")
    (out / "002_run.json").write_text('{"keys": ["hand-edited"]}')
    (out / "director_stems.json").write_text('["001_laugh", "002_run"]')
    with monkeypatch.context() as m:
        if failing == "backup":
            real = shutil.copy2

            def copy2(src, dst, **kw):
                if Path(src).name == "002_run.json":
                    raise OSError("locked")
                return real(src, dst, **kw)

            m.setattr(shutil, "copy2", copy2)
        else:
            real_unlink = Path.unlink

            def unlink(self, *a, **kw):
                if self.name == "002_run.mp4":
                    raise OSError("in use")
                return real_unlink(self, *a, **kw)

            m.setattr(Path, "unlink", unlink)
        _run_director(tmp_path, monkeypatch, out_name="mix")
    assert "002_run" in json.loads((out / "director_stems.json").read_text())
    _run_director(tmp_path, monkeypatch, out_name="mix")  # the failure is gone
    assert not (out / "002_run.mp4").exists() and not (out / "002_run.json").exists()
    assert (out / "002_run.json.bak").read_text() == '{"keys": ["hand-edited"]}'
    assert "002_run" not in json.loads((out / "director_stems.json").read_text())


def test_cmd_director_renumbered_path_taken_over_by_another_segment_is_not_backed_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / "proj" / "pano_paths" / "mix"
    out.mkdir(parents=True)
    seg = lambda start, label: {"source": "x.lrv", "start": start, "end": start + 3, "label": label}  # noqa: E731
    old = {
        "version": D.PATH_VERSION,
        "clip": "001",
        "segment": {"start": 20.0, "end": 23.0},
        "kind": None,
        "vertical": False,
        "params": asdict(D.params_for(None)),
        "keys": ["kept"],
    }
    (out / "002_x.json").write_text(json.dumps(old))  # was segment B, now 003_x
    (out / "director_stems.json").write_text('["002_x"]')
    _run_director(
        tmp_path,
        monkeypatch,
        out_name="mix",
        force=False,
        segments=[seg(1.0, "z"), seg(10.0, "x"), seg(20.0, "x")],
    )
    assert json.loads((out / "003_x.json").read_text())["keys"] == ["kept"]
    assert not list(out.glob("002_*.bak"))
    assert "re-planned" not in caplog.text


def test_cmd_director_preview_without_lrv_renders_from_insv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cutpoint.pano import render as R  # noqa: PLC0415

    srcs: list = []

    def fake_render(clip, path, start, end, out, *, src, **_):
        srcs.append(src)
        out.write_bytes(b"x")
        return {"seconds": 0.0}

    monkeypatch.setattr(R, "render_segment", fake_render)
    clip = SimpleNamespace(clip_id="001", lrv=None, insv=["a_00_.insv", "a_10_.insv"])
    _run_director(tmp_path, monkeypatch, clip=clip, preview=True)
    assert len(srcs) == 1
    assert srcs[0].inputs == ("a_00_.insv", "a_10_.insv")
    assert srcs[0].projection == "dfisheye"


def test_cmd_director_rejects_path_out_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="plain folder name"):
        _run_director(tmp_path, monkeypatch, out_name="../x")


def test_cmd_director_rerun_keeps_bak_of_hand_edited_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    proj, _ = _run_director(tmp_path, monkeypatch, force=True)
    f = proj / "pano_paths" / "director" / "001_laugh.json"
    edited = {"keys": ["hand-edited"]}
    f.write_text(json.dumps(edited))
    caplog.clear()
    _run_director(tmp_path, monkeypatch, force=True)
    bak = f.with_name(f.name + ".bak")
    assert json.loads(bak.read_text()) == edited
    assert "kept as" in caplog.text and bak.name in caplog.text
    assert json.loads(f.read_text()) != edited
