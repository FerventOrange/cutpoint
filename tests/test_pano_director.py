"""Director penalties (PR 3): framing utility, occluders, the wearer in view, selfie
gating, ``DirectorParams.legacy()`` equivalence and ``pano.director`` overrides in
``PanoExporter.path_for``.

No models or footage: the view layer is synthetic and its CLIP scores are stubbed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pano_helpers as helpers
from cutpoint.events import visual as V
from cutpoint.pano import director as D
from cutpoint.pano import geometry as G
from cutpoint.pano import views as VW
from pano_helpers import orientation

FWD_V = [0.0, -1.0, 0.0]  # camera ray far from the wearer cone
N = 12  # view frames, every 1.2 s


def make_views(boxes: dict[int, list[tuple]], source: str) -> VW.PanoViews:
    return helpers.make_views(boxes, source, n=N)


@pytest.fixture
def stub_scores(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Stub the CLIP view scores, voice and forward yaw; tests edit the returned dict."""
    st = {
        "group": np.zeros((N, 9), np.float32),
        "beauty": np.zeros((N, 9), np.float32),
        "occl_zs": np.zeros((N, 8), np.float32),
        "selfie_ok": np.ones(N, np.float32),
        "talk": 0.0,
    }
    monkeypatch.setattr(
        VW,
        "view_scores",
        lambda pv: {"group_zs": st["group"], "beauty": st["beauty"]},
    )
    monkeypatch.setattr(
        VW,
        "occlusion_scores",
        lambda pv: {"occl_zs": st["occl_zs"], "selfie_ok": st["selfie_ok"]},
        raising=False,
    )
    monkeypatch.setattr(VW, "forward_yaw", lambda pv: 0.0)
    monkeypatch.setattr(D, "_voice_at", lambda src, times: np.full(len(times), st["talk"]))
    return st


def path_of(
    pv: VW.PanoViews,
    params: D.DirectorParams | None = None,
    span: tuple[float, float, float] = (1.0, 12.0, 0.0),
    **kw,
) -> dict:
    """Path of ``pv`` for ``span`` = (start, end, margin)."""
    clip = SimpleNamespace(source=pv.meta["source"], duration=N * 1.2, clip_id="001")
    start, end, margin = span
    return D.compute_path(
        clip, start, end, params=params, orient=orientation(), pv=pv, margin=margin, **kw
    )


def mean_abs_yaw(path: dict, centre: float) -> float:
    return float(np.mean([abs((k["yaw"] - centre + 180) % 360 - 180) for k in path["keys"]]))


# --------------------------------------------------------------------------- views
def test_framing_utility_shape() -> None:
    u = VW.framing_utility(
        np.array([0.0, 4.0, 8.0, 12.0, 30.0, 45.0, 52.5, 60.0, 65.0, 70.0, 90.0])
    )
    assert u.tolist() == pytest.approx([0, 0, 0.5, 1, 1, 1, 0.5, 0, -0.25, -0.5, -0.5])
    assert float(VW.framing_utility(20.0)) == 1.0


def test_huge_friend_scores_negative_and_occludes(stub_scores: dict) -> None:
    huge = (0.99, 90.0, -15.0, 80.0, 85.0, FWD_V)  # >70 deg: not the wearer, just too close
    friend = (0.99, 270.0, -8.0, 10.0, 25.0, FWD_V)
    pv = make_views({f: [huge, friend] for f in range(N)}, "huge.lrv")
    ds = VW.direction_scores(pv)
    b90, b270 = 9, 27
    assert ds["people"][3, b90] < 0 < ds["people"][3, b270]
    assert ds["occlude"][3, b90] == 1.0 and ds["occlude"][3, b270] == 0.0
    assert ds["subject"][3, b270].tolist() == pytest.approx([270.0, -8.0, 10.0, 25.0])
    assert np.isnan(ds["subject"][3, b90]).all()
    legacy = VW.direction_scores(pv, people_size="close")  # old rule: >70 deg is the wearer
    assert legacy["people"][3, b90] == pytest.approx(0.0, abs=1e-3)


def test_huge_box_at_yaw_90_is_avoided(stub_scores: dict) -> None:
    huge = (0.99, 90.0, -10.0, 80.0, 85.0, FWD_V)
    stub_scores["beauty"][:, 2] = 1.0  # a vista behind the occluder (view yaw 90)
    pv = make_views({f: [huge] for f in range(N)}, "avoid.lrv")
    new = path_of(pv)
    old = path_of(pv, D.DirectorParams.legacy())
    assert mean_abs_yaw(old, 90.0) < 15.0  # the old director looked into it
    assert mean_abs_yaw(new, 90.0) > 45.0
    assert new["quality"]["too_close"] < old["quality"]["too_close"]


def test_wearer_in_view_is_penalised(stub_scores: dict) -> None:
    p = D.DirectorParams()
    axis = np.asarray(G.WEARER_V)
    times = np.array([1.0, 2.0])
    centres = np.arange(36) * 10.0
    wiv = D.wearer_in_view(orientation(), axis, times, centres, p)
    wy, _ = G.ray_yaw_pitch(orientation().r_vw(1.0).T @ axis)
    near = round(float(wy) % 360 / 10) % 36
    assert wiv[0, near] == 1.0 and wiv[0, (near + 18) % 36] == 0.0
    # equal vistas toward and away from the wearer: the director looks away
    stub_scores["beauty"][:, :8] = 1.0
    pv = make_views({}, "wearer.lrv")
    path = path_of(pv)
    assert mean_abs_yaw(path, float(wy)) > 60.0
    clip = SimpleNamespace(source="wearer.lrv")
    sg = D.score_grid(clip, 1.0, 3.0, p, pv, orient=orientation())
    free = D.score_grid(clip, 1.0, 3.0, D.with_params(p, {"w_wearer": 0}), pv, orient=orientation())
    pen = free["yaw_score"][0] - sg["yaw_score"][0]
    assert pen[near] == pytest.approx(p.w_wearer) and pen[(near + 18) % 36] == 0.0


def test_occluded_selfie_is_gated(stub_scores: dict) -> None:
    stub_scores["talk"] = 1.0  # the wearer talks the whole time: the old director cuts to selfie
    pv = make_views({}, "selfie.lrv")
    base = D.with_params(D.DirectorParams(), {"selfie_base": 0.3, "selfie_mode": "auto"})
    assert path_of(pv, base)["seconds_by_target"]["selfie"] > 5
    stub_scores["selfie_ok"][:] = 0.1  # a glove over the selfie view
    assert path_of(pv, base)["seconds_by_target"]["selfie"] == 0.0
    stub_scores["selfie_ok"][:] = 1.0
    off = D.with_params(base, {"selfie_mode": "off"})
    assert path_of(pv, off)["seconds_by_target"]["selfie"] == 0.0
    assert D.params_for("follow").selfie_mode == "off"
    with pytest.raises(ValueError, match="selfie_mode: one of auto, off"):
        D.with_params(D.DirectorParams(), {"selfie_mode": "sometimes"})


# --------------------------------------------------------------------------- legacy
def legacy_cases(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict]:
    """Synthetic legacy-parameter paths (also run against the pre-PR-3 commit)."""
    out = {}
    group = np.zeros((N, 9), np.float32)
    group[:, 3] = np.linspace(0, 1, N)
    friend = [  # walks right, then someone else at yaw 200 is closer (a cut)
        (0.95, 30.0 + 8 * f if f < 6 else 200.0, -12.0, 10.0, 18.0 + 4 * f, FWD_V) for f in range(N)
    ]
    tall = (0.99, 120.0, -20.0, 80.0, 80.0, FWD_V)
    walk = {f: [friend[f]] for f in range(N)}
    cases = {
        "friend": (walk, 0.0, None, {}),
        "talk": ({f: [friend[f]] for f in range(0, N, 3)}, 0.9, "laughter", {}),
        "tall": ({f: [friend[f], tall] for f in range(N)}, 0.6, "wipeout", {}),
        # PR 4: a segment inside the planned range (cuts near its edges) and 9:16
        "edge": (walk, 0.0, None, {"span": (4.5, 9.0, 4.0)}),
        "vertical": (walk, 0.6, "people", {"vertical": True}),
    }
    for name, (boxes, talk, kind, kw) in cases.items():
        monkeypatch.setattr(
            VW, "view_scores", lambda pv: {"group_zs": group, "beauty": group * 0.5}
        )
        monkeypatch.setattr(
            VW,
            "occlusion_scores",
            lambda pv: {"occl_zs": np.zeros((N, 8)), "selfie_ok": np.zeros(N)},
            raising=False,
        )
        monkeypatch.setattr(VW, "forward_yaw", lambda pv: 10.0)
        monkeypatch.setattr(D, "_voice_at", lambda src, times, v=talk: v * (times > 6.0))
        pv = make_views(boxes, f"legacy_{name}.lrv")
        params = D.DirectorParams.legacy()
        p = path_of(pv, params, kind=kind, **kw)
        out[name] = {k: p[k] for k in ("keys", "cuts", "seconds_by_target")}
    return out


# sha1 of json.dumps(case, sort_keys=True) on commit 6fe1a40 (before the penalties);
# edge / vertical on 99cc953 (legacy params, before subject framing)
LEGACY_SHA1 = {
    "friend": "43095cbf973d3e4129179ebbb03fe20a12b91ff6",
    "talk": "a725db6dca1bdbc8f781a9ce7718290a0454627c",
    "tall": "b133849e9b04ca758fb40df08f053f6ba10c3e71",
    "edge": "67d555af427dd1edb6799109429567a8b9a51b2e",
    "vertical": "ac736fae602f2717cdb22216d3c59e7b4a79f094",
}


def test_legacy_params_reproduce_the_old_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, case in legacy_cases(monkeypatch).items():
        digest = hashlib.sha1(json.dumps(case, sort_keys=True).encode()).hexdigest()
        assert digest == LEGACY_SHA1[name], name


# --------------------------------------------------------------------------- render
def test_path_for_applies_pano_director_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cutpoint import paths  # noqa: PLC0415
    from cutpoint.pano import render as R  # noqa: PLC0415

    paths.config_file().parent.mkdir(parents=True, exist_ok=True)
    paths.config_file().write_text(
        json.dumps({"pano": {"director": {"max_speed": 30, "w_wearer": 0.2}}}), encoding="utf-8"
    )
    monkeypatch.setattr(R, "load", lambda project: {})
    seen = {}

    def fake(clip, start, end, **kw):
        seen.update(kw)
        return {"keys": [], "planned": {"start": start, "end": end}}

    monkeypatch.setattr(D, "compute_path", fake)
    ex = R.PanoExporter(tmp_path, tmp_path / "clips")
    ex.path_for(SimpleNamespace(clip_id="001"), "001_a.mp4", 1.0, 5.0, "laughter")
    assert seen["overrides"] == {"max_speed": 30, "w_wearer": 0.2}
    assert seen["kind"] == "laughter"
    assert (tmp_path / "pano_paths" / "clips" / "001_a.json").is_file()


# --------------------------------------------------------------------------- PR 4: subject framing
def test_track_subject_links_and_interpolates(stub_scores: dict) -> None:
    walk = (0.95, 30.0, -10.0, 10.0, 20.0, FWD_V)
    other = (0.95, 200.0, -10.0, 10.0, 20.0, FWD_V)
    boxes = {f: [(*walk[:1], 30.0 + 6 * f, *walk[2:]), other] for f in range(N)}
    pv = make_views(boxes, "track.lrv")
    ds = VW.direction_scores(pv)
    times = np.arange(0.0, 6.0, 0.5)
    states = np.full(len(times), 3)  # the shot stays on bin 30 deg; the friend walks on
    tr = D._track_subject(pv, ds["subject"], times, states, [(0, len(times))])
    # 6 deg per 1.2 s frame, linear between frames: yaw = 30 + 5 t
    assert tr[:, 0] == pytest.approx(30.0 + 5.0 * times, abs=1e-4)
    assert tr[:, 3] == pytest.approx(20.0)
    assert np.isnan(D._track_subject(pv, None, times, states, [(0, len(times))])).all()
    selfie = np.full(len(times), 36)
    assert np.isnan(D._track_subject(pv, ds["subject"], times, selfie, [(0, len(times))])).all()


def test_fit_fov_inverts_flat_fov() -> None:
    p = D.DirectorParams()
    size = (1920, 1080)
    h = 40.0  # tall: the height decides
    f = float(D.fit_fov(5.0, h, p, size))
    vf = G.flat_fov(f, *size)
    frac = np.tan(np.radians(h) / 2) / np.tan(np.radians(vf) / 2)
    assert frac == pytest.approx(p.subject_max_vfrac)
    w = 50.0  # wide: the width decides
    f = float(D.fit_fov(w, 5.0, p, size))
    frac = np.tan(np.radians(w) / 2) / np.tan(np.radians(f) / 2)
    assert frac == pytest.approx(p.subject_max_wfrac)
    assert D.fit_fov(np.array([5.0, 10.0]), np.array([10.0, 30.0]), p, size)[1] > 60


def safe_ctx(n: int, subj_yaw: float, fov: float) -> dict:
    nb = 36
    return {
        "times": np.arange(n) * 0.5,
        "yaw": np.zeros(n),
        "pitch": np.full(n, -10.0),
        "fov": np.full(n, fov),
        "target": ["friend"] * n,
        "subj": np.tile([subj_yaw, -10.0, 10.0, 25.0], (n, 1)),
        "near": [np.zeros((0, 4))] * n,
        "yaw_score": np.zeros((n, nb)),
        "people": np.zeros((n, nb)),
        "centres": np.arange(nb) * 10.0,
        "forward_yaw": 180.0,
        "size": (1920, 1080),
        "fov_hi": 110.0,
        "fov_lo": 72.0,
        "fov_wide": 100.0,
        "raw_yaw": np.zeros(n),  # no faster re-limit possible
    }


def test_safe_area_pass_widens_then_cuts() -> None:
    p = D.DirectorParams()
    n = 30
    ctx = safe_ctx(n, 36.0, 80.0)  # at the edge of an 80 deg view, inside a 100 deg one
    shots = D._safe_area_pass(ctx, [(0, n)], (0, n), p)
    assert shots == [(0, n)]
    assert ctx["fov"].min() > 80.0 and ctx["fov"].max() <= 110.0
    ctx = safe_ctx(n, 62.0, 80.0)  # beyond half of fov_max: a yaw shift reframes it
    assert D._safe_area_pass(ctx, [(0, n)], (0, n), p) == [(0, n)]
    assert set(ctx["target"]) == {"friend"} and ctx["yaw"].max() > 5.0
    ctx = safe_ctx(n, 90.0, 80.0)  # out of reach: only a cut helps
    ctx["yaw_score"][:, 18] = 0.5  # the forward bin scores best
    shots = D._safe_area_pass(ctx, [(0, n)], (0, n), p)
    assert set(ctx["target"]) == {"forward"}
    assert ctx["yaw"] == pytest.approx(180.0) and np.isnan(ctx["subj"]).all()
    assert shots == [(0, n)]
    ctx = safe_ctx(n, 90.0, 80.0)
    ctx["subj"][:10, 0] = 0.0  # well framed for the first 5 s: cut later
    ctx["yaw_score"][:, 27] = 1.0  # a vista at 270
    shots = D._safe_area_pass(ctx, [(0, n)], (0, n), p)
    assert shots[0] == (0, 10) and len(shots) == 2
    assert ctx["yaw"][10:] == pytest.approx(-90.0) and ctx["target"][10] == "view"
    ctx = safe_ctx(60, 0.0, 80.0)  # two bad stretches in one shot: two cuts
    ctx["subj"][10:20, 0] = ctx["subj"][35:45, 0] = 90.0
    ctx["yaw_score"][:, 27] = 1.0
    shots = D._safe_area_pass(ctx, [(0, 60)], (0, 60), p)
    assert shots[0][0] == 0 and shots[-1][1] == 60 and len(shots) == 5
    assert all(x[1] == y[0] for x, y in pairwise(shots))
    assert ctx["target"][12] == ctx["target"][40] == "view" and ctx["target"][27] == "friend"
    ctx = safe_ctx(60, 0.0, 80.0)  # less than min_hold_s apart: one cut-away
    ctx["subj"][10:14, 0] = ctx["subj"][16:20, 0] = 90.0
    ctx["yaw_score"][:, 27] = 1.0
    shots = D._safe_area_pass(ctx, [(0, 60)], (0, 60), p)
    assert len(shots) == 3 and shots[0] == (0, 10) and shots[-1][1] == 60


def test_safe_area_pass_flags_an_edge_intruder() -> None:
    p = D.DirectorParams()
    n = 10
    ctx = safe_ctx(n, 0.0, 86.0)
    big = np.array([[40.0, -15.0, 30.0, 60.0]])  # a big friend cut by the right edge
    ctx["near"] = [big] * n
    D._safe_area_pass(ctx, [(0, n)], (0, n), p)
    assert ctx["fov"].min() > 86.0 or set(ctx["target"]) != {"friend"}


def test_close_person_mostly_outside_the_view_is_an_edge_intruder() -> None:
    p = D.DirectorParams()
    size = (1920, 1080)
    close = np.array([70.0, -10.0, 80.0, 60.0])  # inches away: only its left part is in view
    assert D.close_edge_share(0.0, -10.0, 86.0, size, close) > D.CLOSE_MIN_SHARE
    assert D.close_edge_share(0.0, -10.0, 86.0, size, np.array([0.0, -10.0, 60.0, 60.0])) == 0
    assert D.close_edge_share(180.0, -10.0, 86.0, size, close) == 0.0
    r = G.project_box_visible(0.0, -10.0, 86.0, *size, *close)
    assert r is None or r[1] < 0.25  # the projected-box rule alone does not see it
    ctx = safe_ctx(10, 0.0, 86.0)
    ctx["near"] = [close[None, :]] * 10
    assert D._framing_checks(ctx, (0, 10), p).bad(0)
    ctx["target"] = ["selfie"] * 10  # the wearer framed on purpose: no intruders
    assert not D._framing_checks(ctx, (0, 10), p).bad(0)


def test_wearer_coming_in_from_the_bottom_is_no_intruder() -> None:
    p = D.DirectorParams()
    n = 10
    ctx = safe_ctx(n, 0.0, 96.0)
    arms = np.array([[0.0, -80.0, 100.0, 90.0, 1.0]])  # the wearer's arms / skis below
    ctx["near"] = [arms] * n
    assert not D._framing_checks(ctx, (0, n), p).bad(0)
    assert D._safe_area_pass(ctx, [(0, n)], (0, n), p) == [(0, n)]
    assert set(ctx["target"]) == {"friend"} and ctx["fov"] == pytest.approx(96.0)
    ctx["near"] = [np.array([[20.0, -80.0, 100.0, 90.0, 0.0]])] * n  # anyone else: intruder
    assert D._framing_checks(ctx, (0, n), p).bad(0)
    ctx["near"] = [np.array([[0.0, -10.0, 60.0, 60.0, 1.0]])] * n  # the wearer mid-frame
    assert D._framing_checks(ctx, (0, n), p).bad(0)


def test_cut_away_only_where_it_helps_and_lasts_within_the_segment() -> None:
    p = D.DirectorParams()
    n = 30
    ctx = safe_ctx(n, 0.0, 80.0)
    ring = np.array([[y, -10.0, 60.0, 60.0] for y in range(0, 360, 45)])  # close all around
    ctx["near"] = [ring] * n
    assert D._safe_area_pass(ctx, [(0, n)], (0, n), p) == [(0, n)]  # no view is better
    assert set(ctx["target"]) == {"friend"}
    ctx = safe_ctx(n, 0.0, 80.0)
    ctx["subj"][15:, 0] = 90.0  # lost from step 15; the segment ends at step 19 (9.5 s)
    ctx["yaw_score"][:, 27] = 1.0
    shots = D._safe_area_pass(ctx, [(0, n)], (0, 20), p)
    c0 = shots[1][0]
    assert c0 <= 13 and ctx["target"][c0] == "view"  # >= min_hold_s of it in the segment


def test_subject_paths_have_subj_keys_and_evaluate_like_v1(stub_scores: dict) -> None:
    friend = (0.95, 40.0, -10.0, 10.0, 30.0, FWD_V)
    pv = make_views({f: [friend] for f in range(N)}, "v2.lrv")
    new = path_of(pv)
    assert new["version"] == 4 == D.PATH_VERSION
    subj = [k for k in new["keys"] if "subj" in k]
    assert subj and subj[0]["subj"] == pytest.approx([40.0, -10.0, 10.0, 30.0])
    assert all(k["fov"] != 96.0 for k in subj if k["target"] == "friend")  # fitted
    assert not any("subj" in k for k in path_of(pv, D.DirectorParams.legacy())["keys"])
    v1 = {
        **new,
        "version": 1,
        "keys": [{k: v for k, v in x.items() if k != "subj"} for x in new["keys"]],
    }
    t = np.linspace(1.0, 12.0, 50)
    for a, b in zip(D.evaluate(new, t), D.evaluate(v1, t), strict=True):
        assert a == pytest.approx(b)


def test_shots_are_measured_within_the_segment() -> None:
    p = D.DirectorParams()
    nb = 36
    centres = np.arange(nb) * 10.0
    times = np.arange(0.0, 20.0, 0.5)
    states = np.r_[np.full(30, 0), np.full(10, 18)]  # cut at 15 s, 5 s each side
    assert D._shots(states.copy(), centres, nb, p) == [(0, 30), (30, 40)]
    st = states.copy()  # the segment ends 0.2 s after the cut: a 0.2 s shot
    assert D._shots(st, centres, nb, p, times, (2.0, 15.2)) == [(0, 40)]
    assert (st == 0).all()
    st = states.copy()  # starts 0.3 s before the cut
    assert D._shots(st, centres, nb, p, times, (14.7, 19.0)) == [(0, 40)]
    assert (st == 18).all()


def test_short_shot_not_folded_into_a_gated_selfie() -> None:
    p = D.with_params(D.DirectorParams(), {"selfie_mode": "auto"})
    nb = 36
    centres = np.arange(nb) * 10.0
    states = np.r_[np.full(10, nb), np.full(1, 18), np.full(10, 0)]  # one-frame block
    selfie = np.zeros(len(states))
    selfie[10] = D.NO_SELFIE  # the selfie is gated (a glove) on that frame
    st = states.copy()
    shots = D._shots(st, centres, nb, p, selfie=selfie)
    assert st[10] == 0 and shots == [(0, 10), (10, 21)]
    st = states.copy()  # ungated: the old rule (previous shot) applies
    assert D._shots(st, centres, nb, p, selfie=np.zeros(len(states)))[0] == (0, 11)
    assert st[10] == nb


def test_path_for_warns_on_a_stale_reused_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from cutpoint.pano import render as R  # noqa: PLC0415

    monkeypatch.setattr(R, "load", lambda project: {})
    monkeypatch.setattr(D, "compute_path", lambda *a, **kw: pytest.fail("re-directed"))
    ex = R.PanoExporter(tmp_path, tmp_path / "clips")
    old = tmp_path / "old.json"
    params = asdict(D.params_for("laughter"))
    path = {
        "version": D.PATH_VERSION,
        "planned": {"start": 0.0, "end": 9.0},
        "keys": [],
        "params": params,
    }
    old.write_text(json.dumps(path), encoding="utf-8")
    clip = SimpleNamespace(clip_id="001")
    with caplog.at_level("WARNING"):
        got, _ = ex.path_for(clip, "001_a.mp4", 1.0, 5.0, "laughter", reuse=old)
    assert got["keys"] == [] and not caplog.records  # same settings: silent
    path.update(params={**params, "max_speed": 30.0})
    old.write_text(json.dumps(path), encoding="utf-8")
    with caplog.at_level("WARNING"):
        ex.path_for(clip, "001_a.mp4", 1.0, 5.0, "laughter", reuse=old)
    assert "max_speed" in caplog.text and "old.json" in caplog.text
    path.update(version=3)  # tighter framing: re-directed
    old.write_text(json.dumps(path), encoding="utf-8")
    monkeypatch.setattr(D, "compute_path", lambda *a, **kw: {"keys": ["new"]})
    got, _ = ex.path_for(clip, "001_a.mp4", 1.0, 5.0, "laughter", reuse=old)
    assert got["keys"] == ["new"] and "version 3 director path" in caplog.text


def test_path_for_keeps_bak_of_replanned_reused_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cutpoint.pano import render as R  # noqa: PLC0415

    monkeypatch.setattr(R, "load", lambda project: {})
    monkeypatch.setattr(D, "compute_path", lambda *a, **kw: {"keys": ["new"]})
    ex = R.PanoExporter(tmp_path, tmp_path / "clips")
    old = ex.paths_dir / "001_a.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"version": 1, "keys": ["hand-edited"]}), encoding="utf-8")
    got, out = ex.path_for(SimpleNamespace(clip_id="001"), "001_a.mp4", 1.0, 5.0, None, reuse=old)
    assert got["keys"] == ["new"] and out == old
    assert json.loads(old.with_name("001_a.json.bak").read_text())["keys"] == ["hand-edited"]
    ex.path_for(SimpleNamespace(clip_id="001"), "001_a.mp4", 1.0, 5.0, None, reuse=old)  # again
    assert json.loads(old.with_name("001_a.json.bak").read_text())["keys"] == ["hand-edited"]
    assert old.with_name("001_a.json.1.bak").is_file()


def test_path_for_backup_failing_midway_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno  # noqa: PLC0415
    import shutil  # noqa: PLC0415

    from cutpoint.pano import render as R  # noqa: PLC0415

    def disk_fills(src: Path, dst: Path) -> None:
        Path(dst).write_bytes(Path(src).read_bytes()[:5])
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(R, "load", lambda project: {})
    monkeypatch.setattr(D, "compute_path", lambda *a, **kw: {"keys": ["new"]})
    monkeypatch.setattr(shutil, "copy2", disk_fills)
    ex = R.PanoExporter(tmp_path, tmp_path / "clips")
    old = ex.paths_dir / "001_a.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"version": 1, "keys": ["hand-edited"]}), encoding="utf-8")
    before = old.read_bytes()
    with pytest.raises(OSError, match="No space"):
        ex.path_for(SimpleNamespace(clip_id="001"), "001_a.mp4", 1.0, 5.0, None, reuse=old)
    assert old.read_bytes() == before
    assert [p.name for p in old.parent.iterdir()] == ["001_a.json"]  # no .bak, no .tmp


def test_views_cache_follows_clip_identity(tmp_path, monkeypatch):
    """A cache written for one CLIP arch/weights is a miss for another; the current one loads."""
    src = tmp_path / "g.lrv"
    src.write_bytes(b"x")
    pv = make_views({}, str(src))
    VW.save_views(src, pv, VW.ViewParams())
    assert VW.load_views(src) is not None
    for attr in ("CLIP_ARCH", "CLIP_PRETRAINED"):
        with monkeypatch.context() as changed:
            changed.setattr(V, attr, f"{getattr(V, attr)}-other")
            assert VW.load_views(src) is None


@pytest.mark.parametrize("existed", [False, True])
def test_failed_director_run_removes_only_the_folder_it_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existed: bool
) -> None:
    import argparse  # noqa: PLC0415

    from cutpoint.pano import cli  # noqa: PLC0415
    from cutpoint.pano import project as P  # noqa: PLC0415

    def boom(*a, **k):
        raise RuntimeError("director failed")

    monkeypatch.setattr(cli, "_project", lambda arg: tmp_path)
    monkeypatch.setattr(P, "load", lambda project: {"x.lrv": SimpleNamespace(clip_id="001")})
    monkeypatch.setattr(cli, "_direct", boom)
    out = tmp_path / "pano_paths" / "director"
    if existed:
        out.mkdir(parents=True)
    a = argparse.Namespace(project="p", vertical=False, param=[], out_name=None)
    with pytest.raises(RuntimeError, match="director failed"):
        cli.cmd_director(a)
    assert out.is_dir() is existed
    assert (tmp_path / "pano_paths").is_dir() is existed


@pytest.mark.parametrize("existed", [False, True])
def test_failed_path_for_removes_only_the_folder_it_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existed: bool
) -> None:
    from cutpoint.pano import render as R  # noqa: PLC0415

    def boom(*a, **k):
        raise RuntimeError("director failed")

    monkeypatch.setattr(R, "load", lambda project: {})
    monkeypatch.setattr(D, "compute_path", boom)
    ex = R.PanoExporter(tmp_path, tmp_path / "clips")
    if existed:
        ex.paths_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="director failed"):
        ex.path_for(SimpleNamespace(clip_id="001"), "001_a.mp4", 1.0, 5.0, None)
    assert ex.paths_dir.is_dir() is existed
    assert (tmp_path / "pano_paths").is_dir() is existed


# ------------------------------------------------------------ evaluate: cubic interpolation
def keys_path(yaw, pitch=None, fov=None, dt=0.5) -> dict:
    n = len(yaw)
    pitch = pitch if pitch is not None else [0.0] * n
    fov = fov if fov is not None else [90.0] * n
    return {
        "keys": [
            {"t": i * dt, "yaw": y, "pitch": p, "fov": f}
            for i, (y, p, f) in enumerate(zip(yaw, pitch, fov, strict=True))
        ]
    }


def test_evaluate_velocity_is_continuous_at_a_middle_key() -> None:
    path = keys_path([0.0, 5.0, 15.0, 30.0, 50.0])  # speeds rise: the key at 1 s is cubic
    e = 1e-4
    t = np.array([1.0 - 2 * e, 1.0 - e, 1.0 + e, 1.0 + 2 * e])
    yaw = D.evaluate(path, t)[0]
    left = (yaw[1] - yaw[0]) / e
    right = (yaw[3] - yaw[2]) / e
    assert left == pytest.approx(right, rel=1e-2)
    assert left > 0  # moving, not a degenerate flat tangent


def test_evaluate_never_overshoots_the_neighbouring_keys() -> None:
    # non-monotone pitch and fov: a spline would overshoot at the turns
    path = keys_path(
        [0.0, 5.0, 10.0, 15.0, 20.0],
        pitch=[0.0, 20.0, 18.0, 30.0, 0.0],
        fov=[90.0, 60.0, 62.0, 100.0, 95.0],
    )
    t = np.linspace(0.0, 2.0, 401)
    _, pitch, fov = D.evaluate(path, t)
    kt = np.array([k["t"] for k in path["keys"]])
    for out, name in ((pitch, "pitch"), (fov, "fov")):
        ky = np.array([k[name] for k in path["keys"]])
        i = np.clip(np.searchsorted(kt, t, side="right") - 1, 0, len(kt) - 2)
        lo, hi = np.minimum(ky[i], ky[i + 1]), np.maximum(ky[i], ky[i + 1])
        assert np.all(out >= lo) and np.all(out <= hi), name


def test_evaluate_yaw_across_the_antimeridian_is_smooth() -> None:
    # +5, +10, +20, +30 deg: always eastward, rising speeds so the intervals stay cubic
    path = keys_path([165.0, 170.0, -180.0, -160.0, -130.0])
    t = np.linspace(0.0, 2.0, 401)
    yaw = D.evaluate(path, t)[0]
    assert yaw[0] == pytest.approx(165.0) and yaw[-1] == pytest.approx(230.0)
    assert np.all(np.diff(yaw) >= 0)  # monotone, no 360 deg jump
    assert np.abs(np.diff(yaw)).max() < 1.0
    v = np.diff(yaw) / np.diff(t)
    # the last interval (from 1.5 s) is linear, its velocity steps at its key: skip it
    assert np.abs(np.diff(v[:295])).max() < 1.0  # velocity has no steps either


def test_evaluate_two_keys_are_linear() -> None:
    path = keys_path([0.0, 20.0], pitch=[0.0, 10.0], fov=[90.0, 70.0])
    t = np.linspace(-0.2, 0.7, 19)  # held outside [0, .5]
    yaw, pitch, fov = D.evaluate(path, t)
    s = np.clip(t / 0.5, 0.0, 1.0)
    assert yaw == pytest.approx(20.0 * s)
    assert pitch == pytest.approx(10.0 * s)
    assert fov == pytest.approx(90.0 - 20.0 * s)


@pytest.mark.parametrize(
    "kx", [[0.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 2.0]], ids=["start", "middle"]
)
def test_pchip_two_keys_at_one_time_step_without_nan(kx: list[float]) -> None:
    # hand-edited path: a step at the shared time, no NaN, no numpy warning
    kx_a, ky = np.array(kx), np.array([0.0, 10.0, 4.0, 8.0])
    x = np.array([-0.5, 0.0, 1.0, 2.0, 2.5])  # outside, at the keys (incl. the step)
    with np.errstate(all="raise"):
        out = D._pchip(x, kx_a, ky)
    assert np.all(np.isfinite(out))
    assert out == pytest.approx(np.interp(x, kx_a, ky))


def test_evaluate_cubic_never_crosses_a_cut() -> None:
    # two shots of 3 keys each; the cut key (t=2) starts the second shot
    rows = [(0, 0), (1, 10), (2, 30), (2, 100), (3, 105), (4, 125)]
    keys = [{"t": float(t), "yaw": float(y), "pitch": 0.0, "fov": 90.0} for t, y in rows]
    keys[3]["cut"] = True
    path = {"keys": keys}
    t = np.array([1.99, 2.0, 5.0])
    yaw = D.evaluate(path, t)[0]
    assert 10.0 < yaw[0] < 30.0  # still in shot 1, no pull towards 100
    assert yaw[1] == pytest.approx(100.0)
    assert yaw[2] == pytest.approx(125.0)  # held after the last key
    grid = np.linspace(0.0, 4.0, 161)
    got = D.evaluate(path, grid)[0]
    first = grid < 2.0
    for sel, kx, ky in (
        (first, np.array([0.0, 1.0, 2.0]), np.array([0.0, 10.0, 30.0])),
        (~first, np.array([2.0, 3.0, 4.0]), np.array([100.0, 105.0, 125.0])),
    ):
        flat = np.zeros(3)
        mask = D._linear_intervals(kx, ky, flat)
        want = D._pchip(grid[sel], kx, np.stack([ky, flat], 1), mask)[:, 0]
        assert got[sel] == pytest.approx(want)


# ------------------------------------------------ evaluate: the cubic never pans faster than linear
def _linear_eval(path: dict, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference: exact linear interpolation of the keys of a path without cuts."""
    keys = path["keys"]
    kt = [k["t"] for k in keys]
    yaw = np.degrees(np.unwrap(np.radians([k["yaw"] for k in keys])))
    return (
        np.interp(times, kt, yaw),
        np.interp(times, kt, [k["pitch"] for k in keys]),
        np.interp(times, kt, [k["fov"] for k in keys]),
    )


_EVALUATE = D.evaluate  # D.evaluate is patched while measuring


def _pan_metrics(path: dict, evaluate, monkeypatch: pytest.MonkeyPatch) -> dict:
    from cutpoint.pano import qa as Q  # noqa: PLC0415

    monkeypatch.setattr(D, "evaluate", evaluate)  # pan_metrics looks it up at call time
    return Q.pan_metrics(path, 0.0, path["keys"][-1]["t"])


def _fails_pan_gate(m: dict) -> bool:
    from cutpoint.pano import qa as Q  # noqa: PLC0415

    th = Q.QAThresholds()
    return m["speed_p95"] > th.max_pan_p95 or m["speed_max"] > th.max_pan_max


def test_evaluate_stop_go_keys_peak_at_the_linear_speed() -> None:
    path = keys_path([0.0, 0.0, 30.0, 30.0])  # the moving interval is 60 deg/s linear
    t = np.linspace(0.0, 1.5, 301)
    yaw = D.evaluate(path, t)[0]
    assert np.abs(np.diff(yaw) / np.diff(t)).max() == pytest.approx(60.0)  # a cubic hits 90


def test_evaluate_near_limit_path_does_not_pan_faster_than_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 8 keys at 0.5 s (random, seed 4, yaw only): linear peaks at 38.2 deg/s and passes
    # the pan gates; the plain cubic reached 56.8 and failed max_pan_max
    yaw = [-117.2, -99.4, -99.0, -79.9, -96.7, -92.4, -97.4, -85.3]
    path = keys_path(yaw)
    lin = _pan_metrics(path, _linear_eval, monkeypatch)
    cubic = _pan_metrics(path, _EVALUATE, monkeypatch)
    assert lin["speed_max"] == pytest.approx(38.2, abs=0.2)
    assert cubic["speed_max"] <= lin["speed_max"] + 0.1
    assert cubic["speed_p95"] <= lin["speed_p95"] + 0.1
    assert not _fails_pan_gate(cubic) or _fails_pan_gate(lin)  # no gate fails only for cubic


def _random_path(rng: np.random.Generator) -> dict:
    n = int(rng.integers(4, 13))
    h = rng.uniform(0.15, 1.2, n - 1)
    vmax = float(rng.choice([35.0, 45.0, 58.0]))
    pattern = rng.choice(["random", "stopgo", "reversal", "nearmax"])
    if pattern == "random":
        v = rng.uniform(-vmax, vmax, n - 1)
    elif pattern == "stopgo":
        v = rng.uniform(0.3, 1.0, n - 1) * vmax * rng.choice([-1, 1])
        v[rng.random(n - 1) < 0.5] = 0.0
    elif pattern == "reversal":
        v = rng.uniform(0.5, 1.0, n - 1) * vmax * np.where(np.arange(n - 1) % 2, 1, -1)
    else:
        v = rng.uniform(0.85, 1.0, n - 1) * vmax
    ang = rng.uniform(0, 0.4, n - 1) * rng.choice([-1, 1], n - 1)  # yaw / pitch split
    kt = np.concatenate([[0.0], np.cumsum(h)])
    pitch = np.clip(
        rng.uniform(-30, 5) + np.concatenate([[0], np.cumsum(v * np.sin(ang) * h)]), -60, 40
    )
    yaw = rng.uniform(-180, 180) + np.concatenate([[0], np.cumsum(v * np.cos(ang) * h)])
    yaw = ((yaw + 180) % 360) - 180  # stored wrapped, as the director's keys are
    keys = [
        {
            "t": round(float(t), 3),
            "yaw": round(float(y), 2),
            "pitch": round(float(p), 2),
            "fov": 90.0,
        }
        for t, y, p in zip(kt, yaw, pitch, strict=True)
    ]
    return {"keys": keys}


def test_evaluate_seeded_fuzz_never_pans_faster_or_fails_a_gate_linear_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = []
    for seed in range(100):
        path = _random_path(np.random.default_rng(seed))
        lin = _pan_metrics(path, _linear_eval, monkeypatch)
        cubic = _pan_metrics(path, _EVALUATE, monkeypatch)
        if cubic["speed_max"] > lin["speed_max"] + 0.1:
            bad.append((seed, "speed_max", lin["speed_max"], cubic["speed_max"]))
        if not _fails_pan_gate(lin) and _fails_pan_gate(cubic):
            bad.append((seed, "gate", lin["speed_max"], cubic["speed_max"]))
    assert not bad
