"""Virtual-camera director: a smooth yaw / pitch / FOV path for one segment.

The path search follows Pano2Vid / AutoCam (research doc section 3.3): score
every display-yaw bin at every time step, then pick the best path by dynamic
programming with a penalty on movement and a fixed price for a hard cut.

Per 0.5 s step and 10-deg yaw bin (from :func:`cutpoint.pano.views.direction_scores`):

    people   other people (wearer excluded) by framing utility  x w_people
             (views.framing_utility: best 12-45 deg tall, negative inches away)
    group    zero-shot friends / group prompt of the nearest view x w_group
    beauty   beauty-shot score (vista prompts x aesthetic x steadiness) x w_beauty
    forward  bias toward the travel direction (yaw 0)            x w_forward
    occlude  huge person boxes / occluder prompts on that yaw     x -w_occlude
    wearer   the wearer's own body inside the view               x -w_wearer

plus one extra state, **selfie**: the downward view at the wearer (seam-free:
it sits inside the ``_10_`` lens), scored from the wearer talking / laughing
(voice layer) and the selfie view's group prompt, and only allowed where the
selfie view shows a usable selfie (``selfie_ok``, not a glove over the lens).

Transitions: staying is free; moving costs ``move_cost`` per ``max_speed * dt``
degrees; anything faster than ``max_speed`` (35 deg/s by default), and every
change to or from the selfie state, is a **hard cut** costing ``cut_cost``.
Cuts closer than ``min_hold_s`` are merged away. Within each shot the yaw is
unwrapped and smoothed (Gaussian ``smooth_s``), then speed- and
acceleration-limited; pitch follows the framed person (clamped) and is
smoothed too; FOV depends on the target. With ``framing="subject"`` (default)
friend shots follow the person tracked across view frames (dead zone,
headroom) with a FOV fitted to its size, and a safe-area pass over the whole
planned span widens, shifts, speeds up or cuts away where the subject or a near
person would still be cut by the frame edge (``edge_rule``: each near person is
framed whole or kept out, see :func:`_include_exclude`); keys then carry ``subj``
(yaw, pitch, width, height; path version 2).

The result is a JSON-able dict saved per clip (``<project>/pano_paths/*.json``)
that the renderer reads, with a ``quality`` block: the geometric direction QA
of the segment (:func:`cutpoint.pano.qa_paths.path_quality`). It is plain JSON on purpose: edit ``keys`` (time,
yaw, pitch, fov, ``cut``) and re-run ``cutpoint extend`` on the clips folder to use
the edited path (``cutpoint 360 director --preview`` renders quick checks). Angles are in the display frame (horizon-levelled, yaw 0 =
smoothed camera forward, + = right, pitch + = up).
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, fields, replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from cutpoint.pano import geometry as G

logger = logging.getLogger(__name__)

PATH_VERSION = 5  # 2: keys may carry "subj" (the framed person); 3: close people at the
# frame edge are intruders; 4: wider framing, the wearer allowed in frame; 5: near people
# framed whole or kept out over the whole planned span ("fixed"; same keys; evaluate
# reads all versions)
MIN_REUSE_VERSION = 4  # older paths (tighter framing) are re-directed, not reused
# a version 4 path fixed its framing within its segment only: reused for that span only
PATH_MARGIN_S = 10.0  # a path is planned this much beyond its segment on both sides


@dataclass(frozen=True)
class DirectorParams:
    """Path-search and smoothing settings."""

    dt: float = 0.5
    n_bins: int = 36
    w_people: float = 1.0
    w_group: float = 0.35
    w_beauty: float = 0.3
    w_forward: float = 0.25
    forward_width_deg: float = 50.0
    w_selfie_talk: float = 0.55  # wearer talking / laughing -> look at the wearer
    w_selfie_group: float = 0.25
    selfie_base: float = 0.0
    # "auto": the selfie state, gated by selfie_ok_min; "off" (tuned default, trip-360:
    # a pole / stick selfie is the wearer's face filling the frame, the QA's BAD 1)
    selfie_mode: str = "off"
    selfie_ok_min: float = 0.55  # selfie view must look like a selfie (views.occlusion_scores)
    w_occlude: float = 1.5  # huge boxes / occluder prompts on the bin (views.direction_scores)
    w_wearer: float = 1.0  # the wearer's own body inside the view
    wearer_clear_deg: float = 5.0  # ... within half the view diagonal plus this
    # 9:16: the wearer cone's half angle (the 16:9 cone of fov_wide 100 + 20 deg clear)
    vertical_wearer_cone_deg: float = 73.821
    people_size: str = "framing"  # people bump height: "framing" utility or legacy "close"
    max_speed: float = 35.0  # deg/s inside a shot (PR 4 tuning; the safe-area pass may reach 1.3x)
    max_accel: float = 60.0  # deg/s^2
    move_cost: float = 0.12  # per max_speed*dt degrees of movement
    cut_cost: float = 1.6
    min_hold_s: float = 3.0
    smooth_s: float = 1.0
    pitch_default: float = -8.0
    pitch_min: float = -32.0
    pitch_max: float = 5.0
    fov_wide: float = 110.0
    fov_friend: float = 96.0
    fov_selfie: float = 110.0
    selfie_pitch_up: float = 12.0  # aim above the wearer so the horizon shows
    vertical_fov_wide: float = 62.0
    vertical_fov_friend: float = 56.0
    vertical_fov_selfie: float = 75.0
    vertical_occlude_fov: float = 86.0  # 9:16: view FOV of the occlusion scores
    # "subject": friend shots follow the tracked person with a fitted FOV and a
    # safe-area pass (PR 4); "bin": the legacy bin-centre path (fixed friend FOV)
    framing: str = "subject"
    subject_max_vfrac: float = 0.62  # fitted FOV: subject height <= this share of the frame
    subject_max_wfrac: float = 0.5  # ... and width
    safe_margin: float = 0.05  # subject kept this far from the left / right edges
    fov_min: float = 86.0  # fitted friend FOV range (16:9)
    fov_max: float = 120.0
    vertical_fov_min: float = 72.0 * 56 / 86  # ... 9:16 (the old 16:9 range x 56 / 86)
    vertical_fov_max: float = 110.0 * 56 / 86
    fov_rate: float = 12.0  # deg/s
    dead_zone: float = 0.25  # share of half the frame width the subject may drift unfollowed
    headroom: float = 0.10  # aim this share of the subject's height above its centre
    # "include_exclude": a near person is framed whole or kept out (widen or shift, by
    # cost), checked as the geometric QA counts edge cuts, over the whole planned span;
    # "legacy": the version 4 edge rule (widen, then reframe; the segment only)
    edge_rule: str = "include_exclude"
    w_shift: float = 1.0  # cost of a yaw shift, per half view width
    w_widen: float = 0.8  # cost of widening, per the room left up to the widest FOV
    # a person framed whole keeps this share off the sides: the detector sees a close person
    # near the edge 0.03-0.07 of the frame wider than its projected box (#11)
    include_margin: float = 0.06
    exclude_clear_deg: float = 3.0  # a person kept out stays this far beyond the edge
    include_max_shift: float = 0.35  # yaw shift of at most this share of half the view
    include_min_subject_h: float = 0.19  # widening keeps the subject this tall (or as is)
    include_bridge_s: float = 2.0  # bad stretches closer than this are fixed as one
    near_min_h_deg: float = 15.0  # other people this tall may intrude at the frame edge
    near_box_scale: float = 1.0  # near people's boxes scaled by this (about the centre)
    near_min_score: float = 0.6  # ... and detected with at least this score
    cut_min_gain_s: float = 1.0  # time the QA counts as cut a cut-away removes, per cut
    cut_min_gap_s: float = 4.0  # ... and starts / ends this far from another cut, or merges

    @classmethod
    def legacy(cls) -> DirectorParams:
        """The settings every path used before the direction-quality work (A/B baseline).

        Pinned here, not taken from the defaults, so it stays the same when they change.
        It covers the parameters only: the ``follow`` kind tweak (:data:`KIND_TWEAKS`,
        added with the penalties) did not exist before, so a ``follow`` moment gets
        w_people 1.2 / w_forward 0.5 / no selfie even with these settings (older
        paths treated it as an unknown kind, no tweak). ``framing="bin"`` keeps the
        bin-centre path (no subject tracking, fitted FOV, safe-area pass or
        segment-edge shot rule).
        """
        return cls(
            dt=0.5,
            n_bins=36,
            w_people=1.0,
            w_group=0.35,
            w_beauty=0.3,
            w_forward=0.25,
            forward_width_deg=50.0,
            w_selfie_talk=0.55,
            w_selfie_group=0.25,
            selfie_base=0.05,
            selfie_mode="auto",
            selfie_ok_min=0.0,
            w_occlude=0.0,
            w_wearer=0.0,
            wearer_clear_deg=20.0,
            vertical_wearer_cone_deg=73.821,
            people_size="close",
            max_speed=45.0,
            max_accel=60.0,
            move_cost=0.12,
            cut_cost=1.6,
            min_hold_s=2.5,
            smooth_s=1.0,
            pitch_default=-8.0,
            pitch_min=-25.0,
            pitch_max=5.0,
            fov_wide=100.0,
            fov_friend=86.0,
            fov_selfie=110.0,
            selfie_pitch_up=12.0,
            vertical_fov_wide=62.0,
            vertical_fov_friend=56.0,
            vertical_fov_selfie=75.0,
            vertical_occlude_fov=86.0,
            framing="bin",
            subject_max_vfrac=0.62,
            subject_max_wfrac=0.5,
            safe_margin=0.05,
            fov_min=72.0,
            fov_max=110.0,
            vertical_fov_min=72.0 * 56 / 86,
            vertical_fov_max=110.0 * 56 / 86,
            fov_rate=12.0,
            dead_zone=0.2,
            headroom=0.10,
            edge_rule="legacy",
            w_shift=1.0,
            w_widen=0.8,
            include_margin=0.03,
            exclude_clear_deg=3.0,
            include_max_shift=0.35,
            include_min_subject_h=0.19,
            include_bridge_s=2.0,
            near_min_h_deg=15.0,
            near_box_scale=1.0,
            near_min_score=0.6,
            cut_min_gain_s=1.0,
            cut_min_gap_s=4.0,
        )


POSITIVE_PARAMS = (
    "dt",
    "n_bins",
    "max_speed",
    "max_accel",
    "subject_max_vfrac",
    "subject_max_wfrac",
    "fov_min",
    "fov_max",
    "vertical_fov_min",
    "vertical_fov_max",
    "fov_rate",
    "near_box_scale",
)
NONNEG_PARAMS = (
    "w_shift",
    "w_widen",
    "include_margin",
    "exclude_clear_deg",
    "include_max_shift",
    "include_min_subject_h",
    "include_bridge_s",
    "near_min_h_deg",
    "near_min_score",
    "cut_min_gain_s",
    "cut_min_gap_s",
)
CHOICE_PARAMS = {
    "selfie_mode": ("auto", "off"),
    "people_size": ("framing", "close"),
    "framing": ("subject", "bin"),
    "edge_rule": ("include_exclude", "legacy"),
}


def _coerce(name: str, default: Any, value: Any) -> Any:
    """``value`` (a string from the command line or a JSON value) as the field's type."""
    try:
        if isinstance(default, bool):
            if isinstance(value, bool):
                return value
            low = str(value).strip().lower()
            if low in ("1", "true", "yes", "on"):
                return True
            if low in ("0", "false", "no", "off"):
                return False
            raise ValueError(value)
        if isinstance(default, int):
            f = float(value)
            if isinstance(value, bool) or not math.isfinite(f) or not f.is_integer():
                raise ValueError(value)
            return int(f)
        if isinstance(default, float):
            f = float(value)
            if isinstance(value, bool) or not math.isfinite(f):
                raise ValueError(value)
            return f
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"director param {name}: {value!r} is not a valid {type(default).__name__}"
        ) from err
    return str(value)


def with_params(base: DirectorParams, overrides: dict[str, Any]) -> DirectorParams:
    """``base`` with ``overrides`` (``DirectorParams`` field -> value, type-coerced).

    Raises:
        ValueError: On an unknown field, a value of the wrong type, a non-finite
            number, a non-positive :data:`POSITIVE_PARAMS` value, a negative
            :data:`NONNEG_PARAMS` value, an ``include_margin`` of 0.5 or more or a
            :data:`CHOICE_PARAMS` value not among its choices.
    """
    defaults = {f.name: getattr(base, f.name) for f in fields(base)}
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ValueError(
            f"unknown director param(s): {', '.join(unknown)} (known: {', '.join(defaults)})"
        )
    vals = {k: _coerce(k, defaults[k], v) for k, v in overrides.items()}
    for k in POSITIVE_PARAMS:
        if k in vals and vals[k] <= 0:
            raise ValueError(f"director param {k}: must be > 0, got {vals[k]!r}")
    for k in NONNEG_PARAMS:
        if k in vals and vals[k] < 0:
            raise ValueError(f"director param {k}: must be >= 0, got {vals[k]!r}")
    if vals.get("include_margin", 0.0) >= 0.5:
        raise ValueError(
            f"director param include_margin: must be < 0.5, got {vals['include_margin']!r}"
        )
    for k, choices in CHOICE_PARAMS.items():
        if k in vals and vals[k] not in choices:
            raise ValueError(f"director param {k}: one of {', '.join(choices)}, got {vals[k]!r}")
    return replace(base, **vals)


def config_overrides() -> dict[str, Any]:
    """``pano.director`` of config.json (``{}`` when unset), validated.

    Raises:
        ValueError: On a non-object ``pano.director``, an unknown field or a bad value.
    """
    from cutpoint.pano.project import pano_config  # noqa: PLC0415

    conf = pano_config().get("director") or {}
    if not isinstance(conf, dict):
        raise ValueError("config.json: 'pano.director' must be an object")
    with_params(DirectorParams(), conf)
    return dict(conf)


def parse_param_args(items: list[str]) -> dict[str, str]:
    """``["k=v", ...]`` (``--param``) as a dict; later keys win.

    Raises:
        ValueError: On an item without ``=`` or with an empty key.
    """
    out: dict[str, str] = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k.strip():
            raise ValueError(f"--param expects KEY=VALUE, got {it!r}")
        out[k.strip()] = v.strip()
    return out


# With selfie_mode "off" (the default) the selfie tweaks below (w_selfie_talk of
# laughter / voice / social / speech, selfie_base of wipeout / crash) have no effect;
# they apply with selfie_mode="auto" (--param selfie_mode=auto, or legacy()).
KIND_TWEAKS: dict[str, dict[str, Any]] = {
    "laughter": {"w_selfie_talk": 0.8},
    "voice_laughter": {"w_selfie_talk": 0.8},
    "voice": {"w_selfie_talk": 0.75},
    "social": {"w_selfie_talk": 0.75},
    "speech": {"w_selfie_talk": 0.7},
    "people": {"w_people": 1.3},
    "group": {"w_group": 0.6},
    "beauty": {"w_beauty": 0.8},
    "scenic": {"w_beauty": 0.8},
    "airtime": {"w_forward": 0.5},
    "speed": {"w_forward": 0.5},
    "carve": {"w_forward": 0.5},
    "follow": {"w_people": 1.2, "w_forward": 0.5, "selfie_mode": "off"},
    "wipeout": {"w_forward": 0.45, "selfie_base": 0.25},
    "crash": {"w_forward": 0.45, "selfie_base": 0.25},
}


def params_for(kind: str | None, base: DirectorParams | None = None) -> DirectorParams:
    """Director settings adjusted for the moment's dominant feature (``kind``)."""
    p = base or DirectorParams()
    return replace(p, **KIND_TWEAKS.get(kind or "", {}))


def _voice_at(src: Path, times: np.ndarray) -> np.ndarray:
    """max(laughter, voice) of the voice layer at ``times`` (0 without the layer)."""
    from cutpoint.events import voice as VO  # noqa: PLC0415

    vg = VO.load_voice(src)
    if vg is None:
        return np.zeros(len(times))
    vf = VO.voice_features(vg)
    x = np.maximum(vf["laughter"], vf["voice"])
    idx = np.clip(np.round(times / 0.1).astype(int), 0, len(x) - 1)
    # 1 s max-pool: talk is bursty
    pooled = np.array([x[max(0, i - 5) : i + 6].max() for i in idx])
    return pooled


def _interp_frames(
    pv_t: np.ndarray, values: np.ndarray, times: np.ndarray, hold: float = 1.0
) -> np.ndarray:
    """Nearest-frame resampling of ``(N, ...)`` values to ``times`` (0 beyond ``hold``)."""
    from cutpoint.events.common import nearest_index  # noqa: PLC0415

    if not len(pv_t):
        return np.zeros((len(times), *values.shape[1:]))
    pick = nearest_index(pv_t, times)
    out = values[pick].astype(float)
    far = np.abs(pv_t[pick] - times) > hold
    out[far] = 0.0
    return out


def _gauss(x: np.ndarray, sigma_steps: float) -> np.ndarray:
    if len(x) < 3 or sigma_steps <= 0:
        return x.copy()
    return G.gauss_smooth(x, sigma_steps)


def _limit(y: np.ndarray, dt: float, vmax: float, amax: float) -> np.ndarray:
    """Speed / acceleration limit (forward and backward pass, averaged)."""

    def one(a: np.ndarray) -> np.ndarray:
        out = a.copy()
        v = 0.0
        for i in range(1, len(out)):
            want = (a[i] - out[i - 1]) / dt
            dv = np.clip(want - v, -amax * dt, amax * dt)
            v = float(np.clip(v + dv, -vmax, vmax))
            out[i] = out[i - 1] + v * dt
        return out

    if len(y) < 3:
        return y.copy()
    f = one(y)
    b = one(y[::-1])[::-1]
    return 0.5 * (f + b)


NO_SELFIE = -1e3  # selfie score where the selfie is off or gated (never chosen)


def wearer_in_view(
    orient: G.Orientation,
    axis: np.ndarray,
    times: np.ndarray,
    centres: np.ndarray,
    p: DirectorParams,
    vertical: bool = False,
) -> np.ndarray:
    """``(T, nb)`` 1 where the wearer's direction is within half the view diagonal
    (``fov_wide`` at 16:9) plus ``wearer_clear_deg`` of the bin's axis at ``pitch_default``
    (9:16: within ``vertical_wearer_cone_deg``)."""
    half_h = math.radians(p.fov_wide) / 2
    half_diag = math.degrees(math.atan(math.tan(half_h) * math.hypot(1.0, 9 / 16)))
    cone = p.vertical_wearer_cone_deg if vertical else half_diag + p.wearer_clear_deg
    lim = math.cos(math.radians(min(180.0, cone)))
    axes = G.yaw_pitch_ray(centres, np.full(len(centres), p.pitch_default))  # (nb, 3)
    wd = np.stack([orient.r_vw(float(t)).T @ axis for t in times])  # (T, 3)
    wd /= np.linalg.norm(wd, axis=1, keepdims=True) + 1e-9
    return (wd @ axes.T > lim).astype(float)


def score_grid(
    clip: Any,
    t0: float,
    t1: float,
    p: DirectorParams,
    pv: Any,
    orient: G.Orientation | None = None,
    vertical: bool = False,
) -> dict[str, Any]:
    """Per-step, per-state scores for ``[t0, t1]`` (states: yaw bins + selfie).

    ``orient`` (the clip's orientation track) places the wearer for the
    ``w_wearer`` penalty (``vertical``: the 9:16 cone); without it the penalty is skipped.
    """
    from cutpoint.pano.views import direction_scores, wearer_axis  # noqa: PLC0415

    times = np.arange(t0, t1 + 1e-6, p.dt)
    nb = p.n_bins
    ds = (
        direction_scores(
            pv,
            nb,
            people_size=p.people_size,
            view_pitch=p.pitch_default,
            view_fov=p.vertical_occlude_fov if vertical else p.fov_friend,
            occlude=bool(p.w_occlude),
        )
        if pv is not None
        else None
    )
    if ds is not None:
        pt = pv.t.astype(float)
        people = _interp_frames(pt, ds["people"], times)
        group = _interp_frames(pt, ds["group"], times)
        beauty = _interp_frames(pt, ds["beauty"], times)
        ppitch = _interp_frames(pt, np.nan_to_num(ds["pitch"], nan=-999.0), times)
        selfie_g = _interp_frames(pt, ds["selfie"], times)
        occlude = _interp_frames(pt, ds["occlude"], times)
        selfie_ok = _interp_frames(pt, ds["selfie_ok"], times)
        centres = ds["centres"]
    else:
        people = group = beauty = occlude = np.zeros((len(times), nb))
        ppitch = np.full((len(times), nb), -999.0)
        selfie_g = np.zeros(len(times))
        selfie_ok = np.ones(len(times))
        centres = np.arange(nb) * 360.0 / nb
    from cutpoint.pano.views import forward_yaw  # noqa: PLC0415

    fy = forward_yaw(pv) if pv is not None else 0.0
    dyaw = G.wrap_deg(centres - fy)
    fwd = np.exp(-0.5 * (dyaw / p.forward_width_deg) ** 2)
    yaw_score = (
        p.w_people * people + p.w_group * group + p.w_beauty * beauty + p.w_forward * fwd[None, :]
    )
    if p.w_occlude:
        yaw_score = yaw_score - p.w_occlude * occlude
    wearer = np.zeros((len(times), nb))
    if p.w_wearer and orient is not None:
        axis = wearer_axis(pv) if pv is not None else G.WEARER_V
        wearer = wearer_in_view(orient, axis, times, centres, p, vertical)
        yaw_score = yaw_score - p.w_wearer * wearer
    talk = _voice_at(Path(clip.source), times)
    selfie = p.selfie_base + p.w_selfie_talk * talk + p.w_selfie_group * selfie_g
    # Talking with a friend in view: the friend wins (people >> selfie).
    selfie = selfie * (1 - 0.6 * np.maximum(people, 0.0).max(1))
    if p.selfie_mode == "off":
        selfie = np.full(len(times), NO_SELFIE)
    else:  # a glove / sleeve / helmet top over the selfie view is no selfie
        selfie = np.where(selfie_ok >= p.selfie_ok_min, selfie, NO_SELFIE)
    return {
        "times": times,
        "centres": centres,
        "yaw_score": yaw_score,
        "forward_yaw": fy,
        "selfie": selfie,
        "person_pitch": ppitch,
        "people": people,
        "talk": talk,
        "occlude": occlude,
        "wearer": wearer,
        "selfie_ok": selfie_ok,
        # per view frame (not resampled): the strongest person per bin, for tracking
        "subject": ds["subject"] if ds is not None else None,
    }


def _viterbi(
    sg: dict[str, Any], p: DirectorParams, start_state: int | None = None, roll: int = 0
) -> np.ndarray:
    """Best state per step (``n_bins`` = selfie).

    ``roll``: solved with bin ``k`` at index ``k - roll``, so exact ties break as in that
    frame (an upright clip: ``n_bins / 2``, the ties of its views' ``-z`` frame)."""
    ys, sf = sg["yaw_score"], sg["selfie"]
    n, nb = ys.shape
    if roll:
        ys = np.roll(ys, -roll, 1)
        if start_state is not None and start_state < nb:
            start_state = (start_state - roll) % nb
    s = np.concatenate([ys, sf[:, None]], 1)
    ns = nb + 1
    c = sg["centres"]
    d = np.abs(G.wrap_deg(c[:, None] - c[None, :]))
    step = p.max_speed * p.dt
    trans = np.where(d <= step + 1e-6, p.move_cost * d / step, p.cut_cost)
    full = np.full((ns, ns), p.cut_cost)
    full[:nb, :nb] = trans
    full[nb, nb] = 0.0
    acc = s[0].copy()
    if start_state is not None:
        acc[np.arange(ns) != start_state] -= p.cut_cost
    back = np.zeros((n, ns), int)
    for i in range(1, n):
        cand = acc[:, None] - full  # from x to
        back[i] = cand.argmax(0)
        acc = cand.max(0) + s[i]
    path = np.zeros(n, int)
    path[-1] = int(acc.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    if roll:
        path = np.where(path < nb, (path + roll) % nb, path)
    return path


def _shots(
    states: np.ndarray,
    centres: np.ndarray,
    nb: int,
    p: DirectorParams,
    times: np.ndarray | None = None,
    segment: tuple[float, float] | None = None,
    selfie: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """``[start, end)`` step ranges between hard cuts, short shots merged away.

    With ``times`` and ``segment`` (``start, end`` s) a shot is also short when
    its part inside the segment is (the rendered clip starts and ends there):
    it then joins its neighbour on the segment side. With ``selfie`` (the
    per-step selfie score) a short shot is not folded into a selfie neighbour
    where the selfie is gated (:data:`NO_SELFIE`); the other neighbour takes it,
    or it stays.
    """
    step = p.max_speed * p.dt

    def is_cut(a: int, b: int) -> bool:
        if (a == nb) != (b == nb):
            return True
        if a == nb:
            return False
        d = abs(G.wrap_deg(centres[a] - centres[b]))
        return d > step + 1e-6

    bounds = [0] + [i for i in range(1, len(states)) if is_cut(states[i - 1], states[i])]
    bounds.append(len(states))
    shots = list(pairwise(bounds))
    min_steps = max(1, round(p.min_hold_s / p.dt))

    def short(k: int) -> tuple[bool, int]:
        """(too short, neighbour to join) of shot ``k``."""
        a, b = shots[k]
        j = k - 1 if k > 0 else k + 1
        if b - a < min_steps:
            return True, j
        if times is None or segment is None:
            return False, j
        s0, s1 = segment
        ta = float(times[a])
        tb = float(times[b]) if b < len(times) else math.inf
        inside = min(tb, s1) - max(ta, s0)
        if inside <= 0 or inside >= min(p.min_hold_s, s1 - s0) - 1e-6:
            return False, j
        if ta < s0 and k + 1 < len(shots):  # crosses the start: join the next shot
            return True, k + 1
        if tb > s1 and k > 0:  # crosses the end: join the previous one
            return True, k - 1
        return False, j

    def gated(k: int, j: int) -> bool:
        a, b = shots[k]
        return (
            selfie is not None
            and states[shots[j][0]] == nb
            and bool((selfie[a:b] <= NO_SELFIE).any())
        )

    changed = True
    while changed and len(shots) > 1:
        changed = False
        for k in range(len(shots)):
            is_short, j = short(k)
            if is_short and gated(k, j):
                other = 2 * k - j
                j = other if 0 <= other < len(shots) and not gated(k, other) else -1
            if is_short and j >= 0:
                # absorb into the neighbour: the neighbour keeps its target
                a, b = shots[k]
                src = states[shots[j][1] - 1] if j < k else states[shots[j][0]]
                states[a:b] = src
                lo, hi = min(shots[j][0], a), max(shots[j][1], b)
                shots[min(j, k)] = (lo, hi)
                del shots[max(j, k)]
                changed = True
                break
    return shots


LINK_DEG = 25.0  # a person in the next view frame this close in yaw is the same subject
TRACK_GAP_S = 2.6  # interpolate between linked detections at most this far apart
TRACK_HOLD_S = 1.0  # ... and hold the first / last one this long
CLOSE_MIN_H_DEG = 35.0  # other people this tall are close: any part at the edge shows
CLOSE_MIN_SHARE = 0.06  # ... once this share of the frame (close_edge_share) is inside
SAFE_MIN_S = 0.5  # the safe-area pass fixes longer violations only
SAFE_SPEED = 1.3  # its re-limit allows max_speed / max_accel times this
REFRAME_SHARE = 0.35  # its reframing shifts the yaw by up to this share of half the view
INCLUDE_TRIES = 6  # include / exclude: the cheapest solutions checked per step before the grid
GRID_TRIES = 12  # ... and the cheapest grid points checked after them
STICKY_MISS = 10.0  # ... cost of a step a forced side leaves unsolved
INCLUDE_MARGIN_RETRY = 0.5  # ... the last try before a cut-away: this share of include_margin
INCLUDE_ROUNDS = 3  # ... rounds per shot while they reduce the bad steps


def _track_subject(
    pv: Any,
    ds_subject: np.ndarray | None,
    times: np.ndarray,
    states: np.ndarray,
    shots: list[tuple[int, int]],
) -> np.ndarray:
    """``(T, 4)`` yaw, pitch, width, height of the framed person per step (NaN: nobody).

    Within each non-selfie shot the person is linked across view frames: the
    candidate (the strongest person of any bin, ``direction_scores`` ``subject``)
    within :data:`LINK_DEG` of the previous one and within 45 deg of the shot's
    bin; otherwise the chosen bin's own person (re-seed). Linked detections are
    interpolated linearly at the step times (yaw unwrapped).
    """
    out = np.full((len(times), 4), np.nan)
    if pv is None or ds_subject is None or not len(pv.t):
        return out
    nb = ds_subject.shape[1]
    centres = np.arange(nb) * 360.0 / nb
    pt = pv.t.astype(float)
    for a, b in shots:
        if states[a] >= nb:
            continue
        lo, hi = times[a] - TRACK_HOLD_S, times[b - 1] + TRACK_HOLD_S
        pts: list[np.ndarray] = []
        for f in np.flatnonzero((pt >= lo) & (pt <= hi)):
            i = a + int(np.abs(times[a:b] - pt[f]).argmin())
            row = ds_subject[f].astype(float)
            cands = np.unique(row[~np.isnan(row).any(1)], axis=0)
            pick = None
            if pts and len(cands):
                d = np.abs(G.wrap_deg(cands[:, 0] - pts[-1][1]))
                off = np.abs(G.wrap_deg(cands[:, 0] - centres[states[i]]))
                ok = np.flatnonzero((d < LINK_DEG) & (off <= 45.0))
                if len(ok):
                    pick = cands[ok[d[ok].argmin()]].copy()
            if pick is None and not np.isnan(row[states[i]]).any():
                pick = row[states[i]].copy()
            if pick is None:
                continue
            if pts:  # unwrap against the previous point
                pick[0] = pts[-1][1] + float(G.wrap_deg(pick[0] - pts[-1][1]))
            pts.append(np.r_[pt[f], pick])
        if not pts:
            continue
        tr = np.asarray(pts)
        for i in range(a, b):
            t = times[i]
            k = int(np.searchsorted(tr[:, 0], t))
            if k < len(tr) and abs(tr[k, 0] - t) < 1e-9:
                out[i] = tr[k, 1:]
            elif 0 < k < len(tr):
                t0, t1 = tr[k - 1, 0], tr[k, 0]
                if t1 - t0 <= TRACK_GAP_S:
                    u = (t - t0) / (t1 - t0)
                    out[i] = (1 - u) * tr[k - 1, 1:] + u * tr[k, 1:]
            elif k == 0 and tr[0, 0] - t <= TRACK_HOLD_S:
                out[i] = tr[0, 1:]
            elif k == len(tr) and t - tr[-1, 0] <= TRACK_HOLD_S:
                out[i] = tr[-1, 1:]
    return out


def fit_fov(
    w: np.ndarray | float, h: np.ndarray | float, p: DirectorParams, size: tuple[int, int]
) -> np.ndarray:
    """Smallest horizontal FOV (deg) at which a centred ``w`` x ``h`` deg subject is at
    most ``subject_max_wfrac`` of the frame width and ``subject_max_vfrac`` of its
    height (the inverse of :func:`geometry.flat_fov`; unclamped)."""
    width, height = size
    th = np.tan(np.radians(np.clip(h, 0.0, 170.0)) / 2) / p.subject_max_vfrac * width / height
    tw = np.tan(np.radians(np.clip(w, 0.0, 170.0)) / 2) / p.subject_max_wfrac
    return np.degrees(2 * np.arctan(np.maximum(th, tw)))


def _dead_zone(target: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """Camera yaw that moves only when ``target`` leaves ``+-zone`` around it."""
    out = np.empty(len(target))
    c = float(target[0]) if len(target) else 0.0
    for i, (t, z) in enumerate(zip(target, zone, strict=True)):
        if t - c > z:
            c = t - z
        elif c - t > z:
            c = t + z
        out[i] = c
    return out


def _refine_shot(
    bins_yaw: np.ndarray,
    subj: np.ndarray,
    people: np.ndarray,
    person_pitch: np.ndarray,
    p: DirectorParams,
    fovs: tuple[float, float, float, float],
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(yaw, pitch, fov)`` of a non-selfie shot framing the tracked subject.

    ``bins_yaw``: the shot's bin centres (unwrapped); ``subj`` ``(n, 4)``
    (:func:`_track_subject`); ``fovs``: friend, wide, fitted min and max FOV.
    Friend steps (people > 0.3) with a subject get the fitted FOV
    (:func:`fit_fov`, clamped) and aim at it with a dead zone and headroom;
    the FOV is smoothed and limited to ``fov_rate``.
    """
    ff, fw, lo, hi = fovs
    sm = p.smooth_s / p.dt
    has = ~np.isnan(subj).any(1)
    friend = people > 0.3
    sy = np.where(has, subj[:, 0], bins_yaw)
    target = bins_yaw + G.wrap_deg(sy - bins_yaw)
    raw_fov = np.where(friend, ff, fw)
    fit = np.clip(fit_fov(subj[:, 2], subj[:, 3], p, size), lo, hi)
    raw_fov = np.where(friend & has, fit, raw_fov)
    zone = np.where(has, p.dead_zone * raw_fov / 2, 0.0)
    yaw = _limit(_gauss(_dead_zone(target, zone), sm), p.dt, p.max_speed, p.max_accel)
    aim = np.clip(subj[:, 1] + p.headroom * subj[:, 3], p.pitch_min, p.pitch_max)
    old = (person_pitch > -900) & (people > 0.15)
    raw_pitch = np.where(old, np.clip(person_pitch, p.pitch_min, p.pitch_max), p.pitch_default)
    raw_pitch = np.where(has & (people > 0.15), aim, raw_pitch)
    pitch = _gauss(raw_pitch, 2 * sm)
    fov = _limit(_gauss(raw_fov, sm), p.dt, p.fov_rate, 2 * p.fov_rate)
    return yaw, pitch, fov


def _near_people(
    pv: Any, times: np.ndarray, min_h: float = 15.0, min_score: float | None = None
) -> list[np.ndarray]:
    """Per time: ``(k, 6)`` yaw, pitch, w, h, is_self (1: the wearer), score of people
    at least ``min_h`` deg tall detected with at least ``min_score`` (default the QA's
    ``person_min_score``) in the nearest view frame (the geometric QA's rows).

    The wearer is judged by its in-frame box (:func:`_framing_checks`)."""
    from cutpoint.pano import qa_paths as QP  # noqa: PLC0415
    from cutpoint.pano.qa import QAThresholds  # noqa: PLC0415

    empty = np.zeros((0, 6))
    if pv is None or not len(pv.det):
        return [empty] * len(times)
    d = pv.det
    person, is_self = QP.person_rows(pv)
    floor = QAThresholds().person_min_score if min_score is None else min_score
    keep = person & (d[:, 2] >= floor)
    keep &= d[:, 6] >= min_h
    rows = np.column_stack([d[:, 3:7].astype(float), is_self.astype(float), d[:, 2]])
    fi = d[:, 0].astype(int)
    idx = QP.view_frame_index(pv.t.astype(float), times)
    by_frame = {int(j): rows[keep & (fi == j)] for j in np.unique(idx) if j >= 0}
    return [by_frame[int(j)] if j >= 0 else empty for j in idx]  # one array per view frame


def close_edge_share(
    view_yaw: float, view_pitch: float, hfov: float, size: tuple[int, int], box: np.ndarray
) -> float:
    """Share of the frame a close person's ``(yaw, pitch, w, h)`` box covers where it
    crosses the left or right frame edge (0 when it does not cross one).

    Angular estimate (box extent against ``hfov`` and the vertical FOV): near the
    camera a person's equirect box is mostly outside any view, and the projected
    box (:func:`geometry.project_box_visible`) drops such slivers, while the
    detector on the rendered frame still sees the head or shoulder at the edge.
    """
    vfov = G.flat_fov(hfov, *size)
    d = float(G.wrap_deg(box[0] - view_yaw))
    lo, hi = d - box[2] / 2, d + box[2] / 2
    half = hfov / 2
    if not (lo < -half < hi or lo < half < hi):
        return 0.0
    w = max(0.0, min(hi, half) - max(lo, -half)) / hfov
    top, bottom = box[1] + box[3] / 2, box[1] - box[3] / 2
    h = max(0.0, min(top, view_pitch + vfov / 2) - max(bottom, view_pitch - vfov / 2)) / vfov
    return float(w * h)


def project_boxes(
    view_yaw: float,
    view_pitch: float,
    hfov: float,
    size: tuple[int, int],
    boxes: np.ndarray,
    steps: int = 9,
    polar_shift: float = 0.5,
) -> list[tuple[tuple[float, float, float, float], float] | None]:
    """:func:`geometry.project_box_visible` of ``(k, 4)`` boxes (yaw, pitch, w, h) in one
    view, all at once (the same points and rules; the safe-area pass projects every
    person of a view per check)."""
    k = len(boxes)
    if not k:
        return []
    b = np.asarray(boxes, float)
    by, bp, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    vfov = G.flat_fov(hfov, *size)
    tx, ty = math.tan(math.radians(hfov) / 2), math.tan(math.radians(vfov) / 2)
    r = G.view_matrix(view_yaw, view_pitch)
    s = np.linspace(-0.5, 0.5, steps)
    lim = G.POLAR_LINE_MAX_DEG
    p_w = np.clip(bp + np.copysign(polar_shift * bh / 2, bp), -lim, lim)
    # vertical lines at every grid yaw (k, steps yaws, steps pitches), then the yaw line
    shape = (k, steps, steps)
    gy = np.broadcast_to(by[:, None, None] + s[None, :, None] * bw[:, None, None], shape)
    gp = np.broadcast_to(bp[:, None, None] + s[None, None, :] * bh[:, None, None], shape)
    hyaw = (by[:, None] + s[None, :] * bw[:, None])[:, None, :]
    py = np.concatenate([gy, hyaw], 1)
    pp = np.concatenate([gp, np.broadcast_to(p_w[:, None, None], hyaw.shape)], 1)
    ax, ay, az = G._view_xy(r, tx, ty, py, pp)  # the same points as project_box_visible
    vx, vy, vz = ax[:, :steps], ay[:, :steps], az[:, :steps]
    hx, hy, hz = ax[:, steps], ay[:, steps], az[:, steps]
    centre = steps // 2
    front_c = vz[:, centre, centre] > 1e-6  # the box centre in front of the camera
    seen = (np.abs(vx) <= 1).any(2)  # (k, steps): per vertical line
    far = np.where(seen, np.abs(np.arange(steps) - centre)[None, :], steps + 1)
    line = np.where(seen[:, centre], centre, np.where(seen.any(1), far.argmin(1), centre))
    rows = np.arange(k)
    xs = np.concatenate([hx, vx[rows, line]], 1)
    y = np.concatenate([hy, vy[rows, line]], 1)
    in_x = np.abs(xs) <= 1
    use = np.where(in_x.any(1)[:, None], in_x, True)
    ys_min = np.where(use, y, np.inf).min(1)
    ys_max = np.where(use, y, -np.inf).max(1)
    front = np.concatenate([hz, vz[:, centre]], 1) > G.FRONT_Z
    yc = np.concatenate([hy, vy[:, centre]], 1)
    fy_min = np.minimum(np.where(front, yc, np.inf).min(1), ys_min)
    fy_max = np.maximum(np.where(front, yc, -np.inf).max(1), ys_max)
    x_min, x_max = xs.min(1), xs.max(1)
    full = np.clip(np.stack([x_min, fy_min, x_max, fy_max], 1), -3.0, 3.0)
    full_area = (full[:, 2] - full[:, 0]) * (full[:, 3] - full[:, 1])
    vis = np.clip(np.stack([x_min, ys_min, x_max, ys_max], 1), -1.0, 1.0)
    vis_area = np.maximum(0.0, vis[:, 2] - vis[:, 0]) * np.maximum(0.0, vis[:, 3] - vis[:, 1])
    share = np.divide(vis_area, full_area, out=np.zeros(k), where=full_area > 0)
    box = (np.stack([x_min, ys_min, x_max, ys_max], 1) + 1) / 2
    return [
        (tuple(float(v) for v in box[i]), float(share[i])) if front_c[i] else None for i in range(k)
    ]


def _framing_checks(ctx: dict[str, Any], fix: tuple[int, int], p: DirectorParams) -> Any:
    """``offenders(i, f, y)``, ``bad(i, f, y)``, ``runs(a, b, min_run)``, ``n_bad(a, b)``,
    ``subject(i)`` and ``constraints(i)`` on the current ``ctx`` path (see
    :func:`_safe_area_pass`); ``runs`` and ``n_bad`` count the steps of ``fix``.

    ``edge_rule="legacy"``: a step is bad by the version 4 rule (``ctx["near"]``).
    ``include_exclude``: by :func:`_include_exclude`'s rule (``ctx["persons"]``, else
    ``near``), and a step is also bad when the view halfway to a neighbouring step is
    (``persons_mid``, the people at that time; else the earlier step's). That view is
    the average of the two (yaw unwrapped); a pair farther apart than the re-limit
    allows, or one across a shot's end (``ctx["cut_after"]``), is a cut, where the
    earlier view is held. ``bad(i, f, y)`` with a view checks step ``i``'s people and
    those halfway to its neighbours in that one view (``mids=False``: the step's only).
    """
    from cutpoint.pano import qa_paths as QP  # noqa: PLC0415
    from cutpoint.pano.qa import (  # noqa: PLC0415
        QAThresholds,
        edge_candidates,
        edge_hit,
        wearer_allowed,
    )

    th = QP.paths_thresholds(QAThresholds())  # the --paths QA's edge floor
    yaw, pitch, fov = ctx["yaw"], ctx["pitch"], ctx["fov"]
    subj, near, size = ctx["subj"], ctx["near"], ctx["size"]
    target = ctx["target"]
    legacy = p.edge_rule == "legacy"
    persons = ctx.get("persons", near)
    persons_mid = ctx.get("persons_mid") or persons[:-1]
    m = p.safe_margin
    base_run = int(SAFE_MIN_S / p.dt) + 1
    cut_deg = SAFE_SPEED * p.max_speed * p.dt + 1e-6  # farther per step: a cut
    cut_after = ctx.setdefault("cut_after", set())  # steps a shot ends at: a cut follows
    memo: dict[tuple, list[float]] = {}
    rays: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def subject(i: int) -> np.ndarray | None:
        """The framed subject of step ``i`` (None: none)."""
        if target[i] == "friend" and not np.isnan(subj[i]).any():
            return subj[i]
        return None

    def is_subject(box: np.ndarray, sj: np.ndarray | None) -> bool:
        return sj is not None and abs(G.wrap_deg(box[0] - sj[0])) < max(5.0, sj[2] / 2)

    def subject_bad(vy: float, vp: float, vf: float, sj: np.ndarray) -> bool:
        res = G.project_box_visible(vy, vp, vf, size[0], size[1], *sj)
        if res is None:
            return True
        x1, y1, x2, y2 = res[0]
        return x1 < m or x2 > 1 - m or (y1 < th.safe_top and y2 < th.edge_top_max_y2)

    def wearer_bad(vy: float, vp: float, vf: float, box: np.ndarray) -> bool:
        r = G.project_box_visible(vy, vp, vf, size[0], size[1], *box)
        return r is not None and not wearer_allowed(np.clip(r[0], 0.0, 1.0), th)

    def legacy_offenders(i: int, f: float | None = None, y: float | None = None) -> list[float]:
        """Yaws of what is badly framed at step ``i`` (empty: fine)."""
        if target[i] == "selfie":
            return []
        vy, vp, vf = yaw[i] if y is None else y, pitch[i], fov[i] if f is None else f
        sj = subject(i)
        if sj is not None and subject_bad(vy, vp, vf, sj):
            return [float(sj[0])]
        out = []
        for row in near[i]:
            box = row[:4]
            if len(row) > 4 and row[4]:  # the wearer: by its whole in-frame box, slivers too
                if wearer_bad(vy, vp, vf, box):
                    out.append(float(box[0]))
                continue
            if is_subject(box, sj):
                continue  # the subject itself
            if box[3] >= CLOSE_MIN_H_DEG and close_edge_share(vy, vp, vf, size, box) >= (
                CLOSE_MIN_SHARE
            ):
                out.append(float(box[0]))
                continue
            r = G.project_box_visible(vy, vp, vf, size[0], size[1], *box)
            if r is None or r[1] < 0.25:
                continue
            bx1, by1, bx2, by2 = (min(1.0, max(0.0, v)) for v in r[0])
            if (bx2 - bx1) * (by2 - by1) < th.edge_min_area:
                continue
            if bx1 <= th.edge_margin or bx2 >= 1 - th.edge_margin:
                out.append(float(box[0]))
        return out

    def score(rows: np.ndarray) -> np.ndarray:
        return rows[:, 5] if rows.shape[1] > 5 else np.ones(len(rows))

    def is_self(rows: np.ndarray) -> np.ndarray:
        return rows[:, 4] > 0 if rows.shape[1] > 4 else np.zeros(len(rows), bool)

    def near_rows(rows: np.ndarray, sj: np.ndarray | None) -> np.ndarray:
        """``(k, 4)`` near people of ``rows`` (not the wearer or the subject), boxes scaled
        by ``near_box_scale``."""
        ok = ~is_self(rows) & (score(rows) >= p.near_min_score) & (rows[:, 3] >= p.near_min_h_deg)
        out = [
            [r[0], r[1], r[2] * p.near_box_scale, r[3] * p.near_box_scale]
            for r in rows[ok]
            if not is_subject(r, sj)
        ]
        return np.asarray(out, float).reshape(-1, 4)

    def in_view(rows: np.ndarray, vy: float, vp: float, vf: float) -> np.ndarray:
        """Per row: the box may reach into the view. The others lie wholly in front of
        the camera and beyond the view's corners (or have their centre behind it): out
        for every rule, so they are not projected."""
        if not len(rows):
            return np.zeros(0, bool)
        t = math.tan(math.radians(vf) / 2)
        corner = math.degrees(math.atan(t * math.hypot(1.0, size[1] / size[0])))
        if id(rows) not in rays:  # per view frame: the box centres and their reach
            reach = (rows[:, 2] + rows[:, 3]) / 2 * max(1.0, p.near_box_scale)
            reach += 2 * p.exclude_clear_deg + np.maximum(np.abs(rows[:, 1]) - 80.0, 0.0) + 1.0
            rays[id(rows)] = (G.yaw_pitch_ray(rows[:, 0], rows[:, 1]), reach)
        ray, reach = rays[id(rows)]
        dist = np.degrees(np.arccos(np.clip(ray @ G.yaw_pitch_ray(vy, vp), -1.0, 1.0)))
        out = (dist - reach > corner) & (dist + reach < 89.0)
        return (dist < 90.5) & ~out

    def view_offenders(
        vy: float, vp: float, vf: float, tg: str, sj: np.ndarray | None, rows: np.ndarray
    ) -> tuple[list[float], bool]:
        """``(yaws, hard)``: what is badly framed in one view by the include / exclude
        rule (the subject's safe area, the wearer, near people split by the edge and the
        geometric QA's edge cuts), and whether the geometric QA counts an edge cut or the
        subject is out of view."""
        if tg == "selfie":
            return [], False
        rows = rows[in_view(rows, vy, vp, vf)]
        near = near_rows(rows, sj)
        c = 2 * p.exclude_clear_deg
        wide = near + np.array([0.0, 0.0, c, c])  # OUT: clear of the view
        own = [] if sj is None else [sj[:4]]
        res = project_boxes(
            vy, vp, vf, size, np.concatenate([np.reshape(own, (-1, 4)), rows[:, :4], near, wide])
        )
        out = []
        lost = False
        if sj is not None:
            r = res.pop(0)
            lost = r is None or not (r[0][0] < 1 and r[0][2] > 0 and r[0][1] < 1 and r[0][3] > 0)
            if r is None:
                out.append(float(sj[0]))
            else:
                x1, y1, x2, y2 = r[0]
                if x1 < m or x2 > 1 - m or (y1 < th.safe_top and y2 < th.edge_top_max_y2):
                    out.append(float(sj[0]))
        res_rows, res_near, res_wide = (
            res[: len(rows)],
            res[len(rows) : len(rows) + len(near)],
            res[len(rows) + len(near) :],
        )
        ok = score(rows) >= th.person_min_score
        for k in np.flatnonzero(is_self(rows) & ok):
            r = res_rows[k]
            if r is not None and not wearer_allowed(np.clip(r[0], 0.0, 1.0), th):
                out.append(float(rows[k, 0]))
        for bx, r, rw in zip(near, res_near, res_wide, strict=True):
            if r is not None and p.include_margin <= r[0][0] <= r[0][2] <= 1 - p.include_margin:
                continue  # IN: framed whole
            if rw is not None and not (
                rw[0][2] <= 0 or rw[0][0] >= 1 or rw[0][3] <= 0 or rw[0][1] >= 1
            ):
                out.append(float(bx[0]))  # SPLIT
        # the geometric QA's edge cuts: only boxes touching the sides or the top count
        touch = [
            r is not None
            and (
                r[0][0] <= th.edge_margin
                or r[0][2] >= 1 - th.edge_margin
                or r[0][1] <= th.edge_margin
            )
            for r in res_rows
        ]
        keep_k = np.flatnonzero(ok)
        if not np.asarray(touch, bool)[keep_k].any():
            return out, lost
        qa_rows = rows[keep_k]
        det, sel, sl, src, sl_src = QP.project_frame(
            qa_rows[:, :4],
            score(qa_rows),
            is_self(qa_rows),
            0,
            vy,
            vp,
            vf,
            size,
            [res_rows[k] for k in keep_k],
        )
        det, sel = np.asarray(det, float).reshape(-1, 7), np.asarray(sel, bool)
        sl = np.asarray(sl, float).reshape(-1, 8)
        keep = QP._nms(det)
        det, sel, src = det[keep], sel[keep], np.asarray(src, int)[keep]
        hit = src[edge_candidates(det, sel, th) & edge_hit(det, th)]
        out += [float(qa_rows[k, 0]) for k in hit]
        qa = len(hit) > 0
        if len(sl) and (
            QP.sliver_edge_frames(sl, th) or QP.sliver_subject_frames(sl, det, sel, th)
        ):
            out += [float(qa_rows[k, 0]) for k in sl_src]
            qa = True
        return out, qa or lost

    def at(
        vy: float, vp: float, vf: float, tg: str, sj: np.ndarray | None, rows: np.ndarray
    ) -> tuple[list[float], bool]:
        """:func:`view_offenders`, memoised (steps of one view frame share ``rows``)."""
        key = (id(rows), vy, vp, vf, tg, None if sj is None else tuple(sj))
        if key not in memo:
            memo[key] = view_offenders(vy, vp, vf, tg, sj, rows)
        return memo[key]

    def is_cut(i: int) -> bool:
        """A cut between steps ``i`` and ``i + 1``."""
        return i in cut_after or abs(float(G.wrap_deg(yaw[i + 1] - yaw[i]))) > cut_deg

    def held(i: int) -> list[np.ndarray]:
        """People step ``i``'s view is held over until a cut to step ``i + 1``: halfway
        and at the cut (a frame just before it shows them)."""
        return [r for r in (persons_mid[i], persons[i + 1]) if r is not persons[i]]

    def view(i: int, f: float | None = None, y: float | None = None) -> tuple[float, ...]:
        return float(yaw[i] if y is None else y), float(pitch[i]), float(fov[i] if f is None else f)

    def offenders(i: int, f: float | None = None, y: float | None = None) -> list[float]:
        """Yaws of what is badly framed at step ``i`` (empty: fine)."""
        if legacy:
            return legacy_offenders(i, f, y)
        return at(*view(i, f, y), target[i], subject(i), persons[i])[0]

    def wearer_seen(i: int) -> bool:
        """Step ``i``'s view shows part of the wearer (any self box, allowed or not)."""
        rows = persons[i]
        own = rows[is_self(rows)] if target[i] != "selfie" else rows[:0]
        if not len(own):
            return False
        return any(  # in front and overlapping the frame
            r is not None and r[0][0] < 1 and r[0][2] > 0 and r[0][1] < 1 and r[0][3] > 0
            for r in project_boxes(*view(i), size, own[:, :4])
        )

    def held_bad(j: int) -> bool:
        """Step ``j``'s view, held until a cut after it, is hard bad at step ``j + 1``'s
        time (:func:`qa_bad`)."""
        if legacy or j < 0 or j + 1 >= len(yaw) or target[j] == "selfie":
            return False
        return at(*view(j), target[j], subject(j), persons[j + 1])[1]

    def qa_bad(i: int, f: float | None = None, y: float | None = None) -> bool:
        """Step ``i`` is hard bad: the geometric QA counts an edge cut or the subject is
        out of view (in its view, or the view held until a cut to it, at its time; with
        ``f`` / ``y``: in that view only)."""
        if legacy:
            return False
        if at(*view(i, f, y), target[i], subject(i), persons[i])[1]:
            return True
        if f is not None or y is not None:
            return False
        j = i - 1
        if j < 0 or target[j] == "selfie" or not is_cut(j) or persons[i] is persons[j]:
            return False
        return at(*view(j), target[j], subject(j), persons[i])[1]

    def mid_bad(i: int) -> bool:
        """The view halfway from step ``i`` to ``i + 1`` is badly framed (not checked
        where it is a step's view with that step's people)."""
        if i < 0 or i + 1 >= len(yaw) or "selfie" in (target[i], target[i + 1]):
            return False
        rows = persons_mid[i]
        d = float(G.wrap_deg(yaw[i + 1] - yaw[i]))
        sj, sj1 = subject(i), subject(i + 1)
        if is_cut(i):  # the earlier view is held
            return any(at(*view(i), target[i], sj, r)[0] for r in held(i))
        vy = float(yaw[i]) + d / 2
        vp, vf = float(pitch[i] + pitch[i + 1]) / 2, float(fov[i] + fov[i + 1]) / 2
        tg = target[i] if target[i] == target[i + 1] else "view"
        for j in (i, i + 1):
            same = abs(vy - yaw[j]) <= 0.5 and abs(vp - pitch[j]) <= 0.5
            if rows is persons[j] and same and abs(vf - fov[j]) <= 0.5 and tg == target[j]:
                return False
        if sj is not None and sj1 is not None:
            sj = (sj + sj1) / 2
            sj[0] = subj[i, 0] + float(G.wrap_deg(subj[i + 1, 0] - subj[i, 0])) / 2
        else:
            sj = None
        return bool(at(vy, vp, vf, tg, sj, rows)[0])

    def bad(i: int, f: float | None = None, y: float | None = None, mids: bool = True) -> bool:
        if legacy:
            return bool(legacy_offenders(i, f, y))
        if f is None and y is None:
            return bool(offenders(i)) or mid_bad(i - 1) or mid_bad(i)
        if offenders(i, f, y):
            return True
        if not mids:
            return False
        sj = subject(i)
        mids = [persons_mid[j] for j in (i - 1, i) if 0 <= j < len(persons_mid)]
        return any(at(*view(i, f, y), target[i], sj, r)[0] for r in mids if r is not persons[i])

    def flags(a: int, b: int) -> tuple[int, list[bool]]:
        lo, hi = max(a, fix[0]), min(b, fix[1])
        return lo, [bad(i) for i in range(lo, hi)]

    def runs(a: int, b: int, min_run: int | None = None) -> list[tuple[int, int]]:
        """Bad stretches of ``[a, b)`` inside ``fix`` at least ``min_run`` steps long
        (default: longer than :data:`SAFE_MIN_S`)."""
        need = base_run if min_run is None else min_run
        lo, fl = flags(a, b)
        out, i = [], 0
        while i < len(fl):
            j = i
            while j < len(fl) and fl[j]:
                j += 1
            if j > i and j - i >= need:
                out.append((lo + i, lo + j))
            i = max(j, i + 1)
        return out

    def n_bad(a: int, b: int) -> tuple[int, int]:
        """``(hard bad steps, bad steps)`` of ``[a, b)`` inside ``fix`` (:func:`qa_bad`;
        legacy: none hard), compared in that order: a fix never adds hard bad steps."""
        lo, fl = flags(a, b)
        return sum(qa_bad(i) for i in range(lo, lo + len(fl))), sum(fl)

    def constraints(i: int) -> np.ndarray:
        """``(k, 4)`` boxes :func:`_solve_step` frames whole or keeps out at step ``i``: the
        near people of the step and halfway to its neighbours, and the other people the
        QA rule counts there now."""
        sj = subject(i)
        sets = [near_rows(persons[i], sj)]
        sets += [near_rows(persons_mid[j], sj) for j in (i - 1, i) if 0 <= j < len(persons_mid)]
        hit = set(offenders(i))
        rows = persons[i]
        extra = [
            r[:4] for r in rows[~is_self(rows)] if float(r[0]) in hit and not is_subject(r, sj)
        ]
        sets.append(np.asarray(extra, float).reshape(-1, 4))
        return np.concatenate(sets)

    return SimpleNamespace(
        offenders=offenders,
        bad=bad,
        runs=runs,
        n_bad=n_bad,
        qa_bad=qa_bad,
        held_bad=held_bad,
        wearer_seen=wearer_seen,
        subject=subject,
        constraints=constraints,
    )


def _keep_if_better(ctx: dict[str, Any], chk: Any, a: int, b: int, apply: Any) -> bool:
    """Run ``apply()`` (edits yaw / fov of ``[a, b)``); revert unless the bad steps drop
    (``chk.n_bad`` of the changed steps and their neighbours: a step's checks reach the
    next one, and a cut from step ``b - 1`` may show at ``b``)."""
    yaw, fov = ctx["yaw"], ctx["fov"]
    keep_y, keep_f = yaw[a:b].copy(), fov[a:b].copy()
    apply()
    ch = np.flatnonzero((yaw[a:b] != keep_y) | (fov[a:b] != keep_f))
    if not len(ch):
        return False
    lo, hi = max(a, a + ch[0] - 1), min(b + 1, a + ch[-1] + 2)
    after = chk.n_bad(lo, hi)
    new_y, new_f = yaw[a:b].copy(), fov[a:b].copy()
    yaw[a:b], fov[a:b] = keep_y, keep_f
    if after < chk.n_bad(lo, hi):
        yaw[a:b], fov[a:b] = new_y, new_f
        return True
    return False


def _widen(ctx: dict[str, Any], chk: Any, a: int, b: int, p: DirectorParams) -> None:
    """Raise the FOV where that alone fixes a bad step (a cone envelope keeps ``fov_rate``)."""
    times, fov, hi = ctx["times"], ctx["fov"], ctx["fov_hi"]
    need = {}
    for r0, r1 in chk.runs(a, b):
        for i in range(r0, r1):
            ok = [f for f in np.arange(fov[i], hi + 1e-6, 2.0) if not chk.bad(i, f=float(f))]
            if ok:
                need[i] = float(ok[0])
    if not need:
        return

    def apply() -> None:
        for j in range(a, b):
            env = max(v - p.fov_rate * abs(times[j] - times[i]) for i, v in need.items())
            fov[j] = max(fov[j], min(env, hi))

    _keep_if_better(ctx, chk, a, b, apply)


def _subject_ceiling(
    ctx: dict[str, Any], chk: Any, a: int, b: int, p: DirectorParams
) -> np.ndarray:
    """Per step of ``[a, b)``: the widest FOV that keeps the subject at ``min(its height
    now, include_min_subject_h)`` (never below the FOV now; ``fov_hi`` without one)."""
    fov, size = ctx["fov"], ctx["size"]
    top = np.maximum(ctx["fov_hi"], fov[a:b])
    for k in range(a, b):
        sj = chk.subject(k)
        if sj is not None:
            floor = min(_height_share(sj[3], fov[k], size), p.include_min_subject_h)
            top[k - a] = max(fov[k], min(top[k - a], _max_fov(sj[3], floor, size)))
    return top


def _widen_only(
    ctx: dict[str, Any], chk: Any, a: int, b: int, p: DirectorParams, hard: bool = False
) -> bool:
    """:func:`_widen` for ``include_exclude``: each bad step of ``[a, b)`` gets the least
    FOV (2 deg steps) that frames it well, else one where it is not hard bad
    (``chk.qa_bad``), up to its :func:`_subject_ceiling`; a cone envelope keeps
    ``fov_rate``; each stretch is kept only when the bad steps drop. ``hard``: only the
    hard bad steps, to the least FOV where they are not. Returns whether any was kept."""
    times, fov = ctx["times"], ctx["fov"]
    top = _subject_ceiling(ctx, chk, a, b, p)
    kept = False
    for r0, r1 in chk.runs(a, b, 1):
        need = {}
        for i in range(r0, r1):
            fs = [float(f) for f in np.arange(fov[i] + 2.0, top[i - a] + 1e-6, 2.0)]
            ok = None if hard else next((f for f in fs if not chk.bad(i, f=f)), None)
            if ok is None and chk.qa_bad(i):
                ok = next((f for f in fs if not chk.qa_bad(i, f=f)), None)
            if ok is not None:
                need[i] = ok
        if not need:
            continue
        base_f = fov[a:b].copy()

        def apply(need=need, base_f=base_f) -> None:
            for j in range(a, b):
                env = max(v - p.fov_rate * abs(times[j] - times[i]) for i, v in need.items())
                fov[j] = max(base_f[j - a], min(env, top[j - a]))

        kept = _keep_if_better(ctx, chk, a, b, apply) or kept
    return kept


def _reframe(ctx: dict[str, Any], chk: Any, a: int, b: int, p: DirectorParams) -> None:
    """Shift the yaw (up to :data:`REFRAME_SHARE` of half the view) and / or change the
    FOV over each bad stretch, smoothed and limited."""
    yaw, fov, lo, hi = ctx["yaw"], ctx["fov"], ctx["fov_lo"], ctx["fov_hi"]
    sm = p.smooth_s / p.dt
    for r0, r1 in chk.runs(a, b):
        dy, df = np.zeros(b - a), np.zeros(b - a)
        hit = np.zeros(b - a, bool)
        for i in range(r0, r1):
            best = None
            for f in np.arange(lo, hi + 1e-6, 4.0):
                lim = REFRAME_SHARE * f / 2
                for o in np.arange(-lim, lim + 1e-6, 2.5):
                    cost = abs(o) + 0.5 * abs(f - fov[i])
                    if (best is None or cost < best[0]) and not chk.bad(
                        i, f=float(f), y=float(yaw[i] + o)
                    ):
                        best = (cost, o, f - fov[i])
            if best is not None:
                dy[i - a], df[i - a], hit[i - a] = best[1], best[2], True
        if not hit.any():
            continue
        idx = np.flatnonzero(hit)
        for j in range(max(0, idx[0] - 2), min(b - a, idx[-1] + 3)):  # hold, then ease
            k = idx[np.abs(idx - j).argmin()]
            dy[j], df[j] = dy[k], df[k]
        base_y, base_f = yaw[a:b].copy(), fov[a:b].copy()

        def apply(base_y=base_y, base_f=base_f, dy=dy, df=df) -> None:
            fov[a:b] = np.clip(
                _limit(base_f + _gauss(df, sm / 2), p.dt, p.fov_rate, 2 * p.fov_rate), lo, hi
            )
            yaw[a:b] = _limit(
                base_y + _gauss(dy, sm / 2),
                p.dt,
                SAFE_SPEED * p.max_speed,
                SAFE_SPEED * p.max_accel,
            )

        _keep_if_better(ctx, chk, a, b, apply)


def _height_share(h: float, f: float, size: tuple[int, int]) -> float:
    """Share of the frame height a centred ``h`` deg tall box covers at horizontal FOV ``f``."""
    vf = G.flat_fov(f, *size)
    return math.tan(math.radians(min(h, 170.0)) / 2) / math.tan(math.radians(vf) / 2)


def _max_fov(h: float, share: float, size: tuple[int, int]) -> float:
    """Widest horizontal FOV at which a ``h`` deg tall box keeps ``share`` of the frame
    height (inf: any)."""
    if share <= 0:
        return math.inf
    t = math.tan(math.radians(min(h, 170.0)) / 2) / share * size[0] / size[1]
    return math.degrees(2 * math.atan(t))


def _intersect(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Intersection of two sorted lists of disjoint closed intervals."""
    out = []
    for lo_a, hi_a in a:
        for lo_b, hi_b in b:
            lo, hi = max(lo_a, lo_b), min(hi_a, hi_b)
            if lo <= hi:
                out.append((lo, hi))
    return sorted(out)


def _solve_step(
    ctx: dict[str, Any], chk: Any, i: int, p: DirectorParams, sign: int = 0
) -> tuple[float, float, float] | None:
    """``(cost, yaw shift, FOV change)`` that frames step ``i`` well, cheapest first (None:
    none found); ``sign`` +1 / -1 allows shifts to that side only.

    Never narrows; widens in 2 deg steps up to ``fov_hi`` while the subject keeps
    ``min(its height now, include_min_subject_h)`` of the frame height. For each FOV the
    yaw shifts (at most ``include_max_shift`` of half the view) that keep the subject in
    the safe area and frame each near person whole (``include_margin`` off the sides)
    or keep it ``exclude_clear_deg`` beyond the edge come from the tangent map
    ``x = 0.5 + tan(a) / (2 tan(f / 2))`` (people above or below the view are out at
    any shift); the smallest one is costed ``w_shift * |shift| / (f / 2) + w_widen *
    (f - f0) / (fov_hi - f0)``. The shift cap counts from ``ctx["yaw0"]`` (the yaw
    before the pass) when given, so rounds do not add up. The :data:`INCLUDE_TRIES`
    cheapest are checked with ``chk.bad``; when all fail, the :data:`GRID_TRIES`
    cheapest points of a grid (FOV every 4 deg, shift every 2.5 deg), at the step only.
    When none frames everyone well and the geometric QA counts an edge cut at the step,
    the cheapest of those it does not count one in (costing :data:`STICKY_MISS` more).
    """
    y0, f0, vp = float(ctx["yaw"][i]), float(ctx["fov"][i]), float(ctx["pitch"][i])
    size = ctx["size"]
    fhi = max(f0, float(ctx["fov_hi"]))
    room = fhi - f0
    sj = chk.subject(i)
    boxes = chk.constraints(i)
    floor = min(_height_share(sj[3], f0, size), p.include_min_subject_h) if sj is not None else 0
    clear = p.exclude_clear_deg
    base = ctx.get("yaw0")
    off = 0.0 if base is None else float(G.wrap_deg(base[i] - y0))  # the yaw before the pass

    def cost(f: float, dy: float) -> float:
        widen = p.w_widen * (f - f0) / room if room > 0 else 0.0
        return p.w_shift * abs(dy) / (f / 2) + widen

    fovs = []
    for f in np.arange(f0, fhi + 1e-6, 2.0):
        if sj is not None and _height_share(sj[3], f, size) < floor - 1e-9:
            break  # wider only makes the subject smaller
        fovs.append(float(f))
    cands = []
    for f in fovs:
        half, t = f / 2, math.tan(math.radians(f) / 2)
        cap = p.include_max_shift * half
        feas = [(-math.inf if sign <= 0 else 0.0, math.inf if sign >= 0 else 0.0)]
        feas = _intersect(feas, [(off - cap, off + cap)])
        if sj is not None:
            a = math.degrees(math.atan((1 - 2 * p.safe_margin) * t))
            d = float(G.wrap_deg(sj[0] - y0))
            feas = _intersect(feas, [(d + sj[2] / 2 - a, d - sj[2] / 2 + a)])
        vh = G.flat_fov(f, *size) / 2
        a_in = math.degrees(math.atan((1 - 2 * p.include_margin) * t))
        for bx in boxes:
            if bx[1] - bx[3] / 2 - clear >= vp + vh or bx[1] + bx[3] / 2 + clear <= vp - vh:
                continue  # above or below the view: out at any shift
            d, w2 = float(G.wrap_deg(bx[0] - y0)), bx[2] / 2
            ok = [(-math.inf, d - w2 - clear - half)]
            if w2 <= a_in:
                ok.append((d + w2 - a_in, d - w2 + a_in))
            ok.append((d + w2 + clear + half, math.inf))
            feas = _intersect(feas, ok)
            if not feas:
                break
        if feas:
            dy = min((min(max(0.0, lo), hi) for lo, hi in feas), key=abs)
            cands.append((cost(f, dy), f, dy))
    cands = sorted(cands)[:INCLUDE_TRIES]
    for c, f, dy in cands:
        if not chk.bad(i, f=f, y=y0 + dy):
            return c, dy, f - f0
    grid = []
    for f in fovs[::2]:
        cap = p.include_max_shift * f / 2
        k = int(cap / 2.5 + 1e-9)
        o = off + 2.5 * np.arange(-k, k + 1)
        grid += [(cost(f, float(x)), f, float(x)) for x in o if sign * x >= 0]
    grid.sort()
    tried = {(f, dy) for _, f, dy in cands}
    if cands:  # the tangent map's picks fail on the exact check
        for c, f, o in [g for g in grid if g[1:] not in tried][:GRID_TRIES]:
            if not chk.bad(i, f=f, y=y0 + o, mids=False):
                return c, o, f - f0
    if chk.qa_bad(i):  # no view frames everyone well: one the geometric QA does not cut
        for c, f, o in [*sorted(cands), *grid][:GRID_TRIES]:
            if not chk.qa_bad(i, f=f, y=y0 + o):
                return c + STICKY_MISS, o, f - f0
    return None


def _include_exclude(ctx: dict[str, Any], chk: Any, a: int, b: int, p: DirectorParams) -> bool:
    """Frame each near person of shot ``[a, b)`` whole or keep it out, by widening and / or
    shifting the yaw (:func:`_solve_step`, per bad step, one step runs too); True when a
    change was kept.

    Bad stretches less than ``include_bridge_s`` apart are fixed as one (the steps
    between hold the shift and FOV, interpolated), else each on its own. Within a
    stretch the shift keeps one side: when the cheapest solutions shift both ways, the
    stretch is solved again with each side forced and the cheaper total kept. The
    changes are held and eased 2 steps beyond, smoothed, limited (:data:`SAFE_SPEED` x
    the yaw limits, ``fov_rate``), capped where the smoothed widening would take a
    subject below its ``include_min_subject_h`` floor, and kept only when they reduce
    the bad steps. When the steps' shifts differ, the largest one held over the whole
    stretch is tried first and kept when it leaves no more bad steps than the
    interpolated shifts (each step's least shift chases small box changes: a drifting
    view on calm footage)."""
    yaw, fov, lo = ctx["yaw"], ctx["fov"], ctx["fov_lo"]
    sm = p.smooth_s / p.dt
    gap = p.include_bridge_s / p.dt - 1e-6

    def solve(sign: int, got: dict) -> dict[int, tuple[float, float, float]]:
        """The solutions with shifts to side ``sign`` only (those of ``got`` on that side
        kept; none stays none)."""
        out = {}
        for i, s in got.items():
            if s is not None and np.sign(round(s[1], 6)) * sign < 0:
                s = _solve_step(ctx, chk, i, p, sign)
            if s is not None:
                out[i] = s
        return out

    def fix(r0: int, r1: int) -> bool:
        steps = [i for i in range(r0, r1) if chk.bad(i)]
        got = {i: _solve_step(ctx, chk, i, p) for i in steps}
        sol = {i: s for i, s in got.items() if s is not None}
        if {np.sign(round(s[1], 6)) for s in sol.values()} >= {-1.0, 1.0}:  # sticky
            sides = [solve(sg, got) for sg in (1, -1)]
            sol = min(
                sides,
                key=lambda x: sum(s[0] for s in x.values()) + STICKY_MISS * (len(steps) - len(x)),
            )
        if not sol:
            return False
        idx = np.array(sorted(sol))
        j = np.arange(max(a, idx[0] - 2), min(b, idx[-1] + 3))
        shifts = [sol[k][1] for k in idx]
        dy, df = np.zeros(b - a), np.zeros(b - a)
        dy[j - a] = np.interp(j, idx, shifts)  # held beyond the ends
        df[j - a] = np.interp(j, idx, [sol[k][2] for k in idx])
        base_y, base_f = yaw[a:b].copy(), fov[a:b].copy()
        top = _subject_ceiling(ctx, chk, a, b, p)  # smoothing must not shrink a subject

        def apply(dy: np.ndarray = dy) -> None:
            f = _limit(base_f + _gauss(df, sm / 2), p.dt, p.fov_rate, 2 * p.fov_rate)
            fov[a:b] = np.clip(f, lo, top)
            yaw[a:b] = _limit(
                base_y + _gauss(dy, sm / 2),
                p.dt,
                SAFE_SPEED * p.max_speed,
                SAFE_SPEED * p.max_accel,
            )

        if max(shifts) - min(shifts) > 1e-6:  # one shift held, not chasing each step's least
            apply()
            n_interp = chk.n_bad(a, b + 1)
            yaw[a:b], fov[a:b] = base_y, base_f
            held = np.zeros(b - a)
            held[j - a] = max(shifts, key=abs)
            if _keep_if_better(ctx, chk, a, b, lambda: apply(held)):
                if chk.n_bad(a, b + 1) <= n_interp:
                    return True
                yaw[a:b], fov[a:b] = base_y, base_f
        return _keep_if_better(ctx, chk, a, b, apply)

    kept = False
    groups: list[list[tuple[int, int]]] = []
    for r in chk.runs(a, b, 1):
        if groups and r[0] - groups[-1][-1][1] < gap:
            groups[-1].append(r)
        else:
            groups.append([r])
    for runs in groups:
        if fix(runs[0][0], runs[-1][1]):
            kept = True
        elif len(runs) > 1:  # bridged, it did not help: each stretch on its own
            for r0, r1 in runs:
                kept |= fix(r0, r1)
    return kept


def _relimit(ctx: dict[str, Any], chk: Any, a: int, b: int, p: DirectorParams) -> None:
    """Follow the subject without the dead zone, at :data:`SAFE_SPEED` x the limits."""
    sm = p.smooth_s / p.dt

    def apply() -> None:
        ctx["yaw"][a:b] = _limit(
            _gauss(ctx["raw_yaw"][a:b], sm / 2),
            p.dt,
            SAFE_SPEED * p.max_speed,
            SAFE_SPEED * p.max_accel,
        )

    _keep_if_better(ctx, chk, a, b, apply)


def _apply_away(ctx: dict[str, Any], c0: int, c1: int, ctr: float, p: DirectorParams) -> None:
    """Turn steps ``[c0, c1)`` into a wide cut-away at yaw ``ctr`` (in place)."""
    yaw = ctx["yaw"]
    prev_yaw = yaw[c0 - 1] if c0 > 0 else ctr
    yaw[c0:c1] = prev_yaw + G.wrap_deg(ctr - prev_yaw)
    ctx["pitch"][c0:c1] = p.pitch_default
    ctx["fov"][c0:c1] = ctx["fov_wide"]
    ctx["subj"][c0:c1] = np.nan
    fwd = abs(G.wrap_deg(ctr - ctx["forward_yaw"])) <= 30
    ctx["target"][c0:c1] = ["forward" if fwd else "view"] * (c1 - c0)


def _best_away(
    ctx: dict[str, Any], chk: Any, c0: int, c1: int, cands: list[float], p: DirectorParams
) -> tuple[tuple[int, int], float]:
    """``(bad steps, yaw)`` of the candidate cut-away of ``[c0, c1)`` with the fewest bad
    steps (``chk.n_bad`` of ``[c0, c1]``: step ``c1`` may show it held; the first of
    equals; ``cands`` is non-empty); ``ctx`` is left unchanged. ``include_exclude``:
    among equal QA edge cuts, the fewest steps that show the wearer first."""
    keys = ("yaw", "pitch", "fov", "subj")
    keep = {k: ctx[k][c0:c1].copy() for k in keys}
    keep_target = ctx["target"][c0:c1]
    cuts = ctx.setdefault("cut_after", set())
    new = {c0 - 1, c1 - 1} - cuts  # the cut-away's ends are cuts while it is tried
    cuts |= new
    tried = []
    for ctr in cands:
        _apply_away(ctx, c0, c1, ctr, p)
        n = chk.n_bad(c0, c1 + 1)
        seen = 0 if p.edge_rule == "legacy" else sum(chk.wearer_seen(i) for i in range(c0, c1))
        for k in keys:
            ctx[k][c0:c1] = keep[k]
        ctx["target"][c0:c1] = keep_target
        tried.append(((n[0], seen, n[1]), n, ctr))
        if not any(n) and not seen:
            break
    cuts -= new
    best = min(tried, key=lambda x: x[0])
    return best[1], best[2]


def _cut_away(
    ctx: dict[str, Any],
    chk: Any,
    a: int,
    b: int,
    span: tuple[int, int],
    p: DirectorParams,
) -> list[tuple[int, int]]:
    """Cut each remaining bad stretch of shot ``[a, b)`` to a bin away from the subject
    and the intruders (forward when none): the one with the fewest bad steps, then the
    best score (:func:`_best_away`). A stretch stays when no cut-away has fewer bad
    steps than the shot. Cut-aways and the pieces they leave last ``min_hold_s`` within
    ``span`` (the segment's steps; or reach the shot's ends); a cut-away in the planned
    margins (``edge_rule="include_exclude"``) may be shorter there. Returns the shot's
    pieces.

    ``include_exclude`` also: a cut-away ends ``cut_min_gap_s`` or more from the shot's
    cuts and the previous cut-away (else it reaches them), and is made only when it
    removes ``cut_min_gain_s`` or more, per cut it adds, of the steps the geometric QA
    counts as cut or where the subject is out of view."""
    subj, cen = ctx["subj"], ctx["centres"]
    min_steps = max(1, round(p.min_hold_s / p.dt))
    snap = min_steps
    gain = 0
    if p.edge_rule != "legacy":
        snap = max(min_steps, round(p.cut_min_gap_s / p.dt))
        gain = max(1, round(p.cut_min_gain_s / p.dt))
    pb, done = b, []
    last_c0, last_sy = -1, []  # the previous cut-away (merged when they would touch)
    last = span[1] - 1  # the segment's last step: a piece ending there ends with the segment
    for r0, r1 in reversed(chk.runs(a, b)):
        if r0 >= pb:  # inside the previous cut-away
            continue
        c0, c1 = r0, min(max(r1, r0 + min_steps), pb)
        if c0 > a and chk.held_bad(c0 - 1):  # the frame before the cut shows it held
            c0 -= 1
        if pb - c1 < snap or 0 <= last - c1 < min_steps:
            c1 = pb
        if c0 <= last and min(c1, last) - c0 < min_steps:  # cut short by the shot / segment end
            c0 = max(a, min(c1, last) - min_steps)
        if c0 < span[0] < c1 < span[0] + min_steps:  # entering the segment: last there too
            c1 = min(pb, span[0] + min_steps)
            if pb - c1 < snap or 0 <= last - c1 < min_steps:
                c1 = pb
        if c0 - a < snap or 0 < c0 - span[0] < min_steps:
            c0 = a
        if c1 <= c0:
            continue
        sy = [y for i in range(c0, c1) for y in chk.offenders(i)]
        sy += [float(y) for y in subj[c0:c1, 0] if not np.isnan(y)]
        merged = None
        if c1 >= pb and pb == last_c0:  # back to back with the previous cut-away: one cut
            merged = done.pop(0)
            c1 = merged[1]
            sy += last_sy
        score = ctx["yaw_score"][c0:c1].mean(0)
        away = np.ones(len(cen), bool)
        for y in sy:
            away &= np.abs(G.wrap_deg(cen - y)) > ctx["fov_wide"] / 2 + 10.0
        away &= ctx["people"][c0:c1].max(0) <= 0.3
        cands = [float(cen[k]) for k in np.flatnonzero(away)[np.argsort(-score[away])]]
        best = _best_away(ctx, chk, c0, c1, cands or [float(ctx["forward_yaw"])], p)
        now = chk.n_bad(c0, c1 + 1)
        cuts = (c0 > a) + (c1 < pb and merged is None)  # cuts the cut-away adds
        if best[0] >= now or now[0] - best[0][0] < gain * max(1, cuts):
            if merged is not None:
                done.insert(0, merged)
            continue
        ctr = best[1]
        _apply_away(ctx, c0, c1, ctr, p)
        ctx.setdefault("cut_after", set()).update({c0 - 1, c1 - 1})
        done[:0] = [x for x in ((c0, c1), (c1, pb)) if x[1] > x[0]]
        pb, last_c0, last_sy = c0, c0, sy
        logger.debug(
            "safe-area cut %.1f-%.1f s to yaw %.0f", ctx["times"][c0], ctx["times"][c1 - 1], ctr
        )
    return ([(a, pb)] if pb > a else []) + done


def _suspect_steps(ctx: dict[str, Any], span: tuple[int, int]) -> set[int]:
    """Steps of ``span`` where a near person other than the framed subject reaches into the
    outer fifth of the view's half width (arms, shoulders and jackets at the edge belong
    to them; a person well inside the frame is no suspect)."""
    out = set()
    subj = ctx["subj"]
    for i in range(*span):
        has = ctx["target"][i] == "friend" and not np.isnan(subj[i]).any()
        for box in ctx["near"][i]:
            if has and abs(G.wrap_deg(box[0] - subj[i, 0])) < max(5.0, subj[i, 2] / 2):
                continue  # the subject itself (as in _framing_checks)
            d, half = abs(G.wrap_deg(box[0] - ctx["yaw"][i])), ctx["fov"][i] / 2
            if d - box[2] / 2 < half and d + box[2] / 2 > 0.8 * half:
                out.add(i)
                break
    return out


def _fix_shot(ctx: dict[str, Any], chk: Any, a: int, b: int, p: DirectorParams) -> None:
    """``include_exclude`` fixes of shot ``[a, b)``: :func:`_include_exclude` (up to
    :data:`INCLUDE_ROUNDS` rounds while it helps), and :func:`_widen_only` before
    them when the rounds leave hard bad steps; the one with fewer bad steps
    (``chk.n_bad``, hard ones first) is kept, then
    hard bad steps left are widened (``_widen_only(hard=True)``)."""
    yaw, fov = ctx["yaw"], ctx["fov"]

    def rounds() -> tuple[int, int]:
        for _ in range(INCLUDE_ROUNDS):  # the eased edges of a fix may need another
            if not chk.runs(a, b, 1) or not _include_exclude(ctx, chk, a, b, p):
                break
        return chk.n_bad(a, b + 1)

    base = yaw[a:b].copy(), fov[a:b].copy()
    first = rounds()
    if first[0]:  # B only where A left hard bad steps
        got = yaw[a:b].copy(), fov[a:b].copy()
        yaw[a:b], fov[a:b] = base
        if not _widen_only(ctx, chk, a, b, p) or rounds() >= first:
            yaw[a:b], fov[a:b] = got
    if chk.n_bad(a, b + 1)[0]:  # the rounds may narrow a step back to a hard cut
        _widen_only(ctx, chk, a, b, p, hard=True)


def _safe_area_pass(
    ctx: dict[str, Any],
    shots: list[tuple[int, int]],
    span: tuple[int, int],
    p: DirectorParams,
    fix: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Fix framing that fails on the smoothed path (in place); returns the shots.

    ``ctx``: ``times``, ``yaw``, ``pitch``, ``fov``, ``target``, ``subj``, ``near``
    (:func:`_near_people`), ``yaw_score``, ``people``, ``centres``,
    ``forward_yaw``, ``size``, ``fov_lo``, ``fov_hi``, ``fov_wide``, ``raw_yaw``; with
    ``edge_rule="include_exclude"`` also ``persons`` and ``persons_mid`` (every
    confident person per step and halfway to the next one; else ``near``). Sets
    ``yaw0`` (the yaw before the pass) and ``cut_after`` (the steps a shot or cut-away
    ends at).
    ``span``: the segment's steps (the cut-aways' hold rule, :func:`_cut_away`);
    ``fix``: the steps that get fixed (default ``span``).

    ``edge_rule="legacy"``: a step is bad when the subject of a friend step (projected
    with :func:`geometry.project_box_visible`) leaves the safe area (``safe_margin``
    left / right; top as the QA's edge rule), or, on any non-selfie step, a near
    non-subject person of the QA's edge size touches the left / right edge, or the
    wearer's in-frame box is not allowed (:func:`qa.wearer_allowed`). Shots with bad
    runs longer than :data:`SAFE_MIN_S` inside ``fix`` get, in turn:
    :func:`_widen`, :func:`_reframe` and :func:`_relimit` (each kept only when it
    reduces the bad steps), then :func:`_cut_away` for what is left.

    ``include_exclude`` (default): the subject and wearer rules as above; a near
    person (``near_min_h_deg`` tall, ``near_min_score``, not the wearer or the subject;
    its unclipped projected box) is fine framed whole (``include_margin`` off the left
    and right edges) or out of view (its box widened by ``exclude_clear_deg`` per side
    entirely outside, or behind the camera), and at any size a bad step when split by
    the edge; and a step is bad where the geometric QA would count an edge cut (any
    confident person, its sliver rules included), or the view halfway to a
    neighbouring step is (:func:`_framing_checks`). Bad runs of any length get
    :func:`_include_exclude` (up to :data:`INCLUDE_ROUNDS` rounds while it helps), then
    :func:`_relimit`. Where a stretch longer than :data:`SAFE_MIN_S` with steps the
    geometric QA counts as cut is left, :func:`_include_exclude` runs once more with
    :data:`INCLUDE_MARGIN_RETRY` x ``include_margin`` (people framed whole may come
    closer to the sides), and
    :func:`_cut_away` takes what is left by that rule.
    """
    fix = span if fix is None else fix
    ctx["yaw0"] = ctx["yaw"].copy()  # the include / exclude shift cap counts from here
    ctx["cut_after"] = {b - 1 for _, b in shots[:-1]}
    chk = _framing_checks(ctx, fix, p)
    legacy = p.edge_rule == "legacy"
    first = None if legacy else 1  # runs of any length are fixed, cut-aways stay longer
    out: list[tuple[int, int]] = []
    for a, b in shots:
        if not chk.runs(a, b, first):
            out.append((a, b))
            continue
        if legacy:
            _widen(ctx, chk, a, b, p)
            if chk.runs(a, b):
                _reframe(ctx, chk, a, b, p)
        else:
            _fix_shot(ctx, chk, a, b, p)
        if chk.runs(a, b, first):
            _relimit(ctx, chk, a, b, p)
        shot_chk = chk
        gain = max(1, round(p.cut_min_gain_s / p.dt))
        if not legacy and chk.runs(a, b) and chk.n_bad(a, b)[0] >= gain:  # a cut-away may come
            p0 = replace(p, include_margin=p.include_margin * INCLUDE_MARGIN_RETRY)
            shot_chk = _framing_checks(ctx, fix, p0)
            _include_exclude(ctx, shot_chk, a, b, p0)
        out += _cut_away(ctx, shot_chk, a, b, span, p)
    return out


def planned_span(
    clip: Any, start: float, end: float, margin: float = PATH_MARGIN_S, dt: float = 0.5
) -> tuple[float, float]:
    """``[start, end]`` widened by ``margin`` s on both sides, within the clip: the range a
    path of that segment is planned over (its ``planned`` span). Cut at the clip's
    start, it starts a whole number of ``dt`` steps before ``start``, so the keys fall
    on ``start + k dt`` (the geometric QA's sample times)."""
    if start >= margin:
        t0 = start - margin
    else:
        t0 = start - dt * math.floor(max(start, 0.0) / dt + 1e-9)
    t1 = min(clip.duration, end + margin) if clip.duration else end + margin
    return t0, t1


def compute_path(
    clip: Any,
    start: float,
    end: float,
    *,
    vertical: bool = False,
    kind: str | None = None,
    params: DirectorParams | None = None,
    overrides: dict[str, Any] | None = None,
    orient: G.Orientation | None = None,
    pv: Any = None,
    margin: float = PATH_MARGIN_S,
    suspects: bool = False,
    grid: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Director path for ``[start, end]`` of a clip (keys cover ``margin`` s more each side).

    Args:
        clip: :class:`cutpoint.pano.x3.X3Clip`.
        start, end: Segment (clip-source seconds).
        vertical: 9:16 output (narrower FOVs).
        kind: The moment's dominant feature (``laughter``, ``people``, ...).
        params: Base settings (``kind`` tweaks are applied on top).
        overrides: User settings (``--param`` / ``pano.director``), applied after
            the ``kind`` tweaks so they win (:func:`with_params`).
        orient, pv: Orientation track and view layer (loaded when None).
        margin: Extra seconds planned on both sides, so ``cutpoint extend`` padding
            and the speech-aware pass stay on a planned path (:func:`planned_span`;
            with ``edge_rule="include_exclude"`` its framing is fixed there too, the
            path's ``fixed`` span; ``legacy`` fixes the segment only).
        suspects: Add ``vl_suspect``: segment times worth a vision-model look
            (near non-subject people reaching the view's outer edge, plus cuts and
            cut-away destinations; subject framing only, see :mod:`cutpoint.pano.vlcheck`).
        grid: If given, filled with the :func:`score_grid` of the planned range and the
            Viterbi ``states`` (before shot merging), for diagnostics; the path is the same.
    """
    from cutpoint.pano.views import load_views  # noqa: PLC0415
    from cutpoint.pano.x3 import imu_source  # noqa: PLC0415

    p = params_for(kind, params)
    if overrides:
        p = with_params(p, overrides)
    src = Path(clip.source)
    if orient is None:
        orient = G.orientation(src, imu_source(src), clip.fps, max(clip.frames, 1))
    if pv is None:
        pv = load_views(src)
    t0, t1 = planned_span(clip, start, end, margin, p.dt)
    from cutpoint.pano.views import wearer_axis  # noqa: PLC0415

    axis = wearer_axis(pv) if pv is not None else G.WEARER_V
    sg = score_grid(clip, t0, t1, p, pv, orient=orient, vertical=vertical)
    nb = p.n_bins
    # upright: ties break as in the views' -z frame (its bin k is bin k + n_bins / 2 here)
    states = _viterbi(sg, p, roll=nb // 2 if orient.forward_axis == G.axis_name(G.LENS_V) else 0)
    if grid is not None:
        grid.update(sg, states=states.copy())
    times = sg["times"]
    subject_mode = p.framing == "subject"
    if subject_mode:  # shots measured within the segment; gated selfie steps stay out
        shots = _shots(states, sg["centres"], nb, p, times, (start, end), sg["selfie"])
    else:
        shots = _shots(states, sg["centres"], nb, p)
    yaw = np.zeros(len(times))
    pitch = np.full(len(times), p.pitch_default)
    fov = np.zeros(len(times))
    target = ["forward"] * len(times)
    sm = p.smooth_s / p.dt
    fw, ff, fs = (
        (p.vertical_fov_wide, p.vertical_fov_friend, p.vertical_fov_selfie)
        if vertical
        else (p.fov_wide, p.fov_friend, p.fov_selfie)
    )
    size = (1080, 1920) if vertical else (1920, 1080)
    lo, hi = (p.vertical_fov_min, p.vertical_fov_max) if vertical else (p.fov_min, p.fov_max)
    subj = np.full((len(times), 4), np.nan)
    raw_yaw = np.zeros(len(times))
    if subject_mode:
        subj = _track_subject(pv, sg.get("subject"), times, states, shots)
    for a, b in shots:
        st = states[a:b]
        if st[0] == nb:  # selfie shot: follow the wearer (fixed in the camera)
            wy, wp = [], []
            for t in times[a:b]:
                d = orient.r_vw(t).T @ axis
                y_, p_ = G.ray_yaw_pitch(d)
                wy.append(float(y_))
                wp.append(float(p_))
            y = np.degrees(np.unwrap(np.radians(wy)))
            yaw[a:b] = _limit(_gauss(y, sm), p.dt, p.max_speed, p.max_accel)
            pitch[a:b] = _gauss(np.clip(np.asarray(wp) + p.selfie_pitch_up, -70.0, 0.0), sm)
            fov[a:b] = fs
            for i in range(a, b):
                target[i] = "selfie"
            continue
        y = np.degrees(np.unwrap(np.radians(sg["centres"][st])))
        pp = np.array([sg["person_pitch"][a + i, s] for i, s in enumerate(st)])
        pe = np.array([sg["people"][a + i, s] for i, s in enumerate(st)])
        if subject_mode:
            yaw[a:b], pitch[a:b], fov[a:b] = _refine_shot(
                y, subj[a:b], pe, pp, p, (ff, fw, lo, hi), size
            )
            has = ~np.isnan(subj[a:b]).any(1)
            raw_yaw[a:b] = y + G.wrap_deg(np.where(has, subj[a:b, 0], y) - y)
        else:
            yaw[a:b] = _limit(_gauss(y, sm), p.dt, p.max_speed, p.max_accel)
            has = (pp > -900) & (pe > 0.15)
            raw_pitch = np.where(has, np.clip(pp, p.pitch_min, p.pitch_max), p.pitch_default)
            pitch[a:b] = _gauss(raw_pitch, 2 * sm)
            fov[a:b] = _gauss(np.where(pe > 0.3, ff, fw), sm)
        for i in range(a, b):
            target[i] = (
                "friend"
                if pe[i - a] > 0.3
                else "forward"
                if abs(G.wrap_deg(sg["centres"][st[i - a]] - sg["forward_yaw"])) <= 30
                else "view"
            )
    legacy_edges = p.edge_rule == "legacy"
    fixed = (t0, t1)
    if subject_mode and pv is not None:
        ctx = {
            "times": times,
            "yaw": yaw,
            "pitch": pitch,
            "fov": fov,
            "target": target,
            "subj": subj,
            "near": _near_people(pv, times, p.near_min_h_deg),
            "yaw_score": sg["yaw_score"],
            "people": sg["people"],
            "centres": sg["centres"],
            "forward_yaw": sg["forward_yaw"],
            "size": size,
            "fov_hi": hi,
            "fov_lo": lo,
            "fov_wide": fw,
            "raw_yaw": raw_yaw,
        }
        span = (int(np.searchsorted(times, start - 1e-6)), int(np.searchsorted(times, end + 1e-6)))
        if legacy_edges:  # the version 4 rule fixes the segment only
            fixed = (start, end)
        else:  # every confident person, at the steps and halfway between them
            from cutpoint.pano.qa import QAThresholds  # noqa: PLC0415

            floor = min(p.near_min_score, QAThresholds().person_min_score)
            rows = _near_people(pv, np.r_[times, times[:-1] + p.dt / 2], 0.0, floor)
            ctx["persons"], ctx["persons_mid"] = rows[: len(times)], rows[len(times) :]
        fix = span if legacy_edges else (0, len(times))
        shots = _safe_area_pass(ctx, shots, span, p, fix)
        sus = _suspect_steps(ctx, span) if suspects else set()  # on the path as rendered
        sus |= {a for a, _ in shots if span[0] < a < span[1]}  # cut / cut-away destinations
    keys = []
    cut_steps = {a for a, _ in shots if a > 0}
    for i, t in enumerate(times):
        keys.append(
            {
                "t": round(float(t), 3),
                "yaw": round(float(yaw[i]), 2),
                "pitch": round(float(pitch[i]), 2),
                "fov": round(float(fov[i]), 1),
                "target": target[i],
                **({"cut": True} if i in cut_steps else {}),
                **(
                    {
                        "subj": [
                            round(float(subj[i, 0]) % 360.0, 1),
                            *np.round(subj[i, 1:], 1).tolist(),
                        ]
                    }
                    if subject_mode and not np.isnan(subj[i]).any()
                    else {}
                ),
            }
        )
    summary = {}
    for tg in ("friend", "selfie", "forward", "view"):
        summary[tg] = round(sum(1 for k in keys if k["target"] == tg) * p.dt, 1)
    path = {
        "version": PATH_VERSION,
        "clip": clip.clip_id,
        "source": str(src),
        "segment": {"start": round(start, 3), "end": round(end, 3)},
        "planned": {"start": round(t0, 3), "end": round(t1, 3)},
        # the span the safe-area pass fixed: a reused path must cover the range with it
        "fixed": {"start": round(fixed[0], 3), "end": round(fixed[1], 3)},
        "vertical": vertical,
        "kind": kind,
        "forward_yaw": round(float(sg["forward_yaw"]), 1),
        "wearer_axis": [round(float(v), 3) for v in axis],
        "frame": "display: horizon-levelled from the IMU; yaw 0 = smoothed camera "
        "forward, + = right; pitch + = up (degrees)",
        "params": asdict(p),
        "cuts": [round(float(times[a]), 3) for a, _ in shots[1:]],
        "seconds_by_target": summary,
        "keys": keys,
    }
    if orient.forward_axis != G.axis_name(G.FORWARD_V):  # upright clip: its yaw 0 axis
        from cutpoint.pano.views import DET_FRAME  # noqa: PLC0415

        path["forward_axis"] = orient.forward_axis
        path["det_frame"] = DET_FRAME  # planned on views sampled in that frame (cli._plan_key)
    if suspects and subject_mode and pv is not None:
        path["vl_suspect"] = [round(float(times[i]), 2) for i in sorted(sus)]
    if pv is not None and len(pv.t):
        from cutpoint.pano.qa_paths import path_quality  # noqa: PLC0415

        path["quality"] = path_quality(path, pv, start, end)
    return path


def _hermite_tangents(kx: np.ndarray, ky: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(h, d, m)`` of keys ``ky`` (one column per channel): interval lengths, secant
    slopes and Fritsch-Butland tangents. Interior tangents are the weighted harmonic mean
    of the two neighbouring secants (so ``|m|`` never exceeds the larger one; 0 at a turn
    or a flat side); the ends use the one-sided secant."""
    h = np.diff(kx)
    # two keys at one time (hand-edited path): a step there, as np.interp does
    d = np.divide(
        np.diff(ky, axis=0), h[:, None], out=np.zeros((len(h), ky.shape[1])), where=h[:, None] > 0
    )
    m = np.empty_like(ky)
    m[0], m[-1] = d[0], d[-1]
    d0, d1 = d[:-1], d[1:]
    w1, w2 = (2 * h[1:] + h[:-1])[:, None], (h[1:] + 2 * h[:-1])[:, None]
    same = d0 * d1 > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        m[1:-1] = np.where(same, (w1 + w2) / (w1 / d0 + w2 / d1), 0.0)
    return h, d, m


def _hermite_basis(s: np.ndarray) -> tuple[np.ndarray, ...]:
    """Cubic Hermite basis ``(h00, h10, h01, h11)`` at ``s`` in [0, 1]."""
    s2, s3 = s * s, s * s * s
    return 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s, -2 * s3 + 3 * s2, s3 - s2


def _pchip(x: np.ndarray, kx: np.ndarray, ky: Any, linear: np.ndarray | None = None) -> np.ndarray:
    """Monotone cubic Hermite (Fritsch-Butland tangents) of ``(kx, ky)`` at ``x``, held
    outside ``kx``; ``ky`` is one channel or one column per channel.

    C1 (continuous velocity) and shape-preserving: every value lies between its two
    keys, so it never overshoots the keys (or any clamp they respect). Key intervals
    flagged in ``linear`` are interpolated linearly instead (velocity steps at their
    keys). Fewer than 3 keys: ``np.interp``.
    """
    ky = np.asarray(ky, float)
    if len(kx) < 3:
        if ky.ndim == 1:
            return np.interp(x, kx, ky)
        return np.stack([np.interp(x, kx, c) for c in ky.T], 1)
    one = ky.ndim == 1
    ky = ky[:, None] if one else ky
    h, _, m = _hermite_tangents(kx, ky)
    xc = np.clip(x, kx[0], kx[-1])
    i = np.clip(np.searchsorted(kx, xc, side="right") - 1, 0, len(kx) - 2)
    s = np.divide(xc - kx[i], h[i], out=np.ones(len(xc)), where=h[i] > 0)[:, None]
    b00, b10, b01, b11 = _hermite_basis(s)
    y0, y1, hi = ky[i], ky[i + 1], h[i][:, None]
    out = b00 * y0 + b10 * hi * m[i] + b01 * y1 + b11 * hi * m[i + 1]
    if linear is not None:
        out = np.where(linear[i][:, None], y0 + s * (y1 - y0), out)
    # rounding only: keep each value between its keys exactly
    out = np.clip(out, np.minimum(y0, y1), np.maximum(y0, y1))
    out = np.where((x < kx[0])[:, None], ky[0], out)  # held at the first key, as np.interp does
    return out[:, 0] if one else out


_PEAK_S = np.linspace(0.0, 1.0, 65)  # samples per key interval for the cubic's peak speed


def _linear_intervals(kx: np.ndarray, yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray | None:
    """Per key interval of one shot: whether to interpolate it linearly rather than
    cubic (``None`` below 3 keys).

    Pan speed is measured as :func:`cutpoint.pano.qa.pan_metrics` does: yaw rate scaled
    by cos pitch, combined with the pitch rate. An interval goes linear when its cubic
    speed peaks above the linear speed of that interval and of both its neighbours (so
    the path never pans faster than with linear keys; between two stops any C1 curve
    peaks at 1.5x the linear speed), or above a QA pan limit (``max_pan_p95``,
    ``max_pan_max``) that its linear speed does not exceed throughout (so in any window
    the cubic is over a limit only where linear is too, and never fails a pan gate
    linear passes).
    """
    if len(kx) < 3:
        return None
    from cutpoint.pano.qa import QAThresholds  # noqa: PLC0415

    h, d, m = _hermite_tangents(kx, np.stack([yaw, pitch], 1))
    s = _PEAK_S[None, :]
    r00, r10, r01 = 6 * (s - s * s), 3 * s * s - 4 * s + 1, 3 * s * s - 2 * s

    def rate(c: int) -> np.ndarray:  # d/dt of the cubic, per interval x sample
        return r00 * d[:, c, None] + r10 * m[:-1, c, None] + r01 * m[1:, c, None]

    b00, b10, b01, b11 = _hermite_basis(s)
    hk = h[:, None]
    pit = b00 * pitch[:-1, None] + b10 * hk * m[:-1, 1, None]
    pit = pit + b01 * pitch[1:, None] + b11 * hk * m[1:, 1, None]
    cubic = np.hypot(rate(0) * np.cos(np.radians(pit)), rate(1)).max(1)
    # linear: constant rates; the yaw rate counts most at the pitch nearest the horizon
    p0, p1 = pitch[:-1], pitch[1:]
    near = np.where(p0 * p1 <= 0, 0.0, np.minimum(np.abs(p0), np.abs(p1)))
    lin = np.hypot(d[:, 0] * np.cos(np.radians(near)), d[:, 1])
    ref = lin.copy()
    ref[1:] = np.maximum(ref[1:], lin[:-1])
    ref[:-1] = np.maximum(ref[:-1], lin[1:])
    out = cubic > ref * (1 + 1e-9) + 1e-9
    # slowest linear speed: at the pitch farthest from the horizon. Kept cubic over a
    # limit only where every linear sample fails it too (qa rounds to 0.1 deg/s)
    slow = np.hypot(d[:, 0] * np.cos(np.radians(np.maximum(np.abs(p0), np.abs(p1)))), d[:, 1])
    th = QAThresholds()
    for limit in (th.max_pan_p95, th.max_pan_max):
        out |= (cubic > limit) & (slow < limit + 0.05)
    return out


def shot_index(path: dict[str, Any], times: np.ndarray) -> np.ndarray:
    """Shot index per time: the number of keys marked ``cut`` at or before it."""
    kt = np.array([k["t"] for k in path["keys"]], float)
    cut_t = kt[[i for i, k in enumerate(path["keys"]) if k.get("cut")]]
    return np.searchsorted(cut_t, times, side="right") if len(cut_t) else np.zeros(len(times), int)


def evaluate(path: dict[str, Any], times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame ``(yaw, pitch, fov)`` from a path's keys (monotone cubic inside shots,
    see :func:`_pchip`, linear on key intervals where the cubic would pan faster than
    linear nearby or over a QA pan limit, see :func:`_linear_intervals`; linear in a
    shot of 2 keys; held outside the planned range, hard jumps at keys marked ``cut``).
    Yaw is interpolated unwrapped (shortest arc between consecutive keys) and returned
    unwrapped, continuous within a shot."""
    keys = path["keys"]
    kt = np.array([k["t"] for k in keys], float)
    cut_idx = [i for i, k in enumerate(keys) if k.get("cut")]
    seg_bounds = [0, *cut_idx, len(keys)]
    yaw = np.zeros(len(times))
    pitch = np.zeros(len(times))
    fov = np.zeros(len(times))
    seg_of = shot_index(path, times)
    for s in range(len(seg_bounds) - 1):
        a, b = seg_bounds[s], seg_bounds[s + 1]
        sel = seg_of == s
        if not sel.any():
            continue
        ty = np.degrees(np.unwrap(np.radians([k["yaw"] for k in keys[a:b]])))
        tp = np.array([k["pitch"] for k in keys[a:b]], float)
        ky = np.stack([ty, tp, [k["fov"] for k in keys[a:b]]], 1)
        out = _pchip(times[sel], kt[a:b], ky, _linear_intervals(kt[a:b], ty, tp))
        yaw[sel], pitch[sel], fov[sel] = out.T
    return yaw, pitch, fov


def describe_quality(path: dict[str, Any]) -> str:
    """One-line summary of the path's ``quality`` block (empty without one)."""
    q = path.get("quality")
    if not q:
        return ""

    def pct(v: float | None) -> str:
        return "-" if v is None else f"{100 * v:.0f}%"

    return (
        f"quality {q['score']:.0f} {'ok' if q['pass'] else 'FAIL'}: occl {pct(q['occluded'])} "
        f"close {pct(q['too_close'])} edge {pct(q['edge_cut'])} subject "
        f"{pct(q['subject_present'])} size {pct(q['size_ok'])} safe {pct(q['safe_area'])}"
        + (f"; {'; '.join(q['fails'])}" if q["fails"] else "")
    )


def describe(path: dict[str, Any]) -> str:
    """One-line summary for logs."""
    s = path.get("seconds_by_target", {})
    return (
        f"{len(path.get('cuts', []))} cut(s); "
        + ", ".join(f"{k} {v:.0f}s" for k, v in s.items() if v)
        if s
        else ""
    )
