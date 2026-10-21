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
degrees; anything faster than ``max_speed`` (45 deg/s by default), and every
change to or from the selfie state, is a **hard cut** costing ``cut_cost``.
Cuts closer than ``min_hold_s`` are merged away. Within each shot the yaw is
unwrapped and smoothed (Gaussian ``smooth_s``), then speed- and
acceleration-limited; pitch follows the framed person (clamped) and is
smoothed too; FOV depends on the target. With ``framing="subject"`` (default)
friend shots follow the person tracked across view frames (dead zone,
headroom) with a FOV fitted to its size, and a safe-area pass widens, speeds
up or cuts away where the subject or a big neighbour would still be cut by the
frame edge; keys then carry ``subj`` (yaw, pitch, width, height; path version 2).

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

PATH_VERSION = 4  # 2: keys may carry "subj" (the framed person); 3: close people at the
# frame edge are intruders; 4: wider framing, the wearer allowed in frame (same keys;
# evaluate reads all versions)
MIN_REUSE_VERSION = 4  # older paths (tighter framing) are re-directed, not reused


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
)
CHOICE_PARAMS = {
    "selfie_mode": ("auto", "off"),
    "people_size": ("framing", "close"),
    "framing": ("subject", "bin"),
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
            number, a non-positive :data:`POSITIVE_PARAMS` value or a
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
    if not len(pv_t):
        return np.zeros((len(times), *values.shape[1:]))
    idx = np.clip(np.searchsorted(pv_t, times), 1, max(1, len(pv_t) - 1))
    left = np.clip(idx - 1, 0, len(pv_t) - 1)
    pick = np.where(np.abs(pv_t[left] - times) <= np.abs(pv_t[idx] - times), left, idx)
    out = values[pick].astype(float)
    far = np.abs(pv_t[pick] - times) > hold
    out[far] = 0.0
    return out


def _gauss(x: np.ndarray, sigma_steps: float) -> np.ndarray:
    if len(x) < 3 or sigma_steps <= 0:
        return x.copy()
    r = int(3 * sigma_steps)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_steps) ** 2)
    k /= k.sum()
    pad = np.pad(x, r, mode="edge")
    return np.convolve(pad, k, mode="valid")


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
    dyaw = (centres - fy + 180.0) % 360.0 - 180.0
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


def _viterbi(sg: dict[str, Any], p: DirectorParams, start_state: int | None = None) -> np.ndarray:
    """Best state per step (``n_bins`` = selfie)."""
    ys, sf = sg["yaw_score"], sg["selfie"]
    n, nb = ys.shape
    s = np.concatenate([ys, sf[:, None]], 1)
    ns = nb + 1
    c = sg["centres"]
    d = np.abs((c[:, None] - c[None, :] + 180.0) % 360.0 - 180.0)
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
        d = abs((centres[a] - centres[b] + 180.0) % 360.0 - 180.0)
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
NEAR_MIN_H_DEG = 15.0  # other people this tall may intrude at the frame edge
CLOSE_MIN_H_DEG = 35.0  # other people this tall are close: any part at the edge shows
CLOSE_MIN_SHARE = 0.06  # ... once this share of the frame (close_edge_share) is inside
SAFE_MIN_S = 0.5  # the safe-area pass fixes longer violations only
SAFE_SPEED = 1.3  # its re-limit allows max_speed / max_accel times this
REFRAME_SHARE = 0.35  # its reframing shifts the yaw by up to this share of half the view


def _circ(d: np.ndarray | float) -> np.ndarray:
    return (np.asarray(d, float) + 180.0) % 360.0 - 180.0


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
                d = np.abs(_circ(cands[:, 0] - pts[-1][1]))
                off = np.abs(_circ(cands[:, 0] - centres[states[i]]))
                ok = np.flatnonzero((d < LINK_DEG) & (off <= 45.0))
                if len(ok):
                    pick = cands[ok[d[ok].argmin()]].copy()
            if pick is None and not np.isnan(row[states[i]]).any():
                pick = row[states[i]].copy()
            if pick is None:
                continue
            if pts:  # unwrap against the previous point
                pick[0] = pts[-1][1] + float(_circ(pick[0] - pts[-1][1]))
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
    target = bins_yaw + _circ(sy - bins_yaw)
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


def _near_people(pv: Any, times: np.ndarray) -> list[np.ndarray]:
    """Per step: ``(k, 5)`` yaw, pitch, w, h, is_self (1: the wearer) of confident people
    at least :data:`NEAR_MIN_H_DEG` tall in the nearest view frame (the geometric QA's rows).

    The wearer is judged by its in-frame box (:func:`_framing_checks`)."""
    from cutpoint.pano import qa_paths as QP  # noqa: PLC0415
    from cutpoint.pano.qa import QAThresholds  # noqa: PLC0415

    empty = np.zeros((0, 5))
    if pv is None or not len(pv.det):
        return [empty] * len(times)
    d = pv.det
    person, is_self = QP.person_rows(pv)
    keep = person & (d[:, 2] >= QAThresholds().person_min_score)
    keep &= d[:, 6] >= NEAR_MIN_H_DEG
    rows = np.column_stack([d[:, 3:7].astype(float), is_self.astype(float)])
    fi = d[:, 0].astype(int)
    idx = QP.view_frame_index(pv.t.astype(float), times)
    return [rows[keep & (fi == j)] if j >= 0 else empty for j in idx]


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
    d = float(_circ(box[0] - view_yaw))
    lo, hi = d - box[2] / 2, d + box[2] / 2
    half = hfov / 2
    if not (lo < -half < hi or lo < half < hi):
        return 0.0
    w = max(0.0, min(hi, half) - max(lo, -half)) / hfov
    top, bottom = box[1] + box[3] / 2, box[1] - box[3] / 2
    h = max(0.0, min(top, view_pitch + vfov / 2) - max(bottom, view_pitch - vfov / 2)) / vfov
    return float(w * h)


def _framing_checks(ctx: dict[str, Any], span: tuple[int, int], p: DirectorParams) -> Any:
    """``offenders(i, f, y)``, ``bad(i, f, y)``, ``runs(a, b)`` and ``n_bad(a, b)`` on
    the current ``ctx`` path (see :func:`_safe_area_pass`)."""
    from cutpoint.pano.qa import QAThresholds, wearer_allowed  # noqa: PLC0415

    th = QAThresholds()
    yaw, pitch, fov = ctx["yaw"], ctx["pitch"], ctx["fov"]
    subj, near, size = ctx["subj"], ctx["near"], ctx["size"]
    m = p.safe_margin
    min_run = int(SAFE_MIN_S / p.dt) + 1

    def offenders(i: int, f: float | None = None, y: float | None = None) -> list[float]:
        """Yaws of what is badly framed at step ``i`` (empty: fine)."""
        if ctx["target"][i] == "selfie":
            return []
        vy, vp, vf = yaw[i] if y is None else y, pitch[i], fov[i] if f is None else f
        has = ctx["target"][i] == "friend" and not np.isnan(subj[i]).any()
        if has:
            res = G.project_box_visible(vy, vp, vf, size[0], size[1], *subj[i])
            if res is None:
                return [float(subj[i, 0])]
            x1, y1, x2, y2 = res[0]
            if x1 < m or x2 > 1 - m or (y1 < th.safe_top and y2 < th.edge_top_max_y2):
                return [float(subj[i, 0])]
        out = []
        for row in near[i]:
            box = row[:4]
            if len(row) > 4 and row[4]:  # the wearer: by its whole in-frame box, slivers too
                r = G.project_box_visible(vy, vp, vf, size[0], size[1], *box)
                if r is not None and not wearer_allowed(np.clip(r[0], 0.0, 1.0), th):
                    out.append(float(box[0]))
                continue
            if has and abs(_circ(box[0] - subj[i, 0])) < max(5.0, subj[i, 2] / 2):
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

    def bad(i: int, f: float | None = None, y: float | None = None) -> bool:
        return bool(offenders(i, f, y))

    def flags(a: int, b: int) -> tuple[int, list[bool]]:
        lo, hi = max(a, span[0]), min(b, span[1])
        return lo, [bad(i) for i in range(lo, hi)]

    def runs(a: int, b: int) -> list[tuple[int, int]]:
        """Bad stretches longer than :data:`SAFE_MIN_S` of ``[a, b)`` inside ``span``."""
        lo, fl = flags(a, b)
        out, i = [], 0
        while i < len(fl):
            j = i
            while j < len(fl) and fl[j]:
                j += 1
            if j - i >= min_run:
                out.append((lo + i, lo + j))
            i = max(j, i + 1)
        return out

    def n_bad(a: int, b: int) -> int:
        return sum(flags(a, b)[1])

    return SimpleNamespace(offenders=offenders, bad=bad, runs=runs, n_bad=n_bad)


def _keep_if_better(ctx: dict[str, Any], chk: Any, a: int, b: int, apply: Any) -> bool:
    """Run ``apply()`` (edits yaw / fov of ``[a, b)``); revert unless the bad steps drop."""
    before = chk.n_bad(a, b)
    keep_y, keep_f = ctx["yaw"][a:b].copy(), ctx["fov"][a:b].copy()
    apply()
    if chk.n_bad(a, b) < before:
        return True
    ctx["yaw"][a:b], ctx["fov"][a:b] = keep_y, keep_f
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
    yaw[c0:c1] = prev_yaw + _circ(ctr - prev_yaw)
    ctx["pitch"][c0:c1] = p.pitch_default
    ctx["fov"][c0:c1] = ctx["fov_wide"]
    ctx["subj"][c0:c1] = np.nan
    fwd = abs(_circ(ctr - ctx["forward_yaw"])) <= 30
    ctx["target"][c0:c1] = ["forward" if fwd else "view"] * (c1 - c0)


def _best_away(
    ctx: dict[str, Any], chk: Any, c0: int, c1: int, cands: list[float], p: DirectorParams
) -> tuple[int, float]:
    """``(bad steps, yaw)`` of the candidate cut-away of ``[c0, c1)`` with the fewest bad
    steps (the first of equals; ``cands`` is non-empty); ``ctx`` is left unchanged."""
    keys = ("yaw", "pitch", "fov", "subj")
    keep = {k: ctx[k][c0:c1].copy() for k in keys}
    keep_target = ctx["target"][c0:c1]
    best = (math.inf, cands[0])
    for ctr in cands:
        _apply_away(ctx, c0, c1, ctr, p)
        n = chk.n_bad(c0, c1)
        for k in keys:
            ctx[k][c0:c1] = keep[k]
        ctx["target"][c0:c1] = keep_target
        if n < best[0]:
            best = (n, ctr)
        if n == 0:
            break
    return int(best[0]), best[1]


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
    ``span`` (or reach the shot's ends). Returns the shot's pieces."""
    subj, cen = ctx["subj"], ctx["centres"]
    min_steps = max(1, round(p.min_hold_s / p.dt))
    pb, done = b, []
    last_c0, last_sy = -1, []  # the previous cut-away (merged when they would touch)
    last = span[1] - 1  # the segment's last step: a piece ending there ends with the segment
    for r0, r1 in reversed(chk.runs(a, b)):
        if r0 >= pb:  # inside the previous cut-away
            continue
        c0, c1 = r0, min(max(r1, r0 + min_steps), pb)
        if pb - c1 < min_steps or 0 <= last - c1 < min_steps:
            c1 = pb
        if min(c1, last) - c0 < min_steps:  # cut short by the shot / segment end
            c0 = max(a, min(c1, last) - min_steps)
        if c0 - a < min_steps or 0 < c0 - span[0] < min_steps:
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
            away &= np.abs(_circ(cen - y)) > ctx["fov_wide"] / 2 + 10.0
        away &= ctx["people"][c0:c1].max(0) <= 0.3
        cands = [float(cen[k]) for k in np.flatnonzero(away)[np.argsort(-score[away])]]
        best = _best_away(ctx, chk, c0, c1, cands or [float(ctx["forward_yaw"])], p)
        if best[0] >= chk.n_bad(c0, c1):
            if merged is not None:
                done.insert(0, merged)
            continue
        ctr = best[1]
        _apply_away(ctx, c0, c1, ctr, p)
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
            if has and abs(_circ(box[0] - subj[i, 0])) < max(5.0, subj[i, 2] / 2):
                continue  # the subject itself (as in _framing_checks)
            d, half = abs(_circ(box[0] - ctx["yaw"][i])), ctx["fov"][i] / 2
            if d - box[2] / 2 < half and d + box[2] / 2 > 0.8 * half:
                out.add(i)
                break
    return out


def _safe_area_pass(
    ctx: dict[str, Any],
    shots: list[tuple[int, int]],
    span: tuple[int, int],
    p: DirectorParams,
) -> list[tuple[int, int]]:
    """Fix framing that fails on the smoothed path (in place); returns the shots.

    ``ctx``: ``times``, ``yaw``, ``pitch``, ``fov``, ``target``, ``subj``, ``near``
    (:func:`_near_people`), ``yaw_score``, ``people``, ``centres``,
    ``forward_yaw``, ``size``, ``fov_lo``, ``fov_hi``, ``fov_wide``, ``raw_yaw``.
    A step is bad when the subject of a friend step (projected with
    :func:`geometry.project_box_visible`) leaves the safe area (``safe_margin``
    left / right; top as the QA's edge rule), or, on any non-selfie step, a near
    non-subject person of the QA's edge size touches the left / right edge, or the
    wearer's in-frame box is not allowed (:func:`qa.wearer_allowed`). Shots with bad
    runs longer than :data:`SAFE_MIN_S` inside ``span`` (steps) get, in turn:
    :func:`_widen`, :func:`_reframe` and :func:`_relimit` (each kept only when it
    reduces the bad steps), then :func:`_cut_away` for what is left.
    """
    chk = _framing_checks(ctx, span, p)
    out: list[tuple[int, int]] = []
    for a, b in shots:
        if not chk.runs(a, b):
            out.append((a, b))
            continue
        _widen(ctx, chk, a, b, p)
        if chk.runs(a, b):
            _reframe(ctx, chk, a, b, p)
        if chk.runs(a, b):
            _relimit(ctx, chk, a, b, p)
        out += _cut_away(ctx, chk, a, b, span, p)
    return out


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
    margin: float = 10.0,
    suspects: bool = False,
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
            and the speech-aware pass stay on a planned path.
        suspects: Add ``vl_suspect``: segment times worth a vision-model look
            (near non-subject people reaching the view's outer edge, plus cuts and
            cut-away destinations; subject framing only, see :mod:`cutpoint.pano.vlcheck`).
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
    t0 = max(0.0, start - margin)
    t1 = min(clip.duration, end + margin) if clip.duration else end + margin
    from cutpoint.pano.views import wearer_axis  # noqa: PLC0415

    axis = wearer_axis(pv) if pv is not None else G.WEARER_V
    sg = score_grid(clip, t0, t1, p, pv, orient=orient, vertical=vertical)
    nb = p.n_bins
    states = _viterbi(sg, p)
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
            raw_yaw[a:b] = y + _circ(np.where(has, subj[a:b, 0], y) - y)
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
                if abs(((sg["centres"][st[i - a]] - sg["forward_yaw"] + 180) % 360) - 180) <= 30
                else "view"
            )
    if subject_mode and pv is not None:
        ctx = {
            "times": times,
            "yaw": yaw,
            "pitch": pitch,
            "fov": fov,
            "target": target,
            "subj": subj,
            "near": _near_people(pv, times),
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
        shots = _safe_area_pass(ctx, shots, span, p)
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
    if suspects and subject_mode and pv is not None:
        path["vl_suspect"] = [round(float(times[i]), 2) for i in sorted(sus)]
    if pv is not None and len(pv.t):
        from cutpoint.pano.qa_paths import path_quality  # noqa: PLC0415

        path["quality"] = path_quality(path, pv, start, end)
    return path


def evaluate(path: dict[str, Any], times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame ``(yaw, pitch, fov)`` from a path's keys (linear inside shots, held
    outside the planned range, hard jumps at keys marked ``cut``)."""
    keys = path["keys"]
    kt = np.array([k["t"] for k in keys], float)
    cut_idx = [i for i, k in enumerate(keys) if k.get("cut")]
    seg_bounds = [0, *cut_idx, len(keys)]
    yaw = np.zeros(len(times))
    pitch = np.zeros(len(times))
    fov = np.zeros(len(times))
    seg_of = (
        np.searchsorted(kt[cut_idx], times, side="right") if cut_idx else np.zeros(len(times), int)
    )
    for s in range(len(seg_bounds) - 1):
        a, b = seg_bounds[s], seg_bounds[s + 1]
        sel = seg_of == s
        if not sel.any():
            continue
        ty = np.degrees(np.unwrap(np.radians([k["yaw"] for k in keys[a:b]])))
        yaw[sel] = np.interp(times[sel], kt[a:b], ty)
        pitch[sel] = np.interp(times[sel], kt[a:b], [k["pitch"] for k in keys[a:b]])
        fov[sel] = np.interp(times[sel], kt[a:b], [k["fov"] for k in keys[a:b]])
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
