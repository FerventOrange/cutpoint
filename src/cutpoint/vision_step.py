"""The vision step of :func:`cutpoint.analyze.rank_parts` (#909): the default edit and the band
below its cut rated by the vision model, centred per prompt kind and blended into their scores.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cutpoint import config, default_fit, jsonout, progress, scorer
from cutpoint.config import remap_source
from cutpoint.events.scoring import heard_laughter, rescore
from cutpoint.styles import ACTION_CAM, Style
from cutpoint.utils import footage_context
from cutpoint.vision import PRIOR_N, RATING_PRIOR, VisionRater, speech_excerpt
from cutpoint.vision_record import vision_blend

if TYPE_CHECKING:
    from cutpoint.activity import Prompts
    from cutpoint.analyze import GridPart

logger = logging.getLogger(__name__)

# the vision band (#909): the default edit plus k below its cut, k = this share of the edit's
# moments, at least VISION_BAND_MIN and at most VISION_BAND_MAX
VISION_BAND_SHARE, VISION_BAND_MIN, VISION_BAND_MAX = 0.3, 2, 24


def prior_group(base: str) -> str:
    """The style group of the vision rating prior (#948) for a style whose built-in base is
    ``base``: ``"action"`` for :data:`~cutpoint.analyze.ACTION_STYLES`, ``"other"`` for every other
    style."""
    from cutpoint.analyze import ACTION_STYLES  # noqa: PLC0415 - analyze imports this module

    return "action" if base in ACTION_STYLES else "other"


def rating_centres(ratings: Iterable[tuple[str, float]], group: str) -> dict[str, float]:
    """The centre of each prompt kind's ratings (#909): ``ratings`` are ``(kind, rating)``
    pairs, the kind ``"speech"`` or ``"frames"``. A kind's centre is the mean of its ratings,
    shrunk toward the ``group``'s (:func:`prior_group`) prior in
    :data:`cutpoint.vision.RATING_PRIOR` while it has fewer than
    :data:`cutpoint.vision.PRIOR_N`: ``(sum + max(0, PRIOR_N - n) * prior) / max(n, PRIOR_N)``.
    Only the kinds rated are in the result."""
    sums: dict[str, list[float]] = {}
    for kind, r in ratings:
        sums.setdefault(kind, []).append(r)
    return {
        kind: (sum(rs) + max(0, PRIOR_N - len(rs)) * RATING_PRIOR[group, kind])
        / max(len(rs), PRIOR_N)
        for kind, rs in sums.items()
    }


def centred_rating(rating: float, centre: float) -> float:
    """``rating`` moved so its kind's ``centre`` (:func:`rating_centres`) is 5, within 0-10."""
    return max(0.0, min(10.0, 5 + rating - centre))


def band_k(suggested_top: int) -> int:
    """How many moments the vision band takes below the default edit's cut (#909):
    :data:`VISION_BAND_SHARE` of the edit's ``suggested_top`` moments, rounded up, within
    :data:`VISION_BAND_MIN` and :data:`VISION_BAND_MAX`."""
    k = math.ceil(round(VISION_BAND_SHARE * suggested_top, 9))  # no float error can add one
    return min(VISION_BAND_MAX, max(VISION_BAND_MIN, k))


def vision_band(
    ranked: list[dict[str, Any]], picks: list[dict[str, Any]], max_n: int | None
) -> tuple[list[dict[str, Any]], int, int]:
    """The moments the vision model rates (#909): the default edit and ``k`` below its cut.

    ``ranked``: the candidates that are neither ``must_include`` nor visual pool moments, in
    :func:`cutpoint.scorer.edit_rank_key` order. ``picks``: the default edit
    (:func:`cutpoint.default_fit.fit_target` before vision). The cut is 1 + the index in
    ``ranked`` of the last pick that is not ``must_include`` (the picks need not be the first
    ones; 0 without one). The band is every pick in ``ranked`` (a moment above the cut that the
    edit skipped for its length or because it overlaps a pick is not one) plus ``ranked[cut : cut + k]``, ``k`` =
    :func:`band_k` of the number of picks, clipped at the end. Closest to the cut first: the
    last pick, the first moment below the cut, the pick before the last, the second below, and
    so on, the rest of one side once the other runs out; then cut to ``max_n``
    (``--vision-top``; None: all). Returns ``(band, cut, k)``."""
    where = {id(c): i for i, c in enumerate(ranked)}
    at = sorted(
        {where[id(c)] for c in picks if not c.get("must_include") and id(c) in where},
        reverse=True,
    )
    cut = at[0] + 1 if at else 0
    k = band_k(len(picks))
    above = [ranked[i] for i in at]
    below = ranked[cut : cut + k]
    order: list[dict[str, Any]] = []
    for i in range(max(len(above), len(below))):
        order.extend(side[i] for side in (above, below) if i < len(side))
    return (order if max_n is None else order[:max_n]), cut, k


def vision_budget(
    share: float, pre_vision_s: float, earlier_total_s: float | None
) -> tuple[float, str]:
    """The vision step's time budget (#909): ``share`` (``vision_time_share``) of the analysis
    before it. A re-rank measures against the first analysis (``earlier_total_s``, its
    ``timings.total_s``), as its own run before vision is mostly cached. Returns
    ``(budget_s, the name of the base)``."""
    if earlier_total_s is not None:
        return share * earlier_total_s, "earlier_total_s"
    return share * pre_vision_s, "pre_vision_s"


def cap_reached(now: float, deadline: float, calls: int) -> bool:
    """Whether the vision time cap stops further model calls: at least one call was made and
    the clock ``now`` is at or past the ``deadline``."""
    return calls >= 1 and now >= deadline


@dataclass
class VisionOutcome:
    """What :func:`_rate_vision` did: ``used`` any rating applied; ``failed`` moments without a
    rating (cache misses of a ``cache_only`` rater included); ``capped`` the time cap stopped
    the model calls and ``skipped_by_cap`` moments were left unrated for it; ``centres`` per
    prompt kind (:func:`rating_centres`); ``by_kind`` the moments rated per prompt kind."""

    used: bool = False
    failed: int = 0
    capped: bool = False
    skipped_by_cap: int = 0
    centres: dict[str, float] = field(default_factory=dict)
    by_kind: dict[str, int] = field(default_factory=dict)


def _ask_rater(
    method: Callable[..., Any], c: dict[str, Any], style: Style, activity: Prompts | None
) -> Any:
    """``VisionRater.rate`` or ``cached`` (``method``) for the candidate ``c``."""
    return method(
        remap_source(Path(c["source"])),
        c["start"],
        c["end"],
        style.base,
        activity=activity,
        transcript=c.get("transcript"),
        laughter=heard_laughter(c),
    )


def _rate_vision(
    order: list[dict[str, Any]],
    rater: VisionRater,
    style: Style,
    activity: Prompts | None = None,
    batch_size: int = 0,
    *,
    deadline: float,
    clock: Callable[[], float] = time.perf_counter,
) -> VisionOutcome:
    """Rate the candidates of ``order`` (the vision band, closest to the cut first,
    :func:`vision_band`) with the vision model and apply the ratings to their analysis score
    (``score_pre_events`` once events rescored them).

    The model is asked in ``order`` until the time cap: once a call was made and ``clock()``
    reached ``deadline`` (:func:`cap_reached`), a moment without a cached rating is skipped; a
    cached one is still taken (it costs nothing). A call under way when the deadline passes
    finishes, so the step can overrun its budget by one call.

    Then each prompt kind's ratings (``"speech"`` when the moment has speech to send with its
    frames, :func:`~cutpoint.vision.speech_excerpt`, else ``"frames"``) are centred on their
    mean (:func:`rating_centres` with the prior of the style's :func:`prior_group`,
    :func:`centred_rating`) and blended in with
    :func:`~cutpoint.vision_record.vision_blend` and the style's ``vision_weight``.
    ``signals.vision`` keeps the model's rating as given; its label goes to ``vision_label``,
    never to ``label`` (#913).
    Each rated candidate keeps ``vision_rated`` (the span rated, the score before and after the
    blend, the prompt kind and its centre) for :func:`~cutpoint.vision_record.sync_vision`.
    ``activity`` words the action-cam prompt (the project's activity
    prompt set). ``batch_size``: ``--batch-size`` (#713), the model is unloaded after the
    candidates of every this many sources (:func:`_vision_batches`); the ratings are the same,
    but once the time cap is reached, which moments near the cut get rated can differ."""
    out = VisionOutcome()

    def cached(c: dict[str, Any]) -> bool:
        return _ask_rater(rater.cached, c, style, activity) is not None

    if not rater.cache_only:  # a resumed run: the ratings made before are cached
        progress.continuing("vision", sum(cached(c) for c in order), len(order))
    got: list[tuple[dict[str, Any], float, str | None]] = []
    done = calls = 0
    try:
        for k, part in enumerate(_vision_batches(order, batch_size)):
            if k:  # a batch boundary: memory back until the next request loads the model
                rater.unload()
                progress.continuing("vision", done, len(order))
            for c in part:
                progress.update(progress.fraction(done, len(order)))
                done += 1
                if cap_reached(clock(), deadline, calls) and not cached(c):
                    out.skipped_by_cap += 1
                    continue
                hits = rater.hits
                res = _ask_rater(rater.rate, c, style, activity)
                if rater.hits == hits and not rater.cache_only:
                    calls += 1
                if res is None:
                    out.failed += 1
                    continue
                got.append((c, res.score, res.label))
                logger.info(
                    "vision %.1f '%s' for %.1f-%.1f", res.score, res.label, c["start"], c["end"]
                )
    finally:  # free the VRAM now, not after Ollama's default 5 min keep_alive
        rater.unload()
    out.capped = out.skipped_by_cap > 0
    logger.info("Vision: %d ratings from the cache, %d asked the model", rater.hits, rater.asked)
    # each prompt kind centred on this run's ratings of it
    kinds = [
        "speech" if speech_excerpt(c.get("transcript"), style.base) else "frames" for c, _, _ in got
    ]
    out.centres = rating_centres(
        ((kind, r) for kind, (_, r, _) in zip(kinds, got, strict=True)), prior_group(style.base)
    )
    for kind, (c, score, label) in zip(kinds, got, strict=True):
        out.used = True
        out.by_kind[kind] = out.by_kind.get(kind, 0) + 1
        c["signals"]["vision"] = score
        if label:
            c["vision_label"] = label
        key = "score_pre_events" if "score_pre_events" in c else "score"
        centre = out.centres[kind]
        base = c[key]
        c[key] = vision_blend(base, centred_rating(score, centre), style.vision_weight)
        # for sync_vision: the rating holds for this span only
        c["vision_rated"] = {
            "start": c["start"],
            "end": c["end"],
            "base": base,
            "blended": c[key],
            "kind": kind,
            "centre": round(centre, 2),
        }
        c.pop("vision_stale", None)
    return out


def _vision_stats(
    rater: VisionRater,
    *,
    rated: int,
    vision_s: float,
    pre_vision_s: float,
    rerank: bool = False,
) -> dict[str, Any]:
    """The vision step's cost for ``settings.vision`` (#909): the ratings made and taken from
    the cache, the step's seconds against the analysis before it, and the run's request
    timings and tokens summed (:attr:`VisionRater.stats`). Logs the one INFO line. ``rerank``:
    the run re-ranks an earlier analysis, so ``pre_vision_s`` is only the cached re-run and no
    share is logged."""
    st = rater.stats
    if rerank:
        logger.info("Vision: %d rated (%d cached) in %.0f s", rated, rater.hits, vision_s)
    else:
        share = 100 * vision_s / pre_vision_s if pre_vision_s > 0 else 0.0
        logger.info(
            "Vision: %d rated (%d cached) in %.0f s (%.0f%% of analyze before vision)",
            rated,
            rater.hits,
            vision_s,
            share,
        )
    return {
        **({"rerank": True} if rerank else {}),
        "rated": rated,
        "cached": rater.hits,
        "calls": int(st["calls"]),
        "vision_s": round(vision_s, 2),
        "pre_vision_s": round(pre_vision_s, 2),
        "grab_s": round(st["grab_s"], 2),
        "load_s": round(st["load_s"], 2),
        "prompt_s": round(st["prompt_s"], 2),
        "reply_s": round(st["reply_s"], 2),
        "prompt_tokens": int(st["prompt_tokens"]),
        "reply_tokens": int(st["reply_tokens"]),
        "model": rater.model,
        "provider": rater.provider_name,
    }


def _vision_batches(top: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    """The candidates to rate in batches for ``--batch-size`` ``size``: those of ``size``
    sources each (sources in the order of their first candidate in ``top``, each one's
    candidates in band order, closest to the cut first). One batch, ``top`` in band order, for
    ``size`` 0 or ``size`` sources or fewer."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in top:
        groups.setdefault(c["source"], []).append(c)
    if size <= 0 or len(groups) <= size:
        return [top]
    parts = list(groups.values())
    return [[c for g in parts[i : i + size] for c in g] for i in range(0, len(parts), size)]


def _vision_step(
    hl: dict[str, Any],
    ev: Any,
    grids: list[GridPart],
    style: Style,
    project: Path,
    steps: progress.Run,
    *,
    t_start: float,
    vision: bool | None,
    vision_top: int | None,
    ollama_url: str | None,
    ollama_model: str | None,
    batch_size: int,
    vision_time_share: float | None,
    earlier_total_s: float | None,
    vision_cache_only: bool,
) -> None:
    """:func:`~cutpoint.analyze.rank_parts`'s vision step, in place: the rating of the default edit
    and the band below its cut (:func:`vision_band`), blended in and clamped; ``hl["timings"]`` gets
    ``vision_s`` and ``hl["settings"]`` the ``vision`` record."""
    timings = hl["timings"]
    cands = hl["candidates"]
    rater: VisionRater | None = None
    bad_url: ValueError | None = None
    if vision and cands:
        try:
            rater = VisionRater(
                **{k: v for k, v in {"url": ollama_url, "model": ollama_model}.items() if v}
            )
        except ValueError as err:  # a bad port in OLLAMA_HOST / ollama_url
            bad_url = err
    if not vision:
        steps.skip("vision", "vision is off (Settings, config.json or --vision)")
    elif not cands:
        steps.skip("vision", "no candidates to rate")
    elif rater is None:
        message = f"vision model not usable ({bad_url}); the vision step is skipped"
        logger.warning("%s", message)
        jsonout.warn("vision_unavailable", message)
        steps.skip("vision", message)
    elif rater.consent_needed:
        # a provider the user did not agree to (a hosted one, or an Ollama on another
        # machine): no request, no cached ratings either
        steps.skip("vision", rater.warn_consent())
    else:
        from cutpoint.visual_pool import is_pool  # noqa: PLC0415

        # the default edit and the band below its cut, from the ranking before vision
        ranked = sorted(
            (c for c in cands if not c.get("must_include") and not is_pool(c)),
            key=scorer.edit_rank_key,
        )
        picks = default_fit.fit_target(
            sorted((c for c in cands if not is_pool(c)), key=scorer.edit_rank_key),
            style.target_total,
        )
        order, cut, k = vision_band(ranked, picks, vision_top)
        in_edit = {id(c) for c in picks}
        # the moments above the cut the edit skipped (too long or overlapping a pick): no
        # one's neighbours in the clamp, as long as they keep this span
        for c in ranked[:cut]:
            if id(c) not in in_edit:
                c["vision_skipped"] = [c["start"], c["end"]]
        # the user's --context words (#1068) go into the prompts, so into the cache keys. An
        # analysis records the words it rated with as settings.vision_context (null: rated
        # without words); a re-rank keeps them, so its cached ratings are found after `crashes`
        # or `profile --context` changed settings.context (a file without the key, from before
        # it: settings.context)
        settings = hl["settings"]
        recorded = settings.get("context")
        if vision_cache_only and "vision_context" in settings:
            recorded = settings["vision_context"]
        rater.context = footage_context(recorded)
        if not vision_cache_only:
            settings["vision_context"] = rater.context
        # action-cam: worded for the project's activity.json (the default without one)
        wording = None
        if style.base == ACTION_CAM:
            from cutpoint import activity  # noqa: PLC0415

            wording = activity.project_prompts(project)
        # available() first: it resolves a hosted provider's unset model, the cache key.
        # Offline, that model stays the placeholder and its cached ratings are not found.
        online, n_cached = (not vision_cache_only and rater.available()), 0
        if vision_cache_only and rater.via is not None:
            from cutpoint.assistant.providers import AUTO_SERVER_MODEL  # noqa: PLC0415

            # a re-rank asks no model, but a hosted server's unset model is the cache key: its
            # check resolves it (a model list, no frames)
            if rater.model == AUTO_SERVER_MODEL and rater.via.available():
                rater.model = rater.via.model
        if not online:  # Ollama down: the ratings already cached, no requests (#413)
            rater.cache_only = True
            n_cached = sum(_ask_rater(rater.cached, c, style, wording) is not None for c in order)
        if online or n_cached:
            if not online and not vision_cache_only:
                n_none = len(order) - n_cached
                message = (
                    "vision rating requested but Ollama/model unavailable; using "
                    f"{n_cached} cached rating{'s' if n_cached != 1 else ''}, "
                    f"{n_none} candidate{'s' if n_none != 1 else ''} had no cached rating"
                )
                logger.warning("%s", message)
                jsonout.warn("vision_unavailable", message)
            with steps.step("vision"):
                t0 = time.perf_counter()
                pre_vision_s = t0 - t_start
                share = (
                    config.vision_time_share() if vision_time_share is None else vision_time_share
                )
                budget_s, budget_base = vision_budget(share, pre_vision_s, earlier_total_s)
                n_edit = sum(id(c) in in_edit for c in order)
                logger.info(
                    "Vision: rating %d moments (%d of the default edit, %d below the cut at "
                    "%d) of %d candidates",
                    len(order),
                    n_edit,
                    len(order) - n_edit,
                    cut,
                    len(cands),
                )
                outcome = _rate_vision(
                    order, rater, style, wording, batch_size, deadline=t0 + budget_s
                )
                if ev:  # the blended analysis scores plus the event boosts again
                    hl["settings"]["vision"] = {}  # rescore_events records the clamp here
                    rescore(hl, ev, project, grids)
                    n_clamped = hl["settings"]["vision"].get("clamped", 0)
                else:  # a rated moment stays between its unrated neighbours
                    n_clamped = scorer.apply_vision_clamp(
                        cands,
                        [c["vision_rated"]["base"] if "vision_rated" in c else None for c in cands],
                    )
                timings["vision_s"] = round(time.perf_counter() - t0, 2)
                if outcome.capped:
                    logger.info(
                        "Vision: time cap reached after %d calls (%.0f s, budget %.0f s = "
                        "%d%% of %.0f s); %d moments near the cut not rated",
                        int(rater.stats["calls"]),
                        timings["vision_s"],
                        budget_s,
                        round(100 * share),
                        pre_vision_s if earlier_total_s is None else earlier_total_s,
                        outcome.skipped_by_cap,
                    )
                hl["settings"]["vision_used"] = outcome.used
                hl["settings"]["vision"] = {
                    "requested": len(order),
                    "failed": outcome.failed - rater.missed,
                    **({} if online else {"uncached": rater.missed}),
                    **_vision_stats(
                        rater,
                        rated=len(order) - outcome.failed - outcome.skipped_by_cap,
                        vision_s=timings["vision_s"],
                        pre_vision_s=pre_vision_s,
                        rerank=earlier_total_s is not None,
                    ),
                    "cut": cut,
                    "band_k": k,
                    "rated_by_kind": outcome.by_kind,
                    "centre": {
                        kind: (round(outcome.centres[kind], 2) if kind in outcome.centres else None)
                        for kind in ("speech", "frames")
                    },
                    "share": share,
                    "budget_s": round(budget_s, 2),
                    "budget_base": budget_base,
                    "capped": outcome.capped,
                    "skipped_by_cap": outcome.skipped_by_cap,
                    "clamped": n_clamped,
                }
        elif vision_cache_only:
            logger.info("Vision: no cached rating for these moments; they stay unrated")
            steps.skip("vision", "no cached ratings (re-rank)")
        else:
            message = "vision rating requested but Ollama/model unavailable; continuing without it"
            logger.warning("%s", message)
            jsonout.warn("vision_unavailable", message)
            steps.skip("vision", "Ollama or its vision model is not reachable")
