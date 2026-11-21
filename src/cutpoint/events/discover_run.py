"""HUD region discovery of one :func:`cutpoint.events.pipeline.run_events` OCR layer: per game
where needed, with the HUD fit check of first-guess keys and the vision HUD map top-up
(:class:`DiscoverRun`, run by :func:`cutpoint.events.pipeline._discover`).

Its lines are logged under the pipeline's logger, as before the move (#1085 A161).
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cutpoint import cache, jsonout, memguard, progress
from cutpoint import paths as cutpoint_paths
from cutpoint.errors import LowMemoryError
from cutpoint.events import discover, game_fit
from cutpoint.events import hud_map as vision_map
from cutpoint.events import regions as hud
from cutpoint.events.killfeed import OcrParams
from cutpoint.events.pipeline import (
    DISCOVERY_PCT,
    USER_PRESET_KEYS,
    _mark_ocr_failure,
    check_game_key,
    logger,
)


class DiscoverRun:
    """One discovery run over ``sources`` (the arguments of
    :func:`cutpoint.events.pipeline._discover`, which documents it).

    ``load_preset`` and ``game_key_for_path`` are the pipeline's (tests patch them there).
    :meth:`run` returns ``out``: ``{game: record}``.
    """

    def __init__(
        self,
        project: Path,
        sources: list[Any],
        ocr: OcrParams,
        force: bool | None,
        bursts: int,
        scope: str,
        hint: str,
        labeler: Callable[..., dict[str, Any]] | None,
        hud_mapper: Any,
        before_top_up: Callable[[], None] | None,
        *,
        load_preset: Callable[..., Any],
        game_key_for_path: Callable[..., str],
    ) -> None:
        self.project = project
        self.sources = sources
        self.ocr = ocr
        self.force = force
        self.scope = scope
        self.hint = hint
        self.labeler = labeler
        self.hud_mapper = hud_mapper
        self.before_top_up = before_top_up
        self.load_preset = load_preset
        self.game_key_for_path = game_key_for_path
        self.params = discover.DiscoverParams(bursts=bursts)
        self.by_game: dict[str, list[Any]] = {}
        self.guesses: dict[str, str] = {}  # path -> first guess, for the sources the check decides
        self.before: dict[
            str, str
        ] = {}  # path -> the key recorded before (stale, or checked again)
        self.decided: dict[str, tuple[str, Any]] = {}  # path -> (key, Fit or None)
        self.started = 0  # discoveries started
        self.total = 1  # discoveries planned (a misfit adds one)
        self.shown = 0.0  # the discovery part's pct never goes back
        self.out: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        """Group the sources per game, then discover and check each game in turn."""
        self._group()
        planned = sum(self.force or self.load_preset(g, self.project) is None for g in self.by_game)
        self.total = max(1, planned)
        for game, srcs in self.by_game.items():
            existing = self.load_preset(game, self.project)
            tested = [s for s in srcs if str(s.path) in self.guesses]
            if existing is None:
                self._new_game(game, srcs, tested)
            else:
                self._known_game(game, srcs, tested, existing)
        return self.out

    # ---- grouping

    def _group(self) -> None:
        """Fill ``by_game`` (and ``guesses``, ``before`` for the sources the check decides)."""
        named = bool(self.ocr.game and self.ocr.game != "auto")
        for s in self.sources:
            path = str(s.path)
            key = self.game_key_for_path(path, self.ocr.game, self.project)
            if not named and hud.preset_for_path(path, None, self.project) is None:
                entry = hud.decided_entry(path, self.project)
                if entry is None or self.force:
                    key = self.guesses[path] = str(entry["guess"]) if entry else key
                    raw = hud.load_game_keys(self.project).get(hud.source_id(path)) or {}
                    if isinstance(raw.get("key"), str):
                        self.before[path] = raw["key"]
            try:
                check_game_key(key)
            except jsonout.ArgumentError:
                if named:
                    raise
                got = self._unplain_key(key, path)
                if got is None:
                    continue
                key = got
            self.by_game.setdefault(key, []).append(s)

    def _unplain_key(self, key: str, path: str) -> str | None:
        """A preset's ``game`` that is no plain name ("Rainbow Six: Siege"): no path is made
        from it. A preset that loads is still read with (OCR finds it by that key) but is not
        refitted, topped up or discovered again (None); else the source's first guess."""
        preset = next(
            (
                p
                for p in hud.preset_files(self.project)
                if str((hud._read_json(p) or {}).get("game")) == key
            ),
            None,
        )
        loads = self.load_preset(key, self.project) is not None
        guess = hud.first_guess(path, self.project)
        logger.warning(
            "HUD preset %s names its game %r, which is no plain name (no /, \\, :, '..'); %s",
            preset or "(unknown file)",
            key,
            f"{path} is read with it but it is not refitted or discovered again"
            if loads
            else f"using {guess!r} for {path}",
        )
        return None if loads else guess

    # ---- per game

    def _new_game(self, game: str, srcs: list[Any], tested: list[Any]) -> None:
        """A game without a preset: discovered; first-guess sources are checked against each
        other, and the preset is installed only after the check when there are several (an
        interrupted run leaves no preset of two games mixed)."""
        split = len(srcs) > 1 and bool(tested)
        proposal = self._discover(game, srcs, None, install=not split)
        if not tested:
            return
        if len(srcs) == 1:  # nothing to compare with
            self.decided[str(tested[0].path)] = (game, None)
            self._persist()
            return
        got = self._fit_frames(game, proposal, srcs, tested)
        fits, _ = self._split(srcs, got)
        misfits = self._decide(game, tested, fits)
        # several sources, some tested: split. The preset from the sources that fit only, or
        # as discovered
        if misfits:
            keep = {discover.src_key(s) for s in srcs if s not in misfits}
            proposal = self._discover(game, srcs, None, keep=keep, install=False)
        self._install_preset(game, proposal)
        self._persist()
        self._place_misfits(game, misfits, fits, got)

    def _known_game(self, game: str, srcs: list[Any], tested: list[Any], existing: Any) -> None:
        """A game with a preset: discovered again only with ``force``; topped up with the vision
        HUD map once; first-guess sources are checked against it."""
        proposal = self._discover(game, srcs, existing) if self.force else None
        self._maybe_top_up(game, srcs, existing, proposal)
        # a forced discovery over an existing preset: the sources are checked against its grid
        # fit_ref, or, without one, against each other like a new discovery; the preset gets
        # the fit_ref of the sources that fit (its regions stay)
        if not tested:
            if proposal is not None:  # every source is the game (--game, a match hint): all of
                # them are the group
                ref = (proposal.get("discovery") or {}).get("fit_ref")
                self._add_fit_ref(game, existing.path, ref)
            return
        got = self._fit_frames(game, proposal, srcs, tested)
        if proposal is not None:
            fits = self._refit(game, srcs, tested, existing, got)
        else:  # against the preset found
            ref = self._fit_ref(existing)  # none, or an earlier version's: no decision, no split
            fits = {
                str(s.path): game_fit.against_ref(
                    game_fit.profile(got.get(str(s.path), []), self.params.min_score), ref
                )
                for s in tested
            }
        misfits = self._decide(game, tested, fits)
        self._persist()
        self._place_misfits(game, misfits, fits, got)

    def _refit(
        self,
        game: str,
        srcs: list[Any],
        tested: list[Any],
        existing: Any,
        got: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """A forced discovery over ``existing``: the fits of ``tested``, and the preset's new
        ``fit_ref`` from the sources that fit."""
        old_ref = self._fit_ref(existing)
        if game_fit.valid_ref(old_ref) and bool(old_ref["grid"]):
            # the preset's grid decides; those that fit make the new one
            profs = self._profiles(srcs, got)
            fits = {str(s.path): game_fit.against_ref(profs[str(s.path)], old_ref) for s in tested}
            if any(f is not None and f.ok for f in fits.values()):  # else the old ref stays
                group = game_fit.Profile()
                for k, prof in profs.items():
                    fit = fits.get(k)
                    if fit is None or fit.ok:  # decided before, inconclusive, or fits
                        group += prof
                self._add_fit_ref(game, existing.path, group.ref())
            return fits
        # the sources against each other; the leader's group: the decided sources and those
        # that fit
        fits, profs = self._split(srcs, got)
        group = game_fit.Profile()
        for k, prof in profs.items():
            fit = fits.get(k)
            if k not in fits or (fit is not None and fit.ok):
                group += prof
        self._add_fit_ref(game, existing.path, group.ref())
        return fits

    def _profiles(self, srcs: list[Any], got: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return {
            str(s.path): game_fit.profile(got.get(str(s.path), []), self.params.min_score)
            for s in srcs
        }

    def _split(
        self, srcs: list[Any], got: dict[str, list[dict[str, Any]]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """The sources of one discovery checked against each other: (fits, profiles)."""
        profs = self._profiles(srcs, got)
        fits = game_fit.split_group(
            profs,
            {str(s.path): float(s.duration) for s in srcs},
            frozenset(p for p in profs if p not in self.guesses),
        )
        return fits, profs

    def _decide(self, game: str, tested: list[Any], fits: dict[str, Any]) -> list[Any]:
        """Record ``game`` for the tested sources that fit (or are inconclusive); returns
        the misfits."""
        misfits = []
        for s in tested:
            fit = fits.get(str(s.path))
            if fit is None:
                logger.debug("%s: HUD fit check for %r inconclusive; keeping it", s.name, game)
            if fit is not None and not fit.ok:
                misfits.append(s)
            else:
                self.decided[str(s.path)] = (game, fit)
        return misfits

    def _place_misfits(
        self,
        game: str,
        misfits: list[Any],
        fits: dict[str, Any],
        got: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Give each misfit the preset discovered for an earlier misfit it fits, else a key
        and a discovery of its own."""
        # presets made for earlier misfits: (key, that misfit's profile, its name)
        made: list[tuple[str, Any, str]] = []
        for s in misfits:
            path, fit = str(s.path), fits[str(s.path)]
            prof = game_fit.profile(got.get(path, []), self.params.min_score)
            for key, other_prof, other in made:
                again = game_fit.against_run(prof, other_prof)
                if again is not None and again.ok:
                    logger.info("%s fits the HUD preset '%s' discovered for %s", s.name, key, other)
                    self.decided[path] = (key, again)
                    self._persist()
                    break
            else:
                new = self._misfit_key(s, path)
                logger.info(
                    "%s does not fit the HUD preset '%s' (score %.2f); discovering it as '%s'",
                    s.name,
                    game,
                    fit.score,
                    new,
                )
                self.decided[path] = (new, fit)
                self._persist()  # before its discovery: an interrupted one keeps the key
                own = self.load_preset(new, self.project)
                if own is None or self.force:
                    prop = self._discover(new, [s], own)
                    if own is not None:  # one source: it is the group
                        ref = (prop.get("discovery") or {}).get("fit_ref")
                        self._add_fit_ref(new, own.path, ref)
                made.append((new, prof, s.name))

    def _misfit_key(self, s: Any, path: str) -> str:
        """A key of its own for the misfit ``s``: the key it had before (also a stale decision)
        or whose preset was discovered from it is its own, not another game's."""
        taken = set(self.by_game) | {k for k, _ in self.decided.values()}
        mine = self.before.get(path)
        return hud.misfit_key(
            path,
            lambda k: (
                k != mine
                and (
                    k in taken
                    or (self.load_preset(k, self.project) is not None and not self._made_from(k, s))
                )
            ),
        )

    # ---- discovery, top-up and presets

    def _discover(
        self,
        game: str,
        srcs: list[Any],
        existing: Any,
        keep: set[str] | None = None,
        install: bool = True,
        topping: bool = False,
    ) -> dict[str, Any]:
        """Discovery for ``game`` on ``srcs`` and its install, as a record in ``out``;
        ``keep``: cluster again from these sources' records only (no OCR, no "No HUD
        preset matches" line);
        ``install`` False: no preset written yet (:meth:`_install_preset` does it once the
        fit check has split off other games); ``topping``: for the HUD map top-up of the
        existing preset only (no feed labels: the preset's regions stay)."""
        gi = self.started
        if keep is None:
            self.started += 1
            self.total = max(self.total, self.started)
        name = "this game" if game == "game" else game
        first = "" if existing is not None else f"first run for {name}: "
        if topping:
            progress.update(detail="mapping the HUD with the vision model")
            logger.info(
                "The HUD preset for %r has no vision HUD map yet; making one from %s "
                "(once per game)",
                game,
                srcs[0].path,
            )
        elif existing is None and keep is None:
            progress.update(detail=f"{first}discovering HUD regions")
            logger.info(
                "No HUD preset matches %s; discovering regions as %r (a few minutes). "
                "Put a preset in %s %s",
                srcs[0].path,
                game,
                cutpoint_paths.presets_dir(),
                self.hint,
            )
        t0 = time.perf_counter()
        try:
            proposal = discover.discover_regions(
                srcs,
                self.project / "discovery" / game,
                game,
                self.params,
                on_frame=functools.partial(self._on_frame, gi, first) if keep is None else None,
                **self._covered(existing),
                **({"labeler": self.labeler} if self.labeler and not topping else {}),
                **({"keep_src_keys": keep} if keep is not None else {}),
                **({"hud_mapper": self.hud_mapper} if self.hud_mapper is not None else {}),
            )
        except LowMemoryError:
            raise
        except Exception as err:  # the games before it have their presets
            memguard.raise_if_out_of_memory(err, "HUD region discovery", game)
            _mark_ocr_failure(err, self.sources, discovery=True, games=self.total)
            raise
        rec = self._record(game, proposal, t0, keep is not None)
        if existing is None:
            if install:
                self._install_preset(game, proposal)
        else:
            rec["preset"] = str(existing.path)
        if existing is not None and not topping:
            self._suggest(game, rec, proposal)
        return proposal

    def _on_frame(self, gi: int, first: str, done: int, total: int) -> None:
        """The first DISCOVERY_PCT of the step, an equal part per discovery (``gi``: its
        index)."""
        share = (gi + progress.fraction(done, total) / 100) / self.total
        self.shown = max(self.shown, DISCOVERY_PCT * share)
        progress.update(self.shown, f"{first}discovering HUD: {done}/{total} frames")

    @staticmethod
    def _covered(existing: Any) -> dict[str, Any]:
        """An existing preset's regions (and big-text check) are the covered ones: extra notice
        regions outside them are only suggested, the preset's regions stay as they are; the
        outside regions it already has count against the extra-notice cap."""
        if existing is None:
            return {}
        data = cache.load_json(existing.path) if existing.path else None
        regs = data.get("regions") if isinstance(data, dict) else None
        outside = sum(isinstance(r, dict) and r.get("origin") == "outside" for r in regs or [])
        return {"covered": existing.all_regions(), "outside": outside}

    def _record(
        self, game: str, proposal: dict[str, Any], t0: float, again: bool
    ) -> dict[str, Any]:
        """The record of a discovery of ``game`` in ``out`` (``again``: a re-cluster, whose
        time adds to the first's)."""
        folder = self.project / "discovery" / game
        rec: dict[str, Any] = {
            "proposal": str(folder / "proposal.json"),
            "overlay": str(folder / "overlay.jpg"),
            "regions": proposal["regions"],
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "installed": False,
        }
        if again and game in self.out:
            rec["elapsed_s"] = round(self.out[game]["elapsed_s"] + rec["elapsed_s"], 1)
        self.out[game] = rec
        return rec

    @staticmethod
    def _suggest(game: str, rec: dict[str, Any], proposal: dict[str, Any]) -> None:
        """The INFO lines of a discovery over a kept preset, and its suggested regions."""
        rec["suggested_regions"] = list(proposal.get("suggested_regions") or [])
        logger.info(
            "Kept existing HUD preset for %r (%s); discovery proposal: %s",
            game,
            rec["preset"],
            rec["proposal"],
        )
        if rec["suggested_regions"]:
            logger.info(
                "Discovery suggests %d extra region(s) for %r the preset does not have "
                "(%s: notices where text shows outside its regions, a score readout); to OCR "
                'them, copy the "suggested_regions" entries of %s into the "regions" of %s',
                len(rec["suggested_regions"]),
                game,
                ", ".join(r["name"] for r in rec["suggested_regions"]),
                rec["proposal"],
                rec["preset"],
            )

    def _maybe_top_up(
        self, game: str, srcs: list[Any], existing: Any, proposal: dict[str, Any] | None
    ) -> None:
        """An installed discovered preset without a vision HUD map gets one, once (#1069)."""
        mapper = self.hud_mapper
        if not (
            mapper is not None
            and vision_map.wants_top_up(existing, cutpoint_paths.presets_dir())
            and mapper.usable(top_up=True)
            and not vision_map.tried(game, mapper.model, existing.path)
        ):
            return
        if self.before_top_up is not None:
            self.before_top_up()
        try:
            # the model is loaded (on a GPU) before the minutes of discovery it is for
            why = None if proposal is not None else mapper.ready()
            if why is None:
                why = self._top_up(
                    game, existing, proposal or self._discover(game, srcs, existing, topping=True)
                )
        finally:
            mapper.release()
        if why is not None:
            vision_map.mark_tried(game, mapper.model, existing.path, why)
            logger.info(
                "No vision HUD map for the preset '%s' (%s); not tried again until the "
                "preset or the vision model changes",
                game,
                why,
            )

    def _top_up(self, game: str, existing: Any, proposal: dict[str, Any] | None) -> str | None:
        """Add the vision HUD map of ``proposal`` (a discovery over the existing preset) to the
        installed discovered preset ``existing`` (:func:`cutpoint.events.hud_map.top_up`),
        written atomically. Returns why there was no map to add (the model answered for no
        frame, or the discovery had none); None when it was added, or the preset was topped up
        meanwhile or cannot be read or written (WARNING)."""
        disc = (proposal or {}).get("discovery") or {}
        got = disc.get("hud_map")
        if not got:
            return "the vision model gave no HUD map"
        if existing.path is None:
            return None
        data = cache.load_json(existing.path)
        if not isinstance(data, dict):
            return None
        if "hud_map" in (data.get("discovery") or {}):  # another run topped it up meanwhile
            return None
        ids = set(disc.get("frames") or [])
        recs = discover.read_ocr_samples(self.project / "discovery" / game / "ocr.jsonl")
        frames = discover.gameplay_frames([f for f in recs if f["id"] in ids], self.params)
        new, changes = vision_map.top_up(data, got, frames)
        try:
            cache.save_json(existing.path, new)
        except OSError as err:
            logger.warning(
                "Could not add the vision HUD map to the preset '%s' (%s): %s",
                game,
                existing.path,
                err,
            )
            return None
        if game in self.out:
            self.out[game]["topped_up"] = changes
        done = "; ".join(f"{k} {', '.join(v)}" for k, v in changes.items() if v)
        logger.info(
            "Topped up the HUD preset '%s' with the vision HUD map (%s): %s",
            game,
            done or "no region changed",
            existing.path,
        )
        return None

    @staticmethod
    def _add_fit_ref(game: str, path: Path | None, ref: Any) -> None:
        """Give an existing preset the fresh grid ``fit_ref`` of a forced discovery (the HUD
        fit check); every other key stays as it is. A ref of too little evidence is not
        written; a preset that cannot be written is left as it is (WARNING)."""
        if path is None or not game_fit.valid_ref(ref) or not ref["grid"]:
            return
        if not (
            isinstance(ref.get("frames"), int)
            and isinstance(ref.get("boxes"), int)
            and ref["frames"] >= game_fit.MIN_FRAMES
            and ref["boxes"] >= game_fit.MIN_BOXES
        ):
            return
        data = cache.load_json(path)
        if not isinstance(data, dict):
            return
        disc = data.get("discovery", {})
        if not isinstance(disc, dict) or disc.get("fit_ref") == ref:
            return
        data["discovery"] = disc | {"fit_ref": ref}
        try:
            cache.save_json(path, data)
        except OSError as err:
            logger.warning(
                "Could not add the HUD fit check to the preset '%s' (%s): %s", game, path, err
            )
            return
        logger.info(
            "Added the HUD fit check to the preset '%s' (its regions are unchanged): %s", game, path
        )

    def _install_preset(self, game: str, proposal: dict[str, Any]) -> None:
        """Write a discovered preset (its record in ``out`` says where)."""
        if self.scope == "project":
            folder = self.project / "hud_presets"
        else:
            folder = cutpoint_paths.presets_dir()
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{check_game_key(game)}.json"
        old = cache.load_json(target) if target.is_file() else None
        if isinstance(old, dict) and hud.stale_mapping(old):
            # a preset made with the wrong y scale (#1085): keep it aside and carry over what
            # the user set in it; the new regions replace the old ones
            backup = target.with_suffix(".json.pre-1085")
            target.replace(backup)
            proposal = proposal | {k: old[k] for k in USER_PRESET_KEYS if k in old}
            if isinstance(old.get("feed"), dict):  # the user's feed keys; row_notices is
                # discovery's, so only the new discovery sets it
                kept = {k: v for k, v in old["feed"].items() if k != "row_notices"}
                proposal["feed"] = kept | (proposal.get("feed") or {})
            logger.warning(
                "Replaced the HUD preset for %r (made with a wrong y scale, #1085); the old "
                "file is kept as %s",
                game,
                backup,
            )
        cache.save_json(target, discover.installable(proposal))
        self.out[game].update(installed=True, preset=str(target))
        logger.info(
            "Installed discovered HUD preset for %r: %s (edit it to calibrate)", game, target
        )

    # ---- the HUD fit check

    def _per_source(
        self, recs: list[dict[str, Any]], srcs: list[Any]
    ) -> dict[str, list[dict[str, Any]]]:
        """The gameplay frames among OCR records, per source path."""
        paths = {discover.src_key(s): str(s.path) for s in srcs}
        got: dict[str, list[dict[str, Any]]] = {}
        for f in discover.gameplay_frames(recs, self.params):
            if discover._src(f) in paths:
                got.setdefault(paths[discover._src(f)], []).append(f)
        return got

    def _fit_frames(
        self, game: str, proposal: dict[str, Any] | None, srcs: list[Any], tested: list[Any]
    ) -> dict[str, list[dict[str, Any]]]:
        """Gameplay frames per source: this run's discovery records, and a sample of their own
        for ``tested`` sources with too few."""
        got: dict[str, list[dict[str, Any]]] = {}
        ids = set(((proposal or {}).get("discovery") or {}).get("frames") or [])
        if ids:
            recs = discover.read_ocr_samples(self.project / "discovery" / game / "ocr.jsonl")
            got = self._per_source([f for f in recs if f["id"] in ids], srcs)
        need = [s for s in tested if len(got.get(str(s.path), [])) < game_fit.TOP_UP_FRAMES]
        if need:
            logger.debug("HUD fit check for %r: sampling %d source(s)", game, len(need))
            try:
                recs = game_fit.sample_frames(
                    need, self.project / "discovery" / "_fit" / game, self.params
                )
            except LowMemoryError:
                raise
            except Exception as err:
                memguard.raise_if_out_of_memory(
                    err, "the HUD fit check", f"{game}: {', '.join(s.name for s in need)}"
                )
                raise
            for path, frames in self._per_source(recs, need).items():
                got.setdefault(path, []).extend(frames)
        return got

    @staticmethod
    def _fit_ref(preset: Any) -> Any:
        data = cache.load_json(preset.path) if preset.path else None
        return (
            ((data or {}).get("discovery") or {}).get("fit_ref") if isinstance(data, dict) else None
        )

    def _persist(self) -> None:
        """Save the decisions made so far (each as soon as it is made: an interrupted run
        keeps them), with the date groups of their folders."""
        entries, groups = hud.load_game_keys(self.project), hud.load_game_groups(self.project)
        for path, (key, fit) in self.decided.items():
            entries[hud.source_id(path)] = {
                "tag": hud.source_tag(path),
                "guess": self.guesses[path],
                "key": key,
                "fit": fit.record() if fit is not None else None,
            }
            folder, stem = hud.date_group(path, self.project)
            if stem is not None:
                groups[folder] = stem
        hud.save_game_keys(self.project, entries, groups)

    def _made_from(self, key: str, s: Any) -> bool:
        """Whether the preset ``key`` was discovered from source ``s``: the OCR records of
        this project's ``discovery/<key>/proposal.json`` (an installed preset keeps none,
        A647); False when that file is missing or unreadable."""
        ids = discover.proposal_frames(self.project / "discovery" / key / "proposal.json")
        prefix = f"{discover.src_key(s)}@"
        return isinstance(ids, list) and any(str(i).startswith(prefix) for i in ids)
