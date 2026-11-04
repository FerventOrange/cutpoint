"""Where the sequence's clips start and end once exported, and where a phrase ends.

A sequence row is the user's raw range. The export moves its ends: the smart tail lets a
clip finish a sentence or a reaction and the speech-aware pass moves cuts that fall mid-word
(:func:`cutpoint.clips.plan_edges`, the planning step of ``cutpoint clips``). :class:`EdgePlanner`
runs that step for a sequence so that the page's preview and totals, and the lengths the model
sees, are those of the export. It reads cached analysis data and writes nothing. Planning is
cached per project, so asking again for the same rows costs nothing.

:meth:`EdgePlanner.phrase_end` is the hook ``ops.apply_ops`` takes (``snap``): a trim that
lengthens a clip ends it where the phrase it lands in ends.
"""

from __future__ import annotations

import bisect
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from cutpoint import cache, sources
from cutpoint.boundaries import SpeechSettings
from cutpoint.clips import plan_edges
from cutpoint.config import remap_source
from cutpoint.events.tails import EventTailSettings, load_events
from cutpoint.ffmpeg_utils import probe
from cutpoint.styles import get_style
from cutpoint.tail import TailSettings

logger = logging.getLogger(__name__)

MAX_CACHED = 64  # planned sequences kept per project
_SPEECH = SpeechSettings()  # the pass's built-in gaps and breath

Edges = dict[str, tuple[float, float]]  # row id -> exported (start, end), source seconds


class EdgePlanner:
    """Exported in/out points of sequence rows, and phrase ends, for one project folder.

    ``resolve``: the file read for a recorded source path (``source_remap``).
    """

    def __init__(self, project: Path, resolve: Callable[[str], Path] = remap_source) -> None:
        self.project = project
        self.resolve = resolve
        self._lock = threading.Lock()
        self._plans: OrderedDict[tuple[Any, ...], Edges] = OrderedDict()
        self._chains: dict[tuple[str, tuple[int, int]], list[tuple[float, float]]] = {}

    # -- exported edges ----------------------------------------------------------------
    def _stamp(self) -> tuple[int, ...]:
        """What the plan depends on besides the rows: the analysis files' modification times."""
        out = []
        for name in ("highlights.json", "events.json"):
            try:
                out.append((self.project / name).stat().st_mtime_ns)
            except OSError:
                out.append(0)
        return tuple(out)

    def plan(
        self, rows: Sequence[dict[str, Any]], *, smart_tail: bool = True, speech_aware: bool = True
    ) -> Edges:
        """``{id: (start, end)}`` of ``rows`` as ``clips`` exports them with smart tails and
        speech-aware cuts on (``smart_tail`` / ``speech_aware`` False turn one off for every
        row; a row's own False does so for it). A row that cannot be planned (a source that
        cannot be read) keeps its raw range; so do all of them if planning fails."""
        raw = {r["id"]: (float(r["start"]), float(r["end"])) for r in rows}
        if not rows:
            return {}
        key = (
            smart_tail,
            speech_aware,
            self._stamp(),
            tuple(
                (r["id"], r["source"], raw[r["id"]], r.get("smart_tail"), r.get("speech_aware"))
                for r in rows
            ),
        )
        with self._lock:
            if key in self._plans:
                self._plans.move_to_end(key)
                return dict(self._plans[key])
            try:
                planned = self._plan(rows, smart_tail, speech_aware)
            except Exception as err:  # noqa: BLE001  the chat and preview go on with raw ranges
                logger.warning("could not plan the exported edges (%s); using the raw ones", err)
                planned = {}
            out = {**raw, **planned}
            self._plans[key] = out
            while len(self._plans) > MAX_CACHED:
                self._plans.popitem(last=False)
            return dict(out)

    def _plan(self, rows: Sequence[dict[str, Any]], smart_tail: bool, speech_aware: bool) -> Edges:
        hl = cache.load_json(self.project / "highlights.json")
        hl = hl if isinstance(hl, dict) else {}
        try:
            style = get_style(str(hl.get("style", "gaming")))
        except ValueError:
            style = get_style("gaming")
        settings = hl.get("settings") if isinstance(hl.get("settings"), dict) else {}
        events = load_events(self.project)
        off = {"smart_tail": False} if not smart_tail else {}
        off |= {"speech_aware": False} if not speech_aware else {}
        moments = [
            {
                "id": r["id"],
                "source": r["source"],
                "start": float(r["start"]),
                "end": float(r["end"]),
                "label": str(r.get("label", "segment")),
                **{k: r[k] for k in ("smart_tail", "speech_aware") if r.get(k) is False},
                **off,
            }
            for r in rows
        ]
        io_paths = {s: self.resolve(s) for s in {m["source"] for m in moments}}
        plans = plan_edges(
            moments,
            self.project,
            io_paths=io_paths,
            infos={s: probe(p) for s, p in io_paths.items()},
            tail=TailSettings.from_style(style, enabled=True),
            events=events,
            event_settings=EventTailSettings(enabled=events is not None),
            speech=SpeechSettings.from_style(style, enabled=True),
            language=settings.get("language"),
        )
        return {p.entry["id"]: (round(p.start, 3), round(p.end, 3)) for p in plans}

    # -- phrase ends -------------------------------------------------------------------
    def _phrases(self, source: str) -> list[tuple[float, float]]:
        """The stretches of ``source`` where speech runs on (words closer than
        ``chain_gap``), as ``(start, end)``; [] without a transcript."""
        path = self.resolve(source)
        try:
            stamp = path.stat()
            version = (stamp.st_mtime_ns, stamp.st_size)
        except OSError:
            version = (0, 0)
        key = (str(path), version)
        if key not in self._chains:
            words = sources.source_words(path) or []
            chains: list[tuple[float, float]] = []
            for w in words:
                start, end = float(w["start"]), float(w["end"])
                if chains and start - chains[-1][1] < _SPEECH.chain_gap:
                    chains[-1] = (chains[-1][0], max(chains[-1][1], end))
                else:
                    chains.append((start, end))
            self._chains[key] = chains
        return self._chains[key]

    def phrase_end(self, source: str, t: float) -> float:
        """Where the phrase spoken at ``t`` of ``source`` ends (its last word and a breath);
        ``t`` itself when no speech runs there."""
        try:
            chains = self._phrases(source)
        except Exception as err:  # noqa: BLE001  no words: the end stays where it was asked
            logger.debug("no phrase end for %s: %s", source, err)
            return t
        i = bisect.bisect_right(chains, (t, float("inf"))) - 1
        if i < 0 or not chains[i][0] <= t < chains[i][1]:
            return t
        end = chains[i][1] + _SPEECH.breath
        if i + 1 < len(chains):
            end = min(end, chains[i + 1][0] - 0.05)
        return max(end, t)


def length(rows: Sequence[dict[str, Any]], edges: Edges | None = None) -> float:
    """Seconds the ``rows`` last: at their exported ``edges`` where given, else raw."""
    total = 0.0
    for r in rows:
        start, end = (edges or {}).get(r["id"], (r["start"], r["end"]))
        total += end - start
    return total
