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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cutpoint import cache, sources
from cutpoint.boundaries import SpeechSettings
from cutpoint.clips import plan_edges
from cutpoint.config import remap_source
from cutpoint.events.tails import EventTailSettings, load_events
from cutpoint.ffmpeg_utils import probe
from cutpoint.planning import Padding
from cutpoint.styles import get_style
from cutpoint.tail import TailSettings

logger = logging.getLogger(__name__)

MAX_CACHED = 64  # planned sequences kept per project
_SPEECH = SpeechSettings()  # the pass's built-in gaps and breath

Edges = dict[str, tuple[float, float]]  # row id -> exported (start, end), source seconds
Reasons = dict[str, str]  # row id -> why its exported range differs from the raw one


@dataclass(frozen=True)
class ExportOptions:
    """The export settings besides Smart tails and Speech-aware cuts that move a clip's ends
    (``None``: the style's or the CLI's default). ``event_tails``: True on, False off, None auto."""

    event_tails: bool | None = None
    event_window: float | None = None
    tail_min: float | None = None
    tail_max: float | None = None
    speech_max_extend: float | None = None
    speech_max_lead: float | None = None
    extend_start: float | None = None
    extend_end: float | None = None


def _reason(plan: Any) -> str | None:
    """Why ``plan`` (a :class:`cutpoint.clips.ClipPlan`) leaves the raw range, or None."""
    tail = getattr(plan, "tail", None)
    why = []
    if tail is not None and str(tail.reason).startswith("event payoff"):
        why.append("event tail")
    elif tail is not None and tail.extension > 0.0005:
        why.append("smart tail")
    speech = getattr(plan, "speech", None) or {}
    if any((speech.get(side) or {}).get("moved") for side in ("start", "end")):
        why.append("speech-aware")
    applied = (getattr(plan, "pad_rec", None) or {}).get("applied") or {}
    if applied.get("start", 0) > 0.0005 or applied.get("end", 0) > 0.0005:
        why.append("extend")
    return ", ".join(why) or None


class EdgePlanner:
    """Exported in/out points of sequence rows, and phrase ends, for one project folder.

    ``resolve``: the file read for a recorded source path (``source_remap``).
    """

    def __init__(self, project: Path, resolve: Callable[[str], Path] = remap_source) -> None:
        self.project = project
        self.resolve = resolve
        self._lock = threading.Lock()
        self._plans: OrderedDict[tuple[Any, ...], tuple[Edges, Reasons]] = OrderedDict()
        self._chains: dict[tuple[Any, ...], list[tuple[float, float]]] = {}

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
        self,
        rows: Sequence[dict[str, Any]],
        *,
        smart_tail: bool = True,
        speech_aware: bool = True,
        options: ExportOptions | None = None,
    ) -> Edges:
        """``{id: (start, end)}`` of ``rows`` as ``clips`` exports them with smart tails and
        speech-aware cuts on (``smart_tail`` / ``speech_aware`` False turn one off for every
        row; a row's own False does so for it; ``options``: the other export settings). A row
        that cannot be planned (a source that cannot be read) keeps its raw range; so do all of
        them if planning fails. Only a plan that worked is kept for the next ask."""
        return self.explain(
            rows, smart_tail=smart_tail, speech_aware=speech_aware, options=options
        )[0]

    def explain(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        smart_tail: bool = True,
        speech_aware: bool = True,
        options: ExportOptions | None = None,
    ) -> tuple[Edges, Reasons]:
        """:meth:`plan` and, for each row whose range changed, why (``event tail``,
        ``smart tail``, ``speech-aware``, ``extend``; several joined by ``, ``)."""
        raw = {r["id"]: (float(r["start"]), float(r["end"])) for r in rows}
        if not rows:
            return {}, {}
        opts = options or ExportOptions()
        key = (
            smart_tail,
            speech_aware,
            opts,
            self._stamp(),
            tuple(self._version(s) for s in sorted({r["source"] for r in rows})),
            tuple(
                (r["id"], r["source"], raw[r["id"]], r.get("smart_tail"), r.get("speech_aware"))
                for r in rows
            ),
        )
        with self._lock:
            if key in self._plans:
                self._plans.move_to_end(key)
                edges, reasons = self._plans[key]
                return dict(edges), dict(reasons)
            try:
                planned, reasons, whole = self._plan(rows, smart_tail, speech_aware, opts)
            except Exception as err:  # noqa: BLE001  the chat and preview go on with raw ranges
                logger.warning("could not plan the exported edges (%s); using the raw ones", err)
                return dict(raw), {}
            out = {**raw, **planned}
            if whole:  # a source that could not be read may be back at the next ask
                self._plans[key] = (out, reasons)
                while len(self._plans) > MAX_CACHED:
                    self._plans.popitem(last=False)
            return dict(out), dict(reasons)

    def _version(self, source: str) -> tuple[str, int, int]:
        """``(resolved path, mtime_ns, size)`` of ``source``; zeros when it cannot be read."""
        path = self.resolve(source)
        try:
            stamp = path.stat()
            return (str(path), stamp.st_mtime_ns, stamp.st_size)
        except OSError:
            return (str(path), 0, 0)

    def _plan(
        self,
        rows: Sequence[dict[str, Any]],
        smart_tail: bool,
        speech_aware: bool,
        opts: ExportOptions,
    ) -> tuple[Edges, Reasons, bool]:
        """``(edges, reasons, whole)``: the planned rows (those of the sources that can be
        read), why each moved, and whether every source could be read."""
        hl = cache.load_json(self.project / "highlights.json")
        hl = hl if isinstance(hl, dict) else {}
        try:
            style = get_style(str(hl.get("style", "gaming")))
        except ValueError:
            style = get_style("gaming")
        settings = hl.get("settings") if isinstance(hl.get("settings"), dict) else {}
        events = load_events(self.project) if opts.event_tails is not False else None
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
        infos = {}
        for src, path in io_paths.items():
            try:
                infos[src] = probe(path)
            except Exception as err:  # noqa: BLE001  its rows keep their raw ranges
                logger.warning("could not read %s (%s); its clips keep their raw ranges", src, err)
        moments = [m for m in moments if m["source"] in infos]
        if not moments:
            return {}, {}, False
        padding = Padding.from_args(None, opts.extend_start, opts.extend_end)
        plans = plan_edges(
            moments,
            self.project,
            io_paths={s: p for s, p in io_paths.items() if s in infos},
            infos=infos,
            tail=TailSettings.from_style(
                style, enabled=True, tail_min=opts.tail_min, tail_max=opts.tail_max
            ),
            events=events,
            event_settings=EventTailSettings(
                enabled=events is not None,
                **({"window": opts.event_window} if opts.event_window else {}),
            ),
            speech=SpeechSettings.from_style(
                style,
                enabled=True,
                max_extend=opts.speech_max_extend,
                max_lead=opts.speech_max_lead,
            ),
            padding=padding if padding.active else None,
            language=settings.get("language"),
        )
        planned = {p.entry["id"]: (round(p.start, 3), round(p.end, 3)) for p in plans}
        reasons = {
            p.entry["id"]: why
            for p in plans
            if (why := _reason(p))
            and (round(p.start, 3), round(p.end, 3))
            != (float(p.entry["start"]), float(p.entry["end"]))
        }
        return planned, reasons, len(infos) == len(io_paths)

    # -- phrase ends -------------------------------------------------------------------
    def _phrases(self, source: str) -> list[tuple[float, float]]:
        """The stretches of ``source`` where speech runs on (words closer than
        ``chain_gap``), as ``(start, end)``; [] without a transcript."""
        path = self.resolve(source)
        language = sources.project_language(self.project)
        transcripts = [
            sources.transcript_stamp(part.cdir, language) for part in sources.parts(path)
        ]
        key = (str(path), language, tuple(transcripts))
        if key not in self._chains:
            words = sources.source_words(path, language) or []
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
        if end - t > _SPEECH.max_extend:  # speech runs on: a cut here is not a phrase end
            return t
        return max(end, t)


def length(rows: Sequence[dict[str, Any]], edges: Edges | None = None) -> float:
    """Seconds the ``rows`` last: at their exported ``edges`` where given, else raw."""
    total = 0.0
    for r in rows:
        start, end = (edges or {}).get(r["id"], (r["start"], r["end"]))
        total += end - start
    return total
