"""The AI Chat routes of ``cutpoint ui``: turns, proposals and history."""

from __future__ import annotations

import contextlib
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from cutpoint.assistant import context as ai_context
from cutpoint.assistant import edges as ai_edges
from cutpoint.assistant import lookfor as ai_lookfor
from cutpoint.assistant import ops as ai_ops
from cutpoint.assistant import providers as ai_providers
from cutpoint.assistant import session as ai_session
from cutpoint.assistant import store as ai_store
from cutpoint.assistant import watch as ai_watch
from cutpoint.config import (
    COLOR_MODES,
    load_config_data,
    remap_source,
    update_section,
    vision_assistant_settings,
)
from cutpoint.pano import project as pano_project
from cutpoint.ui.analysis import _analysis_inputs, _color_after, _color_state, _flag, _seconds
from cutpoint.ui.common import EVENT_TAILS, MAX_EXTEND_S, MAX_TIMING_S, REVERT, UI_DIR, bad
from cutpoint.ui.files import _highlights, _inside, _load, _own, _ui_folder, read_meta
from cutpoint.ui.projects import chat_files, project_dir, validate_segments
from cutpoint.ui.ranking import RankingHook, ranking_key
from cutpoint.ui.routes.context import AppContext
from cutpoint.ui.settings import consent_view


# -- assistant (#346) --------------------------------------------------------------
# A chat turn only proposes sequence edits: the page applies one to its working sequence and
# reports it here for the history. Nothing below writes review.json or highlights.json; GET
# routes only read. A ranking it previews is saved only by the apply route above, which the
# page sends when the user clicks to use it.
def assistant_provider() -> ai_providers.Provider:
    try:
        return ai_providers.make()
    except (ValueError, TypeError) as err:  # a bad Ollama port, an invalid config.json
        raise bad(str(err), 409) from err


def vision_provider() -> ai_providers.Provider:
    try:
        return ai_providers.make_vision(vision_assistant_settings())
    except (ValueError, TypeError) as err:  # a bad Ollama port, an invalid config.json
        raise bad(str(err), 409) from err


def assistant_save(p: Path, name: str) -> None:
    """Check ``ui/assistant/<name>`` may be written and create its folder."""
    _own(p, f"{ai_store.FOLDER}/{name}")
    _ui_folder(p, "assistant")


def frames_state(p: Path, provider: ai_providers.Provider) -> dict[str, Any]:
    """Whether video frames of ``p`` may go to ``provider`` (:func:`ai_watch.frames_view`);
    an unreadable config.json allows none."""
    try:
        data = load_config_data()
    except ValueError:
        data = {}
    return ai_watch.frames_view(provider, p, data)


def frames_given(p: Path) -> tuple[str, ...]:
    try:
        return ai_watch.frames_consent(load_config_data(), p)
    except ValueError:  # config.json became unreadable: no frames go out
        return ()


def register(ctx: AppContext) -> None:
    assistant_revert, assistant_state, media, owned, proxy_error, sources, turns = (
        ctx.assistant_revert,
        ctx.assistant_state,
        ctx.media,
        ctx.owned,
        ctx.proxy_error,
        ctx.sources,
        ctx.turns,
    )

    planners: dict[Path, ai_edges.EdgePlanner] = {}  # one per project: it caches what it planned

    def edge_planner(p: Path) -> ai_edges.EdgePlanner:
        return planners.setdefault(p, ai_edges.EdgePlanner(p, remap_source))

    @owned.post("/projects/{pid}/edges")
    def planned_edges(pid: str, body: dict[str, Any]):
        """Where the clips of the working sequence ``segments`` start and end once exported
        (:meth:`cutpoint.assistant.edges.EdgePlanner.plan`: Smart tails and Speech-aware cuts,
        both on unless ``smart_tail`` / ``speech_aware`` say false) and the other export settings
        the page sends as Export clips does (``event_tails``, ``event_window``, ``tail_min``,
        ``tail_max``, ``speech_max_extend``, ``speech_max_lead``, ``extend_start``,
        ``extend_end``), as ``{id: {start, end, reason}}`` (``reason`` only for a range that moved:
        ``event tail``, ``smart tail``, ``speech-aware`` or ``extend``); the page previews and
        totals the sequence with them. Writes nothing."""
        p = project_dir(pid)
        flags = {"smart_tail": True, "speech_aware": True, **body}
        segments = validate_segments(body.get("segments"), sources.entries(p), media)
        event_tails = body.get("event_tails", "auto")
        if not isinstance(event_tails, str) or event_tails not in EVENT_TAILS:
            raise bad("event_tails must be 'auto', 'on' or 'off'")
        options = ai_edges.ExportOptions(
            event_tails={"auto": None, "on": True, "off": False}[event_tails],
            event_window=_seconds(body, "event_window", MAX_TIMING_S, positive=True),
            tail_min=_seconds(body, "tail_min", MAX_TIMING_S),
            tail_max=_seconds(body, "tail_max", MAX_TIMING_S),
            speech_max_extend=_seconds(body, "speech_max_extend", MAX_TIMING_S),
            speech_max_lead=_seconds(body, "speech_max_lead", MAX_TIMING_S),
            extend_start=_seconds(body, "extend_start", MAX_EXTEND_S),
            extend_end=_seconds(body, "extend_end", MAX_EXTEND_S),
        )
        planned, reasons = edge_planner(p).explain(
            segments,
            smart_tail=_flag(flags, "smart_tail"),
            speech_aware=_flag(flags, "speech_aware"),
            options=options,
        )
        return {
            "edges": {
                k: {"start": a, "end": b, **({"reason": reasons[k]} if k in reasons else {})}
                for k, (a, b) in planned.items()
            }
        }

    def assistant_turn(pid: str, tid: str) -> ai_session.Turn:
        turn = turns.get(str(project_dir(pid)), tid)
        if turn is None:
            raise bad("unknown turn", 404)
        return turn

    @owned.get("/projects/{pid}/assistant")
    def get_assistant(pid: str):
        """The provider in use, whether the user must agree before it gets project text
        (``consent``), whether its visual search can run (``frame_index``: ``ok``, else ``why``
        and ``fix``, :func:`cutpoint.assistant.lookfor.index_state`), the saved chat and the
        turn still running, if any."""
        p = project_dir(pid)
        running = [t.view() for t in turns.of(str(p)) if t.status == "running"]
        provider = assistant_provider()
        return {
            "provider": ai_providers.describe(provider),
            "consent": consent_view(provider),
            "frames": frames_state(p, provider),
            "frame_index": ai_lookfor.index_state([s["path"] for s in sources.entries(p)]),
            "turns": ai_store.turns(p),
            "running": running,
            # the files the chat may name: the page links a reply naming one (Open folder)
            "files": [{"name": f["name"], "target": f["target"]} for f in chat_files(p)],
        }

    @owned.post("/projects/{pid}/assistant/frames-consent")
    def give_frames_consent(pid: str, body: dict[str, Any]):
        """Allow video frames of this project to go to the assistant's provider, named by its
        ``key`` (``frames-v1:<name>@<host>``), kept in ``assistant.frames_consent`` of
        ``config.json`` under the project folder; 409 when the provider in use is another one
        by now or is on this machine, or the user has not agreed to the assistant itself."""
        p = project_dir(pid)
        provider = assistant_provider()
        want = ai_watch.frames_key(provider)
        if want is None or body.get("key") != want:
            raise bad("the assistant's provider changed: check it and allow again", 409)
        if consent_view(provider)["needed"]:
            raise bad(f"agree first to use {provider.host} in AI Chat", 409)
        where = ai_watch.project_key(p)

        def allow(section: dict[str, Any]) -> dict[str, Any] | None:
            given = section.get("frames_consent", {})
            if not isinstance(given, dict):
                raise bad("assistant.frames_consent in config.json must be an object", 409)
            keys_now = list(ai_watch.frames_consent({"assistant": section}, p))
            if want in keys_now:
                return None
            return {**section, "frames_consent": {**given, where: [*keys_now, want]}}

        try:
            update_section("assistant", allow)  # re-read under the config lock
        except ValueError as err:
            raise bad(str(err), 409) from err
        except OSError as err:
            raise bad(f"could not save the consent: {err.strerror or err}", 500) from err
        return {"frames": frames_state(p, provider)}

    @owned.post("/projects/{pid}/assistant/turns")
    def start_turn(pid: str, body: dict[str, Any]):
        """Ask the assistant ``text`` about the working sequence ``segments`` (default: the
        saved review). Poll the turn for its events. A hosted provider the user has not agreed
        to send project text to (``POST /api/assistant/consent``) gets nothing: 409."""
        p = project_dir(pid)
        provider = assistant_provider()
        if consent_view(provider)["needed"]:
            raise bad(
                f"agree first to send this project's analysis results (moment labels, scores "
                f"and signals, vision descriptions, transcript snippets, file names and times) "
                f"and, when analyzing, the whole transcripts of its footage to {provider.host}",
                409,
            )
        text = body.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > ai_session.MAX_TEXT:
            raise bad(f"text must be 1-{ai_session.MAX_TEXT} characters")
        raw = body.get("segments")
        if raw is None:
            review = _load(p, f"{UI_DIR}/review.json")
            raw = (review.get("segments") if isinstance(review, dict) else None) or []
        segments = validate_segments(raw, sources.entries(p), media)
        moments, durations, key = assistant_state(p, segments)
        transcript = ""
        if _inside(p, "transcript.txt"):
            with contextlib.suppress(OSError, UnicodeDecodeError):
                transcript = (p / "transcript.txt").read_text(encoding="utf-8-sig")
        # the events VLM windows: what the vision model saw, which find_moments searches
        windows = ai_context.vlm_windows(_load(p, "events.json"))
        # preview_ranking: an analyzed project that is not 360, ranked in its recorded style
        meta, hl = read_meta(p), _highlights(p)
        ranking = None
        if hl and meta.get("kind") != "360" and not (p / pano_project.PANO_FILE).is_file():
            current = hl.get("style") or _analysis_inputs(meta, hl)[1] or ""
            ranking = RankingHook(p, str(current))
        project = ai_session.Project(
            moments,
            segments,
            durations,
            transcript,
            key,
            windows,
            edge_planner(p),
            frame_index=ai_lookfor.indexed(list(durations)),  # no frames, no look_for
            index_state=ai_lookfor.index_state(list(durations)),  # why not, in the rules
            words=edge_planner(p).words,  # find_words: the cached transcript words
            ranking=ranking,
            ranking_key=ranking_key(p),  # kept in the turn: later turns know when ids changed
            files=chat_files(p),  # names only
            folder=str(p),  # only a model on this machine is given it (Turn.folder)
        )
        if ai_providers.is_subscription(provider) and (why := proxy_error(logged_in_only=True)):
            raise bad(why, 503)
        assistant_save(p, ai_store.CHAT)
        viewer = ai_watch.Viewer(p, provider, resolve=remap_source, consent=lambda: frames_given(p))
        turn = ai_session.Turn(provider, project, text.strip(), ai_store.turns(p), viewer=viewer)
        try:
            turns.start(str(p), turn, lambda t: ai_store.add_turn(p, t.record()))
        except ai_session.BusyError as err:
            raise bad(str(err), 409) from err
        return turn.view()

    @owned.get("/projects/{pid}/assistant/turns/{tid}")
    def poll_turn(pid: str, tid: str, after: int = 0):
        """The turn with its events from number ``after`` on."""
        return assistant_turn(pid, tid).view(after)

    @owned.post("/projects/{pid}/assistant/turns/{tid}/cancel")
    def cancel_turn(pid: str, tid: str):
        turn = assistant_turn(pid, tid)
        turn.cancel()
        return turn.view()

    @owned.post("/projects/{pid}/assistant/proposals/{prop}/applied")
    def proposal_applied(pid: str, prop: str, body: dict[str, Any]):
        """Record that the page applied proposal ``prop`` to its working sequence ``segments``
        (as it was before). When those or the analysis changed since the proposal was made, its
        ops are applied again to them; the result's ``after`` is the sequence to show.
        ``revert-<id>`` (from ``history/<id>/revert``) puts back the "before" of applied change
        ``<id>``; ``base_key`` (from that preview) must still match ``segments``, else 409.
        ``color`` is the page's Color now (``{mode, strength}``); a proposal with a color op
        (or a revert of one) records it as ``before`` with the Color it sets as ``after``."""
        p = project_dir(pid)
        now = _color_state(body.get("color"))
        if prop.startswith(REVERT):
            undone, back = assistant_revert(p, prop.removeprefix(REVERT))
            found = (undone.get("turn"), {"summary": f"Revert: {undone.get('summary', '')}"})
            color = (undone.get("color") or {}).get("before")
        else:
            back = None
            found = next(
                ((t.id, hit) for t in turns.of(str(p)) if (hit := t.proposal(prop)) is not None),
                None,
            )
            if found is None and (saved := ai_store.find_proposal(p, prop)) is not None:
                found = (saved[0].get("id"), saved[1])
            color = (found[1] if found else {}).get("color")
        if found is None:
            raise bad("unknown proposal", 404)
        tid, proposal = found
        before = validate_segments(body.get("segments"), sources.entries(p), media)
        moments, durations, key = assistant_state(p, before)
        after, changes, rebased = proposal.get("after"), proposal.get("diff"), False
        if back is not None:
            if body.get("base_key") != key:
                raise bad(
                    "the sequence changed since this revert was previewed: preview it again", 409
                )
            after, changes = back, ai_ops.diff(before, back)
        elif key != proposal.get("base_key"):
            # look_for moments are not in highlights.json: the proposal carries them
            known = [*moments, *ai_ops.extra_moments(proposal.get("moments"))]
            out = ai_ops.apply_ops(
                before, proposal.get("ops"), known, durations, snap=edge_planner(p).phrase_end
            )
            if out.error:
                raise bad(f"the proposal no longer applies to this sequence: {out.error}", 409)
            after, changes, rebased = out.after, out.diff, True
        entry = {
            "id": secrets.token_hex(6),
            "proposal": prop,
            "turn": tid,
            "summary": proposal.get("summary", ""),
            "applied": datetime.now().isoformat(timespec="seconds"),
            "base_key": key,
            "rebased": rebased,
            "before": before,
            "after": after,
            "diff": changes,
        }
        if isinstance(color, dict) and color.get("mode") in COLOR_MODES:
            entry["color"] = {"before": now, "after": _color_after(color, now)}
        assistant_save(p, ai_store.HISTORY)
        ai_store.add_applied(p, entry)
        return entry

    @owned.delete("/projects/{pid}/assistant/chat")
    def clear_chat(pid: str):
        """Forget the chat and stop a running turn; the history of applied proposals stays."""
        p = project_dir(pid)
        turns.clear(str(p))
        if (p / ai_store.FOLDER / ai_store.CHAT).exists():
            assistant_save(p, ai_store.CHAT)
            ai_store.clear_chat(p)
        return {"turns": []}

    @owned.get("/projects/{pid}/assistant/history")
    def get_history(pid: str):
        return {"applied": ai_store.history(project_dir(pid))}


def register_revert(ctx: AppContext) -> None:
    assistant_revert, assistant_state, media, owned, sources = (
        ctx.assistant_revert,
        ctx.assistant_state,
        ctx.media,
        ctx.owned,
        ctx.sources,
    )

    @owned.post("/projects/{pid}/assistant/history/{hid}/revert")
    def revert_change(pid: str, hid: str, body: dict[str, Any]):
        """A proposal that puts the working sequence ``segments`` back as it was before applied
        change ``hid``, to preview; applying it goes through ``proposals/<id>/applied`` with the
        ``base_key`` returned here. Writes nothing."""
        p = project_dir(pid)
        entry, back = assistant_revert(p, hid)
        now = validate_segments(body.get("segments"), sources.entries(p), media)
        changes = ai_ops.diff(now, back)
        color = (entry.get("color") or {}).get("before")
        if color is not None and color == _color_state(body.get("color")):
            color = None  # the Color is already as it was
        if color is None and not any(row["changes"] for row in changes):
            raise bad("the sequence is already as it was before this change", 409)
        return {
            "id": f"{REVERT}{hid}",
            "summary": f"Revert: {entry.get('summary', '')}",
            "after": back,
            "diff": changes,
            **({"color": color} if color is not None else {}),
            "base_key": assistant_state(p, now)[2],
        }
