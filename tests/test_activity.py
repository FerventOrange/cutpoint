"""Activity profiles: resolution order over the user's library, discovery, the fallbacks,
LLM output validation. The library profiles are invented ones (Cutpoint ships only base
and generic)."""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import HOP, SLED, add_profile, set_config
from cutpoint import activity, jsonout, paths, vision
from cutpoint.errors import LowMemoryError


@pytest.fixture
def fake_clip(monkeypatch: pytest.MonkeyPatch):
    """One embedding axis per library profile (its detect captions) and one per other
    caption (``frames_like("caption:<text>")`` points at it)."""
    axes: dict[str, int] = {}

    def text_emb(prompts: tuple[str, ...]) -> np.ndarray:
        lib = activity.library()
        e = np.zeros((len(prompts), 768), np.float32)
        for k, s in enumerate(prompts):
            owner = next((n for n, p in lib.items() if s in p["detect"]), None)
            e[k, axes.setdefault(owner or f"caption:{s}", len(axes))] = 1.0
        return e

    def frames_like(*names: str) -> np.ndarray:
        v = np.zeros((1, 768), np.float32)
        for n in names:
            v[0, axes.setdefault(n, len(axes))] = 1.0
        return v / np.linalg.norm(v)

    monkeypatch.setattr(activity, "_text_emb", text_emb)
    return frames_like


def _stub(reply: dict) -> activity.Ask:
    calls: list[str] = []

    def ask(prompt: str, images: list[bytes], schema: dict | None) -> dict | None:
        calls.append(prompt)
        return reply

    ask.calls = calls  # type: ignore[attr-defined]
    return ask


@pytest.fixture
def footage(tmp_path: Path, fake_clip, monkeypatch: pytest.MonkeyPatch):
    """Two invented library profiles; keyframes of the project: a clear glacier-sled match
    until the test sets ``emb["v"]``."""
    add_profile(SLED)
    add_profile(HOP)
    emb = {"v": fake_clip("glacier-sled")}
    monkeypatch.setattr(activity, "keyframes", lambda project, **kw: (emb["v"], []))
    return emb


def _vocab(term: str) -> str:
    return f"caption:a first-person video of {term}"


PACK_RAFTING = {
    "name": "Pack Rafting",
    "detect": ["a packraft on a river"],
    "incident": {"pos": ["a flipped packraft"], "neg": ["paddling"], "words": {"swim": 1}},
}


def test_resolve_takes_a_clear_library_match(tmp_path: Path, footage):
    rec = activity.resolve(tmp_path, llm=False)
    assert (rec["name"], rec["source"]) == ("glacier-sled", "library")
    assert rec["profile"] == activity.full(activity.library()["glacier-sled"])


def test_resolve_reuses_the_cached_choice_and_the_flag_overrides_it(tmp_path: Path, footage):
    activity.resolve(tmp_path, llm=False)
    never = _stub({})
    assert activity.resolve(tmp_path, ask=never)["source"] == "library"  # cached
    assert not never.calls
    assert activity.resolve(tmp_path, activity="puddle-hop")["source"] == "flag"


def _ambiguous_and_failing(tmp_path: Path, footage, fake_clip):
    """Ambiguous footage, the LLM fails (one retry): the nearest profile, retried next run."""
    footage["v"] = fake_clip("glacier-sled", "puddle-hop")
    broken = _stub({})
    rec = activity.resolve(tmp_path, refresh=True, context="packrafting trip", ask=broken)
    return rec, broken


def test_failed_synthesis_falls_back_to_the_nearest_profile(tmp_path: Path, footage, fake_clip):
    rec, broken = _ambiguous_and_failing(tmp_path, footage, fake_clip)
    assert (rec["name"], rec["source"], len(broken.calls)) == ("glacier-sled", "fallback", 2)


def test_fallback_is_retried_by_a_plain_call_and_the_discovered_profile_is_saved(
    tmp_path: Path, footage, fake_clip
):
    _ambiguous_and_failing(tmp_path, footage, fake_clip)
    good = _stub(PACK_RAFTING)
    rec = activity.resolve(tmp_path, ask=good)  # a plain call retries with the same context
    assert (rec["name"], rec["source"]) == ("pack-rafting", "synthesized")
    assert "packrafting trip" in good.calls[-1]
    assert "a flipped packraft" in rec["profile"]["incident"]["pos"]
    assert "you okay" in rec["profile"]["incident"]["words"]  # base floor
    saved = json.loads((paths.profiles_dir() / "pack-rafting.json").read_text(encoding="utf-8"))
    assert saved["source"] == "discovered"


def test_discovered_profile_is_reused_by_the_next_similar_video(tmp_path: Path, footage, fake_clip):
    footage["v"] = fake_clip("glacier-sled", "puddle-hop")
    activity.resolve(tmp_path, refresh=True, context="packrafting trip", ask=_stub(PACK_RAFTING))
    footage["v"] = fake_clip("pack-rafting")
    rec = activity.resolve(tmp_path, refresh=True, llm=False)
    assert (rec["name"], rec["source"]) == ("pack-rafting", "library")


def test_discovery_shows_only_the_floor_and_generic_as_examples(tmp_path: Path, footage, fake_clip):
    """No library profile's captions or slang leak into a new profile."""
    footage["v"] = fake_clip(_vocab("kayaking"))
    good = _stub(PACK_RAFTING)
    activity.resolve(tmp_path, refresh=True, ask=good)
    prompt = good.calls[-1]
    examples = prompt.split("Examples:\n", 1)[1]
    assert "sled" not in examples and "puddle" not in examples
    assert '"name": "base"' in examples and '"name": "generic"' in examples
    assert '"vocabulary"' not in examples and '"vlm"' not in examples


def test_the_synthesis_prompt_is_universal():
    words = set(re.findall(r"[a-z]+", activity._SYNTH_PROMPT.lower()))
    assert not words & {"ski", "skis", "mtb", "bike", "bars", "trail", "snow", "surf"}
    schema = activity._schema(10, 20, 4)
    assert "lift" in schema["properties"] and "lift" not in schema["required"]  # optional


def test_a_lone_library_profile_does_not_take_footage_of_another_activity(
    tmp_path: Path, fake_clip, monkeypatch: pytest.MonkeyPatch
):
    """The open vocabulary competes with the library: one discovered profile is not the
    match of every later video."""
    add_profile(SLED)
    monkeypatch.setattr(activity, "keyframes", lambda project, **kw: (emb, []))
    emb = fake_clip(_vocab("kayaking"))
    rec = activity.resolve(tmp_path, llm=False)
    assert rec["source"] == "fallback" and rec["name"] != "glacier-sled"
    assert rec["scores"]["vocabulary:kayaking"] > rec["scores"]["glacier-sled"]
    emb = fake_clip("glacier-sled")
    rec = activity.resolve(tmp_path, refresh=True, llm=False)
    assert (rec["name"], rec["source"]) == ("glacier-sled", "library")


def test_without_a_model_the_profile_is_generic_with_clips_guess_in_the_description(
    tmp_path: Path, fake_clip, monkeypatch: pytest.MonkeyPatch
):
    """#916: a guessed vocabulary term no longer names the profile (a sidewalk walk was
    ``driving-a-car``); CLIP's best three terms are kept in the description."""
    emb = fake_clip(_vocab("kayaking"))
    monkeypatch.setattr(activity, "keyframes", lambda project, **kw: (emb, []))
    rec = activity.resolve(tmp_path, llm=False)
    assert (rec["name"], rec["source"], rec["described_by"]) == ("generic", "fallback", "clip")
    assert rec["profile"] == activity.full(activity.library()["generic"])
    assert re.fullmatch(r"CLIP: kayaking 1\.00, [^,]+ 0\.00, [^,]+ 0\.00", rec["description"])
    assert not activity.user_files()  # never saved
    # nothing clear: generic too
    emb = fake_clip(*(_vocab(t) for t in activity.vocabulary()))
    monkeypatch.setattr(activity, "keyframes", lambda project, **kw: (emb, []))
    rec = activity.resolve(tmp_path, refresh=True, llm=False)
    assert (rec["name"], rec["source"]) == ("generic", "fallback")


def test_a_far_nearer_vocabulary_term_beats_a_weak_library_profile(
    tmp_path: Path, fake_clip, monkeypatch: pytest.MonkeyPatch
):
    """Without a model, a library profile with a fifth of the share is not taken over a
    vocabulary term with far more: the fallback names the term (or is generic)."""
    assert activity.nearest({"glacier-sled": 0.25, "vocabulary:kayaking": 0.75}) == "generic"
    assert activity.nearest({"glacier-sled": 0.75, "vocabulary:kayaking": 0.25}) == "glacier-sled"
    add_profile(SLED)
    emb = fake_clip(_vocab("kayaking"))
    monkeypatch.setattr(activity, "keyframes", lambda project, **kw: (emb, []))
    scores = {"glacier-sled": 0.25, "vocabulary:kayaking": 0.75}
    monkeypatch.setattr(activity, "match_scores", lambda emb, lib: scores)
    rec = activity.resolve(tmp_path, llm=False)
    assert (rec["source"], rec["name"]) == ("fallback", "generic")


def test_a_record_of_a_removed_profile_is_kept_or_discovered_again(
    tmp_path: Path, footage, caplog: pytest.LogCaptureFixture
):
    """Projects of the built-in sport profiles Cutpoint no longer ships keep working."""
    stored = activity.full(activity.validate({**SLED, "name": "ski"}))
    rec = {"version": 1, "request": {"activity": None, "context": None}, "source": "library"}
    rec |= {"name": "ski", "profile": stored}
    (tmp_path / activity.ACTIVITY_FILE).write_text(json.dumps(rec), encoding="utf-8")
    assert activity.resolve(tmp_path, llm=False) == rec  # its full profile is stored
    flagged = rec | {"source": "flag", "request": {"activity": "ski", "context": None}}
    (tmp_path / activity.ACTIVITY_FILE).write_text(json.dumps(flagged), encoding="utf-8")
    assert activity.resolve(tmp_path, activity="ski", llm=False) == flagged  # e.g. resume
    # a failed run asked for it: retried without it, from the footage
    failed = flagged | {"source": "fallback"}
    (tmp_path / activity.ACTIVITY_FILE).write_text(json.dumps(failed), encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="cutpoint.activity"):
        got = activity.resolve(tmp_path, llm=False)
    assert (got["name"], got["source"]) == ("glacier-sled", "library")
    assert got["request"] == {"activity": None, "context": None}
    assert sum("'ski' no longer exists" in r.getMessage() for r in caplog.records) == 1
    with pytest.raises(jsonout.ArgumentError, match="unknown activity 'ski'"):
        activity.resolve(tmp_path, activity="ski", refresh=True)  # a new request: an error


def test_a_context_naming_a_library_profile_picks_it_when_the_llm_fails(
    tmp_path: Path, footage, fake_clip
):
    footage["v"] = fake_clip(_vocab("kayaking"))
    rec = activity.resolve(tmp_path, refresh=True, context="race day on puddle-hop", llm=False)
    assert (rec["name"], rec["source"]) == ("puddle-hop", "fallback")


def test_context_on_a_named_profile_adds_the_llm_additions(tmp_path: Path, footage):
    """The profile plus the LLM's additions."""
    sled = activity.full(activity.library()["glacier-sled"])
    extra = {
        "name": "x",
        "incident": {"pos": ["a sled hitting a gate"], "words": {"tipped the sled": 0.3}},
    }
    rec = activity.resolve(tmp_path, activity="glacier-sled", context="race", ask=_stub(extra))
    assert (rec["name"], rec["source"]) == ("glacier-sled", "context")
    assert rec["profile"]["incident"]["pos"] == [
        *sled["incident"]["pos"],
        extra["incident"]["pos"][0],
    ]
    assert rec["profile"]["incident"]["words"]["tipped the sled"] == 1.0  # never re-weighted


def test_synthesis_validates_and_caps():
    raw = {
        "name": "Moto Trials!",
        "detect": "not a list",
        "incident": {
            "pos": [f"caption {i}" for i in range(30)] + ["x" * 300, 7],
            "neg": ["  a   rider \n balancing  ", "a rider balancing"],
            "words": {
                "Dab": 5,
                "foot down": -1,
                "dabbed my foot": 2,
                "down the run": 1,  # filler only: dropped
                "rider": 1,  # a word of the normal-activity captions: dropped
                "b*": 0.5,  # prefix too short: dropped
                "huh": "x",
                **{f"w{i}": 0.5 for i in range(60)},
            },
        },
        "low_info": ["a dark blurry image"],
        "game": {"hud": ["timer"]},
        "junk": ["dropped"],
    }
    prof = activity.synthesize("trials riding", None, _stub(raw), [])
    inc = prof["incident"]
    assert prof["name"] == "moto-trials"
    assert inc["pos"] == [f"caption {i}" for i in range(activity.MAX_PROMPTS)]
    assert inc["neg"] == ["a rider balancing"]
    w = inc["words"]
    assert 0 < len(w) <= activity.MAX_WORDS
    # single words are capped at GEN_WORD_CAP, phrases at GEN_PHRASE_CAP, lower weights are kept
    assert (w["dab"], w["dabbed my foot"]) == (activity.GEN_WORD_CAP, activity.GEN_PHRASE_CAP)
    assert w["w0"] == 0.5
    assert not {"huh", "foot down", "down the run", "rider", "b*"} & set(w)
    assert "game" not in prof and "junk" not in prof  # LLM output: schema keys only
    assert activity.synthesize("x", None, _stub({"incident": {"pos": []}}), []) is None


def test_default_ask_stays_local_with_hosted_vision_role(home, monkeypatch):
    set_config(home, vision_assistant={"provider": "codex", "consent": []})
    monkeypatch.setattr(
        vision,
        "AssistantVision",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("hosted vision used")),
    )
    monkeypatch.setattr(vision.VisionRater, "available", lambda self: True)
    ask, _ = activity.default_ask()
    rater = getattr(getattr(ask, "func", ask), "__self__", None)
    assert rater is not None and rater.via is None and rater.provider_name == "ollama"


def test_profile_ask_sends_the_short_read_timeout_to_ollama(home, monkeypatch):
    """#608: a stalled CPU-only Ollama must not hold `profile` for ask_json's 600 s default."""
    timeouts: list[float] = []

    class _Resp:
        is_redirect = False

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"message": {"content": '{"name": "x"}'}}

    def post(url, json, timeout, **_k):
        timeouts.append(timeout)
        return _Resp()

    monkeypatch.setattr(vision.VisionRater, "available", lambda self: True)
    monkeypatch.setattr(vision.requests, "post", post)
    ask, _ = activity.default_ask()
    assert ask is not None
    assert ask("describe", [b"jpeg"], None) == {"name": "x"}
    assert timeouts == [activity.PROFILE_TIMEOUT_S] == [120]
    # other callers keep the long default
    vision.VisionRater(local_only=True).ask_json("describe", [b"jpeg"])
    assert timeouts[-1] == 600


def test_resolve_unloads_the_default_ask_model_after_synthesis(
    tmp_path: Path, home, monkeypatch: pytest.MonkeyPatch
):
    """default_ask() returns a functools.partial: resolve must still free the model (VRAM)."""
    add_profile(HOP)
    unloads: list[int] = []
    monkeypatch.setattr(vision.VisionRater, "available", lambda self: True)
    monkeypatch.setattr(vision.VisionRater, "unload", lambda self: unloads.append(1))
    monkeypatch.setattr(activity, "synthesize", lambda *a, **k: None)
    activity.resolve(tmp_path, activity="puddle-hop", context="race day", refresh=True)
    assert unloads == [1]


def test_default_ask_uses_the_ollama_flags_over_config_and_environment(home, monkeypatch):
    """#816: --ollama-url/--ollama-model win for the crash/profile step."""
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
    seen: list[vision.VisionRater] = []

    def available(self):
        seen.append(self)
        return True

    monkeypatch.setattr(vision.VisionRater, "available", available)
    ask, model = activity.default_ask("http://127.0.0.1:2", "my-vl:7b")
    assert ask is not None and model == "my-vl:7b"
    assert seen[0].url == "http://127.0.0.1:2" and seen[0].local_only
    ask, model = activity.default_ask()  # no flags: the environment's Ollama
    assert seen[1].url == "http://127.0.0.1:1" and model == "qwen3.5:9b"


def test_default_ask_survives_a_bad_port(home, monkeypatch, caplog):
    monkeypatch.setenv("OLLAMA_HOST", "myhost:99999")
    with caplog.at_level("WARNING"):
        assert activity.default_ask() == (None, None)
    assert "Activity detection: Ollama not usable" in caplog.text
    assert "invalid port" in caplog.text
    caplog.clear()
    monkeypatch.delenv("OLLAMA_HOST")
    with caplog.at_level("WARNING"):
        assert activity.default_ask("myhost:99999") == (None, None)
    assert "Activity detection: Ollama not usable" in caplog.text  # a flag is checked like the env
    assert "--ollama-url 'myhost:99999'" in caplog.text


def test_default_ask_nobody_asked_for_logs_a_bad_port_at_info(home, monkeypatch, caplog):
    monkeypatch.setenv("OLLAMA_HOST", "myhost:99999")
    with caplog.at_level("INFO"), jsonout.run("analyze") as res:
        assert activity.default_ask(quiet=True) == (None, None)
    (rec,) = [r for r in caplog.records if "Ollama not usable" in r.getMessage()]
    assert rec.levelname == "INFO" and res.warnings == []


def test_resolve_quiet_only_when_nobody_asked_for_the_model(
    tmp_path, monkeypatch, footage, fake_clip
):
    got: list[dict] = []

    def fake(**kw):
        got.append(kw)
        return None, None

    monkeypatch.setattr(activity, "default_ask", fake)
    footage["v"] = fake_clip("glacier-sled", "puddle-hop")  # no clear match: discovery
    activity.resolve(tmp_path, llm=activity.AUTO)  # analyze: discovery nobody asked for
    activity.resolve(tmp_path, llm=activity.AUTO, refresh=True, ollama_url="http://h:1")
    activity.resolve(tmp_path, llm=activity.AUTO, refresh=True, context="a lake")
    activity.resolve(tmp_path, refresh=True)  # cutpoint profile / crashes
    assert got == [{"quiet": True}, {"ollama_url": "http://h:1"}, {}, {}]


def test_low_memory_during_quiet_discovery_falls_back(
    tmp_path, monkeypatch, footage, fake_clip, caplog
):
    """Too little free memory for the vision model must not stop analyze: the profile is
    found as without a model (source fallback), with one INFO line."""

    def ask(*_a, **_k):
        raise LowMemoryError("not enough memory", "1.2 GB RAM free, 3.5 GB needed")

    footage["v"] = fake_clip("glacier-sled", "puddle-hop")  # no clear match: discovery
    without = activity.resolve(tmp_path, llm=False)
    monkeypatch.setattr(activity, "default_ask", lambda **kw: (ask, "m"))
    with caplog.at_level("INFO"):
        rec = activity.resolve(tmp_path, llm=activity.AUTO, refresh=True)
    assert rec["source"] == "fallback" and rec["profile"] == without["profile"]
    assert "description" not in rec
    lines = [r for r in caplog.records if "without the vision model" in r.getMessage()]
    assert [r.levelname for r in lines] == ["INFO"]
    with pytest.raises(LowMemoryError):  # asked for (profile, crashes): the run stops
        activity.resolve(tmp_path, refresh=True)


def test_resolve_hands_the_ollama_flags_to_default_ask(tmp_path, monkeypatch, footage):
    got: list[dict] = []

    def fake(**kw):
        got.append(kw)
        return None, None

    monkeypatch.setattr(activity, "default_ask", fake)
    activity.resolve(tmp_path, context="a lake", ollama_url="http://h:1", ollama_model="m")
    activity.resolve(tmp_path, context="a lake", refresh=True)
    assert got == [{"ollama_url": "http://h:1", "ollama_model": "m"}, {}]


# ---------------------------------------------------------------- per-footage discovery cache
def _synths(ask) -> int:
    """How many synthesis prompts ``ask`` (a :func:`_stub`) was sent."""
    return sum(p.startswith("You write detection profiles") for p in ask.calls)


@pytest.fixture
def same_footage(tmp_path: Path, footage, fake_clip):
    """``make(name)``: a new project over the same source file; its keyframes match no
    library profile clearly, so every resolve discovers."""
    src = tmp_path / "ride.mp4"
    src.write_bytes(b"footage")
    footage["v"] = fake_clip("glacier-sled", "puddle-hop")

    def make(name: str) -> Path:
        project = tmp_path / name
        project.mkdir()
        hl = {"sources": [{"path": str(src), "duration": 60.0}]}
        (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
        return project

    return make


def _cached_discoveries() -> list[Path]:
    folder = paths.cache_dir() / activity.DISCOVERY_DIR
    return sorted(folder.glob("*.json")) if folder.is_dir() else []


def test_the_same_footage_is_synthesized_once_across_projects_and_runs(same_footage):
    """#932: a re-made project over the same sources reuses the discovered profile."""
    good = _stub(PACK_RAFTING)
    first = activity.resolve(same_footage("a"), ask=good)
    again = activity.resolve(same_footage("b"), ask=good)
    third = activity.resolve(same_footage("c"), ask=good, context="  ")  # blank: no context
    assert _synths(good) == 1
    assert first["source"] == again["source"] == third["source"] == "synthesized"
    assert first["profile"] == again["profile"] == third["profile"]
    assert first["description"] == again["description"]
    assert [n for n in activity.user_files() if n.startswith("pack")] == ["pack-rafting"]
    assert len(_cached_discoveries()) == 1


def test_a_context_run_over_the_same_footage_reuses_the_saved_discovery(
    same_footage, footage, fake_clip
):
    """#932: project 2 matches the profile project 1 saved; with --context it must not
    synthesize a second one (the seed would change the examples and the cache key)."""
    good = _stub(PACK_RAFTING)
    first = activity.resolve(same_footage("a"), ask=good, context="a river trip")
    footage["v"] = fake_clip("pack-rafting")  # the library now holds the discovery: a match
    again = activity.resolve(same_footage("b"), ask=good, context="a river trip")
    assert _synths(good) == 1
    assert first["source"] == again["source"] == "synthesized"
    assert first["profile"] == again["profile"]
    assert [n for n in activity.user_files() if n.startswith("pack")] == ["pack-rafting"]


def test_a_changed_context_a_refresh_or_another_model_synthesizes_again(
    same_footage, footage, fake_clip, monkeypatch: pytest.MonkeyPatch
):
    good = _stub(PACK_RAFTING)
    activity.resolve(same_footage("a"), ask=good, context="a river trip")
    activity.resolve(same_footage("b"), ask=good, context="a  river   trip")  # same words
    assert _synths(good) == 1
    activity.resolve(same_footage("c"), ask=good, context="a lake")
    assert _synths(good) == 2
    activity.resolve(same_footage("d"), ask=good, context="a lake")
    assert _synths(good) == 2
    # --refresh-profile: synthesized again, and the new profile replaces the cached one
    # (also once the footage matches the profile the first run saved to the library)
    footage["v"] = fake_clip("pack-rafting")
    other = _stub({**PACK_RAFTING, "description": "Rafting, written again."})
    rec = activity.resolve(same_footage("e"), ask=other, context="a lake", refresh=True)
    assert _synths(other) == 1 and rec["profile"]["description"] == "Rafting, written again."
    rec = activity.resolve(same_footage("f"), ask=good, context="a lake")
    assert _synths(good) == 2 and rec["profile"]["description"] == "Rafting, written again."
    # another model (or the same name with another digest) writes its own (footage that matches
    # no library profile clearly, so each run discovers again)
    footage["v"] = fake_clip("glacier-sled", "puddle-hop")
    models = iter(["m1", "m1", "m2"])
    monkeypatch.setattr(activity, "default_ask", lambda **kw: (good, next(models)))
    for name in ("g", "h", "i"):
        activity.resolve(same_footage(name))
    assert _synths(good) == 4  # m1 once, m2 once
    # one footage keeps one library profile: the refresh and the models rewrote the same file
    assert [n for n in activity.user_files() if n.startswith("pack")] == ["pack-rafting"]


def test_a_failed_discovery_is_cached_until_a_refresh(same_footage, caplog, monkeypatch):
    """#916: the LLM is not asked again for footage it gave nothing usable for."""
    broken = _stub({})
    rec = activity.resolve(same_footage("a"), ask=broken)
    assert rec["source"] == "fallback" and _synths(broken) == 2  # the retry asks for less
    assert rec["discovery"]["attempts"] == 2 and rec["discovery"]["cached"] is False
    (entry,) = _cached_discoveries()
    assert json.loads(entry.read_text(encoding="utf-8"))["failed"] == "no usable profile"
    good = _stub(PACK_RAFTING)
    with caplog.at_level(logging.INFO, logger="cutpoint.activity"):
        rec = activity.resolve(same_footage("b"), ask=good)
    assert rec["source"] == "fallback" and rec["discovery"] == {"cached": True}
    assert not good.calls  # not even the description
    assert "--refresh-profile asks it again" in caplog.text
    # another model is asked
    monkeypatch.setattr(activity, "default_ask", lambda **kw: (good, "m2"))
    assert activity.resolve(same_footage("c"))["source"] == "synthesized"
    # --refresh-profile asks again, and the profile replaces the failure
    rec = activity.resolve(same_footage("d"), ask=good, refresh=True)
    assert rec["source"] == "synthesized" and _synths(good) == 2
    assert activity.resolve(same_footage("e"), ask=good)["source"] == "synthesized"
    assert _synths(good) == 2


def test_footage_without_reachable_sources_is_not_cached(tmp_path: Path, footage, fake_clip):
    footage["v"] = fake_clip("glacier-sled", "puddle-hop")
    good = _stub(PACK_RAFTING)
    activity.resolve(tmp_path, ask=good)  # no highlights.json: nothing to key it by
    activity.resolve(tmp_path, ask=good, refresh=True)
    assert _synths(good) == 2 and not _cached_discoveries()


def test_a_broken_cached_discovery_is_synthesized_again(same_footage):
    good = _stub(PACK_RAFTING)
    activity.resolve(same_footage("a"), ask=good)
    (entry,) = _cached_discoveries()
    v = activity.DISCOVERY_VERSION
    old = json.loads(entry.read_text(encoding="utf-8")) | {"version": v - 1}
    for junk in ("[1]", "{not json", json.dumps({"version": v, "profile": {"name": "x"}})):
        entry.write_text(junk, encoding="utf-8")
        rec = activity.resolve(same_footage(f"p{len(good.calls)}"), ask=good)
        assert rec["source"] == "synthesized"
    entry.write_text(json.dumps(old), encoding="utf-8")  # an older DISCOVERY_VERSION
    assert activity.resolve(same_footage("old"), ask=good)["source"] == "synthesized"
    assert _synths(good) == 5


def test_the_discovery_key_follows_the_prompts_and_the_examples(same_footage, monkeypatch):
    project = same_footage("a")
    model = {"model": "m", "digest": "sha256:1"}
    examples = [activity.builtin(k) for k in activity.BUILTIN_NAMES]
    key = activity.discovery_key(project, model, None, examples)
    assert key == activity.discovery_key(same_footage("b"), model, None, examples)
    assert key != activity.discovery_key(project, {**model, "digest": "sha256:2"}, None, examples)
    assert key != activity.discovery_key(project, model, None, [activity.library()["puddle-hop"]])
    monkeypatch.setattr(activity, "SYNTH_SIZES", ((8, 20, 4),))
    assert key != activity.discovery_key(project, model, None, examples)
    monkeypatch.undo()
    monkeypatch.setattr(activity, "DISCOVERY_SEED", activity.DISCOVERY_SEED + 1)
    assert key != activity.discovery_key(project, model, None, examples)  # #947: sampling
    monkeypatch.undo()
    monkeypatch.setattr(activity, "_DESCRIBE_SCHEMA", {"type": "object"})
    assert key != activity.discovery_key(project, model, None, examples)  # #947: schema


@pytest.mark.parametrize(
    ("name", "description", "expected"),
    [
        ("ski-crash", "", "ski"),
        ("mountain-bike-crash", "", "mountain-bike"),
        ("wipe-out-surfing", "", "surfing"),
        ("mtb-over-the-bars", "", "mtb"),  # the profile's own hyphenated phrase
        ("tumbling", "activity: Trail Running; setting: woods", "trail-running"),
        ("crash", "activity: a crash; setting: woods", activity.DISCOVERED),
        ("", "activity: Trail Running", "trail-running"),  # validate's "custom" placeholder
        ("", "", activity.DISCOVERED),
        ("bumper-cars", "", "bumper-cars"),  # "bump*", but its normal captions say "bumper"
        ("bumper-crash", "", "bumper"),
        ("ski-crash", "", "ski"),  # a floor word, though a normal caption says "crash"
        ("trail-hit-the-deck", "", "trail"),  # own phrase: "the" of a normal caption
    ],
)
def test_a_discovered_name_says_the_activity_not_the_incident(name, description, expected):
    """#916: crash words of the floor and of the profile's own are no part of a name,
    unless the profile's normal-activity captions use them."""
    raw = {
        "name": name,
        "detect": ["a crash on a trail", "a first-person view of a trail"],
        "incident": {
            "pos": ["a fall"],
            "neg": ["a bumper car gliding", "skiing without a crash", "riding down the trail"],
            "words": {
                "bump*": 0.5,
                "over-the-bars": 0.7,
                "over the bars": 0.7,
                "hit the deck": 0.7,
            },
        },
    }
    prof = activity.synthesize(description, None, _stub(raw), [])
    assert prof["name"] == expected
    assert prof["detect"] == ["a first-person view of a trail"]  # no incident captions


def test_without_a_crash_free_detect_caption_the_name_is_one():
    """#916: every ``detect`` caption and the description name a crash: neither is kept."""
    raw = {
        "name": "ski-crash",
        "description": "Skiers who crash.",
        "detect": ["a skier crashing"],
        "incident": {"pos": ["a fall"]},
    }
    prof = activity.synthesize("", None, _stub(raw), [])
    assert prof["detect"] == ["a first-person video of ski"]
    nameless = {**raw, "name": "crash"}  # nothing names it: no detect caption at all
    assert activity.synthesize("", None, _stub(nameless), [])["detect"] == []
    raw = {**raw, "description": "Skiing in deep snow."}
    assert activity.synthesize("", None, _stub(raw), [])["detect"] == ["Skiing in deep snow."]


def test_only_unusable_replies_and_timeouts_are_remembered(same_footage, monkeypatch):
    """A quick None is a request error (Ollama down or loading): the next run asks again. A
    slow None (a timeout) or an unusable reply is cached."""
    down = _stub(None)
    rec = activity.resolve(same_footage("a"), ask=down)
    assert rec["source"] == "fallback" and _synths(down) == 2
    assert rec["discovery"]["answered"] == [False, False] and "failed" not in rec["discovery"]
    assert not _cached_discoveries()
    activity.resolve(same_footage("b"), ask=down)
    assert _synths(down) == 4  # asked again
    monkeypatch.setattr(activity, "PROFILE_TIMEOUT_S", 0)  # every attempt "took" the timeout
    rec = activity.resolve(same_footage("c"), ask=down)
    assert _synths(down) == 5 and rec["discovery"]["timed_out"] == [True]
    assert len(_cached_discoveries()) == 1
    activity.resolve(same_footage("d"), ask=down)
    assert _synths(down) == 5  # remembered


def test_the_retry_is_skipped_after_a_first_attempt_near_the_timeout(monkeypatch):
    monkeypatch.setattr(activity, "PROFILE_TIMEOUT_S", 0)
    broken, stats = _stub({}), {}
    assert activity.synthesize("kayaking", None, broken, [], stats) is None
    assert len(broken.calls) == 1 and stats["attempts"] == 1 and len(stats["synth_s"]) == 1
    assert "took" in stats["failed"]


def test_describe_shows_the_most_different_frames(monkeypatch, fake_clip):
    a, b, c = (fake_clip(f"caption:{k}") for k in "abc")
    emb = np.concatenate([a, a, a, a, b, b, a, c, a])  # mostly a, a little b and c
    assert activity._spread(emb, 3) == [0, 4, 7]  # from the frame nearest the mean
    assert len(activity._spread(emb, 8)) == 3  # copies are not shown twice
    refs = [(Path("v.mp4"), float(t)) for t in range(len(emb))]
    grabbed = _frames(monkeypatch, {})
    _desc, how = activity.describe(emb, refs, _stub({"activity": "x"}))
    assert how == "vl" and grabbed == [0.0, 4.0, 7.0]


def test_describe_asks_for_the_describe_schema(monkeypatch, fake_clip):
    """#947: a structured reply of short strings, so a greedy reply cannot run on."""
    a, b = fake_clip("caption:a"), fake_clip("caption:b")
    refs = [(Path("v.mp4"), float(t)) for t in range(4)]
    _frames(monkeypatch, {})
    seen: list[dict | None] = []

    def ask(prompt, images, schema):
        seen.append(schema)
        return {"activity": "x"}

    activity.describe(np.concatenate([a, a, b, b]), refs, ask)
    assert seen == [activity._DESCRIBE_SCHEMA]
    assert set(seen[0]["required"]) == {"activity", "setting", "camera", "surface", "hazards"}


def _jpeg(kind: str) -> bytes:
    """A JPEG frame: ``scene`` (textured), ``black`` (mean luma 2), ``flat`` (grey, std 6),
    or ``dark`` (underwater-like: mean about 37, textured)."""
    import cv2  # noqa: PLC0415

    rng = np.random.default_rng(0)
    img = {
        "scene": lambda: rng.integers(0, 256, (72, 128)),
        "black": lambda: np.full((72, 128), 2),
        "flat": lambda: 200 + rng.normal(0, 6, (72, 128)),
        "dark": lambda: rng.integers(5, 70, (72, 128)),
    }[kind]()
    return cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8))[1].tobytes()


def _frames(monkeypatch, kinds: dict[float, str]) -> list[float]:
    """Frame grabs return ``kinds[t]`` (``scene`` by default); the grabbed times are logged."""
    grabbed: list[float] = []
    monkeypatch.setattr(activity, "remap_source", lambda p: p)
    monkeypatch.setattr(activity, "GRAB_WORKERS", 1)  # the order of the grabs is the test's

    def grab(p, t, w):
        grabbed.append(t)
        return _jpeg(kinds.get(t, "scene"))

    monkeypatch.setattr(activity, "grab_frame_jpeg", grab)
    return grabbed


def _shown() -> tuple[list[bytes], activity.Ask]:
    """The images the vision model is shown, and the model."""
    shown: list[bytes] = []

    def spy(prompt, images, schema):
        shown.extend(images)
        return {"activity": "x"}

    return shown, spy


def test_describe_never_shows_black_or_flat_frames(monkeypatch, fake_clip):
    """#916: a black frame (cut to black, lens cap) or a flat one (white-out snow) is as far
    from every other keyframe as a frame gets, so the spread picked it; it names nothing."""
    a, b, c, d, e, f = (fake_clip(f"caption:{k}") for k in "abcdef")
    emb = np.concatenate([a, a, b, a, c, a, d, a, e, a, f])
    refs = [(Path("v.mp4"), float(t)) for t in range(len(emb))]
    kinds = {2.0: "black", 4.0: "flat", 6.0: "dark"}
    _frames(monkeypatch, kinds)
    shown, ask = _shown()
    _desc, how = activity.describe(emb, refs, ask)
    assert how == "vl" and len(shown) == activity.DESCRIBE_FRAMES
    assert _jpeg("black") not in shown and _jpeg("flat") not in shown
    assert _jpeg("dark") in shown  # dark but textured (underwater, night) is footage


def test_describe_shows_the_usable_frames_when_too_few_are_left(monkeypatch, fake_clip):
    emb = np.concatenate([fake_clip(f"caption:{k}") for k in "abcdefghi"])
    refs = [(Path("v.mp4"), float(t)) for t in range(len(emb))]
    grabbed = _frames(monkeypatch, {float(t): "black" for t in range(6)})  # 3 usable
    shown, ask = _shown()
    _desc, how = activity.describe(emb, refs, ask)
    assert how == "vl" and shown == [_jpeg("scene")] * 3
    assert grabbed[-3:] == [6.0, 7.0, 8.0] and len(grabbed) == len(set(grabbed))


def test_describe_shows_two_keyframes_once_each(monkeypatch, fake_clip):
    refs = [(Path("v.mp4"), 0.0), (Path("v.mp4"), 1.0)]
    two = np.concatenate([fake_clip(f"caption:{k}") for k in "ab"])
    no_emb = np.zeros((0, 768), np.float32)  # embeddings that do not match the keyframes
    for emb, kinds in ((no_emb, {}), (two, {}), (two, {0.0: "black", 1.0: "black"})):
        grabbed = _frames(monkeypatch, kinds)
        shown, ask = _shown()
        activity.describe(emb, refs, ask)
        assert grabbed == [0.0, 1.0] and len(shown) == 2  # all black: still both, once each


def test_describe_reads_a_bounded_number_of_frames(monkeypatch, fake_clip):
    """A recording that starts with 30 s of black: every black keyframe looks alike to CLIP,
    so one read rules them all out."""
    black = fake_clip("caption:black")
    rest = [fake_clip(f"caption:scene{k}") for k in range(10)]
    emb = np.concatenate([black] * 30 + rest)
    refs = [(Path("v.mp4"), float(t)) for t in range(len(emb))]
    grabbed = _frames(monkeypatch, {float(t): "black" for t in range(30)})
    shown, ask = _shown()
    _desc, how = activity.describe(emb, refs, ask)
    assert how == "vl" and len(shown) == activity.DESCRIBE_FRAMES
    assert _jpeg("black") not in shown
    assert len(grabbed) <= activity.DESCRIBE_READS + activity.DESCRIBE_FRAMES


def test_describe_stops_reading_after_the_read_budget(monkeypatch, fake_clip):
    """Black frames CLIP tells apart are ruled out one read at a time; the reads stop at
    the budget and the frames are spread in time."""
    emb = np.concatenate([fake_clip(f"caption:k{k}") for k in range(40)])
    refs = [(Path("v.mp4"), float(t)) for t in range(len(emb))]
    grabbed = _frames(monkeypatch, {float(t): "black" for t in range(30)})
    shown, ask = _shown()
    _desc, how = activity.describe(emb, refs, ask)
    assert how == "vl" and len(shown) == activity.DESCRIBE_FRAMES
    assert len(grabbed) <= activity.DESCRIBE_READS + activity.DESCRIBE_FRAMES
    assert len(grabbed) < 30  # not every black keyframe read


def _near(x: np.ndarray, y: np.ndarray, cos: float) -> np.ndarray:
    """A unit vector at cosine ``cos`` to ``x``, towards ``y`` (orthogonal unit vectors)."""
    return cos * x + np.sqrt(1 - cos**2) * y


def test_a_good_frame_like_a_blank_one_is_still_shown(monkeypatch, fake_clip):
    """A dark but detailed frame read in the same round as a flat one CLIP finds alike
    (cosine 0.97) is not dropped with it."""
    a, b, x, y = (fake_clip(f"caption:{k}") for k in "abxy")
    emb = np.concatenate([a, x, _near(x, y, 0.97), b])
    refs = [(Path("v.mp4"), float(t)) for t in range(len(emb))]
    _frames(monkeypatch, {1.0: "dark", 2.0: "flat"})
    shown, ask = _shown()
    activity.describe(emb, refs, ask)
    assert _jpeg("dark") in shown and _jpeg("flat") not in shown


def test_describe_ends_whatever_the_likeness_threshold(monkeypatch, fake_clip):
    monkeypatch.setattr(activity, "BLANK_LIKE", 1.5)  # no frame is like another
    emb = np.concatenate([fake_clip(f"caption:{k}") for k in "abcdef"])
    emb[5] = 0.0  # a keyframe without an embedding
    refs = [(Path("v.mp4"), float(t)) for t in range(len(emb))]
    _frames(monkeypatch, {0.0: "black", 5.0: "black"})
    shown, ask = _shown()
    _desc, how = activity.describe(emb, refs, ask)
    assert how == "vl" and _jpeg("black") not in shown


def test_incident_captions_that_fit_the_whole_footage_are_dropped(same_footage, footage, fake_clip):
    """#916: an LLM crash caption that is the best caption of much of this footage
    describes the footage, not a crash (confirmed crashes on ordinary rides)."""
    calm = "a person paddling a packraft"
    raw = {
        **PACK_RAFTING,
        "incident": {**PACK_RAFTING["incident"], "pos": [calm, "a flipped packraft"]},
    }
    footage["v"] = np.concatenate(
        [fake_clip(f"caption:{calm}")] * 3
        + [fake_clip("caption:a flipped packraft")]
        + [fake_clip("glacier-sled", "puddle-hop")] * 16
    )
    rec = activity.resolve(same_footage("a"), ask=_stub(raw))
    assert rec["source"] == "synthesized"
    pos = rec["profile"]["incident"]["pos"]
    assert "a flipped packraft" in pos and calm not in pos  # 1 of 20 kept, 3 of 20 dropped
    assert set(activity.base()["incident"]["pos"]) <= set(pos)  # never empty
    d = rec["discovery"]
    assert d["cached"] is False and d["attempts"] == 1 and len(d["synth_s"]) == 1
    assert d["describe_s"] >= 0
    # every LLM caption dropped: the floor's stay, and the cached discovery is still usable
    footage["v"] = np.concatenate([fake_clip(f"caption:{c}") for c in raw["incident"]["pos"]])
    good = _stub(raw)
    rec = activity.resolve(same_footage("b"), ask=good, context="river")
    assert rec["profile"]["incident"]["pos"] == activity.base()["incident"]["pos"]
    rec = activity.resolve(same_footage("c"), ask=good, context="river")
    assert _synths(good) == 1 and rec["discovery"] == {"cached": True}


def test_identical_discoveries_share_one_library_file(home):
    prof = activity.validate(PACK_RAFTING)
    assert activity.save_to_library(prof) == activity.save_to_library(prof) == "pack-rafting"
    other = activity.validate({**PACK_RAFTING, "description": "another one"})
    second = activity.save_to_library(other)  # the name is taken by other content
    assert second.startswith("pack-rafting-")
    assert activity.save_to_library(other) == second
    assert set(activity.user_files()) == {"pack-rafting", second}


def test_an_unreadable_library_file_is_never_overwritten(home, caplog):
    """A corrupt file with the profile's name stays; the profile gets the next free name."""
    prof = activity.validate(PACK_RAFTING)
    bad = paths.profiles_dir() / "pack-rafting.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("[1, 2", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="cutpoint.activity"):
        name = activity.save_to_library(prof, ["ride.mp4:1"])
    assert name.startswith("pack-rafting-") and name != "pack-rafting"
    assert bad.read_text(encoding="utf-8") == "[1, 2"
    assert any(str(bad) in r.message for r in caplog.records)
    assert activity.save_to_library(prof, ["ride.mp4:1"]) == name  # its own file again
    # the suffixed name is unreadable too: the next free one is used
    again = paths.profiles_dir() / f"{name}.json"
    again.unlink()
    again.write_text("{", encoding="utf-8")
    third = activity.save_to_library(prof, ["other.mp4:2"])
    assert third not in ("pack-rafting", name) and again.read_text(encoding="utf-8") == "{"


def test_a_rediscovery_of_the_same_footage_overwrites_its_library_file(home):
    """#947: --refresh-profile rewrites the footage's profile, no ``<name>-<hash>`` copy."""
    prof = activity.validate(PACK_RAFTING)
    assert activity.save_to_library(prof, ["ride.mp4:1"]) == "pack-rafting"
    newer = activity.validate({**PACK_RAFTING, "description": "written again"})
    assert activity.save_to_library(newer, ["ride.mp4:1"]) == "pack-rafting"
    assert set(activity.user_files()) == {"pack-rafting"}
    saved = json.loads((paths.profiles_dir() / "pack-rafting.json").read_text(encoding="utf-8"))
    assert saved["description"] == "written again" and saved["footage"] == ["ride.mp4:1"]
    assert saved["source"] == "discovered" and saved["name"] == "pack-rafting"
    # other footage, or none: the name is taken by other content, so a copy is made
    other = activity.save_to_library(activity.validate({**PACK_RAFTING, "description": "c"}), ["x"])
    bare = activity.save_to_library(activity.validate({**PACK_RAFTING, "description": "d"}))
    assert other.startswith("pack-rafting-") and bare.startswith("pack-rafting-")
    assert other != bare and len(activity.user_files()) == 3


def test_a_rediscovery_under_a_new_name_renames_the_footages_library_file(home):
    """F1: one footage keeps one library profile, also when the new discovery names it differently."""
    assert (
        activity.save_to_library(activity.validate(PACK_RAFTING), ["ride.mp4:1"]) == "pack-rafting"
    )
    renamed = activity.validate({**PACK_RAFTING, "name": "kayak"})
    assert activity.save_to_library(renamed, ["ride.mp4:1"]) == "kayak"
    assert set(activity.user_files()) == {"kayak"}
    saved = json.loads((paths.profiles_dir() / "kayak.json").read_text(encoding="utf-8"))
    assert saved["name"] == "kayak" and saved["footage"] == ["ride.mp4:1"]
    assert saved["source"] == "discovered"
    # the new name belongs to another library profile: the file stays under its own stem
    activity.create_profile("canoe", HOP)
    taken = activity.validate({**PACK_RAFTING, "name": "canoe", "description": "again"})
    assert activity.save_to_library(taken, ["ride.mp4:1"]) == "kayak"
    assert set(activity.user_files()) == {"kayak", "canoe"}
    assert activity.library()["canoe"]["incident"]["pos"] == HOP["incident"]["pos"]


def test_a_profile_without_incident_negatives_gets_generics(home):
    generic = activity.builtin("generic")["incident"]["neg"]
    assert generic  # the floor has none of its own
    bare = activity.full(activity.validate({"name": "x", "incident": {"pos": ["a fall"]}}))
    assert bare["incident"]["neg"] == generic
    assert activity.prompts(bare).incident_neg == tuple(generic)
    own = activity.full(activity.validate(HOP))
    assert own["incident"]["neg"] == HOP["incident"]["neg"]  # its own are kept as they are
    # an activity.json written before: complete, but no negatives
    old = {"version": 1, "name": "x", "source": "synthesized", "profile": {**bare}}
    old["profile"]["incident"] = {**bare["incident"], "neg": []}
    assert activity.record_profile(old)["incident"]["neg"] == generic


# ---------------------------------------------------------------- the library commands
def test_create_validates_saves_by_hand_and_refuses_to_overwrite(home):
    path, warnings = activity.create_profile("torch-juggle", {**HOP, "junk": 1})
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert path == paths.profiles_dir() / "torch-juggle.json"
    assert saved["name"] == "torch-juggle" and saved["source"] == "hand"  # NAME wins
    assert warnings == ["unknown keys ignored: junk"]
    assert activity.library()["torch-juggle"]["incident"]["pos"] == HOP["incident"]["pos"]
    with pytest.raises(FileExistsError, match="--force"):
        activity.create_profile("torch-juggle", HOP)
    activity.create_profile("torch-juggle", SLED, force=True)
    assert activity.library()["torch-juggle"]["lift"]["pos"] == SLED["lift"]["pos"]


@pytest.mark.parametrize(
    ("name", "data", "match"),
    [
        ("generic", HOP, "built-in profile name"),
        ("Base", HOP, "built-in profile name"),
        ("a b", HOP, "not a usable profile name"),
        ("con", HOP, "Windows"),
        ("x", ["not", "an", "object"], "not a JSON object"),
        ("x", {"incident": {"pos": "a fall"}}, "incident.pos must be a list"),
        ("x", {"name": "empty"}, "no captions and no incident words"),
    ],
)
def test_create_refuses_bad_names_and_content(home, name, data, match):
    with pytest.raises(jsonout.ArgumentError, match=match):
        activity.create_profile(name, data)
    assert not activity.user_files()


def test_create_refuses_a_name_that_differs_only_in_case(home):
    activity.create_profile("Puddle-Hop", HOP)
    with pytest.raises(jsonout.ArgumentError, match="differ only in case"):
        activity.create_profile("puddle-hop", HOP)
    assert list(activity.user_files()) == ["Puddle-Hop"]


@pytest.mark.parametrize("name", ["con", "generic", "Glacier-Sled"])
def test_a_discovered_name_that_is_unusable_or_taken_gets_a_hash_suffix(home, name):
    add_profile(SLED)
    saved = activity.save_to_library(activity.validate({**HOP, "name": name}))
    assert saved.startswith(f"{name.lower()}-") and len(saved) == len(name) + 7
    assert activity.name_problem(saved) is None


def test_list_show_validate_and_delete(home, tmp_path):
    add_profile(HOP)
    activity.save_to_library(activity.validate(SLED))
    (paths.profiles_dir() / "broken.json").write_text("[1]", encoding="utf-8")
    entries = {e["name"]: e for e in activity.list_profiles()}
    assert list(entries) == ["base", "generic", "broken", "glacier-sled", "puddle-hop"]
    assert entries["base"]["floor"] and entries["base"]["source"] == "built-in"
    assert entries["glacier-sled"]["source"] == "discovered"
    assert entries["puddle-hop"]["source"] is None  # a file without one
    assert not entries["broken"]["valid"]
    shown = activity.show_profile("glacier-sled")
    assert shown["profile"]["lift"]["pos"] == SLED["lift"]["pos"]
    assert activity.show_profile("generic")["type"] == "built-in"
    with pytest.raises(FileNotFoundError):
        activity.show_profile("ski")
    outside = tmp_path / "mine.json"
    outside.write_text(json.dumps({**HOP, "vlm": {"tags": {"rapids": "x"}}}), encoding="utf-8")
    res = activity.validate_profile(str(outside))
    assert res["valid"] and res["type"] == "file"
    assert [w["code"] for w in res["warnings"]] == ["profile_entry_ignored"]
    assert not activity.validate_profile("broken")["valid"]
    assert activity.validate_profile("generic")["type"] == "built-in"
    with pytest.raises(jsonout.ArgumentError, match="cannot be deleted"):
        activity.delete_profile("base")
    assert activity.delete_profile("puddle-hop") == paths.profiles_dir() / "puddle-hop.json"
    assert "puddle-hop" not in activity.library()
    with pytest.raises(FileNotFoundError):
        activity.delete_profile("puddle-hop")


# -- the profile verbs on the command line --


@pytest.mark.parametrize(
    ("argv", "verb"),
    [
        (["profile", "list"], True),
        (["-v", "profile", "show", "x"], True),
        (["profile", "--json", "list"], False),  # a verb comes right after ``profile``
        (["profile", "myproject"], False),
        (["profile", "projects/list"], False),  # a project named like a verb
        (["profile"], False),
        (["crashes", "list"], False),
    ],
)
def test_profile_verbs_are_told_from_projects(argv, verb) -> None:
    from cutpoint import cli  # noqa: PLC0415

    assert cli._profile_verb(argv) is verb


def test_profile_verbs_on_the_command_line(home, tmp_path, capsys, monkeypatch) -> None:
    from conftest import run_json  # noqa: PLC0415
    from cutpoint import cli  # noqa: PLC0415

    doc_file = tmp_path / "sled.json"
    doc_file.write_text(json.dumps(SLED), encoding="utf-8")
    code, doc = run_json(capsys, ["profile", "create", "my-sled", "--from", str(doc_file)])
    assert code == 0 and doc["command"] == "profile create", doc
    assert doc["outputs"]["name"] == "my-sled" and doc["outputs"]["replaced"] is False
    code, doc = run_json(capsys, ["profile", "list"])
    assert code == 0 and doc["command"] == "profile list"
    rows = {p["name"]: p for p in doc["outputs"]["profiles"]}
    assert set(rows) == {"base", "generic", "my-sled"}
    assert rows["my-sled"]["source"] == activity.HAND
    code, doc = run_json(capsys, ["profile", "create", "my-sled", "--from", str(doc_file)])
    assert code == 1 and doc["ok"] is False  # exists: --force replaces it
    assert cli.main(["profile", "show", "my-sled"]) == 0
    assert capsys.readouterr().out.startswith("name: my-sled (your profile, source hand)\n")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(HOP)))  # --from -: stdin
    code, doc = run_json(capsys, ["profile", "create", "from-stdin", "--from", "-"])
    assert code == 0 and doc["outputs"]["name"] == "from-stdin"
    assert activity.library()["from-stdin"]["detect"] == HOP["detect"]
    monkeypatch.setattr("sys.stdin", io.StringIO("{not json"))
    code, doc = run_json(capsys, ["profile", "create", "bad", "--from", "-"])
    assert code == 1 and "standard input" in doc["error"]["message"]
    code, doc = run_json(capsys, ["profile", "delete", "my-sled"])
    assert code == 0 and doc["command"] == "profile delete"
    assert "my-sled" not in activity.library()
    code, doc = run_json(capsys, ["profile", "delete", "generic"])  # the built-ins stay
    assert code != 0 and doc["ok"] is False


def test_describe_grabs_the_frames_of_a_round_at_once_in_the_same_order(monkeypatch, fake_clip):
    """#946: the picks of a round are grabbed GRAB_WORKERS at a time; the frames, their order
    and the blank-skip are those of the one-at-a-time loop."""
    import threading  # noqa: PLC0415
    import time  # noqa: PLC0415

    emb = np.concatenate([fake_clip(f"caption:{k}") for k in "abcdefghijkl"])
    refs = [(Path("v.mp4"), float(t)) for t in range(len(emb))]
    monkeypatch.setattr(activity, "remap_source", lambda p: p)
    lock = threading.Lock()
    live = {"now": 0, "max": 0}

    def grab(p, t, w):
        with lock:
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
        time.sleep(0.05)
        with lock:
            live["now"] -= 1
        kind = "black" if t in (1.0, 5.0) else "scene"
        return _jpeg(kind) + int(t).to_bytes(2, "big")  # a trailing marker: JPEG ignores it

    monkeypatch.setattr(activity, "grab_frame_jpeg", grab)
    runs = {}
    for workers in (1, activity.GRAB_WORKERS):
        monkeypatch.setattr(activity, "GRAB_WORKERS", workers)
        live.update(now=0, max=0)
        shown, ask = _shown()
        activity.describe(emb, refs, ask)
        runs[workers] = (list(shown), live["max"])
    assert runs[1][1] == 1 and runs[activity.GRAB_WORKERS][1] > 1
    assert runs[1][0] == runs[activity.GRAB_WORKERS][0] and len(runs[1][0]) == 4


def test_describe_surfaces_a_failed_grab_from_the_pool(monkeypatch, fake_clip):
    from cutpoint.errors import FFmpegError  # noqa: PLC0415

    emb = np.concatenate([fake_clip(f"caption:{k}") for k in "abcdef"])
    refs = [(Path("v.mp4"), float(t)) for t in range(len(emb))]
    monkeypatch.setattr(activity, "remap_source", lambda p: p)

    def grab(p, t, w):
        if t == 3.0:
            raise FFmpegError("boom")
        return _jpeg("scene")

    monkeypatch.setattr(activity, "grab_frame_jpeg", grab)
    shown, ask = _shown()
    activity.describe(emb, refs, ask)  # the grab error falls back to CLIP, as before
    assert shown == []
