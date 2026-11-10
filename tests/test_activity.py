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


def test_without_a_model_a_clear_vocabulary_term_names_the_profile_not_saved(
    tmp_path: Path, fake_clip, monkeypatch: pytest.MonkeyPatch
):
    emb = fake_clip(_vocab("kayaking"))
    monkeypatch.setattr(activity, "keyframes", lambda project, **kw: (emb, []))
    rec = activity.resolve(tmp_path, llm=False)
    assert (rec["name"], rec["source"], rec["described_by"]) == ("kayaking", "fallback", "clip")
    assert rec["profile"]["detect"] == ["a first-person video of kayaking"]
    # the floor's captions, judged against generic's negatives (the floor has none, #932)
    floor, generic = activity.base()["incident"], activity.builtin("generic")["incident"]
    assert rec["profile"]["incident"] == {**floor, "neg": generic["neg"]}
    assert rec["profile"]["incident"]["neg"]
    assert not activity.user_files()  # never saved
    # nothing clear: generic
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
    assert rec["source"] == "fallback" and rec["name"] in ("kayaking", "generic")


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
    # another model (or the same name with another digest) writes its own
    models = iter(["m1", "m1", "m2"])
    monkeypatch.setattr(activity, "default_ask", lambda **kw: (good, next(models)))
    for name in ("g", "h", "i"):
        activity.resolve(same_footage(name))
    assert _synths(good) == 4  # m1 once, m2 once


def test_a_fallback_is_never_cached(same_footage):
    broken = _stub({})
    rec = activity.resolve(same_footage("a"), ask=broken)
    assert rec["source"] == "fallback" and _synths(broken) == 2  # the retry asks for less
    assert not _cached_discoveries()
    good = _stub(PACK_RAFTING)
    assert activity.resolve(same_footage("b"), ask=good)["source"] == "synthesized"
    assert _synths(good) == 1


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
    for junk in ("[1]", "{not json", json.dumps({"version": 1, "profile": {"name": "x"}})):
        entry.write_text(junk, encoding="utf-8")
        rec = activity.resolve(same_footage(f"p{len(good.calls)}"), ask=good)
        assert rec["source"] == "synthesized"
    assert _synths(good) == 4


def test_the_discovery_key_follows_the_prompts_and_the_examples(same_footage, monkeypatch):
    project = same_footage("a")
    model = {"model": "m", "digest": "sha256:1"}
    examples = [activity.builtin(k) for k in activity.BUILTIN_NAMES]
    key = activity.discovery_key(project, model, None, examples)
    assert key == activity.discovery_key(same_footage("b"), model, None, examples)
    assert key != activity.discovery_key(project, {**model, "digest": "sha256:2"}, None, examples)
    assert key != activity.discovery_key(project, model, None, [activity.library()["puddle-hop"]])
    monkeypatch.setattr(activity, "SYNTH_SIZES", ((8, 20, 4, True),))
    assert key != activity.discovery_key(project, model, None, examples)


def test_identical_discoveries_share_one_library_file(home):
    prof = activity.validate(PACK_RAFTING)
    assert activity.save_to_library(prof) == activity.save_to_library(prof) == "pack-rafting"
    other = activity.validate({**PACK_RAFTING, "description": "another one"})
    second = activity.save_to_library(other)  # the name is taken by other content
    assert second.startswith("pack-rafting-")
    assert activity.save_to_library(other) == second
    assert set(activity.user_files()) == {"pack-rafting", second}


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
