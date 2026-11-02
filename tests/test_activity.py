"""Activity profiles: ski regression, resolution order, LLM output validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import set_config
from cutpoint import activity, paths, vision


@pytest.fixture
def fake_clip(monkeypatch: pytest.MonkeyPatch):
    """One embedding axis per library profile (its detect captions); others elsewhere."""
    axes: dict[str, int] = {}

    def text_emb(prompts: tuple[str, ...]) -> np.ndarray:
        lib = activity.library()
        e = np.zeros((len(prompts), 768), np.float32)
        for k, s in enumerate(prompts):
            owner = next((n for n, p in lib.items() if s in p["detect"]), None)
            e[k, axes.setdefault(owner, len(axes)) if owner else 700 + len(s) % 60] = 1.0
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
    """Keyframes of the project: a clear ski match until the test sets ``emb["v"]``."""
    emb = {"v": fake_clip("ski")}
    monkeypatch.setattr(activity, "keyframes", lambda project, **kw: (emb["v"], []))
    return emb


PACK_RAFTING = {
    "name": "Pack Rafting",
    "detect": ["a packraft on a river"],
    "incident": {"pos": ["a flipped packraft"], "neg": ["paddling"], "words": {"swim": 1}},
}


def test_resolve_takes_a_clear_library_match(tmp_path: Path, footage):
    rec = activity.resolve(tmp_path, llm=False)
    assert (rec["name"], rec["source"]) == ("ski", "library")
    assert rec["profile"] == activity.full(activity.library()["ski"])


def test_resolve_reuses_the_cached_choice_and_the_flag_overrides_it(tmp_path: Path, footage):
    activity.resolve(tmp_path, llm=False)
    never = _stub({})
    assert activity.resolve(tmp_path, ask=never)["source"] == "library"  # cached
    assert not never.calls
    assert activity.resolve(tmp_path, activity="mtb")["source"] == "flag"


def _ambiguous_and_failing(tmp_path: Path, footage, fake_clip):
    """Ambiguous footage, the LLM fails (one retry): the nearest profile, retried next run."""
    footage["v"] = fake_clip("ski", "mtb")
    broken = _stub({})
    rec = activity.resolve(tmp_path, refresh=True, context="packrafting trip", ask=broken)
    return rec, broken


def test_failed_synthesis_falls_back_to_the_nearest_profile(tmp_path: Path, footage, fake_clip):
    rec, broken = _ambiguous_and_failing(tmp_path, footage, fake_clip)
    assert (rec["name"], rec["source"], len(broken.calls)) == ("ski", "fallback", 2)


def test_fallback_is_retried_by_a_plain_call_and_the_synthesized_profile_is_saved(
    tmp_path: Path, footage, fake_clip
):
    _ambiguous_and_failing(tmp_path, footage, fake_clip)
    good = _stub(PACK_RAFTING)
    rec = activity.resolve(tmp_path, ask=good)  # a plain call retries with the same context
    assert (rec["name"], rec["source"]) == ("pack-rafting", "synthesized")
    assert "packrafting trip" in good.calls[-1]
    assert "a flipped packraft" in rec["profile"]["incident"]["pos"]
    assert "you okay" in rec["profile"]["incident"]["words"]  # base floor
    assert (paths.profiles_dir() / "pack-rafting.json").exists()


def test_synthesized_profile_is_reused_by_the_next_similar_video(
    tmp_path: Path, footage, fake_clip
):
    footage["v"] = fake_clip("ski", "mtb")
    activity.resolve(tmp_path, refresh=True, context="packrafting trip", ask=_stub(PACK_RAFTING))
    footage["v"] = fake_clip("pack-rafting")
    rec = activity.resolve(tmp_path, refresh=True, llm=False)
    assert (rec["name"], rec["source"]) == ("pack-rafting", "library")


def test_context_on_a_named_profile_adds_the_llm_additions(tmp_path: Path, footage):
    """The profile plus the LLM's additions."""
    ski = activity.full(activity.library()["ski"])
    extra = {
        "name": "x",
        "incident": {"pos": ["a skier hitting a race gate"], "words": {"yard sale": 0.3}},
    }
    rec = activity.resolve(tmp_path, activity="ski", context="slalom race", ask=_stub(extra))
    assert (rec["name"], rec["source"]) == ("ski", "context")
    assert rec["profile"]["incident"]["pos"] == [
        *ski["incident"]["pos"],
        extra["incident"]["pos"][0],
    ]
    assert rec["profile"]["incident"]["words"]["yard sale"] == 1.0  # never re-weighted


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
                "down the trail": 1,  # filler only: dropped
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
    assert not {"huh", "foot down", "down the trail", "rider", "b*"} & set(w)
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
    rater = getattr(ask, "__self__", None)
    assert rater is not None and rater.via is None and rater.provider_name == "ollama"
