"""Activity profiles: ski regression, resolution order, LLM output validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cutpoint import activity, paths


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


def test_resolution_order(tmp_path: Path, fake_clip, monkeypatch: pytest.MonkeyPatch):
    emb = {"v": fake_clip("ski")}
    monkeypatch.setattr(activity, "keyframes", lambda project, **kw: (emb["v"], []))
    ski = activity.full(activity.library()["ski"])

    rec = activity.resolve(tmp_path, llm=False)  # clear library match
    assert (rec["name"], rec["source"]) == ("ski", "library")
    assert rec["profile"] == ski

    never = _stub({})
    assert activity.resolve(tmp_path, ask=never)["source"] == "library"  # cached
    assert not never.calls
    assert activity.resolve(tmp_path, activity="mtb")["source"] == "flag"

    # Ambiguous footage, the LLM fails (one retry): the nearest profile, retried next run.
    emb["v"] = fake_clip("ski", "mtb")
    broken = _stub({})
    rec = activity.resolve(tmp_path, refresh=True, context="packrafting trip", ask=broken)
    assert (rec["name"], rec["source"], len(broken.calls)) == ("ski", "fallback", 2)

    # Synthesis from the stub LLM, saved to the library.
    new = {
        "name": "Pack Rafting",
        "detect": ["a packraft on a river"],
        "incident": {"pos": ["a flipped packraft"], "neg": ["paddling"], "words": {"swim": 1}},
    }
    good = _stub(new)
    rec = activity.resolve(tmp_path, ask=good)  # a plain call retries with the same context
    assert (rec["name"], rec["source"]) == ("pack-rafting", "synthesized")
    assert "packrafting trip" in good.calls[-1]
    assert "a flipped packraft" in rec["profile"]["incident"]["pos"]
    assert "you okay" in rec["profile"]["incident"]["words"]  # base floor
    assert (paths.profiles_dir() / "pack-rafting.json").exists()
    emb["v"] = fake_clip("pack-rafting")  # the next similar video reuses it
    rec = activity.resolve(tmp_path, refresh=True, llm=False)
    assert (rec["name"], rec["source"]) == ("pack-rafting", "library")

    # Context on top of a named profile: the profile plus the LLM's additions.
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
    assert len(inc["words"]) == activity.MAX_WORDS - 4
    w = inc["words"]
    assert (w["dab"], w["dabbed my foot"], w["w0"]) == (0.6, 0.9, 0.5)
    assert not {"huh", "foot down", "down the trail", "rider", "b*"} & set(w)
    assert "game" not in prof and "junk" not in prof  # LLM output: schema keys only
    assert activity.synthesize("x", None, _stub({"incident": {"pos": []}}), []) is None
