"""The Moments grid's kind (web/util.js kindOf) names words as AI Chat does (pacing.moment_kind)."""

import re
from pathlib import Path

from cutpoint.assistant import context, pacing
from cutpoint.assistant.context import _GENERIC

UTIL_JS = Path(__file__).resolve().parents[1] / "src" / "cutpoint" / "web" / "util.js"


def _js_words(name: str) -> set[str]:
    text = UTIL_JS.read_text(encoding="utf-8")
    m = re.search(rf"const {name} = \[([^\]]*)\];", text)
    assert m, f"{name} not found in util.js"
    return set(re.findall(r"'([^']+)'", m.group(1)))


def test_kind_word_sets_match_pacing():
    assert _js_words("SCENIC_WORDS") == set(pacing._SCENIC_WORDS)
    assert _js_words("FUNNY_WORDS") == set(pacing._FUNNY_WORDS)
    assert _js_words("ACTION_WORDS") == set(pacing._ACTION_WORDS)
    assert _js_words("TALK_WORDS") == set(pacing._TALK_WORDS)


def _js_list(name: str) -> list[str]:
    text = UTIL_JS.read_text(encoding="utf-8")
    m = re.search(rf"const {name} = \[([^\]]*)\];", text)
    assert m, f"{name} not found in util.js"
    return re.findall(r"'([^']+)'", m.group(1))


def test_signal_cues_match_context():
    assert _js_list("SIGNAL_CUES") == list(context.SIGNAL_CUES)


def test_cue_kinds_match_pacing():
    text = UTIL_JS.read_text(encoding="utf-8")
    m = re.search(r"const CUE_KIND = \{([^}]*)\};", text)
    assert m, "CUE_KIND not found in util.js"
    js = dict(re.findall(r"(\w+): '([^']+)'", m.group(1)))
    assert js == {k: v for k, v in pacing._CUE_KIND.items() if k in context.SIGNAL_CUES}
    assert re.search(rf"const CUE_STRONG = {pacing.CUE_STRONG};", text)


def test_generic_label_pattern_matches_pacing():
    text = UTIL_JS.read_text(encoding="utf-8")
    m = re.search(r"const GENERIC = /\^(.*)\$/;", text)
    assert m, "GENERIC not found in util.js"
    assert m.group(1) == _GENERIC.pattern


def test_a_moment_made_from_a_talk_stretch_is_dialogue_whatever_its_title_says():
    cand = {
        "label": "fast-run-again",
        "dialogue_title": "Fast Run Again",
        "signals": {"events": {"nominated": ["dialogue-funny"]}},
    }
    assert pacing.moment_kind("Fast Run Again", cand, {}, lambda: False) == pacing.DIALOGUE
    untitled = {"label": "dialogue-funny", "signals": {"events": {"nominated": ["dialogue-story"]}}}
    assert pacing.moment_kind("dialogue-funny", untitled, {}, lambda: False) == pacing.DIALOGUE
    assert "startsWith('dialogue-')" in UTIL_JS.read_text(encoding="utf-8")


def test_a_moment_a_stretch_only_widened_keeps_its_own_type():
    widened = {
        "label": "fast-run-again",  # the title's slug: "fast" and "run" are not its type
        "dialogue_title": "Fast Run Again",
        "widened_from": {"label": "laughter"},
        "signals": {"events": {}},
    }
    assert pacing.moment_kind("Fast Run Again", widened, {}, lambda: False) == pacing.FUNNY
    plain = {**widened, "widened_from": {"label": "ski-moment"}}
    assert pacing.moment_kind("Fast Run Again", plain, {}, lambda: False) == pacing.ACTION
    text = UTIL_JS.read_text(encoding="utf-8")
    assert "c.dialogue_title ? [c.widened_from?.label ?? c.label, c.vision_label]" in text
