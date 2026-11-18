"""ProjectStyle (#1085 A416): one resolver for the styles of a UI project.

The table pins what the seven helpers it replaced returned for each edge case (recorded from
them before the change): a missing or older highlights.json, ``--style auto``, an action-cam
style that fell back (``settings.style_fallback``), a re-rank (``settings.reranked``) before
and after the user's next pick, and a deleted user style with and without ``style_base``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from cutpoint import paths, stylefiles
from cutpoint.ui.analysis import ProjectStyle, _gaming_style

FALLBACK = {"requested": "action-cam", "style": "action", "reason": "no telemetry"}
RERANK = {"from": "gaming", "requested": "funny"}
AUTO_HL = {"style": "action", "settings": {"style_requested": "auto", "inputs": ["/x.mp4"]}}
FELL_BACK = {
    "style": "action",
    "settings": {"style_requested": "action-cam", "style_fallback": FALLBACK, "inputs": ["/x.mp4"]},
}
RERANKED = {
    "style": "funny",
    "settings": {"reranked": RERANK, "rerank_backup": "k1", "style_requested": "funny"},
}

# meta, hl -> inputs, chosen, resolved, reranked, ran, library(chosen), style(ran) as
# [name, base] (None: unknown), ran_deleted, gaming(resolved)
CASES: list[tuple[str, dict[str, Any], Any, tuple[Any, ...]]] = [
    ("nothing yet", {}, None, (None, None, None, None, None, None, None, False, False)),
    ("auto, no analysis", {"style": "auto"}, {}, (None, "auto", "auto", None, "auto", "auto",
                                                  None, False, False)),
    ("command line, older file", {}, {"style": "funny"},
     (None, "funny", "funny", None, "funny", "funny", ["funny", "funny"], False, False)),
    ("auto analysis", {"style": "auto"}, AUTO_HL,
     (["/x.mp4"], "auto", "action", None, "action", "auto", ["action", "action"], False, False)),
    ("auto on the command line", {}, AUTO_HL,
     (["/x.mp4"], "auto", "action", None, "action", "auto", ["action", "action"], False, False)),
    ("auto without a recorded style", {"style": ""},
     {"style": "", "settings": {"style_requested": "auto"}},
     (None, "auto", "", None, "", "auto", None, False, False)),
    ("auto without a recorded style, no meta", {},
     {"style": "", "settings": {"style_requested": "auto"}},
     (None, "auto", None, None, None, "auto", None, False, False)),
    ("action-cam fell back", {"style": "action-cam"}, FELL_BACK,
     (["/x.mp4"], "action-cam", "action", None, "action", "action-cam", ["action", "action"],
      False, False)),
    ("fell back on the command line", {}, FELL_BACK,
     (["/x.mp4"], "action-cam", "action", None, "action", "action-cam", ["action", "action"],
      False, False)),
    ("fell back, then another style chosen", {"style": "funny"}, FELL_BACK,
     (["/x.mp4"], "funny", "funny", None, "action", "funny", ["action", "action"], False, False)),
    ("fell back, no request recorded", {},
     {"style": "action", "settings": {"style_fallback": FALLBACK}},
     (None, "action", "action", None, "action", "action", ["action", "action"], False, False)),
    ("re-ranked over auto", {"style": "auto"}, RERANKED,
     (None, "funny", "funny", "funny", "funny", "funny", ["funny", "funny"], False, False)),
    ("picked after the re-rank", {"style": "action", "style_for": "k1"}, RERANKED,
     (None, "action", "action", None, "funny", "action", ["funny", "funny"], False, False)),
    ("picked before the re-rank", {"style": "action", "style_for": "k0"}, RERANKED,
     (None, "funny", "funny", "funny", "funny", "funny", ["funny", "funny"], False, False)),
    ("re-rank without a mark", {"style": "action", "style_for": "k1"},
     {"style": "funny", "settings": {"reranked": RERANK}},
     (None, "funny", "funny", "funny", "funny", "funny", ["funny", "funny"], False, False)),
    ("re-rank without a style", {},
     {"style": "", "settings": {"reranked": RERANK, "rerank_backup": "k1"}},
     (None, "", None, None, None, "", None, False, False)),
    ("deleted style with its base", {}, {"style": "gone", "style_base": "commentary"},
     (None, "gone", "gone", None, "gone", "commentary", ["commentary", "commentary"], True,
      False)),
    ("deleted style chosen, with its base", {"style": "gone"},
     {"style": "gone", "style_base": "commentary"},
     (None, "gone", "gone", None, "gone", "commentary", ["commentary", "commentary"], True,
      False)),
    ("deleted style of an older file", {}, {"style": "gone"},
     (None, "gone", "gone", None, "gone", 400, None, True, False)),
    ("deleted style of an older file, another chosen", {"style": "funny"}, {"style": "gone"},
     (None, "funny", "funny", None, "gone", "funny", None, True, False)),
    ("user style", {}, {"style": "mine", "settings": {"style_requested": "mine"}},
     (None, "mine", "mine", None, "mine", "mine", ["mine", "gaming"], False, True)),
    ("broken style file", {}, {"style": "broken"},
     (None, "broken", "broken", None, "broken", "broken", None, False, False)),
    ("picked inputs", {"inputs": ["/a.mp4", "/b.mp4"], "style": "gaming"}, None,
     (["/a.mp4", "/b.mp4"], "gaming", "gaming", None, "gaming", "gaming", ["gaming", "gaming"],
      False, True)),
    ("one picked source", {"source": "/s.mp4"}, {},
     (["/s.mp4"], None, None, None, None, None, None, False, False)),
]  # fmt: skip


@pytest.fixture
def library(home):
    """A user style ``mine`` (gaming based) and a broken style file ``broken``."""
    stylefiles.create("mine", base="gaming")
    (paths.styles_dir() / "broken.json").write_text("{}", encoding="utf-8")


@pytest.mark.parametrize(("meta", "hl", "want"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_project_style_resolves_each_case_as_before(library, meta, hl, want) -> None:
    ps = ProjectStyle.of(meta, hl)
    try:
        live: Any = ps.library(ps.chosen)
    except HTTPException as err:
        live = err.status_code
    s = ps.style(ps.ran)
    got = (
        ps.inputs,
        ps.chosen,
        ps.resolved,
        ps.reranked,
        ps.ran,
        live,
        None if s is None else [s.name, s.base],
        ps.ran_deleted,
        _gaming_style(ps.resolved, ps),
    )
    assert got == want
    assert ps.recorded == (hl or {}).get("style")


def test_style_looks_up_the_recorded_base_only_for_the_recorded_style(library) -> None:
    ps = ProjectStyle.of({}, {"style": "gone", "style_base": "commentary"})
    assert ps.style("other-gone") is None  # another deleted name has no recorded base
    assert ps.style("auto") is None and ps.style(None) is None
    assert ProjectStyle.of({}, None).style("gone") is None
