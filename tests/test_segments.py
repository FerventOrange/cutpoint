"""One rule set for a review's segments: the UI's validate_segments and the AI Chat's ops check
run every rule through cutpoint.segments, so they accept and refuse the same sequences."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from cutpoint import ui
from cutpoint.assistant import ops
from cutpoint.segments import MAX_SEGMENTS, check_segments

SRC = "D:/v/a.mp4"
OTHER = "D:/v/b.mp4"  # a project source whose duration is not known up front
GONE = "D:/v/gone.mp4"  # a project source that cannot be read


class FakeMedia:
    def duration(self, path: Any) -> float | None:
        return 50.0 if str(path).endswith("b.mp4") else None  # the UI remaps the path first


def row(**over: Any) -> dict[str, Any]:
    return {"id": "h001", "source": SRC, "start": 1.0, "end": 5.0, "label": "x", **over}


def via_ui(rows: Any) -> Any:
    entries = [
        {"path": SRC, "duration": 100.0},
        {"path": OTHER, "duration": None},
        {"path": GONE, "duration": None},
    ]
    try:
        return ui.validate_segments(rows, entries, FakeMedia())  # type: ignore[arg-type]
    except HTTPException as err:
        assert err.status_code == 400
        return err.detail


def via_ops(rows: Any) -> Any:
    durations = ops.Durations({SRC: 100.0, OTHER: None, GONE: None}, FakeMedia().duration)
    try:
        return ops._check(rows, durations, lambda sid: f"moment {sid}")
    except ops.OpsError as err:
        return str(err)


CALLERS = pytest.mark.parametrize("check", [via_ui, via_ops], ids=["ui", "ops"])


@CALLERS
def test_a_valid_sequence_is_cleaned_the_same(check) -> None:
    rows = [
        row(start=1.00049, end=5, label=" a\x00b \n"),
        row(id="h002", source=OTHER, exact=True, label=""),
        row(id="h003", exact=False),
    ]
    assert check(rows) == [
        {"id": "h001", "source": SRC, "start": 1.0, "end": 5.0, "label": "ab"},
        {
            "id": "h002",
            "source": OTHER,
            "start": 1.0,
            "end": 5.0,
            "label": "segment",
            "exact": True,
        },
        {"id": "h003", "source": SRC, "start": 1.0, "end": 5.0, "label": "x"},
    ]


@CALLERS
def test_a_missing_label_is_segment_and_the_end_may_pass_by_a_hair(check) -> None:
    bare = {"id": "a", "source": SRC, "start": 0, "end": 100.01}
    assert check([bare])[0]["label"] == "segment"
    assert check([row(label="x" * 200)])[0]["label"] == "x" * 200


# (rows, UI text): every rule; ops_text() gives the ops wording
BROKEN = [
    ([row(id=None)], "segment 1: id must be a unique string of 1-64 characters"),
    ([row(id="")], "segment 1: id must be a unique string of 1-64 characters"),
    ([row(id="x" * 65)], "segment 1: id must be a unique string of 1-64 characters"),
    ([row(), row()], "segment 2: id must be a unique string of 1-64 characters"),
    ([row(source="D:/other.mp4")], "segment 1: source is not part of this project"),
    ([row(source=7)], "segment 1: source is not part of this project"),
    ([row(start="1")], "segment 1: start must be a number"),
    ([row(start=True)], "segment 1: start must be a number"),
    ([row(end=float("nan"))], "segment 1: end must be a number"),
    ([row(end=None)], "segment 1: end must be a number"),
    ([row(start=-1)], "segment 1: needs 0 <= start and end at least 0.1s after start"),
    ([row(start=2, end=2.05)], "segment 1: needs 0 <= start and end at least 0.1s after start"),
    ([row(source=GONE)], "segment 1: source cannot be read"),
    ([row(end=100.02)], "segment 1: end 100.02s is past the source's 100s"),
    ([row(end=101)], "segment 1: end 101s is past the source's 100s"),
    ([row(source=OTHER, end=60)], "segment 1: end 60s is past the source's 50s"),
    ([row(label=5)], "segment 1: label must be text of at most 200 characters"),
    ([row(label="x" * 201)], "segment 1: label must be text of at most 200 characters"),
    ([row(exact=1)], "segment 1: exact must be true or false"),
    ([row(exact="yes")], "segment 1: exact must be true or false"),
]
OPS_TEXT = {
    "id": "moment {}: ids must be unique strings of 1-64 characters",
    "source": "moment h001: its source is not part of this project",
    "number": "moment h001: start and end must be numbers",
    "unreadable": "moment h001: its source cannot be read",
    "label": "moment h001: label must be text of at most 200 characters",
    "exact": "moment h001: exact must be true or false",
}


def ops_text(ui_text: str, rows: list[dict[str, Any]]) -> str:
    """The ops wording of a UI message (the same rule, other prefix)."""
    rest = ui_text.split(": ", 1)[1]
    sid = str(rows[min(int(ui_text.split()[1].rstrip(":")), len(rows)) - 1].get("id"))
    if rest.startswith("id must"):
        return OPS_TEXT["id"].format(sid)
    if rest.endswith("must be a number"):
        return OPS_TEXT["number"]
    if rest.startswith("source is not"):
        return OPS_TEXT["source"]
    if rest.startswith("source cannot"):
        return OPS_TEXT["unreadable"]
    if rest.startswith("needs 0"):
        r = rows[0]
        return f"moment h001: {rest} (got {r['start']:g}-{r['end']:g})"
    if rest.startswith("end "):
        return f"moment h001: {rest}"
    return OPS_TEXT["label" if rest.startswith("label") else "exact"]


@pytest.mark.parametrize(("rows", "text"), BROKEN)
def test_each_rule_refuses_in_both_callers_with_their_own_wording(rows, text) -> None:
    assert via_ui(rows) == text
    assert via_ops(rows) == ops_text(text, rows)


def test_the_limit_and_the_list_rule() -> None:
    many = [row(id=f"s{i}") for i in range(MAX_SEGMENTS + 1)]
    assert via_ui(many) == f"at most {MAX_SEGMENTS} segments"
    assert via_ops(many) == f"the sequence can hold at most {MAX_SEGMENTS} moments"
    assert len(via_ui(many[:-1])) == len(via_ops(many[:-1])) == MAX_SEGMENTS
    assert via_ui({"id": "a"}) == "segments must be a list"
    assert via_ui(["x"]) == "segment 1: object expected"


def test_the_first_broken_rule_wins_in_the_same_order() -> None:
    both = row(id="", source="D:/other.mp4", start=-1, label=5, exact=1)
    assert via_ui([both]).split(": ")[1].startswith("id")
    assert via_ops([both]).endswith("ids must be unique strings of 1-64 characters")
    # the duration is asked only after the times pass
    asked: list[str] = []

    def duration(path: str) -> float | None:
        asked.append(path)
        return 100.0

    def fail(rule: str, *_a: Any, **_k: Any) -> ValueError:
        return ValueError(rule)

    with pytest.raises(ValueError, match="range"):
        check_segments([row(start=-1)], has_source=lambda s: True, duration=duration, fail=fail)
    assert asked == []


def test_ops_refuse_a_bool_time_like_the_ui() -> None:
    assert via_ops([row(end=True)]) == "moment h001: start and end must be numbers"
