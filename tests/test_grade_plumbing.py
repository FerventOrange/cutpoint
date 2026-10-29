"""#290 stage 2, PR 1: the Grade protocol plumbing (``grade_from_entry``, per-clip grades in
``render_clips`` / ``write_manifest`` / ``_encode``, and the ``--force`` dropped-grade WARNING)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from conftest import media_info, set_config, write
from cutpoint import clips, color, encode, jsonout, sources
from cutpoint.boundaries import SpeechSettings
from cutpoint.clips import ClipPlan, RenderResult, _entry
from cutpoint.color import AutoGrade
from cutpoint.crop import Crop
from cutpoint.encode import Lut, OutputFormat
from cutpoint.events.tails import EventTailSettings
from cutpoint.ffmpeg_utils import VideoEncoder
from cutpoint.styles import get_style
from cutpoint.tail import TailResult, TailSettings

FMT = OutputFormat(1920, 1080, 30)


@dataclass(frozen=True)
class Warm:
    """A non-Lut grade: what a colour step would plug in."""

    name: str = "warm"

    def filter(self) -> str:
        return "eq=saturation=1.2"

    def to_json(self) -> dict[str, Any]:
        return {"color": {"name": self.name}}


def plan(i: int, tmp_path: Path, pano: bool = False) -> ClipPlan:
    src = tmp_path / "src.mp4"
    m = {"id": f"h{i:03d}", "source": str(src), "end": 10.0 + i, "score": 50}
    tail = TailResult(10.0 + i, 0.0, "none")
    return ClipPlan(
        f"{i:03d}_c.mp4", m, src, media_info(src), float(i), 10.0 + i, (float(i), 10.0 + i),
        tail, pano=pano,
    )  # fmt: skip


# -- grade_from_entry ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({}, None),
        ({"lut": ""}, None),
        ({"lut": None}, None),
        ({"lut": "/l/a.cube"}, Lut(Path("/l/a.cube"), 1.0)),
        ({"lut": "/l/a.cube", "lut_strength": 0.4}, Lut(Path("/l/a.cube"), 0.4)),
    ],
)
def test_grade_from_entry_reads_a_recorded_lut(entry, expected):
    assert encode.grade_from_entry(entry) == expected


def test_grade_from_entry_reads_an_automatic_grade_and_a_lut_wins():
    auto = AutoGrade("underwater", 0.5, black=0.05, white=0.9, mix=0.2)
    skipped = AutoGrade.skipped("auto", color.SKIP_LOG)
    assert encode.grade_from_entry(auto.to_json()) == auto
    assert encode.grade_from_entry(skipped.to_json()) == skipped  # recorded, no filter
    lut = Lut(Path("/l/a.cube"), 0.4)
    both = {"lut": "/l/a.cube", "lut_strength": 0.4} | auto.to_json()
    assert encode.grade_from_entry(both) == lut


# -- a non-Lut grade flows through the encode and the manifest -----------------------------------


def test_a_stand_in_grade_takes_the_place_of_a_lut_in_the_filter_and_the_manifest(
    monkeypatch, tmp_path
):
    calls: list[list[str]] = []
    monkeypatch.setattr(encode, "run", lambda args, **_kw: calls.append(args) or "")
    info = media_info(color_space="bt709")
    encode._encode(VideoEncoder("cpu", None), info.path, info, 1.0, 5.0, Path("o.mp4"), FMT,
                   False, None, 0.0, grade=Warm())  # fmt: skip
    graph = calls[0][calls[0].index("-filter_complex") + 1]
    assert "eq=saturation=1.2" in graph and "lut3d" not in graph

    entry = _entry(
        plan(1, tmp_path), RenderResult(tmp_path / "001_c.mp4", 11.0), 0.25, grade=Warm()
    )
    assert entry["color"] == {"name": "warm"} and "lut" not in entry


# -- render_clips and write_manifest -------------------------------------------------------------


@pytest.fixture
def seen(monkeypatch) -> dict[str, Any]:
    """Stub the encoder; ``seen`` maps each clip name to the grade it was given (and 'pano')."""
    got: dict[str, Any] = {}
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)

    def export(src, info, start, end, out, *args, **_kw):
        got[out.name] = args[-1]
        out.write_bytes(b"x")

    monkeypatch.setattr(clips, "export_clip_checked", export)
    return got


@pytest.mark.parametrize("workers", [1, 2])
def test_render_clips_gives_each_clip_its_own_grade(seen, tmp_path, workers):
    g = Warm()
    plans = [plan(i, tmp_path) for i in range(3)]
    clips.render_clips(
        plans, tmp_path, fmt=FMT, vertical=False, audio_fade=0.0,
        grades=[g, None, g], workers=workers,
    )  # fmt: skip
    assert seen == {"000_c.mp4": g, "001_c.mp4": None, "002_c.mp4": g}


def test_render_clips_hands_a_360_clip_its_grade(seen, tmp_path):
    got: dict[str, Any] = {}

    class Pano:
        def export(self, source, start, end, out, **kw):
            got.update(kw)
            out.write_bytes(b"x")
            return None, {}

    g = Warm()
    clips.render_clips(
        [plan(0, tmp_path, pano=True)], tmp_path, fmt=FMT, vertical=False, audio_fade=0.0,
        pano=Pano(), grades=[g],
    )  # fmt: skip
    assert got["grade"] is g and "lut" not in got


def _write_manifest(tmp_path: Path, **kw) -> list[dict]:
    plans = [plan(0, tmp_path), plan(1, tmp_path)]
    results = [RenderResult(tmp_path / p.name, p.end) for p in plans]
    clips.write_manifest(
        tmp_path, tmp_path, plans, results, order="segments", fmt=FMT, vertical=False,
        style=get_style("gaming"), tail=TailSettings(enabled=False),
        event_settings=EventTailSettings(enabled=False), padding=None,
        speech=SpeechSettings(enabled=False), audio_fade=0.25, crop=None, **kw,
    )  # fmt: skip
    return json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["clips"]


def test_write_manifest_records_each_clips_own_grade(tmp_path):
    lut = Lut(tmp_path / "look.cube", 0.5)
    first, second = _write_manifest(tmp_path, grades=[lut, None])
    assert (first["lut"], first["lut_strength"]) == (str(lut.path), 0.5)
    assert "lut" not in second and "lut_strength" not in second


def test_write_manifest_without_grades_records_none(tmp_path):
    entries = _write_manifest(tmp_path)
    assert all("lut" not in e and "color" not in e for e in entries)


# -- --force and a dropped grade -----------------------------------------------------------------


def _old_manifest(project: Path, clips_entry: dict, **top) -> None:
    folder = project / "clips"
    write(folder / "001_a.mp4")
    data = {"generator": "cutpoint", "clips": [{"file": "001_a.mp4", **clips_entry}]} | top
    write(folder / "manifest.json", json.dumps(data))


def _run(home, monkeypatch, *, force=True, lut=None, old: dict | str | None = None, **top):
    project = home / "projects" / "p"
    src = write(project / "src.mp4")
    seg = write(
        project / "seg.json",
        json.dumps([{"source": str(src), "start": 0.0, "end": 2.0, "label": "a"}]),
    )
    if isinstance(old, dict):
        _old_manifest(project, old, **top)
    elif isinstance(old, str):
        write(project / "clips" / "manifest.json", old)
    monkeypatch.setattr(clips, "probe", media_info)

    def export(src, info, start, end, out, *_a, **_kw):
        out.write_bytes(b"n")

    monkeypatch.setattr(clips, "export_clip_checked", export)
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)
    return clips.export_clips(project, segments=seg, force=force, lut=lut)


GRADED = {"lut": "/l/a.cube", "lut_strength": 1.0}


@pytest.mark.parametrize(
    ("old", "keys"),
    [
        (GRADED, "lut"),
        ({"color": {"name": "warm"}}, "color"),
        (GRADED | {"color": {"name": "warm"}}, "color, lut"),
    ],
    ids=["lut", "color", "both"],
)
def test_force_warns_when_the_replaced_clips_were_graded_and_this_run_is_not(
    home, monkeypatch, caplog, old, keys
):
    _run(home, monkeypatch, old=old)
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("were graded" in m and f"({keys})" in m and "1 clip(s)" in m for m in msgs)


def test_force_does_not_warn_with_a_lut_this_run(home, monkeypatch, caplog):
    _run(home, monkeypatch, old=GRADED, lut=Lut(Path("/l/a.cube"), 1.0))
    assert "were graded" not in caplog.text


def test_no_warning_without_force(home, monkeypatch, caplog):
    with pytest.raises(FileExistsError):
        _run(home, monkeypatch, force=False, old=GRADED)
    assert "were graded" not in caplog.text


@pytest.mark.parametrize("old", [{}, None], ids=["ungraded", "no-manifest"])
def test_no_warning_when_the_old_clips_were_not_graded(home, monkeypatch, caplog, old):
    _run(home, monkeypatch, old=old)
    assert "were graded" not in caplog.text


def test_no_warning_for_an_unreadable_manifest(tmp_path, caplog):
    write(tmp_path / "manifest.json", "{not json")
    clips._warn_dropped_grade(tmp_path)
    clips._warn_dropped_grade(tmp_path / "missing")
    assert "were graded" not in caplog.text


def test_a_foreign_manifest_still_stops_the_run_before_the_warning(home, monkeypatch, caplog):
    with pytest.raises(FileExistsError, match="may not be one cutpoint wrote"):
        _run(home, monkeypatch, old=GRADED, generator="other-tool")
    assert "were graded" not in caplog.text


@pytest.fixture
def dropped(monkeypatch) -> list[tuple[str, str]]:
    """The ``jsonout.warn`` calls (code, message) of ``_warn_dropped_grade``."""
    got: list[tuple[str, str]] = []
    monkeypatch.setattr(jsonout, "warn", lambda code, msg, files=(): got.append((code, msg)))
    return got


@pytest.mark.parametrize(
    ("entries", "again"),
    [
        ([{"lut": "/l/a.cube"}], "--lut"),
        ([AutoGrade("auto", black=0.05).to_json()], "--color"),
        ([{"lut": "/l/a.cube"}, AutoGrade("auto", black=0.05).to_json()], "--color or --lut"),
    ],
    ids=["lut", "color", "both"],
)
def test_dropped_grade_is_a_json_warning_naming_the_flags_to_give_again(
    tmp_path, dropped, entries, again
):
    write(tmp_path / "manifest.json", json.dumps({"clips": entries}))
    clips._warn_dropped_grade(tmp_path)
    ((code, message),) = dropped
    assert code == "grade_dropped" and f"give {again} again" in message


@pytest.mark.parametrize("skip", [color.SKIP_GOOD, color.SKIP_360, color.SKIP_LOG, color.VETO_CAST])
def test_skipped_color_records_are_not_counted_as_graded(tmp_path, caplog, dropped, skip):
    skipped = AutoGrade.skipped("auto", skip).to_json()
    write(tmp_path / "manifest.json", json.dumps({"clips": [skipped]}))
    clips._warn_dropped_grade(tmp_path)
    assert dropped == [] and "were graded" not in caplog.text

    graded = AutoGrade("auto", black=0.05).to_json()  # one graded clip among skipped ones
    write(tmp_path / "manifest.json", json.dumps({"clips": [skipped, graded, skipped]}))
    clips._warn_dropped_grade(tmp_path)
    assert "1 clip(s)" in dropped[0][1]


# -- _color_mode ---------------------------------------------------------------------------------

FUNNY, GAMING = get_style("funny"), get_style("gaming")
LUT = Lut(Path("/l/a.cube"), 1.0)


@pytest.mark.parametrize(
    ("cfg", "given", "lut", "style", "want", "level"),
    [
        ("auto", None, None, FUNNY, "auto", None),
        ("underwater", None, None, FUNNY, "underwater", None),
        ("off", None, None, FUNNY, "off", None),
        ("auto", None, LUT, FUNNY, "off", None),  # config never applies with a LUT
        ("auto", None, None, GAMING, "off", "INFO"),  # ... nor to gaming footage
        ("auto", "off", None, FUNNY, "off", None),  # a given mode wins over config
        ("off", "underwater", None, FUNNY, "underwater", None),
        ("off", "auto", None, GAMING, "auto", "WARNING"),  # explicit on gaming: allowed, warned
        ("off", "underwater", None, GAMING, "underwater", "WARNING"),
        ("auto", "off", None, GAMING, "off", None),
    ],
)
def test_color_mode_resolves_flag_config_lut_and_gaming(
    home, caplog, cfg, given, lut, style, want, level
):
    set_config(home, color=cfg)
    caplog.set_level(logging.INFO)
    assert clips._color_mode(given, lut, style) == want
    seen = {r.levelname for r in caplog.records if r.name == "cutpoint.clips"}
    assert seen == ({level} if level else set())
    if level:
        assert "gaming" in caplog.text


@pytest.mark.parametrize("given", ["auto", "underwater"])
def test_color_mode_on_gaming_footage_warns_on_json_too(home, monkeypatch, caplog, given):
    got: list[tuple[str, str]] = []
    monkeypatch.setattr(jsonout, "warn", lambda code, msg, files=(): got.append((code, msg)))
    assert clips._color_mode(given, None, GAMING) == given
    ((code, message),) = got
    assert code == "color_on_gaming" and message in caplog.text and "gaming" in message
    # underwater is made for footage shot underwater: a stronger warning, still graded
    stronger = ("may look worse", "--color auto or a LUT", "not_underwater")
    assert all((s in message) is (given == "underwater") for s in stronger)


@pytest.mark.parametrize(("cfg", "given"), [("auto", None), ("off", "auto")])
def test_color_mode_with_an_unknown_style_never_counts_as_gaming(home, caplog, cfg, given):
    # segments without highlights.json: the gaming fallback is not the footage's style
    set_config(home, color=cfg)
    caplog.set_level(logging.INFO)
    assert clips._color_mode(given, None, GAMING, known_style=False) == "auto"
    assert not [r for r in caplog.records if r.name == "cutpoint.clips"]


@pytest.mark.parametrize("given", ["auto", "underwater"])
def test_color_mode_refuses_a_given_mode_with_a_lut(home, given):
    with pytest.raises(jsonout.ArgumentError, match="cannot be combined with --lut"):
        clips._color_mode(given, LUT, FUNNY)


# -- _color_grades -------------------------------------------------------------------------------


def test_color_grades_skip_360_hint_log_and_sum_up(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    crop = Crop(1280, 720, 0, 0)
    plans = [plan(i, tmp_path, pano=i == 1) for i in range(5)]
    plans[0] = replace(plans[0], crop=crop)
    results = {
        0.0: AutoGrade("auto", black=0.05),
        2.0: AutoGrade.skipped("auto", color.SKIP_LOG),
        3.0: AutoGrade("auto", black=0.05, underwater_hint=True),
        4.0: AutoGrade.skipped("auto", color.SKIP_GOOD),
    }
    seen: list[tuple] = []

    def fake(src, info, start, end, mode, fmt, **kw):
        seen.append((start, mode, kw))
        return results[start]

    monkeypatch.setattr(clips, "auto_grade", fake)
    grades = clips._color_grades(plans, "auto", 0.6, FMT, True)

    assert [s[0] for s in seen] == [0.0, 2.0, 3.0, 4.0]  # the 360 clip is not measured
    assert all(m == "auto" and kw["strength"] == 0.6 and kw["vertical"] for _, m, kw in seen)
    assert [kw["crop"] for _, _, kw in seen] == [crop, None, None, None]
    assert grades[1].skip == color.SKIP_360 and grades[1].strength == 0.6
    lines = [r.getMessage() for r in caplog.records if r.name == "cutpoint.clips"]
    assert lines[1:] == [
        "Color: 001_c.mp4 not graded (360)",
        "Color: 002_c.mp4 not graded (log): looks like log/flat footage; use --lut with the "
        "camera's LUT",
        "Color: 004_c.mp4 not graded (good)",
        "Color: 003_c.mp4 look underwater; --color underwater corrects underwater colors "
        "(--color auto does not)",
        "Color auto: 2 of 5 clips graded; not graded: 360 1, log 1, good 1",
    ]


def test_color_grades_name_each_skipped_clip_with_its_code(tmp_path, monkeypatch, caplog):
    """One INFO line per clip not graded, its skip code as recorded; ``not_underwater`` points
    to ``--color auto``; a graded clip gets none."""
    caplog.set_level(logging.INFO)
    results = {
        0.0: AutoGrade.skipped("underwater", color.SKIP_NOT_UNDERWATER),
        1.0: AutoGrade("underwater", black=0.05, skip=color.VETO_CLIPPING),
        2.0: AutoGrade("underwater", black=0.05),
    }
    monkeypatch.setattr(clips, "auto_grade", lambda src, info, start, *a, **kw: results[start])
    clips._color_grades([plan(i, tmp_path) for i in range(3)], "underwater", 1.0, FMT, False)
    lines = [r.getMessage() for r in caplog.records if r.name == "cutpoint.clips"]
    assert lines[1:] == [
        "Color: 000_c.mp4 not graded (not_underwater): it does not look underwater; "
        "--color auto may suit it",
        "Color: 001_c.mp4 not graded (veto_clipped)",
        "Color underwater: 1 of 3 clips graded; not graded: not_underwater 1, veto_clipped 1",
    ]


def test_color_grades_name_no_underwater_clip_when_none_looks_it(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(clips, "auto_grade", lambda *a, **kw: AutoGrade("underwater", black=0.05))
    clips._color_grades([plan(0, tmp_path)], "underwater", 1.0, FMT, False)
    assert "look underwater" not in caplog.text and "Color underwater: 1 of 1" in caplog.text
