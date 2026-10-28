"""#290 stage 2, PR 1: the Grade protocol plumbing (``grade_from_entry``, per-clip grades in
``render_clips`` / ``write_manifest`` / ``_encode``, and the ``--force`` dropped-grade WARNING)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from conftest import media_info, write
from cutpoint import clips, encode, sources
from cutpoint.boundaries import SpeechSettings
from cutpoint.clips import ClipPlan, RenderResult, _entry
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
