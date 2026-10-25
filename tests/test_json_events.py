"""``--json`` on events (#4).

No model, no GPU: ``run_events`` is a stub that writes events.json / events.txt, the rescore
and the ski re-analysis are stubs, and the failure cases stop before any layer runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import write
from cutpoint import analyze, cache, cli, crash
from cutpoint.events import cli as events_cli
from cutpoint.events import pipeline, scoring
from test_jsonout import _envelope


def _json(capsys, argv: list[str]) -> tuple[int, dict]:
    """Run ``cutpoint <argv> --json``: exit code and the one envelope on stdout."""
    code = cli.main([*argv, "--json"])
    return code, _envelope(capsys.readouterr().out)


def _codes(doc: dict) -> list[str]:
    return [w["code"] for w in doc["warnings"]]


@pytest.fixture
def project(home, tmp_path) -> Path:
    """Project ``ev`` with one reachable source and an analysis."""
    src = write(tmp_path / "talk.mp4", "x")
    project = home / "projects" / "ev"
    project.mkdir(parents=True)
    cache.save_json(project / "highlights.json", {"sources": [{"path": str(src)}]})
    return project


def _fake_run_events(monkeypatch, layers: list[str], stats: dict | None = None) -> None:
    """``run_events`` writes a small events.json and events.txt, as the real one does."""

    def run(project: Path, **kwargs):
        jpath, tpath = project / "events.json", project / "events.txt"
        cache.save_json(
            jpath,
            {
                "settings": {"layers": layers},
                "stats": stats or {},
                "events": [{"kind": "audio"}, {"kind": "ocr"}],
                "windows": [{"start": 0.0, "end": 15.0}],
            },
        )
        tpath.write_text("events\n", encoding="utf-8")
        return jpath, tpath

    monkeypatch.setattr(events_cli, "run_events", run)


def test_events_json(project, monkeypatch, capsys):
    stats = {
        "sheets": [str(project / "visual_check" / "beauty.jpg")],
        "discovery": {
            "game": {
                "proposal": str(project / "discovery" / "game" / "proposal.json"),
                "overlay": str(project / "discovery" / "game" / "overlay.jpg"),
                "preset": str(project / "hud_presets" / "game.json"),
                "installed": True,
                "regions": [],
                "elapsed_s": 1.0,
            }
        },
    }
    _fake_run_events(monkeypatch, ["audio", "ocr"], stats)

    def rescore(p: Path) -> int:  # the real one writes the backup once, then highlights.json
        cache.save_json(p / "highlights.pre-events.json", {})
        return 0

    monkeypatch.setattr(scoring, "rescore_highlights", rescore)
    code, doc = _json(capsys, ["events", "ev"])
    assert code == 0 and doc["ok"] is True and doc["command"] == "events"
    assert doc["project"] == str(project) and doc["warnings"] == []
    out = doc["outputs"]
    assert out["events"] == str(project / "events.json")
    assert out["events_txt"] == str(project / "events.txt")
    assert out["layers"] == ["audio", "ocr"]
    assert out["count"] == 2 and out["windows"] == 1
    assert out["highlights"] == str(project / "highlights.json")
    assert out["highlights_backup"] == str(project / "highlights.pre-events.json")
    assert "crashes" not in out
    assert out["sheets"] == stats["sheets"]
    assert out["discovery"] == {
        "game": {k: stats["discovery"]["game"][k] for k in ("proposal", "overlay", "preset")}
        | {"installed": True}
    }


def test_events_json_no_rescore(project, monkeypatch, capsys):
    _fake_run_events(monkeypatch, ["audio"])
    cache.save_json(project / "highlights.pre-events.json", {})  # from an earlier run
    assert cli.main(["events", "ev", "--no-rescore"]) == 0  # without --json: nothing on stdout
    assert capsys.readouterr().out == ""
    code, doc = _json(capsys, ["events", "ev", "--no-rescore", "--layers", "audio"])
    assert code == 0 and doc["ok"] is True
    assert doc["outputs"]["highlights"] is None and "highlights_backup" not in doc["outputs"]


def test_events_json_ski_rerank_skipped(project, monkeypatch, capsys):
    hl = cache.load_json(project / "highlights.json")
    cache.save_json(project / "highlights.json", hl | {"style_kind": "ski"})
    _fake_run_events(monkeypatch, ["audio", "voice", "visual", "crash", "vlm"])
    monkeypatch.setattr(analyze, "reanalyze_if_reachable", lambda project: None)
    code, doc = _json(capsys, ["events", "ev"])
    assert code == 0 and doc["ok"] is True
    assert _codes(doc) == ["highlights_not_updated"]
    assert doc["warnings"][0]["files"] == [str(project / "highlights.json")]
    out = doc["outputs"]
    assert out["highlights"] is None and out["crashes"] == str(project / "crashes.json")


def test_events_json_missing_extra(project, monkeypatch, capsys):
    monkeypatch.setattr(pipeline.importlib.util, "find_spec", lambda name: None)
    code, doc = _json(capsys, ["events", "ev", "--layers", "audio"])
    assert code == 1 and doc["ok"] is False
    assert doc["error"]["code"] == "missing_extra"
    assert "'events' extra" in doc["error"]["message"]
    assert doc["project"] == str(project) and doc["outputs"] == {}


def test_events_json_footage_unreachable(project, tmp_path, monkeypatch, capsys):
    gone = tmp_path / "gone.mp4"
    cache.save_json(project / "highlights.json", {"sources": [{"path": str(gone)}]})
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    code, doc = _json(capsys, ["events", "ev", "--layers", "audio"])
    assert code == 1 and doc["ok"] is False and doc["error"]["code"] == "error"
    assert "No footage of ev is reachable" in doc["error"]["message"]
    assert _codes(doc) == ["footage_unreachable"]
    assert doc["warnings"][0]["files"] == [str(gone)]


def test_events_json_footage_unreachable_warned_once(
    project, tmp_path, monkeypatch, capsys, caplog
):
    """run_events warns and logs per unreachable source; the crash layer adds neither again."""
    gone = tmp_path / "gone.mp4"
    (src,) = [s["path"] for s in cache.load_json(project / "highlights.json")["sources"]]
    cache.save_json(project / "highlights.json", {"sources": [{"path": src}, {"path": str(gone)}]})
    seen: dict = {}

    def detect(project: Path, **kwargs) -> Path:
        seen.update(kwargs)
        path = project / "crashes.json"
        cache.save_json(path, {"crashes": []})
        return path

    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr("cutpoint.crash.detect_project", detect)
    with caplog.at_level(logging.WARNING):
        code, doc = _json(capsys, ["events", "ev", "--layers", "crash", "--no-rescore"])
    assert code == 0 and doc["ok"] is True
    assert seen["warn_unreachable"] is False  # detect_project then logs it at DEBUG only
    warns = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "footage not reachable" in r.getMessage()
    ]
    assert len(warns) == 1 and warns[0].startswith(str(gone)), warns
    assert _codes(doc).count("footage_unreachable") == 1, doc
    assert [w["files"] for w in doc["warnings"] if w["code"] == "footage_unreachable"] == [
        [str(gone)]
    ]


def test_events_crash_layer_one_warning_line_per_unreachable_source(
    project, tmp_path, monkeypatch, capsys, caplog
):
    """The real crash layer, its profile detected from the footage: the unreachable source
    gets run_events' one WARNING line, none from detect_project or the keyframe read."""
    gone = tmp_path / "gone.mp4"
    (src,) = [s["path"] for s in cache.load_json(project / "highlights.json")["sources"]]
    sources = [{"path": src, "duration": 60}, {"path": str(gone), "duration": 60}]
    cache.save_json(project / "highlights.json", {"sources": sources})
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(
        crash,
        "load_inputs",
        lambda s, dur, p, tmp: crash.SourceInputs(
            path=s, duration=dur, tel=None, imu=None, visual=SimpleNamespace(t=[]), kf_crash=None
        ),
    )
    monkeypatch.setattr("cutpoint.activity.default_ask", lambda: (None, None))
    with caplog.at_level(logging.DEBUG):
        code, doc = _json(capsys, ["events", "ev", "--layers", "crash", "--no-rescore"])
    assert code == 0 and doc["ok"] is True, doc
    warns = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and str(gone) in r.getMessage()
    ]
    assert len(warns) == 1, warns
    assert _codes(doc).count("footage_unreachable") == 1, doc
