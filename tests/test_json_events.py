"""``--json`` on events (#4).

No model, no GPU: ``run_events`` is a stub that writes events.json / events.txt, the rescore
and the ski re-analysis are stubs, and the failure cases stop before any layer runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import envelope, run_json, warning_codes, write
from cutpoint import analyze, cache, cli, crash
from cutpoint.events import pipeline, scoring


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

    monkeypatch.setattr(pipeline, "run_events", run)


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
    code, doc = run_json(capsys, ["events", "ev"])
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
        | {"installed": True, "suggested": 0}
    }


def test_events_json_no_rescore(project, monkeypatch, capsys):
    _fake_run_events(monkeypatch, ["audio"])
    cache.save_json(project / "highlights.pre-events.json", {})  # from an earlier run
    assert cli.main(["events", "ev", "--no-rescore"]) == 0  # without --json: nothing on stdout
    assert capsys.readouterr().out == ""
    code, doc = run_json(capsys, ["events", "ev", "--no-rescore", "--layers", "audio"])
    assert code == 0 and doc["ok"] is True
    assert doc["outputs"]["highlights"] is None and "highlights_backup" not in doc["outputs"]


def test_events_json_ski_rerank_skipped(project, monkeypatch, capsys):
    hl = cache.load_json(project / "highlights.json")
    cache.save_json(project / "highlights.json", hl | {"style_kind": "action-cam"})
    _fake_run_events(monkeypatch, ["audio", "voice", "visual", "crash", "vlm"])
    monkeypatch.setattr(analyze, "reanalyze_if_reachable", lambda project: None)
    code, doc = run_json(capsys, ["events", "ev"])
    assert code == 0 and doc["ok"] is True
    assert warning_codes(doc) == ["highlights_not_updated"]
    assert doc["warnings"][0]["files"] == [str(project / "highlights.json")]
    out = doc["outputs"]
    assert out["highlights"] is None and out["crashes"] == str(project / "crashes.json")


def test_events_json_missing_extra(project, monkeypatch, capsys):
    monkeypatch.setattr(pipeline.importlib.util, "find_spec", lambda name: None)
    code, doc = run_json(capsys, ["events", "ev", "--layers", "audio"])
    assert code == 1 and doc["ok"] is False
    assert doc["error"]["code"] == "missing_extra"
    assert "'events' extra" in doc["error"]["message"]
    assert doc["project"] == str(project) and doc["outputs"] == {}


def test_events_json_missing_project_is_not_found(home, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    code, doc = run_json(capsys, ["events", "nosuch", "--layers", "audio"])
    assert code == 1 and doc["ok"] is False
    assert doc["error"]["code"] == "not_found"
    assert "nosuch" in doc["error"]["message"]
    assert not (home / "projects" / "nosuch").exists()


def test_events_json_interrupted(project, monkeypatch, capsys):
    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "run_events", interrupted)
    code, doc = run_json(capsys, ["events", "ev", "--layers", "audio"])
    assert code == 130 and doc["ok"] is False
    assert doc["error"]["code"] == "interrupted"
    assert doc["project"] == str(project)


def test_events_json_footage_unreachable(project, tmp_path, monkeypatch, capsys):
    gone = tmp_path / "gone.mp4"
    cache.save_json(project / "highlights.json", {"sources": [{"path": str(gone)}]})
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    code, doc = run_json(capsys, ["events", "ev", "--layers", "audio"])
    assert code == 1 and doc["ok"] is False and doc["error"]["code"] == "error"
    assert "No footage of ev is reachable" in doc["error"]["message"]
    assert warning_codes(doc) == ["footage_unreachable"]
    assert doc["warnings"][0]["files"] == [str(gone)]


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
        lambda s, dur, p, tmp, lang: crash.SourceInputs(
            path=s, duration=dur, tel=None, imu=None, visual=SimpleNamespace(t=[]), kf_crash=None
        ),
    )
    monkeypatch.setattr("cutpoint.activity.default_ask", lambda: (None, None))
    with caplog.at_level(logging.DEBUG):
        code, doc = run_json(capsys, ["events", "ev", "--layers", "crash", "--no-rescore"])
    assert code == 0 and doc["ok"] is True, doc
    warns = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and str(gone) in r.getMessage()
    ]
    assert len(warns) == 1, warns
    assert warning_codes(doc).count("footage_unreachable") == 1, doc
    assert [w["files"] for w in doc["warnings"] if w["code"] == "footage_unreachable"] == [
        [str(gone)]
    ]


def test_a_bad_ollama_url_port_is_bad_args(home, capsys):
    """#825: --ollama-url with an out-of-range port is refused at the command line."""
    for cmd in (["analyze", "talk.mp4"], ["edit", "talk.mp4"]):
        with pytest.raises(SystemExit) as exc:
            run_json(capsys, [*cmd, "--ollama-url", "myhost:99999"])
        assert exc.value.code == 2
        doc = envelope(capsys.readouterr().out)
        assert doc["ok"] is False and doc["error"]["code"] == "bad_args"
        message = doc["error"]["message"]
        assert "invalid port" in message and "myhost:99999" in message
    with pytest.raises(SystemExit) as exc:  # no host
        run_json(capsys, ["analyze", "talk.mp4", "--ollama-url", ":11434"])
    assert exc.value.code == 2
    assert "no host" in envelope(capsys.readouterr().out)["error"]["message"]
    args = cli._parse(cli.build_parser(), ["analyze", "talk.mp4", "--ollama-url", "myhost:11434"])
    assert args.ollama_url == "myhost:11434"


@pytest.mark.parametrize("cmd", ["profile", "crashes"])
def test_profile_and_crashes_pass_the_ollama_flags(home, monkeypatch, cmd):
    """#825: the Ollama flags of profile and crashes reach the activity resolution."""
    from cutpoint import activity, crash  # noqa: PLC0415 - action-cam modules

    seen = {}

    def fake(*a, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(activity, "resolve", fake)
    monkeypatch.setattr(crash, "detect_project", fake)
    args = cli._parse(
        cli.build_parser(),
        [cmd, "p", "--ollama-url", "myhost:11434", "--ollama-model", "m:1"],
    )
    with pytest.raises(RuntimeError):
        args.func(args)
    assert (seen["ollama_url"], seen["ollama_model"]) == ("myhost:11434", "m:1")
