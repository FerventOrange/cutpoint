"""``--json`` on crashes and profile (#4, phase 3b).

No model, no GPU: the per-source inputs are synthetic, the profile comes from ``--activity``
and the LLM is a stub.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import envelope, run_json, warning_codes, write
from cutpoint import activity, analyze, cache, cli, crash, jsonout

N = 600  # 60 s on the 0.1 s grid


def _inputs(src: Path, *, imu: bool = True, visual: bool = True) -> crash.SourceInputs:
    """One 8 g impact at 30 s, then standing still, and keyframes that look like a crash."""
    acc = np.zeros(N)
    acc[300:303] = 8.0
    speed = np.full(N, 30.0)
    speed[303:] = 0.0
    kf = np.zeros(60)
    kf[30:34] = 0.95
    grids = {k: np.zeros(N) for k in ("gyro_dps", "offaxis_deg", "sat")} | {"acc_g": acc}
    tel = SimpleNamespace(n=N, grid={"speed_kmh": speed, "lift_mask": np.zeros(N)})
    return crash.SourceInputs(
        path=src,
        duration=60.0,
        tel=tel if imu else None,
        imu=grids if imu else None,
        visual=SimpleNamespace(t=np.arange(60, dtype=float)) if visual else None,
        kf_crash=kf if visual else None,
    )


@pytest.fixture
def ski(home, tmp_path, monkeypatch) -> tuple[Path, Path]:
    """Project ``ski`` with one reachable source; load_inputs is synthetic."""
    src = write(tmp_path / "run.mp4", "x")
    project = home / "projects" / "ski"
    project.mkdir(parents=True)
    cache.save_json(project / "highlights.json", {"sources": [{"path": str(src), "duration": 60}]})
    monkeypatch.setattr(crash, "load_inputs", lambda s, dur, p, tmp, lang: _inputs(s))
    return project, src


CRASHES = [
    "crashes",
    "ski",
    "--activity",
    "generic",
    "--no-dense",
    "--sheets",
    "0",
    "--no-reanalyze",
]


# --------------------------------------------------------------------------------- crashes
def test_crashes_json(ski, capsys):
    project, src = ski
    code, doc = run_json(capsys, CRASHES)
    assert code == 0 and doc["ok"] is True and doc["command"] == "crashes"
    assert doc["project"] == str(project) and doc["warnings"] == []
    out = doc["outputs"]
    assert out["crashes"] == str(project / "crashes.json")
    assert out["activity_file"] == str(project / "activity.json")
    assert out["profile_name"] == "generic"
    data = json.loads((project / "crashes.json").read_text(encoding="utf-8"))
    assert data["timings"]["profile_s"] >= 0  # the profile step's own time (#916)
    saved = data["crashes"]
    assert out["count"] == len(saved) == len(out["candidates"]) == 1
    (c,) = out["candidates"]
    assert set(c) == {"id", "source", "t0", "t1", "confidence", "confirmed", "origin", "clip"}
    assert {k: saved[0][k] for k in c} == c  # crashes.json's own keys and values
    assert c["id"] == "c001" and c["source"] == str(src) and c["confirmed"] is False
    assert c["clip"]["start"] < c["t0"] <= c["t1"] < c["clip"]["end"]


@pytest.mark.parametrize("language", [None, "de"])
def test_crashes_read_the_transcript_of_the_analysis_language(ski, monkeypatch, capsys, language):
    """#371: the words layer reads the transcript of the language the analysis forced."""
    project, src = ski
    hl = {"sources": [{"path": str(src), "duration": 60}], "settings": {"language": language}}
    cache.save_json(project / "highlights.json", hl)
    langs = []

    def inputs(s, dur, p, tmp, lang):
        langs.append(lang)
        return _inputs(s)

    monkeypatch.setattr(crash, "load_inputs", inputs)
    code, _ = run_json(capsys, CRASHES)
    assert code == 0 and langs == [language]


def test_crashes_json_warnings(ski, tmp_path, monkeypatch, capsys):
    project, src = ski
    gone = tmp_path / "gone.mp4"
    sources = [{"path": str(gone), "duration": 60}, {"path": str(src), "duration": 60}]
    cache.save_json(project / "highlights.json", {"sources": sources})
    monkeypatch.setattr(
        crash, "load_inputs", lambda s, dur, p, tmp, lang: _inputs(s, imu=False, visual=False)
    )
    monkeypatch.setattr(analyze, "reanalyze_if_reachable", lambda project: None)
    code, doc = run_json(capsys, [*CRASHES[:-1], "--reanalyze"])
    assert code == 0 and doc["ok"] is True
    assert warning_codes(doc) == [
        "footage_unreachable",
        "no_telemetry",
        "no_visual_layer",
        "highlights_not_updated",
    ]
    files = [w["files"] for w in doc["warnings"]]
    assert files[:3] == [[str(gone)], [str(src)], [str(src)]]
    assert files[3] == [str(project / "highlights.json")]
    assert doc["outputs"]["count"] == 0 and doc["outputs"]["candidates"] == []


@pytest.mark.parametrize("warn_unreachable, expected", [(True, 1), (False, 0)])
def test_detect_project_warns_once_for_an_unreachable_source(
    ski, tmp_path, capsys, caplog, warn_unreachable, expected
):
    """``crashes`` gives one WARNING line and one JSON warning for an unreachable source; the
    events and analyze paths (which already warned, ``warn_unreachable=False``) neither. The
    profile is detected from the footage, so the keyframe read skips the source too."""
    project, src = ski
    gone = tmp_path / "gone.mp4"
    sources = [{"path": str(gone), "duration": 60}, {"path": str(src), "duration": 60}]
    cache.save_json(project / "highlights.json", {"sources": sources})
    with caplog.at_level(logging.DEBUG), jsonout.run("crashes") as result:
        crash.detect_project(
            project, dense=False, sheets=0, llm=False, warn_unreachable=warn_unreachable
        )
    capsys.readouterr()
    assert warning_codes(result.warnings).count("footage_unreachable") == expected
    warns = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and str(gone) in r.getMessage()
    ]
    assert len(warns) == expected, warns


def test_crashes_json_reanalyze_failure_keeps_outputs(ski, monkeypatch, capsys):
    project, _ = ski

    def boom(project):
        raise RuntimeError("reanalyze broke")

    monkeypatch.setattr(analyze, "reanalyze_if_reachable", boom)
    code, doc = run_json(capsys, [*CRASHES[:-1], "--reanalyze"])
    assert code == 1 and doc["ok"] is False and doc["error"]["message"] == "reanalyze broke"
    # crashes.json was written before reanalyze ran; the envelope still names it.
    assert doc["outputs"]["crashes"] == str(project / "crashes.json")
    assert (project / "crashes.json").is_file() and doc["outputs"]["count"] == 1


def test_crashes_json_without_clip_is_missing_extra(ski, monkeypatch, capsys):
    monkeypatch.delenv("TORCH_HOME", raising=False)  # set by the encoder before its imports
    monkeypatch.setitem(sys.modules, "open_clip", None)  # import open_clip fails
    argv = [a for a in CRASHES if a != "--no-dense"]
    code, doc = run_json(capsys, argv)
    assert code == 1 and doc["ok"] is False and doc["command"] == "crashes"
    assert doc["error"]["code"] == "missing_extra"
    assert doc["error"]["message"].startswith(
        "open_clip is not installed: this feature is not available in Cutpoint lite"
    )


def test_crashes_json_unknown_activity_is_bad_args(ski, capsys):
    code, doc = run_json(capsys, ["crashes", "ski", "--activity", "curling", "--no-dense"])
    assert code == 1 and doc["error"]["code"] == "bad_args"
    assert "unknown activity 'curling'" in doc["error"]["message"]


# --------------------------------------------------------------------------------- profile
def test_profile_json_keeps_the_record_keys(ski, capsys):
    project, _ = ski
    code, doc = run_json(capsys, ["profile", "ski", "--activity", "generic"])
    assert code == 0 and doc["ok"] is True and doc["command"] == "profile"
    assert doc["project"] == str(project) and doc["warnings"] == []
    rec = json.loads((project / "activity.json").read_text(encoding="utf-8"))
    # The old --json printed the activity.json record; its keys stay, plus the file's path.
    assert doc["outputs"] == rec | {"activity_file": str(project / "activity.json")}
    assert {"version", "request", "source", "name", "profile"} <= set(doc["outputs"])
    assert (doc["outputs"]["name"], doc["outputs"]["source"]) == ("generic", "flag")
    assert doc["outputs"]["profile"]["incident"]["pos"]


def test_profile_json_context_never_prints_to_stdout(ski, monkeypatch, capsys):
    def chatty(prompt, images, schema):
        print("model says hi")  # a stray print from the local model's client
        return {}  # unusable, twice: --context is not applied

    monkeypatch.setattr(activity, "default_ask", lambda: (chatty, None))
    code = cli.main(
        ["profile", "ski", "--activity", "generic", "--context", "icy groomer", "--json"]
    )
    captured = capsys.readouterr()
    doc = envelope(captured.out)
    assert code == 0 and doc["ok"] is True
    assert "model says hi" in captured.err
    assert warning_codes(doc) == ["context_not_applied"]
    assert doc["outputs"]["request"] == {"activity": "generic", "context": "icy groomer"}


def test_profile_text_unchanged_without_json(ski, capsys):
    assert cli.main(["profile", "ski", "--activity", "generic"]) == 0
    assert capsys.readouterr().out.startswith("activity  generic (flag)\n")


def test_profile_shows_the_negatives_crashes_uses_for_an_old_record(ski, capsys):
    """An activity.json from before profiles had incident negatives shows generic's, as
    ``crashes`` reads it (record_profile), in the text and in --json."""
    project, _ = ski
    generic = activity.full(activity.library()["generic"])
    old = {**generic, "incident": {**generic["incident"], "neg": []}}
    request = {"activity": "generic", "context": None}
    rec = {"version": activity.PROFILE_VERSION, "request": request, "source": "flag"}
    cache.save_json(project / activity.ACTIVITY_FILE, {**rec, "name": "generic", "profile": old})
    want = activity.record_profile(cache.load_json(project / activity.ACTIVITY_FILE))
    assert want["incident"]["neg"]
    code, doc = run_json(capsys, ["profile", "ski"])
    assert code == 0 and doc["outputs"]["profile"]["incident"]["neg"] == want["incident"]["neg"]
    assert cli.main(["profile", "ski"]) == 0
    assert f"{len(want['incident']['neg'])} negatives" in capsys.readouterr().out


def test_profile_warns_once_for_an_unreachable_source(ski, tmp_path, monkeypatch, capsys, caplog):
    """The profile detected from the footage: one WARNING line per unreachable source and,
    with --json, one ``footage_unreachable`` warning, as on ``crashes``."""
    project, src = ski
    gone = tmp_path / "gone.mp4"
    sources = [{"path": str(gone), "duration": 60}, {"path": str(src), "duration": 60}]
    cache.save_json(project / "highlights.json", {"sources": sources})
    monkeypatch.setattr(activity, "default_ask", lambda: (None, None))  # no Ollama
    with caplog.at_level(logging.DEBUG):
        code, doc = run_json(capsys, ["profile", "ski"])
    assert code == 0 and doc["ok"] is True, doc
    assert [w["files"] for w in doc["warnings"] if w["code"] == "footage_unreachable"] == [
        [str(gone)]
    ]
    warns = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and str(gone) in r.getMessage()
    ]
    assert len(warns) == 1 and warns[0].endswith("; skipped"), warns


@pytest.mark.parametrize(
    ("reason", "expected"),
    [(None, "cutpoint events"), ("no CUDA GPU for the keyframes step", "(no CUDA GPU")],
)
def test_no_visual_layer_says_why_inside_analyze(ski, monkeypatch, reason, expected):
    """Inside analyze (no_visual_reason), the warning keeps its code and gives the reason."""
    project, src = ski
    monkeypatch.setattr(crash, "load_inputs", lambda s, dur, p, tmp, lang: _inputs(s, visual=False))
    with jsonout.run("crashes") as result, crash.no_visual_reason(reason):
        crash.detect_project(project, dense=False, sheets=0, activity_name="generic")
    (w,) = [w for w in result.warnings if w["code"] == "no_visual_layer"]
    assert expected in w["message"] and w["files"] == [str(src)]
