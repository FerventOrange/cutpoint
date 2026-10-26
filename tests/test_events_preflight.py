"""Pre-flight checks: model files per layer, fail-fast ordering, Ollama unload, 360 analyze."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint import activity, crash
from cutpoint import analyze as analyze_mod
from cutpoint.events import common, paths, pipeline, visual
from cutpoint.pano import cli, views
from cutpoint.pano import geometry as G


def _create(items: list[tuple[str, Path]]) -> None:
    for _, p in items:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")


def test_clip_weights_found_in_hf_snapshot(home: Path):
    assert not paths.clip_weights().is_file()
    repo = paths.clip_dir() / "models--timm--vit_large_patch14_clip_224.openai"
    snap = repo / "snapshots" / "abc"
    snap.mkdir(parents=True)
    f = snap / "open_clip_model.safetensors"
    f.write_text("x", encoding="utf-8")
    assert paths.clip_weights() == f


# --------------------------------------------------------------------------- pipeline
@pytest.fixture
def extra_ok(monkeypatch):
    """The events extra counts as installed; no llama-server on PATH."""
    monkeypatch.setattr(pipeline.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(paths.shutil, "which", lambda *a, **k: None)


def test_preflight_lists_everything_missing_in_setup_order(home: Path, extra_ok):
    layers = ("voice", "vlm", "visual")
    with pytest.raises(FileNotFoundError) as exc:
        pipeline.preflight(layers)
    msg = str(exc.value)
    for name in (
        "Cnn14_", "class_labels_indices.csv", "silero_vad.onnx", "llama-server",
        "Qwen2.5-Omni-7B-Q4_K_M.gguf", "mmproj-", "vit_large_patch14", "fasterrcnn_",
        "sac+logos",
    ):  # fmt: skip
        assert name in msg
    # setup order is pip, panns, llama, omni, visual, voice
    assert msg.endswith("cutpoint setup --only panns llama omni visual voice")


def test_preflight_passes_when_files_present_or_ocr(home: Path, extra_ok):
    pipeline.preflight(("ocr",))
    _create(paths.layer_files(["audio"]))
    pipeline.preflight(("audio",))


@pytest.fixture
def no_unload(monkeypatch) -> list:
    """``unload_ollama`` records its calls instead of running; returns the list."""
    calls: list = []
    monkeypatch.setattr(pipeline, "unload_ollama", lambda *a, **k: calls.append(a), raising=False)
    return calls


def test_run_events_missing_models_fails_before_unload(
    home: Path, monkeypatch, tmp_path: Path, extra_ok, no_unload
):
    monkeypatch.setattr(pipeline, "load_highlights", lambda project: {"sources": []})
    with pytest.raises(FileNotFoundError, match="cutpoint setup --only"):
        pipeline.run_events(tmp_path, layers=("audio", "visual"))
    assert no_unload == []


# run_events fails fast, before side effects, when the events extra is missing
def _no_modules(monkeypatch):
    monkeypatch.setattr(pipeline.importlib.util, "find_spec", lambda name: None)


def test_vlm_only_skips_check(monkeypatch):
    _no_modules(monkeypatch)
    pipeline._require_events_extra(("vlm",))


def test_run_events_fails_before_unloading_ollama(monkeypatch, tmp_path, no_unload):
    _no_modules(monkeypatch)
    monkeypatch.setattr(pipeline, "load_highlights", lambda project: {"sources": []})
    with pytest.raises(ModuleNotFoundError) as exc:
        pipeline.run_events(tmp_path)
    assert exc.value.name == pipeline.EVENTS_MODULES[0]
    assert no_unload == []


def test_run_events_all_footage_offline_fails_before_side_effects(monkeypatch, tmp_path, no_unload):
    hl = {"sources": [{"path": str(tmp_path / "gone.mp4"), "duration": 60.0}]}
    monkeypatch.setattr(pipeline, "load_highlights", lambda project: hl)
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    with pytest.raises(ValueError, match="reachable"):
        pipeline.run_events(tmp_path, layers=("audio",))
    assert no_unload == []
    assert not (tmp_path / "events.json").exists()


# --------------------------------------------------------------------------- unload_ollama
class _Resp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def _no_sleep(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not wait")

    monkeypatch.setattr(common.time, "sleep", boom)


def test_unload_ollama_all_unloads_fail_returns_without_waiting(monkeypatch):
    _no_sleep(monkeypatch)
    gets = []

    def get(url, **k):
        gets.append(url)
        return _Resp({"models": [{"name": "m"}]})

    def post(*a, **k):
        raise common.requests.ConnectionError("down")

    monkeypatch.setattr(common.requests, "get", get)
    monkeypatch.setattr(common.requests, "post", post)
    assert common.unload_ollama("http://x") == []
    assert len(gets) == 1


# --------------------------------------------------------------------------- 360 analyze
def _analyze_stubs(monkeypatch, tmp_path: Path) -> list[str]:

    order: list[str] = []
    clip = argparse.Namespace(lrv="a.lrv", clip_id="c1")
    monkeypatch.setattr(cli, "_project", lambda arg: tmp_path)
    monkeypatch.setattr(cli, "_clips", lambda project: [clip])
    monkeypatch.setattr(pipeline, "preflight", lambda layers: order.append("preflight"))
    monkeypatch.setattr(pipeline, "run_events", lambda *a, **k: order.append("events"))
    monkeypatch.setattr(views, "run_layer", lambda clips, force=False: order.append("views") or {})
    monkeypatch.setattr(analyze_mod, "analyze", lambda *a, **k: order.append("analyze"))
    monkeypatch.setattr(analyze_mod, "reanalyze", lambda project: None)
    monkeypatch.setattr("cutpoint.pano.project.default_style", lambda: "ski")
    monkeypatch.setattr("cutpoint.styles.get_style", lambda name: name)
    return order


def _args(**kw) -> argparse.Namespace:
    base = {
        "project": "p",
        "only": None,
        "scan": None,
        "no_events": False,
        "views_only": False,
        "force_views": False,
        "style": None,
        "model": "m",
        "language": None,
    }
    return argparse.Namespace(**{**base, **kw})


def test_cmd_analyze_preflight_failure_stops_before_views(monkeypatch, tmp_path: Path):
    order = _analyze_stubs(monkeypatch, tmp_path)

    def fail(layers):
        raise FileNotFoundError("missing")

    monkeypatch.setattr("cutpoint.events.pipeline.preflight", fail)
    with pytest.raises(FileNotFoundError):
        cli.cmd_analyze(_args())
    assert order == []


@pytest.mark.parametrize("flag", ["no_events", "views_only"])
def test_cmd_analyze_skips_preflight(monkeypatch, tmp_path: Path, flag: str):
    order = _analyze_stubs(monkeypatch, tmp_path)
    cli.cmd_analyze(_args(**{flag: True}))
    assert "preflight" not in order
    assert "events" not in order


def _progress_steps(capsys) -> list[tuple[str, str]]:
    out = []
    for line in capsys.readouterr().err.splitlines():
        if line.startswith("PROGRESS "):
            d = json.loads(line[len("PROGRESS ") :])
            out.append((d["step"], d["state"]))
    return out


def test_cmd_analyze_scan_runs_before_views_and_reports_the_four_steps(
    monkeypatch, tmp_path: Path, capsys
):
    order = _analyze_stubs(monkeypatch, tmp_path)
    monkeypatch.setenv("CUTPOINT_PROGRESS", "1")
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli, "_scan", lambda *a, **k: order.append("scan"))
    cli.cmd_analyze(_args(scan=["cam"]))
    assert order == ["preflight", "scan", "views", "analyze", "events"]
    steps = [s for s in _progress_steps(capsys) if s[1] in ("running", "done")]
    assert [s for s, st in steps if st == "running"] == ["scan", "views", "analyze", "events"]


def test_cmd_analyze_without_scan_skips_the_scan_step(monkeypatch, tmp_path: Path, capsys):
    _analyze_stubs(monkeypatch, tmp_path)
    monkeypatch.setenv("CUTPOINT_PROGRESS", "1")
    cli.cmd_analyze(_args())
    assert ("scan", "skipped") in _progress_steps(capsys)


def test_scan_only_reads_and_records_just_the_named_clips(monkeypatch, tmp_path: Path):
    from cutpoint.pano import x3  # noqa: PLC0415
    from pano_helpers import orientation, x3clip  # noqa: PLC0415

    clips = [
        x3clip(clip_id="001", lrv=str(tmp_path / "a.lrv")),
        x3clip(clip_id="002", lrv=str(tmp_path / "b.lrv")),
    ]
    oriented: list[str] = []

    def fake_orientation(source, *a, **k):
        oriented.append(Path(source).name)
        return orientation(n=60)

    monkeypatch.setattr(x3, "scan", lambda folder: clips)
    monkeypatch.setattr(x3, "imu_source", lambda source: Path("imu.insv"))
    monkeypatch.setattr(G, "orientation", fake_orientation)
    project = tmp_path / "proj"
    cli._scan(project, [str(tmp_path)], only=["001"])
    saved = json.loads((project / "pano.json").read_text(encoding="utf-8"))
    assert [c["clip_id"] for c in saved["clips"]] == ["001"]
    assert oriented == ["a.lrv"]


def test_cmd_analyze_scan_only_naming_a_missing_clip_says_so(monkeypatch, tmp_path: Path):
    from cutpoint.pano import x3  # noqa: PLC0415
    from pano_helpers import x3clip  # noqa: PLC0415

    order = _analyze_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(x3, "scan", lambda folder: [x3clip(clip_id="001")])
    monkeypatch.setattr(x3, "imu_source", lambda source: None)
    with pytest.raises(FileNotFoundError, match=r"no Insta360 X3 clip 009 found") as e:
        cli.cmd_analyze(_args(scan=[str(tmp_path)], only=["009"]))
    assert "scan first" not in str(e.value)
    assert "views" not in order
    assert not (tmp_path / "pano.json").exists()


def test_cmd_analyze_scan_without_telemetry_parser_fails_before_the_scan(
    monkeypatch, tmp_path: Path
):
    order = _analyze_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(cli, "_scan", lambda *a, **k: order.append("scan"))
    with pytest.raises(ModuleNotFoundError, match="telemetry_parser"):
        cli.cmd_analyze(_args(scan=["cam"]))
    assert order == []


def test_cmd_analyze_footage_gone_before_rescoring_keeps_highlights(
    monkeypatch, tmp_path: Path, caplog
):
    reanalyze = analyze_mod.reanalyze
    _analyze_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(analyze_mod, "reanalyze", reanalyze)
    lrv = tmp_path / "a.lrv"
    lrv.write_bytes(b"x")
    hl = tmp_path / "highlights.json"
    written = json.dumps({"style": "ski", "sources": [{"path": str(lrv)}]})

    def fake_analyze(*a, **k):
        if not lrv.exists():  # the real analyze refuses a missing input
            raise FileNotFoundError(lrv)
        hl.write_text(written, encoding="utf-8")

    monkeypatch.setattr(analyze_mod, "analyze", fake_analyze)
    monkeypatch.setattr(pipeline, "run_events", lambda *a, **k: lrv.unlink())
    with caplog.at_level(logging.INFO):
        cli.cmd_analyze(_args())
    assert hl.read_bytes() == written.encode()
    assert "rerun cutpoint 360 analyze p" in caplog.text


# --------------------------------------------------------------------------- activity
@pytest.fixture
def ambiguous(monkeypatch):
    """Footage that matches no library profile clearly (every embedding identical)."""
    v = np.ones((1, 4), np.float32) / 2
    monkeypatch.setattr(
        activity, "_text_emb", lambda prompts: np.ones((len(prompts), 4), np.float32)
    )
    monkeypatch.setattr(activity, "keyframes", lambda project, **kw: (v, []))


def test_profile_unloads_only_the_model_it_used(monkeypatch, tmp_path: Path, ambiguous):
    """Free just the vision model the profile used (VRAM for the CLIP steps); models the user
    loaded in Ollama for other things stay loaded."""
    unloads: list[str] = []

    class Rater(activity.VisionRater):
        def ask_json(self, *_a, **_k) -> dict:
            return {}

        def unload(self) -> None:
            unloads.append(self.model)

    rater = Rater(url="http://x:1", model="some-model")
    monkeypatch.setattr(activity, "default_ask", lambda: (rater.ask_json, rater.model))
    for verb in ("get", "post"):  # e.g. unload_ollama listing and unloading every model
        monkeypatch.setattr(common.requests, verb, lambda url, *a, **k: unloads.append(url))
    activity.resolve(tmp_path, context="packrafting trip")
    assert unloads == ["some-model"]


@pytest.mark.parametrize(
    ("context", "llm", "ask", "reason", "profile"),
    [
        ("downhill mtb chest mount", True, (None, None), "unavailable", "mtb"),
        ("downhill mtb chest mount", False, None, "disabled", "mtb"),
        ("downhill mtb chest mount", True, (lambda p, i, s: {}, "m"), "no usable", "mtb"),
        # names two profiles: ambiguous, so the nearest profile stays
        ("ski touring after a hike", True, (None, None), "unavailable", None),
    ],
)
def test_context_not_applied_warns_and_uses_named_profile(
    monkeypatch, tmp_path: Path, caplog, ambiguous, context, llm, ask, reason, profile
):
    if ask is not None:
        monkeypatch.setattr(activity, "default_ask", lambda: ask)
    with caplog.at_level(logging.WARNING, logger=activity.logger.name):
        rec = activity.resolve(tmp_path, context=context, llm=llm)
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("--context not applied" in w and reason in w for w in warns)
    assert rec["source"] == "fallback"
    if profile:
        assert rec["name"] == profile
        assert rec["profile"]["name"] == profile
    else:
        assert rec["name"] not in ("ski", "hike")


def test_no_context_no_warning(tmp_path: Path, caplog, ambiguous):
    with caplog.at_level(logging.WARNING, logger=activity.logger.name):
        activity.resolve(tmp_path, llm=False)
    assert not any("--context not applied" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- visual -> crash handoff
class _Sentinel(Exception):
    pass


@pytest.mark.parametrize(
    ("layers", "only", "handed"),
    [
        (("visual", "crash"), (), True),
        (("visual", "crash"), ("a*",), False),
        (("visual",), (), False),
    ],
    ids=["crash-run", "only-sources", "no-crash-layer"],
)
def test_run_events_hands_the_visual_clip_model_to_the_crash_layer(
    monkeypatch, tmp_path: Path, layers, only, handed
):
    src = SimpleNamespace(path=tmp_path / "a.mp4", name="a.mp4", duration=60.0)
    monkeypatch.setattr(
        pipeline, "load_highlights", lambda project: {"sources": [{"path": str(src.path)}]}
    )
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "load_source", lambda path, dur: src)
    monkeypatch.setattr(pipeline, "bad_ranges", lambda *a: {})
    monkeypatch.setattr(pipeline, "load_transcript_lines", lambda project: [])
    monkeypatch.setattr(pipeline, "detect_audio_events", lambda *a, **k: [])

    class Enc:
        closed = 0

        def close(self):
            Enc.closed += 1

    enc, got, models = Enc(), {}, object()
    monkeypatch.setattr(
        crash.ClipEncoder, "from_visual", classmethod(lambda cls, m: enc if m is models else None)
    )

    def run_layer(sources, params, force=False, bad=None, handoff=None):
        got["handoff"] = handoff
        if handoff is not None:
            handoff(models)
        if handoff is None:
            raise _Sentinel  # nothing to hand over: stop here
        return {}

    def detect(project, **kw):
        got["encoder"] = kw.get("encoder", "missing")
        raise _Sentinel

    monkeypatch.setattr(visual, "run_layer", run_layer)
    monkeypatch.setattr(crash, "detect_project", detect)
    with pytest.raises(_Sentinel):
        pipeline.run_events(tmp_path, layers=layers, only=only)
    assert (got["handoff"] is not None) is handed
    if handed:
        assert got["encoder"] is enc and Enc.closed >= 1
