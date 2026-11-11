"""``scripts/bench.py``, the cross-category benchmark: manifest and label checks, the holdout
rules, the machine lock, the resource sampler, the runner (no real analyze: ``subprocess.run`` is
faked and writes a synthetic ``highlights.json``), the per-bucket scores, A vs B with the gate,
``--compare`` and the report. Footage, labels and projects are invented."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import requests

from conftest import load_script

pytestmark = pytest.mark.script
bench = load_script("bench")

VIDEO = "clip-a.mp4"


# -- helpers -----------------------------------------------------------------------------------


def grid(n: int = 20) -> list[dict]:
    """``n`` candidates of 10 s: candidate i covers [i*50, i*50+10], its score falls with i."""
    return [
        {
            "id": f"h{i:03d}",
            "source": f"D:\\Videos\\{VIDEO}",
            "source_index": 0,
            "start": i * 50.0,
            "end": i * 50.0 + 10,
            "duration": 10.0,
            "score": 100.0 - i,
            "label": "x",
        }
        for i in range(n)
    ]


def write_highlights(project: Path, category: str = "gameplay", ambiguous: bool = False) -> None:
    project.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "style": "gaming",
        "detection": {
            "category": category,
            "ambiguous": ambiguous,
            "style": "gaming",
            "fallback": None,
            "decided_by": [] if ambiguous else ["hud"],
            "reason": "invented",
            "sources": [],
        },
        "settings": {"target_total": 20.0},
        "sources": [{"path": f"D:\\Videos\\{VIDEO}", "duration": 1000.0}],
        "suggested_top": 2,
        "candidates": grid(),
    }
    (project / "highlights.json").write_text(json.dumps(data), encoding="utf-8")


def source(sid: str, **over) -> dict:
    return {
        "id": sid,
        "input": f"footage/{sid}.mp4",
        "category": "gameplay",
        "bucket": "gameplay-kills",
        "holdout": False,
        **over,
    }


def write_manifest(tmp_path: Path, sources: list[dict], labels: dict | None = None, **top) -> Path:
    """A manifest in ``tmp_path/bench`` with its footage files; ``labels`` maps a label file
    name to its JSON."""
    root = tmp_path / "bench"
    (root / "footage").mkdir(parents=True, exist_ok=True)
    for s in sources:
        (root / s["input"]).write_bytes(b"x")
    for name, data in (labels or {}).items():
        (root / name).write_text(json.dumps(data), encoding="utf-8")
    held = [s for s in sources if s.get("holdout")]
    if held and "holdout_sha256" not in top:
        top["holdout_sha256"] = bench.holdout_hash(
            [bench.Source(s["id"], Path(), "", "", True) for s in held]
        )
    path = root / "manifest.json"
    path.write_text(json.dumps({"schema": 1, **top, "sources": sources}), encoding="utf-8")
    return path


class FakeProc:
    """A finished ``subprocess.Popen``; ``communicate`` may raise ``raises`` instead."""

    def __init__(self, returncode: int, stdout: str, raises: BaseException | None = None):
        self.returncode, self.stdout, self.raises = returncode, stdout, raises
        self.actions: list[str] = []
        self.running = raises is not None

    def communicate(self):
        if self.raises is not None:
            raise self.raises
        return self.stdout, None

    def poll(self):
        return None if self.running else self.returncode

    def wait(self, timeout=None):
        self.actions.append("wait")
        if self.running:
            raise subprocess.TimeoutExpired("analyze", timeout)
        return self.returncode

    def terminate(self):
        self.actions.append("terminate")

    def kill(self):
        self.actions.append("kill")
        self.running = False


class FakeRun:
    """``subprocess.Popen`` for ``analyze``: writes the project's highlights.json with the
    detection ``results`` gives the input's stem (default: gameplay, sure). Stems in ``fail``
    exit 1, in ``missing`` cannot start, in ``interrupt`` get a Ctrl+C."""

    def __init__(self) -> None:
        self.results: dict = {}
        self.fail: set = set()
        self.missing: set = set()
        self.interrupt: set = set()
        self.calls: list[tuple[list[str], dict]] = []
        self.procs: list[FakeProc] = []

    def __call__(self, argv, **kw):
        self.calls.append((list(argv), kw.get("env") or {}))
        stem = Path(argv[argv.index("analyze") + 1]).stem
        if stem in self.missing:
            raise FileNotFoundError(2, "No such file or directory", argv[0])
        if stem in self.interrupt:
            proc = FakeProc(0, "", KeyboardInterrupt())
        elif stem in self.fail:
            err = {"ok": False, "error": {"code": "error", "message": "boom"}, "warnings": []}
            proc = FakeProc(1, json.dumps(err) + "\n")
        else:
            project = Path(argv[argv.index("--out") + 1])
            write_highlights(project, *self.results.get(stem, ("gameplay", False)))
            env = {"ok": True, "error": None, "warnings": [{"code": "style_ambiguous"}]}
            proc = FakeProc(0, "note\n" + json.dumps(env) + "\n")
        self.procs.append(proc)
        return proc

    def analyzed(self) -> list[str]:
        return [Path(a[a.index("analyze") + 1]).stem for a, _ in self.calls]


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    for name in ("CUTPOINT_HOME", "CUTPOINT_CACHE_DIR", "CUTPOINT_MODELS_DIR", "CUTPOINT_CONFIG"):
        monkeypatch.delenv(name, raising=False)  # main sets them; restored afterwards
    monkeypatch.setattr(bench, "_git", lambda *a: None)
    monkeypatch.setattr(bench, "machine_lock", lambda: tmp_path / "machine" / "bench.lock")
    run = FakeRun()
    monkeypatch.setattr(bench.subprocess, "Popen", run)
    return run


def results_files(manifest: Path) -> dict[str, dict]:
    out = manifest.parent / "results"
    return {p.name: json.loads(p.read_text()) for p in out.glob("*.json")}


# -- manifest and labels -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sources", "labels", "message"),
    [
        ([source("a", category="sport")], None, "category 'sport'"),
        ([source("a", bucket="kills")], None, "bucket 'kills'"),
        ([source("a"), source("a")], None, "duplicate id"),
        (
            [source("a", labels="a.json")],
            {"a.json": {"video": VIDEO, "keep": [[1, 2]], "holdout": True}},
            "label file says holdout true",
        ),
    ],
)
def test_manifest_errors(tmp_path, sources, labels, message):
    path = write_manifest(tmp_path, sources, labels)
    with pytest.raises(bench.ManifestError, match=message):
        bench.load_manifest(path)


def test_manifest_loads_labels_with_time_strings_and_without_holdout(tmp_path):
    labels = {
        "a.json": {
            "video": VIDEO,
            "duration": "16:40",
            "keep": [["1:40", "1:50"], [300, 310]],
            "reference": [["0:20:00", "0:21:00.5"]],
        }
    }
    m = bench.load_manifest(write_manifest(tmp_path, [source("a", labels="a.json")], labels))
    lab = m.sources[0].labels
    assert lab["duration"] == 1000.0
    assert lab["keep"] == [(100.0, 110.0), (300.0, 310.0)]
    assert lab["reference"] == [(1200.0, 1260.5)]
    assert lab["holdout"] is False
    assert [v.name for v in m.variants] == ["base"]


@pytest.mark.parametrize(
    ("value", "seconds"),
    [(75, 75), ("75.5", 75.5), ("1:15", 75.0), ("1:00:01", 3601.0), ("1:75", None), ("x", None)],
)
def test_time_strings(value, seconds):
    if seconds is None:
        with pytest.raises(ValueError, match="mm:ss"):
            bench.to_seconds(value, "here")
    else:
        assert bench.to_seconds(value, "here") == seconds


# -- holdout -----------------------------------------------------------------------------------


def test_holdout_hash_missing_wrong_and_rehash(tmp_path, capsys, caplog, fake_run):
    sources = [source("a"), source("h", holdout=True)]
    good = bench.holdout_hash([bench.Source("h", Path(), "", "", True)])
    missing = write_manifest(tmp_path, sources, holdout_sha256=None)
    with pytest.raises(bench.ManifestError, match=good):
        bench.check_holdout_hash(bench.load_manifest(missing))
    wrong = write_manifest(tmp_path, sources, holdout_sha256="00" * 32)
    assert bench.main([str(wrong), "--sample-interval", "0"]) == 2
    assert "--rehash" in caplog.text
    assert fake_run.calls == []  # refused before any run
    assert bench.main([str(wrong), "--rehash"]) == 0
    assert capsys.readouterr().out.strip() == good
    assert fake_run.calls == []


def test_holdout_hidden_by_default_and_shown_with_flag(tmp_path, fake_run):
    labels = {"h.json": {"video": VIDEO, "keep": [[0, 10]]}}
    sources = [source("a"), source("h", holdout=True, labels="h.json")]
    manifest = write_manifest(tmp_path, sources, labels)
    assert bench.main([str(manifest), "--sample-interval", "0", "--runs", "10"]) == 0
    assert fake_run.analyzed() == ["a"]
    (res,) = results_files(manifest).values()
    assert [s["id"] for s in res["sources"]] == ["a"]
    assert res["holdout_viewed"] is False and "holdout" not in res["summary"]
    report = next((manifest.parent / "results").glob("*-report.md")).read_text()
    assert "HOLDOUT" not in report

    for p in (manifest.parent / "results").iterdir():
        p.unlink()
    argv = [str(manifest), "--sample-interval", "0", "--runs", "10", "--holdout"]
    assert bench.main(argv) == 0
    assert fake_run.analyzed()[-1] == "h"  # a is reused, h runs
    (res,) = results_files(manifest).values()
    assert res["holdout_viewed"] is True
    held = res["summary"]["holdout"]
    assert held["ranking"]["gameplay-kills"]["metrics"]["keep_top10"]["value"] == 1
    assert res["summary"]["tune"]["detection"][bench.ALL]["sources"] == 1  # a only
    report = next((manifest.parent / "results").glob("*-report.md")).read_text()
    assert "HOLDOUT" in report and "Holdout viewed" in report


# -- lock --------------------------------------------------------------------------------------


def test_lock_contention_and_stale_pid(tmp_path, monkeypatch):
    assert bench.machine_lock().parent == Path(bench.tempfile.gettempdir())  # one per machine
    path = tmp_path / "locks" / "bench.lock"
    with bench.bench_lock(path) as lock:
        assert lock.read_text() == str(os.getpid())
        with (
            pytest.raises(bench.BenchBusyError, match="another bench run"),
            bench.bench_lock(path),
        ):
            pass
    assert not lock.exists()
    alive = {4242: True}
    monkeypatch.setattr(bench.procs, "pid_alive", lambda pid, not_after=None: alive[pid])
    lock.write_text("4242")
    with pytest.raises(bench.BenchBusyError, match="pid 4242"), bench.bench_lock(path):
        pass
    assert lock.read_text() == "4242"  # another live bench's lock is left alone
    alive[4242] = False
    with bench.bench_lock(path):
        assert lock.read_text() == str(os.getpid())  # the stale lock was replaced
    assert not lock.exists()


# -- sampler -----------------------------------------------------------------------------------


def test_sampler_parsing_and_unavailable(monkeypatch):
    smi = "0, 5120, 24576, 87\n1, [N/A], 8192, 0\n"

    def run(argv, **kw):
        assert argv == bench.SMI_QUERY
        return subprocess.CompletedProcess(argv, 0, stdout=smi)

    monkeypatch.setattr(bench.subprocess, "run", run)
    assert bench.sample_gpu() == [
        {"index": 0, "memory_used_mib": 5120, "memory_total_mib": 24576, "util_pct": 87}
    ]

    def missing(argv, **kw):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(bench.subprocess, "run", missing)
    assert bench.sample_gpu() is None

    class Resp:
        status_code = 200

        def json(self):
            return {"models": [{"name": "qwen3.5:9b", "size_vram": 6 * 2**30, "size": 7}]}

    monkeypatch.setattr(bench.requests, "get", lambda url, **kw: Resp())
    assert bench.sample_ollama("http://127.0.0.1:11434")[0]["name"] == "qwen3.5:9b"
    assert bench.sample_ollama(None) is None

    def down(url, **kw):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(bench.requests, "get", down)
    assert bench.sample_ollama("http://127.0.0.1:11434") is None

    samples = [
        {"t": 0, "gpu": [{"index": 0, "memory_used_mib": 1000, "memory_total_mib": 8000,
                          "util_pct": 10}],
         "ollama": [{"name": "m", "size_vram": 2**30}]},
        {"t": 5, "gpu": [{"index": 0, "memory_used_mib": 3000, "memory_total_mib": 8000,
                          "util_pct": 30}],
         "ollama": None},
        {"t": 10, "gpu": None, "ollama": []},
    ]  # fmt: skip
    s = bench.summarize_samples(samples)
    assert (s["samples"], s["gpu_available"], s["ollama_available"]) == (3, 2, 2)
    assert s["gpu"]["0"]["memory_used_mib"] == {"min": 1000, "max": 3000, "mean": 2000.0}
    assert s["ollama_vram_mib"] == {"min": 0, "max": 1024, "mean": 512.0}
    assert s["ollama_models"] == ["m"]


def test_sampler_thread_samples_while_the_run_goes(monkeypatch):
    monkeypatch.setattr(bench, "take_sample", lambda t, url: {"t": t, "gpu": None, "ollama": None})
    with bench.Sampler(0.01, None) as s:
        while len(s.samples) < 2:
            pass
    assert s.result()["summary"]["samples"] >= 2
    with bench.Sampler(0, None) as s:
        pass
    assert s.samples == []


# -- runner ------------------------------------------------------------------------------------


def test_runner_argv_env_and_stamp_reuse(tmp_path, fake_run):
    manifest = write_manifest(tmp_path, [source("a")])
    root = manifest.parent
    argv = [str(manifest), "--sample-interval", "0", "--variant", "nov:--no-vision --top 3"]
    assert bench.main(argv) == 0
    ((call, env),) = fake_run.calls
    project = root / "runs" / "nov" / "a"
    assert call[-10:] == [
        "analyze", str(root / "footage" / "a.mp4"), "--style", "auto", "--out", str(project),
        "--json", "--no-vision", "--top", "3",
    ]  # fmt: skip
    assert call[:-10] == [sys.executable, "-m", "cutpoint"]
    assert env["CUTPOINT_HOME"] == str(root / "home")
    assert env["CUTPOINT_CACHE_DIR"] == str(root / "cache")
    assert env["CUTPOINT_MODELS_DIR"]
    assert (project.parent / "a.stamp.json").is_file()
    (res,) = results_files(manifest).values()
    assert res["sources"][0]["run"]["warnings"] == ["style_ambiguous"]

    assert bench.main(argv) == 0
    assert len(fake_run.calls) == 1  # same stamp: reused
    assert bench.main([*argv, "--rerun"]) == 0
    assert len(fake_run.calls) == 2
    other = [str(manifest), "--sample-interval", "0", "--variant", "nov:--no-vision"]
    assert bench.main(other) == 0
    assert len(fake_run.calls) == 3  # other args: a new run
    footage = root / "footage" / "a.mp4"
    footage.write_bytes(b"longer")
    assert bench.main(other) == 0
    assert len(fake_run.calls) == 4  # the input changed


def test_failed_run_keeps_no_stamp_and_reuse_takes_a_folder(tmp_path, fake_run):
    manifest = write_manifest(tmp_path, [source("a"), source("b")])
    fake_run.fail = {"a"}
    given = tmp_path / "earlier"
    write_highlights(given, "talk")
    argv = [str(manifest), "--sample-interval", "0", "--reuse", f"b={given}"]
    assert bench.main(argv) == 0
    assert fake_run.analyzed() == ["a"]
    assert not (manifest.parent / "runs" / "base" / "a.stamp.json").exists()
    (res,) = results_files(manifest).values()
    a, b = res["sources"]
    assert a["run"]["status"] == "failed" and a["run"]["error"]["message"] == "boom"
    assert b["run"]["status"] == "given" and b["detection"]["verdict"] == "sure_wrong"
    assert bench.main([str(manifest), "--reuse", "zzz=x"]) == 2


def test_bench_next_to_a_source_is_refused(tmp_path, fake_run):
    manifest = write_manifest(tmp_path, [source("a", input="a.mp4")])
    (manifest.parent / "a.mp4").write_bytes(b"x")
    assert bench.main([str(manifest), "--sample-interval", "0"]) == 2
    assert fake_run.calls == []
    assert (
        bench.main([str(manifest), "--sample-interval", "0", "--bench", str(tmp_path / "o")]) == 0
    )


# -- scoring -----------------------------------------------------------------------------------


def test_per_bucket_detection_and_ranking(tmp_path):
    keeps = {"video": VIDEO, "keep": [[0, 10], [50, 60], [900, 910]], "skip": [[100, 110]]}
    labels = {"k.json": keeps}
    sources = [
        source("g1", labels="k.json"),
        source("g2", labels="k.json"),
        source("t1", category="talk", bucket="talk"),
        source("t2", category="talk", bucket="talk"),
    ]
    m = bench.load_manifest(write_manifest(tmp_path, sources, labels))
    detected = {"g1": ("gameplay", False), "g2": ("gameplay", True), "t1": ("gameplay", False),
                "t2": ("talk", False)}  # fmt: skip
    entries = []
    for s in m.sources:
        project = tmp_path / "p" / s.id
        write_highlights(project, *detected[s.id])
        entries.append(bench.source_entry(s) | bench.score_source(s, project, 20, 7, 0.5))
    det = bench.summarize(entries)["tune"]["detection"]
    assert (det["gameplay-kills"]["right"], det["gameplay-kills"]["ambiguous"]) == (1, 1)
    assert det["gameplay-kills"]["ambiguous_rate"] == 0.5
    assert det["talk"]["sure_wrong"] == ["t1"] and det["talk"]["accuracy"] == 0.5
    assert det["talk"]["confusion"] == {
        "gameplay": {"sure": 1, "ambiguous": 0},
        "talk": {"sure": 1, "ambiguous": 0},
    }
    assert det[bench.ALL]["sources"] == 4
    rank = bench.ranking_summary(entries)
    assert set(rank) == {"gameplay-kills", bench.ALL}  # talk has no labels
    k10 = rank["gameplay-kills"]["metrics"]["keep_top10"]
    assert (k10["value"], k10["of"], k10["videos"]) == (4, 6, 2)  # 2 of 3 keeps per video
    assert rank["gameplay-kills"]["metrics"]["skip_top10"]["value"] == 2


# -- compare and gate --------------------------------------------------------------------------


def entry(
    sid,
    bucket="gameplay-kills",
    det=("gameplay", False),
    keep20=8,
    keep30=8,
    skip20=1,
    noise=True,
    holdout=False,
):
    category = "gameplay" if bucket.startswith("gameplay") else "talk"
    d = None
    if det is not None:
        verdict = "ambiguous" if det[1] else "right" if det[0] == category else "sure_wrong"
        d = {"category": det[0], "ambiguous": det[1], "verdict": verdict}
    metrics = {
        "keep_top20": {"value": keep20, "beats_noise": noise},
        "keep_30": {"value": keep30, "beats_noise": noise},
        "skip_top20": {"value": skip20, "beats_noise": False},
    }
    return {
        "id": sid,
        "bucket": bucket,
        "category": category,
        "holdout": holdout,
        "detection": d,
        "ranking": {"labels": {"keep": 10, "skip": 5, "reference": 0}, "metrics": metrics},
    }


def run_result(name, sources, edition="full", holdout=False):
    return {
        "kind": bench.KIND,
        "schema": bench.SCHEMA,
        "created": "20261111-120000",
        "variant": {"name": name, "args": [], "cutpoint": ["cutpoint"]},
        "cutpoint": {"cutpoint": ["cutpoint"], "version": "0.5.0", "git": None, "edition": edition},
        "edition": edition,
        "manifest": "m.json",
        "settings": {"runs": 1000, "seed": 7, "min_overlap": 0.5},
        "holdout_viewed": holdout,
        "sources": sources,
        "summary": bench.summarize(sources),
    }


BASE = [entry(f"g{i}") for i in range(5)] + [
    entry(f"t{i}", "talk", ("talk", False)) for i in range(2)
]


@pytest.mark.parametrize(
    ("change", "failure"),
    [
        ({"t0": {"det": ("gameplay", False)}}, "t0: new sure-wrong"),
        ({"g0": {"det": ("gameplay", True)}}, "gameplay-kills: ambiguous rate up 20.0"),
        ({"g0": {"keep20": 3}, "g1": {"keep20": 4}}, "keep_top20 down 9"),
        ({"g0": {"skip20": 3}}, "skip_top20 up 2"),
        ({"t0": {"noise": False}}, "talk: keep_30 beats noise in 1 fewer"),
        ({"g4": {"det": None}}, "g4: scored in A, failed"),
    ],
)
def test_gate_rules(change, failure):
    a = run_result("A", BASE)
    new = []
    for s in BASE:
        kw = {"bucket": s["bucket"]}
        if s["detection"]:
            kw["det"] = (s["detection"]["category"], s["detection"]["ambiguous"])
        kw |= change.get(s["id"], {})
        new.append(entry(s["id"], **kw))
    cmp = bench.compare_results(a, run_result("B", new), bench.GateRules())
    assert cmp["gate"]["verdict"] == "fail"
    assert any(failure in f for f in cmp["gate"]["failures"]), cmp["gate"]["failures"]


def test_gate_tolerances_pass():
    a = run_result("A", BASE)
    # 40 keeps in the bucket: 10% is 4, so a drop of 4 passes; one more skip passes
    small = [entry(s["id"], s["bucket"], (s["detection"]["category"], False)) for s in BASE]
    small[0] = entry("g0", keep20=4, keep30=5, skip20=2)
    cmp = bench.compare_results(a, run_result("B", small), bench.GateRules())
    assert cmp["gate"] == {"verdict": "pass", "failures": []}
    rows = {r["id"]: r for r in cmp["tune"]["sources"]}
    assert rows["g0"]["ranking"] == {"keep_top20": -4, "keep_30": -3, "skip_top20": 1}
    tight = bench.GateRules(keep_drop_share=0.05)
    assert bench.compare_results(a, run_result("B", small), tight)["gate"]["verdict"] == "fail"


def test_lite_against_full_gives_deltas_but_no_gate():
    worse = [entry(s["id"], s["bucket"], ("general", False)) for s in BASE]
    cmp = bench.compare_results(
        run_result("A", BASE), run_result("B", worse, "lite"), bench.GateRules()
    )
    assert cmp["gate"]["verdict"] == "refused"
    assert any("edition differs" in r for r in cmp["refusals"])
    assert len(cmp["tune"]["sources"]) == len(BASE)
    text = "\n".join(bench.compare_lines(cmp))
    assert "NOT GIVEN" in text and "lite baseline" in text


def test_compare_files_and_strict_gate(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    held = [entry("h0", holdout=True)]
    a.write_text(json.dumps(run_result("A", BASE + held, holdout=True)))
    worse = [entry(s["id"], s["bucket"], keep20=0) for s in BASE[:5]] + BASE[5:]
    worse.append(entry("h0", keep30=0, holdout=True))
    b.write_text(json.dumps(run_result("B", worse, holdout=True)))
    assert bench.main(["--compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "## B (20261111-120000) vs A" in out
    assert "Gate (advice, tune split only): FAIL" in out
    assert "| gameplay-kills | 8 -> 8" not in out and "40 -> 0 (-40)" in out
    assert "HOLDOUT" not in out and "h0" not in out  # the holdout needs --holdout
    assert bench.main(["--compare", str(a), str(b), "--holdout"]) == 0
    out = capsys.readouterr().out
    assert "HOLDOUT" in out and "h0 (gameplay-kills): keep_30 -8" in out
    assert bench.main(["--compare", str(a), str(b), "--strict-gate"]) == 3
    assert bench.main(["--compare", str(a), str(a), "--strict-gate"]) == 0
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"schema": 2, "videos": []}))  # a moment_eval file
    assert bench.main(["--compare", str(a), str(other)]) == 2


def test_relabelled_sources_are_left_out_of_the_gate():
    a = run_result("A", [{**s, "labels_sha256": "aa"} for s in BASE])
    new = [{**s, "labels_sha256": "aa"} for s in BASE]
    new[0] = {**entry("g0", keep20=0, keep30=0), "labels_sha256": "bb"}  # relabelled, worse
    new[1] = {**entry("g1", bucket="gameplay-nokills"), "labels_sha256": "aa"}  # new bucket
    cmp = bench.compare_results(a, run_result("B", new), bench.GateRules())
    assert cmp["gate"] == {"verdict": "pass", "failures": []}
    assert any(w.startswith("g0: labels_sha256 differ") for w in cmp["warnings"])
    assert any(w.startswith("g1: bucket differ") for w in cmp["warnings"])
    assert cmp["tune"]["detection"]["b"][bench.ALL]["sources"] == len(BASE) - 2


def test_relabelled_holdout_is_named_only_with_holdout():
    held = {**entry("h0", holdout=True), "labels_sha256": "aa"}
    a = run_result("A", [*BASE, held], holdout=True)
    # relabelled, and moved into the tune split: still kept out of the tune deltas
    b = run_result("B", [*BASE, {**entry("h0"), "labels_sha256": "bb"}], holdout=True)
    for flag in (False, True):
        cmp = bench.compare_results(a, b, bench.GateRules(), holdout=flag)
        assert any("h0" in w for w in cmp["warnings"]) is flag
        assert "h0" not in json.dumps(cmp["tune"])


def test_two_variants_report(tmp_path, fake_run):
    labels = {"k.json": {"video": VIDEO, "keep": [[0, 10], [900, 910]]}}
    manifest = write_manifest(
        tmp_path,
        [source("g", labels="k.json"), source("t", category="talk", bucket="talk")],
        labels,
        variants=[{"name": "base", "args": []}, {"name": "cand", "args": ["--no-vision"]}],
    )
    assert bench.main([str(manifest), "--sample-interval", "0", "--runs", "10"]) == 0
    assert len(fake_run.calls) == 4
    files = results_files(manifest)
    assert sorted(n.split("-", 2)[-1] for n in files) == ["base.json", "cand.json"]
    report = next((manifest.parent / "results").glob("*-report.md")).read_text()
    for part in (
        "## base: detection (tune)",
        "## cand: ranking (tune)",
        "| talk | 1 | 0 | 0 | 1 |",  # detected gameplay: sure-wrong
        "Sure-wrong: t",
        "| gameplay-kills | 1 | 1/2 |",
        "vs base",
        "Gate (advice, tune split only): PASS",
        "## Resources",
    ):
        assert part in report, part


# -- review fixes ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "windows", "argv"),
    [
        (
            "uv run --project ../old cutpoint",
            False,
            ["uv", "run", "--project", "../old", "cutpoint"],
        ),
        ("'/opt/my tools/cutpoint' -v", False, ["/opt/my tools/cutpoint", "-v"]),
        (
            r"H:\cp-main\.venv\Scripts\cutpoint.exe",
            True,
            [r"H:\cp-main\.venv\Scripts\cutpoint.exe"],
        ),
        (r'"C:\My Tools\cutpoint.exe" --x', True, [r"C:\My Tools\cutpoint.exe", "--x"]),
    ],
)
def test_split_cmdline_keeps_windows_backslashes(text, windows, argv):
    assert bench.split_cmdline(text, windows=windows) == argv


def test_ctrl_c_stops_analyze_and_frees_the_lock(tmp_path, fake_run):
    manifest = write_manifest(tmp_path, [source("a"), source("b")])
    fake_run.interrupt = {"a"}
    assert bench.main([str(manifest), "--sample-interval", "0"]) == 130
    (proc,) = fake_run.procs
    # the child got the Ctrl+C too: waited for (not terminated), then killed when it hangs
    assert proc.actions == ["wait", "kill", "wait"]
    assert fake_run.analyzed() == ["a"]  # the bench stopped
    assert not bench.machine_lock().exists()
    other = FakeProc(0, "", RuntimeError())  # any other error terminates it first
    bench.stop_analyze(other, RuntimeError())
    assert other.actions[0] == "terminate"


def test_a_command_that_cannot_start_is_a_failed_job(tmp_path, fake_run):
    manifest = write_manifest(tmp_path, [source("a"), source("b")])
    fake_run.missing = {"a"}
    assert bench.main([str(manifest), "--sample-interval", "0"]) == 0
    assert fake_run.analyzed() == ["a", "b"]  # the bench went on
    (res,) = results_files(manifest).values()
    a, b = res["sources"]
    assert a["run"]["status"] == "failed" and a["run"]["error"]["code"] == "start_failed"
    assert "No such file" in a["errors"][0]
    assert b["run"]["status"] == "ok"


def test_git_state_hashes_untracked_files(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "REPO", tmp_path)
    (tmp_path / "src").mkdir()
    new = tmp_path / "src" / "new.py"
    new.write_text("A = 1\n")
    answers = {
        "rev-parse": "abc123\n",
        "status": "?? src/new.py\n",
        "diff": "",
        "ls-files": "src/new.py\0",
    }
    monkeypatch.setattr(bench, "_git", lambda *a: answers[a[0]])
    first = bench.git_state()
    assert first.startswith("abc123+dirty.")
    new.write_text("A = 2\n")
    assert bench.git_state() != first  # same status and diff, other bytes
    answers["status"] = ""
    assert bench.git_state() == "abc123"


def test_config_and_cache_are_part_of_the_stamp(tmp_path, fake_run):
    manifest = write_manifest(
        tmp_path,
        [source("a")],
        variants=[{"name": "base"}, {"name": "own", "args": [], "cache": "cache-own"}],
    )
    root = manifest.parent
    argv = [str(manifest), "--sample-interval", "0"]
    assert bench.main(argv) == 0
    envs = {Path(c[c.index("--out") + 1]).parent.name: e for c, e in fake_run.calls}
    assert envs["base"]["CUTPOINT_CACHE_DIR"] == str(root / "cache")
    assert envs["own"]["CUTPOINT_CACHE_DIR"] == str(root / "cache-own")
    assert bench.main(argv) == 0
    assert len(fake_run.calls) == 2  # reused
    (root / "home").mkdir(exist_ok=True)
    (root / "home" / "config.json").write_text('{"vision": true}')
    assert bench.main(argv) == 0
    assert len(fake_run.calls) == 4  # the config changed
    assert bench.main([*argv, "--cache", str(tmp_path / "elsewhere")]) == 0
    assert len(fake_run.calls) == 5  # base has another cache now; own keeps its own
    assert fake_run.calls[-1][1]["CUTPOINT_CACHE_DIR"] == str(tmp_path / "elsewhere")


def test_ctrl_break_is_a_ctrl_c_on_windows_only(monkeypatch):
    fake_break = signal.SIGINT  # stands in for signal.SIGBREAK, which only Windows has
    monkeypatch.setattr(bench.signal, "SIGBREAK", fake_break, raising=False)
    before = signal.getsignal(fake_break)
    monkeypatch.setattr(bench.sys, "platform", "linux")
    with bench.break_as_interrupt():
        assert signal.getsignal(fake_break) is before
    monkeypatch.setattr(bench.sys, "platform", "win32")
    with bench.break_as_interrupt():
        handler = signal.getsignal(fake_break)
        assert handler is bench._raise_interrupt
        with pytest.raises(KeyboardInterrupt):
            handler(fake_break, None)
    assert signal.getsignal(fake_break) is before  # restored


def test_interrupt_while_taking_the_lock_or_writing_the_report_frees_it(
    tmp_path, monkeypatch, fake_run
):
    lock = tmp_path / "x.lock"

    def broken_fdopen(fd, *a, **kw):
        os.close(fd)
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(bench.os, "fdopen", broken_fdopen)
        with pytest.raises(KeyboardInterrupt), bench.bench_lock(lock):
            pass
    assert not lock.exists()

    manifest = write_manifest(tmp_path, [source("a")])

    def interrupted(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(bench, "render_report", interrupted)
    assert bench.main([str(manifest), "--sample-interval", "0"]) == 130
    assert not bench.machine_lock().exists()
