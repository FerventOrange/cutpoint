"""``cutpoint rerank``: a project re-ranked with another style from its caches (#902).

Synthetic projects only: stand-in source files with hand-written caches (loudness, transcript,
scene cuts) and a made-up highlights.json and events.json. No ranking pins a score, so the
tests hold in Cutpoint full and lite alike.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import envelope
from cutpoint import analyze, cache, cli, folders, rerank, runstate
from cutpoint.errors import FolderBusyError, NeedsAnalyzeError
from cutpoint.sources import TRANSCRIPT_MODELS
from cutpoint.styles import get_style
from cutpoint.transcribe import transcript_name

SPAN = 240.0
WORDS = ("haha", "no", "way", "that", "worked", "go", "nice", "wow", "okay", "run")


# --------------------------------------------------------------------------- fixtures
def write_caches(src: Path, *, loud: bool = True, said: bool = True, scenes: bool = True) -> None:
    """The caches an analysis leaves for ``src``: loudness with a peak every 40 s, a word every
    0.4 s in 6-word bursts every 7 s, a scene cut every 20 s."""
    cdir = cache.source_dir(src)
    t = np.arange(1, int(SPAN * 10) + 1) * 0.1
    if loud:
        m = np.full(t.size, -32.0)
        for i, c in enumerate(np.arange(20.0, SPAN - 10, 40.0)):
            m += (8.0 + i) * np.exp(-(((t - c) / 1.5) ** 2))
        cache.save_json(cdir / "loudness.json", {"t": t.round(2).tolist(), "m": m.tolist()})
    if said:
        segs = []
        for k, a in enumerate(np.arange(3.0, SPAN - 5, 7.0)):
            ws = [
                {
                    "word": WORDS[(k + i) % len(WORDS)],
                    "start": float(a + 0.4 * i),
                    "end": float(a + 0.4 * i + 0.3),
                    "probability": 0.9,
                }
                for i in range(6)
            ]
            text = " ".join(w["word"] for w in ws)
            segs.append({"start": ws[0]["start"], "end": ws[-1]["end"], "text": text, "words": ws})
        cache.save_json(cdir / transcript_name(TRANSCRIPT_MODELS[0]), segs)
    if scenes:
        cache.save_json(cdir / analyze.SCENES_FILE, [float(x) for x in np.arange(10.0, SPAN, 20)])


def make_project(
    home: Path,
    tmp_path: Path,
    n: int = 2,
    style: str = "gaming",
    layers: tuple[str, ...] = ("audio", "ocr"),
    **settings: Any,
) -> Path:
    """Project ``p`` of ``n`` sources with their caches, analysed (re-ranked once) with
    ``style``; its events.json lists ``layers``."""
    srcs = []
    for i in range(n):
        src = tmp_path / "footage" / f"s{i}.mp4"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"x")
        write_caches(src)
        srcs.append(src)
    project = home / "projects" / "p"
    project.mkdir(parents=True)
    st = get_style(style)
    hl = {
        "version": 1,
        "style": st.name,
        "style_kind": st.kind,
        "style_base": st.base,
        "created": None,
        "settings": {
            "scenes": True,
            "whisper_model": TRANSCRIPT_MODELS[0],
            "vision_requested": False,
            "inputs": [str(s) for s in srcs],
            "must_include_crash": st.must_include_crash,
            **settings,
        },
        "timings": {"total_s": 900.0},
        "suggested_top": 0,
        "sources": [
            {"path": str(s), "duration": SPAN, "width": 1920, "height": 1080, "fps": 30.0}
            for s in srcs
        ],
        "candidates": [],
    }
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    ev = {"settings": {"layers": list(layers)}, "sources": [str(s) for s in srcs], "events": []}
    (project / "events.json").write_text(json.dumps(ev), encoding="utf-8")
    seeded = rerank.ranked(project, hl, st)
    seeded["settings"].pop("reranked", None)
    (project / "highlights.json").write_text(json.dumps(seeded), encoding="utf-8")
    return project


@pytest.fixture(autouse=True)
def lite_layers(monkeypatch):
    """No events extra (as in CI): the layer check plans nothing unless a test says so."""
    monkeypatch.setattr(rerank, "_events_extra", lambda: False)


def read(project: Path) -> dict[str, Any]:
    return json.loads((project / "highlights.json").read_text(encoding="utf-8"))


def features_of(project: Path, style: str) -> set[str]:
    return {m.feature for m in rerank.readiness(project, read(project), get_style(style)).missing}


def ids(hl: dict[str, Any]) -> list[str]:
    return [c["id"] for c in hl["candidates"]]


# --------------------------------------------------------------------------- readiness
def test_a_project_with_every_cache_is_ready(home, tmp_path):
    project = make_project(home, tmp_path)
    for style in ("gaming", "funny", "commentary", "action", "sport"):
        assert rerank.readiness(project, read(project), get_style(style)).ready, style


def test_a_360_project_needs_analyze(home, tmp_path):
    project = make_project(home, tmp_path)
    (project / "pano.json").write_text("{}", encoding="utf-8")
    assert "360" in features_of(project, "funny")


def test_an_action_cam_style_needs_analyze(home, tmp_path):
    project = make_project(home, tmp_path)
    assert features_of(project, "action-cam") == {"action_cam"}


def test_a_change_of_style_kind_needs_analyze(home, tmp_path):
    project = make_project(home, tmp_path)
    hl = read(project)
    hl["style_kind"] = "action-cam"
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    assert features_of(project, "funny") == {"style_kind"}


@pytest.mark.parametrize("entry", [{"offset": 120.0}, {"read_errors": [{"failure_time": 60}]}])
def test_a_source_read_in_parts_needs_analyze(home, tmp_path, entry):
    project = make_project(home, tmp_path)
    hl = read(project)
    hl["sources"][1].update(entry)
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    (missing,) = rerank.readiness(project, hl, get_style("funny")).missing
    assert (missing.feature, missing.source) == ("recovered", hl["sources"][1]["path"])


def test_an_offline_source_needs_analyze_and_leaves_no_cache_folder(home, tmp_path):
    project = make_project(home, tmp_path)
    gone = Path(read(project)["sources"][0]["path"])
    gone.unlink()
    before = sorted(cache.cache_root().iterdir())
    (missing,) = rerank.readiness(project, read(project), get_style("funny")).missing
    assert (missing.feature, missing.source) == ("offline", str(gone))
    assert sorted(cache.cache_root().iterdir()) == before


def test_source_dir_without_create_names_the_folder_only(home, tmp_path):
    src = tmp_path / "new.mp4"
    src.write_bytes(b"x")
    folder = cache.source_dir(src, create=False)
    assert not folder.exists()
    assert cache.source_dir(src) == folder and folder.is_dir()


@pytest.mark.parametrize(("kind", "kept"), [("loudness", {"said": True}), ("transcript", {})])
def test_missing_loudness_or_transcript_needs_analyze(home, tmp_path, monkeypatch, kind, kept):
    project = make_project(home, tmp_path)
    src = Path(read(project)["sources"][0]["path"])
    cdir = cache.source_dir(src)
    for f in cdir.glob("loudness.json" if kind == "loudness" else "transcript-*.json"):
        f.unlink()
    monkeypatch.setattr(rerank, "_has_audio", lambda s: True)
    (missing,) = rerank.readiness(project, read(project), get_style("funny")).missing
    assert (missing.feature, missing.source) == (kind, str(src))


def test_a_source_without_audio_needs_no_loudness_or_transcript(home, tmp_path, monkeypatch):
    project = make_project(home, tmp_path)
    cdir = cache.source_dir(Path(read(project)["sources"][0]["path"]))
    for f in [*cdir.glob("loudness.json"), *cdir.glob("transcript-*.json")]:
        f.unlink()
    monkeypatch.setattr(rerank, "_has_audio", lambda s: False)
    assert rerank.readiness(project, read(project), get_style("funny")).ready


def test_scenes_are_needed_only_by_a_style_that_weighs_them(home, tmp_path):
    project = make_project(home, tmp_path)
    src = Path(read(project)["sources"][0]["path"])
    (cache.source_dir(src) / analyze.SCENES_FILE).unlink()
    assert analyze.scores_scenes(get_style("gaming"))
    assert not analyze.scores_scenes(get_style("funny"))
    assert features_of(project, "gaming") == {"scenes"}  # weighs scenes: a full pass
    assert features_of(project, "funny") == set()  # edge windows, decoded when needed


def test_a_long_source_needs_its_full_scene_pass_for_any_style(home, tmp_path):
    project = make_project(home, tmp_path)
    hl = read(project)
    hl["sources"][0]["duration"] = analyze.LONG_FILE_S + 60.0  # no edge windows that long
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    (cache.source_dir(Path(hl["sources"][0]["path"])) / analyze.SCENES_FILE).unlink()
    assert features_of(project, "funny") == {"scenes"}


def test_scenes_off_needs_no_scene_pass(home, tmp_path):
    project = make_project(home, tmp_path, scenes=False)
    src = Path(read(project)["sources"][0]["path"])
    (cache.source_dir(src) / analyze.SCENES_FILE).unlink()
    assert features_of(project, "gaming") == set()


def test_an_event_layer_the_style_plans_and_did_not_run_needs_analyze(home, tmp_path, monkeypatch):
    project = make_project(home, tmp_path)  # gameplay layers: audio, ocr
    monkeypatch.setattr(rerank, "_events_extra", lambda: True)
    plan = analyze.LayerPlan(
        layers={"crashes": ("crash",), "reactions": ("audio", "voice"), "keyframes": ("visual",)}
    )
    asked: list[str] = []

    def plan_layers(style, sources, skip, cat=None, visual_index=False):
        asked.append(cat)
        assert sources == []  # no telemetry probe of the sources
        return plan

    monkeypatch.setattr(analyze, "plan_layers", plan_layers)
    got = features_of(project, "action")
    assert got == {"layer:voice", "layer:visual", "layer:crash"}
    assert asked == ["action"]
    asked.clear()
    assert features_of(project, "funny") == set()  # talk: audio ran, no plan needed
    assert asked == []


def test_no_events_extra_needs_no_layer(home, tmp_path):
    project = make_project(home, tmp_path, layers=())
    assert features_of(project, "action") == set()


def test_dialogue_never_read_is_a_note_only(home, tmp_path):
    project = make_project(home, tmp_path, style="funny", layers=("audio",))
    got = rerank.readiness(project, read(project), get_style("gaming"))
    assert got.ready
    (note,) = got.notes
    assert "Dialogue was never read" in note
    ev = json.loads((project / "events.json").read_text(encoding="utf-8"))
    ev["events"] = [{"kind": "text", "type": "dialogue", "source": "x", "start": 1.0}]
    (project / "events.json").write_text(json.dumps(ev), encoding="utf-8")
    assert rerank.readiness(project, read(project), get_style("gaming")).notes == []
    assert rerank.readiness(project, read(project), get_style("funny")).notes == []


# --------------------------------------------------------------------------- ranking and ids
def test_the_preview_equals_the_saved_ranking_and_writes_nothing(home, tmp_path):
    project = make_project(home, tmp_path)
    files = {p.name: p.read_bytes() for p in project.iterdir()}
    preview = rerank.rerank(project, get_style("funny"), dry_run=True)
    assert not preview.saved and preview.backup is None
    assert {p.name: p.read_bytes() for p in project.iterdir()} == files  # nothing written
    saved = rerank.rerank(project, get_style("funny"))
    assert saved.saved
    assert saved.after["candidates"] == preview.after["candidates"]
    assert saved.after["suggested_top"] == preview.after["suggested_top"]
    assert read(project)["candidates"] == preview.after["candidates"]
    assert read(project)["style"] == "funny"


def test_save_keeps_the_earlier_file_as_the_backup(home, tmp_path):
    project = make_project(home, tmp_path)
    before = (project / "highlights.json").read_bytes()
    res = rerank.rerank(project, get_style("commentary"))
    assert res.backup == project / rerank.BACKUP
    assert (project / rerank.BACKUP).read_bytes() == before
    rerank.rerank(project, get_style("funny"))  # one backup: the file just before
    assert json.loads((project / rerank.BACKUP).read_bytes())["style"] == "commentary"


def test_the_same_style_keeps_every_id(home, tmp_path):
    project = make_project(home, tmp_path)
    hl = read(project)
    res = rerank.rerank(project, get_style("gaming"))
    assert ids(res.after) == ids(hl)
    assert [c["start"] for c in res.after["candidates"]] == [c["start"] for c in hl["candidates"]]
    assert res.after["settings"]["last_candidate_id"] == hl["settings"]["last_candidate_id"]


def test_unchanged_spans_keep_their_id_and_new_ones_come_after_the_last(home, tmp_path):
    project = make_project(home, tmp_path)
    preview = rerank.rerank(project, get_style("funny"), dry_run=True).after
    target = preview["candidates"][2]  # a span the funny ranking has
    hl = read(project)
    hl["candidates"][0].update(
        id="h007", source=target["source"], start=target["start"], end=target["end"]
    )
    hl["candidates"] = [hl["candidates"][0]] + [
        c for c in hl["candidates"][1:] if c["id"] != "h007"
    ]
    hl["settings"]["last_candidate_id"] = 500  # ids up to here were given once
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    after = rerank.rerank(project, get_style("funny")).after
    by_span = {analyze.span_key(c): c["id"] for c in after["candidates"]}
    assert by_span[analyze.span_key(target)] == "h007"
    others = sorted(int(i[1:]) for i in ids(after) if i != "h007")
    assert others == list(range(501, 501 + len(after["candidates"]) - 1))
    assert after["settings"]["last_candidate_id"] == 500 + len(after["candidates"]) - 1
    assert len(set(ids(after))) == len(after["candidates"])


def test_the_new_style_replaces_the_style_fields(home, tmp_path):
    project = make_project(home, tmp_path)
    after = rerank.rerank(project, get_style("commentary")).after
    style = get_style("commentary")
    assert (after["style"], after["style_kind"], after["style_base"]) == (
        "commentary",
        style.kind,
        style.base,
    )
    st = after["settings"]
    assert st["weights"] == style.weights and st["target_total"] == style.target_total
    assert st["reranked"] == {"from": "gaming", "requested": None}
    assert st["style_requested"] == "commentary"
    assert after["timings"]["total_s"] == 900.0 and "rerank_s" in after["timings"]
    assert {c["source_index"] for c in after["candidates"]} <= {0, 1}


def test_an_auto_project_keeps_its_detection_and_records_the_choice(home, tmp_path):
    detection = {"category": "gameplay", "emphasis": "hype", "style": "gaming"}
    project = make_project(home, tmp_path, style_requested="auto")
    hl = read(project)
    hl["detection"] = detection
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    after = rerank.rerank(project, get_style("funny")).after
    assert after["detection"] == detection
    assert after["settings"]["style_requested"] == "funny"
    assert after["settings"]["reranked"] == {"from": "gaming", "requested": "auto"}


def test_an_analyze_override_of_must_include_crash_is_kept(home, tmp_path):
    project = make_project(home, tmp_path, must_include_crash=0.9)
    after = rerank.rerank(project, get_style("action")).after
    assert after["settings"]["must_include_crash"] == 0.9
    plain = make_project(home / "other", tmp_path / "other")
    assert rerank.rerank(plain, get_style("action")).after["settings"]["must_include_crash"] == (
        get_style("action").must_include_crash
    )


def test_a_save_waits_for_no_other_run_but_a_preview_reads(home, tmp_path):
    project = make_project(home, tmp_path)
    with folders.swap_lock(project / runstate.LOCK_NAME, project):
        assert rerank.rerank(project, get_style("funny"), dry_run=True).after["style"] == "funny"
    lock = folders.lock_path(project / runstate.LOCK_NAME)
    lock.write_text(json.dumps({"pid": 1, "host": "elsewhere", "token": "t"}), encoding="utf-8")
    with pytest.raises(FolderBusyError):
        rerank.rerank(project, get_style("funny"))
    assert read(project)["style"] == "gaming"


# --------------------------------------------------------------------------- refused, --analyze
def test_refused_names_the_missing_features_and_the_command(home, tmp_path):
    project = make_project(home, tmp_path)
    with pytest.raises(NeedsAnalyzeError) as got:
        rerank.rerank(project, get_style("action-cam"))
    err = got.value
    assert err.command == "cutpoint rerank p --style action-cam --analyze"
    assert [m["feature"] for m in err.missing] == ["action_cam"]
    assert err.command in str(err)
    assert not (project / rerank.BACKUP).exists()


def test_analyze_runs_reanalyze_with_the_style_when_refused(home, tmp_path, monkeypatch):
    project = make_project(home, tmp_path)
    (project / "pano.json").write_text("{}", encoding="utf-8")
    before = (project / "highlights.json").read_bytes()
    called: list[Any] = []

    def fake_reanalyze(p: Path, style=None) -> Path:
        called.append((p, style.name))
        hl = read(p)
        hl.update(style=style.name, candidates=[])
        (p / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
        return p / "highlights.json"

    monkeypatch.setattr(analyze, "reanalyze", fake_reanalyze)
    with pytest.raises(NeedsAnalyzeError):  # a preview never analyzes
        rerank.rerank(project, get_style("action-cam"), dry_run=True, analyze_missing=True)
    assert called == []
    res = rerank.rerank(project, get_style("action-cam"), analyze_missing=True)
    assert called == [(project, "action-cam")]
    assert res.analyzed and res.saved and res.after["style"] == "action-cam"
    assert (project / rerank.BACKUP).read_bytes() == before


def test_analyze_is_not_run_when_the_fast_path_can(home, tmp_path, monkeypatch):
    project = make_project(home, tmp_path)
    monkeypatch.setattr(analyze, "reanalyze", lambda *a, **k: pytest.fail("analyzed"))
    assert not rerank.rerank(project, get_style("funny"), analyze_missing=True).analyzed


def _captured_analyze(monkeypatch) -> dict[str, Any]:
    got: dict[str, Any] = {}

    def fake(target, style, out_dir, **kw):
        got.update(kw, style=style, target=target)
        return out_dir / "highlights.json"

    monkeypatch.setattr(analyze, "analyze", fake)
    return got


def test_reanalyze_with_a_style_runs_its_layers_and_records_the_choice(home, tmp_path, monkeypatch):
    detection = {"category": "gameplay", "emphasis": "hype"}
    project = make_project(home, tmp_path, style_requested="auto")
    hl = read(project)
    hl["detection"] = detection
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    got = _captured_analyze(monkeypatch)
    analyze.reanalyze(project, get_style("funny"))
    assert got["style"].name == "funny"
    assert got["layers"] is True
    assert got["reranked"] == {"from": "gaming", "requested": "auto"}
    assert got["detection"] == detection  # kept as a record; the layers follow the style


def test_reanalyze_of_a_reranked_project_keeps_the_users_style(home, tmp_path, monkeypatch):
    project = make_project(home, tmp_path, style_requested="auto")
    hl = read(project)
    hl["detection"] = {"category": "gameplay"}
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    rerank.rerank(project, get_style("funny"))
    got = _captured_analyze(monkeypatch)
    analyze.reanalyze(project)  # e.g. cutpoint events
    assert got["style"].name == "funny"
    assert got["reranked"] == {"from": "gaming", "requested": "auto"}
    assert got["detection"] == {"category": "gameplay"}
    assert got["layers"] is False


def test_reanalyze_of_a_360_project_with_a_style_leaves_the_layers(home, tmp_path, monkeypatch):
    project = make_project(home, tmp_path)
    (project / "pano.json").write_text("{}", encoding="utf-8")
    got = _captured_analyze(monkeypatch)
    analyze.reanalyze(project, get_style("action-cam"))
    assert got["layers"] is False and got["crash_moments"] is False


def test_a_plain_reanalyze_is_unchanged(home, tmp_path, monkeypatch):
    project = make_project(home, tmp_path)
    got = _captured_analyze(monkeypatch)
    analyze.reanalyze(project)
    assert got["style"].name == "gaming"
    assert got["layers"] is False and got["reranked"] is None and got["detection"] is None


# --------------------------------------------------------------------------- the command
def _json(capsys, argv: list[str]) -> tuple[int, dict]:
    code = cli.main([*argv, "--json"])
    return code, envelope(capsys.readouterr().out)


def test_rerank_json_preview(home, tmp_path, capsys):
    project = make_project(home, tmp_path)
    code, doc = _json(capsys, ["rerank", "p", "--style", "funny", "--dry-run", "--top", "3"])
    assert code == 0 and doc["ok"] and doc["command"] == "rerank"
    out = doc["outputs"]
    assert doc["project"] == str(project)
    assert out["saved"] is False and out["backup"] is None and out["analyzed"] is False
    assert (out["style"], out["from"]) == ("funny", "gaming")
    assert len(out["top"]) == len(out["before"]) == 3
    assert set(out["top"][0]) == {"rank", "id", "source", "start", "end", "score", "label", "was"}
    assert out["candidates"] > 0 and isinstance(out["suggested_top"], int)
    assert out["highlights"] == str(project / "highlights.json")
    assert read(project)["style"] == "gaming"


def test_rerank_json_save(home, tmp_path, capsys):
    project = make_project(home, tmp_path)
    code, doc = _json(capsys, ["rerank", "p", "--style", "commentary"])
    assert code == 0 and doc["outputs"]["saved"] is True
    assert doc["outputs"]["backup"] == str(project / rerank.BACKUP)
    assert read(project)["style"] == "commentary"


def test_rerank_json_needs_analyze(home, tmp_path, capsys):
    make_project(home, tmp_path)
    code, doc = _json(capsys, ["rerank", "p", "--style", "action-cam"])
    assert code == 1 and not doc["ok"]
    assert doc["error"]["code"] == "needs_analyze"
    assert doc["outputs"]["missing"] == [
        {"feature": "action_cam", "detail": doc["outputs"]["missing"][0]["detail"], "source": None}
    ]
    assert doc["outputs"]["analyze_command"] == "cutpoint rerank p --style action-cam --analyze"


def test_rerank_without_a_project_is_not_found(home, capsys):
    code, doc = _json(capsys, ["rerank", "nothing", "--style", "funny"])
    assert code == 1 and doc["error"]["code"] == "not_found"


def test_rerank_refuses_auto_and_dry_run_with_analyze(home, capsys):
    with pytest.raises(SystemExit) as got:
        cli.main(["rerank", "p", "--style", "auto"])
    assert got.value.code == 2
    with pytest.raises(SystemExit) as got:
        cli.main(["rerank", "p", "--style", "funny", "--dry-run", "--analyze"])
    assert got.value.code == 2


def test_rerank_prints_the_new_ranking(home, tmp_path, capsys):
    make_project(home, tmp_path)
    assert cli.main(["rerank", "p", "--style", "funny", "--dry-run", "--top", "2"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("gaming -> funny (preview, nothing saved")
    assert len(out) == 4  # header, counts, two moments
