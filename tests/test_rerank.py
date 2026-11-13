"""``cutpoint rerank``: a project re-ranked with another style from its caches (#902).

Synthetic projects only: stand-in source files with hand-written caches (loudness, transcript,
scene cuts) and a made-up highlights.json and events.json. No ranking pins a score, so the
tests hold in Cutpoint full and lite alike.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from conftest import envelope
from cutpoint import analyze, cache, cli, folders, jsonout, rerank, runstate
from cutpoint.assistant.providers import AUTO_SERVER_MODEL
from cutpoint.errors import FolderBusyError, NeedsAnalyzeError, StaleRankingError
from cutpoint.ffmpeg_utils import MediaInfo
from cutpoint.sources import TRANSCRIPT_MODELS
from cutpoint.styles import get_style
from cutpoint.transcribe import transcript_name
from test_analyze_steps import run  # noqa: F401 (the fixture: analyze with its fakes)

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


def test_an_offline_refusal_asks_to_reconnect_the_footage(home, tmp_path):
    project = make_project(home, tmp_path)
    style = get_style("funny")
    gone = rerank.Missing("offline", "a.mp4 is not reachable", "a.mp4")
    only = str(rerank._refused(project, style, [gone]))
    assert "Reconnect the footage and run the rerank again" in only
    assert "--analyze" not in only
    both = str(rerank._refused(project, style, [gone, rerank.Missing("loudness", "no loudness")]))
    assert "Reconnect the footage, then run: " in both and "--analyze" in both


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


@pytest.mark.parametrize("scenes", [True, False])
def test_a_source_with_nothing_cached_needs_analyze(home, tmp_path, monkeypatch, scenes):
    """No audio and no scene cuts (scene detection off, or a source too long for edge
    windows): cached_grid_parts has nothing to build its grid from."""
    project = make_project(home, tmp_path, scenes=scenes)
    hl = read(project)
    if scenes:
        hl["sources"][0]["duration"] = analyze.LONG_FILE_S + 60.0
        (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    src = Path(hl["sources"][0]["path"])
    for f in cache.source_dir(src).iterdir():
        f.unlink()
    monkeypatch.setattr(rerank, "_has_audio", lambda s: False)
    assert features_of(project, "funny") == ({"scenes"} if scenes else {"caches"})
    if not scenes:
        (missing,) = rerank.readiness(project, read(project), get_style("funny")).missing
        assert (missing.feature, missing.source) == ("caches", str(src))
    assert [p.index for p in analyze.cached_grid_parts(read(project), get_style("funny"))] == [1]


def test_ranked_never_drops_a_source_without_a_grid(home, tmp_path, monkeypatch):
    project = make_project(home, tmp_path)
    hl = read(project)
    real = analyze.cached_grid_parts
    monkeypatch.setattr(analyze, "cached_grid_parts", lambda h, st: real(h, st)[1:])
    with pytest.raises(NeedsAnalyzeError) as got:
        rerank.ranked(project, hl, get_style("funny"))
    assert got.value.missing == [
        {
            "feature": "caches",
            "detail": "s0.mp4 has nothing cached",
            "source": hl["sources"][0]["path"],
        }
    ]


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

    def plan_layers(style, sources, skip, cat=None, visual_index=False, cuda=None):
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


def test_an_injected_cuda_answer_is_used_instead_of_torch(home, tmp_path, monkeypatch):
    project = make_project(home, tmp_path)  # gameplay layers: audio, ocr
    monkeypatch.setattr(rerank, "_events_extra", lambda: True)
    monkeypatch.setattr(analyze, "EVENTS_MODULES", ())  # the extra is "installed"

    def boom() -> bool:
        raise AssertionError("torch's own CUDA check ran")

    monkeypatch.setattr(analyze, "_cuda", boom)
    calls: list[int] = []

    def probe() -> bool:
        calls.append(1)
        return False

    miss = rerank.readiness(project, read(project), get_style("action"), probe).missing
    assert calls, "the injected CUDA answer was never asked"
    assert "layer:visual" not in {m.feature for m in miss}  # no GPU: the visual layer is skipped


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


def _by_span(hl: dict[str, Any]) -> dict[tuple[str, float, float], str]:
    return {analyze.span_key(c): c["id"] for c in hl["candidates"]}


def _a_to_b(home: Path, tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """A gaming project re-ranked to funny: the project, the gaming ranking and the funny one
    (which drops gaming spans and adds spans of its own)."""
    project = make_project(home, tmp_path)
    a = read(project)
    b = rerank.rerank(project, get_style("funny")).after
    assert set(_by_span(a)) - set(_by_span(b)), "funny drops no gaming span"
    assert b["settings"]["last_candidate_id"] > a["settings"]["last_candidate_id"]
    return project, a, b


def test_switching_style_and_back_restores_the_ids(home, tmp_path):
    """#1005: gaming -> funny -> gaming gives every gaming span its gaming id back, from the
    backup for the spans funny dropped; last_candidate_id stays where funny took it."""
    project, a, b = _a_to_b(home, tmp_path)
    preview = rerank.rerank(project, get_style("gaming"), dry_run=True).after
    back = rerank.rerank(project, get_style("gaming")).after
    assert back["candidates"] == preview["candidates"]  # the preview gives the ids back too
    assert back["settings"]["last_candidate_id"] == preview["settings"]["last_candidate_id"]
    assert back["settings"]["rerank_backup"] == preview["settings"]["rerank_backup"]
    assert back["settings"]["rerank_backup"] == analyze.backup_digest(
        (project / rerank.BACKUP).read_bytes()
    )
    assert _by_span(back) == _by_span(a)
    assert back["settings"]["last_candidate_id"] == b["settings"]["last_candidate_id"]
    assert json.loads((project / rerank.BACKUP).read_bytes())["style"] == "funny"


def test_a_backup_id_another_span_has_is_not_given_again(home, tmp_path):
    """#1005: the current ranking's ids come first; a backup id it gives to another span is
    not reused, and the span that had it there gets a new id."""
    project, a, b = _a_to_b(home, tmp_path)
    dropped = next(k for k in _by_span(a) if k not in _by_span(b))
    taken = _by_span(a)[dropped]
    other = next(c for c in a["candidates"] if analyze.span_key(c) != dropped)
    kept = b["candidates"][0]  # the funny ranking names another gaming span with that id
    kept.update(id=taken, source=other["source"], start=other["start"], end=other["end"])
    (project / "highlights.json").write_text(json.dumps(b), encoding="utf-8")
    back = rerank.rerank(project, get_style("gaming")).after
    got = _by_span(back)
    assert got[analyze.span_key(kept)] == taken
    assert int(got[dropped][1:]) > b["settings"]["last_candidate_id"]
    assert len(set(got.values())) == len(back["candidates"])


@pytest.mark.parametrize("backup", [b"{not json", b'{"candidates": [{"id": "h001"}]}', b"[]"])
def test_a_backup_that_cannot_be_read_is_ignored(home, tmp_path, backup):
    project, _, b = _a_to_b(home, tmp_path)
    (project / rerank.BACKUP).write_bytes(backup)
    back = rerank.rerank(project, get_style("gaming")).after
    got, was, last = _by_span(back), _by_span(b), b["settings"]["last_candidate_id"]
    for key, i in got.items():  # funny's ids kept, the spans only gaming had are new
        assert i == was[key] if key in was else int(i[1:]) > last
    assert len(set(got.values())) == len(back["candidates"])


def test_a_linked_backup_with_other_ids_for_a_shared_span_is_ignored(home, tmp_path):
    """#1005 review F2: highlights.json numbered anew by hand, its link to the backup kept: a
    span both files have has another id in each, so the backup is stale and gives no id
    back."""
    project, a, b = _a_to_b(home, tmp_path)
    shared = a["candidates"][0]
    b["candidates"].append(
        {**b["candidates"][-1], **{k: shared[k] for k in ("source", "start", "end")}}
    )
    for n, c in enumerate(b["candidates"], 1):  # renumbered from h001, the shared span too
        c["id"] = f"h{n:03d}"
    assert _by_span(b)[analyze.span_key(shared)] != shared["id"]
    (project / "highlights.json").write_text(json.dumps(b), encoding="utf-8")
    back = rerank.rerank(project, get_style("gaming")).after
    got, was = _by_span(back), _by_span(b)
    last = max(b["settings"]["last_candidate_id"], a["settings"]["last_candidate_id"])
    for key, i in got.items():  # no id from the backup: kept from b, or new
        assert i == was[key] if key in was else int(i[1:]) > last
    assert len(set(got.values())) == len(back["candidates"])


def _renumbered(b: dict[str, Any], *, linked: bool) -> dict[str, Any]:
    """``b`` as a fresh analysis would leave it: numbered from h001, one moment dropped since
    (its id free again) and, unless ``linked``, no link to the backup."""
    fresh = json.loads(json.dumps(b))
    fresh["candidates"] = fresh["candidates"][:-1]
    for n, c in enumerate(fresh["candidates"], 1):
        c["id"] = f"h{n:03d}"
    fresh["settings"]["last_candidate_id"] = len(fresh["candidates"]) + 1
    if not linked:
        fresh["settings"].pop("rerank_backup")
    return fresh


@pytest.mark.parametrize("linked", [False, True])
def test_a_backup_the_ranking_is_not_linked_to_gives_no_id(home, tmp_path, linked):
    """#1005 review F5: after a fresh analysis (no settings.rerank_backup) that shares no span
    with the backup, the backup gives no id back, even an id that is free again. The same
    file still linked to it (the control) does."""
    project, a, b = _a_to_b(home, tmp_path)
    fresh = _renumbered(b, linked=linked)
    (project / "highlights.json").write_text(json.dumps(fresh), encoding="utf-8")
    back = rerank.rerank(project, get_style("gaming")).after
    got, was = _by_span(back), _by_span(fresh)
    restored = {k: i for k, i in got.items() if k not in was and _by_span(a).get(k) == i}
    assert bool(restored) is linked
    if not linked:
        last = fresh["settings"]["last_candidate_id"]
        assert all(i == was[k] if k in was else int(i[1:]) > last for k, i in got.items())
    assert len(set(got.values())) == len(back["candidates"])


def test_a_plain_reanalyze_keeps_the_link_to_the_backup(home, tmp_path, monkeypatch):
    """#1005 review F5: the re-analysis of cutpoint events between two reranks records the
    link again, so the backup still gives ids back; rerank --analyze links its own backup."""
    project, _, b = _a_to_b(home, tmp_path)
    got = _captured_analyze(monkeypatch)
    analyze.reanalyze(project)
    assert got["rerank_backup"] == b["settings"]["rerank_backup"]
    monkeypatch.setattr(
        rerank, "readiness", lambda *x: rerank.Readiness([rerank.Missing("x", "y")])
    )
    raw = (project / "highlights.json").read_bytes()
    rerank.rerank(project, get_style("gaming"), analyze_missing=True)
    assert got["rerank_backup"] == analyze.backup_digest(raw)
    assert (project / rerank.BACKUP).read_bytes() == raw


def test_rerank_analyze_restores_the_ids_from_the_backup_it_replaces(home, tmp_path, monkeypatch):
    """#1005 with --analyze: the backup before this rerank's own (gaming) gives the ids back;
    last_candidate_id stays where funny took it."""
    project, a, b = _a_to_b(home, tmp_path)
    monkeypatch.setattr(
        rerank, "readiness", lambda *x: rerank.Readiness([rerank.Missing("x", "y")])
    )
    got = _captured_analyze(monkeypatch)
    rerank.rerank(project, get_style("gaming"), analyze_missing=True)
    assert json.loads((project / rerank.BACKUP).read_bytes())["style"] == "funny"
    assert got["last_candidate_id"] == b["settings"]["last_candidate_id"]
    assert {k: i for k, i in got["ids"].items() if k in _by_span(a)} == _by_span(a)
    assert {k: i for k, i in got["ids"].items() if k in _by_span(b)} == _by_span(b)


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


def test_a_save_with_the_previewed_key_writes(home, tmp_path):
    project = make_project(home, tmp_path)
    token = rerank.key((project / "highlights.json").read_bytes())
    res = rerank.rerank(project, get_style("funny"), expect=token)
    assert res.saved and read(project)["style"] == "funny"
    assert (project / rerank.BACKUP).exists()


def test_a_save_with_a_stale_key_raises_before_any_write(home, tmp_path):
    project = make_project(home, tmp_path)
    token = rerank.key((project / "highlights.json").read_bytes())
    rerank.rerank(project, get_style("commentary"))  # another ranking ran meanwhile
    files = {p.name: p.read_bytes() for p in project.iterdir()}
    with pytest.raises(StaleRankingError):
        rerank.rerank(project, get_style("funny"), expect=token)
    assert {p.name: p.read_bytes() for p in project.iterdir()} == files
    assert read(project)["style"] == "commentary"


def test_a_stale_key_is_the_stale_error_code():
    assert jsonout.error_code(StaleRankingError("x")) == "stale"


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
    before = (project / "highlights.json").read_bytes()
    called: list[Any] = []

    def fake_reanalyze(p: Path, style=None, *, backup=None, rerank_backup=None) -> Path:
        called.append((p, style.name))
        assert (p / rerank.BACKUP).read_bytes() == before  # written before analyze replaces it
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


def test_a_360_project_names_360_analyze_and_analyze_refuses_it(home, tmp_path, monkeypatch):
    project = make_project(home, tmp_path)
    (project / "pano.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(analyze, "reanalyze", lambda *a, **k: pytest.fail("analyzed"))
    with pytest.raises(NeedsAnalyzeError) as got:
        rerank.rerank(project, get_style("funny"), analyze_missing=True)
    assert got.value.command == "cutpoint 360 analyze p --style funny"
    assert got.value.command in str(got.value)
    assert not (project / rerank.BACKUP).exists()
    assert rerank.analyze_command(project, get_style("action-cam")) == (
        "cutpoint 360 analyze p --style action-cam"
    )
    args = cli.build_parser().parse_args(["360", "analyze", "p", "--style", "funny"])
    assert (args.project, args.style) == ("p", "funny")


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


def test_reanalyze_with_a_style_passes_the_earlier_ids(home, tmp_path, monkeypatch):
    project = make_project(home, tmp_path)
    hl = read(project)
    hl["settings"]["last_candidate_id"] = 300
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    got = _captured_analyze(monkeypatch)
    analyze.reanalyze(project, get_style("funny"))
    assert got["ids"] == {analyze.span_key(c): c["id"] for c in hl["candidates"]}
    assert got["last_candidate_id"] == 300
    got.clear()
    analyze.reanalyze(project)  # e.g. cutpoint events: the ids kept too (#1006)
    assert got["ids"] == {analyze.span_key(c): c["id"] for c in hl["candidates"]}
    assert got["last_candidate_id"] == 300


def test_rerank_analyze_keeps_the_ids_of_unchanged_spans(run, monkeypatch):  # noqa: F811
    """#902 F1: the analysis --analyze runs keeps the id of an unchanged span and gives a new
    span the next id after settings.last_candidate_id, as the fast path does."""
    run.go("funny")  # one candidate, 10-20 s
    hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
    (only,) = hl["candidates"]
    only["id"] = "h007"
    hl["settings"]["last_candidate_id"] = 40
    (run.proj / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")

    def two(src, style, tr, use_scenes, max_cand, timings, **kw):
        info = MediaInfo(
            src, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
        )
        cands = [
            analyze.Candidate(src, 10.0, 20.0, 15.0, 1.0, {"loudness": 1.0}, score=50.0),
            analyze.Candidate(src, 30.0, 40.0, 35.0, 1.0, {"loudness": 1.0}, score=60.0),
        ]
        return analyze.SourceResult(src, info, 0.0, 60.0, cands, [], None)

    monkeypatch.setattr(analyze, "analyze_source", two)
    monkeypatch.setattr(
        rerank, "readiness", lambda *a: rerank.Readiness([rerank.Missing("x", "y")])
    )
    res = rerank.rerank(run.proj, get_style("commentary"), analyze_missing=True)
    assert res.analyzed
    after = read(run.proj)
    by_start = {c["start"]: c["id"] for c in after["candidates"]}
    assert by_start == {10.0: "h007", 30.0: "h041"}
    assert after["settings"]["last_candidate_id"] == 41
    assert rerank.changes(res.before, res.after) == (["h041"], [])


def test_a_plain_reanalyze_keeps_the_ids_and_never_lowers_the_last(run, monkeypatch):  # noqa: F811
    """#1006: the re-analysis of cutpoint events keeps the id of an unchanged span, gives a
    new span the next id after settings.last_candidate_id and never lowers it."""
    run.go("funny")  # one candidate, 10-20 s
    hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
    (only,) = hl["candidates"]
    only["id"] = "h007"
    hl["settings"]["last_candidate_id"] = 40
    (run.proj / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    analyze.reanalyze(run.proj)
    after = read(run.proj)
    assert [c["id"] for c in after["candidates"]] == ["h007"]
    assert after["settings"]["last_candidate_id"] == 40

    def two(src, style, tr, use_scenes, max_cand, timings, **kw):
        info = MediaInfo(
            src, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
        )
        cands = [
            analyze.Candidate(src, 10.0, 20.0, 15.0, 1.0, {"loudness": 1.0}, score=50.0),
            analyze.Candidate(src, 30.0, 40.0, 35.0, 1.0, {"loudness": 1.0}, score=60.0),
        ]
        return analyze.SourceResult(src, info, 0.0, 60.0, cands, [], None)

    monkeypatch.setattr(analyze, "analyze_source", two)
    analyze.reanalyze(run.proj)
    after = read(run.proj)
    assert {c["start"]: c["id"] for c in after["candidates"]} == {10.0: "h007", 30.0: "h041"}
    assert after["settings"]["last_candidate_id"] == 41


def test_a_fresh_analyze_into_a_project_keeps_the_ids_and_never_lowers_the_last(run, monkeypatch):  # noqa: F811
    """#1009: analyze/edit again into a project (no ids given) keeps the id of an unchanged
    span, gives a new span the next id after settings.last_candidate_id, never lowers it."""
    run.go("funny")  # one candidate, 10-20 s
    hl = read(run.proj)
    (only,) = hl["candidates"]
    only["id"] = "h007"
    hl["settings"]["last_candidate_id"] = 40
    (run.proj / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    run.go("funny", layers=False)
    after = read(run.proj)
    assert [c["id"] for c in after["candidates"]] == ["h007"]
    assert after["settings"]["last_candidate_id"] == 40

    def two(src, style, tr, use_scenes, max_cand, timings, **kw):
        info = MediaInfo(
            src, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
        )
        cands = [
            analyze.Candidate(src, 10.0, 20.0, 15.0, 1.0, {"loudness": 1.0}, score=50.0),
            analyze.Candidate(src, 30.0, 40.0, 35.0, 1.0, {"loudness": 1.0}, score=60.0),
        ]
        return analyze.SourceResult(src, info, 0.0, 60.0, cands, [], None)

    monkeypatch.setattr(analyze, "analyze_source", two)
    run.go("funny", layers=False)
    after = read(run.proj)
    assert {c["start"]: c["id"] for c in after["candidates"]} == {10.0: "h007", 30.0: "h041"}
    assert after["settings"]["last_candidate_id"] == 41


def test_a_fresh_analyze_gives_a_span_its_id_from_the_linked_backup(run, monkeypatch):  # noqa: F811
    """#1009 review F1: analyze again into a re-ranked project: a span highlights.json lacks
    gets its id from the backup it is linked to, and the backup's last id counts."""
    run.go("funny")  # one candidate, 10-20 s
    hl = read(run.proj)
    (only,) = hl["candidates"]
    only["id"] = "h007"
    hl["settings"]["last_candidate_id"] = 40
    backup = {
        "candidates": [
            {**only, "id": "h007"},
            {**only, "id": "h009", "start": 30.0, "end": 40.0},
        ],
        "settings": {"last_candidate_id": 60},
    }
    raw = json.dumps(backup).encode()
    (run.proj / rerank.BACKUP).write_bytes(raw)
    hl["settings"]["rerank_backup"] = analyze.backup_digest(raw)
    (run.proj / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")

    def two(src, style, tr, use_scenes, max_cand, timings, **kw):
        info = MediaInfo(
            src, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
        )
        cands = [
            analyze.Candidate(src, 10.0, 20.0, 15.0, 1.0, {"loudness": 1.0}, score=50.0),
            analyze.Candidate(src, 30.0, 40.0, 35.0, 1.0, {"loudness": 1.0}, score=60.0),
        ]
        return analyze.SourceResult(src, info, 0.0, 60.0, cands, [], None)

    monkeypatch.setattr(analyze, "analyze_source", two)
    run.go("funny", layers=False)
    after = read(run.proj)
    assert {c["start"]: c["id"] for c in after["candidates"]} == {10.0: "h007", 30.0: "h009"}
    assert after["settings"]["last_candidate_id"] >= 60


def _spans(monkeypatch, *spans: tuple[float, float]) -> None:
    def fake(src, style, tr, use_scenes, max_cand, timings, **kw):
        info = MediaInfo(
            src, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
        )
        cands = [
            analyze.Candidate(src, a, b, (a + b) / 2, 1.0, {"loudness": 1.0}, score=50.0)
            for a, b in spans
        ]
        return analyze.SourceResult(src, info, 0.0, 60.0, cands, [], None)

    monkeypatch.setattr(analyze, "analyze_source", fake)


def _link_backup(proj: Path, raw: bytes, *, link: bool = True) -> None:
    (proj / rerank.BACKUP).write_bytes(raw)
    hl = read(proj)
    if link:
        hl["settings"]["rerank_backup"] = analyze.backup_digest(raw)
    (proj / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")


def test_a_fresh_analyze_carries_the_link_so_a_second_one_still_gets_backup_ids(
    run,  # noqa: F811
    monkeypatch,
):
    """#1009 (W1): action -> rerank -> analyze -> analyze: the backup keeps giving its ids."""
    run.go("funny")  # one candidate, 10-20 s
    hl = read(run.proj)
    hl["candidates"][0]["id"] = "h014"
    hl["settings"]["last_candidate_id"] = 14
    (run.proj / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    backup = {
        "candidates": [
            {**hl["candidates"][0], "id": "h001", "start": 50.0, "end": 58.0},
            {**hl["candidates"][0], "id": "h002", "start": 30.0, "end": 40.0},
        ],
        "settings": {"last_candidate_id": 2},
    }
    raw = json.dumps(backup).encode()
    _link_backup(run.proj, raw)
    _spans(monkeypatch, (10.0, 20.0))
    run.go("funny", layers=False)  # the style the rerank went to, analysed fresh
    after = read(run.proj)
    assert after["settings"]["rerank_backup"] == analyze.backup_digest(raw)
    _spans(monkeypatch, (10.0, 20.0), (30.0, 40.0))
    run.go("funny", layers=False)
    after = read(run.proj)
    assert {c["start"]: c["id"] for c in after["candidates"]}[30.0] == "h002"


def test_action_rerank_sport_analyze_sport_analyze_action_gives_the_original_ids(
    run,  # noqa: F811
    monkeypatch,
):
    """#1009 (W1), the whole chain for real: action analysed, a real rerank to sport, a fresh
    analyze as sport, a fresh analyze as action: the action spans get their first ids back."""
    action = [(10.0, 20.0), (30.0, 40.0), (50.0, 58.0)]
    sport = [(5.0, 9.0), (22.0, 27.0)]
    write_caches(run.video)
    _spans(monkeypatch, *action)
    run.go("action", layers=False)
    first = {c["start"]: c["id"] for c in read(run.proj)["candidates"]}
    assert sorted(first.values()) == ["h001", "h002", "h003"]
    rerank.rerank(run.proj, get_style("sport"))
    assert (run.proj / rerank.BACKUP).exists()
    assert "rerank_backup" in read(run.proj)["settings"]
    _spans(monkeypatch, *sport)
    run.go("sport", layers=False)
    assert "rerank_backup" in read(run.proj)["settings"]  # carried on
    _spans(monkeypatch, *action)
    run.go("action", layers=False)
    assert {c["start"]: c["id"] for c in read(run.proj)["candidates"]} == first


def test_a_fresh_analyze_records_no_link_for_a_stale_or_unlinked_backup(run, monkeypatch):  # noqa: F811
    run.go("funny")
    _link_backup(run.proj, b"{}", link=False)
    run.go("funny", layers=False)
    assert "rerank_backup" not in read(run.proj)["settings"]
    hl = read(run.proj)
    hl["settings"]["rerank_backup"] = "0123456789abcdef"  # not this backup's digest
    (run.proj / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    run.go("funny", layers=False)
    assert "rerank_backup" not in read(run.proj)["settings"]


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '{"no": "analysis"}', b"\xff\xfe"])
def test_a_fresh_analyze_over_an_unreadable_highlights_numbers_from_h001(run, raw):  # noqa: F811
    run.go("funny")
    path = run.proj / "highlights.json"
    path.write_bytes(raw if isinstance(raw, bytes) else raw.encode())
    run.go("funny", layers=False)
    after = read(run.proj)
    assert [c["id"] for c in after["candidates"]] == ["h001"]
    assert after["settings"]["last_candidate_id"] == 1


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


# --------------------------------------------------------------------------- vision, cache only
def test_a_hosted_unset_model_is_resolved_before_the_cached_ratings(home, tmp_path, monkeypatch):
    """#902 F6: a hosted Vision provider whose model is unset (the AUTO_SERVER_MODEL
    placeholder) is checked (a model list, no frames) so the ratings are looked up under the
    model it resolves to; the rater itself is never asked."""
    project = make_project(home, tmp_path)
    hl = read(project)
    hl["settings"]["vision_requested"] = True
    looked_up: list[str] = []
    checks: list[int] = []

    class Via:
        model = AUTO_SERVER_MODEL

        def available(self):
            checks.append(1)
            self.model = "served-model"
            return True

    class Rater:
        cache_only = False
        hits = asked = missed = 0
        stats = dict.fromkeys(
            ("calls", "grab_s", "load_s", "prompt_s", "reply_s", "prompt_tokens", "reply_tokens"), 0
        )
        provider_name = "hosted"
        consent_needed = None

        def __init__(self, **_):
            self.via = Via()
            self.model = self.via.model

        def available(self):
            pytest.fail("a re-rank asks no model")

        def cached(self, src, start, end, base, **kw):
            looked_up.append(self.model)
            hit = self.model == "served-model"
            return SimpleNamespace(score=8.0, label="nice") if hit else None

        def rate(self, src, start, end, base, **kw):
            assert self.cache_only
            res = self.cached(src, start, end, base)
            if res is not None:
                self.hits += 1
            return res

        def unload(self):
            pass

    monkeypatch.setattr(analyze, "VisionRater", Rater)
    after = rerank.ranked(project, hl, get_style("funny"))
    assert checks == [1]
    assert looked_up and set(looked_up) == {"served-model"}
    assert after["settings"]["vision_used"] is True
    assert after["settings"]["vision"]["model"] == "served-model"
