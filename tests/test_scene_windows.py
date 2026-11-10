"""Scene cuts around candidate edges only (#730): a style that does not weigh the scene curve
(``funny``, ``commentary``, a user style with scene 0) runs no full scene pass; the cuts it
snaps edges to are detected in windows around the candidate edges and cached in
``scenes-windows.json``. ffmpeg, ffprobe, PySceneDetect, Whisper and the signal files are
faked; the cache folder is real."""

from __future__ import annotations

import contextlib
import json
import os
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import StubTranscriber, media_info, write
from cutpoint import analyze, cache
from cutpoint.styles import STYLES, get_style

WINDOWS = analyze.SCENE_WINDOWS_FILE
PAD = analyze.WINDOW_PAD_S


# ----------------------------------------------------------------- which styles
@pytest.mark.parametrize(
    ("name", "weighs"),
    [("funny", False), ("commentary", False), ("gaming", True), ("action", True), ("sport", True)],
)
def test_only_the_builtins_with_scene_weight_0_detect_around_edges(name, weighs):
    assert analyze.scores_scenes(get_style(name)) is weighs
    assert (STYLES[name].weights.get("scene", 0) > 0) is weighs


def test_a_user_style_with_scene_weight_0_detects_around_edges(home, monkeypatch):
    folder = home / "my-styles"
    monkeypatch.setenv("CUTPOINT_STYLES_DIR", str(folder))
    doc = {"schema": 1, "base": "action", "source": "hand", "weights": {"scene": 0}}
    write(folder / "calm.json", json.dumps({**doc, "name": "calm"}))
    write(folder / "cuts.json", json.dumps({**doc, "name": "cuts", "base": "funny",
                                            "weights": {"scene": 0.3}}))  # fmt: skip
    assert analyze.scores_scenes(get_style("calm")) is False
    assert analyze.scores_scenes(get_style("cuts")) is True


# ----------------------------------------------------------------- windows
def test_an_edge_window_reaches_the_pad_either_side_on_the_proxy_grid():
    assert analyze._edge_window(20.0, 60.0) == (17.5, 22.5)
    assert analyze._edge_window(20.04, 60.0) == (17.5, 22.54)  # start floored to 0.1 s
    assert analyze._edge_window(1.0, 60.0) == (0.0, 1.0 + PAD)  # clamped to the part
    assert analyze._edge_window(59.0, 60.0) == (56.5, 60.0)


def test_windows_that_overlap_or_touch_merge():
    got = analyze._merge_windows([(30.0, 35.0), (10.0, 15.0), (14.0, 19.0), (19.0, 20.0)])
    assert got == [(10.0, 20.0), (30.0, 35.0)]


def test_a_window_proxy_seeks_to_the_part_offset_plus_the_window(tmp_path, monkeypatch):
    """The proxy is the 10 fps one of a normal file, cut to the window; cuts come back in part
    time (window start added)."""
    src = write(tmp_path / "a.mp4")
    seen = []
    monkeypatch.setattr(analyze, "decoder_for", lambda s: None)
    monkeypatch.setattr(analyze, "run_progress", lambda args, total, label, **k: seen.append(args))
    monkeypatch.setattr(analyze, "_scan", lambda proxy, cancel: [0.4, 2.0])
    cuts = analyze._window_cuts(src, tmp_path, 100.0, 17.5, 22.5)
    assert cuts == [17.9, 19.5]
    (args,) = seen
    assert args[args.index("-ss") + 1] == "117.500" and args[args.index("-t") + 1] == "5.000"
    assert "fps=10,scale=320:-2" in args and "-skip_frame" not in args
    assert not list(tmp_path.glob("proxy-window*"))


def test_a_window_proxy_has_its_own_name_and_goes_after_a_failed_encode(tmp_path, monkeypatch):
    """Two runs on the same source share its cache folder: each names its proxy by process and
    thread, and removes it also when the encode is interrupted."""
    names = []

    def encode(args, total, label, **k):
        names.append(Path(args[-1]).name)
        write(Path(args[-1]), "partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(analyze, "decoder_for", lambda s: None)
    monkeypatch.setattr(analyze, "run_progress", encode)
    with pytest.raises(KeyboardInterrupt):
        analyze._window_cuts(write(tmp_path / "a.mp4"), tmp_path, 0.0, 17.5, 22.5)
    (name,) = names
    assert name == f"proxy-window-{os.getpid()}-{threading.get_ident()}.mp4"
    assert not list(tmp_path.glob("proxy-window*"))


@pytest.mark.parametrize("kind", ["cuda", "vaapi"])
def test_a_window_proxy_decodes_on_the_cpu_with_a_gpu_decoder(tmp_path, monkeypatch, kind):
    """Each window is a short ffmpeg run: a GPU decoder's start-up costs more than it saves, so
    the windows decode on the CPU whatever decoder the full pass would use (same 10 fps chain)."""
    seen = []
    hw = analyze.HwDecoder(kind, "/dev/dri/renderD128" if kind == "vaapi" else None)
    monkeypatch.setattr(analyze, "decoder_for", lambda s: hw)
    monkeypatch.setattr(analyze, "run_progress", lambda args, total, label, **k: seen.append(args))
    monkeypatch.setattr(analyze, "_scan", lambda proxy, cancel: [0.4])
    assert analyze._window_cuts(write(tmp_path / "a.mp4"), tmp_path, 0.0, 17.5, 22.5) == [17.9]
    (args,) = seen
    assert "-hwaccel" not in args and "hwdownload" not in " ".join(args)
    assert args[args.index("-vf") + 1] == "fps=10,scale=320:-2"


@pytest.mark.parametrize(
    ("span", "windows"),
    [(60.0, True), (analyze.LONG_FILE_S, True), (analyze.LONG_FILE_S + 1, False)],
)
def test_edge_windows_only_up_to_the_long_file_length(span, windows):
    """Past LONG_FILE_S the full pass decodes keyframes only, faster than the 10 fps windows."""
    assert analyze.edge_windows(get_style("funny"), span) is windows
    assert analyze.edge_windows(get_style("gaming"), span) is False


def test_scene_snapping_reaches_one_second_but_sentence_snapping_goes_further():
    """Moves past SNAP_REACH_S come from the sentence snap after the scene snap, not from a
    longer cut reach: two cut lists that differ only within 1 s of the edge put the start of a
    ``commentary`` candidate on two sentence starts 2.5 s apart."""

    def seg(a, b):
        d = (b - a) / 4
        words = [{"start": a + i * d, "end": a + (i + 1) * d - 0.02, "word": "w"} for i in range(4)]
        return {"start": a, "end": b, "text": "one two three four.", "words": words}

    segs = [seg(96.0, 99.5), seg(99.6, 102.0), seg(102.1, 106.0)]
    got = []
    for cuts in ([102.4], [101.3]):  # each within reach of the unsnapped start 101.9
        c = analyze.Candidate(Path("a.mp4"), 101.9, 130.0, 110.0, 1.0, {})
        analyze._snap(c, cuts, segs, 600.0, get_style("commentary"))
        got.append(c.start)
    assert got[0] - got[1] > 2 * analyze.SNAP_REACH_S


# ----------------------------------------------------------------- EdgeCuts and its cache
@pytest.fixture
def decodes(monkeypatch):
    """``_window_cuts`` faked: a cut 0.6 s after every whole 10 s inside the window."""
    calls: list[tuple[float, float, float]] = []

    def fake(io, cdir, offset, a, b):
        calls.append((offset, a, b))
        return [float(t) + 0.6 for t in range(0, 600, 10) if a <= t + 0.6 <= b]

    monkeypatch.setattr(analyze, "_window_cuts", fake)
    return calls


def saved(cdir: Path) -> dict:
    return json.loads((cdir / WINDOWS).read_text(encoding="utf-8"))


def test_edge_cuts_decode_merged_windows_and_save_them(tmp_path, decodes):
    took = []
    ec = analyze.EdgeCuts(tmp_path / "a.mp4", tmp_path, 120.0, on_time=took.append)
    cuts = ec.around([20.0, 22.0, 60.0])
    assert decodes == [(0.0, 17.5, 24.5), (0.0, 57.5, 62.5)]
    assert cuts == [20.6, 60.6]
    assert saved(tmp_path) == {"version": 1, "windows": [[17.5, 24.5], [57.5, 62.5]],
                               "cuts": [20.6, 60.6]}  # fmt: skip
    assert len(took) == 1 and took[0] >= 0


def test_edge_cuts_reuse_covered_windows_and_append_missing_ones(tmp_path, decodes):
    analyze.EdgeCuts(tmp_path / "a.mp4", tmp_path, 120.0).around([20.0, 60.0])
    decodes.clear()
    again = analyze.EdgeCuts(tmp_path / "a.mp4", tmp_path, 120.0)
    assert again.around([20.0, 60.0]) == [20.6, 60.6] and decodes == []  # all cached
    assert again.around([90.0]) == [20.6, 60.6, 90.6]
    assert decodes == [(0.0, 87.5, 92.5)]  # only the missing window
    assert saved(tmp_path)["windows"] == [[17.5, 22.5], [57.5, 62.5], [87.5, 92.5]]


def test_a_window_touching_saved_ones_is_decoded_with_them(tmp_path, decodes):
    """Saved windows never overlap: each cut comes from one decode of its whole window."""
    ec = analyze.EdgeCuts(tmp_path / "a.mp4", tmp_path, 120.0)
    ec.around([20.0])
    decodes.clear()
    assert ec.around([20.0, 24.0]) == [20.6]  # 24 needs 21.5-26.5: overlaps 17.5-22.5
    assert decodes == [(0.0, 17.5, 26.5)]
    assert saved(tmp_path) == {"version": 1, "windows": [[17.5, 26.5]], "cuts": [20.6]}


def test_a_full_scenes_json_is_preferred_over_windows(tmp_path, decodes):
    cache.save_json(tmp_path / "scenes.json", [5.0, 21.0])
    assert analyze.EdgeCuts(tmp_path / "a.mp4", tmp_path, 120.0).around([20.0]) == [5.0, 21.0]
    assert decodes == [] and not (tmp_path / WINDOWS).exists()


def test_an_unreadable_windows_file_is_decoded_again(tmp_path, decodes):
    write(tmp_path / WINDOWS, json.dumps({"cuts": "?"}))
    assert analyze.EdgeCuts(tmp_path / "a.mp4", tmp_path, 120.0).around([20.0]) == [20.6]
    assert decodes == [(0.0, 17.5, 22.5)]


def test_a_recovered_part_decodes_at_its_offset(tmp_path, decodes):
    analyze.EdgeCuts(tmp_path / "a.mp4", tmp_path, 50.0, offset=300.0).around([49.0])
    assert decodes == [(300.0, 46.5, 50.0)]  # part time, clamped to the part


# ----------------------------------------------------------------- analyze_source
@pytest.fixture
def world(home, tmp_path, monkeypatch, decodes):
    """``analyze_source`` with ffprobe, the audio, Whisper and the loudness faked: one loud burst
    at 20 s of a 60 s source. ``w.full`` records full scene passes."""
    monkeypatch.delenv("CUTPOINT_ANALYZE_SCENE_OVERLAP", raising=False)
    src = write(tmp_path / "videos" / "a.mp4")
    w = type("World", (), {})()
    w.src, w.decodes, w.full = src, decodes, []
    w.cdir = cache.source_dir(src, 0.0)

    def full(src_, cdir, duration, offset=0.0, limit=None, shared=None, cancel=None):
        w.full.append(threading.current_thread())
        cache.save_json(cdir / "scenes.json", [16.4])
        return [16.4]

    monkeypatch.setattr(analyze, "scene_cuts", full)
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, 60.0, audio_streams=1))
    monkeypatch.setattr(analyze, "extract_audio", lambda io, wav, *a, **k: write(wav, "wav"))
    monkeypatch.setattr(analyze, "check_read", lambda *a, **k: None)
    times = np.arange(0.0, 60.0, 0.5)
    lufs = -50.0 + 35.0 * np.exp(-(((times - 20.0) / 3.0) ** 2))
    monkeypatch.setattr(analyze, "_loudness", lambda wav, cdir: (times, lufs))
    w.timings = {}
    w.source = lambda style: analyze.analyze_source(
        src, style, StubTranscriber(), True, None, w.timings
    )
    return w


def edges(res) -> list[float]:
    return sorted(t for c in res.candidates for t in (c.start, c.end))


@pytest.mark.parametrize("name", ["funny", "commentary"])
def test_a_style_without_scene_weight_runs_no_full_pass(world, monkeypatch, name):
    monkeypatch.setenv("CUTPOINT_ANALYZE_SCENE_OVERLAP", "1")  # no scene thread either
    monkeypatch.setattr(analyze, "_SceneJob", lambda *a, **k: pytest.fail("scene thread"))
    res = world.source(get_style(name))
    assert res.candidates and world.full == []
    assert world.decodes  # windows around the edges instead
    assert res.stages["scene_windows"] is True and res.stages["scenes_s"] >= 0
    assert world.timings["scenes_s"] == res.stages["scenes_s"]
    assert not res.signals.sig["scene"].any()  # weight 0: no curve
    assert not (world.cdir / "scenes.json").exists() and (world.cdir / WINDOWS).is_file()
    rec = analyze._source_timings([res], None)[0]
    assert rec["scene_windows"] is True and "overlap" not in rec


def test_every_unsnapped_edge_lies_inside_a_decoded_window(world, monkeypatch):
    seen = []
    real = analyze._snap

    def spy(c, cuts, segments, duration, style):
        seen.append((c.start, c.end))
        real(c, cuts, segments, duration, style)

    monkeypatch.setattr(analyze, "_snap", spy)
    world.source(get_style("funny"))
    wins = [(a, b) for _off, a, b in world.decodes]
    for t in (t for pair in seen for t in pair):
        a, b = analyze._edge_window(t, 60.0)
        assert any(x <= a + 1e-6 and b - 1e-6 <= y for x, y in wins), t


def test_edges_snap_to_window_cuts_as_to_full_ones(world, monkeypatch):
    """The same cuts near the edges give the same candidates, from windows or a full pass."""
    from_windows = edges(world.source(get_style("funny")))
    near = {round(x, 3) for _o, a, b in world.decodes for x in [t + 0.6 for t in range(0, 60, 10)]
            if a <= x <= b}  # fmt: skip
    assert set(from_windows) & near  # a cut was in reach and taken
    cache.save_json(world.cdir / "scenes.json", sorted(near))
    world.decodes.clear()
    assert edges(world.source(get_style("funny"))) == from_windows
    assert world.decodes == [] and world.full == []  # the cached full pass was used


@pytest.mark.parametrize("name", ["funny", "commentary"])
def test_a_long_source_keeps_the_full_pass(world, monkeypatch, name):
    """Over LONG_FILE_S the full pass decodes keyframes only and is faster than the windows:
    a style without scene weight runs it as main did, and no windows."""
    long = analyze.LONG_FILE_S + 1
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, long, audio_streams=1))
    res = world.source(get_style(name))
    assert world.full == [threading.current_thread()] and world.decodes == []
    assert res.signals.cuts == [16.4] and res.signals.edge_cuts is None
    assert not res.signals.sig["scene"].any()  # weight 0: still no curve
    assert "scene_windows" not in res.stages and res.stages["scenes_s"] >= 0
    assert not (world.cdir / WINDOWS).exists()
    assert "scene_windows" not in analyze._source_timings([res], None)[0]


def test_a_long_source_of_a_style_without_scene_weight_overlaps_as_before(world, monkeypatch):
    """With CUTPOINT_ANALYZE_SCENE_OVERLAP=1 its full pass runs alongside Whisper, as on main."""
    monkeypatch.setenv("CUTPOINT_ANALYZE_SCENE_OVERLAP", "1")
    long = analyze.LONG_FILE_S + 1
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, long, audio_streams=1))
    started = []

    class Job:
        def __init__(self, *args):
            started.append(args)
            self.cuts, self.took = [16.4], 0.5

        def result(self):
            return self.cuts

    monkeypatch.setattr(analyze, "_SceneJob", Job)
    tr = type("Tr", (StubTranscriber,), {"load": lambda self: None})()
    res = analyze.analyze_source(world.src, get_style("funny"), tr, True, None, {})
    assert len(started) == 1 and world.decodes == [] and res.stages["overlap"] is True
    assert res.signals.cuts == [16.4] and "scene_windows" not in res.stages


def test_a_cached_full_pass_is_used_by_a_style_without_scene_weight(world):
    cache.save_json(world.cdir / "scenes.json", [16.4])
    res = world.source(get_style("funny"))
    assert world.full == [] and world.decodes == []
    assert res.signals.cuts == [16.4] and res.signals.edge_cuts is None
    assert "scene_windows" not in res.stages


def test_the_window_cache_is_reused_by_the_next_run(world):
    first = edges(world.source(get_style("funny")))
    world.decodes.clear()
    assert edges(world.source(get_style("funny"))) == first and world.decodes == []


@pytest.mark.parametrize("name", ["gaming", "action", "sport"])
def test_a_style_with_scene_weight_keeps_the_full_pass(world, name):
    res = world.source(get_style(name))
    assert world.full == [threading.current_thread()] and world.decodes == []
    assert res.signals.cuts == [16.4] and res.signals.sig["scene"].max() > 0
    assert "scene_windows" not in res.stages


def test_a_user_style_with_scene_weight_0_runs_no_full_pass(world):
    calm = replace(get_style("action"), name="calm", weights={**get_style("action").weights,
                                                             "scene": 0.0})  # fmt: skip
    res = world.source(calm)
    assert world.full == [] and world.decodes and res.stages["scene_windows"] is True


def test_no_scenes_detects_nothing(world):
    res = analyze.analyze_source(world.src, get_style("funny"), StubTranscriber(), False, None, {})
    assert world.full == [] and world.decodes == [] and res.signals.edge_cuts is None


def test_a_style_chosen_later_that_weighs_scenes_gets_the_full_pass(world):
    """--style auto decides after the transcript: windows first, then a full pass."""
    res = world.source(get_style("funny"))
    assert world.full == []
    analyze.scene_signal(res.signals, get_style("action"))
    assert len(world.full) == 1 and res.signals.edge_cuts is None
    assert res.signals.cuts == [16.4] and res.signals.sig["scene"].max() > 0


def test_an_auto_switch_to_a_scene_weighted_style_drops_the_windows_mark(world, monkeypatch):
    """--style auto, pending: picked funny before the transcript (no full pass), action after
    it (the full pass runs); timings.sources then no longer says ``scene_windows``."""
    det = SimpleNamespace(pending=True, category="general", reason="test", record=lambda *a: {})
    picks = iter(["funny", "action"])
    monkeypatch.setattr(analyze.category, "detect", lambda sources, *a, **kw: ([], det))
    monkeypatch.setattr(analyze.category, "decide", lambda probes, speech: det)
    monkeypatch.setattr(analyze.category, "source_speech", lambda *a: {})
    monkeypatch.setattr(analyze.category, "choose_style", lambda d, e: (next(picks), None))
    monkeypatch.setattr(analyze.category, "warn", lambda *a: None)
    monkeypatch.setattr(analyze, "Transcriber", StubTranscriber)
    proj = world.src.parent.parent / "p-auto"
    analyze.analyze(world.src, analyze.AutoRequest("hype"), proj, vision=False,
                    use_scenes=True, layers=False)  # fmt: skip
    hl = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    assert hl["style"] == "action" and len(world.full) == 1
    (rec,) = hl["timings"]["sources"]
    assert "scene_windows" not in rec and rec["scenes_s"] >= 0


def test_the_shared_scene_proxy_is_not_planned_without_scene_weight(world, monkeypatch):
    """In a shared-decode mode the background proxy is planned only for styles that weigh it."""
    planned = []
    monkeypatch.setattr(analyze, "_scene_plan", lambda src, d: planned.append(src))

    pre = SimpleNamespace(mode="split", waits={}, offer=lambda *a: None)
    monkeypatch.setattr(analyze, "_ocr_prefetch", lambda *a, **k: contextlib.nullcontext(pre))
    monkeypatch.setattr(analyze, "Transcriber", type("Tr", (StubTranscriber,), {"device": "cuda"}))
    for name in ("funny", "gaming"):
        analyze.analyze(world.src, get_style(name), world.src.parent.parent / f"p-{name}",
                        vision=False, use_scenes=True, layers=False)  # fmt: skip
    assert planned == [world.src]  # gaming only


# ----------------------------------------------------------------- the cached grids (events)
def test_cached_grids_detect_around_edges_for_a_style_without_scene_weight(tmp_path, decodes):
    src = write(tmp_path / "a.mp4")
    cdir = cache.source_dir(src)
    cache.save_json(cdir / "loudness.json", {"t": [0.2, 0.7], "m": [-30.0, -30.0]})
    hl = {"settings": {"scenes": True}, "sources": [{"path": str(src), "duration": 60.0}]}
    (part,) = analyze.cached_grid_parts(hl, get_style("funny"))
    assert part.signals.edge_cuts is not None and part.signals.cuts == []
    (part,) = analyze.cached_grid_parts(hl, get_style("gaming"))
    assert part.signals.edge_cuts is None
    cache.save_json(cdir / "scenes.json", [12.0])
    (part,) = analyze.cached_grid_parts(hl, get_style("funny"))
    assert part.signals.edge_cuts is None and part.signals.cuts == [12.0]
    assert decodes == []  # nothing decoded until a candidate edge needs it


def test_cached_grids_keep_a_source_without_audio_for_a_style_without_scene_weight(
    tmp_path, decodes
):
    """No loudness, no transcript, no scenes.json: the edge cuts still serve its part (crash
    rebuild, nominations); a style that weighs scenes has nothing to build it from."""
    src = write(tmp_path / "silent.mp4")
    hl = {"settings": {"scenes": True}, "sources": [{"path": str(src), "duration": 60.0}]}
    (part,) = analyze.cached_grid_parts(hl, get_style("funny"))
    assert part.source == src and part.signals.edge_cuts is not None
    assert not part.signals.sig["loudness"].any()
    assert analyze.cached_grid_parts(hl, get_style("gaming")) == []
    hl["settings"]["scenes"] = False  # --no-scenes: no edge cuts either
    assert analyze.cached_grid_parts(hl, get_style("funny")) == []


def test_cached_grids_of_a_long_source_decode_no_windows(tmp_path, decodes):
    """Over LONG_FILE_S, as in analyze: no edge windows; without scenes.json it has no cuts."""
    src = write(tmp_path / "a.mp4")
    cache.save_json(cache.source_dir(src) / "loudness.json", {"t": [0.2], "m": [-30.0]})
    entry = {"path": str(src), "duration": analyze.LONG_FILE_S + 1}
    hl = {"settings": {"scenes": True}, "sources": [entry]}
    (part,) = analyze.cached_grid_parts(hl, get_style("funny"))
    assert part.signals.edge_cuts is None and part.signals.cuts == [] and decodes == []
    entry["path"] = str(write(tmp_path / "silent.mp4"))  # nothing cached at all
    assert analyze.cached_grid_parts(hl, get_style("funny")) == []


def test_cached_grids_name_a_source_missing_its_scene_score(tmp_path, caplog):
    src = write(tmp_path / "a.mp4")
    cache.save_json(cache.source_dir(src) / "loudness.json", {"t": [0.2], "m": [-30.0]})
    hl = {"settings": {"scenes": True}, "sources": [{"path": str(src), "duration": 60.0}]}
    with caplog.at_level("INFO", logger="cutpoint.analyze"):
        analyze.cached_grid_parts(hl, get_style("gaming"))
    assert "a.mp4: no scene cuts cached; its scene score is missing" in caplog.text
    caplog.clear()
    with caplog.at_level("INFO", logger="cutpoint.analyze"):
        analyze.cached_grid_parts(hl, get_style("funny"))  # weight 0: nothing missing
    assert "no scene cuts cached" not in caplog.text
