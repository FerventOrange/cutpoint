"""360 analyze loads CLIP once: the views' model goes on to the crash dense confirm (#376)."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint import analyze as analyze_mod
from cutpoint import crash, styles
from cutpoint.events import common, pipeline
from cutpoint.events import visual as V
from cutpoint.pano import cli, project, views, x3
from cutpoint.pano import geometry as G
from cutpoint.pano.x3 import X3Clip


class _Stop(Exception):
    pass


class _Clip:
    """A CLIP module: ``.to`` keeps the object (the identity the tests follow)."""

    def to(self, *args):
        return self


def _args(**over) -> argparse.Namespace:
    base = {
        "project": "p",
        "scan": None,
        "only": None,
        "no_events": False,
        "views_only": False,
        "force_views": False,
        "style": None,
        "model": None,
        "language": None,
    }
    return argparse.Namespace(**(base | over))


@pytest.fixture
def pano(tmp_path: Path, monkeypatch):
    """Runs ``cmd_analyze`` on one synthetic clip with the real view-layer driver and fake
    models; returns a record of the CLIP models built, models closed, CLIP releases and the
    encoder ``run_events`` got."""
    lrv = tmp_path / "VID_001.lrv"
    lrv.write_bytes(b"x")
    clip = X3Clip("001", "2026-01-01", "5.7k", str(lrv), [], frames=10, duration=10.0)
    rec = SimpleNamespace(built=[], closed=[], released=0, encoders=[], dense=True)

    class Models:
        def __init__(self, **kw):
            self.torch, self.device, self.dtype = None, "cuda", "float16"
            self.clip, self.det, self.mean, self.std = _Clip(), object(), "mean", "std"
            rec.built.append(self)

        def close(self):
            rec.closed.append(self)
            del self.clip, self.det

    def free_torch():
        rec.released += 1

    def compute(c, models, p, orient, wearer_v=G.WEARER_V):
        if rec.views_fail:
            raise RuntimeError("views failed")
        return SimpleNamespace(t=np.zeros(3), det=np.zeros((0, 7)))

    def run_events(project, *, layers, encoder=None):
        rec.encoders.append(encoder)
        if rec.events_fail:
            raise RuntimeError("events failed")
        if encoder is not None:  # what the crash layer does after the dense step
            encoder.close()

    monkeypatch.setattr(cli, "_project", lambda arg: tmp_path)
    monkeypatch.setattr(cli, "_clips", lambda p: [clip])
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "run_events", run_events)
    monkeypatch.setattr(V, "VisualModels", Models)
    monkeypatch.setattr(V, "text_embeddings", lambda prompts, models=None: None)
    monkeypatch.setattr(common, "free_torch", free_torch)
    monkeypatch.setattr(x3, "imu_source", lambda src: src)
    monkeypatch.setattr(G, "orientation", lambda *a: None)
    monkeypatch.setattr(views, "load_views", lambda src, p=None: None)
    monkeypatch.setattr(views, "compute_views", compute)
    monkeypatch.setattr(views, "wearer_axis", lambda data: G.WEARER_V)
    monkeypatch.setattr(views, "save_views", lambda src, data, p: None)
    monkeypatch.setattr(crash, "dense_needs_run", lambda srcs: rec.dense)
    monkeypatch.setattr(analyze_mod, "analyze", lambda *a, **k: rec.analyze_hook())
    monkeypatch.setattr(analyze_mod, "reanalyze_if_reachable", lambda p: {})
    monkeypatch.setattr(analyze_mod, "report_highlights", lambda p, hl: None)
    monkeypatch.setattr(project, "default_style", lambda: "ski")
    monkeypatch.setattr(styles, "get_style", lambda name: None)
    rec.views_fail = rec.events_fail = False
    rec.analyze_hook = lambda: None
    return rec


def test_the_crash_layer_gets_the_views_clip_and_it_is_released_once(pano):
    cli.cmd_analyze(_args())
    assert len(pano.built) == 1
    models = pano.built[0]
    assert pano.closed == [models]  # the detector is freed after the views
    (enc,) = pano.encoders
    assert isinstance(enc, crash.ClipEncoder)
    assert enc.model is None and pano.released == 1  # one release, by the crash layer


def test_run_events_gets_the_views_clip_object(pano, monkeypatch):
    clips: list[object] = []
    real = crash.ClipEncoder.from_visual

    def spy(models):
        clips.append(models.clip)
        return real(models)

    monkeypatch.setattr(crash.ClipEncoder, "from_visual", spy)
    got = []
    monkeypatch.setattr(
        pipeline, "run_events", lambda project, *, layers, encoder=None: got.append(encoder.model)
    )
    cli.cmd_analyze(_args())
    assert len(pano.built) == 1 and len(got) == 1 and got[0] is clips[0]
    assert pano.released == 1  # closed by cmd_analyze when the crash layer did not


def test_cached_dense_confirm_closes_clip_with_the_views(pano):
    pano.dense = False
    cli.cmd_analyze(_args())
    assert len(pano.built) == 1 and pano.closed == pano.built
    assert pano.encoders == [None] and pano.released == 0


@pytest.mark.parametrize("flag", ["no_events", "views_only"])
def test_skipped_events_close_clip_with_the_views(pano, flag):
    cli.cmd_analyze(_args(**{flag: True}))
    assert len(pano.built) == 1 and pano.closed == pano.built
    assert pano.encoders == [] and pano.released == 0


def test_failed_views_close_clip_and_hand_nothing_over(pano):
    pano.views_fail = True
    with pytest.raises(RuntimeError, match="views failed"):
        cli.cmd_analyze(_args())
    assert len(pano.built) == 1 and pano.closed == pano.built
    assert pano.encoders == [] and pano.released == 0


def test_failed_analysis_releases_the_kept_clip(pano):
    def boom():
        raise RuntimeError("whisper failed")

    pano.analyze_hook = boom
    with pytest.raises(RuntimeError, match="whisper failed"):
        cli.cmd_analyze(_args())
    assert pano.encoders == [] and pano.released == 1


def test_failed_events_release_the_kept_clip_once(pano):
    pano.events_fail = True
    with pytest.raises(RuntimeError, match="events failed"):
        cli.cmd_analyze(_args())
    assert len(pano.encoders) == 1 and pano.released == 1


def test_dense_needs_run_checks_the_cache_folders(tmp_path: Path):
    a, b = tmp_path / "a.lrv", tmp_path / "b.lrv"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    assert crash.dense_needs_run([a])
    (crash.cache.source_dir(a) / "crashdense-0123.json").write_text("{}")
    assert not crash.dense_needs_run([a])
    assert crash.dense_needs_run([a, b])
    assert not crash.dense_needs_run([a, tmp_path / "offline.lrv"])  # skipped there too


# --------------------------------------------------------------------------- run_events
@pytest.fixture
def events(tmp_path: Path, monkeypatch):
    """``run_events`` with the visual and crash layers faked; returns the handoff the
    visual layer got and the encoder the crash layer got."""
    src = common.Source(path=tmp_path / "a.lrv", duration=10.0, cdir=tmp_path / "a", audio_parts=[])
    monkeypatch.setattr(pipeline, "load_highlights", lambda p: {"sources": [{"path": "a"}]})
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "load_source", lambda path, dur: src)
    monkeypatch.setattr(pipeline, "bad_ranges", lambda *a: {})
    monkeypatch.setattr(pipeline, "load_transcript_lines", lambda project: [])
    monkeypatch.setattr(common, "gpu_memory_used_mib", lambda: None)
    rec = SimpleNamespace(handoff=[], encoder=[], fail=False)

    def visual(sources, params, force=False, bad=None, handoff=None):
        rec.handoff.append(handoff)
        return {}

    def detect(project, force=False, encoder=None, warn_unreachable=True, llm=True):
        rec.encoder.append(encoder)
        if rec.fail:
            raise RuntimeError("crash failed")
        if encoder is not None:  # closed after the dense step
            encoder.close()

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(V, "run_layer", visual)
    monkeypatch.setattr(crash, "detect_project", detect)
    monkeypatch.setattr(pipeline, "select_windows", stop)
    return rec


def _encoder() -> crash.ClipEncoder:
    enc = crash.ClipEncoder.__new__(crash.ClipEncoder)
    enc.model = _Clip()
    return enc


def test_run_events_gives_the_encoder_to_the_crash_layer(tmp_path: Path, events, monkeypatch):
    released: list[int] = []
    monkeypatch.setattr(common, "free_torch", lambda: released.append(1))
    enc = _encoder()
    with pytest.raises(_Stop):
        pipeline.run_events(
            tmp_path, layers=("visual", "crash"), discover=False, cached_ocr=False, encoder=enc
        )
    assert events.handoff == [None]  # the visual layer keeps no second CLIP
    assert events.encoder == [enc] and enc.model is None and released == [1]


def test_run_events_closes_the_encoder_when_the_crash_layer_fails(
    tmp_path: Path, events, monkeypatch
):
    released: list[int] = []
    monkeypatch.setattr(common, "free_torch", lambda: released.append(1))
    events.fail = True
    enc = _encoder()
    with pytest.raises(RuntimeError, match="crash failed"):
        pipeline.run_events(
            tmp_path, layers=("visual", "crash"), discover=False, cached_ocr=False, encoder=enc
        )
    assert events.encoder == [enc] and enc.model is None and released == [1]


def test_without_an_encoder_the_visual_layer_still_hands_over(tmp_path: Path, events):
    with pytest.raises(_Stop):
        pipeline.run_events(tmp_path, layers=("visual", "crash"), discover=False, cached_ocr=False)
    assert events.handoff[0] is not None and events.encoder == [None]
