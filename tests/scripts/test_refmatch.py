"""``scripts/refmatch.py`` (formerly ``cutpoint refmatch`` / ``fit``): the ``mask_friends``
search (soft and hard mask only; friends no longer lift it), cached thumbnails and audio, and
check-frame cleanup."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import load_script, write
from cutpoint import cache, reset_state, ski
from cutpoint import ffmpeg_utils as fu
from cutpoint.errors import FFmpegError
from cutpoint.styles import get_style

pytestmark = pytest.mark.script
refmatch = load_script("refmatch")


def interrupt(*_a: object, **_k: object) -> None:
    raise KeyboardInterrupt  # what Ctrl+C raises inside a step


def test_mask_modes_collapse_old_threshold_lists() -> None:
    assert refmatch.mask_modes((0, 5, 6, 7, 8, 11), 7.0) == (0.0, 7.0)
    assert refmatch.mask_modes((0, 7, 11), 7.0) == (0.0, 7.0)
    assert refmatch.mask_modes((11,), 7.0) == (7.0,)  # cross-validation refits one mode
    assert refmatch.mask_modes((5, 0), 0.0) == (float(ski.FRIENDS_STRONG), 0.0)


def test_fit_weights_fits_each_mask_once(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def stop(_ctx, _keys, thresholds, *_a, **_k):
        seen.append(thresholds)
        raise RuntimeError("stop")

    monkeypatch.setattr(refmatch, "_train", stop)
    with pytest.raises(RuntimeError, match="stop"):
        refmatch.fit_weights([], thresholds=(0, 5, 6, 7, 8, 11), save=False)
    assert seen == [(0.0, get_style("action-cam").mask_friends)]


def test_log_fit_reads_a_preset_saved_with_a_bom(tmp_path: Path, caplog) -> None:
    preset = tmp_path / "learned.json"
    preset.write_bytes(b"\xef\xbb\xbf" + json.dumps({"weights": {"speed": 2.0}}).encode())
    pr = {"precision": 0.5, "recall": 0.5}
    hit = {**pr, "moments_hit": 1, "moments": 2}
    rep = {
        "preset": str(preset),
        "features_fitted": ["speed"],
        "features": {},
        "at_edit_duration": {"default": hit, "learned_in_sample": hit},
        "cv_at_edit_duration": {"default": pr, "learned": pr},
        "auc": {"default": 0.5, "learned_cv": 0.6},
        "mask": {
            "positives_s": 10,
            "in_lift_or_stop_s": 2,
            "share_of_footage_masked": 0.1,
            "unlocked_by_vlm_friends_s": 1,
            "still_masked_s": 1,
        },
        "mask_friends": 0,
    }
    with caplog.at_level("INFO", logger=refmatch.logger.name):
        refmatch._log_fit(rep, {"speed": 1.0})
    assert "speed      1.00 -> 2.00" in caplog.text
    assert "Saved learned preset" in caplog.text


def test_interrupted_check_frames_leaves_no_refmatch_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shot = {
        "matched": True, "shot": 1, "ref_start": 0.0, "ref_end": 2.0, "src_start": 5.0,
        "source": str(tmp_path / "a.mp4"),
    }  # fmt: skip
    result = {"shots": [shot], "reference": str(tmp_path / "ref.mp4")}
    out = tmp_path / "proj" / "refmatch_check"
    monkeypatch.setattr(refmatch, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        refmatch.check_frames(result, out)
    assert not out.exists()
    assert not (tmp_path / "proj").exists()


@pytest.fixture
def thumbs_env(monkeypatch, tmp_path):
    """refmatch.thumbnails with a 10 s video and a fake decoder (frames per pass)."""
    state = SimpleNamespace(per_pass={"auto": 100, None: 100}, calls=[])
    monkeypatch.setattr(cache, "source_dir", lambda _p: tmp_path)
    monkeypatch.setattr(refmatch, "probe", lambda _p: SimpleNamespace(vduration=10.0))
    state.gpu = [fu.HwDecoder("cuda")]  # the decoder found for the source; [None] = CPU only
    monkeypatch.setattr(refmatch, "decoder_for", lambda _p: state.gpu[0])

    def fake_decode(_path, *, size, vf, pix_fmt, hwaccel, fallback_cpu=True):
        state.calls.append((hwaccel, fallback_cpu))
        n = state.per_pass[hwaccel]
        if isinstance(n, Exception):
            raise n
        w, h = size
        for k in range(n):
            yield k / refmatch.FPS, np.full((h, w), k % 256, np.uint8)

    monkeypatch.setattr(refmatch, "decode_raw", fake_decode)
    state.cpath = tmp_path / refmatch.THUMBS_FILE
    return state


def test_thumbnails_full_decode_is_cached(thumbs_env):
    frames = refmatch.thumbnails(Path("x.mp4"))
    assert frames.shape == (100, refmatch.W * refmatch.H)
    assert frames[3, 0] == 3
    assert thumbs_env.calls == [("auto", False)]  # no hidden CPU pass in decode_raw
    assert thumbs_env.cpath.exists()


def test_thumbnails_short_cuda_decode_is_redone_on_cpu(thumbs_env):
    thumbs_env.per_pass["auto"] = 40
    frames = refmatch.thumbnails(Path("x.mp4"))
    assert thumbs_env.calls == [("auto", False), (None, True)]
    assert len(frames) == 100 and thumbs_env.cpath.exists()


def test_thumbnails_without_a_gpu_decoder_decode_once_on_the_cpu(thumbs_env):
    """VAAPI declines HEVC/10-bit: "auto" would be the CPU pass the retry then repeats."""
    thumbs_env.gpu[0] = None
    frames = refmatch.thumbnails(Path("x.mp4"))
    assert len(frames) == 100
    assert thumbs_env.calls == [(None, True)]


def test_thumbnails_still_short_is_not_cached(thumbs_env):
    thumbs_env.per_pass.update({"auto": 40, None: 60})
    frames = refmatch.thumbnails(Path("x.mp4"))
    assert len(frames) == 60
    assert not thumbs_env.cpath.exists()


def test_thumbnails_failed_cuda_decode_runs_cpu(thumbs_env):
    thumbs_env.per_pass["auto"] = FFmpegError("No device")
    frames = refmatch.thumbnails(Path("x.mp4"))
    assert thumbs_env.calls == [("auto", False), (None, True)]
    assert len(frames) == 100 and thumbs_env.cpath.exists()


def test_thumbnails_keeps_cuda_frames_when_cpu_retry_fails(thumbs_env):
    thumbs_env.per_pass.update({"auto": 40, None: FFmpegError("read error")})
    frames = refmatch.thumbnails(Path("x.mp4"))
    assert len(frames) == 40
    assert not thumbs_env.cpath.exists()


def test_thumbnails_raise_when_neither_pass_decodes(thumbs_env):
    thumbs_env.per_pass.update({"auto": FFmpegError("a"), None: FFmpegError("b")})
    with pytest.raises(FFmpegError, match="Could not decode thumbnails"):
        refmatch.thumbnails(Path("x.mp4"))
    assert not thumbs_env.cpath.exists()


@pytest.fixture
def audio_env(monkeypatch, tmp_path):
    """refmatch.audio_8k with a failing decoder; ``streams`` = probed audio streams."""
    state = SimpleNamespace(streams=1, decodes=0, cpath=tmp_path / "audio8k.npy")

    def failing(*_a, **_kw):
        state.decodes += 1
        raise FFmpegError("decode failed")

    monkeypatch.setattr(cache, "source_dir", lambda _p: tmp_path)
    monkeypatch.setattr(refmatch, "decode_pcm", failing)
    monkeypatch.setattr(refmatch, "probe", lambda _p: SimpleNamespace(audio_streams=state.streams))
    monkeypatch.setattr(refmatch, "_AUDIO_FAILED", set())
    return state


def test_audio_8k_no_audio_stream_is_cached(audio_env):
    audio_env.streams = 0
    x = refmatch.audio_8k(Path("x.mp4"))
    assert x.shape == (0,) and x.dtype == np.float32
    assert audio_env.cpath.exists()


def test_audio_8k_other_failure_not_cached_nor_retried_in_process(audio_env, caplog):
    for _ in range(3):
        x = refmatch.audio_8k(Path("x.mp4"))
        assert x.shape == (0,) and x.dtype == np.float32
    assert not audio_env.cpath.exists()
    assert audio_env.decodes == 1
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


# -- F11: refmatch audio failures keyed by cache folder ------------------------------
def test_audio_failure_keyed_by_source_cache_dir(tmp_path, monkeypatch):
    calls: list[Path] = []

    def fail(path, sr):
        calls.append(path)
        raise FFmpegError("decode failed")

    monkeypatch.setattr(refmatch, "decode_pcm", fail)
    monkeypatch.setattr(refmatch, "probe", lambda p: SimpleNamespace(audio_streams=1))
    src = write(tmp_path / "clip.mp4", "a")
    assert refmatch.audio_8k(src).size == 0
    assert refmatch.audio_8k(src).size == 0
    assert len(calls) == 1
    failed = refmatch._AUDIO_FAILED
    assert failed == {str(cache.source_dir(src))}
    write(src, "changed")  # new size -> new cache folder -> retried
    refmatch.audio_8k(src)
    assert len(calls) == 2
    reset_state()
    assert not refmatch._AUDIO_FAILED
    refmatch.audio_8k(src)
    assert len(calls) == 3
