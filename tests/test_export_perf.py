"""Audit #1085 export perf rows: A233 (encode.py warnings), A731 (limiter passes), A733
(parallel peak measurement), A734 (extend in parallel), A737 (stored encoder verdicts)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from conftest import media_info
from cutpoint import assemble, clips, encode, extend, ffmpeg_utils, jsonout, paths
from cutpoint.boundaries import SpeechSettings
from cutpoint.planning import Padding
from test_extend import _two_clips, reads  # noqa: F401 - reads is a fixture

# -- A233 ---------------------------------------------------------------------------------------


def test_complete_clip_with_read_errors_is_a_json_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(encode, "_export", lambda *a, **kw: "Invalid data found")
    monkeypatch.setattr(encode, "probe", lambda p: media_info(p, 10.0))
    monkeypatch.setattr(encode, "READ_ERROR_RE", type("R", (), {"findall": lambda s, t: ["bad"]})())
    src = tmp_path / "s.mp4"
    with jsonout.run("clips") as res:
        got = encode._export_checked(
            src, media_info(src), 0.0, 10.0, tmp_path / "p.mp4", encode.OutputFormat(1280, 720, 30),
            False, None, 0.0, tmp_path / "001_a.mp4",
        )  # fmt: skip
    assert got is not None and got["failure_time"] is None
    assert [(w["code"], w["files"]) for w in res.warnings] == [("read_errors_in_clip", [str(src)])]


def test_log_footage_is_a_json_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(encode, "is_high_bit_sdr", lambda info: True)
    monkeypatch.setattr(encode, "decode_raw", lambda *a, **kw: (f for f in [(0.0, np.zeros(1))]))
    monkeypatch.setattr(encode, "log_stats", lambda frame: (0.1, 0.2, 0.7))
    src = tmp_path / "log.mp4"
    with jsonout.run("clips") as res:
        encode.warn_if_log(src, media_info(src), 1.0)
    assert [(w["code"], w["files"]) for w in res.warnings] == [("log_footage", [str(src)])]


# -- A731 ---------------------------------------------------------------------------------------


def test_limiter_passes_burn_the_captions_and_measure_the_gain_once(monkeypatch, tmp_path):
    calls: list[dict] = []
    measured: list[Path] = []
    results = iter([(-14.0, -0.5), (-14.1, -2.5)])
    final = tmp_path / ".f.part.mp4"

    def post_process(stage, out, loudnorm, captions, model, limit_db, ass=None, dst=None, **kw):
        calls.append({"captions": captions, "dst": dst, **kw})
        (dst or out).write_bytes(b"x")
        return tmp_path / "f.ass"

    monkeypatch.setattr(assemble, "post_process", post_process)
    monkeypatch.setattr(assemble, "loudnorm_filter", lambda s: measured.append(s) or "volume=1dB")
    monkeypatch.setattr(assemble, "measure_loudness", lambda p: next(results))
    got = assemble.normalize_with_ceiling(
        tmp_path / "s.mp4", tmp_path / "f.mp4", True, None, final, True
    )
    assert got == (-14.1, -2.5)
    assert measured == [tmp_path / "s.mp4"]  # measured once
    assert [c["captions"] for c in calls] == [True, False]
    assert calls[0].get("video_from") is None
    assert calls[1]["video_from"] == final and calls[1]["gain"] == "volume=1dB"
    assert calls[1]["dst"] != final and not calls[1]["dst"].exists()  # moved over the final
    assert final.read_bytes() == b"x"


def test_post_process_with_video_from_copies_that_video_and_the_stage_audio(monkeypatch, tmp_path):
    ran: list[list[str]] = []
    monkeypatch.setattr(assemble, "run", lambda args, **_: ran.append(args) or "")
    monkeypatch.setattr(assemble, "has_soxr", lambda: True)
    monkeypatch.setattr(assemble, "probe", lambda p: media_info(p, 9.0, vduration=8.9))
    monkeypatch.setattr(assemble, "loudnorm_filter", lambda s: pytest.fail("gain was passed in"))
    assemble.post_process(
        tmp_path / "s.mp4", tmp_path / "o.mp4", True, False, None,
        dst=tmp_path / "r.mp4", gain="volume=2.00dB", video_from=tmp_path / "v.mp4",
    )  # fmt: skip
    args = ran[-1]
    assert [args[i + 1] for i, a in enumerate(args) if a == "-i"] == [
        str((tmp_path / "s.mp4").resolve()),
        str((tmp_path / "v.mp4").resolve()),
    ]
    assert args[args.index("-map") + 1] == "1:v:0" and "0:a:0" in args
    assert args[args.index("-c:v") + 1] == "copy"
    assert "volume=2.00dB" in args[args.index("-af") + 1]
    assert "-vf" not in args


# -- A733 ---------------------------------------------------------------------------------------


def test_plan_set_measures_the_clips_at_once_in_order():
    barrier = threading.Barrier(2, timeout=10)  # both measurements must be in flight together

    def measure(i: int, chain: str) -> list[float]:
        if not chain:
            barrier.wait()
        return [-20.0 - i, -18.0 - i, -16.0]

    assert encode.plan_set("light", measure, [5.0, 5.0], [True, True]) is not None


# -- A734 ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.usefixtures("reads")
def test_extend_exports_on_the_pool_only_with_several_workers(monkeypatch, tmp_path, workers):
    pooled: list[int] = []
    real = clips.export_pool

    def export_pool(jobs, n):
        pooled.append(n)
        return real(jobs, n)

    out = extend.extend_clips(
        _two_clips(tmp_path),
        Padding(1.0, 1.0),
        speech=SpeechSettings(enabled=False),
        parallel=extend.ParallelExport(
            lambda count, pano: workers, clips.warm_source_checks, export_pool
        ),
    )
    entries = json.loads((out / "manifest.json").read_text("utf-8"))["clips"]
    assert [e["file"] for e in entries] == ["001_a.mp4", "002_b.mp4"]  # plan order either way
    assert pooled == ([2] if workers > 1 else [])


# -- A737 ---------------------------------------------------------------------------------------


@pytest.fixture
def fake_probe(monkeypatch, tmp_path):
    """An ffmpeg file, a counting encoder probe and a way to start a new process."""
    exe = tmp_path / "ffmpeg.exe"
    exe.write_bytes(b"ffmpeg 1")
    monkeypatch.setattr(ffmpeg_utils, "find_tool", lambda name: str(exe))
    calls: list[str] = []

    def probe(enc):
        calls.append(enc.kind)
        return None

    monkeypatch.setattr(ffmpeg_utils, "_encoder_probe", probe)

    def new_process():
        ffmpeg_utils.encoder_error.cache_clear()
        ffmpeg_utils.decoder_error.cache_clear()
        ffmpeg_utils._probe_stamp.cache_clear()

    return exe, calls, new_process


def test_encoder_verdict_is_kept_for_the_next_process(fake_probe):
    _exe, calls, new_process = fake_probe
    enc = ffmpeg_utils.VideoEncoder("amf")
    assert ffmpeg_utils.encoder_error(enc) is None
    new_process()
    assert ffmpeg_utils.encoder_error(enc) is None
    assert calls == ["amf"]
    assert (paths.cache_dir() / ffmpeg_utils.PROBE_FILE).is_file()


def test_encoder_verdict_is_probed_again_for_another_ffmpeg_build(fake_probe):
    exe, calls, new_process = fake_probe
    enc = ffmpeg_utils.VideoEncoder("amf")
    ffmpeg_utils.encoder_error(enc)
    exe.write_bytes(b"ffmpeg 2, longer")  # another size: an upgraded ffmpeg
    new_process()
    ffmpeg_utils.encoder_error(enc)
    assert calls == ["amf", "amf"]


def test_working_verdict_expires(fake_probe):
    _exe, calls, new_process = fake_probe
    enc = ffmpeg_utils.VideoEncoder("amf")
    ffmpeg_utils.encoder_error(enc)
    file = paths.cache_dir() / ffmpeg_utils.PROBE_FILE
    data = json.loads(file.read_text("utf-8"))
    for v in data.values():
        v["at"] -= ffmpeg_utils.PROBE_OK_TTL_S + 1
    file.write_text(json.dumps(data), encoding="utf-8")
    new_process()
    ffmpeg_utils.encoder_error(enc)
    assert calls == ["amf", "amf"]


def test_find_encoder_refresh_probes_again(fake_probe):
    _exe, calls, new_process = fake_probe
    ffmpeg_utils.find_encoder("amf")
    new_process()
    ffmpeg_utils.find_encoder("amf")
    assert calls == ["amf"]
    ffmpeg_utils.find_encoder("amf", refresh=True)
    assert calls == ["amf", "amf"]


def test_a_corrupt_verdict_file_is_probed_again(fake_probe):
    _exe, calls, _new = fake_probe
    file = paths.cache_dir() / ffmpeg_utils.PROBE_FILE
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("{not json", encoding="utf-8")
    assert ffmpeg_utils.encoder_error(ffmpeg_utils.VideoEncoder("amf")) is None
    assert calls == ["amf"]


def test_a_failed_verdict_is_not_stored(fake_probe, monkeypatch):
    _exe, calls, new_process = fake_probe
    monkeypatch.setattr(ffmpeg_utils, "_encoder_probe", lambda enc: calls.append("x") or "busy")
    enc = ffmpeg_utils.VideoEncoder("amf")
    assert ffmpeg_utils.encoder_error(enc) == "busy"
    assert ffmpeg_utils.encoder_error(enc) == "busy"  # this process remembers it
    assert calls == ["x"]
    assert not (paths.cache_dir() / ffmpeg_utils.PROBE_FILE).exists()
    new_process()
    ffmpeg_utils.encoder_error(enc)
    assert calls == ["x", "x"]  # the next process probes again


def test_cpu_retry_success_forgets_the_stored_working_verdict(fake_probe, monkeypatch):
    _exe, calls, new_process = fake_probe
    enc = ffmpeg_utils.VideoEncoder("amf")
    assert ffmpeg_utils.encoder_error(enc) is None  # stored as working
    monkeypatch.setattr(ffmpeg_utils, "_GPU_ENCODE_FAILED", set())
    monkeypatch.setattr(ffmpeg_utils, "_GPU_ENCODE_SUSPECT", {})
    monkeypatch.setattr(ffmpeg_utils, "video_encoder", lambda: enc)
    monkeypatch.setattr(ffmpeg_utils, "_gpu_off", lambda: False)

    def encode_once(e):
        if e.kind != "cpu":
            raise ffmpeg_utils.FFmpegError("out of memory")

    ffmpeg_utils.encode_with_fallback(encode_once, "001_a.mp4")
    new_process()
    ffmpeg_utils.encoder_error(enc)
    assert calls == ["amf", "amf"]  # probed again, not read from the file


def test_decoder_verdict_is_kept_for_the_next_process(fake_probe, monkeypatch):
    _exe, _calls, new_process = fake_probe
    n: list[int] = []
    monkeypatch.setattr(ffmpeg_utils, "_decoder_probe", lambda dec: n.append(1) or None)
    dec = ffmpeg_utils.HwDecoder("vaapi", "/dev/dri/renderD128")
    assert ffmpeg_utils.decoder_error(dec) is None
    new_process()
    assert ffmpeg_utils.decoder_error(dec) is None
    assert n == [1]
