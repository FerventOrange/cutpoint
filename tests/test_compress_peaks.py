"""Tests for --compress-peaks (#326): planning, filter chains, wiring, real ffmpeg."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import media_info, needs_ffmpeg, set_config
from cutpoint import assemble, cli, clips, config, encode, extend, ffmpeg_utils
from cutpoint.ffmpeg_utils import VideoEncoder, clip_momentary, has_soxr, measure_loudness

SPEECH = [-30.0] * 50  # 5 s of speech-level blocks
BURST = [-30.0] * 40 + [-10.0] * 10  # speech, then a burst 20 dB louder


# -- planning ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["light", "strong"])
def test_threshold_sits_above_the_typical_level_of_the_whole_set(preset):
    plan = encode.plan_peaks(preset, [SPEECH, BURST, [-31.0] * 50], [5.0, 5.0, 5.0])
    assert plan is not None
    # the typical level is speech (-30, median of the non-background blocks), not the burst
    assert plan.threshold == pytest.approx(-30.0 + encode.PEAK_PRESETS[preset].threshold, abs=1)


def test_gains_level_clips_toward_each_other_within_the_caps():
    quiet, loud = [-45.0] * 50, [-25.0] * 50
    light = encode.plan_peaks("light", [quiet, SPEECH, loud], [5.0] * 3)
    strong = encode.plan_peaks("strong", [quiet, SPEECH, loud], [5.0] * 3)
    assert light and strong
    assert light.gains[0] > 0 > light.gains[2]  # quiet up, loud down
    assert strong.gains[0] > light.gains[0]  # strong levels further
    for plan in (light, strong):
        p = encode.PEAK_PRESETS[plan.preset]
        assert all(-p.max_cut <= g <= p.max_boost for g in plan.gains if g is not None)
    assert strong.gains[0] == encode.PEAK_PRESETS["strong"].max_boost  # 15 dB away: capped


def test_silent_and_skipped_clips_get_no_filter_but_silence_alone_plans_nothing():
    plan = encode.plan_peaks(
        "light", [SPEECH, [-120.0] * 20, BURST], [5, 2, 5], [True, True, False]
    )
    assert plan is not None and plan.gains[1] is None and plan.gains[2] is None
    assert plan.filter(1) == plan.filter(2) == ""
    assert encode.plan_peaks("strong", [[-120.0] * 10, []], [1.0, 1.0]) is None


@pytest.mark.parametrize("preset", ["light", "strong"])
def test_filter_is_gain_then_rms_compressor_with_the_preset(preset):
    p = encode.PEAK_PRESETS[preset]
    plan = encode.PeakPlan(preset, -22.0, (-3.5, 0.0))
    chain = plan.filter(0)
    assert chain.startswith(",volume=-3.50dB,acompressor=")
    assert f"threshold={10 ** (-22 / 20):.6f}" in chain
    assert f":ratio={p.ratio:g}:attack={p.attack:g}:release={p.release:g}:" in chain
    assert chain.endswith("detection=rms:link=maximum")
    assert plan.filter(1).startswith(",acompressor=")  # no 0 dB volume step


def test_presets_match_the_config_choices():
    assert set(encode.PEAK_PRESETS) == set(config.COMPRESS_PEAKS) - {"off"}


def test_assemble_limiter_is_the_shared_limiter_chain(monkeypatch):
    monkeypatch.setattr(assemble, "has_soxr", lambda: True)
    assert assemble.limiter_filter(-2.3) == encode.limiter_chain(-2.3, True)


# -- clip export chain ------------------------------------------------------------------


def _clip_audio(monkeypatch, peaks: str, streams: int = 1) -> str:
    calls: list[list[str]] = []
    monkeypatch.setattr(encode, "run", lambda args, **_: calls.append(args) or "")
    monkeypatch.setattr(encode, "has_soxr", lambda: False)
    info = media_info(Path("v.mp4"), 12.0, audio_streams=streams)
    fmt = encode.OutputFormat(64, 48, 25)
    encode._encode(VideoEncoder("cpu"), info.path, info, 10.0, 11.5, Path("o.mp4"), fmt,
                   False, None, 0.25, peaks)  # fmt: skip
    graph = calls[0][calls[0].index("-filter_complex") + 1]
    return graph.split(";", 1)[1]


@pytest.mark.parametrize("streams", [0, 1, 2])
def test_clip_chain_off_is_unchanged(monkeypatch, streams):
    fade = encode.audio_fade_filter(1.5, 0.25)
    audio = _clip_audio(monkeypatch, "", streams)
    assert audio.endswith(f"{encode.A_FMT}{fade}[a]")
    assert "acompressor" not in audio and "alimiter" not in audio


def test_clip_chain_compresses_before_the_fade_and_limits_last(monkeypatch):
    peaks = encode.PeakPlan("light", -22.0, (2.0,)).filter(0)
    audio = _clip_audio(monkeypatch, peaks)
    fade = encode.audio_fade_filter(1.5, 0.25)
    limiter = encode.limiter_chain(encode.CLIP_LIMIT_DB, False)
    assert audio == f"[0:a:0]{encode.A_FMT}{peaks}{fade},{limiter}[a]"


# -- assemble ---------------------------------------------------------------------------


def test_assemble_chain_puts_peaks_before_the_fade():
    info = media_info(Path("x.mp4"), 2.0)
    off = assemble._audio_chain(0, info, 0.25)
    on = assemble._audio_chain(0, info, 0.25, peaks=",acompressor=x")
    assert on == off.replace("asetpts=PTS-STARTPTS,", "asetpts=PTS-STARTPTS,acompressor=x,", 1)


def test_concat_copy_rebuilds_the_audio_for_peaks_without_fades(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(assemble, "run", lambda args, **_: calls.append(args) or "")
    files = [tmp_path / "1.mp4", tmp_path / "2.mp4"]
    infos = [media_info(f, 3.0) for f in files]
    assemble.concat_copy(files, tmp_path / "o.mp4", infos, [0.0, 0.0], ["", ""])
    assert "-filter_complex" not in calls[-1]  # nothing to do: plain stream copy
    assemble.concat_copy(files, tmp_path / "o.mp4", infos, [0.0, 0.0], [",acompressor=x", ""])
    graph = calls[-1][calls[-1].index("-filter_complex") + 1]
    assert "[1:a:0]" in graph and graph.count("acompressor=x") == 1
    assert calls[-1][calls[-1].index("-c:v") + 1] == "copy"


def test_limit_without_loudnorm_runs_the_limiter_alone(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(assemble, "run", lambda args, **_: calls.append(args) or "")
    monkeypatch.setattr(assemble, "has_soxr", lambda: True)
    assemble.post_process(tmp_path / "s.mp4", tmp_path / "o.mp4", False, False, None, limit=True)
    args = calls[-1]
    assert args[args.index("-af") + 1] == assemble.limiter_filter(assemble.LIMIT_START_DB)
    assert "loudnorm=" not in " ".join(args) and "320k" in args
    assemble.post_process(tmp_path / "s.mp4", tmp_path / "o.mp4", False, False, None)
    assert "-af" not in calls[-1] and "copy" in calls[-1]


def _manifest(d: Path, entries: list[dict]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"clips": entries}), encoding="utf-8")


def test_peak_filters_skip_baked_clips_and_measure_nothing_when_off(monkeypatch, tmp_path):
    measured: list[Path] = []

    def momentary(f, streams):
        measured.append(f)
        return BURST if f.name == "2.mp4" else SPEECH

    monkeypatch.setattr(assemble, "clip_momentary", momentary)
    files = [tmp_path / "1.mp4", tmp_path / "2.mp4"]
    infos = [media_info(f, 5.0) for f in files]
    assert assemble.peak_filters(tmp_path, files, infos, "off") == ["", ""]
    assert measured == []
    _manifest(tmp_path, [{"file": "1.mp4", "compress_peaks": "light"}, {"file": "2.mp4"}])
    got = assemble.peak_filters(tmp_path, files, infos, "strong")
    assert got[0] == "" and "acompressor" in got[1]
    assert measured == files  # the baked clip still counts for the typical level
    _manifest(tmp_path, [{"file": f.name, "compress_peaks": "strong"} for f in files])
    assert assemble.peak_filters(tmp_path, files, infos, "strong") == ["", ""]


# -- clips manifest, extend -------------------------------------------------------------


def test_extend_does_not_reuse_or_keep_compressed_clips():
    old = {"source": "a.mp4", "start": 1.0, "end": 2.0, "crop": None, "audio_fade": 0.25}
    new = {"source": "a.mp4", "start": 1.0, "end": 2.0, "crop": None}
    assert extend._reusable(old, new, True, 0.25)
    assert not extend._reusable(old | {"compress_peaks": "light"}, new, True, 0.25)


def test_clips_peak_filters_measure_each_span_with_the_export_tracks(monkeypatch):
    seen: list[tuple] = []

    def momentary(src, streams, start, duration):
        seen.append((src.name, streams, start, duration))
        return BURST

    monkeypatch.setattr(clips, "clip_momentary", momentary)
    info = media_info(Path("s.mp4"), 100.0, audio_streams=2)

    class Plan:  # the ClipPlan fields peak_filters reads
        def __init__(self, start, pano=False):
            self.src, self.info, self.start, self.end, self.pano = (
                info.path,
                info,
                start,
                start + 5,
                pano,
            )

    got = clips.peak_filters([Plan(10.0), Plan(50.0, pano=True)], "light")
    assert seen == [("s.mp4", 2, 10.0, 5.0)]  # the 360 clip is not measured
    assert "acompressor" in got[0] and got[1] == ""
    assert clips.peak_filters([Plan(10.0)], "off") == [""]


# -- config and CLI ---------------------------------------------------------------------


@pytest.mark.parametrize(("data", "want"), [({}, "off"), ({"compress_peaks": None}, "off"),
                                           ({"compress_peaks": "strong"}, "strong")])  # fmt: skip
def test_compress_peaks_config_default(home, data, want):
    set_config(home, **data)
    assert config.compress_peaks_default() == want


@pytest.mark.parametrize("value", ["loud", True, 3])
def test_compress_peaks_config_is_checked(home, value):
    set_config(home, compress_peaks=value)
    with pytest.raises(ValueError, match="compress_peaks"):
        config.compress_peaks_default()


@pytest.mark.parametrize("command", ["clips", "assemble", "edit"])
def test_cli_flag_overrides_config_and_rejects_unknown_presets(home, command, capsys):
    parser = cli.build_parser()
    target = ["x"]
    set_config(home, compress_peaks="light")
    a = parser.parse_args([command, *target])
    assert a.compress_peaks is None and cli._compress_peaks(a) == "light"
    a = parser.parse_args([command, *target, "--compress-peaks", "off"])
    assert cli._compress_peaks(a) == "off"
    with pytest.raises(SystemExit):
        parser.parse_args([command, *target, "--compress-peaks", "max"])


def test_assemble_cli_passes_the_preset(home, tmp_path, monkeypatch):
    got: dict = {}
    monkeypatch.setattr(cli, "assemble", lambda d, out, **kw: got.update(kw))
    (tmp_path / "clips").mkdir()
    assert cli.main(["assemble", str(tmp_path / "clips"), "--compress-peaks", "strong"]) == 0
    assert got["compress_peaks"] == "strong" and got["loudnorm"] is False


# -- real ffmpeg ------------------------------------------------------------------------


def _momentary_peak_over_bed(path: Path) -> float:
    m = np.asarray(clip_momentary(path, 1))
    m = m[m > -70]
    return float(m.max() - np.median(m))


@needs_ffmpeg
@pytest.mark.parametrize("preset", ["light", "strong"])
def test_real_burst_is_pulled_toward_the_bed_and_true_peak_stays_under_the_ceiling(
    tmp_path, preset
):
    # 8 s of a speech-level tone with a 1 s burst 20 dB louder, next to a plain tone clip.
    expr = "(0.05+0.45*between(t\\,4\\,5))*sin(2*PI*440*t)"
    loud, bed = tmp_path / "loud.wav", tmp_path / "bed.wav"
    for path, e in ((loud, expr), (bed, "0.04*sin(2*PI*330*t)")):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"aevalsrc=exprs='{e}':s=48000:d=8:c=stereo", str(path)], check=True)  # fmt: skip
    plan = encode.plan_peaks(preset, [clip_momentary(loud, 1), clip_momentary(bed, 1)], [8.0, 8.0])
    assert plan is not None
    out = tmp_path / "out.m4a"
    chain = (
        f"{encode.A_FMT}{plan.filter(0)},{encode.limiter_chain(encode.CLIP_LIMIT_DB, has_soxr())}"
    )
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(loud), "-af", chain,
                    *encode.AUDIO_OUT, str(out)], check=True)  # fmt: skip
    before, after = _momentary_peak_over_bed(loud), _momentary_peak_over_bed(out)
    assert before > 18
    assert after < before - (6 if preset == "light" else 12)
    assert measure_loudness(out)[1] <= assemble.TP_CEILING


# -- review fixes -----------------------------------------------------------------------


def test_clip_momentary_reads_audio_only(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(ffmpeg_utils, "run", lambda args, **_: calls.append(args) or "")
    ffmpeg_utils.clip_momentary(Path("v.mp4"), 2, 1.0, 2.0)
    assert {"-vn", "-sn", "-dn"} <= set(calls[0])


@pytest.mark.parametrize("level", [-66.0, 5.0])
def test_threshold_stays_in_the_range_acompressor_takes(level):
    plan = encode.plan_peaks("strong", [[level] * 50], [5.0])
    assert plan is not None
    assert encode.PEAK_THRESHOLD_MIN <= plan.threshold <= encode.PEAK_THRESHOLD_MAX


def test_clip_export_tags_compressed_clips_only(monkeypatch):
    def args(peaks):
        calls: list[list[str]] = []
        monkeypatch.setattr(encode, "run", lambda a, **_: calls.append(a) or "")
        info = media_info(Path("v.mp4"), 12.0)
        encode._encode(VideoEncoder("cpu"), info.path, info, 1.0, 2.0, Path("o.mp4"),
                       encode.OutputFormat(64, 48, 25), False, None, 0.25, peaks)  # fmt: skip
        return calls[0]

    assert "-metadata" not in args("")
    on = args(",acompressor=x")
    assert on[on.index("-metadata") + 1] == f"comment={encode.PEAK_TAG}"


def test_a_renamed_tagged_clip_counts_as_baked(monkeypatch, tmp_path):
    files = [tmp_path / "001b_x.mp4", tmp_path / "002_y.mp4"]
    _manifest(tmp_path, [{"file": "001_x.mp4", "compress_peaks": "light"}])
    tagged = {files[0].name}
    monkeypatch.setattr(assemble, "_peak_tagged", lambda f: f.name in tagged)
    assert assemble.baked_peaks(tmp_path, files) == [True, False]


def test_a_manifest_entry_wins_over_an_inherited_tag(monkeypatch, tmp_path):
    files = [tmp_path / "001_x.mp4", tmp_path / "002_y.mp4"]
    _manifest(tmp_path, [{"file": "001_x.mp4"}, {"file": "002_y.mp4", "compress_peaks": "light"}])
    monkeypatch.setattr(assemble, "_peak_tagged", lambda f: True)
    assert assemble.baked_peaks(tmp_path, files) == [False, True]


@pytest.mark.parametrize(("preset", "baked", "limited"), [
    ("off", False, False), ("light", False, True), ("light", True, True),
])  # fmt: skip
def test_assemble_limits_without_loudnorm_whenever_compress_peaks_is_on(
    monkeypatch, tmp_path, preset, baked, limited
):
    d = tmp_path / "clips"
    d.mkdir()
    (d / "001_a.mp4").write_bytes(b"x")
    calls: dict = {}
    monkeypatch.setattr(assemble, "probe", lambda p: media_info(Path(p), 3.0))
    monkeypatch.setattr(
        assemble, "peak_filters",
        lambda _d, files, _i, p: [""] * len(files) if p == "off" or baked else [",acompressor=x"],
    )  # fmt: skip

    def join(files, infos, clips_dir, out, *a):
        calls["join_peaks"] = a[-1]
        out.write_bytes(b"joined")
        return 0.0

    def ceiling(stage, out, captions, model, dst, loudnorm, language=None):
        calls["ceiling"] = loudnorm
        dst.write_bytes(b"final")
        return -14.0, -2.0

    monkeypatch.setattr(assemble, "_join", join)
    monkeypatch.setattr(assemble, "normalize_with_ceiling", ceiling)
    monkeypatch.setattr(assemble, "write_timelines", lambda i, out, f: (out, out))
    monkeypatch.setattr("cutpoint.srt.write_final_srt", lambda *a, **k: None)
    assemble.assemble(d, tmp_path / "final.mp4", compress_peaks=preset)
    assert calls.get("ceiling", "not run") == (False if limited else "not run")
    assert calls["join_peaks"] == ([""] if preset == "off" or baked else [",acompressor=x"])
    assert (tmp_path / "final.mp4").read_bytes() == (b"final" if limited else b"joined")


@needs_ffmpeg
def test_real_peak_tag_survives_in_the_mp4(tmp_path):
    tagged, plain = tmp_path / "t.mp4", tmp_path / "p.mp4"
    for path, meta in ((tagged, ["-metadata", f"comment={encode.PEAK_TAG}"]), (plain, [])):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=d=0.2",
                        *encode.AUDIO_OUT, *meta, "-movflags", "+faststart", str(path)],
                       check=True)  # fmt: skip
    assert assemble._peak_tagged(tagged) and not assemble._peak_tagged(plain)
