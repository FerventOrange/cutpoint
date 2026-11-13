"""#1038: the scene pass of a recovered part in a container that seeks inexactly (MPEG-TS) seeks a
lead-in earlier; the retry without it; the old scene files removed; WebM split like MP4/MKV.
Mocked unit tests first, then real-ffmpeg regressions on small low-res clips."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import media_info, needs_ffmpeg
from cutpoint import analyze, ffmpeg_utils
from cutpoint.errors import FFmpegError


# ----------------------------------------------------------------- first_frame_time
@pytest.mark.parametrize(
    ("kw", "seek"),
    [({}, None), ({"start": 0.0}, None), ({"start": 6.4}, "6.400")],
    ids=["default", "zero", "positive"],
)
def test_the_first_frame_time_seeks_only_for_a_positive_start(monkeypatch, kw, seek):
    seen = []

    def fake_run(args, **k):
        seen.append(args)
        return "[x] config in time_base: 1/1000\n[x] n:   0 pts:     33 pts_time:0.033"

    monkeypatch.setattr(ffmpeg_utils, "run", fake_run)
    assert ffmpeg_utils.first_frame_time("a.ts", **kw) == pytest.approx(0.033)
    (cmd,) = seen
    if seek is None:
        assert "-ss" not in cmd
    else:  # an input option: before -copyts -i
        assert cmd[cmd.index("-ss") + 1] == seek
        assert cmd.index("-ss") < cmd.index("-copyts") < cmd.index("-i")


# ----------------------------------------------------------------- _lead_in
@pytest.fixture
def first_time(monkeypatch):
    """``first_frame_time`` fake (records its calls; ``box["value"]`` is returned or raised);
    ``probe`` gives a container start of 1.0."""
    calls = []
    box = {"value": 12.0}

    def fake(path, **k):
        calls.append(k)
        if isinstance(box["value"], Exception):
            raise box["value"]
        return box["value"]

    monkeypatch.setattr(analyze, "first_frame_time", fake)
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, 60.0, fstart=1.0))
    return calls, box


@pytest.mark.parametrize(
    ("name", "offset"),
    [("a.ts", 0.0), ("a.mp4", 300.0), ("a.mkv", 300.0), ("a.webm", 300.0)],
    ids=["from-the-start", "mp4", "mkv", "webm"],
)
def test_no_lead_in_for_exact_seekers_or_a_pass_from_the_start(first_time, name, offset):
    calls, _box = first_time
    assert analyze._lead_in(Path(name), offset, 60.0, 0.0) is None
    assert calls == []  # nothing is measured


@pytest.mark.parametrize("offset", [300.0, 6.4, 10.0, 10.5])
def test_the_lead_in_seeks_earlier_and_clamps_at_the_start(first_time, offset):
    calls, _box = first_time
    seek = max(0.0, round(offset - analyze.LEAD_IN_S, 3))
    got = analyze._lead_in(Path("a.ts"), offset, 60.0, 0.0)
    assert got is not None and got[0] == seek
    assert got[1] == pytest.approx(12.0 - 1.0 - offset)  # first frame on the clock, less offset
    assert calls == [{"keyframes": False, "start": seek}]


def test_a_failed_measurement_keeps_the_old_seek(first_time):
    _calls, box = first_time
    box["value"] = FFmpegError("no frame")
    assert analyze._lead_in(Path("a.ts"), 300.0, 60.0, 0.0) is None


@pytest.mark.parametrize(
    ("offset", "readable_from", "seek"),
    [
        (300.0, 295.0, 295.0),  # the damage ended 5 s before: the seek stops there, not at 290
        (300.0, 298.0, 298.0),  # exactly LEAD_IN_MIN_S left: still a lead-in
        (300.0, 120.0, 290.0),  # damage far back: the plain lead-in
        (12.0, 9.5, 9.5),
    ],
    ids=["clamped", "min-left", "far-damage", "near-start"],
)
def test_the_lead_in_never_seeks_before_the_end_of_the_damage(
    first_time, offset, readable_from, seek
):
    calls, _box = first_time
    got = analyze._lead_in(Path("a.ts"), offset, 60.0, readable_from)
    assert got is not None and got[0] == seek >= readable_from
    assert calls == [{"keyframes": False, "start": seek}]  # nothing was decoded before it


@pytest.mark.parametrize(
    ("offset", "readable_from"),
    [(300.0, 298.5), (300.0, 300.0), (300.0, 305.0), (300.0, None)],
    ids=["under-2s-left", "at-the-offset", "past-the-offset", "old-record"],
)
def test_no_lead_in_with_under_two_seconds_left_or_an_unknown_end_of_the_damage(
    first_time, offset, readable_from
):
    calls, _box = first_time
    assert analyze._lead_in(Path("a.ts"), offset, 60.0, readable_from) is None
    assert calls == []


def test_a_seek_clamped_to_zero_has_no_ss_in_the_proxy_args():
    pre, _post, vf, _enc = analyze._proxy_args(60.0, 6.4, None, None, (0.0, 1.25))
    assert "-ss" not in pre
    assert vf.startswith("setpts=PTS-STARTPTS+1.250000/TB,fps=")
    pre, *_ = analyze._proxy_args(60.0, 300.0, None, None, (290.0, -8.5))
    assert pre[pre.index("-ss") + 1] == "290.000"


# ----------------------------------------------------------------- the retry
@pytest.fixture
def retry(tmp_path, monkeypatch):
    """``scene_cuts`` of a recovered part with ffmpeg faked: ``runs`` has each command; ``plan``
    says per run: "" = ok, "raise" = FFmpegError, other text = the stderr of a run exiting 0 that
    made no output (a ``(stderr, out_time)`` tuple: with that much output)."""
    runs, plan = [], []
    monkeypatch.setattr(analyze, "decoder_for", lambda src: None)
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, 60.0, fstart=0.0))
    monkeypatch.setattr(analyze, "first_frame_time", lambda p, **k: 291.0)
    monkeypatch.setattr(analyze.importlib, "import_module", lambda name: None)
    monkeypatch.setattr(analyze, "_detect", lambda proxy, path, **k: [3.0])

    def run(args, total, label, **kw):
        runs.append(args)
        step = plan[len(runs) - 1] if len(runs) <= len(plan) else ""
        if step == "raise":
            raise FFmpegError("ffmpeg failed (exit 1): read error")
        # ffmpeg stops at a read error: one in the lead-in leaves no output past the part start
        if isinstance(step, tuple):
            return ffmpeg_utils.ProgressResult(stderr=step[0], out_time=step[1])
        return ffmpeg_utils.ProgressResult(stderr=step, out_time=0.0 if step else total)

    monkeypatch.setattr(analyze, "run_progress", run)

    def call(name="s.ts", offset=300.0, readable_from="far"):
        if readable_from == "far":  # the default: damage long before the part
            readable_from = offset - 20.0
        return analyze.scene_cuts(
            tmp_path / name, tmp_path, 60.0, offset=offset, readable_from=readable_from
        )

    return runs, plan, call


def seek_of(cmd: list[str]) -> str:
    return cmd[cmd.index("-ss") + 1]


def test_a_clean_lead_in_decode_runs_once(retry):
    runs, _plan, call = retry
    assert call() == [3.0]
    assert len(runs) == 1 and seek_of(runs[0]) == "290.000"


@pytest.mark.parametrize(
    "first", ["raise", "Error during demuxing: Input/output error"], ids=["fails", "read-error"]
)
def test_a_lead_in_that_hits_a_read_error_is_run_again_without_it(retry, caplog, first):
    runs, plan, call = retry
    plan.append(first)
    with caplog.at_level(logging.INFO, logger=analyze.logger.name):
        assert call() == [3.0]
    assert len(runs) == 2
    assert seek_of(runs[0]) == "290.000"
    assert seek_of(runs[1]) == "300.000"  # the old seek
    assert not any("setpts" in a for a in runs[1])
    assert any("lead-in" in r.getMessage() for r in caplog.records if r.levelno == logging.INFO)


def test_a_read_error_later_in_the_part_keeps_the_lead_in_proxy(retry):
    """F2: ffmpeg decoded 30 s of the part before the damage inside it: the plain seek would
    stop there too, so one run, no retry."""
    runs, plan, call = retry
    plan.append(("Error during demuxing: Input/output error", 30.0))
    assert call() == [3.0]
    assert len(runs) == 1 and seek_of(runs[0]) == "290.000"


def test_a_read_error_with_no_output_in_the_part_is_run_again(retry):
    runs, plan, call = retry
    plan.append(("Error during demuxing: Input/output error", 0.0))
    assert call() == [3.0]
    assert [seek_of(r) for r in runs] == ["290.000", "300.000"]


def test_the_lead_in_range_covering_the_damage_is_cut_back_to_its_end(retry, monkeypatch):
    """The recorded damage ends at 296.5, so the pass seeks there, not 10 s before the part, and
    nothing is decoded before it (the measuring decode included)."""
    runs, _plan, call = retry
    starts = []
    monkeypatch.setattr(
        analyze, "first_frame_time", lambda p, **k: starts.append(k["start"]) or 297.0
    )
    assert call(readable_from=296.5) == [3.0]
    assert starts == [296.5] and [seek_of(r) for r in runs] == ["296.500"]


def test_a_part_with_under_two_seconds_after_the_damage_seeks_as_before(retry):
    runs, _plan, call = retry
    assert call(readable_from=299.0) == [3.0]
    assert [seek_of(r) for r in runs] == ["300.000"]
    assert not any("setpts" in a for a in runs[0])


def test_a_part_whose_record_has_no_end_of_the_damage_seeks_as_before(retry):
    runs, _plan, call = retry
    assert call(readable_from=None) == [3.0]
    assert [seek_of(r) for r in runs] == ["300.000"]


def test_the_retry_without_the_lead_in_may_fail_for_good(retry):
    runs, plan, call = retry
    plan += ["raise", "raise"]
    with pytest.raises(FFmpegError):
        call()
    assert len(runs) == 2


def test_without_a_lead_in_a_failure_propagates_and_a_read_error_is_not_retried(retry):
    runs, plan, call = retry
    plan.append("raise")
    with pytest.raises(FFmpegError):
        call("s.mp4")  # exact seeker: no lead-in
    assert len(runs) == 1
    runs.clear()
    plan[:] = ["Input/output error"]
    assert call("s.mp4") == [3.0]
    assert len(runs) == 1


# ----------------------------------------------------------------- the old scene files
def test_the_old_scene_files_go_after_the_save(tmp_path, monkeypatch):
    cdir = tmp_path / "src"
    cdir.mkdir()
    for name in ("scenes.json", "scenes-v2.json", "scenes-windows.json", "other.json"):
        (cdir / name).write_text("[]")
    proxy = cdir / "proxy.mp4"
    proxy.write_bytes(b"x")
    monkeypatch.setattr(analyze, "_scan", lambda p, c: [1.0])
    assert analyze._detect(proxy, cdir / analyze.SCENES_FILE) == [1.0]
    assert sorted(p.name for p in cdir.iterdir()) == sorted(
        ["other.json", "scenes-windows.json", analyze.SCENES_FILE]
    )


def test_the_old_scene_files_stay_when_the_detection_fails(tmp_path, monkeypatch):
    (tmp_path / "scenes.json").write_text("[]")

    def boom(p, c):
        raise FFmpegError("x")

    monkeypatch.setattr(analyze, "_scan", boom)
    with pytest.raises(FFmpegError):
        analyze._detect(tmp_path / "proxy.mp4", tmp_path / analyze.SCENES_FILE)
    assert (tmp_path / "scenes.json").exists()


def test_the_split_pass_drops_the_old_scene_files_too(tmp_path, monkeypatch):
    (tmp_path / "scenes-v2.json").write_text("[]")
    (tmp_path / "keep.json").write_text("[]")
    monkeypatch.setattr(analyze.importlib, "import_module", lambda name: None)
    monkeypatch.setattr(analyze, "_hybrid_plan", lambda *a: (None, 5.0, 2))
    monkeypatch.setattr(analyze, "_hybrid_cuts", lambda *a: [2.0])
    assert analyze.scene_cuts(tmp_path / "s.mp4", tmp_path, 60.0, cpu_free=True) == [2.0]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.json", analyze.SCENES_FILE]


def test_a_file_that_cannot_be_removed_is_ignored(tmp_path, monkeypatch):
    (tmp_path / "scenes.json").write_text("[]")
    (tmp_path / "scenes-v2.json").write_text("[]")
    real_unlink = Path.unlink

    def deny(self, *a, **k):
        if self.name == "scenes.json":
            raise PermissionError("in use")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", deny)
    analyze._drop_old_scene_files(tmp_path)  # no exception; the other file is still removed
    assert (tmp_path / "scenes.json").exists() and not (tmp_path / "scenes-v2.json").exists()


# ----------------------------------------------------------------- real ffmpeg
CUT_PARTS = [
    ("testsrc2", 4.0), ("color=c=red", 4.0), ("testsrc", 3.0), ("smptebars", 4.0),
    ("rgbtestsrc", 4.0), ("color=c=white", 5.0),
]  # fmt: skip
CUT_TIMES = [4.0, 8.0, 11.0, 15.0, 19.0]  # of the 24 s clip


def make_clip(out: Path, *codec: str, fmt: str | None = None) -> Path:
    """The 24 s 160x120 10 fps clip with hard cuts at CUT_TIMES, encoded with ``codec``."""
    args: list[str] = []
    for name, dur in CUT_PARTS:
        sep = ":" if "=" in name else "="
        args += ["-f", "lavfi", "-i", f"{name}{sep}size=160x120:rate=10:duration={dur}"]
    graph = "".join(f"[{i}]" for i in range(len(CUT_PARTS))) + f"concat=n={len(CUT_PARTS)}:v=1[v]"
    fmt_args = ["-f", fmt] if fmt else []
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", *args, "-filter_complex", graph, "-map", "[v]",
         *codec, *fmt_args, str(out)],
        check=True,
    )  # fmt: skip
    return out


def late_copy(src: Path, out: Path, acodec: str) -> Path:
    """``src`` plus a sine audio track, its video 0.25 s late."""
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-itsoffset", "0.25", "-i", str(src), "-f",
         "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=24", "-map", "0:v",
         "-map", "1:a", "-c:v", "copy", "-c:a", acodec, str(out)],
        check=True,
    )  # fmt: skip
    return out


def near(got: list[float], want: list[float]) -> bool:
    return len(got) == len(want) and all(
        abs(a - b) <= analyze.SCENE_GRID_S + 1e-6 for a, b in zip(got, want, strict=True)
    )


def scene(src: Path, cdir: Path, duration: float, offset: float = 0.0) -> list[float]:
    cdir.mkdir()
    lead = {"readable_from": 0.0} if offset else {}  # a recovered part, clean before it
    return analyze.scene_cuts(src, cdir, duration, offset, **lead)


@pytest.fixture
def real(monkeypatch):
    monkeypatch.setattr(analyze, "probe", ffmpeg_utils.probe)
    monkeypatch.setattr(analyze, "decoder_for", lambda src: None)


@pytest.fixture
def ts_2s(tmp_path) -> Path:
    """The clip as MPEG-TS with a keyframe every 2 s (an input seek lands up to 2 s late)."""
    return make_clip(
        tmp_path / "gop2.ts",
        *("-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "20", "-keyint_min", "20",
          "-sc_threshold", "0", "-bf", "0"),
        fmt="mpegts",
    )  # fmt: skip


@needs_ffmpeg
@pytest.mark.parametrize("offset", [6.4, 1.3, 7.0], ids=["6.4", "1.3-mid-gop", "7.0"])
def test_a_recovered_ts_part_keeps_the_cut_a_keyframe_seek_skips(ts_2s, tmp_path, real, offset):
    """``-ss 6.4`` in MPEG-TS decodes from the keyframe at 8.0 and the cut at 8.0 was lost; the
    lead-in decode keeps it (and every cut after the part's first 0.65 s)."""
    duration = ffmpeg_utils.probe(ts_2s).duration
    full = scene(ts_2s, tmp_path / "full", duration)
    assert near(full, CUT_TIMES), full
    part = scene(ts_2s, tmp_path / "part", duration - offset, offset)
    got = [c + offset for c in part]
    want = [c for c in full if c - offset >= 0.65]
    assert near(got, want), (got, want)
    if offset < 8.0 - 0.65:
        assert any(abs(c - 8.0) <= 0.2 for c in got), got


# offsets above LEAD_IN_S: the lead-in seek is above 0 (0.2 and 4.4), so the real seek and the
# measured first frame after it are exercised; the keyframe seek lands at 12.0 and 16.0
LATE_PARTS = [(10.2, 11.0), (14.4, 15.0)]


@needs_ffmpeg
@pytest.mark.parametrize(("offset", "cut"), LATE_PARTS, ids=["10.2", "14.4"])
def test_a_recovered_ts_part_far_into_the_file_keeps_its_first_cut(
    ts_2s, tmp_path, real, offset, cut
):
    duration = ffmpeg_utils.probe(ts_2s).duration
    assert offset - analyze.LEAD_IN_S > 0
    lead = analyze._lead_in(ts_2s, offset, duration - offset, 0.0)
    assert lead is not None and lead[0] == pytest.approx(offset - analyze.LEAD_IN_S)
    got = [c + offset for c in scene(ts_2s, tmp_path / "part", duration - offset, offset)]
    assert any(abs(c - cut) <= 0.2 for c in got), got
    assert near([c for c in got if c > cut + 1.0], [c for c in CUT_TIMES if c > cut + 1.0]), got


@needs_ffmpeg
@pytest.mark.parametrize(("offset", "cut"), [(6.4, 8.0), *LATE_PARTS], ids=["6.4", "10.2", "14.4"])
def test_the_ts_regression_fails_without_the_lead_in(
    ts_2s, tmp_path, monkeypatch, real, offset, cut
):
    """Control: with no lead-in (the code before #1038) the part loses the cut after its start."""
    monkeypatch.setattr(analyze, "_lead_in", lambda *a: None)
    duration = ffmpeg_utils.probe(ts_2s).duration
    part = scene(ts_2s, tmp_path / "part", duration - offset, offset)
    assert not any(abs(c + offset - cut) <= 0.2 for c in part), part


@needs_ffmpeg
def test_a_forced_split_of_a_late_video_matches_one_pass_on_the_gpu(tmp_path, monkeypatch):
    """GPU twin of the late-video split test (#1038 lifted the ``vstart`` gate): the NVDEC half
    and the CPU half of an MP4 whose video starts 0.25 s late give the cuts of one pass."""
    dec = ffmpeg_utils.gpu_decoder()
    if dec is None:
        pytest.skip("no GPU decoder works here")
    base = make_clip(tmp_path / "base.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "20")
    late = late_copy(base, tmp_path / "late.mp4", "aac")
    assert ffmpeg_utils.probe(late).vstart > 0.2
    monkeypatch.setattr(analyze, "probe", ffmpeg_utils.probe)
    monkeypatch.setattr(analyze, "decoder_for", lambda src: None)
    duration = ffmpeg_utils.probe(late).duration
    one = scene(late, tmp_path / "one", duration)
    assert near(one, [c + 0.25 for c in CUT_TIMES]), one
    monkeypatch.setattr(analyze, "decoder_for", lambda src: dec)
    cdir = tmp_path / "hy"
    cdir.mkdir()
    for split in (11.0, 10.9, 15.2, 6.0):
        got = analyze._hybrid_cuts(late, cdir, duration, dec, split, 2)
        assert near(got, one), (split, got, one)


@pytest.fixture
def webm(tmp_path) -> Path:
    encoders = subprocess.run(
        [shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True,
    ).stdout  # fmt: skip
    if "libvpx-vp9" not in encoders:
        pytest.skip("ffmpeg has no libvpx-vp9")
    base = make_clip(
        tmp_path / "base.webm",
        *("-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-g", "20", "-b:v", "200k",
          "-deadline", "realtime", "-cpu-used", "8"),
    )  # fmt: skip
    return late_copy(base, tmp_path / "late.webm", "libopus")


@needs_ffmpeg
def test_a_late_webm_seeks_exactly_for_windows_and_a_split(webm, tmp_path, real):
    """#1038: WebM is an exact-seek container. Edge windows (which seek) match the full pass and a
    forced split of the pass matches one pass."""
    info = ffmpeg_utils.probe(webm)
    assert info.vcodec == "vp9" and info.vstart > 0.2
    assert analyze._seeks_exactly(webm)
    one = scene(webm, tmp_path / "one", info.duration)
    assert near(one, [c + info.vstart for c in CUT_TIMES]), one
    edges = [2.5, 8.0, 11.5, 15.0, 19.0]
    wdir = tmp_path / "windows"
    wdir.mkdir()
    got = analyze.EdgeCuts(webm, wdir, info.duration).around(edges)
    assert near(got, one), (got, one)
    cdir = tmp_path / "hy"
    cdir.mkdir()
    for split in (11.0, 10.9, 15.2):
        got = analyze._hybrid_cuts(webm, cdir, info.duration, None, split, 2)
        assert near(got, one), (split, got, one)
