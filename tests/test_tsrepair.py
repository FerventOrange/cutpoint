"""Joined MPEG-TS/PS recordings whose timestamps jump: a stream copy on ffmpeg's clock (#1052).

The media are made here: short low-res parts joined by binary concatenation, so the second
part's timestamps start over (or ahead) where the first ends.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections import namedtuple
from pathlib import Path

import pytest

from conftest import StubTranscriber
from cutpoint import analyze, cache, cli, clips, extend, jsonout, stage, tsrepair, ui
from cutpoint.boundaries import SpeechSettings
from cutpoint.errors import FFmpegError
from cutpoint.ffmpeg_utils import probe
from cutpoint.planning import Padding
from cutpoint.styles import get_style
from cutpoint.utils import find_videos

pytestmark = [
    pytest.mark.ffmpeg,
    pytest.mark.skipif(
        not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not on PATH"
    ),
]

PART_S = 4  # length of one part (s)


def _part(d: Path, name: str, *, off: float = 0.0, audio: bool = True, fmt: str = "mpegts") -> Path:
    """One short part; ``off`` shifts its timestamps (``-output_ts_offset``)."""
    out = d / name
    cmd = [shutil.which("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
           f"testsrc=size=128x96:rate=10:duration={PART_S}"]  # fmt: skip
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={PART_S}"]
    cmd += ["-c:v", "mpeg2video" if fmt == "mpeg" else "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "mp2" if fmt == "mpeg" else "aac", "-b:a", "32k"]
    cmd += ["-output_ts_offset", str(off), "-f", fmt, str(out)]
    subprocess.run(cmd, check=True)
    return out


def _join(out: Path, *parts: Path) -> Path:
    out.write_bytes(b"".join(p.read_bytes() for p in parts))
    return out


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("tsrepair-media")
    a, b = _part(d, "a.ts"), _part(d, "b.ts")
    silent = [_part(d, "na.ts", audio=False), _part(d, "nb.ts", audio=False)]
    mpg = [_part(d, "pa.mpg", fmt="mpeg"), _part(d, "pb.mpg", fmt="mpeg")]
    mp4 = d / "plain.mp4"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-y", "-i", str(a), "-c", "copy", str(mp4)],
        check=True,
    )
    mkv = d / "plain.mkv"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-y", "-i", str(a), "-c", "copy", str(mkv)],
        check=True,
    )
    return {
        "back": _join(d / "back.ts", a, b),  # the second part starts over: 4 s read as 8
        "three": _join(d / "three.ts", a, b, _part(d, "c.ts")),
        "forward": _join(d / "forward.ts", a, _part(d, "f.ts", off=20)),  # a gap of 20 s
        "noaudio": _join(d / "noaudio.ts", *silent),
        "ps": _join(d / "back.mpg", *mpg),
        "normal": a,
        # backward by 0.1 s only (a 3.9 s offset; 3.4 s sits at 0.93 s, too near the 1 s gate to
        # assert on), a gap of 5 s: ffmpeg corrects (or ignores) them, the length the file says
        # stays well within a second of the one it plays
        "small": _join(d / "small.ts", a, _part(d, "s.ts", off=3.9)),
        "gap5": _join(d / "gap5.ts", a, _part(d, "g.ts", off=9)),
        "mp4": mp4,
        "mkv": mkv,
    }


class Scans:
    """Counts the ffmpeg runs of tsrepair (the stream-copy scan and the remux) and lets a test
    make the next one fail."""

    def __init__(self, monkeypatch) -> None:
        self.calls: list[list[str]] = []
        self.fail: BaseException | None = None
        self._real = tsrepair.run_progress
        monkeypatch.setattr(tsrepair, "run_progress", self)

    def __call__(self, args, total, label, *rest, **kw):
        self.calls.append(list(args))
        if self.fail is not None:
            # as an ffmpeg killed halfway: a part of the output exists
            Path(args[-1]).write_bytes(b"half") if args[-1].endswith(".part.mkv") else None
            raise self.fail
        return self._real(args, total, label, *rest, **kw)

    @property
    def scans(self) -> list[list[str]]:
        return [c for c in self.calls if c[-1] == "-"]

    @property
    def remuxes(self) -> list[list[str]]:
        return [c for c in self.calls if c[-1].endswith(".part.mkv")]


@pytest.fixture
def scans(monkeypatch) -> Scans:
    return Scans(monkeypatch)


def _copy_of(src: Path) -> Path:
    return cache.source_dir(src, create=False) / tsrepair.COPY_FILE


def _no_copy(src: Path) -> None:
    assert not _copy_of(src).exists()
    assert not list(cache.cache_root().glob("*/timeline.*part.mkv"))
    assert not (cache.cache_root() / stage.TIMELINE_INDEX).exists()


# --- detection ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "jumps", "plays_s"),
    [
        ("back", 1, 8), ("three", 2, 12), ("forward", 1, 8), ("noaudio", 1, 8), ("ps", 1, 8),
    ],
    ids=["backward-jump", "three-parts", "forward-gap-over-10s", "no-audio", "program-stream"],
)  # fmt: skip
def test_joined_recording_gets_a_remux_verdict(media, scans, name, jumps, plays_s):
    verdict = tsrepair.check(media[name])

    assert verdict is not None
    assert verdict.remux
    assert verdict.discontinuities == jumps
    assert verdict.read_s == pytest.approx(plays_s, abs=0.5)
    assert abs(verdict.read_s - verdict.probed_s) >= tsrepair.MISMATCH_S
    assert len(scans.scans) == 1


def test_a_joined_ts_in_a_folder_input_is_found_and_checked(media, scans, tmp_path):
    folder = tmp_path / "recordings"
    folder.mkdir()
    shutil.copy(media["back"], folder / "rec.ts")
    shutil.copy(media["ps"], folder / "rec.mpg")

    found = find_videos([folder])

    assert [p.name for p in found] == ["rec.mpg", "rec.ts"]
    assert all(tsrepair.scanned(probe(p)) for p in found)
    assert all(tsrepair.check(p).remux for p in found)


def test_a_recording_without_audio_is_scanned_on_its_video(media, scans):
    tsrepair.check(media["noaudio"])
    tsrepair.check(media["back"])

    maps = [c[c.index("-map") + 1] for c in scans.scans]
    assert maps == ["0:v:0", "0:a:0"]


@pytest.mark.parametrize("name", ["normal", "small", "gap5"])
def test_untroubled_or_small_jumps_are_read_in_place(media, scans, name):
    verdict = tsrepair.check(media[name])

    assert verdict is not None
    assert not verdict.remux
    assert verdict.read_s == pytest.approx(verdict.probed_s, abs=tsrepair.MISMATCH_S)
    if name == "small":
        assert abs(verdict.read_s - verdict.probed_s) < 0.5 * tsrepair.MISMATCH_S
    assert tsrepair.prepare_sources([media[name]]) == {}
    assert not scans.remuxes
    _no_copy(media[name])


@pytest.mark.parametrize("name", ["mp4", "mkv"])
def test_other_containers_run_no_scan(media, scans, name):
    assert tsrepair.check(media[name]) is None
    assert tsrepair.prepare_sources([media[name]]) == {}
    assert scans.calls == []


# --- redirect and clock ------------------------------------------------------------------------


def test_reads_go_through_the_copy_and_the_original_stays_the_path(media, scans):
    src = media["back"]
    assert probe(src).duration == pytest.approx(PART_S, abs=0.3)  # as the file says

    record = tsrepair.prepare_sources([src])

    assert record[str(src)]["remuxed"] is True
    assert record[str(src)]["discontinuities"] == 1
    info = probe(src)
    assert info.duration == pytest.approx(2 * PART_S, abs=0.5)
    assert info.path == src
    copy = _copy_of(src)
    assert copy.is_file()
    assert stage.staged_path(str(src)) == str(copy)
    assert stage.timeline_copy(str(src)) == str(copy)
    # the scan and the repair of the recording itself read the original
    assert stage.staged_path(str(src), repaired=False) == str(src)
    with stage.unrepaired():
        assert stage.staged_path(str(src)) == str(src)
        assert stage.timeline_copy(str(src)) == str(copy)  # a probe of the copy, not a read
    assert stage.staged_path(str(src)) == str(copy)


def test_the_copy_is_a_stream_copy_of_both_streams(media, scans):
    src = media["back"]
    tsrepair.prepare_sources([src])

    info = probe(_copy_of(src))

    assert info.container.startswith("matroska")
    assert info.vcodec == "h264"
    assert info.acodec == "aac"
    assert info.audio_streams == 1


def test_a_clip_after_the_join_is_exported_from_the_copy(media, scans, home, tmp_path, monkeypatch):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    src = media["back"]
    project = tmp_path / "proj"
    project.mkdir()
    record = tsrepair.prepare_sources([src])
    cand = {"id": "h001", "source": str(src), "start": 5.0, "end": 7.0, "duration": 2.0,
            "score": 90, "label": "late"}  # fmt: skip
    hl = {"style": "funny", "suggested_top": 1, "candidates": [cand],
          "sources": [{"path": str(src), "duration": 8.0, "timeline": record[str(src)]}]}  # fmt: skip
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")

    out = clips.export_clips(project, top=1)

    exported = sorted(out.glob("*.mp4"))
    assert len(exported) == 1
    assert exported[0].stat().st_size > 0
    assert probe(exported[0]).duration == pytest.approx(2.0, abs=0.4)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["clips"][0]["source"] == str(src)
    # the copy a project needs is made again by clips when it is gone
    _copy_of(src).unlink()
    scans.calls.clear()
    clips.export_clips(project, top=1, force=True)
    assert _copy_of(src).is_file()
    assert len(scans.remuxes) == 1


def test_analyze_records_the_original_and_the_playing_length(media, scans, monkeypatch):
    monkeypatch.setattr(analyze, "Transcriber", StubTranscriber)
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    src = media["back"]
    proj = cache.cache_root().parent / "proj"

    analyze.analyze(
        src, get_style("funny"), proj, vision=False, use_scenes=False,
        skip=["reactions", "keyframes", "crashes", "dialogue"],
    )  # fmt: skip

    hl = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    (entry,) = hl["sources"]
    assert entry["path"] == str(src)
    assert entry["duration"] == pytest.approx(2 * PART_S, abs=0.5)
    assert entry["timeline"]["remuxed"] is True
    assert entry["timeline"]["probed_s"] == pytest.approx(PART_S, abs=0.3)
    assert all(c["source"] == str(src) for c in hl["candidates"])


# --- cache and lifecycle -----------------------------------------------------------------------


def test_a_second_prepare_runs_no_ffmpeg(media, scans):
    src = media["back"]
    first = tsrepair.prepare_sources([src])
    ran = len(scans.calls)

    assert ran == 2  # the scan and the remux
    assert tsrepair.prepare_sources([src]) == first
    assert len(scans.calls) == ran


def test_a_changed_source_is_checked_again_and_read_in_place_until_then(media, scans, tmp_path):
    src = tmp_path / "rec.ts"
    shutil.copyfile(media["back"], src)
    tsrepair.prepare_sources([src])
    assert stage.staged_path(str(src)) == str(_copy_of(src))
    scans.calls.clear()

    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))

    stage.reload_stage_index()  # a new process: the index entry no longer matches the source
    assert stage.staged_path(str(src)) == str(src)
    tsrepair.prepare_sources([src])
    assert len(scans.scans) == 1
    assert stage.staged_path(str(src)) != str(src)


@pytest.mark.parametrize(
    "failure", [FFmpegError("boom"), KeyboardInterrupt()], ids=["fails", "ctrl-c"]
)
def test_a_failed_or_cancelled_remux_leaves_no_part_and_no_index_entry(media, scans, failure):
    src = media["back"]
    verdict = tsrepair.check(src)
    assert verdict is not None
    scans.fail = failure
    crashed = isinstance(failure, KeyboardInterrupt)

    if crashed:
        with pytest.raises(KeyboardInterrupt):
            tsrepair.ensure_copy(src, verdict)
    else:
        assert tsrepair.ensure_copy(src, verdict) is None

    assert len(scans.remuxes) == 1
    _no_copy(src)
    assert stage.staged_path(str(src)) == str(src)


def test_a_failed_remux_warns_and_the_source_is_read_in_place(media, scans, monkeypatch, caplog):
    result = jsonout.Result("test")
    monkeypatch.setattr(jsonout, "_active", result)
    src = media["back"]
    tsrepair.check(src)
    scans.fail = FFmpegError("boom")

    assert tsrepair.prepare_sources([src]) == {}

    assert [w["code"] for w in result.warnings] == ["timeline_remux_failed"]
    assert result.warnings[0]["files"] == [str(src.absolute())]
    assert "WARNING" in {r.levelname for r in caplog.records}
    assert probe(src).duration == pytest.approx(PART_S, abs=0.3)


def test_a_part_of_a_dead_run_is_removed_before_the_next_remux(media, scans):
    src = media["back"]
    cdir = cache.source_dir(src)
    stale = cdir / "timeline.5000000000.0a1b2c3d.part.mkv"  # no such process
    stale.write_bytes(b"half")

    tsrepair.prepare_sources([src])

    assert not stale.exists()
    assert _copy_of(src).is_file()
    assert not list(cdir.glob("timeline.*.part.mkv"))


FREE = namedtuple("FREE", "total used free")


def test_too_little_free_space_warns_and_continues(media, scans, monkeypatch):
    result = jsonout.Result("test")
    monkeypatch.setattr(jsonout, "_active", result)
    monkeypatch.setattr(tsrepair.shutil, "disk_usage", lambda p: FREE(10, 10, 0))
    src = media["back"]

    assert tsrepair.prepare_sources([src]) == {}
    assert tsrepair.prepare_sources([src], required=["other.ts"]) == {}

    assert [w["code"] for w in result.warnings] == ["timeline_remux_failed"]
    assert "too little free space" in result.warnings[0]["message"]
    assert not scans.remuxes
    _no_copy(src)


def test_a_required_source_without_a_copy_stops_the_run(media, scans, monkeypatch):
    monkeypatch.setattr(tsrepair.shutil, "disk_usage", lambda p: FREE(10, 10, 0))
    src = media["back"]

    with pytest.raises(FFmpegError, match="stream copy"):
        tsrepair.prepare_sources([src], required=[str(src)])


def test_the_scan_runs_at_debug_level_and_still_finds_the_join(media, scans):
    verdict = tsrepair.check(media["back"])  # ffmpeg 5.1 logs the jump only at debug

    (scan,) = scans.scans
    assert scan[scan.index("-v") + 1] == "debug"
    assert verdict is not None
    assert verdict.remux
    assert verdict.discontinuities == 1


def _fill_cache(src: Path) -> Path:
    cdir = cache.source_dir(src)
    (cdir / "scenes").mkdir()
    (cdir / "scenes" / "cuts.json").write_text("[]", encoding="utf-8")
    for name in ("read_errors.json", "scenes-0.json", "audio16k.wav", "audio16k-tracks.json",
                 "loudness.json", "transcript-small.json", "fallback-transcript-small.json"):  # fmt: skip
        (cdir / name).write_text("x", encoding="utf-8")
    return cdir


def test_the_first_copy_drops_the_cut_short_caches_and_keeps_the_audio(media, scans):
    src = media["back"]
    cdir = _fill_cache(src)

    assert tsrepair.prepare_sources([src])

    assert sorted(p.name for p in cdir.iterdir()) == [
        "audio16k-tracks.json", "audio16k.wav", "fallback-transcript-small.json",
        "loudness.json", "timeline.json", "timeline.mkv", "transcript-small.json",
    ]  # fmt: skip


def test_an_unmarked_source_keeps_all_its_caches(media, scans):
    src = media["normal"]
    cdir = _fill_cache(src)
    before = sorted(p.name for p in cdir.rglob("*"))

    assert not tsrepair.check(src).remux

    assert sorted(p.name for p in cdir.rglob("*") if p.name != "timeline.json") == before


def test_json_envelope_carries_the_warning_code(media, scans, monkeypatch, capsys):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    src = media["back"]

    code = cli.main(["detect", str(src), "--no-vision", "--json"])

    assert code == 0
    envelope = json.loads(capsys.readouterr().out)
    warnings = [w for w in envelope["warnings"] if w["code"] == "timeline_remuxed"]
    assert len(warnings) == 1
    assert warnings[0]["files"] == [str(src.absolute())]
    assert _copy_of(src).is_file()


# --- UI ----------------------------------------------------------------------------------------


def test_media_key_changes_when_a_copy_exists(media, scans):
    src = media["back"]
    before = ui.Media.key(src)

    tsrepair.prepare_sources([src])

    after = ui.Media.key(src)
    assert before and after and before != after
    _copy_of(src).unlink()
    assert ui.Media.key(src) == before


# --- review fixes (#1052) ----------------------------------------------------------------------


def _sentinel(cdir: Path) -> Path:
    """A cache file of a read of the recording cut short, next to the ones a copy keeps."""
    (cdir / "audio16k.wav").write_text("x", encoding="utf-8")
    (cdir / "loudness.json").write_text("x", encoding="utf-8")
    (cdir / "transcript-small.json").write_text("x", encoding="utf-8")
    sentinel = cdir / "scenes-v3.json"
    sentinel.write_text("x", encoding="utf-8")
    return sentinel


def _kept(cdir: Path) -> bool:
    return all(
        (cdir / n).exists() for n in ("audio16k.wav", "loudness.json", "transcript-small.json")
    )


def test_caches_read_without_a_copy_are_dropped_by_the_next_successful_copy(
    media, scans, monkeypatch, tmp_path
):
    src = tmp_path / "rec.ts"
    shutil.copyfile(media["back"], src)
    with monkeypatch.context() as m:
        m.setattr(tsrepair.shutil, "disk_usage", lambda p: FREE(10, 10, 0))
        assert tsrepair.prepare_sources([src]) == {}  # read in place
    tsrepair._FAILED.clear()  # a later run: the failed copy is tried again
    cdir = cache.source_dir(src)
    sentinel = _sentinel(cdir)  # a read in place caches a cut-short result
    assert "copy_size" not in json.loads((cdir / "timeline.json").read_text(encoding="utf-8"))

    assert tsrepair.prepare_sources([src])

    assert not sentinel.exists()
    assert _kept(cdir)
    # the copy is there now: later reads cache against it and stay
    again = _sentinel(cdir)
    assert tsrepair.prepare_sources([src])
    assert again.exists()


def test_a_lost_copy_that_cannot_be_made_again_marks_the_caches_for_the_next_copy(
    media, scans, monkeypatch, tmp_path
):
    src = tmp_path / "rec.ts"
    shutil.copyfile(media["back"], src)
    assert tsrepair.prepare_sources([src])
    cdir = cache.source_dir(src)
    (cdir / tsrepair.COPY_FILE).unlink()
    with monkeypatch.context() as m:
        m.setattr(tsrepair.shutil, "disk_usage", lambda p: FREE(10, 10, 0))
        assert tsrepair.prepare_sources([src]) == {}
    tsrepair._FAILED.clear()  # a later run: the failed copy is tried again
    sentinel = _sentinel(cdir)

    assert tsrepair.prepare_sources([src])

    assert not sentinel.exists()
    assert _kept(cdir)
    assert (cdir / tsrepair.COPY_FILE).is_file()


def test_a_source_replaced_at_its_path_stops_getting_the_old_copy(
    media, scans, monkeypatch, tmp_path
):
    src = tmp_path / "rec.ts"
    shutil.copyfile(media["back"], src)
    tsrepair.prepare_sources([src])
    assert stage.staged_path(str(src)) == str(_copy_of(src))  # remembered
    assert stage.timeline_copy(str(src)) == str(_copy_of(src))
    reloads: list[int] = []
    real = stage.reload_stage_index
    monkeypatch.setattr(stage, "reload_stage_index", lambda: (reloads.append(1), real()))

    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))

    assert stage.staged_path(str(src)) == str(src)
    assert stage.timeline_copy(str(src)) is None
    assert reloads == []


@pytest.mark.parametrize("case", ["check-raised", "no-verdict", "no-remux"])
def test_a_required_source_without_a_verdict_to_copy_stops_the_run(media, scans, monkeypatch, case):
    src = {"check-raised": media["back"], "no-verdict": media["mp4"], "no-remux": media["normal"]}[
        case
    ]
    if case == "check-raised":

        def boom(path, info=None):
            raise OSError("unreadable")

        monkeypatch.setattr(tsrepair, "check", boom)

    with pytest.raises(FFmpegError, match="its times are on the timeline of its stream copy"):
        tsrepair.prepare_sources([src], required=[str(src)])


def test_a_missing_required_source_is_left_to_the_callers_read(media, scans, tmp_path):
    gone = tmp_path / "gone.ts"

    try:
        assert tsrepair.prepare_sources([gone], required=[str(gone)]) == {}
    except FFmpegError as err:  # pragma: no cover - the failure being tested
        pytest.fail(f"a missing file must not raise the no-copy error: {err}")


def test_a_copy_missing_from_the_shared_index_is_found_from_the_sources_own_folder(
    media, scans, tmp_path
):
    src = tmp_path / "rec.ts"
    shutil.copyfile(media["back"], src)
    tsrepair.prepare_sources([src])
    copy = _copy_of(src)
    index_path = cache.cache_root() / stage.TIMELINE_INDEX
    # two runs wrote the index at once and this source's entry was lost (another one stays)
    index_path.write_text(json.dumps({"other": {"file": "x", "size": 1, "mtime_ns": 1, "copy_size": 1}}),
                          encoding="utf-8")  # fmt: skip
    stage.reload_stage_index()

    assert stage.staged_path(str(src)) == str(copy)

    timeline = copy.parent / tsrepair.TIMELINE_FILE
    data = json.loads(timeline.read_text(encoding="utf-8"))
    data["copy_size"] += 1
    timeline.write_text(json.dumps(data), encoding="utf-8")
    stage.reload_stage_index()

    assert stage.staged_path(str(src)) == str(src)


# --- review fixes, cycle 2 (#1062) -------------------------------------------------------------


def _exported(media, tmp_path) -> tuple[Path, Path]:
    """``(source, clips folder)`` of one clip after the join, exported from the copy."""
    src = media["back"]
    project = tmp_path / "proj"
    project.mkdir()
    record = tsrepair.prepare_sources([src])
    cand = {"id": "h001", "source": str(src), "start": 5.0, "end": 7.0, "duration": 2.0,
            "score": 90, "label": "late"}  # fmt: skip
    hl = {"style": "funny", "suggested_top": 1, "candidates": [cand],
          "sources": [{"path": str(src), "duration": 8.0, "timeline": record[str(src)]}]}  # fmt: skip
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    return src, clips.export_clips(project, top=1)


def test_extend_makes_the_copy_again_when_it_is_gone(media, scans, home, tmp_path, monkeypatch):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    src, out = _exported(media, tmp_path)
    _copy_of(src).unlink()
    scans.calls.clear()

    ext = extend.extend_clips(out, Padding(1.0, 1.0), speech=SpeechSettings(enabled=False))

    assert len(scans.remuxes) == 1
    assert _copy_of(src).is_file()
    (clip,) = sorted(ext.glob("*.mp4"))
    assert probe(clip).duration == pytest.approx(4.0, abs=0.5)


def test_extend_never_reads_a_remuxed_source_in_place(media, scans, home, tmp_path, monkeypatch):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    src, out = _exported(media, tmp_path)
    _copy_of(src).unlink()
    monkeypatch.setattr(tsrepair.shutil, "disk_usage", lambda p: FREE(10, 10, 0))

    with pytest.raises(FFmpegError, match="stream copy"):
        extend.extend_clips(out, Padding(1.0, 1.0), speech=SpeechSettings(enabled=False))


def test_a_failed_copy_is_not_tried_again_but_still_warned_of(media, scans, monkeypatch, caplog):
    src = media["back"]
    with monkeypatch.context() as m:
        m.setattr(tsrepair.shutil, "disk_usage", lambda p: FREE(10, 10, 0))
        assert tsrepair.prepare_sources([src]) == {}
    logged = len([r for r in caplog.records if r.levelname == "WARNING"])
    assert logged == 1
    result = jsonout.Result("test")
    monkeypatch.setattr(jsonout, "_active", result)

    # space is free again, but this run already read the source as it is
    assert tsrepair.prepare_sources([src]) == {}
    assert tsrepair.prepare_sources([src]) == {}

    assert not scans.remuxes
    _no_copy(src)
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == logged
    assert [w["code"] for w in result.warnings] == ["timeline_remux_failed"]
    # a new process (state reset) tries again
    from cutpoint import state  # noqa: PLC0415

    state.reset_state()
    assert tsrepair.prepare_sources([src])


def test_a_part_name_is_unique_per_remux(media, scans):
    src = media["back"]
    tsrepair.prepare_sources([src])
    names = [Path(c[-1]).name for c in scans.remuxes]
    assert len(names) == 1
    assert tsrepair._PART.fullmatch(names[0])
    assert names[0].startswith(f"timeline.{os.getpid()}.")


def test_a_second_run_waits_for_the_copy_and_uses_it(media, scans, monkeypatch, tmp_path):
    src = tmp_path / "rec.ts"
    shutil.copyfile(media["back"], src)
    verdict = tsrepair.check(src)
    assert verdict is not None
    first_in = threading.Event()
    release = threading.Event()
    real = scans._real

    def slow(args, total, label, *rest, **kw):
        scans.calls.append(list(args))
        first_in.set()
        assert release.wait(30)
        return real(args, total, label, *rest, **kw)

    monkeypatch.setattr(tsrepair, "run_progress", slow)
    try:
        results: list[Path | None] = []
        t1 = threading.Thread(target=lambda: results.append(tsrepair.ensure_copy(src, verdict)))
        t1.start()
        assert first_in.wait(30)
        t2 = threading.Thread(target=lambda: results.append(tsrepair.ensure_copy(src, verdict)))
        t2.start()
        t2.join(1.0)
        assert t2.is_alive()  # waiting for the first run's lock
        release.set()
        t1.join(60)
        t2.join(60)
    finally:
        release.set()
    assert results == [_copy_of(src), _copy_of(src)]
    assert len(scans.remuxes) == 1  # the second run found the copy the first one made
    assert not list(cache.source_dir(src).glob("timeline.*.part.mkv"))
    assert (cache.source_dir(src) / ".timeline.cutpoint-lock").exists() is False


def test_the_lock_file_survives_the_drop_of_stale_caches(media, scans, tmp_path):
    cdir = cache.source_dir(media["back"])
    lock = cdir / ".timeline.cutpoint-lock"
    lock.write_text("x", encoding="utf-8")
    other = cdir / "scenes-v3.json"
    other.write_text("x", encoding="utf-8")

    tsrepair._drop_stale_caches(cdir)

    assert lock.exists()
    assert not other.exists()
