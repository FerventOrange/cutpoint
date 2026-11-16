"""The analysis audio and the waveform follow the container's clock: a forward gap in a TS's
timestamps is silence in them, not cut out (#1056)."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from _tsmedia import PART_S, join_parts, make_part
from conftest import media_info
from cutpoint import analyze, cache, envelope, tsrepair
from cutpoint.ffmpeg_utils import AUDIO_CLOCK, extract_audio, probe, wav_duration
from cutpoint.sources import READ_ERRORS_FILE

pytestmark = [
    pytest.mark.ffmpeg,
    pytest.mark.skipif(
        not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not on PATH"
    ),
]

GAP = (PART_S + 0.5, 9 - 0.5)  # inside the silent stretch of gap5/gap5_2tr (s): 4 to 9


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("audio-clock-media")
    a, b = make_part(d, "a.ts"), make_part(d, "b.ts")
    a2 = make_part(d, "a2.ts", tracks=2)
    mkv = d / "delayed.mkv"  # the audio starts 0.5 s after the video
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-y", "-i", str(a), "-itsoffset", "0.5",
         "-i", str(a), "-map", "0:v", "-map", "1:a", "-c", "copy", str(mkv)],
        check=True,
    )  # fmt: skip
    return {
        "normal": a,
        "back": join_parts(d / "back.ts", a, b),
        "small": join_parts(d / "small.ts", a, make_part(d, "s.ts", off=3.9)),
        "gap5": join_parts(d / "gap5.ts", a, make_part(d, "g.ts", off=9)),
        "gap5_2tr": join_parts(d / "gap5-2tr.ts", a2, make_part(d, "g2.ts", off=9, tracks=2)),
        "delayed": mkv,
    }


def _samples(wav: Path) -> np.ndarray:
    with wave.open(str(wav), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)


def _rms(wav: Path, t0: float, t1: float) -> float:
    x = _samples(wav)[int(t0 * 16000) : int(t1 * 16000)]
    return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0


def _extract(src: Path, tmp_path: Path, **kw) -> tuple[Path, float]:
    wav = tmp_path / "audio16k.wav"
    extract_audio(src, wav, probe(src), **kw)
    return wav, probe(src).duration


@pytest.mark.parametrize("name", ["gap5", "gap5_2tr"])
def test_a_gap_in_the_timestamps_is_silence_in_the_audio(media, tmp_path, name):
    wav, probed = _extract(media[name], tmp_path)

    assert probed > 2 * PART_S + 4  # the file plays the gap
    assert wav_duration(wav) == pytest.approx(probed, abs=0.05)
    assert _rms(wav, *GAP) < 5  # silent
    assert _rms(wav, 0.5, PART_S - 0.5) > 200  # the first part is sound
    assert _rms(wav, 9.5, 12.5) > 200  # and so is the part after the gap, at its own time


@pytest.mark.parametrize(
    ("name", "length"), [("normal", PART_S), ("back", 2 * PART_S), ("small", 2 * PART_S - 0.1)]
)
def test_audio_without_a_forward_gap_is_as_long_as_before(media, tmp_path, name, length):
    wav, _ = _extract(media[name], tmp_path)  # a backward jump: the probe says less than it plays

    assert wav_duration(wav) == pytest.approx(length, abs=0.5)
    assert _rms(wav, 0.5, PART_S - 0.5) > 200


def test_audio_that_starts_after_the_video_starts_with_silence(media, tmp_path):
    wav, _ = _extract(media["delayed"], tmp_path)

    assert _rms(wav, 0.0, 0.45) < 5
    assert _rms(wav, 0.7, 3.5) > 200


def test_a_seeked_extract_is_as_long_as_the_rest_of_the_file(media, tmp_path):
    start = 10.0  # past the gap
    wav, probed = _extract(media["gap5"], tmp_path, start=start)

    assert wav_duration(wav) == pytest.approx(probed - start, abs=0.15)
    assert _rms(wav, 0.2, 1.5) > 200


def test_a_gap_is_no_read_error(media, tmp_path):
    src = media["gap5"]
    info = probe(src)
    wav = tmp_path / "audio16k.wav"
    res = extract_audio(src, wav, info)

    assert analyze.check_read(src, info, 0.0, wav, tmp_path, res) is None
    assert not (tmp_path / READ_ERRORS_FILE).exists()


def test_the_waveform_follows_the_clock_too(media):
    one = envelope.ffmpeg_args("a.ts")
    assert one[one.index("-af") + 1] == AUDIO_CLOCK
    two = envelope.ffmpeg_args("a.ts", 2)
    graph = two[two.index("-filter_complex") + 1]
    assert graph.endswith(f"normalize=0:duration=longest,{AUDIO_CLOCK}[a]")
    done = subprocess.run(
        [shutil.which("ffmpeg"), *envelope.ffmpeg_args(str(media["gap5"]))[1:]],
        capture_output=True, check=True,
    )  # fmt: skip
    seconds = len(done.stdout) / 2 / envelope.RATE
    assert seconds == pytest.approx(probe(media["gap5"]).duration, abs=0.1)


# --- caches made before the audio followed the clock --------------------------------------------


def _wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))


def _old_cache(cdir: Path, wav_s: float, record: dict | None = None) -> None:
    """The folder of a run before the clock file: the WAV, what is made of it, an unrelated cache."""
    cdir.mkdir(parents=True, exist_ok=True)
    _wav(cdir / "audio16k.wav", wav_s)
    for name in ("loudness.json", "transcript-small.json", "fallback-transcript-small.json",
                 "events-audio-ab12.json", "voice-ab12.npz", "scenes-v3.json"):  # fmt: skip
        (cdir / name).write_text("x", encoding="utf-8")
    if record is not None:
        cache.save_json(cdir / READ_ERRORS_FILE, record)


LENGTH_ONLY = {"version": 1, "messages": [], "missing_s": 5.0, "audio_s": 20.0}
FFMPEG_ERROR = {"version": 1, "messages": ["Error during demuxing: Input/output error"]}


def test_a_wav_that_left_out_a_gap_is_made_again_with_what_came_of_it(tmp_path, caplog):
    cdir = tmp_path / "cdir"
    _old_cache(cdir, 20.0, LENGTH_ONLY)
    caplog.set_level("INFO", logger="cutpoint.analyze")

    analyze._audio_clock(Path("a.ts"), media_info(Path("a.ts"), 25.0), 0.0, cdir)

    assert sorted(p.name for p in cdir.iterdir()) == ["scenes-v3.json"]
    assert "left out 5.0 s where the timestamps jump; extracting it again" in caplog.text


def test_a_wav_of_full_length_is_kept_and_marked(tmp_path):
    cdir = tmp_path / "cdir"
    _old_cache(cdir, 25.0)
    before = {p.name for p in cdir.iterdir()}

    analyze._audio_clock(Path("a.ts"), media_info(Path("a.ts"), 25.0), 0.0, cdir)

    assert {p.name for p in cdir.iterdir()} == before | {analyze.CLOCK_FILE}
    assert cache.load_json(cdir / analyze.CLOCK_FILE) == {"version": 1}


def test_a_wav_cut_short_by_an_ffmpeg_error_is_kept_and_marked(tmp_path):
    cdir = tmp_path / "cdir"
    _old_cache(cdir, 20.0, FFMPEG_ERROR)
    before = {p.name for p in cdir.iterdir()}

    analyze._audio_clock(Path("a.ts"), media_info(Path("a.ts"), 25.0), 0.0, cdir)

    assert {p.name for p in cdir.iterdir()} == before | {analyze.CLOCK_FILE}
    assert cache.load_json(cdir / READ_ERRORS_FILE) == FFMPEG_ERROR


def test_a_recovered_part_is_measured_from_its_offset(tmp_path):
    cdir = tmp_path / "cdir"
    _old_cache(cdir, 15.0)  # a part from 10 s of a 25 s source

    analyze._audio_clock(Path("a.ts"), media_info(Path("a.ts"), 25.0), 10.0, cdir)

    assert (cdir / "audio16k.wav").exists()
    assert (cdir / analyze.CLOCK_FILE).exists()


def test_the_check_is_made_once(tmp_path):
    cdir = tmp_path / "cdir"
    _old_cache(cdir, 25.0)
    analyze._audio_clock(Path("a.ts"), media_info(Path("a.ts"), 25.0), 0.0, cdir)
    before = {p.name: p.stat().st_mtime_ns for p in cdir.iterdir()}

    # a probe that says the file is longer: the marked WAV is not judged again
    analyze._audio_clock(Path("a.ts"), media_info(Path("a.ts"), 40.0), 0.0, cdir)

    assert {p.name: p.stat().st_mtime_ns for p in cdir.iterdir()} == before


def test_without_a_wav_nothing_is_written(tmp_path):
    analyze._audio_clock(Path("a.ts"), media_info(Path("a.ts"), 25.0), 0.0, tmp_path)

    assert not list(tmp_path.iterdir())


def test_a_wav_that_cannot_be_removed_is_judged_again_next_run(monkeypatch, tmp_path):
    cdir = tmp_path / "cdir"
    _old_cache(cdir, 20.0, LENGTH_ONLY)
    unlink = Path.unlink

    def locked(self, missing_ok=False):
        if self.name == "audio16k.wav":
            raise PermissionError("in use")
        unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked)
    analyze._audio_clock(Path("a.ts"), media_info(Path("a.ts"), 25.0), 0.0, cdir)
    assert (cdir / "audio16k.wav").exists()
    assert not (cdir / analyze.CLOCK_FILE).exists()

    monkeypatch.setattr(Path, "unlink", unlink)
    analyze._audio_clock(Path("a.ts"), media_info(Path("a.ts"), 25.0), 0.0, cdir)
    assert not (cdir / "audio16k.wav").exists()


def test_a_repaired_source_is_measured_by_the_probe_of_its_copy(media, tmp_path):
    """A source with a jump is read through its copy, so ``info`` is the copy's probe (8 s; the
    file itself says 4 s) and the WAV, 8 s long, is not short of it."""
    src = media["back"]
    cdir = tmp_path / "cdir"
    cdir.mkdir()
    extract_audio(src, cdir / "audio16k.wav", probe(src))
    assert tsrepair.prepare_sources([src])
    info = probe(src)  # redirected to the copy

    analyze._audio_clock(src, info, 0.0, cdir)

    assert wav_duration(cdir / "audio16k.wav") == pytest.approx(info.duration, abs=0.5)
    assert (cdir / analyze.CLOCK_FILE).exists()


def test_a_gap_free_wav_of_a_gap_source_is_kept(media, tmp_path):
    src = media["gap5"]
    cdir = tmp_path / "cdir"
    cdir.mkdir()
    extract_audio(src, cdir / "audio16k.wav", probe(src))

    analyze._audio_clock(src, probe(src), 0.0, cdir)

    assert (cdir / "audio16k.wav").exists()
    assert (cdir / analyze.CLOCK_FILE).exists()
