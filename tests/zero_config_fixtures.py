"""Synthetic footage for the zero-config tests (#899, tests/test_zero_config*.py): one tiny,
real video file per footage category, made with ffmpeg's lavfi sources, and the helpers the
test modules share. Not collected as tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from dataclasses import replace
from pathlib import Path

from conftest import envelope
from cutpoint import analyze, category, cli, edit, extras

SIZE = "320x180"  # the footage probe works on 320x180 frames
RATE = 10
TALK_SECONDS = 500  # a talk recording is at least 480 s (category.TALK_MIN_S)
APPLE = "Apple Video Media Handler"


def _ffmpeg(*args: str | Path) -> None:
    exe = shutil.which("ffmpeg")
    assert exe, "ffmpeg is needed"
    subprocess.run([exe, "-v", "error", "-y", *map(str, args)], check=True)


def _tone(seconds: int, rate: int = 16000) -> list[str]:
    # quiet, with two loud bursts, so the signal analysis has candidates to find
    loud = "volume=enable='between(t,5,8)+between(t,13,16)':volume=30"
    src = f"sine=frequency=440:sample_rate={rate}:duration={seconds},volume=0.02,{loud}"
    return ["-f", "lavfi", "-i", src]


def _moving(seconds: int) -> list[str]:
    src = f"gradients=size={SIZE}:rate={RATE}:speed=0.05:nb_colors=4"
    return ["-f", "lavfi", "-i", src, "-t", str(seconds)]


ENCODE = ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac"]


def screen_capture(out: Path, seconds: int = 20) -> Path:
    """A game-like screen recording: a moving picture under a HUD that never changes (high
    contrast stripes in a corner) and the signature of NVIDIA's recorder."""
    hud = f"nullsrc=size=96x24:rate={RATE},geq=lum='if(lt(mod(X,4),2),255,0)':cb=128:cr=128"
    _ffmpeg(
        *_moving(seconds), "-f", "lavfi", "-i", hud, *_tone(seconds),
        "-filter_complex", "[0][1]overlay=8:8:shortest=1,format=yuv420p",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
        "-metadata:s:v", "handler_name=VideoHandle", "-shortest", out,
    )  # fmt: skip
    return out


def phone_video(out: Path, seconds: int = 20, tagged: bool = True) -> Path:
    """Footage with nothing fixed on screen. ``tagged``: the container names a phone camera
    (an Apple handler); untagged, nothing says what it is."""
    tags = ["-metadata:s:v", f"handler_name={APPLE}"] if tagged else []
    _ffmpeg(*_moving(seconds), *_tone(seconds), *ENCODE, *tags, "-shortest", out)
    return out


def untagged_video(out: Path, seconds: int = 20) -> Path:
    return phone_video(out, seconds, tagged=False)


def action_cam(out: Path, seconds: int = 20) -> Path:
    """Footage with a GoPro-style telemetry track: a data stream whose sample entry is
    ``gpmd``. ffmpeg cannot write that tag, so a timed-text track is written and its
    four-letter sample entry renamed. The track holds no readable GPMF samples."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        srt, plain = Path(tmp, "t.srt"), Path(tmp, "plain.mp4")
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        _ffmpeg(
            *_moving(seconds), *_tone(seconds), "-i", srt, "-map", "0", "-map", "1", "-map", "2",
            *ENCODE, "-c:s", "mov_text", "-t", str(seconds), plain,
        )  # fmt: skip
        data = plain.read_bytes()
    assert data.count(b"tx3g") == 1
    out.write_bytes(data.replace(b"tx3g", b"gpmd"))
    return out


def talk_recording(out: Path, seconds: int = TALK_SECONDS) -> Path:
    """A long, still shot (colour bars) with a tone; the words come from the test's stub
    transcriber."""
    _ffmpeg(
        "-f", "lavfi", "-i", f"smptebars=size={SIZE}:rate=1",
        *_tone(seconds, 8000), *ENCODE, "-t", str(seconds), out,
    )  # fmt: skip
    return out


# -- shared by the test modules ----------------------------------------------------------------

# model-backed steps (sound classifiers, CLIP, crash detection, the dialogue text model)
NO_MODELS = ["--skip", "reactions", "keyframes", "crashes", "dialogue"]


# varied text: a transcript of one repeated word is dropped as a Whisper hallucination
VOCABULARY = [
    "so",
    "today",
    "we",
    "are",
    "going",
    "to",
    "talk",
    "about",
    "how",
    "the",
    "new",
    "schedule",
    "changes",
    "what",
    "every",
    "team",
    "does",
    "during",
    "the",
    "first",
    "week",
    "and",
    "why",
    "that",
    "matters",
    "for",
    "the",
    "people",
    "who",
    "build",
    "and",
    "ship",
    "the",
    "product",
    "before",
    "the",
    "quarter",
    "ends",
    "and",
    "the",
    "next",
    "plan",
    "begins",
]


class StubWhisper:
    """No Whisper: the speaker talks from the start for 80% of a talk recording; the other
    footage has no speech."""

    model_name, device = "stub", "cpu"

    def __init__(self, language: str | None = None, **_kw) -> None:
        self.language = language

    def transcribe(self, wav: Path) -> list[dict]:
        with wave.open(str(wav)) as w:
            seconds = w.getnframes() / w.getframerate()
        if seconds < TALK_SECONDS - 1:
            return []
        out, t, n = [], 0.0, 0
        while t < 0.8 * seconds:
            said = [VOCABULARY[(n * 8 + i) % len(VOCABULARY)] for i in range(8)]
            words = [
                {"start": t + 0.5 * i, "end": t + 0.5 * i + 0.4, "word": f" {w}"}
                for i, w in enumerate(said)
            ]
            out.append({"start": t, "end": t + 4.0, "text": " " + " ".join(said), "words": words})
            t += 4.5
            n += 1
        return out

    def close(self) -> None:
        pass


def lite_without_models(monkeypatch) -> None:
    """A lite install with no models: Whisper is :class:`StubWhisper`, no OCR engine, the CPU
    encoder, and no loudnorm on the highlights edit."""
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(analyze, "Transcriber", StubWhisper)
    monkeypatch.setattr(extras, "edition", lambda: "lite")
    monkeypatch.setattr(category, "ocr_installed", lambda: False)  # no HUD OCR engine in lite
    # loudnorm's limiter resamples with soxr, which some ffmpeg builds lack
    monkeypatch.setitem(edit.KINDS, "highlights", replace(edit.KINDS["highlights"], loudnorm=False))


def _run(capsys, argv: list[str]) -> dict:
    code = cli.main([*argv, "--json"])
    doc = envelope(capsys.readouterr().out)
    assert code == 0 and doc["ok"] is True, doc["error"]
    return doc


def _check(doc: dict, command: str, case: tuple) -> dict:
    """The detection in the envelope of ``command``, with what #899 asks of it for ``case``
    (builder, category, style, ambiguous, decided_by, warning codes)."""
    _build, cat, style, ambiguous, decided_by, warnings = case
    assert doc["command"] == command
    out = doc["outputs"]
    det = out["detection"]
    assert (det["category"], det["ambiguous"], det["decided_by"]) == (cat, ambiguous, decided_by)
    assert out["style"] == det["style"] == style
    assert det["fallback"] == (style if ambiguous else None)
    hl = json.loads(Path(out["highlights"]).read_text(encoding="utf-8"))
    assert hl["detection"] == det and hl["settings"]["style_requested"] == "auto"
    assert [w["code"] for w in doc["warnings"]] == warnings
    return det


def check_flagless(capsys, src: Path, case: tuple) -> None:
    """``analyze`` and then ``edit`` of ``src``, flagless: both detect the category, pick its
    style, report it in the envelope and warn of nothing the category does not explain."""
    analyzed = _run(capsys, ["analyze", str(src), "--out", "a", "--no-scenes", *NO_MODELS])
    det = _check(analyzed, "analyze", case)
    doc = _run(capsys, ["edit", str(src), "--out", "e", *NO_MODELS])
    assert _check(doc, "edit", case) == det  # edit detects what analyze did
    out = doc["outputs"]
    assert out["kind"] == "highlights" and Path(out["final"]).is_file()  # the default edit ran
