"""``cutpoint extend``: which file a manifest entry is read from."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import media_info, remap, write
from cutpoint import extend, planning
from cutpoint.boundaries import SpeechSettings
from cutpoint.planning import Padding

FMT = {"width": 1920, "height": 1080, "fps": 30, "vertical": False}


@pytest.fixture
def reads(monkeypatch) -> list[Path]:
    """Stub probe and the encoder (no ffmpeg); returns the files clips were encoded from."""
    encoded: list[Path] = []

    def fake_export(src, info, start, end, out, *args):
        encoded.append(src)
        write(out, "encoded")

    monkeypatch.setattr(extend, "probe", media_info)
    monkeypatch.setattr(extend, "export_clip_checked", fake_export)
    monkeypatch.setattr(planning.sources, "readable_range", lambda src, t, d: (0.0, d))
    return encoded


def _clips_dir(tmp_path: Path, **entry) -> Path:
    clips = tmp_path / "proj" / "clips"
    write(clips / "001_a.mp4")
    e = {"file": "001_a.mp4", "start": 10.0, "end": 20.0} | entry
    write(clips / "manifest.json", json.dumps({"format": FMT, "clips": [e]}))
    return clips


def _run(clips: Path, **kw) -> dict:
    out = extend.extend_clips(clips, Padding(1.0, 1.0), speech=SpeechSettings(enabled=False), **kw)
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))["clips"][0]


@pytest.mark.parametrize("case", ["stale-read-from", "remap-wins", "fallback-to-read-from"])
def test_extend_reads_the_source_through_remap_and_read_from_only_as_fallback(
    home, tmp_path, monkeypatch, reads, case
):
    orig, copy, other = tmp_path / "old" / "a.mp4", tmp_path / "new" / "a.mp4", tmp_path / "b.mp4"
    if case == "stale-read-from":  # the copy it read was deleted; the original is back
        write(orig)
        entry = {"source": str(orig), "read_from": str(tmp_path / "gone.mp4")}
        want, recorded = orig, None
    elif case == "remap-wins":  # source_remap reaches the moved footage
        write(copy)
        write(other)
        remap(home, monkeypatch, orig.parent, copy.parent)
        entry = {"source": str(orig), "read_from": str(other)}
        want, recorded = copy, str(copy)
    else:  # neither the original nor a remap target exists; the copy it read does
        write(copy)
        entry = {"source": str(orig), "read_from": str(copy)}
        want, recorded = copy, str(copy)

    new = _run(_clips_dir(tmp_path, **entry))

    assert reads == [want]
    assert new.get("read_from") == recorded


def test_extend_reuse_copies_unchanged_clips_when_only_the_file_read_differs(tmp_path, reads):
    orig = write(tmp_path / "a.mp4")
    prior = tmp_path / "proj" / "earlier"
    write(prior / "001_a.mp4", "earlier")
    entry = {"file": "001_a.mp4", "source": str(orig), "start": 9.0, "end": 21.0}
    entry |= {"read_from": str(tmp_path / "copy.mp4"), "audio_fade": 0.25}
    write(prior / "manifest.json", json.dumps({"format": FMT, "clips": [entry]}))

    out = extend.extend_clips(
        _clips_dir(tmp_path, source=str(orig)),
        Padding(1.0, 1.0),
        speech=SpeechSettings(enabled=False),
        reuse_dir=prior,
    )

    assert reads == []
    assert (out / "001_a.mp4").read_text(encoding="utf-8") == "earlier"
