"""probe() memo: one ffprobe per unchanged file, re-probe on change, no cached failures (#372)."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import link_file
from cutpoint import ffmpeg_utils as fu
from cutpoint import reset_state, stage


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fake ffprobe_json; the list records the path of every call."""
    seen: list[str] = []

    def fake(path, **_kw):
        seen.append(str(path))
        return {
            "format": {"duration": "1"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 2,
                    "height": 2,
                    "avg_frame_rate": "30/1",
                }
            ],
        }

    monkeypatch.setattr(fu, "ffprobe_json", fake)
    return seen


@pytest.fixture
def media(tmp_path: Path) -> Path:
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    return f


def test_unchanged_file_is_probed_once(calls, media):
    first = fu.probe(media)
    assert fu.probe(media) is first
    assert fu.probe(str(media)) is first  # type: ignore[arg-type]
    assert len(calls) == 1


def test_rewritten_file_is_probed_again(calls, media):
    fu.probe(media)
    media.write_bytes(b"longer content")
    fu.probe(media)
    assert len(calls) == 2


def test_other_spelling_of_the_path_is_probed_again(calls, media):
    link = media.parent / "link.mp4"
    link_file(link, media)
    fu.probe(media)
    info = fu.probe(link)
    assert len(calls) == 2
    assert info.path == link


def test_staged_copy_appearing_changes_the_key(calls, media, monkeypatch):
    fu.probe(media)
    copy = media.parent / "staged.mp4"
    copy.write_bytes(b"xyz")
    monkeypatch.setattr(stage, "staged_path", lambda p: str(copy))
    info = fu.probe(media)
    assert len(calls) == 2
    assert calls[-1] == str(media)  # ffprobe still gets the caller's path
    assert info.path == media


def test_failed_probe_is_not_cached(media, monkeypatch):
    n = {"calls": 0}

    def boom(path, **_kw):
        n["calls"] += 1
        raise fu.FFmpegError("broken")

    monkeypatch.setattr(fu, "ffprobe_json", boom)
    for _ in range(2):
        with pytest.raises(fu.FFmpegError):
            fu.probe(media)
    assert n["calls"] == 2
    assert not fu._PROBED


def test_missing_file_is_not_cached(calls, tmp_path):
    gone = tmp_path / "missing.mp4"
    fu.probe(gone)
    fu.probe(gone)
    assert len(calls) == 2
    assert not fu._PROBED


def test_clear_probe_cache_forces_a_new_probe(calls, media):
    fu.probe(media)
    fu.clear_probe_cache()
    assert not fu._PROBED
    fu.probe(media)
    assert len(calls) == 2


def test_reset_state_clears_the_memo(calls, media):
    fu.probe(media)
    reset_state()
    assert not fu._PROBED


def test_cap_evicts_the_least_recently_used(calls, tmp_path, monkeypatch):
    monkeypatch.setattr(fu, "_PROBED_MAX", 2)
    files = []
    for i in range(3):
        f = tmp_path / f"{i}.mp4"
        f.write_bytes(b"x")
        files.append(f)
    fu.probe(files[0])
    fu.probe(files[1])
    fu.probe(files[0])  # touch: files[1] is now the oldest
    fu.probe(files[2])  # evicts files[1]
    assert len(fu._PROBED) == 2
    assert len(calls) == 3
    fu.probe(files[0])
    fu.probe(files[2])
    assert len(calls) == 3
    fu.probe(files[1])
    assert len(calls) == 4
