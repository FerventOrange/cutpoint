"""path_key folds case per folder: Windows always, elsewhere where the folder is
case-insensitive (macOS APFS/HFS+, exFAT or FAT drives on Linux, NTFS only when mounted
case-insensitive) (#247).

Linux CI has a case-sensitive file system, so a case-insensitive folder is simulated:
``config.folds_case`` says so for it and :func:`case_insensitive` makes ``Path`` find a name
in another case there, as macOS does.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from conftest import media_info, write
from cutpoint import clips, config, ffmpeg_utils, reset_state, sources, srt

WORDS = [{"start": 101.0, "end": 101.4, "word": "hello"}]


def _case_sensitive(folder: Path) -> bool:
    probe = write(folder / ".case-probe")
    try:
        return not (folder / ".CASE-PROBE").exists()
    finally:
        probe.unlink()


@pytest.fixture
def case_sensitive_fs(tmp_path):
    if sys.platform == "win32" or not _case_sensitive(tmp_path):
        pytest.skip("needs a case-sensitive file system")


# -- detection ----------------------------------------------------------------------------


def test_probe_finds_a_case_sensitive_folder(tmp_path, case_sensitive_fs):
    write(tmp_path / "d" / "Clip.mp4")
    assert config._probe_case(tmp_path / "d") is False
    write(tmp_path / "e" / "123.mp4")  # no letters in the entry: the folder's own name tells
    assert config._probe_case(tmp_path / "e") is False
    assert config.folds_case(tmp_path / "d") is False


def test_probe_finds_a_case_insensitive_folder(tmp_path, case_sensitive_fs):
    """A case-swapped spelling that is the same file (a hard link stands in for macOS)."""
    f = write(tmp_path / "d" / "clip.mp4")
    try:
        os.link(f, tmp_path / "d" / "CLIP.MP4")
    except OSError:
        pytest.skip("no hard links here")
    assert config._probe_case(tmp_path / "d") is True


def test_probe_without_letters_is_unknown_and_walks_up(tmp_path, monkeypatch):
    d = tmp_path / "123"
    write(d / "1.2")
    monkeypatch.setattr(sys, "platform", "linux")
    calls: list[Path] = []

    def probe(p: Path) -> bool | None:
        calls.append(p)
        return None if p.name == "123" else True

    assert config._probe_case(d.resolve()) is None
    monkeypatch.setattr(config, "_probe_case", probe)
    assert config.folds_case(d) is True
    assert calls == [d.resolve(), tmp_path.resolve()]


def test_folds_case_is_cached_per_folder_and_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    calls: list[Path] = []

    def probe(p: Path) -> bool:
        calls.append(p)
        return p.name == "ci"

    monkeypatch.setattr(config, "_probe_case", probe)
    ci, cs = (tmp_path / "ci", tmp_path / "cs")
    ci.mkdir()
    cs.mkdir()
    for _ in range(3):
        assert config.path_key("001_A.mp4", ci) == "001_a.mp4"
        assert config.path_key(str(cs / "001_A.mp4")) == str(cs / "001_A.mp4")
    assert calls == [ci.resolve(), cs.resolve()]
    reset_state()
    assert config.folds_case(ci) is True
    assert len(calls) == 3


def test_folds_case_cache_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(config, "_probe_case", lambda p: False)
    monkeypatch.setattr(config, "_FOLDS_CASE_MAX", 2)
    for n in "abc":
        (tmp_path / n).mkdir()
        config.folds_case(tmp_path / n)
    assert len(config._FOLDS_CASE) == 1


def test_a_missing_output_folder_uses_its_nearest_existing_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    calls: list[Path] = []
    monkeypatch.setattr(config, "_probe_case", lambda p: calls.append(p) or True)
    out = tmp_path / "project" / "clips"  # neither exists yet
    assert config.path_key("001_A.mp4", out) == "001_a.mp4"
    assert config.path_key(str(out / "001_A.mp4")) == str(out / "001_A.mp4").lower()
    assert calls == [tmp_path.resolve()]  # probed once, then cached


def test_windows_always_folds(monkeypatch):
    monkeypatch.setattr(config, "_probe_case", lambda p: pytest.fail("no probe expected"))
    monkeypatch.setattr(sys, "platform", "win32")
    assert config.folds_case("C:\\missing") is True
    assert config.path_key("D:\\Clip.MP4") == "d:\\clip.mp4"


@pytest.mark.skipif(os.name == "nt", reason="ntpath: D:\\ is absolute")
def test_a_windows_path_on_posix_is_not_looked_up_in_the_cwd(monkeypatch):
    monkeypatch.setattr(config, "_probe_case", lambda p: pytest.fail("no probe expected"))
    monkeypatch.setattr(sys, "platform", "linux")
    assert config.path_key("D:\\Clip.MP4") == "D:\\Clip.MP4"


def test_case_sensitive_folder_keeps_two_spellings_apart(tmp_path, case_sensitive_fs):
    write(tmp_path / "001_A.mp4", "A")
    write(tmp_path / "001_a.mp4", "a")
    assert config.path_key("001_A.mp4", tmp_path) != config.path_key("001_a.mp4", tmp_path)


# -- swaps and clips --force on a case-insensitive folder ---------------------------------


def test_swap_in_moves_a_case_variant_name_once(tmp_path, case_insensitive):
    """The old manifest names ``001_A.mp4``, the folder has ``001_a.mp4`` and the run writes
    ``001_a.mp4``: one file, moved to the backup once."""
    dst = tmp_path / "clips"
    write(dst / "001_a.mp4", "old")
    case_insensitive(dst)
    new = tmp_path / "staged"
    write(new / "001_a.mp4", "new")

    ffmpeg_utils.swap_in(new, dst, ["001_A.mp4"])

    assert sorted(p.name for p in tmp_path.iterdir()) == ["clips"]  # no backup or staging
    assert [p.name for p in dst.iterdir()] == ["001_a.mp4"]
    assert (dst / "001_a.mp4").read_text(encoding="utf-8") == "new"


def test_swap_in_without_folding_moves_it_twice_and_fails(tmp_path, case_insensitive, monkeypatch):
    """What #247 fixes, on the simulated folder: two spellings, one file, moved twice."""
    if sys.platform == "win32":
        pytest.skip("path_key always folds on Windows")
    dst = tmp_path / "clips"
    write(dst / "001_a.mp4", "old")
    case_insensitive(dst)
    monkeypatch.setattr(config, "folds_case", lambda f: False)
    new = tmp_path / "staged"
    write(new / "001_a.mp4", "new")

    with pytest.raises(FileNotFoundError):
        ffmpeg_utils.swap_in(new, dst, ["001_A.mp4"])
    # put back (under the listed spelling), so no data is lost, but the run failed
    assert [p.read_text(encoding="utf-8") for p in dst.iterdir()] == ["old"]


def _project(home: Path) -> tuple[Path, Path, Path]:
    project = home / "projects" / "p"
    src = write(project / "src.mp4", b"x")
    seg = write(
        project / "seg.json",
        json.dumps([{"source": str(src), "start": 100.0, "end": 104.0, "label": "a",
                     "speech_aware": False, "smart_tail": False}]),
    )  # fmt: skip
    return project, seg, project / "clips"


def _fake_export(src, info, start, end, out, *_a, **_kw) -> None:
    out.write_bytes(b"new")


@pytest.fixture
def exported(home, monkeypatch):
    """A project whose first ``clips`` run wrote ``001_a.mp4``, ``001_a.srt`` and a
    manifest.json that names them ``001_A.mp4`` / ``001_A.srt`` (edited by hand)."""
    project, seg, out = _project(home)
    monkeypatch.setattr(clips, "probe", lambda p: media_info(p, 200.0))
    monkeypatch.setattr(clips, "export_clip_checked", _fake_export)
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: WORDS)
    clips.export_clips(project, segments=seg)
    write(out / "001_a.mp4", b"old")
    data = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    data["clips"][0] |= {"file": "001_A.mp4", "srt": "001_A.srt"}
    write(out / "manifest.json", json.dumps(data))
    return project, seg, out


def test_clips_force_replaces_a_clip_listed_in_another_case(exported, case_insensitive):
    project, seg, out = exported
    case_insensitive(out)

    clips.export_clips(project, segments=seg, force=True)

    assert sorted(p.name for p in out.iterdir()) == ["001_a.mp4", "001_a.srt", "manifest.json"]
    assert (out / "001_a.mp4").read_bytes() == b"new"
    assert sorted(p.name for p in out.parent.iterdir() if p.name.startswith(".")) == []


def test_clips_force_on_a_case_sensitive_folder_keeps_the_other_spelling(
    exported, case_sensitive_fs
):
    """On Linux ``001_A.mp4`` (listed) and ``001_a.mp4`` (on disk) are two names: the one on
    disk is not the listed clip, so the run stops before replacing it."""
    project, seg, out = exported
    with pytest.raises(FileExistsError, match=r"001_a\.mp4"):
        clips.export_clips(project, segments=seg, force=True)
    assert (out / "001_a.mp4").read_bytes() == b"old"


def test_clips_force_pairs_a_listed_srt_renamed_in_case(home, monkeypatch, case_insensitive):
    """The #256 pairing pass, now also outside Windows: review renamed 001_a.srt to 001_A.srt
    and the run replaces 001_a.mp4 without a transcript, so the old .srt goes too."""
    project, seg, out = _project(home)
    monkeypatch.setattr(clips, "probe", lambda p: media_info(p, 200.0))
    monkeypatch.setattr(clips, "export_clip_checked", _fake_export)
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: WORDS)
    clips.export_clips(project, segments=seg)
    (out / "001_a.srt").rename(out / "001_A.srt")
    case_insensitive(out)
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)

    clips.export_clips(project, segments=seg, force=True)

    assert sorted(p.name for p in out.iterdir()) == ["001_a.mp4", "manifest.json"]


def test_protected_srt_is_matched_in_the_clips_folder_not_the_staging_one(
    tmp_path, monkeypatch, case_insensitive
):
    out, staging = tmp_path / "clips", tmp_path / "staging"
    out.mkdir()
    staging.mkdir()
    write(out / "001_a.srt", "mine")
    case_insensitive(out)
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: WORDS)
    subs = srt.ClipSubtitles(srt.unlisted_srts(out, []), None, out)
    assert subs.write(staging / "001_A.mp4", "src.mp4", 100.0, 104.0) is None
    assert not (staging / "001_A.srt").exists()


def test_folds_case_remembers_a_missing_folder_until_reset(tmp_path, monkeypatch):
    """F1: a missing (or offline) folder is walked up once, not on every call."""
    monkeypatch.setattr(sys, "platform", "linux")
    stats: list[str] = []
    real_stat = Path.stat
    monkeypatch.setattr(
        Path, "stat", lambda self, **k: stats.append(str(self)) or real_stat(self, **k)
    )
    monkeypatch.setattr(config, "_probe_case", lambda p: True)
    out = tmp_path / "offline" / "clips"
    assert config.path_key(str(out / "001_A.mp4")) == str(out / "001_a.mp4").lower()
    assert stats
    stats.clear()
    for _ in range(3):
        assert config.path_key(str(out / "001_B.mp4")) == str(out / "001_b.mp4").lower()
        assert config.folds_case(out) is True
    assert stats == []  # no I/O on a repeat call
    reset_state()
    config.folds_case(out)
    assert stats
