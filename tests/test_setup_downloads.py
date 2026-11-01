"""events/setup: verified downloads and the PANNs checkpoint mirrors (issues #30, #35)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from conftest import messages, write
from cutpoint.events import paths as ev_paths
from cutpoint.events import setup


def write_bytes(path: Path, data: bytes) -> str:
    return write(path, data).as_uri()


def _panns_env(monkeypatch, tmp_path: Path, good: bytes, mirrors: tuple[str, ...]) -> Path:
    """Point the PANNs download at ``mirrors`` serving ``good``; returns the checkpoint path."""
    ckpt = tmp_path / "models" / "panns" / "ckpt.pth"
    monkeypatch.setattr(setup, "PANNS_CKPT_MIRRORS", mirrors)
    monkeypatch.setattr(setup, "PANNS_CKPT_SIZE", len(good))
    monkeypatch.setattr(setup, "PANNS_CKPT_SHA256", hashlib.sha256(good).hexdigest())
    labels = write_bytes(tmp_path / "srv" / "labels.csv", b"a,b\n")
    monkeypatch.setattr(setup, "PANNS_LABELS_URL", labels)
    monkeypatch.setattr(ev_paths, "panns_ckpt", lambda: ckpt)
    monkeypatch.setattr(ev_paths, "panns_labels", lambda: tmp_path / "models" / "panns" / "l.csv")
    return ckpt


# -- #35: downloads are verified ------------------------------------------------------------------


def test_corrupt_cached_archive_is_downloaded_again(tmp_path):
    data = b"payload-bytes" * 50
    url = write_bytes(tmp_path / "srv" / "a.bin", data)
    dst = tmp_path / "out" / "a.bin"
    dst.parent.mkdir()
    dst.write_bytes(b"X" * len(data))  # same size, corrupt content
    setup.download(url, dst, len(data), hashlib.sha256(data).hexdigest())
    assert dst.read_bytes() == data


# -- #30: PANNs checkpoint mirrors ----------------------------------------------------------------


@pytest.mark.parametrize("first", ["missing", "wrong_sha", "slow"])
def test_install_panns_falls_back_to_next_mirror(tmp_path, monkeypatch, caplog, first):
    good = b"good-checkpoint" * 20
    srv = tmp_path / "srv"
    bad_url = (
        (srv / "gone.pth").as_uri()
        if first == "missing"
        else write_bytes(srv / "bad.pth", b"X" * len(good))  # right size, wrong content
    )
    if first == "slow":  # both mirrors serve the good file; the first is judged too slow
        bad_url = write_bytes(srv / "slow.pth", good)
        monkeypatch.setattr(setup, "SLOW_START_WINDOW", 0.0)
        monkeypatch.setattr(setup, "PANNS_MIN_RATE", float("inf"))
    good_url = write_bytes(srv / "good.pth", good)
    ckpt = _panns_env(monkeypatch, tmp_path, good, (bad_url, good_url))
    monkeypatch.delenv("CUTPOINT_PANNS_URL", raising=False)

    with caplog.at_level("INFO"):
        setup.install_panns()

    assert ckpt.read_bytes() == good
    # the user can see which mirror the checkpoint finally came from (#52); slow keeps its .part
    verb = "resuming" if first == "slow" else "downloading"
    assert f"{verb} {ckpt.name} from {good_url}" in caplog.text
    warnings = messages(caplog)
    assert len(warnings) == 1
    assert bad_url in warnings[0]
    assert good_url not in warnings[0]
    if first == "slow":  # the last mirror is exempt, so it still completes
        assert "too slow" in warnings[0]


@pytest.mark.parametrize(
    ("bad", "shown"),
    [
        ("http://[foo", "[foo"),
        ("https://user:TOKEN@[foo/x?t=SECRET", "[foo"),
        ("TOKEN@host/x?t=SECRET", "host"),  # no scheme: "unknown url type"
        ("file:https://user:TOKEN@host/x?t=SECRET", "host"),  # the file: branch of the redaction
    ],
    ids=["plain", "credentials", "no-scheme", "file-nested"],
)
def test_install_panns_malformed_env_url_is_a_failed_mirror(
    tmp_path, monkeypatch, caplog, bad, shown
):
    good = b"good-checkpoint" * 20
    srv = tmp_path / "srv"
    good_url = write_bytes(srv / "good.pth", good)
    ckpt = _panns_env(monkeypatch, tmp_path, good, (good_url,))
    monkeypatch.setenv("CUTPOINT_PANNS_URL", bad)

    with caplog.at_level("INFO"):
        setup.install_panns()

    assert ckpt.read_bytes() == good  # the next mirror served it
    warnings = messages(caplog)
    assert len(warnings) == 1
    assert shown in warnings[0]
    assert "token" not in caplog.text.lower()  # urllib lowercases the scheme in its text
    assert "secret" not in caplog.text.lower()
