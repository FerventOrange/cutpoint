"""``scripts/stamp_installers.py``: the release stamping of install.sh and install.ps1 (#684)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from conftest import load_script

pytestmark = pytest.mark.script

st = load_script("stamp_installers")
ROOT = Path(__file__).resolve().parents[2]


def test_stamps_real_installers_and_writes_checksums(tmp_path):
    out = tmp_path / "out"
    st.stamp("9.8.7", out)
    sh = (out / "install.sh").read_text(encoding="utf-8")
    ps = (out / "install.ps1").read_text(encoding="utf-8")
    assert re.findall(r'^CUTPOINT_VERSION_DEFAULT="(.*)"$', sh, re.M) == ["9.8.7"]
    assert re.findall(r'^\s+\$CUTPOINT_VERSION_DEFAULT = "(.*)"', ps, re.M) == ["9.8.7"]
    for name in ("install.sh", "install.ps1"):
        assert (ROOT / name).read_bytes().count(b"\r\n") == (out / name).read_bytes().count(b"\r\n")
    # Only the version line differs from the sources.
    for name in ("install.sh", "install.ps1"):
        a = (ROOT / name).read_text(encoding="utf-8").splitlines()
        b = (out / name).read_text(encoding="utf-8").splitlines()
        assert len(a) == len(b) and sum(x != y for x, y in zip(a, b, strict=True)) <= 1
    sums = (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert sums == [
        f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}"
        for n in ("install.sh", "install.ps1")
    ]


def test_fails_when_stamp_does_not_apply_once(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "install.sh").write_text('CUTPOINT_VERSION_DEFAULT="1.0.0"\n', encoding="utf-8")
    (src / "install.ps1").write_text("no version here\n", encoding="utf-8")
    assert st.main(["1.2.3", str(tmp_path / "o"), "--src", str(src)]) == 1
    (src / "install.ps1").write_text(
        '    $CUTPOINT_VERSION_DEFAULT = "1"\n    $CUTPOINT_VERSION_DEFAULT = "2"\n',
        encoding="utf-8",
    )
    assert st.main(["1.2.3", str(tmp_path / "o"), "--src", str(src)]) == 1


@pytest.mark.parametrize("bad", ["v1.2.3", "", '1"; rm -rf /', "latest"])
def test_rejects_bad_version(tmp_path, bad):
    with pytest.raises(ValueError):
        st.stamp_text("install.sh", 'CUTPOINT_VERSION_DEFAULT="1.0.0"\n', bad)
