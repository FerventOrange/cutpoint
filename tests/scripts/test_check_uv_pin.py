"""``scripts/check_uv_pin.py``: the uv pin must agree in the Dockerfile, install.sh and
install.ps1 (#613). Offline: only the parsing and mismatch logic; the URL check runs in CI."""

from __future__ import annotations

import pytest

from conftest import load_script

pytestmark = pytest.mark.script

check = load_script("check_uv_pin")

SHA = "@sha256:" + "a" * 64


def test_parses_dockerfile_with_and_without_digest():
    assert (
        check.parse_version(f"FROM ghcr.io/astral-sh/uv:0.13.6{SHA} AS uv\n", check.DOCKERFILE_RE)
        == "0.13.6"
    )
    assert (
        check.parse_version("FROM ghcr.io/astral-sh/uv:0.13.6 AS uv\n", check.DOCKERFILE_RE)
        == "0.13.6"
    )


def test_parses_installers():
    assert check.parse_version('x=1\nUV_VERSION="0.13.6"\n', check.SH_RE) == "0.13.6"
    assert (
        check.parse_version('if ($x) {\n    $UV_VERSION = "0.13.6"\n}\n', check.PS1_RE) == "0.13.6"
    )


@pytest.mark.parametrize("pattern", ["DOCKERFILE_RE", "SH_RE", "PS1_RE"])
def test_missing_version_is_none(pattern):
    assert check.parse_version("nothing here\n", getattr(check, pattern)) is None


def test_agreeing_versions_have_no_problems():
    assert (
        check.find_problems(
            {"Dockerfile": "0.13.6", "install.sh": "0.13.6", "install.ps1": "0.13.6"}
        )
        == []
    )


def test_mismatch_names_the_files_and_the_fix():
    problems = check.find_problems(
        {"Dockerfile": "0.14.0", "install.sh": "0.13.6", "install.ps1": "0.13.6"}
    )
    assert len(problems) == 1
    assert "Dockerfile=0.14.0" in problems[0]
    assert "install.sh" in problems[0] and "install.ps1" in problems[0]


def test_missing_version_is_a_problem():
    problems = check.find_problems(
        {"Dockerfile": "0.13.6", "install.sh": None, "install.ps1": "0.13.6"}
    )
    assert problems == ["install.sh: no uv version found"]


def test_installer_urls():
    assert check.installer_urls("0.13.6") == [
        "https://astral.sh/uv/0.13.6/install.sh",
        "https://astral.sh/uv/0.13.6/install.ps1",
    ]


def test_repository_pin_is_consistent():
    """The real files agree (offline half of the CI check)."""
    assert check.main([]) == 0


SHA_A = "a" * 64
SHA_B = "b" * 64


def test_parses_installer_shas():
    assert check.parse_version(f'x=1\nUV_INSTALLER_SHA256="{SHA_A}"\n', check.SH_SHA_RE) == SHA_A
    ps1 = f'if ($x) {{\n    $UV_INSTALLER_SHA256 = "{SHA_A}"\n}}\n'
    assert check.parse_version(ps1, check.PS1_SHA_RE) == SHA_A


def test_sha_problems():
    assert check.sha_problems({"install.sh": SHA_A, "install.ps1": SHA_B}) == []
    problems = check.sha_problems({"install.sh": None, "install.ps1": "A" * 64})
    assert problems[0] == "install.sh: no UV_INSTALLER_SHA256 found"
    assert "install.ps1" in problems[1] and "64 lowercase hex" in problems[1]


def test_url_problems_compare_the_downloaded_hash():
    hashes = {
        "https://astral.sh/uv/0.13.6/install.sh": SHA_A,
        "https://astral.sh/uv/0.13.6/install.ps1": SHA_B,
    }
    shas = {"install.sh": SHA_A, "install.ps1": SHA_A}
    problems = check.url_problems("0.13.6", shas, fetch=hashes.get)
    assert len(problems) == 1
    assert "install.ps1" in problems[0] and f"to {SHA_B}" in problems[0]
    assert (
        check.url_problems("0.13.6", {"install.sh": SHA_A, "install.ps1": SHA_B}, hashes.get) == []
    )


def test_url_problems_report_an_unreachable_url():
    problems = check.url_problems(
        "0.13.6", {"install.sh": SHA_A, "install.ps1": SHA_B}, lambda u: None
    )
    assert len(problems) == 2 and all("does not resolve" in p for p in problems)


def test_main_fails_without_an_installer_sha(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM ghcr.io/astral-sh/uv:0.13.6 AS uv\n")
    (tmp_path / "install.sh").write_text('UV_VERSION="0.13.6"\n')
    (tmp_path / "install.ps1").write_text(
        f'    $UV_VERSION = "0.13.6"\n    $UV_INSTALLER_SHA256 = "{SHA_B}"\n'
    )
    assert check.main([], root=tmp_path) == 1
    (tmp_path / "install.sh").write_text(f'UV_VERSION="0.13.6"\nUV_INSTALLER_SHA256="{SHA_A}"\n')
    assert check.main([], root=tmp_path) == 0
