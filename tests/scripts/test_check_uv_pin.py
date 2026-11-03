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
