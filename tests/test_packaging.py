"""pyproject.toml packaging contract: core UI dependencies, extra aliases, default groups."""

import tomllib
from pathlib import Path

import pytest

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))


def _names(requirements: list[str]) -> set[str]:
    return {r.split(";")[0].split("[")[0].split(">")[0].split("=")[0].strip() for r in requirements}


def test_ui_keyring_ski_packages_are_core(pyproject):
    core = _names(pyproject["project"]["dependencies"])
    assert {"fastapi", "uvicorn", "keyring", "telemetry-parser"} <= core


def test_extras_and_aliases_are_declared(pyproject):
    extras = pyproject["project"]["optional-dependencies"]
    assert {"ui", "keyring", "ski", "events", "gpu", "all"} <= set(extras)
    for alias in ("ui", "keyring", "ski"):  # old names stay, with nothing to install
        assert extras[alias] == []
    assert extras["all"] == ["cutpoint[events,gpu]"]


def test_full_group_is_a_default_group(pyproject):
    assert pyproject["dependency-groups"]["full"] == ["cutpoint[all]"]
    assert pyproject["tool"]["uv"]["default-groups"] == ["dev", "full"]
    assert "dev" in pyproject["dependency-groups"]
