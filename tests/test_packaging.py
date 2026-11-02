"""Packaging contract: the old extra names still install, and the built wheel carries the files
the package reads at run time."""

import os
import shutil
import subprocess
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
# the environment before conftest's isolated_env moves LOCALAPPDATA/HOME (and so uv's cache)
_REAL_ENV = {k: v for k, v in os.environ.items() if not k.startswith("CUTPOINT_")}


def test_old_extra_names_still_install_with_the_ui_in_core():
    """``cutpoint[ui]``, ``[keyring]`` and ``[ski]`` in old install commands keep working: the
    names stay as empty extras, their packages are core dependencies."""
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    for alias in ("ui", "keyring", "ski"):
        assert extras[alias] == []
    core = {r.split(";")[0].split("[")[0].split(">")[0].split("=")[0].strip()
            for r in project["dependencies"]}  # fmt: skip
    assert {"fastapi", "uvicorn", "keyring", "telemetry-parser"} <= core


@pytest.mark.slow
def test_the_wheel_ships_the_web_page_skill_guide_and_activity_profiles(tmp_path):
    """The UI, ``cutpoint skill``, ``cutpoint guide`` and the activity profiles read files
    from the installed package; hatchling's default include must keep putting them in the wheel.
    Builds offline (from the uv cache); skips when the build backend is not cached."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH")
    built = subprocess.run(
        [uv, "build", "--wheel", "--offline", "--out-dir", str(tmp_path), str(_ROOT)],
        capture_output=True, text=True, timeout=300, check=False, env=_REAL_ENV,
    )  # fmt: skip
    if built.returncode != 0:
        pytest.skip(f"cannot build the wheel offline: {built.stderr.strip()[-200:]}")
    (wheel,) = tmp_path.glob("cutpoint-*.whl")
    with zipfile.ZipFile(wheel) as z:
        names = set(z.namelist())
        meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        meta = Parser().parsestr(z.read(meta_name).decode("utf-8"))
    web = {p.name for p in (_ROOT / "src" / "cutpoint" / "web").iterdir() if p.is_file()}
    profiles = {
        p.name for p in (_ROOT / "src" / "cutpoint" / "presets" / "activities").glob("*.json")
    }
    assert {f"cutpoint/web/{n}" for n in web} <= names
    assert {f"cutpoint/presets/activities/{n}" for n in profiles} <= names
    assert {"cutpoint/skill/SKILL.md", "cutpoint/AGENTS.md"} <= names
    assert "base.json" in profiles and "index.html" in web  # the globs found the real folders
    assert {"ui", "keyring", "ski", "events", "gpu"} <= set(meta.get_all("Provides-Extra"))
