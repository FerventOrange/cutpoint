"""Packaging contract: ui and keyring are core dependencies, and the built wheel carries the
files the package reads at run time."""

import os
import re
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


def test_ui_and_keyring_are_core_dependencies_not_extras():
    """The browser workspace and the keyring are in every install: neither installer asks for
    them as extras (pip would warn about an unknown extra)."""
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    for script in ("install.sh", "install.ps1"):
        text = (_ROOT / script).read_text(encoding="utf-8")
        assert not re.search(r"cutpoint(\$extra|\$\{extra\})?\[[^\]]*\b(ui|keyring)\b", text)
        assert not re.search(r'extra\s*=\s*"\[[^"]*\b(ui|keyring)\b', text)
    core = {r.split(";")[0].split("[")[0].split(">")[0].split("=")[0].strip()
            for r in project["dependencies"]}  # fmt: skip
    assert {"fastapi", "uvicorn", "keyring", "telemetry-parser"} <= core


def _run_sh_version_check(version: str) -> subprocess.CompletedProcess[str]:
    """Run install.sh's ``check_version`` alone (the script runs ``main`` when sourced)."""
    text = (_ROOT / "install.sh").read_text(encoding="utf-8")
    die = re.search(r"^die\(\) \{.*?^\}", text, re.S | re.M)
    check = re.search(r"^check_version\(\) \{.*?^\}", text, re.S | re.M)
    assert die and check
    script = f"{die.group(0)}\n{check.group(0)}\nCUTPOINT_VERSION={version}\ncheck_version\n"
    return subprocess.run(
        ["sh", "-c", script], capture_output=True, text=True, timeout=30, check=False
    )  # fmt: skip


@pytest.mark.skipif(shutil.which("sh") is None, reason="sh not on PATH")
@pytest.mark.parametrize("version", ["0.3.0", "0.3.9", "v0.3.0", "0.1.0", "0.0.1", "0.3.0rc1"])
def test_install_sh_refuses_releases_before_0_4_0(version):
    done = _run_sh_version_check(version)
    assert done.returncode == 1 and done.stdout == ""
    assert done.stderr == (
        f"ERROR: Cutpoint {version} is older than 0.4.0, the oldest release this installer "
        "can install; install it with pip or uv directly, or use a newer version.\n"
    )


@pytest.mark.skipif(shutil.which("sh") is None, reason="sh not on PATH")
@pytest.mark.parametrize(
    "version",
    ["0.4.0", "0.4.0rc1", "0.4.0.dev1", "0.5.0", "0.10.0", "0.30.1", "1.0.0", "v0.4.0", "weird"],
)
def test_install_sh_allows_0_4_0_and_later_and_unparsable(version):
    done = _run_sh_version_check(version)
    assert (done.returncode, done.stdout, done.stderr) == (0, "", "")


def test_install_ps1_has_the_same_version_floor():
    text = (_ROOT / "install.ps1").read_text(encoding="utf-8")
    assert 'if ($S.Source -eq "pypi") { Test-Version $S.Version }' in text
    assert r"-match '^v?(\d+)\.(\d+)'" in text
    assert "[int]$Matches[1] -eq 0 -and [int]$Matches[2] -lt 4" in text
    sh = (_ROOT / "install.sh").read_text(encoding="utf-8")
    msg = "is older than 0.4.0, the oldest release this installer can install; install it with pip or uv directly, or use a newer version."
    assert msg in text and msg in sh


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
    assert {"events", "gpu"} <= set(meta.get_all("Provides-Extra"))
    assert {"ui", "keyring"}.isdisjoint(meta.get_all("Provides-Extra"))
    assert "ski" not in set(meta.get_all("Provides-Extra"))
