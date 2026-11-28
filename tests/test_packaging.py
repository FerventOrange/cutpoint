"""Packaging contract: ui and keyring are core dependencies, and the built wheel carries the
files the package reads at run time."""

import hashlib
import os
import re
import shutil
import subprocess
import sys
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


def test_runtime_dependencies_have_an_upper_bound():
    """`uv tool install` does not read uv.lock: every runtime dependency stops below its next
    major (or is pinned), so a breaking release is never picked up (#1085, A809)."""
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    reqs = list(project["dependencies"])
    for items in project["optional-dependencies"].values():
        reqs += [r for r in items if not r.startswith("cutpoint[")]
    unbounded = [
        r
        for r in reqs
        if not re.search(r"(<|==)", r.split(";")[0]) and not r.startswith("cryptography")
    ]
    assert unbounded == []


def _run_sh_version_check(version: str) -> subprocess.CompletedProcess[str]:
    """Run install.sh's ``check_version`` alone (the script runs ``main`` when sourced)."""
    text = (_ROOT / "install.sh").read_text(encoding="utf-8")
    die = re.search(r"^die\(\) \{.*?^\}", text, re.S | re.M)
    check = re.search(r"^check_version\(\) \{.*?^\}", text, re.S | re.M)
    clamp = re.search(r"^clamp_num\(\) \{.*?^\}", text, re.S | re.M)
    assert die and check and clamp
    script = f"{die.group(0)}\n{clamp.group(0)}\n{check.group(0)}\nCUTPOINT_VERSION={version}\ncheck_version\n"
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


@pytest.mark.skipif(shutil.which("sh") is None, reason="sh not on PATH")
@pytest.mark.parametrize("version", ["0.99999999999.0", "99999999999999999999.1.0", "0.00004.0"])
def test_install_sh_leaves_an_oversized_version_part_to_uv(version):
    """A part too big for shell arithmetic is a huge number, not an `integer expression` error."""
    done = _run_sh_version_check(version)
    assert (done.returncode, done.stdout, done.stderr) == (0, "", "")


@pytest.mark.skipif(shutil.which("sh") is None, reason="sh not on PATH")
@pytest.mark.parametrize("version", ["0.00003.0", "000.3.1"])
def test_install_sh_refuses_an_old_version_written_with_leading_zeros(version):
    assert _run_sh_version_check(version).returncode == 1


@pytest.mark.skipif(shutil.which("sh") is None, reason="sh not on PATH")
@pytest.mark.parametrize(
    ("uv_version", "ok"),
    [("0.13.0", 0), ("0.12.9", 1), ("1.0.0", 0), ("0.99999999999.0", 0),
     ("99999999999999999999.0.0", 0), ("0.5", 1)],
)  # fmt: skip
def test_install_sh_uv_version_check_survives_oversized_parts(uv_version, ok):
    text = (_ROOT / "install.sh").read_text(encoding="utf-8")
    funcs = [
        re.search(rf"^{n}\(\) \{{.*?^\}}", text, re.S | re.M)
        for n in ("clamp_num", "uv_version_ok")
    ]
    assert all(funcs)
    script = (
        "UV_MIN_MAJOR=0\nUV_MIN_MINOR=13\n"
        + "\n".join(m.group(0) for m in funcs)
        + f"\nuv() {{ echo 'uv {uv_version}'; }}\nuv_version_ok\n"
    )
    done = subprocess.run(
        ["sh", "-c", script], capture_output=True, text=True, timeout=30, check=False
    )
    assert (done.returncode, done.stderr) == (ok, "")


@pytest.mark.skipif(shutil.which("pwsh") is None and shutil.which("powershell") is None,
                    reason="PowerShell not on PATH")  # fmt: skip
@pytest.mark.parametrize("version", ["0.99999999999.0", "99999999999999999999.1.0", "0.00004.0"])
def test_install_ps1_test_version_survives_an_oversized_part(version):
    """Windows CI runs this: an oversized part is left to uv, never a cast error."""
    text = (_ROOT / "install.ps1").read_text(encoding="utf-8")
    parts = [re.search(rf"^    function {n}\b.*?^    \}}", text, re.S | re.M)
             for n in ("ConvertTo-VersionPart", "Test-Version")]  # fmt: skip
    assert all(parts)
    script = (
        "$ErrorActionPreference = 'Stop'\nfunction Die($m) { throw $m }\n"
        + "\n".join(m.group(0) for m in parts)
        + f"\nTest-Version '{version}'\n'ok'\n"
    )
    exe = shutil.which("pwsh") or shutil.which("powershell")
    done = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-Command", script],
                          capture_output=True, text=True, timeout=60, check=False)  # fmt: skip
    assert (done.returncode, done.stdout.strip(), done.stderr) == (0, "ok", "")


@pytest.mark.skipif(shutil.which("pwsh") is None and shutil.which("powershell") is None,
                    reason="PowerShell not on PATH")  # fmt: skip
@pytest.mark.parametrize("version", ["0.3.0", "0.00003.0", "000.3.1"])
def test_install_ps1_test_version_still_refuses_an_old_version(version):
    text = (_ROOT / "install.ps1").read_text(encoding="utf-8")
    parts = [re.search(rf"^    function {n}\b.*?^    \}}", text, re.S | re.M)
             for n in ("ConvertTo-VersionPart", "Test-Version")]  # fmt: skip
    assert all(parts)
    script = (
        "$ErrorActionPreference = 'Stop'\nfunction Die($m) { throw $m }\n"
        + "\n".join(m.group(0) for m in parts)
        + f"\nTest-Version '{version}'\n'ok'\n"
    )
    exe = shutil.which("pwsh") or shutil.which("powershell")
    done = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-Command", script],
                          capture_output=True, text=True, timeout=60, check=False)  # fmt: skip
    assert done.returncode != 0
    assert "older than 0.4.0" in done.stderr


@pytest.mark.skipif(shutil.which("pwsh") is None and shutil.which("powershell") is None,
                    reason="PowerShell not on PATH")  # fmt: skip
@pytest.mark.parametrize(
    ("uv_version", "ok"),
    [("0.12.9", "False"), ("0.13.0", "True"), ("0.99999999999.0", "True")],
)
def test_install_ps1_uv_version_check_survives_oversized_parts(uv_version, ok):
    text = (_ROOT / "install.ps1").read_text(encoding="utf-8")
    parts = [re.search(rf"^    function {n}\b.*?^    \}}", text, re.S | re.M)
             for n in ("ConvertTo-VersionPart", "Test-UvVersionOk")]  # fmt: skip
    assert all(parts)
    script = (
        "$ErrorActionPreference = 'Stop'\n$UV_MIN_MAJOR = 0\n$UV_MIN_MINOR = 13\n"
        + f"function Get-UvVersionText {{ '{uv_version}' }}\n"
        + "\n".join(m.group(0) for m in parts)
        + "\nTest-UvVersionOk\n"
    )
    exe = shutil.which("pwsh") or shutil.which("powershell")
    done = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-Command", script],
                          capture_output=True, text=True, timeout=60, check=False)  # fmt: skip
    assert (done.returncode, done.stdout.strip(), done.stderr) == (0, ok, "")


def test_install_ps1_has_the_same_version_floor():
    text = (_ROOT / "install.ps1").read_text(encoding="utf-8")
    assert 'if ($S.Source -eq "pypi") { Test-Version $S.Version }' in text
    assert r"-match '^v?(\d+)\.(\d+)'" in text
    assert "(ConvertTo-VersionPart $Matches[1]) -eq 0" in text
    assert "(ConvertTo-VersionPart $Matches[2]) -lt 4" in text
    assert "[int]$Matches[2]" not in text
    sh = (_ROOT / "install.sh").read_text(encoding="utf-8")
    msg = "is older than 0.4.0, the oldest release this installer can install; install it with pip or uv directly, or use a newer version."
    assert msg in text and msg in sh


def _install_sh_functions() -> str:
    """install.sh without its last line (``main "$@"``), so sourcing it only defines things."""
    text = (_ROOT / "install.sh").read_text(encoding="utf-8")
    body, last = text.rstrip("\n").rsplit("\n", 1)
    assert last == 'main "$@"'
    return body + "\n"


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason="install.sh is POSIX-only (Git Bash mangles the sh -c script on Windows)",
)
def test_install_sh_refuses_a_uv_installer_with_the_wrong_sha256(tmp_path):
    """A downloaded uv installer whose SHA256 is not the pinned one is never run (#1085)."""
    ran = tmp_path / "ran"
    harness = (
        _install_sh_functions()
        + f"""
DRY_RUN=0; BIN_DIR={tmp_path}; TMPDIR={tmp_path}
curl() {{ printf 'touch {ran}\\n' > "$4"; }}
install_uv
"""
    )
    done = subprocess.run(
        ["sh", "-c", harness], capture_output=True, text=True, timeout=30, check=False
    )
    assert done.returncode == 1
    assert "has SHA256" in done.stderr and "not running it" in done.stderr
    assert not ran.exists()
    assert not list(tmp_path.glob("uv-install.*"))  # the download is removed


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason="install.sh is POSIX-only (Git Bash mangles the sh -c script on Windows)",
)
def test_install_sh_runs_a_uv_installer_with_the_pinned_sha256(tmp_path):
    """The downloaded uv installer runs when its SHA256 is the pinned one; the download goes."""
    ran = tmp_path / "ran"
    fake = f"touch {ran}\n"
    sha = hashlib.sha256(fake.encode()).hexdigest()
    harness = (
        _install_sh_functions()
        + f"""
DRY_RUN=0; BIN_DIR={tmp_path}; TMPDIR={tmp_path}; UV_INSTALLER_SHA256={sha}
curl() {{ printf 'touch {ran}\\n' > "$4"; }}
have() {{ [ "$1" = uv ] || command -v "$1" >/dev/null 2>&1; }}
install_uv
"""
    )
    done = subprocess.run(
        ["sh", "-c", harness], capture_output=True, text=True, timeout=30, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "SHA256 checked" in done.stdout
    assert ran.exists()
    assert not list(tmp_path.glob("uv-install.*"))


def test_installers_pin_the_uv_installer_sha256():
    sh = (_ROOT / "install.sh").read_text(encoding="utf-8")
    ps1 = (_ROOT / "install.ps1").read_text(encoding="utf-8")
    assert re.search(r'^UV_INSTALLER_SHA256="[0-9a-f]{64}"$', sh, re.M)
    assert re.search(r'^\s*\$UV_INSTALLER_SHA256 = "[0-9a-f]{64}"\s*$', ps1, re.M)
    # install.sh checks the hash before it runs the script; install.ps1 likewise
    assert sh.index('if [ "$got" != "$UV_INSTALLER_SHA256" ]') < sh.index('sh "$tmp"')
    assert ps1.index("if ($got -ne $UV_INSTALLER_SHA256)") < ps1.index("-File $tmp")


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason="install.sh is POSIX-only (Git Bash mangles the sh -c script on Windows)",
)
def test_install_sh_prints_the_ollama_linux_command_instead_of_running_it(tmp_path):
    """Ollama's Linux script has no published checksum: --ollama prints it, never runs it."""
    harness = (
        _install_sh_functions()
        + """
DRY_RUN=0; OS=Linux; OLLAMA_CHOICE=1
have() { [ "$1" != ollama ] && command -v "$1" >/dev/null 2>&1; }
ollama_up() { return 1; }
curl() { echo "curl ran" >&2; return 1; }
sudo() { echo "sudo ran" >&2; return 1; }
offer_ollama
echo "status=$OLLAMA_STATUS"
echo "hint=$OLLAMA_HINT"
"""
    )
    done = subprocess.run(
        ["sh", "-c", harness], capture_output=True, text=True, timeout=30, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "ran" not in done.stderr
    assert "status=not installed (Linux: install it yourself" in done.stdout
    assert "curl -fsSL https://ollama.com/install.sh | sh" in done.stdout


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
