"""Check that the uv pin is the same in the Dockerfile, install.sh and install.ps1.

Dependabot bumps only the Dockerfile `FROM ghcr.io/astral-sh/uv:<ver>` line, so the two
installers must be bumped by hand, with the SHA256 of uv's installer script each of them checks
before running it (`UV_INSTALLER_SHA256`). Offline the script checks that both pins are there
and well formed. With `--urls` it also downloads the pinned installer scripts
(https://astral.sh/uv/<ver>/install.sh and install.ps1; text, never run) and compares their
SHA256 with the pins; on a mismatch it prints the hash to pin.

Stdlib only. Run from anywhere: `python scripts/check_uv_pin.py [--urls]`.
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCKERFILE_RE = re.compile(r"^FROM\s+ghcr\.io/astral-sh/uv:(\d+\.\d+\.\d+)(?:@\S+)?(?:\s|$)", re.M)
SH_RE = re.compile(r'^UV_VERSION="(\d+\.\d+\.\d+)"', re.M)
PS1_RE = re.compile(r'^\s*\$UV_VERSION\s*=\s*"(\d+\.\d+\.\d+)"', re.M)

FILES = (
    ("Dockerfile", DOCKERFILE_RE),
    ("install.sh", SH_RE),
    ("install.ps1", PS1_RE),
)

# The SHA256 each installer checks uv's installer script against, and the script it is for.
SH_SHA_RE = re.compile(r'^UV_INSTALLER_SHA256="([^"]*)"', re.M)
PS1_SHA_RE = re.compile(r'^\s*\$UV_INSTALLER_SHA256\s*=\s*"([^"]*)"', re.M)
SHA_FILES = (
    ("install.sh", SH_SHA_RE, "install.sh"),
    ("install.ps1", PS1_SHA_RE, "install.ps1"),
)
SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def parse_version(text: str, pattern: re.Pattern[str]) -> str | None:
    """Return the first uv version `pattern` matches in `text`, or None."""
    match = pattern.search(text)
    return match.group(1) if match else None


def find_problems(versions: dict[str, str | None]) -> list[str]:
    """Return messages for missing or differing versions (empty when all agree)."""
    problems = [f"{name}: no uv version found" for name, ver in versions.items() if ver is None]
    found = {ver for ver in versions.values() if ver is not None}
    if len(found) > 1:
        listing = ", ".join(f"{name}={ver}" for name, ver in versions.items())
        problems.append(
            f"uv pin mismatch ({listing}). Dependabot bumps only the Dockerfile: "
            "bump UV_VERSION in install.sh and install.ps1 to the same version."
        )
    return problems


def sha_problems(shas: dict[str, str | None]) -> list[str]:
    """Return messages for missing or malformed installer SHA256 pins (empty when all are fine)."""
    problems = []
    for name, sha in shas.items():
        if sha is None:
            problems.append(f"{name}: no UV_INSTALLER_SHA256 found")
        elif not SHA256_HEX.fullmatch(sha):
            problems.append(f"{name}: UV_INSTALLER_SHA256 is not 64 lowercase hex digits: {sha!r}")
    return problems


def installer_url(version: str, script: str) -> str:
    """The URL of uv's installer `script` (install.sh or install.ps1) for `version`."""
    return f"https://astral.sh/uv/{version}/{script}"


def installer_urls(version: str) -> list[str]:
    return [installer_url(version, "install.sh"), installer_url(version, "install.ps1")]


def fetch_sha256(
    url: str, attempts: int = 2, timeout: float = 20.0, retry_delay: float = 3.0
) -> str | None:
    """SHA256 of the body at `url` (redirects are followed by urllib), or None when it cannot be
    downloaded; retry once on any failure. The body is only hashed, never run."""
    for attempt in range(attempts):
        if attempt:
            time.sleep(retry_delay)
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "cutpoint-ci-check-uv-pin"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if 200 <= response.status < 300:
                    return hashlib.sha256(response.read()).hexdigest()
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
    return None


def url_problems(version: str, shas: dict[str, str | None], fetch=fetch_sha256) -> list[str]:
    """Download each pinned installer script and compare its SHA256 with the pin."""
    problems = []
    for name, _pattern, script in SHA_FILES:
        url = installer_url(version, script)
        got = fetch(url)
        if got is None:
            problems.append(
                f"installer URL does not resolve: {url} "
                "(is the pinned uv version released? Bump the pin in the Dockerfile, "
                "install.sh and install.ps1)"
            )
        elif got != shas.get(name):
            problems.append(
                f"{name}: {url} has SHA256 {got}, but UV_INSTALLER_SHA256 is {shas.get(name)}: "
                f"set UV_INSTALLER_SHA256 in {name} to {got} (after checking the uv release)"
            )
        else:
            print(f"ok: {url} matches UV_INSTALLER_SHA256 in {name}")
    return problems


def main(argv: list[str], root: Path = ROOT) -> int:
    texts = {name: (root / name).read_text(encoding="utf-8") for name, _ in FILES}
    versions = {name: parse_version(texts[name], pattern) for name, pattern in FILES}
    shas = {name: parse_version(texts[name], pattern) for name, pattern, _ in SHA_FILES}
    problems = find_problems(versions) + sha_problems(shas)
    pinned = next((v for v in versions.values() if v), None)
    if "--urls" in argv and not problems and pinned:
        problems += url_problems(pinned, shas)
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    if not problems:
        print(
            f"ok: uv pin {pinned} matches in Dockerfile, install.sh and install.ps1, "
            "with an installer SHA256 in each installer"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
