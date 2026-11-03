"""Check that the uv pin is the same in the Dockerfile, install.sh and install.ps1.

Dependabot bumps only the Dockerfile `FROM ghcr.io/astral-sh/uv:<ver>` line, so the two
installers must be bumped by hand. With `--urls` the script also checks that the pinned
installer URLs (https://astral.sh/uv/<ver>/install.sh and install.ps1) resolve.

Stdlib only. Run from anywhere: `python scripts/check_uv_pin.py [--urls]`.
"""

from __future__ import annotations

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


def installer_urls(version: str) -> list[str]:
    return [
        f"https://astral.sh/uv/{version}/install.sh",
        f"https://astral.sh/uv/{version}/install.ps1",
    ]


def url_resolves(
    url: str, attempts: int = 2, timeout: float = 20.0, retry_delay: float = 3.0
) -> bool:
    """HEAD the URL (redirects are followed by urllib); retry once on any failure."""
    for attempt in range(attempts):
        if attempt:
            time.sleep(retry_delay)
        try:
            request = urllib.request.Request(
                url, method="HEAD", headers={"User-Agent": "cutpoint-ci-check-uv-pin"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if 200 <= response.status < 300:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
    return False


def main(argv: list[str]) -> int:
    versions = {
        name: parse_version((ROOT / name).read_text(encoding="utf-8"), pattern)
        for name, pattern in FILES
    }
    problems = find_problems(versions)
    pinned = next((v for v in versions.values() if v), None)
    if "--urls" in argv and not problems and pinned:
        for url in installer_urls(pinned):
            if url_resolves(url):
                print(f"ok: {url}")
            else:
                problems.append(
                    f"installer URL does not resolve: {url} "
                    "(is the pinned uv version released? Bump the pin in the Dockerfile, "
                    "install.sh and install.ps1)"
                )
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    if not problems:
        print(f"ok: uv pin {pinned} matches in Dockerfile, install.sh and install.ps1")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
