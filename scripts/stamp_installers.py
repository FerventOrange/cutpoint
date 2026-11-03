"""Stamp a release version into copies of install.sh and install.ps1 and write SHA256SUMS (#684).

    python scripts/stamp_installers.py VERSION OUT_DIR [--src DIR]

Copies install.sh and install.ps1 from --src (default: the repository root) into OUT_DIR, sets
CUTPOINT_VERSION_DEFAULT in each to VERSION (no leading "v"), fails unless the stamp matched
exactly one line per file, and writes OUT_DIR/SHA256SUMS in `sha256sum` format. The sources are
never modified. Used by .github/workflows/release.yml and tested in tests/scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_RE = re.compile(r"[0-9]+(\.[0-9]+)*([A-Za-z0-9.+-]*)")
# (file name, pattern whose groups 1, 3 and 4 surround the quoted version and keep the line end)
PATTERNS = {
    "install.sh": re.compile(r'^(CUTPOINT_VERSION_DEFAULT=")([^"\n]*)(")([ \t\r]*)$', re.M),
    "install.ps1": re.compile(
        r'^([ \t]*\$CUTPOINT_VERSION_DEFAULT = ")([^"\n]*)(")([ \t\r]*)$', re.M
    ),
}


def stamp_text(name: str, text: str, version: str) -> str:
    """Return *text* with the default version set; raise ValueError unless exactly one line matched."""
    if not VERSION_RE.fullmatch(version):
        raise ValueError(f"not a release version: {version!r} (give it without a leading v)")
    new, count = PATTERNS[name].subn(lambda m: m.group(1) + version + m.group(3) + m.group(4), text)
    if count != 1:
        raise ValueError(
            f"{name}: expected exactly one CUTPOINT_VERSION_DEFAULT line, found {count}"
        )
    return new


def stamp(version: str, out: Path, src: Path = ROOT) -> list[str]:
    """Write the stamped installers and SHA256SUMS into *out*; return the file names written."""
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for name in PATTERNS:
        data = stamp_text(name, (src / name).read_bytes().decode("utf-8"), version)
        raw = data.encode("utf-8")
        (out / name).write_bytes(raw)
        lines.append(f"{hashlib.sha256(raw).hexdigest()}  {name}\n")
    (out / "SHA256SUMS").write_text("".join(lines), encoding="utf-8", newline="\n")
    return [*PATTERNS, "SHA256SUMS"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("out", type=Path)
    ap.add_argument("--src", type=Path, default=ROOT)
    args = ap.parse_args(argv)
    try:
        names = stamp(args.version, args.out, args.src)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("stamped", args.version, "->", ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
