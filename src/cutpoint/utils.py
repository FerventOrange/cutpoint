"""Small shared helpers."""

from __future__ import annotations

import fnmatch
import re
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePath

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm"}


def slugify(text: str, max_len: int | None = 40, default: str = "moment") -> str:
    """Lowercase ASCII slug safe for Windows filenames.

    Args:
        text: Any text; accents are folded to ASCII, other runs become ``-``.
        max_len: Truncate to this many characters (None = no limit).
        default: Returned when nothing is left.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if max_len is not None:
        slug = slug[:max_len].rstrip("-")
    return slug or default


def clock(t: float) -> str:
    """Format seconds as ``HH:MM:SS.s`` (same style as transcript.txt).

    Rounds to tenths first so carries propagate (``59.96`` -> ``00:01:00.0``);
    negative values get a leading ``-``.
    """
    tenths = round(abs(t) * 10)
    sign = "-" if t < 0 and tenths else ""
    h, rem = divmod(tenths, 36000)
    m, rem = divmod(rem, 600)
    return f"{sign}{h:02d}:{m:02d}:{rem // 10:02d}.{rem % 10}"


def natural_key(path: Path) -> list[object]:
    """Sort key so that ``2_x`` < ``10_x`` (case-insensitive)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def matches_glob(path: str | PurePath, patterns: Iterable[str]) -> bool:
    """True if ``path``'s file name or full path matches any glob.

    Case-insensitive; backslashes and forward slashes are treated alike, so
    ``"*/Captures/*"`` also matches Windows paths.
    """
    full = str(path).replace("\\", "/").rstrip("/").lower()
    name = full.rsplit("/", 1)[-1]
    for pat in patterns:
        pat = pat.lower().replace("\\", "/")
        if fnmatch.fnmatchcase(name, pat) or fnmatch.fnmatchcase(full, pat):
            return True
    return False


def find_videos(
    targets: Path | Sequence[Path], *, exclude: Sequence[str] = (), recursive: bool = False
) -> list[Path]:
    """Resolve files and/or folders into a de-duplicated list of video files.

    Args:
        targets: One path or several. Files are taken as given (any extension);
            folders contribute the videos directly inside them.
        exclude: Glob patterns matched against each file's name and its full
            path, e.g. ``"*DVR*"`` or ``"*/Captures/*"``.
        recursive: Also search sub-folders of folder inputs.

    Returns:
        Videos in input order; files inside one folder are naturally sorted.

    Raises:
        FileNotFoundError: If an input path does not exist.
    """
    if isinstance(targets, (str, Path)):
        targets = [Path(targets)]
    found: list[Path] = []
    for target in map(Path, targets):
        if target.is_file():
            batch = [target.resolve()]
        elif target.is_dir():
            walker = target.rglob("*") if recursive else target.glob("*")
            batch = sorted(
                (p.resolve() for p in walker if p.is_file() and p.suffix.lower() in VIDEO_EXTS),
                key=lambda p: [natural_key(Path(part)) for part in p.parts],
            )
        else:
            raise FileNotFoundError(f"Input not found: {target}")
        found.extend(p for p in batch if not matches_glob(p, exclude))
    return list(dict.fromkeys(found))
