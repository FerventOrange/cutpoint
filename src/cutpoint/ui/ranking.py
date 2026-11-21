"""Re-ranking a project with another style from ``cutpoint ui``, and its undo."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from cutpoint import jsonout
from cutpoint.assistant import ranking as ai_ranking
from cutpoint.pano import project as pano_project
from cutpoint.projectfiles import HIGHLIGHTS, RERANK_BACKUP
from cutpoint.segments import moment_id
from cutpoint.styles import AUTO, Style, get_style
from cutpoint.styles import style_name as fallback_style_name
from cutpoint.ui.analysis import ui_styles
from cutpoint.ui.common import BUSY_TEXT, bad, job_env, logger
from cutpoint.ui.files import _inside, _own, read_meta
from cutpoint.ui.projects import RERANKING, project_busy, run_state

CUDA_PROBE_S = 120.0  # seconds the CUDA check's child process may take (it imports torch)


class CudaProbe:
    """Whether torch sees a CUDA GPU, as :func:`cutpoint.analyze.plan_layers` asks it (the
    visual layer of a ranking's readiness, #902), from a child ``python`` once per UI session:
    the UI process never imports torch. False when torch is missing or broken, or the check
    fails or takes longer than :data:`CUDA_PROBE_S`."""

    CODE = "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"

    def __init__(self) -> None:
        self.lock = threading.Lock()  # one check at a time; later callers take its answer
        self.seen: bool | None = None

    def __call__(self) -> bool:
        with self.lock:
            if self.seen is None:
                try:
                    done = subprocess.run(
                        [sys.executable, "-c", self.CODE],
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        env=job_env(),
                        timeout=CUDA_PROBE_S,
                        check=False,
                    )
                    self.seen = done.returncode == 0
                except (OSError, subprocess.TimeoutExpired):
                    logger.debug("the CUDA check failed", exc_info=True)
                    self.seen = False
            return self.seen


cuda_probe = CudaProbe()


# --------------------------------------------------------------------------- ranking (#902)
PREVIEW_TOP = 10  # moments a ranking preview lists, before and after
STALE_TEXT = (
    "The moments changed since this preview (an analysis or another ranking ran); preview it again"
)
RERANK_BUSY_TEXT = "A ranking is being applied to this project; wait a moment"
UNDO_STALE_TEXT = "The moments changed since the page loaded (an analysis or another ranking ran)"
NO_UNDO_TEXT = "There is no earlier ranking of this analysis to go back to"


class PlainJSON(JSONResponse):
    """The JSON reply of plain data, the same bytes as FastAPI's own (:func:`jsonable_encoder`,
    then :class:`JSONResponse`) without first copying it value by value: a project's reply holds
    every moment, thousands for a long gameplay recording (A742). ``jsonable_encoder`` still
    takes a value JSON has no type for."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            default=jsonable_encoder,
        ).encode("utf-8")


class RankingRefused(Exception):
    """A ranking preview or apply refused with a ``code`` the page reads (``ApiError.code``):
    :meth:`response` is the JSON answer, ``extra`` keys beside ``detail`` and ``code``."""

    def __init__(self, detail: str, code: str, status: int = 409, **extra: Any) -> None:
        super().__init__(detail)
        self.detail, self.code, self.status, self.extra = detail, code, status, extra

    def response(self) -> JSONResponse:
        return JSONResponse(
            {"detail": self.detail, "code": self.code, **self.extra}, status_code=self.status
        )


def ranking_key(p: Path) -> str | None:
    """The ranking key (:func:`cutpoint.analyze.backup_digest`) of the project's
    ``highlights.json``, which a preview is applied against; None without a readable file of
    its own."""
    from cutpoint import analyze  # noqa: PLC0415

    if not _inside(p, HIGHLIGHTS):
        return None
    try:
        return analyze.backup_digest((p / HIGHLIGHTS).read_bytes())
    except OSError:
        return None


def _ranking_style(name: object) -> Style:
    """The library style (built in or saved) ``name`` a ranking takes, else 400 (auto too:
    a ranking needs a style, not a detection)."""
    if not isinstance(name, str) or not name or name == AUTO:
        raise bad("style must be a style from the library, not auto")
    if name not in ui_styles():
        raise bad(f"unknown style {name!r}")
    return get_style(name)


def _ranking_check(p: Path) -> None:
    """Refuse a ranking of ``p``: 409 for a 360 project; :class:`RankingRefused`
    ``folder_busy`` while a run holds it or a ranking is being applied to it."""
    if read_meta(p).get("kind") == "360" or (p / pano_project.PANO_FILE).is_file():
        raise bad("A 360 project is ranked by its analysis; analyze it again in the style", 409)
    if p in RERANKING:
        raise RankingRefused(RERANK_BUSY_TEXT, jsonout.FOLDER_BUSY)
    if project_busy(p, run_state(p, True)):
        raise RankingRefused(BUSY_TEXT, jsonout.FOLDER_BUSY)


def _ranking_load(p: Path) -> tuple[bytes, dict[str, Any]]:
    """The bytes and object of the project's ``highlights.json``; 409 without an analysis."""
    from cutpoint import rerank  # noqa: PLC0415

    _own(p, HIGHLIGHTS)  # never read through a link out of the project
    try:
        return rerank.load(p)
    except FileNotFoundError:
        raise bad("This project has no analysis to rank yet; analyze it first", 409) from None
    except ValueError:  # its message names the full path: the page gets file names only
        raise bad("highlights.json records no analysis; analyze the project again", 409) from None


def _ranking_rows(hl: dict[str, Any], other: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The first :data:`PREVIEW_TOP` moments of ``hl`` for a preview card: ``rank``, the page's
    ``id`` (:func:`moment_id`), ``cli_id``, the source's ``file`` name (never its path),
    ``start``, ``end``, ``score``, ``label`` and, with ``other``, ``was`` (the rank there)."""
    from cutpoint import rerank  # noqa: PLC0415

    out = []
    for r in rerank.rows(hl, PREVIEW_TOP, other):
        src = str(r.get("source") or "")
        row = {
            "rank": r["rank"],
            "id": moment_id(src, r["start"], r["end"]),
            "cli_id": r["id"],
            "file": Path(src).name,
            "start": r["start"],
            "end": r["end"],
            "score": r["score"],
            "label": r["label"],
        }
        if "was" in r:
            row["was"] = r["was"]
        out.append(row)
    return out


def _needs_analyze(style: Style, missing: list[dict[str, str | None]]) -> dict[str, Any]:
    """What keeps a ranking in ``style`` from the caches (``missing``,
    :meth:`cutpoint.rerank.Missing.record`): ``missing`` with file names only, whether
    analyzing again can help (``can_analyze``: not when only offline footage is missing) and
    the ``message`` the page shows."""
    rows = [
        {
            "feature": m.get("feature"),
            "detail": m.get("detail"),
            "file": Path(str(m["source"])).name if m.get("source") else None,
        }
        for m in missing
    ]
    shown = "; ".join(str(m.get("detail")) for m in missing[:5])
    more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
    offline = [m for m in missing if m.get("feature") == "offline"]
    if offline and len(offline) == len(missing):
        then = "Reconnect the footage and preview it again"
    elif offline:
        then = f"Reconnect the footage, then analyze again as {style.name}"
    else:
        then = f"Analyze again as {style.name} to rank it"
    return {
        "missing": rows,
        "can_analyze": len(offline) < len(missing),
        "message": f"Ranking with the {style.name} style needs analyze: {shown}{more}. {then}",
    }


def ranking_preview(p: Path, style_name: object) -> dict[str, Any]:
    """The project ``p`` re-ranked in the library style ``style_name`` from its caches
    (:func:`cutpoint.rerank.ranked`), for the page's before/after card; writes nothing.

    ``ready`` true: ``style``, ``from`` (the style it is ranked in now), ``base`` (the
    :func:`ranking_key` the apply route checks), the first :data:`PREVIEW_TOP` moments ``top``
    (with ``was``) and ``before``, ``new_ids_count``, ``dropped_count``, ``suggested_top``
    ``before`` and ``after``, ``notes`` and ``seconds``. ``ready`` false (analyze needed):
    ``style``, ``from``, ``base`` and :func:`_needs_analyze`'s keys.

    Raises:
        HTTPException: 400 for auto or a style not in the library, 409 for a 360 project or
            one without an analysis.
        RankingRefused: ``folder_busy`` while a run holds the project.
    """
    from cutpoint import analyze, rerank  # noqa: PLC0415
    from cutpoint.errors import NeedsAnalyzeError  # noqa: PLC0415

    style = _ranking_style(style_name)
    _ranking_check(p)
    raw, hl = _ranking_load(p)
    head = {
        "style": style.name,
        "from": fallback_style_name(hl),
        "base": analyze.backup_digest(raw),
    }
    ready = rerank.readiness(p, hl, style, cuda_probe)
    missing = [m.record() for m in ready.missing]
    after: dict[str, Any] = {}
    if not missing:
        try:
            after = rerank.ranked(p, hl, style, raw, check=False)
        except NeedsAnalyzeError as err:  # a source with nothing cached after all
            missing = err.missing
    if missing:
        return {"ready": False, **head, **_needs_analyze(style, missing)}
    new_ids, dropped = rerank.changes(hl, after)
    return {
        "ready": True,
        **head,
        "top": _ranking_rows(after, hl),
        "before": _ranking_rows(hl),
        "new_ids_count": len(new_ids),
        "dropped_count": len(dropped),
        "suggested_top": {"before": hl.get("suggested_top"), "after": after.get("suggested_top")},
        "notes": ready.notes,
    }


# the last ranking_undo_view per project: (highlights.json key, backup size and mtime) -> view
_UNDO_SEEN: dict[Path, tuple[tuple[str, int, int], dict[str, Any] | None]] = {}


def ranking_undo_view(p: Path, meta: dict[str, Any]) -> dict[str, Any] | None:
    """What the page's Undo ranking offers (:func:`cutpoint.rerank.undo_target`): ``action``
    ``undo`` or ``redo`` and ``style``, the style of the ranking it swaps in; None when there is
    none to swap in, and for 360 projects. Remembered per project until either file changes."""
    from cutpoint import analyze, rerank  # noqa: PLC0415

    if meta.get("kind") == "360" or (p / pano_project.PANO_FILE).is_file():
        return None
    if not _inside(p, RERANK_BACKUP) or not _inside(p, HIGHLIGHTS):
        return None
    try:
        st = (p / RERANK_BACKUP).stat()
        raw = (p / HIGHLIGHTS).read_bytes()
    except OSError:
        return None
    seen = (analyze.backup_digest(raw), st.st_size, st.st_mtime_ns)
    if (got := _UNDO_SEEN.get(p)) is not None and got[0] == seen:
        return got[1]
    try:
        hl = json.loads(raw.decode("utf-8-sig"))
    except ValueError:
        hl = None
    found = rerank.undo_target(p, hl) if isinstance(hl, dict) else None
    view = (
        None
        if found is None
        else {"action": "redo" if found[1] else "undo", "style": found[0].get("style")}
    )
    _UNDO_SEEN[p] = (seen, view)
    return view


def forget_undo_view(p: Path) -> None:
    """Drop the remembered Undo view of the project folder ``p`` (it was deleted)."""
    _UNDO_SEEN.pop(p, None)


def ranking_swap(p: Path, base: str) -> tuple[dict[str, Any], dict[str, Any], bool] | None:
    """Undo (or Redo) the last ranking change of ``p`` (:func:`cutpoint.rerank.swap`, its
    returns and errors); a swap forgets the page's remembered Undo view of ``p``."""
    from cutpoint import rerank  # noqa: PLC0415

    swapped = rerank.swap(p, base)
    if swapped is not None:
        _UNDO_SEEN.pop(p, None)
    return swapped


class RankingHook:
    """The chat's ``preview_ranking`` for project ``p`` (:class:`cutpoint.assistant.ranking.Hook`):
    the library styles with their saved description, the style the moments are ranked in now
    (``current``) and :func:`ranking_preview`, its refusals as ``RankingError``."""

    def __init__(self, p: Path, current: str) -> None:
        self.p, self._current = p, current

    def styles(self) -> list[tuple[str, str]]:
        return [
            (n, str((get_style(n).learned or {}).get("description") or "")) for n in ui_styles()
        ]

    def current(self) -> str:
        return self._current

    def preview(self, name: str) -> dict[str, Any]:
        try:
            return ranking_preview(self.p, name)
        except RankingRefused as err:
            raise ai_ranking.RankingError(err.detail, err.code == jsonout.FOLDER_BUSY) from None
        except HTTPException as err:
            raise ai_ranking.RankingError(str(err.detail)) from None
