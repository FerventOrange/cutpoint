"""Style choice and the command lines of the jobs ``cutpoint ui`` starts."""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from cutpoint import cache, progress, selfcheck, transcribe
from cutpoint.assistant import providers as ai_providers
from cutpoint.config import (
    COLOR_MODES,
    COMPRESS_PEAKS,
    analysis_skip,
    assistant_settings,
    get_config,
    projects_dir,
    vision_assistant_settings,
    vision_detect_all_default,
)
from cutpoint.pano import project as pano_project
from cutpoint.pano import stitch as pano_stitch
from cutpoint.projectfiles import ACTIVITY, CRASHES
from cutpoint.segments import MAX_SEGMENTS
from cutpoint.styles import (
    ACTION_CAM,
    AUTO,
    HYPE,
    Style,
    auto_fallback,
    get_style,
    learned_styles,
    recorded_style,
    style_names,
)
from cutpoint.ui.common import (
    ANALYSIS_FILE,
    CLIP_ORDERS,
    CLIP_SIZE,
    CROP_BOX,
    MAX_TIMING_S,
    MAX_TOTAL_S,
    META_FILE,
    SAFE_NAME,
    STYLE_FOR,
    UI_DIR,
    bad,
    cutpoint_cmd,
    logger,
    number_in,
)
from cutpoint.ui.files import (
    _highlights,
    _inside,
    _load,
    _own,
    _ui_folder,
    _values,
    picked_filters,
    picked_inputs,
    read_meta,
)
from cutpoint.ui.jobs import Job
from cutpoint.utils import slugify


def new_project(name: str, source: str, style: str, **extra: Any) -> dict[str, str]:
    """Create the project folder and its ``ui/meta.json`` (``extra``: e.g. ``kind`` and
    ``clips`` of a 360 project; ``inputs``, ``recursive`` and ``exclude`` of a project made from
    several inputs or with those ``analyze`` options)."""
    root = projects_dir()
    with cache.created_dirs() as made:  # a failed request leaves no empty folder behind
        cache.make_dirs(root, made)
        base, n = slugify(name, 60, "project"), 1
        while True:
            pid = base if n == 1 else f"{base}-{n}"
            try:
                (root / pid).mkdir()
                break
            except FileExistsError:
                n += 1
        made.insert(0, root / pid)
        (root / pid / UI_DIR).mkdir()
        made.insert(0, root / pid / UI_DIR)
        cache.save_json(
            root / pid / UI_DIR / "meta.json",
            {
                "name": name,
                "source": source,
                "style": style,
                "created": datetime.now().isoformat(),
                **extra,
            },
        )
    return {"id": pid, "name": name}


def set_style(p: Path, style: str, detection: dict[str, Any] | None = None) -> Callable[[], None]:
    """Make ``style`` the project's chosen style in ``ui/meta.json`` (created with only it for a
    project made on the command line), with ``detection`` (when given) as the copy of the last
    detection the header shows after the user left Auto, and :data:`STYLE_FOR`, the re-rank the
    choice was made after (:func:`_rerank_mark` of the current highlights.json; left out when it
    is not re-ranked). Returns what puts the file back."""
    meta_file = _own(p, META_FILE)
    try:
        before: bytes | None = meta_file.read_bytes()
    except FileNotFoundError:
        before = None
    meta = {**read_meta(p), "style": style}
    meta.pop(STYLE_FOR, None)  # chosen after no re-rank: any later one wins
    if (mark := _rerank_mark(_highlights(p))) is not None:
        meta[STYLE_FOR] = mark
    if detection is not None:
        meta["detection"] = detection
    cache.save_json(meta_file, meta)

    def back() -> None:
        if before is None:
            meta_file.unlink(missing_ok=True)
        else:
            cache.save_bytes(meta_file, before)

    return back


def ui_styles() -> list[str]:
    """Styles for standard video, the telemetry-driven action-cam styles included (#633).

    A learned style that fails to load is left out with a warning, so one bad file does not
    break the UI.
    """
    names = []
    for n in style_names():
        try:
            get_style(n)
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("skipping learned style %s: %s", learned_styles().get(n, n), e)
            continue
        names.append(n)
    return names


def _gaming_style(name: object, ps: ProjectStyle) -> bool:
    """``name`` is a ``gaming`` style (or one based on it): config.json's ``color`` never grades
    it, so the page's Color starts at Off for it. ``ps``: the project's style, whose recorded
    base stands in for its deleted style (:meth:`ProjectStyle.style`)."""
    s = ps.style(name)
    return s is not None and s.base == "gaming"


def _action_cam_style(name: str) -> bool:
    """``name`` is a telemetry-driven action-cam style (``kind`` action-cam).

    Such a style is kept as given for 360 sources.
    """
    try:
        return get_style(name).kind == ACTION_CAM
    except (ValueError, TypeError, AttributeError):
        return False


# --------------------------------------------------------------------------- app
def _flag(body: dict[str, Any], key: str) -> bool:
    v = body.get(key, False)
    if not isinstance(v, bool):
        raise bad(f"{key} must be true or false")
    return v


def _compress_peaks(body: dict[str, Any]) -> list[str]:
    """``--compress-peaks`` with the page's choice, else ``compress_peaks`` in config.json;
    explicit, so the job does what the page showed even if config.json changes before it runs."""
    v = body.get("compress_peaks", get_config().compress_peaks)
    if v not in COMPRESS_PEAKS:
        raise bad(f"compress_peaks must be one of {', '.join(COMPRESS_PEAKS)}")
    return ["--compress-peaks", v]


def _color_state(raw: Any) -> dict[str, Any] | None:
    """The page's Color as ``{mode, strength}`` (strength 0-1) from an assistant request, or
    None when it did not send one (or sent nonsense)."""
    if not isinstance(raw, dict) or raw.get("mode") not in COLOR_MODES:
        return None
    strength = number_in(raw.get("strength"), 0, 1)
    if strength is None:
        return None
    return {"mode": raw["mode"], "strength": strength}


def _color_after(target: dict[str, Any], now: dict[str, Any] | None) -> dict[str, Any]:
    """The Color a proposal's color op gives: its ``target`` ``{mode, strength}``, where a
    strength it leaves out (None) stays what the page has ``now``."""
    strength = target.get("strength")
    if strength is None:
        strength = now["strength"] if now else 1.0
    return {"mode": target["mode"], "strength": strength}


def _color_args(body: dict[str, Any]) -> list[str]:
    """``--color`` and ``--color-strength`` with the page's choice. Not sent: no flag, so the CLI
    applies ``color`` in config.json itself (never to gaming footage, which an explicit flag
    would grade). The page has no LUT, but a ``lut`` with a grade is refused as the CLI refuses
    ``--color`` with ``--lut``."""
    color = body.get("color")
    strength = body.get("color_strength")
    if color is not None and (not isinstance(color, str) or color not in COLOR_MODES):
        raise bad(f"color must be one of {', '.join(COLOR_MODES)}")
    if body.get("lut") is not None and (color not in (None, "off") or strength is not None):
        raise bad("color cannot be combined with lut; use one")
    if strength is None:
        return [] if color is None else ["--color", color]
    value = number_in(strength, 0, 1)
    if value is None:
        raise bad("color_strength must be a number from 0 to 1")
    if color not in ("auto", "underwater"):
        raise bad("color_strength needs color auto or underwater")
    return ["--color", color, "--color-strength", f"{value:g}"]


def _rerank_mark(hl: dict[str, Any] | None) -> str | None:
    """What tells the re-rank of ``hl`` from any other: its ``settings.rerank_backup`` (the key of
    the file it was re-ranked from, #902); None when ``hl`` is not re-ranked (or records none)."""
    st = (hl or {}).get("settings") or {}
    mark = st.get("rerank_backup")
    return mark if isinstance(st.get("reranked"), dict) and isinstance(mark, str) else None


@dataclass(frozen=True)
class ProjectStyle:
    """A project's style, resolved once from its ``ui/meta.json`` (``meta``) and its
    ``highlights.json`` (``hl``, None before the first analysis, perhaps an older file without
    the newer keys): each style a part of the UI needs, and the style library lookups that
    stand in a style's recorded built-in base for a deleted style (``style_base``).

    The styles differ on purpose. ``chosen`` is what the next analysis runs on, so an ``auto``
    or an action-cam request that fell back stays as asked and the detection (or the telemetry
    check) repeats; ``resolved`` is what the current analysis ran as (the header's style and the
    page's Color); ``ran`` is the recorded style itself, else the one in ``ui/meta.json`` (the
    Analyze options and presets judge what the analysis did, not a style that may be
    mid-switch).
    """

    inputs: Any
    """The inputs the user picked, else (made on the command line) ``settings.inputs``."""
    chosen: Any
    """The style a project's analysis runs on: :attr:`reranked`, else the style the user chose,
    else ``auto`` for an analysis run with ``--style auto``, else the style asked for when an
    action-cam one fell back for footage without telemetry (#939, ``settings.style_fallback``),
    else the recorded one. A re-rank on the command line leaves ``ui/meta.json`` as it was."""
    resolved: Any
    """The style the analysis ran as: :attr:`reranked`, else the project's chosen style, or for
    ``auto`` (or none) the style the footage was detected as (recorded), else the chosen one.
    An action-cam style that fell back for footage without telemetry resolves to the style
    that ran."""
    reranked: Any
    """The style a re-ranked analysis (``settings.reranked``, #902) runs on, else None. None too
    when the chosen style was picked after that re-rank (:data:`STYLE_FOR` names it): the
    user's last pick wins, even when its analysis did not finish, until the next re-rank."""
    recorded: Any
    """The style ``highlights.json`` records (``style``), as it is (None without one)."""
    ran: Any
    """:attr:`recorded`, else the style the user chose (as it is)."""
    hl: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    @classmethod
    def of(cls, meta: dict[str, Any], hl: dict[str, Any] | None) -> ProjectStyle:
        """The style of the project with ``meta`` and ``hl``."""
        st = (hl or {}).get("settings") or {}
        recorded = (hl or {}).get("style")
        picked = meta.get("style")
        reranked = None
        if isinstance(st.get("reranked"), dict):
            mark = _rerank_mark(hl)
            if not (picked and mark is not None and meta.get(STYLE_FOR) == mark):
                reranked = recorded or None
        fell_back = isinstance(st.get("style_fallback"), dict)
        requested = st.get("style_requested")
        if reranked:
            chosen = reranked
        elif picked:
            chosen = picked
        elif requested == AUTO or (fell_back and requested):
            chosen = requested
        else:
            chosen = recorded
        if reranked:
            resolved = reranked
        elif not picked or picked == AUTO or (fell_back and requested == picked):
            resolved = recorded or picked
        else:
            resolved = picked
        return cls(
            inputs=picked_inputs(meta) or st.get("inputs"),
            chosen=chosen,
            resolved=resolved,
            reranked=reranked,
            recorded=recorded,
            ran=recorded or picked,
            hl=hl,
        )

    def style(self, name: object) -> Style | None:
        """The style ``name``, else None. The recorded style, gone from the library (deleted), is
        its recorded built-in base; None when the file records none (an older file): the page
        then shows the style's options as unknown."""
        try:
            return get_style(str(name)) if name else None
        except (ValueError, TypeError, AttributeError):
            if isinstance(self.hl, dict) and name and name == self.hl.get("style"):
                try:
                    return recorded_style(self.hl, logging.DEBUG)
                except ValueError:
                    pass
            return None

    def library(self, style: object) -> object:
        """``style`` as a job may pass it to ``--style``: the recorded style gone from the
        library (deleted) is the built-in base the file records for it (the job then records
        the base); a bad request (400) when the file records none (an older file), or for any
        other deleted style. Auto, a style in the library and a style whose file is there but
        broken (the CLI says why) stay as they are."""
        if not isinstance(style, str) or not style or style == AUTO:
            return style
        try:
            get_style(style)
        except (ValueError, TypeError, AttributeError):
            if style in learned_styles():
                return style
            try:
                if not isinstance(self.hl, dict) or self.hl.get("style") != style:
                    raise ValueError(style)
                return recorded_style(self.hl).name
            except ValueError:
                raise bad(
                    f"The style {style!r} of this analysis is not in the style library "
                    "(deleted?); pick another style"
                ) from None
        return style

    @property
    def ran_deleted(self) -> bool:
        """:attr:`ran` is a style gone from the library (deleted): not auto, not built in, no
        file."""
        name = self.ran
        if not isinstance(name, str) or not name or name == AUTO:
            return False
        try:
            get_style(name)
        except (ValueError, TypeError, AttributeError):
            return name not in learned_styles()
        return False


def _no_options(values: Any, what: str) -> None:
    """A bad request when one of ``values`` (paths from ``ui/meta.json`` or ``highlights.json``,
    which may have been edited by hand) would reach a job's command line as an option."""
    for v in values:
        if str(v).startswith("-"):
            raise bad(f"{what} {str(v)!r} in the project's files is not a path")


def _keep_style(
    p: Path,
    style: str,
    undo: list[Callable[[], None]],
    detection: dict[str, Any] | None = None,
) -> None:
    """In a job's ``prepare``: make ``style`` the project's chosen style (``ui/meta.json``, as
    :func:`set_style`, with ``detection``), and have ``undo`` put it back, with the ui folder
    this created."""
    made: list[Path] = []
    _ui_folder(p, made=made)  # a command-line project has no ui folder (nor meta.json) yet
    back = set_style(p, style, detection)

    def put_back() -> None:
        try:
            back()
        finally:
            for folder in made:  # the ui folder this created, if still empty
                with contextlib.suppress(OSError):
                    folder.rmdir()

    undo.append(put_back)


def detected_view(meta: dict[str, Any], hl: dict[str, Any] | None) -> dict[str, Any] | None:
    """What the header's style button shows (#897): the style the user chose (``chosen``,
    ``auto`` for Auto) and the footage category the analysis detected, with its reason, whether
    it was unsure (``ambiguous``, then ``fallback_label`` names the menu choice whose style it used,
    ``used_style``) and the sources of another category (:func:`cutpoint.category.disagreeing`).
    The detection is the current analysis's (``--style auto``), else the copy saved when the
    user left Auto, else the one a re-ranked analysis kept (``stale`` both); ``category`` is
    null without any. ``choices`` are the
    categories with their style (General is a detection label only, not a choice), ``main`` false for Action cam unless the footage has camera
    telemetry or the style is already action-cam (the page lists it under Other styles then).
    None for 360 projects and before the first analysis."""
    if meta.get("kind") == "360" or not hl:
        return None
    from cutpoint import category  # noqa: PLC0415
    from cutpoint.styles import auto_fallback_category, resolve_auto  # noqa: PLC0415

    chosen = ProjectStyle.of(meta, hl).chosen or None
    st = hl.get("settings") or {}
    det = hl.get("detection") if chosen == AUTO and st.get("style_requested") == AUTO else None
    stale = not isinstance(det, dict)
    if stale:
        det = meta.get("detection")
        if not isinstance(det, dict) and isinstance(st.get("reranked"), dict):
            det = hl.get("detection")  # re-ranked on the command line: no copy in meta.json
    if not isinstance(det, dict) or det.get("category") not in category.CATEGORIES:
        det = None
    srcs = [s for s in (det or {}).get("sources") or [] if isinstance(s, dict)]
    camera = (det or {}).get("category") == category.ACTION_CAM or any(
        s.get("decided_by") == "telemetry" for s in srcs
    )
    choices = [
        {
            "category": c,
            "label": category.label(c),
            "style": resolve_auto(c, "hype"),
            "main": c != category.ACTION_CAM or camera or ACTION_CAM in (chosen, hl.get("style")),
        }
        for c in (category.GAMEPLAY, category.ACTION, category.ACTION_CAM, category.TALK)
    ]
    view: dict[str, Any] = {
        "auto": chosen == AUTO,
        "chosen": chosen,
        "style": hl.get("style"),
        "choices": choices,
        "category": None,
        "label": None,
        "reason": None,
        "ambiguous": False,
        "fallback_label": None,
        "used_style": None,
        "suggest": None,
        "stale": False,
        "sources": [],
    }
    if det is None:
        return view
    cat = det["category"]
    ambiguous = det.get("ambiguous") is True

    def text(v: object) -> str | None:
        return v if isinstance(v, str) and v else None

    def fallback_label(c: str) -> str:
        # the menu's name for the style the fallback ran (General's style is Action's)
        fb = auto_fallback_category(c)
        style = resolve_auto(fb, "hype")
        return next((ch["label"] for ch in choices if ch["style"] == style), category.label(fb))

    return {
        **view,
        "category": cat,
        "label": category.label(cat),
        "reason": text(det.get("reason")),
        "ambiguous": ambiguous,
        "fallback_label": fallback_label(cat) if ambiguous else None,
        "used_style": text(det.get("style")),
        "suggest": text(det.get("suggest")),
        "stale": stale,
        "sources": [
            {
                "name": Path(str(s.get("path") or "")).name,
                "path": s.get("path"),
                "category": s.get("category"),
                "label": category.label(s["category"])
                if s.get("category") in category.CATEGORIES
                else None,
                "reason": text(s.get("reason")),
                "ambiguous": s.get("ambiguous") is True,
            }
            for s in category.disagreeing(det)
            if isinstance(s, dict)
        ],
    }


def _subscription_roles(args: list[str], style: object) -> bool:
    """Whether an analyze/edit job with ``args`` on ``style`` has a role on a subscription
    provider (#832): Vision (rating on) or the AI Chat model of the ``dialogue`` step (open-list
    styles; ``auto`` may pick one; not when ``--skip`` names it, ``analysis.skip`` when there is
    none). An explicit ``--ollama-url`` or ``--ollama-model`` wins over both."""
    if any(a.startswith(("--ollama-url", "--ollama-model")) for a in args):
        return False
    try:
        setting = get_config().vision  # unset: only ``auto`` asks the vision model (unsure footage and camera footage without telemetry)
        rating = "--vision" in args or (
            "--no-vision" not in args and (setting is True or (setting is None and style == AUTO))
        )
        if rating and ai_providers.is_subscription(vision_assistant_settings().provider):
            return True
        given = any(a == "--skip" or a.startswith("--skip=") for a in args)
        skipped = _values(args, "--skip") if given else analysis_skip()
        if "dialogue" in skipped or not ai_providers.is_subscription(assistant_settings().provider):
            return False
        return style == AUTO or get_style(str(style)).open_list
    except (ValueError, TypeError, AttributeError):  # an invalid config.json or style: the job says
        return False


def _vision_args(st: dict[str, Any]) -> list[str]:
    """Vision options for an analysis: the Settings values now (explicit, so the job and the page
    agree even if config.json changes before the CLI reads it), including the visual index, and
    the recorded Ollama options. With no ``vision`` key in config.json no vision flag is passed:
    the CLI default asks the vision model about unsure sources and camera-tagged sources without
    telemetry."""
    cfg = get_config()
    setting = cfg.vision
    args = [] if setting is None else ["--vision" if setting else "--no-vision"]
    args.append("--visual-index" if cfg.visual_index else "--no-visual-index")
    vision_top = st.get("vision_top")
    if number_in(vision_top, 1, integer=True) is not None:
        args.append(f"--vision-top={vision_top}")
    for key, opt in (("ollama_url", "--ollama-url"), ("ollama_model", "--ollama-model")):
        if isinstance(st.get(key), str) and st[key]:
            args.append(f"{opt}={st[key]}")
    return args


# The Analyze options (#483), each one analyze flag: the request's key and its flag.
ANALYZE_TEXT = {"model": "--model", "language": "--language"}
ANALYZE_TEXT_FORMS = {  # what the page may send for them: a model name, a language code
    "model": (
        re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"),
        "a whisper model name such as small or large-v3",
    ),
    "language": (re.compile(r"[A-Za-z]{2,3}"), "a language code such as en"),
}
ANALYZE_FLAGS = {"no_scenes": "--no-scenes", "recover": "--recover"}
ANALYZE_KEYS = ("model", "language", "max_candidates", "no_scenes", "skip", "must_include_crash",
                "recover", "vision_detect_all")  # fmt: skip


def _default_model(st: dict[str, Any]) -> str:
    """The whisper model ``analyze`` picks without ``--model`` on the device the analysis ran on
    (Transcriber's default)."""
    return transcribe.default_model("cuda" if st.get("whisper_device") == "cuda" else "cpu")


def _style_crash(style: object, ps: ProjectStyle | None = None) -> float | None:
    """The ``must_include_crash`` of ``style`` (what ``--must-include-crash`` overrides), else
    None (no known style). ``auto`` picks a built-in style, all with the same value: its
    fallback's. ``ps``: the project's style (:meth:`ProjectStyle.style`)."""
    ps = ps or ProjectStyle.of({}, None)
    s = ps.style(auto_fallback(HYPE) if style == AUTO else style)
    return s.must_include_crash if s is not None else None


def _no_scenes(st: dict[str, Any], style: object, ps: ProjectStyle) -> bool:
    """The analysis ran with ``--no-scenes``: it recorded no scene detection although its style
    detects scenes (action-cam never does, so it records none without the flag)."""
    s = ps.style(style)
    return st.get("scenes") is False and (s is None or s.use_scenes)


def _saved_skip(p: Path, hl: dict[str, Any] | None) -> list[str]:
    """The ``--skip`` steps the UI ran the project's current analysis with; none when that file
    belongs to an earlier analysis (``created`` differs), as after a command-line re-run."""
    saved = _load(p, ANALYSIS_FILE)
    if not isinstance(saved, dict) or not hl or saved.get("created") != hl.get("created"):
        return []
    skip = saved.get("skip")
    return [s for s in progress.SKIPPABLE if isinstance(skip, list) and s in skip]


def _vision_detect_all(hl: dict[str, Any] | None) -> bool:
    """``--vision-detect-all`` as the project's analysis used it (``settings.vision_detect_all``):
    on when recorded, off for an auto analysis without it; with no auto analysis yet
    ``vision_detect_all`` in config.json (off when config.json is unreadable)."""
    st = (hl or {}).get("settings") or {}
    if st.get("vision_detect_all") is True:
        return True
    if st.get("style_requested") == AUTO and isinstance((hl or {}).get("detection"), dict):
        return False
    try:
        return vision_detect_all_default()
    except ValueError:  # an invalid config.json: the job says
        return False


def analyze_options(p: Path, meta: dict[str, Any], hl: dict[str, Any] | None) -> dict[str, Any]:
    """The Analyze options the project's analysis used, as the page sends them: what
    ``highlights.json`` recorded (a model only when not the default), the saved ``--skip`` and
    ``vision_detect_all`` (:func:`_vision_detect_all`)."""
    st = (hl or {}).get("settings") or {}
    model = st.get("whisper_model")
    mc = st.get("max_candidates")
    crash = st.get("must_include_crash")
    ps = ProjectStyle.of(meta, hl)
    ran = ps.ran  # not meta's: it may be mid-switch
    default_crash = _style_crash(ran, ps)
    return {
        "model": model
        if isinstance(model, str) and model and model != _default_model(st)
        else None,
        "language": st["language"]
        if isinstance(st.get("language"), str) and st["language"]
        else None,
        "max_candidates": number_in(mc, 1, integer=True),
        "no_scenes": _no_scenes(st, ran, ps),
        "skip": _saved_skip(p, hl),
        "must_include_crash": number_in(crash, 0, 1)
        if default_crash is not None and crash != default_crash
        else None,
        "recover": st.get("recover") is True,
        "vision_detect_all": _vision_detect_all(hl),
    }


# Warnings of a refining job's envelope (crashes, events) the user must read: where a vision frame
# would go and how to allow it, a vision model that is not there, and a description that was not used
REFINE_WARNINGS = (
    "vision_consent_needed",
    "vision_unavailable",
    "vlm_unavailable",
    "context_not_applied",
    "step_failed",
    *selfcheck.CODES,  # the analysis self-checks (#1073): a detection that silently failed
)


def events_unavailable() -> str | None:
    """Why the events extra's steps (``events``, crash detection) cannot run here: the lite
    edition's message with the install command (:func:`cutpoint.extras.missing_module_message`),
    else None."""
    from cutpoint import extras  # noqa: PLC0415

    if extras.edition() != "lite":
        return None
    missing = next(
        (
            m
            for m, e in extras.MODULE_EXTRA.items()
            if e == "events" and not importlib.util.find_spec(m)
        ),
        None,
    )
    return extras.missing_module_message(missing) if missing else None


def crash_step_runs(p: Path, meta: dict[str, Any], hl: dict[str, Any] | None) -> bool:
    """The project's analysis ran (or would run) the crash step (``plan_layers``): not a 360
    project; an action, sport or action-cam style, footage detected as action-cam, action or general
    (``--style auto``), or sources with camera telemetry (the analysis then wrote crashes.json).
    Before the first analysis (``hl`` None) the project's chosen style answers: yes for those
    styles, ``auto`` or none (the footage decides), no for any other."""
    if meta.get("kind") == "360":
        return False
    ps = ProjectStyle.of(meta, hl)
    if hl is None:
        style = meta.get("style")
        if not style or style == AUTO:
            return True
        s = ps.style(style)
        return s is not None and (s.kind == ACTION_CAM or s.base in ("action", "sport"))
    from cutpoint import category  # noqa: PLC0415

    s = ps.style(ps.ran)
    if s is not None and (s.kind == ACTION_CAM or s.base in ("action", "sport")):
        return True
    det = hl.get("detection")
    if (
        isinstance(det, dict)
        and not det.get("fallback")
        and det.get("category") in (category.ACTION_CAM, category.ACTION, category.GENERAL)
    ):
        return True
    return _inside(p, CRASHES) and (p / CRASHES).is_file()


def footage_context(p: Path) -> str | None:
    """The description of the footage ``activity.json`` was made with (``request.context``)."""
    rec = _load(p, ACTIVITY)
    req = rec.get("request") if isinstance(rec, dict) else None
    ctx = req.get("context") if isinstance(req, dict) else None
    return ctx if isinstance(ctx, str) and ctx.strip() else None


def refine_state(p: Path, meta: dict[str, Any], hl: dict[str, Any] | None) -> dict[str, Any]:
    """What the page's "Describe the footage" and "Look closer with vision" need: whether the
    crash step applies (``crash_step``), the saved description (``context``) and why the events
    extra's steps cannot run here (``unavailable``)."""
    return {
        "crash_step": crash_step_runs(p, meta, hl),
        "context": footage_context(p),
        "unavailable": events_unavailable(),
    }


def _note_warnings(job: Job) -> None:
    """Put the envelope's :data:`REFINE_WARNINGS` into the job's log as WARNING notes, when the
    CLI's own line (logged as it ran) is not there already; the warning's ``files`` (the sources
    it is about) follow the message."""
    for w in (job.result or {}).get("warnings") or []:
        msg = w.get("message") if isinstance(w, dict) else None
        if (
            isinstance(msg, str)
            and w.get("code") in REFINE_WARNINGS
            and not any(msg in line for line in job.logs)
        ):
            files = [f for f in w.get("files") or [] if isinstance(f, str)]
            job.note("WARNING", f"{msg} ({', '.join(files)})" if files else msg)


def _vision_detect_all_arg(value: object) -> list[str]:
    """``--vision-detect-all`` or ``--no-vision-detect-all`` for ``value`` (explicit, so the job
    and the page agree even if config.json changes before the CLI reads it); none for None."""
    if value is None:
        return []
    if not isinstance(value, bool):
        raise bad("vision_detect_all must be true or false")
    return ["--vision-detect-all" if value else "--no-vision-detect-all"]


def _analyze_args(
    body: dict[str, Any], recorded: dict[str, Any], auto: bool = False
) -> tuple[list[str], list[str]]:
    """The Analyze options' flags and the ``--skip`` steps: each from the request, else (a key it
    does not send) what the project's analysis used, so Analyze again repeats it.
    ``--vision-detect-all`` only for an ``auto`` analysis (the other styles detect nothing)."""
    opts = {k: body.get(k, v) for k, v in recorded.items()}
    args = []
    for key, flag in ANALYZE_TEXT.items():
        v = opts[key]
        if v is not None and not isinstance(v, str):
            raise bad(f"{key} must be text")
        v = (v or "").strip()
        if not v:
            continue
        pattern, what = ANALYZE_TEXT_FORMS[key]
        if not pattern.fullmatch(v):
            raise bad(f"{key} must be {what}")
        args.append(f"{flag}={v}")
    mc = opts["max_candidates"]
    if mc is not None:
        if number_in(mc, 1, integer=True) is None:
            raise bad("max_candidates must be a whole number above 0")
        args.append(f"--max-candidates={mc}")
    crash = opts["must_include_crash"]
    if crash is not None:
        if (value := number_in(crash, 0, 1)) is None:
            raise bad("must_include_crash must be a number from 0 to 1")
        args.append(f"--must-include-crash={value:g}")
    for key, flag in ANALYZE_FLAGS.items():
        if _flag(opts, key):
            args.append(flag)
    vda = _vision_detect_all_arg(opts["vision_detect_all"])
    if auto:
        args += vda
    skip = opts["skip"]
    if skip is None:
        skip = []
    if not isinstance(skip, list) or any(
        not isinstance(s, str) or s not in progress.SKIPPABLE for s in skip
    ):
        raise bad(f"skip must be a list of {', '.join(progress.SKIPPABLE)}")
    return args, [s for s in progress.SKIPPABLE if s in skip]


def preset_problem(meta: dict[str, Any], hl: dict[str, Any] | None) -> str | None:
    """Why ``cutpoint edit`` cannot remake this project's analysis (None when it can)."""
    if meta.get("kind") == "360":
        return "Presets are not available for 360 projects (cutpoint edit has no 360 mode)."
    st = (hl or {}).get("settings") or {}
    exclude, recursive = picked_filters(meta)
    if st.get("exclude") or st.get("recursive") is True or exclude or recursive:
        return (
            "Presets are not available here: this project's analysis used --exclude or "
            "--recursive, which cutpoint edit does not take."
        )
    # edit has no --model or --no-scenes: it would transcribe again with the default model or
    # detect scenes, and rewrite highlights.json. The default model is Transcriber's for the
    # device the analysis ran on.
    model = st.get("whisper_model")
    if model is not None and model != _default_model(st):
        return (
            f"Presets are not available here: this project was transcribed with the {model} "
            "model (analyze --model), which cutpoint edit does not take."
        )
    ps = ProjectStyle.of(meta, hl)
    inputs, style, ran = ps.inputs, ps.chosen, ps.ran  # ran: as analyze_options
    if _no_scenes(st, ran, ps):
        return (
            "Presets are not available here: this project's analysis used --no-scenes, which "
            "cutpoint edit does not take."
        )
    if st.get("recover") is True:
        return (
            "Presets are not available here: this project's analysis used --recover, which "
            "cutpoint edit does not take."
        )
    # nor --language or --max-candidates (recorded as None when not given)
    for key, opt in (("language", "--language"), ("max_candidates", "--max-candidates")):
        if st.get(key) is not None:
            return (
                f"Presets are not available here: this project's analysis used {opt}, which "
                "cutpoint edit does not take."
            )
    # nor --must-include-crash: edit would analyze with the style's own confidence
    crash, default_crash = st.get("must_include_crash"), _style_crash(ran, ps)
    # (a deleted style runs in its base: the base's confidence, whatever the style had)
    if (
        default_crash is not None
        and crash is not None
        and crash != default_crash
        and not ps.ran_deleted
    ):
        return (
            "Presets are not available here: this project's analysis used --must-include-crash, "
            "which cutpoint edit does not take."
        )
    if not inputs or not style:
        return "Presets need a recorded source to analyze."
    return None


def preset_taken(p: Path) -> list[str]:
    """The folders a preset writes (``clips``, ``clips_vertical``) that hold files already, so
    ``edit`` needs ``--force``."""
    taken = []
    for name in ("clips", "clips_vertical"):
        with contextlib.suppress(OSError):
            if (p / name).is_dir() and any((p / name).iterdir()):
                taken.append(name)
    return taken


def _seconds(body: dict[str, Any], key: str, hi: float, *, positive: bool = False) -> float | None:
    """``body[key]`` as seconds, 0 (or above 0 with ``positive``) to ``hi``; None when not sent."""
    if body.get(key) is None:
        return None
    v = number_in(body[key], 0, hi)
    if v is None or (positive and v == 0):
        rule = f"above 0 and at most {hi:g}" if positive else f"0-{hi:g}"
        raise bad(f"{key} must be {rule} seconds")
    return v


def _opt(flag: str, v: float | None) -> list[str]:
    return [] if v is None else [flag, f"{v:g}"]


def _picture_args(body: dict[str, Any]) -> list[str]:
    """``clips --crop`` and ``--size`` as the page sent them; not sent: the CLI's defaults (the
    project's per-source crop, else none; the largest source's size, capped at 1080p)."""
    args = []
    crop, size = body.get("crop"), body.get("size")
    if crop is not None:
        if not isinstance(crop, str) or not (crop in ("auto", "none") or CROP_BOX.fullmatch(crop)):
            raise bad("crop must be 'auto', 'none' or W:H:X:Y in pixels")
        args += ["--crop", crop]
    if size is not None:
        if not isinstance(size, str) or not CLIP_SIZE.fullmatch(size):
            raise bad("size must be WxH in even pixels, such as 1920x1080")
        args += ["--size", size]
    return args


def _speech_args(body: dict[str, Any]) -> list[str]:
    """``--speech-max-extend`` / ``--speech-max-lead`` when the page sent them."""
    return _opt("--speech-max-extend", _seconds(body, "speech_max_extend", MAX_TIMING_S)) + _opt(
        "--speech-max-lead", _seconds(body, "speech_max_lead", MAX_TIMING_S)
    )


def _auto_picks(body: dict[str, Any], hl: dict[str, Any] | None) -> list[str]:
    """``clips`` selection flags for automatic picks: ``--top`` or ``--ids``, ``--max-total`` and
    ``--order``, each only when sent (else the CLI's defaults: the style's count and order)."""
    cands = (hl or {}).get("candidates")
    if not isinstance(cands, list) or not cands:
        raise bad("analyze the footage before exporting automatic picks")
    top, ids, order = body.get("top"), body.get("ids"), body.get("order")
    if top is not None and ids is not None:
        raise bad("give top or ids, not both")
    args: list[str] = []
    if top is not None:
        if number_in(top, 1, MAX_SEGMENTS, integer=True) is None:
            raise bad(f"top must be a whole number from 1 to {MAX_SEGMENTS}")
        args += ["--top", str(top)]
    if ids is not None:
        known = {c.get("id") for c in cands if isinstance(c, dict)}
        if (
            not isinstance(ids, list)
            or not 0 < len(ids) <= MAX_SEGMENTS
            or len(set(map(repr, ids))) != len(ids)
            or not all(isinstance(i, str) and SAFE_NAME.fullmatch(i) and i in known for i in ids)
        ):
            raise bad("ids must be different candidate ids of this analysis, such as h001")
        args += ["--ids", *ids]
    args += _opt("--max-total", _seconds(body, "max_total", MAX_TOTAL_S, positive=True))
    if order is not None:
        if order not in CLIP_ORDERS:
            raise bad(f"order must be one of {', '.join(CLIP_ORDERS)}")
        args += ["--order", order]
    return args


PREVIEW_NOTE = "exporting from the 360 preview (not stitched)"


def stitch_first(
    job: Job, project: Path, seg_file: Path | None, picks: dict[str, Any]
) -> list[str] | None:
    """``cutpoint 360 stitch`` for the recordings a 360 export needs that are not stitched yet,
    or None (``pano.auto_stitch`` off, no working SDK or NVIDIA GPU, all stitched, too little
    free space).

    ``seg_file``: the export's segments file; else ``picks`` (``top``, ``ids``, ``max_total``)
    choose the moments as ``clips`` will. Says why in the job log; never raises for a reason
    to skip (``Jobs.start`` turns anything else into a warning too).
    """
    try:
        on = pano_project.auto_stitch()
    except ValueError as err:
        job.note("WARNING", f"{err}; {PREVIEW_NOTE}")
        return None
    if not on:
        job.note("INFO", f"pano.auto_stitch is off: {PREVIEW_NOTE}")
        return None
    sdk = pano_stitch.sdk_status()
    if not sdk.get("runs"):
        why = "does not run" if sdk.get("found") else "was not found"
        job.note("WARNING", f"the Insta360 SDK {why} (see cutpoint 360 status): {PREVIEW_NOTE}")
        return None
    from cutpoint import capabilities  # noqa: PLC0415

    if not capabilities.nvidia_present():  # the SDK needs one: the stitch would likely fail
        job.note("WARNING", "no NVIDIA GPU; exporting the 360 preview")
        return None
    if seg_file is not None:
        sources = [m["source"] for m in cache.read_json(seg_file)]
    else:
        from cutpoint.planning import select_moments  # noqa: PLC0415

        sources = [m["source"] for m in select_moments(project, **picks)[0]]
    recorded = pano_project.load(project)
    chosen = {c.clip_id: c for s in sources if (c := pano_project.clip_for(recorded, s))}
    missing = [c for _, c in sorted(chosen.items()) if not pano_project.sdk_output(c).is_file()]
    if not missing:
        if chosen:
            job.note("INFO", f"the {len(chosen)} recording(s) of this export are stitched")
        return None
    need = pano_stitch.estimate_gb(missing)
    out_dir = pano_project.stitched_dir()
    free = pano_stitch.free_gb(out_dir)
    room = "unknown" if free is None else f"{free:.0f} GB"
    job.note(
        "INFO",
        f"stitching {len(missing)} recording(s) with the Insta360 SDK first: about "
        f"{need:.0f} GB in {out_dir} ({room} free)",
    )
    if free is None:
        job.note("WARNING", f"cannot read the free space in {out_dir}: {PREVIEW_NOTE}")
        return None
    if free - need < pano_stitch.MIN_FREE_GB:
        job.note(
            "WARNING",
            f"not enough free space in {out_dir} (needs about {need:.0f} GB plus "
            f"{pano_stitch.MIN_FREE_GB:.0f} GB to spare): {PREVIEW_NOTE}",
        )
        return None
    if seg_file is not None:  # the stitch skips the recordings already stitched
        return cutpoint_cmd("360", "stitch", str(project), "--segments", str(seg_file))
    return cutpoint_cmd("360", "stitch", str(project), "--clips", *(c.clip_id for c in missing))
