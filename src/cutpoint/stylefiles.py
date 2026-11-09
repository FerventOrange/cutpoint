"""Style files: the user's own scoring styles in ``<CUTPOINT_HOME>/styles`` (``cutpoint style``).

A style file is ``<styles>/<name>.json`` (:func:`cutpoint.paths.styles_dir`, or
``CUTPOINT_STYLES_DIR``). Schema 1::

    {"schema": 1,
     "name": "bmx",                       # the file's name without .json
     "base": "action",                    # a built-in style (not another user style)
     "source": "hand",                    # hand | fitted | assistant
     "description": "...",                # optional
     "profile": "bmx",                    # optional: an activity profile name, stored only
     "applies_to": {"categories": ["action-cam"], "profiles": ["bmx"]},  # optional
     "weights": {"scene": 0.4},           # registry grid features, over the base's weights
     "mask_friends": 0, "must_include_crash": 0.6}  # optional, as in legacy presets

Any other key (``fit``, ``created``, ...) is kept as it is. ``applies_to`` and ``profile`` are
stored only: nothing picks a style by them yet, a style is used by ``--style NAME``.

A file without ``schema`` is a legacy preset (``scripts/refmatch.py``): it loads leniently as
before (:func:`cutpoint.styles.load_learned`), reads as ``source`` ``fitted`` and is never
rewritten. A schema-1 file is checked strictly (:func:`check`); one that fails is unusable.

This module holds the schema check and the logic of the ``cutpoint style`` verbs; the CLI
handlers in :mod:`cutpoint.cli` only print. Errors: :class:`cutpoint.jsonout.ArgumentError`
for a rejected value (exit 1, ``bad_args``), FileNotFoundError for a missing style or file
(``not_found``), FileExistsError for an existing name without ``--force``.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cutpoint import cache, features, jsonout, paths
from cutpoint.category import ACTION_CAM, CATEGORIES
from cutpoint.styles import (
    AUTO,
    RENAMED,
    STYLES,
    Style,
    load_learned,
    own_weights,
    style_hash,
)

logger = logging.getLogger(__name__)

SCHEMA = 1
SOURCES = ("hand", "fitted", "assistant")
LEGACY_SOURCE = "fitted"  # what a legacy preset (no "schema") reads as
APPLIES_KEYS = ("categories", "profiles")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
# File names Windows refuses whatever the extension
_RESERVED_STEMS = frozenset(
    ["con", "prn", "aux", "nul"] + [f"{p}{i}" for p in ("com", "lpt") for i in range(1, 10)]
)
_ORDER = ("schema", "name", "base", "source", "description", "profile", "applies_to", "weights")


UNKNOWN_PROFILE = "unknown_profile"  # the codes of Check.warnings
FEATURE_NOT_COMPUTED = "feature_not_computed"


@dataclass
class Check:
    """What :func:`check` found: ``problems`` make the file unusable, ``warnings`` do not."""

    problems: list[str] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)  # (code, message)

    @property
    def ok(self) -> bool:
        return not self.problems


def computed_features(base: Style) -> frozenset[str]:
    """The grid features the scorer of ``base`` measures: the action-cam grids of
    :mod:`cutpoint.ski` (:data:`cutpoint.features.SKI_FEATURES`) or, for the generic scorer,
    the other grid features plus ``speech``, which both measure."""
    grid = set(features.names("grid"))
    ski = set(features.SKI_FEATURES) & grid
    return frozenset(ski if base.kind == ACTION_CAM else (grid - ski) | {"speech"})


def is_legacy(data: dict[str, Any]) -> bool:
    """A legacy preset: a style file without ``"schema"``."""
    return "schema" not in data


def _number(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _names(items: Iterable[str]) -> str:
    return ", ".join(items)


def name_problem(name: object) -> str | None:
    """Why ``name`` cannot name a user style, or None: it must be a safe file name and not a
    built-in, renamed or reserved style name (compared ignoring case)."""
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        return (
            f"{name!r} is not a usable style name: use up to 64 letters, digits, '-' and '_', "
            "starting with a letter or digit"
        )
    low = name.lower()
    if low == AUTO:
        return f"{name!r} is reserved for --style {AUTO}"
    if low in RENAMED:
        return f"{name!r} is an old style name (now {RENAMED[low]!r}) and cannot be used"
    if low in STYLES:
        return f"{name!r} is a built-in style name; choose another name"
    if low in _RESERVED_STEMS:
        return f"{name!r} cannot be a file name on Windows; choose another name"
    return None


def user_files() -> dict[str, Path]:
    """Every ``*.json`` in the styles folder (name -> file), sorted by name; unlike
    :func:`cutpoint.styles.learned_styles`, ``auto.json`` and files named like a built-in
    are included (``style list`` reports why they are not used)."""
    folder = paths.styles_dir()
    if not folder.is_dir():
        return {}
    return {p.stem: p for p in sorted(folder.glob("*.json")) if p.is_file()}


def profile_names() -> set[str]:
    """The activity profile library's names (:func:`cutpoint.activity.library`)."""
    from cutpoint import activity  # noqa: PLC0415 - numpy and the profile files, only when asked

    return set(activity.library())


def check(data: object, stem: str, profiles: Collection[str] | None = None) -> Check:
    """Check a schema-1 style file's content ``data`` (file name ``<stem>.json``) strictly,
    collecting every problem in plain words.

    Args:
        data: The file's parsed JSON.
        stem: The file name without ``.json``; ``name`` must equal it.
        profiles: The activity profile names that exist: a ``profile`` or ``applies_to``
            profile not among them is a warning. None skips that check (loading for analyze).
    """
    res = Check()
    if not isinstance(data, dict):
        res.problems.append("the file is not a JSON object")
        return res
    p = res.problems
    schema = data.get("schema")
    if type(schema) is not int or schema != SCHEMA:
        p.append(f'"schema" is {schema!r}; this Cutpoint reads schema {SCHEMA}')

    name = data.get("name")
    if name is None:
        p.append('"name" is missing; it must be the file name without .json')
    elif name != stem:
        p.append(f'"name" is {name!r} but the file is {stem}.json; they must match')
    elif msg := name_problem(name):
        p.append(msg)

    base_style = None
    base = data.get("base")
    built_in = _names(sorted(STYLES))
    if base is None:
        p.append(f'"base" is missing; name a built-in style: {built_in}')
    elif not isinstance(base, str):
        p.append(f'"base" is {base!r}; name a built-in style: {built_in}')
    elif base in RENAMED:
        p.append(f'"base" {base!r} was renamed to {RENAMED[base]!r}')
    elif base in STYLES:
        base_style = STYLES[base]
    elif base in user_files():
        p.append(
            f'"base" {base!r} is one of your own styles; a style can only build on a built-in '
            f"style: {built_in}"
        )
    else:
        p.append(f'"base" {base!r} is not a built-in style; choose from {built_in}')

    weights = data.get("weights", {})
    weights_ok = isinstance(weights, dict)
    if not weights_ok:
        p.append('"weights" must be an object of feature names and numbers')
        weights = {}
    grid = set(features.names("grid"))
    unknown = [k for k in weights if k not in grid]
    if unknown:
        many = len(unknown) > 1
        p.append(
            f"unknown feature{'s' if many else ''} {_names(map(repr, unknown))} in "
            '"weights"; `cutpoint style features` lists the names'
        )
    numbers = True
    for k, v in weights.items():
        if not _number(v):
            p.append(f"the weight of {k!r} is {v!r}, not a number")
            numbers = False
        elif v < 0:
            p.append(f"the weight of {k!r} is {v}; weights cannot be negative")
            numbers = False
    if base_style is not None and weights_ok and numbers and not unknown:
        own = own_weights(data, base_style, Path(f"{stem}.json"))
        if not any(w > 0 for w in {**base_style.weights, **own}.values()):
            p.append(
                f"every weight is 0 once the weights of {base!r} are added; at least one "
                "must be above 0"
            )

    source = data.get("source")
    if source is None:
        p.append(f'"source" is missing; use {_names(SOURCES)}')
    elif source not in SOURCES:
        p.append(f'"source" is {source!r}; use {_names(SOURCES)}')

    if "description" in data and not isinstance(data["description"], str):
        p.append('"description" must be text')

    named_profiles: list[str] = []
    if "profile" in data:
        if isinstance(data["profile"], str) and data["profile"]:
            named_profiles.append(data["profile"])
        else:
            p.append('"profile" must be the name of an activity profile')

    if "applies_to" in data:
        named_profiles += _check_applies(data["applies_to"], p)

    if "mask_friends" in data and not (_number(data["mask_friends"]) and data["mask_friends"] >= 0):
        p.append(f'"mask_friends" is {data["mask_friends"]!r}; use a number of 0 or more')
    crash = data.get("must_include_crash")
    if "must_include_crash" in data and not (_number(crash) and 0 <= crash <= 1):
        p.append(f'"must_include_crash" is {crash!r}; use a number from 0 to 1')

    if base_style is not None and weights_ok and numbers and not unknown:
        own = own_weights(data, base_style, Path(f"{stem}.json"))
        measured = computed_features(base_style)
        idle = [k for k, w in own.items() if w > 0 and k not in measured]
        if idle:
            one = len(idle) == 1
            msg = (
                f"{_names(map(repr, idle))} {'is' if one else 'are'} never measured for base "
                f"{base!r}, so {'it' if one else 'they'} only lower{'s' if one else ''} every "
                "score; the file is kept"
            )
            res.warnings.append((FEATURE_NOT_COMPUTED, msg))

    if profiles is not None:
        for prof in dict.fromkeys(named_profiles):
            if prof not in profiles:
                res.warnings.append(
                    (
                        UNKNOWN_PROFILE,
                        f"profile {prof!r} is not in the activity profile library; it is kept, "
                        "but nothing matches it yet",
                    )
                )
    return res


def _check_applies(applies: object, p: list[str]) -> list[str]:
    """Add the problems of ``applies_to`` to ``p``; returns the profile names it gives."""
    if not isinstance(applies, dict):
        p.append('"applies_to" must be an object with "categories" and "profiles" lists')
        return []
    extra = [k for k in applies if k not in APPLIES_KEYS]
    if extra:
        p.append(
            f'"applies_to" has unknown key(s) {_names(map(repr, extra))}; '
            f"it takes {_names(APPLIES_KEYS)}"
        )
    out: list[str] = []
    for key in APPLIES_KEYS:
        items = applies.get(key, [])
        if not isinstance(items, list) or not all(isinstance(i, str) and i for i in items):
            p.append(f'"applies_to" "{key}" must be a list of names')
            continue
        if key == "categories":
            bad = [c for c in items if c not in CATEGORIES]
            if bad:
                p.append(
                    f'"applies_to" names unknown categor{"ies" if len(bad) > 1 else "y"} '
                    f"{_names(map(repr, bad))}; choose from {_names(CATEGORIES)}"
                )
        else:
            out += items
    return out


def _applies(data: dict[str, Any]) -> dict[str, list[str]]:
    """``applies_to`` of a style file as listed (both keys; what is not a list is empty)."""
    raw = data.get("applies_to")
    out: dict[str, list[str]] = {k: [] for k in APPLIES_KEYS}
    if isinstance(raw, dict):
        for k in APPLIES_KEYS:
            if isinstance(raw.get(k), list):
                out[k] = list(raw[k])
    return out


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


# --------------------------------------------------------------------------- list


def _builtin_entry(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "built-in",
        "legacy": False,
        "valid": True,
        "problems": [],
        "warnings": [],
        "source": None,
        "base": name,
        "description": None,
        "profile": None,
        "applies_to": {k: [] for k in APPLIES_KEYS},
        "path": None,
    }


def _file_entry(stem: str, path: Path, profiles: Collection[str] | None) -> dict[str, Any]:
    data = cache.load_json(path)
    doc = data if isinstance(data, dict) else {}
    legacy = isinstance(data, dict) and is_legacy(data)
    entry: dict[str, Any] = {
        "name": stem,
        "type": "user",
        "legacy": legacy,
        "valid": False,
        "problems": [],
        "warnings": [],
        "source": LEGACY_SOURCE if legacy else _text(doc.get("source")),
        "base": _text(doc.get("base")),
        "description": _text(doc.get("description")),
        "profile": _text(doc.get("profile")),
        "applies_to": _applies(doc),
        "path": path,
    }
    entry["problems"], warnings = _file_problems(stem, path, data, profiles)
    entry["warnings"] = _warning_dicts(warnings)
    entry["valid"] = not entry["problems"]
    return entry


def _warning_dicts(warnings: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"code": c, "message": m} for c, m in warnings]


def _file_problems(
    stem: str, path: Path, data: object, profiles: Collection[str] | None
) -> tuple[list[str], list[tuple[str, str]]]:
    """The problems and warnings of the style file ``path`` as ``--style`` would load it."""
    if not isinstance(data, dict):
        return ["the file cannot be read as a JSON object"], []
    if stem.lower() == AUTO:
        return [f"{AUTO!r} is reserved for --style {AUTO}; the file is ignored"], []
    if stem in STYLES:
        return [f"the built-in style {stem!r} has this name; the file is ignored"], []
    if stem in RENAMED:
        return [f"{stem!r} is an old style name (now {RENAMED[stem]!r}); the file is ignored"], []
    if not is_legacy(data):
        res = check(data, stem, profiles)
        return res.problems, res.warnings
    return _legacy_problems(path), []


def _legacy_problems(path: Path) -> list[str]:
    """Why the legacy preset ``path`` does not load (leniently, as for ``--style``), if so."""
    try:
        load_learned(path)
    except (ValueError, TypeError, AttributeError) as err:
        return [str(err)]
    return []


def list_styles() -> list[dict[str, Any]]:
    """Every style: the built-ins (sorted), then the files of the styles folder (sorted).

    Each entry: ``name``, ``type`` (``built-in`` or ``user``), ``legacy``, ``valid``,
    ``problems``, ``warnings``, ``source`` (None for built-ins; ``fitted`` for legacy presets),
    ``base``, ``description``, ``profile``, ``applies_to`` (``categories``, ``profiles``) and
    ``path`` (None for built-ins). A broken file is listed as not valid, never raised.
    """
    files = user_files()
    profiles = profile_names() if files else None
    return [_builtin_entry(n) for n in sorted(STYLES)] + [
        _file_entry(stem, path, profiles) for stem, path in files.items()
    ]


# --------------------------------------------------------------------------- show


def _find(name: str) -> Path:
    """The style file of the user style ``name`` (exact name).

    Raises:
        FileNotFoundError: If the styles folder has no such file.
    """
    path = user_files().get(name)
    if path is None:
        raise FileNotFoundError(
            f"no style named {name!r}; `cutpoint style list` shows the styles "
            f"(your styles are in {paths.styles_dir()})"
        )
    return path


def show(name: str) -> dict[str, Any]:
    """The style ``name`` with its effective weights.

    Returns:
        ``name``, ``type``, ``legacy``, ``source``, ``base``, ``kind``, ``description``,
        ``profile``, ``applies_to``, ``path``, ``hash`` (:func:`cutpoint.styles.style_hash`),
        ``mask_friends``, ``must_include_crash`` and ``weights``: a list of ``{"feature",
        "weight", "from"}`` in scoring order, ``from`` being ``base`` (the base style's
        weight; every weight of a built-in) or ``style`` (set by the style file).

    Raises:
        FileNotFoundError: For an unknown style.
        jsonout.ArgumentError: For ``auto``, a renamed name, or a style file that cannot
            be used (naming every problem).
    """
    if name == AUTO:
        raise jsonout.ArgumentError(
            f"{AUTO!r} detects a style from the footage; it is not a style itself"
        )
    if name in STYLES:
        style = STYLES[name]
        entry = _builtin_entry(name)
        own: dict[str, float] = {}
    else:
        if name in RENAMED and name not in user_files():
            raise jsonout.ArgumentError(
                f"style {name!r} was renamed to {RENAMED[name]!r}; use {RENAMED[name]!r}"
            )
        path = _find(name)
        entry = _file_entry(name, path, None)
        if not entry["valid"]:
            raise jsonout.ArgumentError(f"{path} cannot be used: {'; '.join(entry['problems'])}")
        style = load_learned(path)
        data = cache.load_json(path) or {}
        own = own_weights(data, STYLES[style.base], path)
    return {
        "name": entry["name"],
        "type": entry["type"],
        "legacy": entry["legacy"],
        "source": entry["source"],
        "base": style.base,
        "kind": style.kind,
        "description": entry["description"],
        "profile": entry["profile"],
        "applies_to": entry["applies_to"],
        "path": entry["path"],
        "hash": style_hash(style),
        "mask_friends": style.mask_friends,
        "must_include_crash": style.must_include_crash,
        "weights": [
            {"feature": k, "weight": w, "from": "style" if k in own else "base"}
            for k, w in style.weights.items()
        ],
    }


# --------------------------------------------------------------------------- create


def parse_weights(items: Iterable[str]) -> dict[str, float]:
    """``--weight FEATURE=W`` values as ``{feature: weight}`` (a later one wins).

    Raises:
        jsonout.ArgumentError: For a value without ``=`` or a weight that is not a number.
    """
    out: dict[str, float] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise jsonout.ArgumentError(f"--weight {item}: expected FEATURE=WEIGHT")
        try:
            weight = float(value)
        except ValueError:
            raise jsonout.ArgumentError(f"--weight {item}: {value!r} is not a number") from None
        out[key.strip()] = weight
    return out


def parse_applies(items: Iterable[str]) -> dict[str, list[str]]:
    """``--applies-to category:X profile:Y ...`` as ``{"categories": [...], "profiles": [...]}``.

    Raises:
        jsonout.ArgumentError: For an item that is not ``category:NAME`` or ``profile:NAME``.
    """
    out: dict[str, list[str]] = {k: [] for k in APPLIES_KEYS}
    for item in items:
        kind, sep, value = item.partition(":")
        key = {"category": "categories", "profile": "profiles"}.get(kind.strip())
        if not sep or key is None or not value.strip():
            raise jsonout.ArgumentError(
                f"--applies-to {item}: expected category:NAME or profile:NAME"
            )
        if value.strip() not in out[key]:
            out[key].append(value.strip())
    return out


def _ordered(doc: dict[str, Any]) -> dict[str, Any]:
    """``doc`` with the schema keys first, in :data:`_ORDER`, then the others as they were."""
    head = {k: doc[k] for k in _ORDER if k in doc}
    return {**head, **{k: v for k, v in doc.items() if k not in head}}


def create(
    name: str,
    *,
    base: str | None = None,
    weights: dict[str, float] | None = None,
    description: str | None = None,
    applies_to: dict[str, list[str]] | None = None,
    profile: str | None = None,
    from_file: Path | None = None,
    force: bool = False,
) -> tuple[Path, Check]:
    """Write the schema-1 style ``<styles>/<name>.json``.

    With ``from_file``, its content is the start (a schema-1 or legacy style file, or any
    JSON object): every key is kept, ``name`` becomes ``name``, and the given options replace
    its ``base``, ``description``, ``profile`` and ``applies_to``; ``weights`` are added over
    its weights. ``source`` is the file's, else ``hand``.

    Returns:
        The file written and its :class:`Check` (warnings only).

    Raises:
        jsonout.ArgumentError: For an unusable name, a missing base, a ``from_file`` that is
            not a JSON object, or content that fails :func:`check` (naming every problem).
        FileNotFoundError: If ``from_file`` does not exist.
        FileExistsError: If the style exists and ``force`` is not given.
    """
    if msg := name_problem(name):
        raise jsonout.ArgumentError(msg)
    doc: dict[str, Any] = {}
    if from_file is not None:
        if not from_file.is_file():
            raise FileNotFoundError(f"--from {from_file}: file not found")
        data = cache.load_json(from_file)
        if not isinstance(data, dict):
            raise jsonout.ArgumentError(f"--from {from_file}: not a JSON object")
        doc = dict(data)
    legacy = is_legacy(doc)
    if legacy and isinstance(doc.get("weights"), dict):  # loads as if never weighed
        gone = [k for k in doc["weights"] if k in features.REMOVED]
        if gone:
            logger.info("--from %s: dropping the weights of removed features %s", from_file, gone)
            doc["weights"] = {k: w for k, w in doc["weights"].items() if k not in gone}
    if base is None and "base" not in doc:
        raise jsonout.ArgumentError('--base is required (or a --from file with a "base")')
    doc.setdefault("schema", SCHEMA)
    doc["name"] = name
    if base is not None:
        doc["base"] = base
    doc.setdefault("source", "hand")
    if description is not None:
        doc["description"] = description
    if profile is not None:
        doc["profile"] = profile
    if applies_to is not None:
        doc["applies_to"] = applies_to
    own = doc.get("weights", {})
    doc["weights"] = {**own, **(weights or {})} if isinstance(own, dict) else own

    files = user_files()
    clash = next((s for s in files if s.lower() == name.lower() and s != name), None)
    if clash is not None:
        raise jsonout.ArgumentError(
            f"a style named {clash!r} exists; names that differ only in case clash on "
            "Windows and macOS"
        )
    path = paths.styles_dir() / f"{name}.json"
    if name in files and not force:
        raise FileExistsError(f"style {name!r} already exists ({path}); add --force to replace it")

    res = check(doc, name, profile_names())
    if not res.ok:
        raise jsonout.ArgumentError(f"style {name!r} not saved: {'; '.join(res.problems)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.save_json(path, _ordered(doc))
    return path, res


# --------------------------------------------------------------------------- validate


def _looks_like_file(target: str) -> bool:
    return target.endswith(".json") or "/" in target or "\\" in target or Path(target).is_file()


def validate(target: str) -> dict[str, Any]:
    """Check the style ``target``: a style name, or a file (a path, or a name ending in
    ``.json``).

    Returns:
        ``name``, ``path`` (None for a built-in), ``type`` (``built-in``, ``user``; None for a
        renamed name), ``legacy``, ``valid``,
        ``problems``, ``warnings`` and ``notes``. The caller fails the command when
        ``valid`` is false.

    Raises:
        FileNotFoundError: For an unknown style name or a missing file.
    """
    if not _looks_like_file(target):
        if target in STYLES:
            return {
                "name": target,
                "path": None,
                "type": "built-in",
                "legacy": False,
                "valid": True,
                "problems": [],
                "warnings": [],
                "notes": [f"{target!r} is a built-in style"],
            }
        if target in RENAMED and target not in user_files():
            problem = f"style {target!r} was renamed to {RENAMED[target]!r}"
            return {
                "name": target,
                "path": None,
                "type": None,
                "legacy": False,
                "valid": False,
                "problems": [problem],
                "warnings": [],
                "notes": [],
            }
        path = _find(target)
    else:
        path = Path(target)
        if not path.is_file():
            raise FileNotFoundError(f"{target}: file not found")
    data = cache.load_json(path)
    legacy = isinstance(data, dict) and is_legacy(data)
    in_folder = path.parent.resolve() == paths.styles_dir().resolve()
    if not isinstance(data, dict):
        problems, warnings = ["the file cannot be read as a JSON object"], []
    elif in_folder:  # also why --style would ignore it (auto.json, a built-in's name)
        problems, warnings = _file_problems(path.stem, path, data, profile_names())
    elif legacy:
        problems, warnings = _legacy_problems(path), []
    else:
        res = check(data, path.stem, profile_names())
        problems, warnings = res.problems, res.warnings
    notes = []
    if legacy and not problems:
        notes.append(
            'a legacy preset (no "schema"): it loads as before and is never rewritten; '
            f"`cutpoint style create NAME --from {path}` makes a schema-{SCHEMA} copy"
        )
    return {
        "name": path.stem,
        "path": path,
        "type": "user",
        "legacy": legacy,
        "valid": not problems,
        "problems": problems,
        "warnings": _warning_dicts(warnings),
        "notes": notes,
    }


# --------------------------------------------------------------------------- delete


def delete(name: str) -> Path:
    """Delete the user style ``name`` (its file in the styles folder).

    Raises:
        jsonout.ArgumentError: For a built-in style.
        FileNotFoundError: If there is no such style file.
    """
    if name in STYLES:
        shadow = user_files().get(name)
        extra = f" (the file {shadow} is ignored; remove it by hand)" if shadow else ""
        raise jsonout.ArgumentError(f"{name!r} is a built-in style and cannot be deleted{extra}")
    path = _find(name)
    path.unlink()
    return path


# --------------------------------------------------------------------------- features


def feature_list() -> list[dict[str, Any]]:
    """The registry's grid features (what style weights can name), sorted: ``name``,
    ``layer``, ``description``, ``extra`` (None for core) and ``available`` (in this
    install)."""
    avail = features.available()
    return [
        {
            "name": f.name,
            "layer": f.layer,
            "description": f.description,
            "extra": f.extra,
            "available": f.name in avail,
        }
        for f in (features.get(n) for n in features.names("grid"))
    ]
