"""Development benchmark across footage categories (#900): runs ``cutpoint analyze --style auto``
on labelled footage, scores the detection against the labelled category and the ranking against
the owner's moment labels (with :mod:`moment_eval`), compares variants (A vs B) and gives an
advisory gate verdict. Not part of the package. See ``docs/workflows/benchmark.md``.

Usage::

    uv run --offline python scripts/bench.py H:/bench/manifest.json
    uv run --offline python scripts/bench.py manifest.json --variant base: --variant novis:--no-vision
    uv run --offline python scripts/bench.py manifest.json --holdout
    uv run --offline python scripts/bench.py manifest.json --rehash
    uv run --offline python scripts/bench.py --compare results/A.json results/B.json

The manifest (keep it and the label files out of the repository; paths are relative to it)::

    {"schema": 1, "holdout_sha256": "<hex>",
     "variants": [{"name": "base", "args": []},
                  {"name": "old", "args": [], "cutpoint": ["uv", "run", "--project", "../old",
                                                           "cutpoint"]}],
     "sources": [{"id": "race-1", "input": "footage/race-1.mp4", "category": "gameplay",
                  "bucket": "gameplay-nokills", "labels": "labels/race-1.json",
                  "holdout": false, "note": "free text"}]}

- ``category``: the true footage category (``cutpoint.category.CATEGORIES``); for ``mixed`` the
  category of most of the footage by duration, set by the labeller.
- ``bucket``: one of :data:`BUCKETS`; the labeller sets it (kills or no kills, telemetry or not).
- ``labels`` (optional): a :mod:`moment_eval` label file. Times may also be ``"mm:ss"`` or
  ``"h:mm:ss"`` strings. Without labels the source is scored for detection only. A label file's
  own ``holdout``, when present, must agree with the manifest.
- ``holdout``: never analyzed or reported unless ``--holdout``. ``holdout_sha256`` is the sha256
  of the sorted holdout ids joined by newlines; a manifest whose holdout ids do not give it is
  refused (``--rehash`` prints the hash to store; the script never rewrites the manifest).

Per source and variant, one at a time: ``<cutpoint> analyze <input> --style auto --out
<bench>/runs/<variant>/<id> --json <variant args>``, with ``CUTPOINT_HOME=<bench>/home``,
``CUTPOINT_CACHE_DIR=<bench>/cache`` (``--cache``, or the variant's own ``cache``) and the user's
models folder. ``<cutpoint>`` is the
variant's ``cutpoint`` (a list, or a string split like a shell command line), else
``python -m cutpoint`` of the interpreter running this script. A run whose stamp (cutpoint,
version, git state, edition, variant args, input sizes and times) matches is reused
(``--rerun`` forces it). A background thread samples Ollama's ``/api/ps`` and ``nvidia-smi``
while each run goes. Results go to ``<bench>/results/<timestamp>-<variant>.json`` and
``<timestamp>-report.md`` (also printed). One bench runs at a time on the machine (a lock in the
temporary folder). Exit codes: 0 done (the gate is advice), 1 error, 2 bad arguments, a refused
manifest or a file ``--compare`` cannot read, 3 a failed gate with ``--strict-gate``, 130
interrupted.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import logging
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from cutpoint import __version__, extras, paths, procs, reset_state
from cutpoint.category import CATEGORIES

logger = logging.getLogger(__name__)


def _load_moment_eval() -> Any:
    """``scripts/moment_eval.py`` (a sibling script, not a package module)."""
    name = "moment_eval"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().with_name("moment_eval.py")
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


me = _load_moment_eval()

SCHEMA = 1
KIND = "cutpoint-bench"
REPO = Path(__file__).resolve().parents[1]
BUCKETS = (
    "gameplay-kills",
    "gameplay-nokills",
    "actioncam-telemetry",
    "actioncam-notelemetry",
    "talk",
    "vlog",
    "mixed",
)
ALL = "all"
DEFAULT_SAMPLE_S = 5.0
LOCK_NAME = "cutpoint-bench.lock"
STAMP_SUFFIX = ".stamp.json"
LOCK_FRESH_S = 5.0
STOP_S = 30.0  # a stopped analyze gets this long to clean up before it is killed
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
#: Summed per bucket (``refsec_*`` are shares and are not summed).
RANK_METRICS = (
    "keep_top10",
    "keep_top20",
    "skip_top10",
    "skip_top20",
    "keep_15",
    "keep_30",
    "ref_15",
    "ref_30",
)
REPORT_METRICS = (
    "keep_top10",
    "keep_top20",
    "skip_top10",
    "skip_top20",
    "keep_15",
    "keep_30",
    "ref_30",
)
SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
]


class ManifestError(ValueError):
    """A manifest or holdout problem: exit 2."""


# -- manifest and labels -----------------------------------------------------------------------


def to_seconds(value: Any, where: str) -> Any:
    """``value`` in seconds when it is a time string (``"75.5"``, ``"1:15.5"``, ``"0:01:15"``);
    anything else is returned as is for :func:`moment_eval.parse_labels` to check."""
    if not isinstance(value, str):
        return value
    parts = value.strip().split(":")
    try:
        if len(parts) > 3 or not all(parts):
            raise ValueError
        nums = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"{where}: {value!r} is not seconds, mm:ss or h:mm:ss") from None
    if any(n < 0 for n in nums) or any(n >= 60 for n in nums[1:]):
        raise ValueError(f"{where}: {value!r} is not seconds, mm:ss or h:mm:ss")
    total = 0.0
    for n in nums:
        total = total * 60 + n
    return total


def convert_times(data: dict[str, Any], where: str) -> dict[str, Any]:
    """A copy of label ``data`` with time strings in ``duration`` and the ranges as seconds."""
    out = dict(data)
    if "duration" in out:
        out["duration"] = to_seconds(out["duration"], where)
    for key in ("keep", "skip", "reference"):
        items = out.get(key)
        if isinstance(items, list):
            out[key] = [
                [to_seconds(x, f"{where}: {key}") for x in item] if isinstance(item, list) else item
                for item in items
            ]
    return out


def load_label_file(path: Path) -> tuple[dict[str, Any], bool | None]:
    """A label file as :func:`moment_eval.parse_labels` checks it, and its own ``holdout``
    (None when it has none)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    own = data.get("holdout") if "holdout" in data else None
    return me.parse_labels(convert_times(data, str(path)), str(path)), own


@dataclass
class Source:
    id: str
    input: Path
    category: str
    bucket: str
    holdout: bool
    labels_path: Path | None = None
    labels: dict[str, Any] | None = None
    note: str = ""


@dataclass
class Variant:
    name: str
    args: list[str] = field(default_factory=list)
    cutpoint: list[str] | None = None  # None: ``python -m cutpoint`` of this interpreter
    cache: Path | None = None  # None: the bench's shared cache

    def command(self) -> list[str]:
        return list(self.cutpoint) if self.cutpoint else [sys.executable, "-m", "cutpoint"]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "cutpoint": self.command(),
            "cache": str(self.cache) if self.cache else None,
        }


@dataclass
class Manifest:
    path: Path
    sources: list[Source]
    variants: list[Variant]
    holdout_sha256: str | None


def split_cmdline(text: str, windows: bool | None = None) -> list[str]:
    """``text`` split as a command line: POSIX rules, or on Windows (``windows`` None: this
    system) non-POSIX rules, which keep backslashes, with one pair of surrounding quotes taken
    off each token."""
    if windows is None:
        windows = os.name == "nt"
    if not windows:
        return shlex.split(text)
    out = []
    for token in shlex.split(text, posix=False):
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            token = token[1:-1]
        out.append(token)
    return out


def _argv(value: Any, where: str) -> list[str]:
    """A list of strings, or a string split as a command line (:func:`split_cmdline`)."""
    if isinstance(value, str):
        return split_cmdline(value)
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return list(value)
    raise ValueError(f"{where} must be a string or a list of strings")


def parse_variant(raw: Any, where: str, base: Path | None = None) -> Variant:
    """A variant; its ``cache`` is relative to ``base`` (the manifest's folder)."""
    if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
        raise ValueError(f"{where}: a variant needs a 'name'")
    name = raw["name"]
    if not NAME_RE.match(name):
        raise ValueError(f"{where}: variant name {name!r} must be letters, digits, '.', '_', '-'")
    cutpoint, cache = raw.get("cutpoint"), raw.get("cache")
    if cache is not None and (not isinstance(cache, str) or not cache):
        raise ValueError(f"{where}: cache must be a folder path")
    return Variant(
        name,
        _argv(raw.get("args") or [], f"{where}: args"),
        _argv(cutpoint, f"{where}: cutpoint") if cutpoint else None,
        paths.absolute((base or Path()) / cache) if cache else None,
    )


def variant_flag(text: str) -> Variant:
    """``--variant NAME:ARGS``."""
    name, sep, args = text.partition(":")
    if not sep:
        raise ValueError(f"--variant {text!r}: expected NAME:ARGS (ARGS may be empty)")
    return parse_variant({"name": name, "args": args}, "--variant")


def check_variants(variants: list[Variant]) -> list[Variant]:
    names = [v.name for v in variants]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"variant names must be unique: {', '.join(dup)}")
    return variants or [Variant("base")]


def load_manifest(path: Path) -> Manifest:
    """Read and check a manifest; every problem is listed in one :class:`ManifestError`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise ManifestError(f'{path}: expected a JSON object with "schema": 1')
    raw = data.get("sources")
    if not isinstance(raw, list) or not raw:
        raise ManifestError(f"{path}: 'sources' must list at least one source")
    base = path.parent
    errors: list[str] = []
    sources: list[Source] = []
    seen: set[str] = set()
    for i, s in enumerate(raw, 1):
        where = f"source {i}"
        if not isinstance(s, dict):
            errors.append(f"{where}: not an object")
            continue
        sid = s.get("id")
        if not isinstance(sid, str) or not NAME_RE.match(sid):
            errors.append(f"{where}: 'id' must be letters, digits, '.', '_', '-' (got {sid!r})")
            continue
        where = f"source {sid!r}"
        if sid in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(sid)
        if s.get("category") not in CATEGORIES:
            errors.append(
                f"{where}: category {s.get('category')!r} is not one of {', '.join(CATEGORIES)}"
            )
        if s.get("bucket") not in BUCKETS:
            errors.append(f"{where}: bucket {s.get('bucket')!r} is not one of {', '.join(BUCKETS)}")
        holdout = s.get("holdout", False)
        if not isinstance(holdout, bool):
            errors.append(f"{where}: 'holdout' must be true or false")
            holdout = False
        if not isinstance(s.get("input"), str) or not s["input"]:
            errors.append(f"{where}: 'input' (a file or folder) is required")
            continue
        inp = paths.absolute(base / s["input"])
        src = Source(sid, inp, s.get("category"), s.get("bucket"), holdout)
        src.note = str(s.get("note") or "")
        if not src.input.exists():
            errors.append(f"{where}: input {src.input} does not exist")
        if s.get("labels"):
            src.labels_path = paths.absolute(base / s["labels"])
            try:
                src.labels, own = load_label_file(src.labels_path)
            except (OSError, ValueError) as err:
                errors.append(f"{where}: {err}")
            else:
                if own is not None and own != holdout:
                    errors.append(
                        f"{where}: the label file says holdout {str(own).lower()}, the manifest "
                        f"{str(holdout).lower()}"
                    )
                src.labels["holdout"] = holdout
        sources.append(src)
    variants: list[Variant] = []
    for i, v in enumerate(data.get("variants") or [], 1):
        try:
            variants.append(parse_variant(v, f"variant {i}", base))
        except ValueError as err:
            errors.append(str(err))
    try:
        variants = check_variants(variants)
    except ValueError as err:
        errors.append(str(err))
    hash_ = data.get("holdout_sha256")
    if hash_ is not None and not isinstance(hash_, str):
        errors.append("'holdout_sha256' must be a string")
    if errors:
        raise ManifestError(f"{path}:\n  " + "\n  ".join(errors))
    return Manifest(path, sources, variants, hash_)


def holdout_hash(sources: Sequence[Source]) -> str:
    """sha256 of the sorted holdout ids joined by newlines."""
    ids = sorted(s.id for s in sources if s.holdout)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def check_holdout_hash(m: Manifest) -> None:
    """Refuse a manifest whose holdout ids do not give its ``holdout_sha256``."""
    got = holdout_hash(m.sources)
    if m.holdout_sha256 is None:
        if any(s.holdout for s in m.sources):
            raise ManifestError(
                f"{m.path}: holdout sources but no 'holdout_sha256'; add "
                f'"holdout_sha256": "{got}"'
            )
        return
    if m.holdout_sha256.lower() != got:
        raise ManifestError(
            f"{m.path}: the holdout sources changed (holdout_sha256 {m.holdout_sha256}, the "
            f"holdout ids give {got}). If the change is intended, run with --rehash and store "
            "the hash it prints"
        )


def check_outputs(dirs: Sequence[Path], sources: Sequence[Source]) -> None:
    """Refuse a bench folder (or home) that is a file source's folder or inside a folder
    source: the sources' folders are read-only."""
    for d in dirs:
        out = d.resolve()
        for s in sources:
            inp = s.input.resolve()
            home = inp if inp.is_dir() else inp.parent
            if out == home or (inp.is_dir() and out.is_relative_to(inp)):
                raise ManifestError(f"{d} is in the folder of source {s.id!r}; use --bench")


# -- lock --------------------------------------------------------------------------------------


class BenchBusyError(RuntimeError):
    pass


def machine_lock() -> Path:
    """The machine's bench lock: one bench job at a time, whatever the bench folder."""
    return Path(tempfile.gettempdir()) / LOCK_NAME


@contextlib.contextmanager
def bench_lock(lock: Path) -> Iterator[Path]:
    """Hold ``lock`` (the pid, created exclusively) for the whole run. A lock whose process is
    gone is replaced; :func:`cutpoint.procs.pid_alive` checks it (OpenProcess on Windows, where
    ``os.kill`` would end the process). A lock without a pid that is younger than
    :data:`LOCK_FRESH_S` is being written: busy."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                pid = int(lock.read_text(encoding="utf-8").strip() or 0)
                mtime = lock.stat().st_mtime
            except (OSError, ValueError):
                pid, mtime = 0, None
            fresh = mtime is not None and time.time() - mtime < LOCK_FRESH_S
            if (pid and procs.pid_alive(pid, not_after=mtime)) or (not pid and fresh):
                raise BenchBusyError(
                    f"another bench run (pid {pid or '?'}) holds {lock}; wait for it, or "
                    "delete the file if no bench is running"
                ) from None
            logger.warning("Replacing the stale lock %s (pid %s)", lock, pid or "?")
            with contextlib.suppress(FileNotFoundError):
                lock.unlink()
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        break
    else:
        raise BenchBusyError(f"could not take {lock}")
    try:
        yield lock
    finally:
        with contextlib.suppress(OSError):
            if lock.read_text(encoding="utf-8").strip() == str(os.getpid()):
                lock.unlink()


# -- samples -----------------------------------------------------------------------------------


def sample_gpu() -> list[dict[str, Any]] | None:
    """One row per GPU from nvidia-smi; None when it is missing or fails."""
    try:
        out = subprocess.run(
            SMI_QUERY, capture_output=True, text=True, timeout=10, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in out.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != 4:
            continue
        try:
            idx, used, total, util = (int(c) for c in cells)
        except ValueError:  # [N/A]
            continue
        rows.append(
            {"index": idx, "memory_used_mib": used, "memory_total_mib": total, "util_pct": util}
        )
    return rows or None


def sample_ollama(url: str | None) -> list[dict[str, Any]] | None:
    """The models Ollama at ``url`` has loaded (``/api/ps``); None when it does not answer."""
    if not url:
        return None
    try:
        resp = requests.get(f"{url}/api/ps", timeout=3, allow_redirects=False)
        if resp.status_code != 200:
            return None
        models = resp.json().get("models", [])
    except (requests.RequestException, ValueError, AttributeError):
        return None
    if not isinstance(models, list):
        return None
    return [
        {
            "name": m.get("name") or m.get("model"),
            "size_vram": m.get("size_vram"),
            "size": m.get("size"),
            "context_length": m.get("context_length"),
        }
        for m in models
        if isinstance(m, dict)
    ]


def take_sample(t: float, ollama_url: str | None) -> dict[str, Any]:
    return {"t": round(t, 1), "gpu": sample_gpu(), "ollama": sample_ollama(ollama_url)}


def _stats(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"min": min(values), "max": max(values), "mean": round(statistics.fmean(values), 1)}


def summarize_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Min, max and mean per GPU and of Ollama's loaded VRAM, and how often each answered."""
    gpus: dict[int, dict[str, list[float]]] = {}
    vram: list[float] = []
    models: set[str] = set()
    for s in samples:
        for g in s.get("gpu") or []:
            d = gpus.setdefault(g["index"], {"used": [], "util": [], "total": []})
            d["used"].append(g["memory_used_mib"])
            d["util"].append(g["util_pct"])
            d["total"].append(g["memory_total_mib"])
        if s.get("ollama") is not None:
            vram.append(sum((m.get("size_vram") or 0) for m in s["ollama"]) / 2**20)
            models |= {m["name"] for m in s["ollama"] if m.get("name")}
    return {
        "samples": len(samples),
        "gpu_available": sum(s.get("gpu") is not None for s in samples),
        "ollama_available": sum(s.get("ollama") is not None for s in samples),
        "gpu": {
            str(i): {
                "memory_used_mib": _stats(d["used"]),
                "util_pct": _stats(d["util"]),
                "memory_total_mib": max(d["total"]),
            }
            for i, d in sorted(gpus.items())
        },
        "ollama_vram_mib": _stats([round(v) for v in vram]),
        "ollama_models": sorted(models),
    }


class Sampler:
    """Samples every ``interval`` seconds on a background thread while a run goes (the first at
    once); ``interval`` <= 0 takes none."""

    def __init__(self, interval: float, ollama_url: str | None) -> None:
        self.interval = interval
        self.ollama_url = ollama_url
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        t0 = time.monotonic()
        while True:
            self.samples.append(take_sample(time.monotonic() - t0, self.ollama_url))
            if self._stop.wait(self.interval):
                return

    def __enter__(self) -> Sampler:
        if self.interval > 0:
            self._thread = threading.Thread(target=self._loop, name="bench-sampler", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)

    def result(self) -> dict[str, Any]:
        return {"raw": list(self.samples), "summary": summarize_samples(self.samples)}


# -- runner ------------------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def git_state() -> str | None:
    """HEAD of this checkout, with ``+dirty.<hash>`` of the uncommitted changes under ``src``
    (the status, the diff and the bytes of the untracked files)."""
    head = _git("rev-parse", "HEAD")
    if not head:
        return None
    status = _git("status", "--porcelain", "--", "src") or ""
    if not status.strip():
        return head.strip()
    h = hashlib.sha256((status + (_git("diff", "HEAD", "--", "src") or "")).encode("utf-8"))
    untracked = _git("ls-files", "--others", "--exclude-standard", "-z", "--", "src") or ""
    for name in sorted(n for n in untracked.split("\0") if n):
        h.update(name.encode("utf-8") + b"\0")
        with contextlib.suppress(OSError):
            h.update((REPO / name).read_bytes())
    return f"{head.strip()}+dirty.{h.hexdigest()[:12]}"


def identify(variant: Variant, env: dict[str, str]) -> dict[str, Any]:
    """The cutpoint a variant runs: command, version, git state (this checkout only) and
    edition."""
    if variant.cutpoint is None:
        return {
            "cutpoint": variant.command(),
            "version": __version__,
            "git": git_state(),
            "edition": extras.edition(),
        }
    cmd = variant.command()
    version = edition = "unknown"
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        out = subprocess.run(
            [*cmd, "--version"], env=env, capture_output=True, text=True, timeout=300, check=True
        ).stdout
        version = out.strip().removeprefix("cutpoint").strip() or "unknown"
    with contextlib.suppress(OSError, subprocess.SubprocessError, ValueError, AttributeError):
        out = subprocess.run(
            [*cmd, "capabilities", "--json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        ).stdout
        edition = json.loads(out)["outputs"]["edition"]
    return {"cutpoint": cmd, "version": version, "git": None, "edition": edition}


def input_fingerprint(path: Path) -> list[dict[str, Any]]:
    """Name, size and modification time of the input file, or of every file under a folder."""
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    out = []
    for f in files:
        st = f.stat()
        rel = f.name if f == path else f.relative_to(path).as_posix()
        out.append({"name": rel, "size": st.st_size, "mtime": round(st.st_mtime, 3)})
    return out


def file_sha256(path: Path) -> str | None:
    """sha256 of a file's bytes; None when it does not exist or cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def job_stamp(
    identity: dict[str, Any], variant: Variant, source: Source, cache: Path
) -> dict[str, Any]:
    """What makes a run reusable: the cutpoint, its settings (the config file the run reads,
    the cache) and the input."""
    return {
        "cutpoint": identity["cutpoint"],
        "version": identity["version"],
        "git": identity["git"],
        "edition": identity["edition"],
        "args": variant.args,
        "cache": str(cache),
        "config_sha256": file_sha256(paths.config_file()),
        "inputs": input_fingerprint(source.input),
    }


def analyze_argv(variant: Variant, source: Source, project: Path) -> list[str]:
    return [
        *variant.command(),
        "analyze",
        str(source.input),
        "--style",
        "auto",
        "--out",
        str(project),
        "--json",
        *variant.args,
    ]


def bench_env(bench: Path, home: Path | None = None, cache: Path | None = None) -> dict[str, str]:
    """The variables every run gets: a bench home, the shared bench cache (a variant's own
    ``cache`` replaces it per run), and the user's models folder (as resolved before the home
    changes) unless one is set."""
    out = {
        "CUTPOINT_HOME": str((home or bench / "home").resolve()),
        "CUTPOINT_CACHE_DIR": str((cache or bench / "cache").resolve()),
    }
    if not os.environ.get("CUTPOINT_MODELS_DIR"):
        out["CUTPOINT_MODELS_DIR"] = str(paths.models_dir())
    return out


def run_job(
    variant: Variant,
    source: Source,
    project: Path,
    identity: dict[str, Any],
    *,
    cache: Path,
    rerun: bool,
    interval: float,
    ollama_url: str | None,
) -> dict[str, Any]:
    """Analyze one source for one variant with ``cache`` as its cache (or reuse the matching
    earlier run). A command that cannot start is a failed job; a Ctrl+C (or any other
    exception) stops analyze (:func:`stop_analyze`) and is raised again."""
    stamp_path = project.with_name(project.name + STAMP_SUFFIX)
    stamp = job_stamp(identity, variant, source, cache)
    if not rerun and (project / "highlights.json").is_file():
        with contextlib.suppress(OSError, ValueError):
            if json.loads(stamp_path.read_text(encoding="utf-8")) == stamp:
                logger.info("%s / %s: reusing %s", variant.name, source.id, project)
                return {"status": "reused", "project": str(project), "samples": None}
    argv = analyze_argv(variant, source, project)
    project.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        stamp_path.unlink()  # a stale stamp must not survive a failed run
    logger.info("%s / %s: %s", variant.name, source.id, shlex.join(argv))
    env = {**os.environ, "CUTPOINT_CACHE_DIR": str(cache)}
    t0 = time.monotonic()
    with Sampler(interval, ollama_url) as sampler:
        try:
            proc = subprocess.Popen(
                argv,
                env=env,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as err:
            logger.error("%s / %s: cannot start %s: %s", variant.name, source.id, argv[0], err)
            return {
                "status": "failed",
                "project": str(project),
                "returncode": None,
                "seconds": 0.0,
                "error": {"code": "start_failed", "message": str(err)},
                "warnings": [],
                "samples": None,
            }
        try:
            stdout, _ = proc.communicate()
        except BaseException as err:
            stop_analyze(proc, err)
            raise
    seconds = round(time.monotonic() - t0, 1)
    envelope = parse_envelope(stdout)
    ok = proc.returncode == 0 and (project / "highlights.json").is_file()
    out = {
        "status": "ok" if ok else "failed",
        "project": str(project),
        "returncode": proc.returncode,
        "seconds": seconds,
        "error": envelope.get("error") or None,
        "warnings": sorted(
            {
                w["code"]
                for w in envelope.get("warnings") or []
                if isinstance(w, dict) and w.get("code")
            }
        ),
        "samples": sampler.result(),
    }
    if ok:
        stamp_path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    else:
        logger.error(
            "%s / %s: analyze failed (exit %s)%s",
            variant.name,
            source.id,
            proc.returncode,
            f": {out['error']}" if out["error"] else "",
        )
    return out


def stop_analyze(proc: subprocess.Popen[str], err: BaseException) -> None:
    """Let a stopped analyze clean up (its lock, ``progress.json``, its own ffmpeg): after a
    Ctrl+C the child got the same signal, so it is waited for; after anything else it is asked
    to stop (terminate). If it is still running after :data:`STOP_S` it is killed. A second
    Ctrl+C during the wait kills it at once."""
    try:
        if not isinstance(err, KeyboardInterrupt):
            proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(STOP_S)
    finally:
        if proc.poll() is None:
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(STOP_S)


def parse_envelope(stdout: str | None) -> dict[str, Any]:
    """The ``--json`` envelope (the last stdout line); {} when there is none."""
    lines = (stdout or "").strip().splitlines()
    try:
        data = json.loads(lines[-1]) if lines else {}
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


# -- scoring -----------------------------------------------------------------------------------


def score_detection(source: Source, hl: dict[str, Any]) -> dict[str, Any] | None:
    """The project's ``detection`` record against the labelled category; None without one."""
    det = hl.get("detection")
    if not isinstance(det, dict) or det.get("category") is None:
        return None
    ambiguous = bool(det.get("ambiguous"))
    right = det["category"] == source.category
    verdict = "ambiguous" if ambiguous else "right" if right else "sure_wrong"
    return {
        "category": det["category"],
        "ambiguous": ambiguous,
        "verdict": verdict,
        "style": det.get("style"),
        "fallback": det.get("fallback"),
        "decided_by": det.get("decided_by"),
        "reason": det.get("reason"),
        "sources": det.get("sources"),
    }


def score_source(
    source: Source, project: Path, runs: int, seed: int, min_overlap: float
) -> dict[str, Any]:
    """Detection and (with labels) ranking of one analyzed project."""
    out: dict[str, Any] = {"detection": None, "ranking": None, "errors": []}
    try:
        hl = json.loads((project / "highlights.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        out["errors"].append(f"highlights.json: {err}")
        return out
    out["detection"] = score_detection(source, hl)
    if out["detection"] is None:
        out["errors"].append("highlights.json has no detection record")
    if source.labels is not None:
        try:
            out["ranking"] = me.evaluate_video(
                project, source.labels, runs, seed, min_overlap, None
            )
        except (OSError, ValueError, KeyError) as err:
            out["errors"].append(f"ranking: {err}")
    return out


def source_entry(source: Source) -> dict[str, Any]:
    return {
        "id": source.id,
        "input": str(source.input),
        "category": source.category,
        "bucket": source.bucket,
        "holdout": source.holdout,
        "labels": str(source.labels_path) if source.labels_path else None,
        "labels_sha256": file_sha256(source.labels_path) if source.labels_path else None,
    }


def detection_summary(sources: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per bucket (and ``all``): right, ambiguous and sure-wrong counts and rates, the
    confusion (detected category -> sure, ambiguous) and the sources without a detection."""
    out: dict[str, dict[str, Any]] = {}
    for s in sources:
        for key in (s["bucket"], ALL):
            b = out.setdefault(
                key,
                {
                    "sources": 0,
                    "right": 0,
                    "ambiguous": 0,
                    "sure_wrong": [],
                    "missing": [],
                    "confusion": {},
                },
            )
            d = s.get("detection")
            if d is None:
                b["missing"].append(s["id"])
                continue
            b["sources"] += 1
            cell = b["confusion"].setdefault(d["category"], {"sure": 0, "ambiguous": 0})
            cell["ambiguous" if d["ambiguous"] else "sure"] += 1
            if d["verdict"] == "right":
                b["right"] += 1
            elif d["verdict"] == "ambiguous":
                b["ambiguous"] += 1
            else:
                b["sure_wrong"].append(s["id"])
    for b in out.values():
        n = b["sources"]
        b["accuracy"] = b["right"] / n if n else None
        b["ambiguous_rate"] = b["ambiguous"] / n if n else None
    return out


def _kind(metric: str) -> str:
    return "reference" if metric.startswith("ref") else metric.split("_")[0]


def ranking_summary(sources: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per bucket (and ``all``): each metric summed over the labelled videos, the labels it is
    out of, and how many videos beat the random control."""
    out: dict[str, dict[str, Any]] = {}
    for s in sources:
        r = s.get("ranking")
        if r is None:
            continue
        for key in (s["bucket"], ALL):
            b = out.setdefault(key, {"videos": 0, "metrics": {}})
            b["videos"] += 1
            for name in RANK_METRICS:
                m = r["metrics"].get(name)
                if m is None:
                    continue
                agg = b["metrics"].setdefault(
                    name, {"value": 0.0, "of": 0, "videos": 0, "beats_noise": 0}
                )
                agg["value"] += m["value"]
                agg["of"] += r["labels"][_kind(name)]
                agg["videos"] += 1
                agg["beats_noise"] += bool(m["beats_noise"])
    return out


def summarize(sources: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Detection and ranking summaries of the tune split, and of the holdout when present."""
    tune = [s for s in sources if not s["holdout"]]
    held = [s for s in sources if s["holdout"]]
    out = {"tune": {"detection": detection_summary(tune), "ranking": ranking_summary(tune)}}
    if held:
        out["holdout"] = {"detection": detection_summary(held), "ranking": ranking_summary(held)}
    return out


# -- compare and gate --------------------------------------------------------------------------


@dataclass
class GateRules:
    ambiguous_pp: float = 5.0  # a bucket's ambiguous rate may rise this many points
    keep_drop: float = 1.0  # keep_top20 / keep_30 may drop this many ranges ...
    keep_drop_share: float = 0.10  # ... or this share of the baseline, whichever is more
    skip_rise: float = 1.0  # skip_top20 may rise this many ranges


def compare_issues(a: dict[str, Any], b: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(warnings, refusals): settings that make the runs less comparable, and those that rule
    out the gate (deltas are still reported)."""
    warnings, refusals = [], []
    for key in ("kind", "schema"):
        if a.get(key) != b.get(key):
            refusals.append(f"{key} differs: {a.get(key)!r} -> {b.get(key)!r}")
    if a.get("edition") != b.get("edition"):
        refusals.append(
            f"edition differs: {a.get('edition')} -> {b.get('edition')}; compare lite against a "
            "lite baseline"
        )
    sa, sb = a.get("settings") or {}, b.get("settings") or {}
    if sa.get("min_overlap") != sb.get("min_overlap"):
        refusals.append(f"min_overlap differs: {sa.get('min_overlap')} -> {sb.get('min_overlap')}")
    for key in ("seed", "runs"):
        if sa.get(key) != sb.get(key):
            warnings.append(f"{key} differs: {sa.get(key)} -> {sb.get(key)} (beats noise shifts)")
    return warnings, refusals


#: Per source, what must be the same in both runs for them to be compared.
LABEL_KEYS = ("category", "bucket", "holdout", "labels_sha256")


def label_mismatches(a: dict[str, Any], b: dict[str, Any]) -> dict[str, list[str]]:
    """Per source in both runs, the :data:`LABEL_KEYS` that differ (labelled differently)."""
    by_id = {s["id"]: s for s in a["sources"]}
    out = {}
    for s in b["sources"]:
        o = by_id.get(s["id"])
        if o is not None:
            keys = [k for k in LABEL_KEYS if o.get(k) != s.get(k)]
            if keys:
                out[s["id"]] = keys
    return out


def _split_pairs(
    a: dict[str, Any], b: dict[str, Any], holdout: bool, skip: Sequence[str] = ()
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_id = {s["id"]: s for s in a["sources"]}
    return [
        (by_id[s["id"]], s)
        for s in b["sources"]
        if s["id"] in by_id and s["id"] not in skip and s["holdout"] == holdout
    ]


def split_delta(pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """Detection and ranking summaries of A and B over the sources both scored, per source
    changes, and the sources B lost."""
    det_pairs = [(x, y) for x, y in pairs if x.get("detection") and y.get("detection")]
    rank_pairs = [(x, y) for x, y in pairs if x.get("ranking") and y.get("ranking")]
    lost = [
        y["id"]
        for x, y in pairs
        if (x.get("detection") and not y.get("detection"))
        or (x.get("ranking") and not y.get("ranking"))
    ]
    rows = []
    for x, y in pairs:
        row: dict[str, Any] = {"id": y["id"], "bucket": y["bucket"]}
        dx, dy = x.get("detection"), y.get("detection")
        if (dx and dx["category"], dx and dx["ambiguous"]) != (
            dy and dy["category"],
            dy and dy["ambiguous"],
        ):
            row["detection"] = {"old": _det_text(dx), "new": _det_text(dy)}
        rx, ry = x.get("ranking"), y.get("ranking")
        if rx and ry:
            deltas = {
                n: ry["metrics"][n]["value"] - rx["metrics"][n]["value"]
                for n in RANK_METRICS
                if n in rx["metrics"] and n in ry["metrics"]
            }
            deltas = {n: d for n, d in deltas.items() if d}
            if deltas:
                row["ranking"] = deltas
        if len(row) > 2:
            rows.append(row)
    return {
        "detection": {
            "a": detection_summary([x for x, _ in det_pairs]),
            "b": detection_summary([y for _, y in det_pairs]),
        },
        "ranking": {
            "a": ranking_summary([x for x, _ in rank_pairs]),
            "b": ranking_summary([y for _, y in rank_pairs]),
        },
        "sources": rows,
        "lost": lost,
    }


def _det_text(d: dict[str, Any] | None) -> str:
    if d is None:
        return "none"
    return f"{d['category']}{'?' if d['ambiguous'] else ''}"


def gate(delta: dict[str, Any], rules: GateRules) -> list[str]:
    """The failed rules of the tune split (empty: pass)."""
    fails = [f"{sid}: scored in A, failed or unscored in B" for sid in delta["lost"]]
    da, db = delta["detection"]["a"], delta["detection"]["b"]
    old_wrong = set(da.get(ALL, {}).get("sure_wrong", []))
    fails += [
        f"{sid}: new sure-wrong detection"
        for sid in db.get(ALL, {}).get("sure_wrong", [])
        if sid not in old_wrong
    ]
    for bucket, bb in db.items():
        ba = da.get(bucket)
        if bucket == ALL or ba is None or bb["ambiguous_rate"] is None:
            continue
        rise = (bb["ambiguous_rate"] - (ba["ambiguous_rate"] or 0)) * 100
        if rise > rules.ambiguous_pp:
            fails.append(
                f"{bucket}: ambiguous rate up {rise:.1f} points (limit {rules.ambiguous_pp:g})"
            )
    ra, rb = delta["ranking"]["a"], delta["ranking"]["b"]
    for bucket, bb in rb.items():
        ba = ra.get(bucket)
        if bucket == ALL or ba is None:
            continue
        ma, mb = ba["metrics"], bb["metrics"]
        for name in ("keep_top20", "keep_30"):
            if name in ma and name in mb:
                drop = ma[name]["value"] - mb[name]["value"]
                allowed = max(rules.keep_drop, rules.keep_drop_share * ma[name]["value"])
                if drop > allowed:
                    fails.append(f"{bucket}: {name} down {drop:g} (allowed {allowed:g})")
        if "skip_top20" in ma and "skip_top20" in mb:
            rise = mb["skip_top20"]["value"] - ma["skip_top20"]["value"]
            if rise > rules.skip_rise:
                fails.append(f"{bucket}: skip_top20 up {rise:g} (allowed {rules.skip_rise:g})")
        if "keep_30" in ma and "keep_30" in mb:
            lost = ma["keep_30"]["beats_noise"] - mb["keep_30"]["beats_noise"]
            if lost > 0:
                fails.append(f"{bucket}: keep_30 beats noise in {lost} fewer video(s)")
    return fails


def compare_results(
    a: dict[str, Any], b: dict[str, Any], rules: GateRules, holdout: bool = False
) -> dict[str, Any]:
    """B against A: deltas of the tune split (and with ``holdout`` of the holdout, when both
    runs viewed it), and the gate verdict of the tune split (``refused`` when the runs are not
    comparable). A source labelled differently in the two runs (:func:`label_mismatches`) is
    left out of both, with a warning; without ``holdout`` a source that is holdout in either run
    is left out silently, so no holdout id is named."""
    warnings, refusals = compare_issues(a, b)
    skip = label_mismatches(a, b)
    held = {s["id"] for r in (a, b) for s in r["sources"] if s.get("holdout")}
    warnings += [
        f"{sid}: {', '.join(keys)} differ; left out of the comparison and the gate"
        for sid, keys in skip.items()
        if holdout or sid not in held
    ]
    tune = split_delta(_split_pairs(a, b, False, list(skip)))
    out: dict[str, Any] = {
        "a": _run_name(a),
        "b": _run_name(b),
        "warnings": warnings,
        "refusals": refusals,
        "tune": tune,
    }
    if holdout and a.get("holdout_viewed") and b.get("holdout_viewed"):
        out["holdout"] = split_delta(_split_pairs(a, b, True, list(skip)))
    if refusals:
        out["gate"] = {"verdict": "refused", "failures": []}
    else:
        fails = gate(tune, rules)
        out["gate"] = {"verdict": "fail" if fails else "pass", "failures": fails}
    return out


def _run_name(r: dict[str, Any]) -> str:
    return f"{(r.get('variant') or {}).get('name', '?')} ({r.get('created', '?')})"


# -- report ------------------------------------------------------------------------------------


def _n(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.0f}%"


def _detection_table(det: dict[str, Any]) -> list[str]:
    cats = list(CATEGORIES)
    lines = [
        "| bucket | sources | right | ambiguous | sure-wrong | accuracy | ambiguous rate | "
        + " | ".join(cats)
        + " | no detection |",
        "|---" * (8 + len(cats)) + "|",
    ]
    for bucket in [*[b for b in BUCKETS if b in det], ALL] if det else []:
        b = det[bucket]
        cells = []
        for c in cats:
            cell = b["confusion"].get(c, {"sure": 0, "ambiguous": 0})
            amb = f" +{cell['ambiguous']}?" if cell["ambiguous"] else ""
            cells.append(f"{cell['sure']}{amb}")
        lines.append(
            f"| {bucket} | {b['sources']} | {b['right']} | {b['ambiguous']} | "
            f"{len(b['sure_wrong'])} | {_pct(b['accuracy'])} | {_pct(b['ambiguous_rate'])} | "
            + " | ".join(cells)
            + f" | {len(b['missing'])} |"
        )
    wrong = det.get(ALL, {}).get("sure_wrong", [])
    if wrong:
        lines.append("")
        lines.append("Sure-wrong: " + ", ".join(wrong))
    missing = det.get(ALL, {}).get("missing", [])
    if missing:
        lines.append("")
        lines.append("No detection (failed run or no record): " + ", ".join(missing))
    return lines


def _ranking_table(rank: dict[str, Any]) -> list[str]:
    if not rank:
        return ["No labelled source was scored."]
    lines = [
        "| bucket | videos | " + " | ".join(REPORT_METRICS) + " | keep_30 beats noise |",
        "|---" * (3 + len(REPORT_METRICS)) + "|",
    ]
    for bucket in [*[b for b in BUCKETS if b in rank], ALL]:
        b = rank[bucket]
        cells = []
        for name in REPORT_METRICS:
            m = b["metrics"].get(name)
            cells.append(f"{_n(m['value'])}/{m['of']}" if m else "-")
        k30 = b["metrics"].get("keep_30")
        noise = f"{k30['beats_noise']}/{k30['videos']}" if k30 else "-"
        lines.append(f"| {bucket} | {b['videos']} | " + " | ".join(cells) + f" | {noise} |")
    return lines


def _delta_lines(delta: dict[str, Any]) -> list[str]:
    lines = ["Detection (sources scored in both):", ""]
    lines += [
        "| bucket | accuracy | ambiguous rate | sure-wrong |",
        "|---|---|---|---|",
    ]
    da, db = delta["detection"]["a"], delta["detection"]["b"]
    for bucket in [*[b for b in BUCKETS if b in db], ALL] if db else []:
        x, y = da[bucket], db[bucket]
        lines.append(
            f"| {bucket} | {_pct(x['accuracy'])} -> {_pct(y['accuracy'])} | "
            f"{_pct(x['ambiguous_rate'])} -> {_pct(y['ambiguous_rate'])} | "
            f"{len(x['sure_wrong'])} -> {len(y['sure_wrong'])} |"
        )
    lines += ["", "Ranking (labelled sources scored in both; A -> B (delta)):", ""]
    ra, rb = delta["ranking"]["a"], delta["ranking"]["b"]
    if rb:
        lines += [
            "| bucket | " + " | ".join(REPORT_METRICS) + " |",
            "|---" * (1 + len(REPORT_METRICS)) + "|",
        ]
        for bucket in [*[b for b in BUCKETS if b in rb], ALL]:
            cells = []
            for name in REPORT_METRICS:
                x, y = ra[bucket]["metrics"].get(name), rb[bucket]["metrics"].get(name)
                cells.append(
                    f"{_n(x['value'])} -> {_n(y['value'])} ({y['value'] - x['value']:+g})"
                    if x and y
                    else "-"
                )
            lines.append(f"| {bucket} | " + " | ".join(cells) + " |")
    else:
        lines.append("No labelled source scored in both.")
    if delta["lost"]:
        lines += ["", "Scored in A, not in B: " + ", ".join(delta["lost"])]
    lines += ["", "Per source:", ""]
    if not delta["sources"]:
        lines.append("- no source changed")
    for row in delta["sources"]:
        parts = []
        if "detection" in row:
            parts.append(f"detection {row['detection']['old']} -> {row['detection']['new']}")
        parts += [f"{n} {d:+g}" for n, d in row.get("ranking", {}).items()]
        lines.append(f"- {row['id']} ({row['bucket']}): " + ", ".join(parts))
    return lines


def compare_lines(cmp: dict[str, Any]) -> list[str]:
    lines = [f"## {cmp['b']} vs {cmp['a']}", ""]
    for w in cmp["warnings"]:
        lines.append(f"- WARNING: {w}")
    for r in cmp["refusals"]:
        lines.append(f"- WARNING: {r}; deltas only, no gate")
    if cmp["warnings"] or cmp["refusals"]:
        lines.append("")
    lines += ["### Tune split", "", *_delta_lines(cmp["tune"]), ""]
    if "holdout" in cmp:
        lines += [
            "### HOLDOUT (checking only; do not tune on it; not gated)",
            "",
            *_delta_lines(cmp["holdout"]),
            "",
        ]
    g = cmp["gate"]
    verdict = {"pass": "PASS", "fail": "FAIL", "refused": "NOT GIVEN (runs not comparable)"}
    lines += [f"### Gate (advice, tune split only): {verdict[g['verdict']]}", ""]
    lines += [f"- {f}" for f in g["failures"]]
    return lines


def _resources(results: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "| variant | source | status | seconds | GPU MiB max | GPU util mean | Ollama VRAM MiB max | Ollama models |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        for s in r["sources"]:
            run = s.get("run") or {}
            summ = (run.get("samples") or {}).get("summary") or {}
            gpus = summ.get("gpu") or {}
            used = [g["memory_used_mib"]["max"] for g in gpus.values() if g["memory_used_mib"]]
            util = [g["util_pct"]["mean"] for g in gpus.values() if g["util_pct"]]
            vram = summ.get("ollama_vram_mib")
            lines.append(
                f"| {r['variant']['name']} | {s['id']} | {run.get('status', '-')} | "
                f"{run.get('seconds', '-')} | {max(used) if used else '-'} | "
                f"{max(util) if util else '-'} | {vram['max'] if vram else '-'} | "
                f"{', '.join(summ.get('ollama_models') or []) or '-'} |"
            )
    return lines


def render_report(results: Sequence[dict[str, Any]], comparisons: Sequence[dict[str, Any]]) -> str:
    first = results[0]
    lines = [f"# Cutpoint bench {first['created']}", ""]
    lines += [f"Manifest: `{first['manifest']}`", ""]
    lines += [
        "| variant | cutpoint | version | git | edition | args |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        c = r["cutpoint"]
        lines.append(
            f"| {r['variant']['name']} | `{shlex.join(c['cutpoint'])}` | {c['version']} | "
            f"{c['git'] or '-'} | {r['edition']} | `{shlex.join(r['variant']['args'])}` |"
        )
    if first["holdout_viewed"]:
        lines += ["", "**Holdout viewed** (`--holdout`): for the release check only."]
    for r in results:
        name = r["variant"]["name"]
        tune = r["summary"]["tune"]
        lines += ["", f"## {name}: detection (tune)", "", *_detection_table(tune["detection"])]
        lines += ["", f"## {name}: ranking (tune)", "", *_ranking_table(tune["ranking"])]
        held = r["summary"].get("holdout")
        if held:
            lines += [
                "",
                f"## {name}: HOLDOUT (checking only; do not tune on it)",
                "",
                *_detection_table(held["detection"]),
                "",
                *_ranking_table(held["ranking"]),
            ]
        errs = [(s["id"], e) for s in r["sources"] for e in s.get("errors") or []]
        if errs:
            lines += ["", f"Problems ({name}):", ""] + [f"- {i}: {e}" for i, e in errs]
    for cmp in comparisons:
        lines += ["", *compare_lines(cmp)]
    lines += ["", "## Resources (sampled while each run went)", "", *_resources(results)]
    return "\n".join(lines) + "\n"


# -- main --------------------------------------------------------------------------------------


def run_bench(
    m: Manifest,
    bench: Path,
    variants: list[Variant],
    *,
    holdout: bool,
    rerun: bool,
    reuse: dict[str, Path],
    cache: Path,
    interval: float,
    ollama_url: str | None,
    runs: int,
    seed: int,
    min_overlap: float,
    argv: list[str],
) -> list[dict[str, Any]]:
    """Run (or reuse) and score every source for every variant; one results dict per variant.
    ``cache`` is the shared cache, for the variants without their own."""
    chosen = [s for s in m.sources if holdout or not s.holdout]
    created = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = []
    for variant in variants:
        vcache = variant.cache or cache
        identity = identify(variant, {**os.environ, "CUTPOINT_CACHE_DIR": str(vcache)})
        entries = []
        for source in chosen:
            if source.id in reuse:
                project = reuse[source.id]
                run = {"status": "given", "project": str(project), "samples": None}
            else:
                project = bench / "runs" / variant.name / source.id
                run = run_job(
                    variant,
                    source,
                    project,
                    identity,
                    cache=vcache,
                    rerun=rerun,
                    interval=interval,
                    ollama_url=ollama_url,
                )
            entry = source_entry(source) | {"run": run}
            if run["status"] == "failed":
                why = (run.get("error") or {}).get("message")
                err = f"analyze failed: {why}" if why else "analyze failed"
                entry |= {"detection": None, "ranking": None, "errors": [err]}
            else:
                entry |= score_source(source, project, runs, seed, min_overlap)
            entries.append(entry)
        out.append(
            {
                "kind": KIND,
                "schema": SCHEMA,
                "created": created,
                "manifest": str(m.path),
                "bench": str(bench),
                "variant": variant.to_json(),
                "cutpoint": identity,
                "edition": identity["edition"],
                "args": argv,
                "settings": {
                    "runs": runs,
                    "seed": seed,
                    "min_overlap": min_overlap,
                    "sample_interval_s": interval,
                    "cache": str(vcache),
                    "holdout": holdout,
                },
                "holdout_viewed": holdout,
                "sources": entries,
                "summary": summarize(entries),
            }
        )
    return out


def load_results(path: Path) -> dict[str, Any]:
    """A results file of this script.

    Raises:
        ValueError: Not one (or not JSON).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("kind") != KIND
        or not isinstance(data.get("sources"), list)
        or not all(isinstance(s, dict) and "id" in s for s in data["sources"])
    ):
        raise ValueError(f"{path}: not a results file of scripts/bench.py")
    return data


def _reuse_flags(values: Sequence[str], m: Manifest) -> dict[str, Path]:
    known = {s.id for s in m.sources}
    out = {}
    for v in values:
        sid, sep, folder = v.partition("=")
        if not sep or not folder:
            raise ValueError(f"--reuse {v!r}: expected ID=DIR")
        if sid not in known:
            raise ValueError(f"--reuse {v!r}: no source {sid!r} in the manifest")
        if not (Path(folder) / "highlights.json").is_file():
            raise ValueError(f"--reuse {v!r}: {folder} has no highlights.json")
        out[sid] = Path(folder)
    return out


def sampler_ollama_url() -> str | None:
    """Ollama's URL as the product finds it (``OLLAMA_HOST``, the bench home's config, the
    default); None when it is invalid."""
    from cutpoint import backends  # noqa: PLC0415

    try:
        return backends.ollama_url()
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest", type=Path, nargs="?", help="the bench manifest (JSON)")
    ap.add_argument("--bench", type=Path, help="bench folder (default: the manifest's folder)")
    ap.add_argument("--home", type=Path, help="CUTPOINT_HOME of the runs (default <bench>/home)")
    ap.add_argument(
        "--cache",
        type=Path,
        help="shared CUTPOINT_CACHE_DIR of the runs (default <bench>/cache; a variant's own "
        "'cache' replaces it)",
    )
    ap.add_argument("--holdout", action="store_true", help="also run and report the holdout")
    ap.add_argument(
        "--rehash", action="store_true", help="print the manifest's holdout hash and stop"
    )
    ap.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME:ARGS",
        help="a variant (replaces the manifest's); the first is the baseline",
    )
    ap.add_argument("--rerun", action="store_true", help="analyze again even if a run matches")
    ap.add_argument(
        "--reuse", action="append", default=[], metavar="ID=DIR", help="use this project folder"
    )
    ap.add_argument(
        "--sample-interval",
        type=float,
        default=DEFAULT_SAMPLE_S,
        help="seconds between resource samples (0: none; default 5)",
    )
    ap.add_argument("--compare", type=Path, nargs=2, metavar=("A", "B"), help="two results files")
    ap.add_argument("--runs", type=int, default=me.DEFAULT_RUNS, help="random orders (1000)")
    ap.add_argument("--seed", type=int, default=me.DEFAULT_SEED, help="control seed (7)")
    ap.add_argument("--min-overlap", type=float, default=me.DEFAULT_MIN_OVERLAP)
    rules = GateRules()
    ap.add_argument("--gate-ambiguous-pp", type=float, default=rules.ambiguous_pp)
    ap.add_argument("--gate-keep-drop", type=float, default=rules.keep_drop)
    ap.add_argument("--gate-keep-drop-share", type=float, default=rules.keep_drop_share)
    ap.add_argument("--gate-skip-rise", type=float, default=rules.skip_rise)
    ap.add_argument("--strict-gate", action="store_true", help="exit 3 when a gate fails")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug output")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    rules = GateRules(
        args.gate_ambiguous_pp, args.gate_keep_drop, args.gate_keep_drop_share, args.gate_skip_rise
    )
    if args.compare:
        if args.manifest:
            ap.error("--compare takes two results files and no manifest")
        try:
            a, b = (load_results(p) for p in args.compare)
        except (OSError, ValueError) as err:
            logger.error("%s", err)
            return 2
        cmp = compare_results(a, b, rules, args.holdout)
        print("\n".join(compare_lines(cmp)))
        return 3 if args.strict_gate and cmp["gate"]["verdict"] == "fail" else 0
    if not args.manifest:
        ap.error("give a manifest or --compare A B")
    if args.runs < 1:
        ap.error("--runs must be at least 1")
    if not 0 < args.min_overlap <= 1:
        ap.error("--min-overlap must be above 0 and at most 1")
    try:
        m = load_manifest(args.manifest)
        if args.rehash:
            print(holdout_hash(m.sources))
            logger.info('Store it in the manifest as "holdout_sha256"')
            return 0
        check_holdout_hash(m)
        variants = (
            check_variants([variant_flag(v) for v in args.variant]) if args.variant else m.variants
        )
        bench = (args.bench or args.manifest.parent).resolve()
        extra = [args.home, args.cache, *(v.cache for v in variants)]
        check_outputs([bench, *(d for d in extra if d)], m.sources)
        reuse = _reuse_flags(args.reuse, m)
    except ManifestError as err:
        logger.error("%s", err)
        return 2
    except (OSError, ValueError) as err:  # JSONDecodeError is a ValueError
        logger.error("%s", err)
        return 2
    try:
        with bench_lock(machine_lock()):
            os.environ.update(bench_env(bench, args.home, args.cache))
            reset_state()  # the config of the bench home from here on
            results = run_bench(
                m,
                bench,
                variants,
                holdout=args.holdout,
                rerun=args.rerun,
                reuse=reuse,
                cache=Path(os.environ["CUTPOINT_CACHE_DIR"]),
                interval=args.sample_interval,
                ollama_url=sampler_ollama_url() if args.sample_interval > 0 else None,
                runs=args.runs,
                seed=args.seed,
                min_overlap=args.min_overlap,
                argv=sys.argv[1:] if argv is None else list(argv),
            )
            comparisons = [compare_results(results[0], r, rules, args.holdout) for r in results[1:]]
            report = render_report(results, comparisons)
            out_dir = bench / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = results[0]["created"]
            for r in results:
                path = out_dir / f"{stamp}-{r['variant']['name']}.json"
                path.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
                logger.info("Wrote %s", path)
            report_path = out_dir / f"{stamp}-report.md"
            report_path.write_text(report, encoding="utf-8")
            logger.info("Wrote %s", report_path)
    except BenchBusyError as err:
        logger.error("%s", err)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted; finished runs keep their stamps, so a re-run reuses them")
        return 130
    except (OSError, ValueError) as err:
        logger.error("%s: %s", type(err).__name__, err)
        return 1
    print(report)
    failed = any(c["gate"]["verdict"] == "fail" for c in comparisons)
    return 3 if args.strict_gate and failed else 0


if __name__ == "__main__":
    sys.exit(main())
