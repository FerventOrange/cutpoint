"""Insta360 Media SDK wrapper: stitch whole X3 clips to equirect (``cutpoint 360 stitch``).

Runs ``MediaSDKTest`` (``.exe`` on Windows) as
``MediaSDKTest -inputs <_00_.insv> <_10_.insv> -output <out.mp4> <pano.stitch_args>``;
the parameters are those in Insta360's public Media SDK docs.

Cutpoint judges success from the SDK's progress and error output plus other
status markers it prints at runtime (see :func:`parse_sdk_output` and
:func:`failure_reasons`), and verifies the output file with ffprobe. Cutpoint
accepts only the stitch types listed in Insta360's public docs
(:func:`cutpoint.pano.project.validate_stitch_args`).

The SDK cannot trim, so whole clips are stitched; only clips that contain a
chosen segment are stitched (``--clips`` or the clips in a manifest /
highlights selection). Output goes to ``pano.stitched_dir`` (pick a drive with room:
about 54 GB per hour of footage), through a ``.partial`` file that is deleted on
failure, timeout (``pano.stitch_timeout_s``) or Ctrl-C, and is checked against
the clip's frame count. The full stdout/stderr of each run is written to
``<stitched_dir>/logs/<stem>.log``. Each finished stitch gets a sidecar
``<stem>.equirect.json`` (:func:`write_sidecar`) recording its arguments, so
renders know whether it was made with FlowState.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cutpoint import cache
from cutpoint.pano.project import (
    EQUIRECT_META_VERSION,
    SDK_EXE_NAME,
    equirect_flowstate,
    equirect_sidecar,
    pano_config,
    sdk_exe,
    sdk_output,
    sdk_version,
    stitch_args,
    stitch_timeout_s,
    stitched_dir,
    validate_stitch_args,
)
from cutpoint.pano.x3 import X3Clip

logger = logging.getLogger(__name__)

MIN_FREE_GB = 20.0
MIN_OUTPUT_BYTES = 1_000_000  # an empty/aborted MP4 is far smaller than any real stitch
MIN_OUTPUT_BYTES_PER_S = 125_000  # 1 Mb/s floor (the default -bitrate is 100 Mb/s)
TAIL_LINES = 20
PROGRESS_STEP = 5  # log progress every 5 % ...
PROGRESS_EVERY_S = 10.0  # ... or every 10 s, whichever comes first
HELP_TIMEOUT_S = 60.0
READER_JOIN_S = 10.0  # wait for the output reader threads after the process ends
PARTIAL_RM_ATTEMPTS = 5  # removing a .partial file the SDK may still hold open ...
PARTIAL_RM_DELAY_S = 0.5  # ... is retried this often, this far apart

# Markers the parser matches (code literals only).
PROGRESS_KEY = "process"
ERROR_KEY = "error"
END_MARKER = "end stitch"
COST_KEY = "cost"
EXPORT_MARKER = "[Export Summary]"
FEATURES_MARKER = "--- Feature Status"
BLOCK_END = "---"
VERSION_KEY = "SDK version:"
FEATURE_OFF = ("OFF", "0")  # name / public GetFeatureStatusMap code
FEATURE_FAILED = ("FAILED", "3")
FEATURE_UNKNOWN = ("UNKNOWN", "-1")

_PROGRESS_RE = re.compile(rf"{PROGRESS_KEY}\s*=\s*(\d+)\s*%")
_ERROR_RE = re.compile(rf"^\s*{ERROR_KEY}:\s*(.*)$", re.IGNORECASE)
_COST_RE = re.compile(rf"^\s*{COST_KEY}\s*=\s*([0-9.eE+-]+)")
_FEATURE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(\S.*?)\s*$")
_VERSION_RE = re.compile(re.escape(VERSION_KEY) + r"\s*([0-9][0-9.]*)", re.IGNORECASE)


class SdkNotConfigured(FileNotFoundError):
    """The ``MediaSDKTest`` executable is not configured or not found."""


def require_sdk() -> Path:
    """The configured or auto-detected SDK executable.

    Raises:
        SdkNotConfigured: With instructions when it is missing.
    """
    exe = sdk_exe()
    if exe is None or not exe.is_file():
        where = f" ({exe} does not exist)" if exe else ""
        raise SdkNotConfigured(
            "Insta360 Media SDK not found" + where + ". Apply for the Media SDK in the "
            "developer/SDK section of insta360.com (https://www.insta360.com/sdk/apply), "
            "extract it outside the repository, e.g. under <CUTPOINT_HOME>/sdk/insta360/ "
            f"(searched automatically), or set "
            f'"pano": {{"sdk_exe": "<path to {SDK_EXE_NAME}>"}} in config.json or the '
            "CUTPOINT_INSTA360_SDK environment variable. Until then render with "
            "--pano-source studio (export equirect from Insta360 Studio) or dfisheye "
            "(preview quality)."
        )
    return exe


def stitch_command(exe: Path, clip: X3Clip, out: Path, extra: list[str] | None = None) -> list[str]:
    """The ``MediaSDKTest`` argument list for one clip.

    Raises:
        ValueError: If the arguments hold an unsupported ``-stitch_type``.
    """
    from cutpoint.config import remap_source  # noqa: PLC0415

    args = validate_stitch_args([*stitch_args(), *(extra or [])])
    inputs = [str(remap_source(p)) for p in clip.insv]  # moved footage: read the new copy
    return [str(exe), "-inputs", *inputs, "-output", str(out), *args]


# --------------------------------------------------------------------------- output parsing
@dataclass
class SdkResult:
    """What one ``MediaSDKTest`` run reported on stdout.

    Attributes:
        progress: Highest ``process = N%`` seen (-1 when none).
        ended: The end marker was seen.
        errors: Text of each ``error:`` line.
        cost_s: The reported run time in seconds, or None.
        export: Reported export fields.
        features: Reported status per feature.
        lines: All non-empty stdout lines (split on CR and LF).
    """

    progress: int = -1
    ended: bool = False
    errors: list[str] = field(default_factory=list)
    cost_s: float | None = None
    export: dict[str, str] = field(default_factory=dict)
    features: dict[str, str] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        """Progress reached 100 %."""
        return self.progress >= 100

    def summary(self) -> dict[str, Any]:
        """JSON-friendly digest for the per-clip record."""
        return {
            "progress": self.progress,
            "cost_s": self.cost_s,
            "export": dict(self.export),
            "features": dict(self.features),
            "errors": list(self.errors),
        }


def split_lines(text: str) -> list[str]:
    """Non-empty lines of ``text``, split on CR as well as LF."""
    return [ln for ln in re.split(r"[\r\n]+", text) if ln.strip()]


def parse_progress(line: str) -> list[int]:
    """Every ``process = N%`` value in one line (several may share a line)."""
    return [int(v) for v in _PROGRESS_RE.findall(line)]


def parse_export_summary(line: str) -> dict[str, str]:
    """``{key: value}`` of an export status line (``{}`` for any other line)."""
    _, sep, rest = line.partition(EXPORT_MARKER)
    if not sep:
        return {}
    out: dict[str, str] = {}
    for item in rest.split("|"):
        key, colon, value = item.partition(":")
        if colon and key.strip():
            out[key.strip()] = value.strip()
    return out


def parse_sdk_output(text: str) -> SdkResult:
    """Parse ``MediaSDKTest`` stdout: progress, errors and the other status markers.

    Pure function: suitable for synthetic input in tests.
    """
    res = SdkResult(lines=split_lines(text))
    in_features = False
    for raw in res.lines:
        pcts = parse_progress(raw)
        if pcts:
            res.progress = max(res.progress, *pcts)
        line = _PROGRESS_RE.sub("", raw).strip()  # the rest of the line
        if not line:
            continue
        if in_features:
            if line.startswith(BLOCK_END):
                in_features = False
                continue
            m = _FEATURE_RE.match(line)
            if m:
                res.features[m.group(1)] = m.group(2)
                continue
            in_features = False  # anything else ends the block
        if line.startswith(FEATURES_MARKER):
            in_features = True
            continue
        m = _ERROR_RE.match(line)
        if m:
            res.errors.append(m.group(1).strip())
            continue
        if line.startswith(END_MARKER):
            res.ended = True
            continue
        m = _COST_RE.match(line)
        if m:
            with contextlib.suppress(ValueError):
                res.cost_s = float(m.group(1))
            continue
        summary = parse_export_summary(line)
        if summary:
            res.export.update(summary)
    return res


def _norm_feature(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def requested_features(args: list[str]) -> list[str]:
    """Feature names requested by ``-enable_<name>`` flags (normalised: no ``_``).

    Flags that are not features (``-enable_h265_encoder``, ``-enable_soft_encode``,
    ...) get no reported feature status, so :func:`failure_reasons` ignores them.
    """
    out = []
    for a in args:
        a = str(a)
        if a.startswith("-enable_"):
            out.append(_norm_feature(a[len("-enable_") :]))
    return list(dict.fromkeys(out))


def failure_reasons(
    result: SdkResult,
    returncode: int | None,
    requested: list[str],
    output_bytes: int | None,
    min_bytes: int = MIN_OUTPUT_BYTES,
) -> list[str]:
    """Why a run failed (empty list = success). Pure function.

    Args:
        result: Parsed stdout.
        returncode: Process exit code (None if it was killed).
        requested: :func:`requested_features` of the arguments.
        output_bytes: Size of the output file, or None if it is missing.
        min_bytes: Smallest plausible output size.
    """
    from cutpoint.ffmpeg_utils import exit_code  # noqa: PLC0415

    reasons = []
    if returncode is None:
        reasons.append("process killed")
    elif returncode != 0:
        reasons.append(f"exit code {exit_code(returncode)} (bad arguments?)")
    reasons += [f"error: {e}" for e in result.errors]
    if not result.finished:
        reasons.append(
            "stitch did not complete"
            + (f" (progress {result.progress}%)" if result.progress >= 0 else " (no progress)")
        )
    if output_bytes is None:
        reasons.append("no output file")
    elif output_bytes < min_bytes:
        reasons.append(f"output only {output_bytes} bytes (expected >= {min_bytes})")
    by_norm = {_norm_feature(k): (k, v) for k, v in result.features.items()}
    for f in requested:
        if f not in by_norm:
            continue
        name, status = by_norm[f]
        st = status.strip().upper()
        if st.startswith(FEATURE_FAILED[0]) or st in FEATURE_FAILED:
            reasons.append(f"feature {name} FAILED")
        elif st in FEATURE_OFF:
            reasons.append(f"feature {name} is OFF although requested")
    return reasons


_FEATURE_HIDDEN = (*FEATURE_OFF, *FEATURE_UNKNOWN)
_FEATURE_LABELS = {FEATURE_FAILED[1]: FEATURE_FAILED[0]}  # known public numeric codes


def features_summary(features: dict[str, str]) -> str:
    """``k=v`` list of features that are not off/unknown; known numeric codes as names."""
    shown = []
    for k, v in features.items():
        st = v.strip().upper()
        if st in _FEATURE_HIDDEN:
            continue
        shown.append(f"{k}={_FEATURE_LABELS.get(st, v.strip())}")
    return ", ".join(shown)


def parse_help_version(text: str) -> str | None:
    """The version number in ``MediaSDKTest -help`` output, or None."""
    m = _VERSION_RE.search(text)
    return m.group(1) if m else None


class ProgressThrottle:
    """Decides when to log progress: every :data:`PROGRESS_STEP` % or :data:`PROGRESS_EVERY_S`."""

    def __init__(self, step: int = PROGRESS_STEP, every_s: float = PROGRESS_EVERY_S) -> None:
        self.step = step
        self.every_s = every_s
        self.last_pct = -1
        self.last_t: float | None = None

    def should_log(self, pct: int, now: float) -> bool:
        """True when ``pct`` (seen at time ``now``) should be logged."""
        if pct <= self.last_pct:
            return False
        if (
            self.last_t is None
            or pct >= 100
            or pct - self.last_pct >= self.step
            or now - self.last_t >= self.every_s
        ):
            self.last_pct, self.last_t = pct, now
            return True
        return False


# --------------------------------------------------------------------------- running
@dataclass
class SdkRun:
    """Raw outcome of one ``MediaSDKTest`` process."""

    returncode: int | None
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False


class SdkInterrupted(KeyboardInterrupt):
    """Ctrl-C during a run; ``run`` holds the output captured until the kill."""

    def __init__(self, run: SdkRun) -> None:
        super().__init__()
        self.run = run


def _kill(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        logger.warning("MediaSDKTest (pid %s) did not exit after kill", proc.pid)


def sdk_env(exe: str | Path) -> dict[str, str] | None:
    """Child environment for the SDK: on Linux with its ``lib`` folder on ``LD_LIBRARY_PATH``.

    The folder is ``lib`` next to the executable or next to its ``bin`` folder
    (``/opt/MediaSDK-<version>-linux/lib``), if it holds shared libraries.
    None (inherit the environment) elsewhere or when there is none.
    """
    if not sys.platform.startswith("linux"):
        return None
    exe = Path(exe)
    for lib in (exe.parent / "lib", exe.parent.parent / "lib"):
        if lib.is_dir() and any(lib.glob("*.so*")):
            env = dict(os.environ)
            old = env.get("LD_LIBRARY_PATH")
            env["LD_LIBRARY_PATH"] = f"{lib}{os.pathsep}{old}" if old else str(lib)
            return env
    return None


def run_sdk(cmd: list[str], label: str = "", timeout_s: float | None = None) -> SdkRun:
    """Run ``MediaSDKTest``, logging throttled progress; kill it on timeout or Ctrl-C.

    Args:
        cmd: Full command line.
        label: Prefix of the progress log lines (e.g. ``clip 029``).
        timeout_s: Kill the process after this many seconds (None = no limit).

    Raises:
        SdkInterrupted: (a ``KeyboardInterrupt``) after killing the process on
            Ctrl-C, carrying the output captured so far.
    """
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        encoding="utf-8",
        errors="replace",
        env=sdk_env(cmd[0]),
    )
    out_chunks: list[str] = []
    err_chunks: list[str] = []
    throttle = ProgressThrottle()

    def read_stdout() -> None:
        assert proc.stdout is not None
        # text mode = universal newlines: readline() also stops at a bare CR
        for chunk in iter(proc.stdout.readline, ""):
            out_chunks.append(chunk)
            for pct in parse_progress(chunk):
                if throttle.should_log(pct, time.perf_counter()):
                    logger.info("%s: %d%% (%.0fs)", label, pct, time.perf_counter() - t0)

    def read_stderr() -> None:
        assert proc.stderr is not None
        err_chunks.append(proc.stderr.read())

    readers = [
        threading.Thread(target=read_stdout, daemon=True),
        threading.Thread(target=read_stderr, daemon=True),
    ]
    for th in readers:
        th.start()
    timed_out = interrupted = False
    try:
        while proc.poll() is None:
            if timeout_s is not None and time.perf_counter() - t0 > timeout_s:
                timed_out = True
                _kill(proc)
                break
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=0.5)
    except KeyboardInterrupt:
        interrupted = True
        _kill(proc)
    finally:
        for th in readers:
            th.join(timeout=READER_JOIN_S)
            if th.is_alive():
                logger.warning(
                    "%s: output reader still running after %.0fs; the captured "
                    "output (and log) may be truncated",
                    label or "MediaSDKTest",
                    READER_JOIN_S,
                )
    run = SdkRun(
        None if timed_out or interrupted else proc.returncode,
        "".join(out_chunks),
        "".join(err_chunks),
        round(time.perf_counter() - t0, 1),
        timed_out,
    )
    if interrupted:
        raise SdkInterrupted(run)
    return run


def probe_sdk(exe: Path, timeout_s: float = HELP_TIMEOUT_S) -> dict[str, Any]:
    """Run ``<exe> -help``: ``{"runs", "version", "seconds", "error"}``.

    The SDK "runs" when its ``-help`` output names a version within
    ``timeout_s``; the exit code is not used.
    """
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(exe), "-help"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
            check=False,
            env=sdk_env(exe),
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        return {
            "runs": False,
            "version": None,
            "seconds": round(time.perf_counter() - t0, 1),
            "error": str(err),
        }
    version = parse_help_version(proc.stdout + proc.stderr)
    return {
        "runs": version is not None,
        "version": version,
        "seconds": round(time.perf_counter() - t0, 1),
        "error": None if version else "no version line in -help output",
    }


def sdk_status() -> dict[str, Any]:
    """SDK path, origin, version (folder name, else ``-help``) and whether it runs."""
    from cutpoint.pano.project import sdk_location  # noqa: PLC0415

    exe, origin = sdk_location()
    info: dict[str, Any] = {"exe": str(exe) if exe else None, "origin": origin}
    if exe is None or not exe.is_file():
        info.update(found=False, runs=False, version=None)
        return info
    probe = probe_sdk(exe)
    info.update(
        found=True,
        runs=probe["runs"],
        version=sdk_version(exe) or probe["version"],
        help_seconds=probe["seconds"],
        error=probe["error"],
    )
    return info


def _write_log(path: Path, cmd: list[str], run: SdkRun | None, note: str = "") -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(f"command: {subprocess.list2cmdline(cmd)}\n")
            if run is not None:
                f.write(
                    f"exit code: {run.returncode}  seconds: {run.seconds}"
                    f"{'  TIMED OUT' if run.timed_out else ''}\n"
                )
            if note:
                f.write(f"result: {note}\n")
            if run is not None:
                f.write("\n===== stdout =====\n")
                f.write("\n".join(split_lines(run.stdout)) + "\n")
                f.write("\n===== stderr =====\n")
                f.write(run.stderr.replace("\r\n", "\n"))
    except OSError as err:
        logger.warning("could not write %s: %s", path, err)


def write_sidecar(out: Path, cmd: list[str]) -> Path:
    """Write :func:`~cutpoint.pano.project.equirect_sidecar` for the stitch ``out`` made by ``cmd``.

    Records the schema version, the arguments after ``-output <file>`` and the
    SDK version from the executable's folder name (None when it has none).
    :func:`~cutpoint.pano.project.resolve_source` reads it to tell FlowState
    stitches from plain ones. On a write error the stale sidecar (if any) is
    removed, so the file counts as an unknown (FlowState) stitch.
    """
    path = equirect_sidecar(out)
    i = cmd.index("-output")
    meta = {
        "version": EQUIRECT_META_VERSION,
        "stitch_args": [str(a) for a in cmd[i + 2 :]],
        "sdk_version": sdk_version(cmd[0]),
    }
    try:
        cache.save_json(path, meta)
    except OSError as err:
        logger.warning("could not write %s: %s", path, err)
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
    return path


def _need_gb(clip: X3Clip) -> float:
    """Estimated size of one stitched clip (~100 Mb/s, the default -bitrate, +20%)."""
    return clip.duration / 3600 * 45 * 1.2


def _bitrate(args: list[str] | None) -> int | None:
    """The last positive ``-bitrate`` value (bits/s) in ``args``, or None."""
    items = [str(a) for a in args or []]
    value = None
    for flag, raw in itertools.pairwise(items):
        if flag == "-bitrate":
            with contextlib.suppress(ValueError):
                value = int(raw)
    return value if value and value > 0 else None


def _min_bytes(clip: X3Clip, args: list[str] | None = None) -> int:
    """Smallest plausible output size for ``clip`` run with ``args``.

    The per-second floor is :data:`MIN_OUTPUT_BYTES_PER_S`, or a quarter of a
    user-set ``-bitrate`` (in bytes/s) when that is lower.
    """
    rate = MIN_OUTPUT_BYTES_PER_S
    bitrate = _bitrate(args)
    if bitrate is not None:
        rate = min(rate, bitrate // 8 // 4)
    return max(MIN_OUTPUT_BYTES, int(clip.duration * rate))


def _on_root_fs(path: Path) -> bool:
    """POSIX: ``path`` (or its nearest existing parent) is on the file system of ``/``."""
    probe = path.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return probe.stat().st_dev == Path("/").stat().st_dev
    except OSError:
        return False


def _warn_system_drive(out_dir: Path, clips: list[X3Clip]) -> None:
    """Warn when the default stitch folder is on the system drive (does not block).

    Only when ``pano.stitched_dir`` is unset: stitched equirects (~54 GB per hour
    of footage) can fill the system drive quickly. The system drive is
    ``%SystemDrive%`` on Windows and the file system of ``/`` elsewhere.
    """
    if not clips or pano_config().get("stitched_dir"):
        return
    if sys.platform == "win32":
        system = os.environ.get("SystemDrive") or "C:"  # noqa: SIM112 - Windows spelling
        if out_dir.resolve().drive.lower() != system.lower():
            return
    else:
        system = "/"
        if not _on_root_fs(out_dir):
            return
    logger.warning(
        "Stitching to %s on the system drive (%s): about %.0f GB for %d clip(s). "
        'Set "pano": {"stitched_dir": "<folder>"} in config.json to use '
        "another drive.",
        out_dir,
        system,
        sum(map(_need_gb, clips)),
        len(clips),
    )


def stitch_clips(
    clips: list[X3Clip], force: bool = False, dry_run: bool = False
) -> list[dict[str, Any]]:
    """Stitch each clip to ``<stitched_dir>/<stem>.equirect.mp4`` (skips existing ones).

    Args:
        clips: Clips to stitch (whole clips: the SDK cannot trim).
        force: Re-stitch existing outputs.
        dry_run: Only log the commands (works without the SDK).

    Returns:
        One record per clip: ``clip``, ``output``, ``status`` (``ok``, the
        verification notes, ``failed``, ``exists`` or ``dry-run``); after a run
        also ``seconds``, ``log_file`` and ``sdk`` (the parsed status, see
        :meth:`SdkResult.summary`); on failure ``error`` (the reasons)
        and ``log`` (the last stdout/stderr lines).

    Raises:
        SdkNotConfigured: If the SDK is missing (unless ``dry_run``).
        ValueError: If the output folder lacks free space, or for invalid
            ``pano.stitch_args`` / ``pano.stitch_timeout_s``.
        KeyboardInterrupt: After killing the SDK and removing the partial output.
    """
    import shutil  # noqa: PLC0415

    out_dir = stitched_dir()
    exe = sdk_exe() if dry_run else require_sdk()
    timeout_s = stitch_timeout_s()
    _warn_system_drive(out_dir, [c for c in clips if force or not sdk_output(c).is_file()])
    records = []
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    for c in clips:
        out = sdk_output(c)
        if out.is_file() and not force:
            flowstate = equirect_flowstate(out)
            if flowstate is not False:
                logger.warning(
                    "clip %s: existing %s %s; its framing drifts with the camera's "
                    "heading. Restitch: cutpoint 360 stitch <project> --clips %s --force",
                    c.clip_id,
                    out.name,
                    "has no sidecar (assumed FlowState)"
                    if flowstate is None
                    else "was stitched with FlowState",
                    c.clip_id,
                )
            records.append({"clip": c.clip_id, "output": str(out), "status": "exists"})
            continue
        need_gb = _need_gb(c)
        if not dry_run:
            free = shutil.disk_usage(out_dir).free / 1e9
            if free - need_gb < MIN_FREE_GB:
                raise ValueError(
                    f"{out_dir}: {free:.0f} GB free, clip {c.clip_id} needs ~{need_gb:.0f} GB"
                )
        part = out.with_name(out.name + ".partial.mp4")
        cmd = stitch_command(exe or Path(SDK_EXE_NAME), c, part)
        logger.info("clip %s: %s", c.clip_id, subprocess.list2cmdline(cmd))
        if dry_run:
            records.append(
                {
                    "clip": c.clip_id,
                    "output": str(out),
                    "status": "dry-run",
                    "command": cmd,
                    "estimated_gb": round(need_gb, 1),
                }
            )
            continue
        records.append(_stitch_one(c, cmd, part, out, out_dir, timeout_s))
    return records


def _remove_partial(part: Path, attempts: int | None = None, delay_s: float | None = None) -> None:
    """Delete ``part``, retrying while it is locked (e.g. the SDK has not exited yet).

    Tries ``attempts`` times (default :data:`PARTIAL_RM_ATTEMPTS`), ``delay_s``
    apart (default :data:`PARTIAL_RM_DELAY_S`); logs a warning instead of raising
    when it still cannot be removed.
    """
    attempts = max(PARTIAL_RM_ATTEMPTS if attempts is None else attempts, 1)
    delay_s = PARTIAL_RM_DELAY_S if delay_s is None else delay_s
    for i in range(attempts):
        try:
            part.unlink(missing_ok=True)
            return
        except OSError as err:
            if i >= attempts - 1:
                logger.warning("could not remove partial output %s: %s", part, err)
                return
            time.sleep(delay_s)


def _stitch_one(
    c: X3Clip, cmd: list[str], part: Path, out: Path, out_dir: Path, timeout_s: float | None
) -> dict[str, Any]:
    """Run, judge, log and finalise one clip (see :func:`stitch_clips`)."""
    log_file = out_dir / "logs" / f"{out.name.removesuffix('.equirect.mp4')}.log"
    part.unlink(missing_ok=True)
    try:
        run = run_sdk(cmd, label=f"clip {c.clip_id}", timeout_s=timeout_s)
    except KeyboardInterrupt as err:
        _remove_partial(part)
        partial = err.run if isinstance(err, SdkInterrupted) else None
        _write_log(log_file, cmd, partial, "interrupted (Ctrl-C); partial output removed")
        logger.error("clip %s: interrupted; removed %s", c.clip_id, part)
        raise
    except OSError as err:
        _remove_partial(part)
        _write_log(log_file, cmd, None, f"could not start: {err}")
        logger.error("clip %s: could not start MediaSDKTest: %s", c.clip_id, err)
        return {
            "clip": c.clip_id,
            "output": str(out),
            "status": "failed",
            "error": f"could not start: {err}",
            "log_file": str(log_file),
        }
    res = parse_sdk_output(run.stdout)
    size = part.stat().st_size if part.is_file() else None
    reasons = failure_reasons(
        res, run.returncode, requested_features(cmd), size, _min_bytes(c, cmd)
    )
    if run.timed_out:
        reasons.insert(0, f"timed out after {timeout_s:.0f}s (pano.stitch_timeout_s)")
    rec: dict[str, Any] = {
        "clip": c.clip_id,
        "output": str(out),
        "seconds": run.seconds,
        "log_file": str(log_file),
        "sdk": res.summary(),
    }
    if reasons:
        _remove_partial(part)
        text = [ln for ln in res.lines if _PROGRESS_RE.sub("", ln).strip()]
        tail = "\n".join((text or split_lines(run.stderr))[-TAIL_LINES:])
        error = "; ".join(reasons)
        _write_log(log_file, cmd, run, f"FAILED: {error}")
        rec.update(status="failed", error=error, log=tail)
        logger.error(
            "clip %s: stitch FAILED: %s (partial output removed; full log %s)\n%s",
            c.clip_id,
            error,
            log_file,
            tail,
        )
        return rec
    status = _verify(c, part)
    # --force: drop the old sidecar first, so an interruption before the new one is
    # written leaves a sidecar-less (warned, FlowState) file rather than a stale record
    with contextlib.suppress(OSError):
        equirect_sidecar(out).unlink(missing_ok=True)
    part.replace(out)
    write_sidecar(out, cmd)
    _write_log(log_file, cmd, run, f"ok; verify: {status}")
    rec["status"] = status
    ex = res.export
    feats = features_summary(res.features)
    logger.info(
        "clip %s: stitched in %.0fs (%.2fx realtime): %s; encoder %s (%s); features: %s",
        c.clip_id,
        run.seconds,
        c.duration / max(run.seconds, 1e-6),
        status,
        ex.get("Encoder", "?"),
        ex.get("Encode", "?"),
        feats or "none",
    )
    return rec


def _verify(clip: X3Clip, path: Path) -> str:
    """Compare the stitched file's frames and size with the clip."""
    from cutpoint.pano.x3 import probe_video_stream  # noqa: PLC0415

    s = probe_video_stream(path)
    w, h, n = int(s.get("width") or 0), int(s.get("height") or 0), int(s.get("nb_frames") or 0)
    notes = []
    if w != 2 * h:
        notes.append(f"not 2:1 ({w}x{h})")
    if clip.frames and abs(n - clip.frames) > 1:
        notes.append(f"{n} frames vs {clip.frames} in the proxy: time mapping is off")
    return "ok" if not notes else "; ".join(notes)
