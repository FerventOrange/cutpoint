"""Browser proxies, stills and envelopes of a project's sources for ``cutpoint ui``."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from cutpoint import cache, procs, stage
from cutpoint import envelope as audio_envelope
from cutpoint.config import remap_source
from cutpoint.encode import BT709_TAGS, TV_RANGE, preview_filter, still_filter
from cutpoint.errors import FFmpegError
from cutpoint.ffmpeg_utils import VideoEncoder, audio_mix_filter, grab_images, probe, video_encoder
from cutpoint.ui.common import (
    BROWSER_AUDIO,
    BROWSER_CONTAINERS,
    BROWSER_VIDEO,
    ENVELOPE_STALE_S,
    ENVELOPE_TIMEOUT_S,
    ENVELOPE_VERSION,
    FRAGMENT_SCAN_BOXES,
    FRAME_CACHE_MAX,
    FRAME_MAX_AT,
    FRAME_SLOTS,
    FRAME_TIMEOUT_S,
    UI_DIR,
    logger,
    spawn,
)
from cutpoint.ui.files import _inside, _own, _ui_folder
from cutpoint.ui.lanes import Lane

# Proxies: a keyframe every PROXY_GOP frames and at most PROXY_FPS_MAX fps keep seeks quick (the
# encoder default GOP made them take up to ~0.4 s) and the encode light.
PROXY_GOP = 30
PROXY_FPS_MAX = 30
# NVENC at CQ 28 alone was up to four times libx264's size at CRF 27. At CQ 32 with this cap, a
# 1080p60 recording's proxy was about 1.2 times libx264's (scoring a higher SSIM) and needed
# about half the CPU time of libx264.
# VAAPI at QP 32 (AMD, 1080p60, 15-20 s of four synthetic contents) came out 0.6 to 1.0 times
# libx264's size, so it needs no cap. AMF was not measured (no Windows machine): AMF proxies
# stay on libx264 (Media._gpu_encoder).
PROXY_GPU_QUALITY = 32  # NVENC CQ; the same number as QP for VAAPI
PROXY_NVENC_RATE = ["-maxrate", "1500k", "-bufsize", "3000k"]
# Window proxies (#323): while the whole proxy builds, the focused moment plays from a short
# proxy of [start - BEFORE, end + AFTER], its edges rounded outward to STEP so nearby requests
# share one. At most WINDOWS_MAX per source are kept; the least recently used go first.
PROXY_WINDOW_BEFORE = 10.0
PROXY_WINDOW_AFTER = 20.0
PROXY_WINDOW_STEP = 5.0
PROXY_WINDOWS_MAX = 8
# "Play sequence" queues the windows of its next clips behind the focus (``queue``); at most this
# many wait, so a long sequence cannot evict the windows it is about to play
PROXY_WINDOWS_QUEUED = 4
# The page plays a window up to this many centiseconds before its end (app.js ``covers``, FRAME)
PROXY_WINDOW_FRAME_CS = 4
# ui/proxy/<proxy key>.w<lo>-<hi>.mp4 (see Media._proxy_key), the range in centiseconds of
# the source
PROXY_WINDOW_FILE = re.compile(r"([0-9a-f]{16})\.w(\d+)-(\d+)\.mp4")
# a whole proxy ui/proxy/<proxy key>.mp4 or a window
PROXY_FILE = re.compile(r"([0-9a-f]{16})(?:\.w\d+-\d+)?\.mp4")
# prune_proxies leaves files written or used this recently: another run on the project may be
# encoding one (a temporary grows as it encodes), or about to serve or serving one
# (Media.window and the proxy routes touch what they serve, see _mark_used)
PROXY_PRUNE_GRACE_S = 3600.0
# while a source cannot be read, a proxy or window that names no other source goes only once
# it has been idle this long (a drive that stays offline must not keep them forever)
PROXY_OFFLINE_PRUNE_S = 30 * 86400.0
# A frames folder over FRAME_CACHE_MAX stills is evicted to this share of the cap below it, so
# the next evictions (each stats every still) wait for that many new stills
FRAME_EVICT_SHARE = 0.05
# a ``key=value`` line of ffmpeg's ``-progress`` output (frame=, out_time_us=, progress=...)
PROXY_PROGRESS_LINE = re.compile(r"([a-z0-9_]+)= *(\S*)")  # bitrate= 4.1kbits/s


def proxy_args(
    real: Path,
    out: Path,
    audio_streams: int,
    *,
    encoder: VideoEncoder | None = None,
    color: str = "",
    lo: float | None = None,
    hi: float | None = None,
) -> list[str]:
    """ffmpeg command for a 720p H.264/AAC preview; several audio tracks are mixed as in exports.

    ``encoder``: a GPU encoder (NVENC, VAAPI or AMF), else libx264. Decoding stays on the CPU:
    ``-hwaccel cuda`` halved the CPU time but made the build two to three times slower on an RTX 3080.
    ``color``: the source's :func:`~cutpoint.encode.preview_filter` (HDR and BT.2020 become SDR
    Rec.709, tagged so; full-range BT.601 becomes TV range, its tags kept). ``lo``/``hi``: only
    that part of the source, in seconds (a window proxy, whose 0 is ``lo``).
    """
    mix = audio_mix_filter(audio_streams)
    audio = ["-filter_complex", f"{mix}[a]", "-map", "[a]"] if mix else ["-map", "0:a:0?"]
    video = (
        [*encoder.args(PROXY_GPU_QUALITY), *(PROXY_NVENC_RATE if encoder.kind == "nvenc" else [])]
        if encoder
        else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p"]
    )
    scale = "scale=-2:min(720\\,ih)"
    vf = f"{scale},{color},format=yuv420p" if color else scale
    if encoder:
        vf = encoder.chain(vf)
    start = ["-ss", f"{lo:.3f}"] if lo is not None else []
    length = ["-t", f"{hi - (lo or 0.0):.3f}"] if hi is not None else []
    return [
        "ffmpeg", "-y", "-nostdin", "-v", "error", *start, "-i", str(real), "-map", "0:v:0",
        *audio, "-vf", vf, *video,
        *(BT709_TAGS if color and color != TV_RANGE else []), *length,
        "-g", str(PROXY_GOP), "-fpsmax", str(PROXY_FPS_MAX),
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out),
    ]  # fmt: skip


def _tail(output: str, lines: int = 5) -> str:
    """The last ``lines`` non-empty lines of an encode's output, trimmed: why it failed."""
    kept = [line.strip() for line in output.splitlines() if line.strip()]
    return "\n".join(kept[-lines:])[-500:]


def _exit_reason(code: int, tail: str) -> str:
    """``tail``, or the exit code when the encode printed nothing (e.g. it was killed)."""
    if tail:
        return tail
    hexa = f" (0x{code & 0xFFFFFFFF:08X})" if code < 0 or code > 255 else ""  # Windows NTSTATUS
    return f"ffmpeg exited with code {code}{hexa}"


def _idle(file: Path, cutoff_ns: int) -> bool:
    """``file`` was last written or touched before ``cutoff_ns``; False if it is gone."""
    try:
        return file.stat().st_mtime_ns < cutoff_ns
    except OSError:
        return False


def _mark_used(file: Path) -> None:
    """A proxy is being served: touch it so another run's prune_proxies leaves it. Only once a
    quarter of the grace period has passed, so its Last-Modified rarely changes mid-playback."""
    with contextlib.suppress(OSError):
        if time.time() - file.stat().st_mtime > PROXY_PRUNE_GRACE_S / 4:
            os.utime(file)


def _covers(span: tuple[int, int], want: tuple[int, int], dur_cs: int) -> bool:
    """The window ``span`` plays all of ``want`` (centiseconds) as the page sees it (app.js
    ``covers``): not its last frame, unless it ends with the source (``dur_cs``, 0 if unknown)."""
    lo, hi = span
    return want[0] >= lo and (
        want[1] < hi - PROXY_WINDOW_FRAME_CS
        or (dur_cs > 0 and hi >= dur_cs - PROXY_WINDOW_FRAME_CS and want[1] <= hi)
    )


class ProxyWindow:
    """One queued or running window proxy encode: ``lo``/``hi`` in centiseconds of the source."""

    def __init__(
        self, project: Path, real: Path, key: str, lo: int, hi: int, audio_streams: int, color: str
    ) -> None:
        self.project, self.real, self.key = project, real, key
        self.lo, self.hi = lo, hi
        self.audio_streams, self.color = audio_streams, color
        self.proc: subprocess.Popen[str] | None = None  # set and cleared under ``Media.lock``
        self.cancelled = False  # superseded or no longer needed: never starts, or is killed

    @property
    def name(self) -> str:
        return f"{self.key}.w{self.lo}-{self.hi}"


# --------------------------------------------------------------------------- playable media
def fragmented_mp4(path: Path) -> bool:
    """Whether ``path`` is a fragmented MP4 or MOV without a segment index: a ``moof`` box among
    its first top-level boxes before any ``sidx``. A browser opens such a file with a Range
    request per fragment (hundreds, #1014). Reads only box headers; False if unreadable."""
    try:
        with path.open("rb") as f:
            end = os.fstat(f.fileno()).st_size
            pos = 0
            for _ in range(FRAGMENT_SCAN_BOXES):
                f.seek(pos)
                head = f.read(16)
                if len(head) < 8:
                    return False
                size, kind, start = int.from_bytes(head[:4], "big"), head[4:8], 8
                if size == 1 and len(head) == 16:  # a 64-bit size follows the type
                    size, start = int.from_bytes(head[8:16], "big"), 16
                elif size == 0:  # the box runs to the end of the file
                    size = end - pos
                if kind in (b"moof", b"sidx"):
                    return kind == b"moof"
                if size < start or pos + size >= end:
                    return False
                pos += size
    except OSError:
        pass
    return False


class Media:
    """Probe cache and on-demand H.264/AAC proxies for sources a browser cannot play.

    Two lanes, one worker each: whole-source proxies (``pool``, first in, first out) and window
    proxies of the focused moment (``window_lane``: a focus request supersedes the queued
    windows; a ``queue`` request waits behind them, at most PROXY_WINDOWS_QUEUED), so a moment plays
    within seconds while its source's whole proxy still builds.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # key -> (browser plays it as is, duration, audio tracks, preview_filter of a proxy)
        # or None
        self.probed: dict[str, tuple[bool, float, int, str] | None] = {}
        # key -> set once its probe ends: the one ffprobe a key runs at a time (see info)
        self.probing: dict[str, threading.Event] = {}
        self.probe_lock = threading.Lock()  # guards ``probing`` only, never held while probing
        # (project, key) -> "preparing" | "failed": each project builds its own proxy file
        self.proxies: dict[tuple[Path, str], str] = {}
        # (project, key) -> share of its whole proxy encoded, 0-1
        self.progress: dict[tuple[Path, str], float] = {}
        self.procs: set[subprocess.Popen[Any]] = set()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cutpoint-proxy")
        # (project, key) -> {(lo, hi) centiseconds: file} of ready windows, least recently used first
        self.windows: dict[tuple[Path, str], dict[tuple[int, int], Path]] = {}
        # (project, key) -> windows of a source whose whole proxy landed: no longer listed, but
        # still served (a page may be playing one) until the next run's prune_proxies deletes them
        self.retired: dict[tuple[Path, str], dict[tuple[int, int], Path]] = {}
        # (project, key) -> {(lo, hi) centiseconds: short reason} of window encodes that failed,
        # oldest first; listed so the page stops waiting, cleared when one is asked for again
        self.window_failures: dict[tuple[Path, str], dict[tuple[int, int], str]] = {}
        # the window proxies' one worker, beside the whole-source lane; its thread outlives
        # any window that raises (#1085). ``run`` looks ``_build_window`` up per window.
        self.window_lane: Lane[ProxyWindow] = Lane(
            self.lock,
            lambda win: self._build_window(win),
            name="cutpoint-proxy-window",
            failed=lambda _win: logger.exception("window proxy lane: a window failed"),
        )
        self.proxies_pruned: set[Path] = set()  # projects whose proxy leftovers went this run
        self.closing = False
        self.gpu_failed = False  # a proxy failed with the GPU encoder and worked with libx264
        # still file name -> (its builder lock, requests using it); dropped when the last one leaves
        self.frame_locks: dict[str, tuple[threading.Lock, int]] = {}
        self.frame_slots = threading.BoundedSemaphore(FRAME_SLOTS)
        self.frames_pruned: set[Path] = set()  # projects whose stale stills went this run
        # one envelope decodes at a time, on its own worker; content keys queued or decoding, and
        # the failure of a finished decode, until the next request reports it
        self.envelope_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cutpoint-wave")
        self.envelopes_running: set[str] = set()
        self.envelope_asked: dict[str, float] = {}  # content key -> monotonic time of the last ask
        self.envelope_errors: dict[str, str] = {}
        self.envelopes_pruned: set[Path] = set()  # projects whose stale envelopes went this run
        # key -> still_filter of its stills (#412, #441): also for a source the browser plays
        # as is
        self.still_color: dict[str, str] = {}
        # source path -> whether the probed file the browser plays as is is a fragmented MP4
        # (#1014), for the page's slow-start notice; the last probe of that path wins
        self.fragmented: dict[str, bool] = {}

    @staticmethod
    def key(real: Path) -> str | None:
        """Content key of ``real``: its path, size and mtime, plus whether ffmpeg reads it
        through a timeline copy (:mod:`cutpoint.tsrepair`, #1052), which probes and plays
        longer: a copy made or gone probes it again and names its proxies apart."""
        try:
            st = real.stat()
        except OSError:
            return None
        copy = "|timeline" if stage.timeline_copy(str(real)) else ""
        return hashlib.sha1(f"{real}|{st.st_size}|{st.st_mtime_ns}{copy}".encode()).hexdigest()[:16]

    def info(self, real: Path, key: str) -> tuple[bool, float, int, str] | None:
        """``(browser plays it as is, duration, audio tracks, colour filter of its proxy)``, or
        None if unreadable. The filter is :func:`~cutpoint.encode.preview_filter`'s, "" for a
        source that plays as is. One ffprobe per key: concurrent callers of a key not probed yet
        wait for the first one's result (no lock is held while it probes)."""
        while key not in self.probed:
            with self.probe_lock:
                gate = self.probing.get(key)
                mine = gate is None and key not in self.probed
                if mine:
                    gate = self.probing[key] = threading.Event()
            if gate is None:  # probed meanwhile
                break
            if not mine:
                gate.wait()  # then probed, or its probe raised: this caller probes
                continue
            try:
                self._probe(real, key)
            finally:
                with self.probe_lock:
                    del self.probing[key]
                gate.set()
        return self.probed[key]

    def _probe(self, real: Path, key: str) -> None:
        """Probe ``real`` into ``self.probed[key]`` (see :meth:`info`)."""
        try:
            i = probe(real)
            ok = (
                real.suffix.lower() in BROWSER_CONTAINERS
                and i.vcodec in BROWSER_VIDEO
                and (i.vcodec != "h264" or i.pix_fmt in {"yuv420p", "yuvj420p"})
                and i.acodec in BROWSER_AUDIO
                and i.audio_streams <= 1  # a browser plays one track; exports mix them all
            )
            color = preview_filter(i)
            self.still_color[key] = still_filter(i)
            self.fragmented[str(real)] = ok and fragmented_mp4(real)
            self.probed[key] = (ok, i.duration, i.audio_streams, "" if ok else color)
        except (FFmpegError, OSError, ValueError, KeyError):
            self.probed[key] = None

    def _proxy_key(self, key: str | None) -> str | None:
        """The key naming a source's proxies (whole, windows, builds): its content key, mixed
        with the colour filter of a probed source that has one (HDR, BT.2020, full-range
        BT.601), so a changed filter rebuilds its proxies while the others keep their names.
        Never runs ffprobe."""
        res = self.probed.get(key) if key else None
        if res is None or not res[3]:
            return key
        return hashlib.sha1(f"{key}|{res[3]}".encode()).hexdigest()[:16]

    def _lookup(
        self, real: Path, key: str | None = None
    ) -> tuple[str | None, tuple[bool, float, int, str] | None]:
        """``(content key, info)`` of ``real``; both None if it is missing. ``key``: its content
        key if the caller has it already (:meth:`MediaLookups.content_key`), so no stat."""
        key = key or self.key(real)
        return key, self.info(real, key) if key else None

    def duration(self, real: Path) -> float | None:
        res = self._lookup(real)[1]
        return res[1] if res else None

    def probed_duration(self, path: str, key: str | None = None) -> float | None:
        """The duration of ``path`` if it is probed already; never runs ffprobe. ``key``: its
        content key if known (see :meth:`_lookup`)."""
        key = key or self.key(remap_source(path))
        res = self.probed.get(key) if key else None
        return res[1] if res else None

    def status(self, project: Path, path: str, key: str | None = None) -> tuple[str, Path | None]:
        """``(status, file to serve)``: ready | needs-proxy | preparing | failed. No side effects.
        ``key``: the source's content key if known (see :meth:`_lookup`)."""
        real = remap_source(path)
        key, res = self._lookup(real, key)
        if res is None or key is None:
            return "failed", None
        if res[0]:
            return "ready", real
        key = self._proxy_key(key)
        proxy = project / UI_DIR / "proxy" / f"{key}.mp4"
        if proxy.is_file() and _inside(project, f"{UI_DIR}/proxy/{key}.mp4"):
            return "ready", proxy
        return {"preparing": "preparing", "failed": "failed"}.get(
            self.proxies.get((project, key), ""), "needs-proxy"
        ), None

    def cached_view(
        self, project: Path, path: str, key: str | None = None
    ) -> tuple[str, bool, float | None]:
        """``(the cached status, the source plays from a proxy, share of its whole proxy encoded or
        None while that is not encoding)``; never runs ffprobe. ``key``: the source's content key
        if known (see :meth:`_lookup`)."""
        key = key or self.key(remap_source(path))
        if key is None:  # missing: what status says, without a second stat
            return "failed", False, None
        if key not in self.probed:
            return "unknown", False, None
        status = self.status(project, path, key)[0]  # the key once: one stat of the source
        res = self.probed.get(key) if key else None
        pkey = self._proxy_key(key)
        with self.lock:
            done = self.progress.get((project, pkey)) if pkey and status == "preparing" else None
        return status, res is not None and not res[0], done

    def prepare(
        self,
        project: Path,
        path: str,
        window: tuple[float, float] | None = None,
        *,
        queue: bool = False,
    ) -> str:
        """Queue the proxy build of one source (also retries a failed build); returns its status.

        ``window``: a ``(lo, hi)`` range in seconds the user is looking at; a window proxy around
        it is queued first, in its own lane (see :meth:`_focus`). ``queue``: it waits behind the
        windows already queued instead of superseding them ("Play sequence" asks for its next
        clips so).
        """
        real = remap_source(path)
        content, res = self._lookup(real)
        key = self._proxy_key(content)
        if res is not None and key is not None and not res[0] and not self.closing:
            proxy = _own(project, f"{UI_DIR}/proxy/{key}.mp4")  # never built through a link
            _own(project, f"{UI_DIR}/proxy/{key}.part.mp4")  # nor its temporary
            if window is not None and not proxy.is_file():
                self._focus(project, real, key, res, *window, queue=queue)
            with self.lock:
                if not proxy.is_file() and self.proxies.get((project, key)) != "preparing":
                    self.proxies[project, key] = "preparing"
                    self.pool.submit(self._build, project, real, proxy, key, res[2], res[3], res[1])
        return self.status(project, path)[0]

    def _focus(
        self,
        project: Path,
        real: Path,
        key: str,
        res: tuple[bool, float, int, str],
        lo: float,
        hi: float,
        *,
        queue: bool = False,
    ) -> None:
        """The user looks at ``[lo, hi]`` s of a source: queue a window proxy around it unless a
        ready or running one covers it. The request goes ahead of (and drops) earlier queued
        ones, and a running window encode that does not overlap it is killed.

        ``queue``: it goes behind the queued ones instead, unless one of them covers it or
        PROXY_WINDOWS_QUEUED wait already (the page asks again later), and kills nothing."""
        dur = res[1]
        if dur > 0:
            lo, hi = min(lo, dur), min(hi, dur)
        step = PROXY_WINDOW_STEP
        a = max(0.0, math.floor((lo - PROXY_WINDOW_BEFORE) / step) * step)
        b = math.ceil((hi + PROXY_WINDOW_AFTER) / step) * step
        dur_cs = math.floor(dur * 100) if dur > 0 else 0
        span = (round(a * 100), min(math.floor(b * 100), dur_cs) if dur_cs else math.floor(b * 100))
        if span[1] <= span[0]:
            return
        want = (min(math.floor(lo * 100), span[1]), min(math.ceil(hi * 100), span[1]))
        win = ProxyWindow(project, real, key, *span, res[2], res[3])
        _own(project, f"{UI_DIR}/proxy/{win.name}.mp4")  # never built through a link
        _own(project, f"{UI_DIR}/proxy/{win.name}.part.mp4")  # nor its temporary
        with self.lock:
            ready = self.windows.get((project, key), {})
            covered = {s: f for s, f in ready.items() if _covers(s, want, dur_cs)}
        gone = [s for s, f in covered.items() if not f.is_file()]  # deleted by hand meanwhile
        kill = None
        with self.lock:
            if self.closing:
                return
            for s in gone:
                ready.pop(s, None)
            if not queue:
                self.window_lane.queue.clear()  # superseded: the user moved on
            run = self.window_lane.running
            mine = run is not None and (run.project, run.key) == (project, key)
            waits = queue and (
                len(self.window_lane.queue) >= PROXY_WINDOWS_QUEUED
                or any(
                    (w.project, w.key) == (project, key) and _covers((w.lo, w.hi), want, dur_cs)
                    for w in self.window_lane.queue
                )
            )
            if mine and not run.cancelled and _covers((run.lo, run.hi), want, dur_cs):
                pass  # the running encode covers it
            elif waits:
                pass  # a queued one covers it, or the queue is full
            elif len(covered) == len(gone):
                self.window_lane.queue.append(win)
                failed = self.window_failures.get((project, key), {})
                for sp in [sp for sp in failed if sp[0] < win.hi and win.lo < sp[1]]:
                    del failed[sp]  # asked for again: no longer a failure to show
                self.window_lane.wake()
            if (
                not queue
                and run is not None
                and not (mine and run.lo < span[1] and span[0] < run.hi)
            ):
                run.cancelled = True  # does not start if it has not yet
                kill, run.proc = run.proc, None
        if kill is not None:
            threading.Thread(
                target=procs.kill_tree, args=(kill, procs.SOFT_STOP_S), daemon=True
            ).start()

    def _build_window(self, win: ProxyWindow) -> None:
        p, name = win.project, f"{UI_DIR}/proxy/{win.name}"
        tmp, out = p / f"{name}.part.mp4", p / f"{name}.mp4"
        what = f"{win.real.name} {win.lo / 100:.2f}-{win.hi / 100:.2f} s"
        drop: list[Path] = []
        try:
            _ui_folder(p, "proxy")  # ui or ui/proxy a file: a clear refusal, not a traceback
            _own(p, f"{name}.part.mp4")  # a link made while this window was queued
            gpu = self._gpu_encoder()

            def encode(enc: VideoEncoder | None) -> tuple[int, str]:
                args = proxy_args(
                    win.real, tmp, win.audio_streams, encoder=enc, color=win.color,
                    lo=win.lo / 100, hi=win.hi / 100,
                )  # fmt: skip
                return self._encode(args, win)

            code, tail = encode(gpu)
            if code != 0 and gpu and not self.closing and not win.cancelled:
                logger.warning(
                    "window proxy of %s: %s failed (%s); retrying with libx264",
                    what, gpu.codec, tail,
                )  # fmt: skip
                code, tail = encode(None)
                self.gpu_failed = code == 0
            if code == 0 and not self.closing and not win.cancelled:
                _own(p, f"{name}.mp4")  # a link made during the encode
                with self.lock:
                    # the whole proxy landed meanwhile: this window is not needed
                    if not (p / UI_DIR / "proxy" / f"{win.key}.mp4").is_file():
                        tmp.replace(out)
                        drop = self._keep_window((p, win.key), (win.lo, win.hi), out)
            elif not win.cancelled and not self.closing:
                reason = _exit_reason(code, tail)
                logger.warning("window proxy of %s failed: %s", what, reason)
                self._window_failed(win, reason)
        except HTTPException as err:  # refused: the temporary is removed below
            logger.warning("window proxy of %s failed: %s", what, err.detail)
            self._window_failed(win, str(err.detail))
        except Exception as err:  # a failed window never stops the lane: listed as failed
            logger.exception("window proxy of %s failed", what)
            self._window_failed(win, str(err) or type(err).__name__)
        finally:
            with contextlib.suppress(OSError):  # e.g. NotADirectoryError when ui is a file
                tmp.unlink(missing_ok=True)
            self._delete(drop)

    def _window_failed(self, win: ProxyWindow, reason: str) -> None:
        """Record a failed window encode for the listing: its last line is the reason shown."""
        last = reason.splitlines()[-1] if reason else ""
        with self.lock:
            failed = self.window_failures.setdefault((win.project, win.key), {})
            failed.pop((win.lo, win.hi), None)
            failed[(win.lo, win.hi)] = last[:200]
            while len(failed) > PROXY_WINDOWS_MAX:
                del failed[next(iter(failed))]

    def failed_windows(
        self, project: Path, path: str, key: str | None = None
    ) -> list[tuple[int, int, str]]:
        """``(lo, hi, reason)`` of a source's failed window encodes, in centiseconds; no stat
        while the project has none. ``key``: the source's content key if known (see
        :meth:`_lookup`)."""
        with self.lock:
            if not any(p == project and f for (p, _), f in self.window_failures.items()):
                return []
        key = self._proxy_key(key or self.key(remap_source(path)))
        with self.lock:
            return [
                (lo, hi, r) for (lo, hi), r in self.window_failures.get((project, key), {}).items()
            ]

    def _keep_window(self, pid: tuple[Path, str], span: tuple[int, int], file: Path) -> list[Path]:
        """Record a ready window (call with ``self.lock`` held); returns the windows evicted for
        it, least recently used first, for the caller to delete."""
        ready = self.windows.setdefault(pid, {})
        ready.pop(span, None)
        ready[span] = file
        drop = []
        while len(ready) > PROXY_WINDOWS_MAX:
            drop.append(ready.pop(next(iter(ready))))
        return drop

    @staticmethod
    def _delete(files: list[Path]) -> None:
        for f in files:
            try:
                f.unlink(missing_ok=True)
            except OSError as err:  # e.g. still being sent on Windows; pruned next run
                logger.debug("could not delete %s: %s", f.name, err)

    def _drop_windows(self, project: Path, key: str) -> None:
        """The whole proxy of ``key`` landed: stop that source's window encodes and retire its
        windows (still served, no longer listed; the next run's :meth:`prune_proxies` deletes
        them, so a page playing one is not cut off)."""
        kill = None
        with self.lock:
            done = self.windows.pop((project, key), None)
            if done:
                self.retired.setdefault((project, key), {}).update(done)
            self.window_failures.pop((project, key), None)
            for win in [w for w in self.window_lane.queue if (w.project, w.key) == (project, key)]:
                self.window_lane.queue.remove(win)
            run = self.window_lane.running
            if run is not None and (run.project, run.key) == (project, key):
                run.cancelled = True
                kill, run.proc = run.proc, None
        if kill is not None:
            threading.Thread(
                target=procs.kill_tree, args=(kill, procs.SOFT_STOP_S), daemon=True
            ).start()

    def window(
        self, project: Path, path: str, lo: int, hi: int, key: str | None = None
    ) -> Path | None:
        """The ready window ``lo``-``hi`` (centiseconds) of a source, else None; marks it used.
        ``key``: the source's content key if known (see :meth:`_lookup`)."""
        key = self._proxy_key(key or self.key(remap_source(path)))
        if key is None:
            return None
        with self.lock:
            ready = self.windows.get((project, key))
            file = ready.pop((lo, hi), None) if ready else None
            if ready is not None and file is not None:
                ready[(lo, hi)] = file  # recently used: evicted last
            elif file is None:
                file = self.retired.get((project, key), {}).get((lo, hi))
        if (
            file is None
            or not file.is_file()
            or not _inside(project, f"{UI_DIR}/proxy/{file.name}")
        ):
            return None
        _mark_used(file)
        return file

    def windows_of(
        self, project: Path, path: str, key: str | None = None
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int, str]]]:
        """``(ready windows, pending ones with "encoding" | "queued")`` of a source, in
        centiseconds, from memory only; no stat at all while the project has none. ``key``: the
        source's content key if known (see :meth:`_lookup`)."""
        with self.lock:
            busy = [
                w for w in (self.window_lane.running, *self.window_lane.queue)
                if w is not None and w.project == project and not w.cancelled
            ]  # fmt: skip
            any_ready = any(p == project and r for (p, _), r in self.windows.items())
        if not busy and not any_ready:
            return [], []
        key = self._proxy_key(key or self.key(remap_source(path)))
        with self.lock:
            ready = sorted(self.windows.get((project, key), {}))
            run = self.window_lane.running
        pending = [(w.lo, w.hi, "encoding" if w is run else "queued") for w in busy if w.key == key]
        return ready, pending

    def prune_proxies(self, project: Path, paths: list[str]) -> None:
        """Once per run and project: delete leftover temporaries, the whole proxies and windows
        of sources that changed or left or whose tone map changed, and windows whose whole proxy
        is built; adopt the other windows on disk. A file written or used in the last
        PROXY_PRUNE_GRACE_S stays (another run may be encoding or serving it), as does one that
        cannot be deleted (open on Windows); a later run deletes it. While a source cannot be
        read (an offline drive), a proxy or window naming no current source goes only once idle
        for PROXY_OFFLINE_PRUNE_S (it may be the offline source's), and the readable sources are
        probed only when such a proxy has been idle that long.

        A proxy no known key names may be of a source with a colour filter (HDR, BT.2020,
        full-range BT.601) not probed yet (its proxy key holds the filter): sources are then
        probed, in order, until every such proxy is matched, so this may run ffprobe. Proxies
        and windows such a source's bare content key names (built before it had a filter) go
        once a proxy or window of its current filter exists, on a later run."""
        folder = project / UI_DIR / "proxy"
        if project in self.proxies_pruned:
            return
        cutoff = time.time_ns() - int(PROXY_PRUNE_GRACE_S * 1e9)
        offline_cutoff = time.time_ns() - int(PROXY_OFFLINE_PRUNE_S * 1e9)
        reals = [remap_source(path) for path in paths]
        keys = {self._proxy_key(self.key(real)) for real in reals}
        unsure = None in keys  # a source is unreachable: its proxies' names are unknown
        keys.discard(None)
        if unsure:  # probe only for a proxy idle long enough to go if no source names it
            left = {
                m[1]
                for f in folder.glob("*.mp4")
                if (m := PROXY_FILE.fullmatch(f.name))
                and m[1] not in keys
                and _idle(f, offline_cutoff)
            }
        else:
            named = {
                m[1] for m in (PROXY_FILE.fullmatch(f.name) for f in folder.glob("*.mp4")) if m
            }
            left = named - keys
        for real in reals:
            if not left:
                break
            key, res = self._lookup(real)
            if res is not None and res[3]:  # only a colour filter changes the key
                pkey = self._proxy_key(key)
                keys.discard(key)  # its windows from an earlier tone map go
                keys.add(pkey)
                left.discard(pkey)
        drop: list[Path] = []
        with self.lock:
            self.proxies_pruned.add(project)
            if not folder.is_dir() or not _inside(project, f"{UI_DIR}/proxy"):
                return  # never delete in a folder a link takes outside the project
            busy = {
                f"{k}.part.mp4"
                for (q, k), st in self.proxies.items()
                if q == project and st == "preparing"
            }
            busy |= {
                f"{w.name}.part.mp4"
                for w in (self.window_lane.running, *self.window_lane.queue)
                if w is not None and w.project == project
            }
            drop += [f for f in folder.glob("*.part.mp4") if f.name not in busy]
            drop += [
                f
                for f in folder.glob("*.mp4")
                if (m := PROXY_FILE.fullmatch(f.name))
                and ".w" not in f.name
                and m[1] not in keys
                and (not unsure or _idle(f, offline_cutoff))
            ]  # whole proxies
            found: list[tuple[int, Path, str, tuple[int, int]]] = []
            for f in folder.glob("*.w*.mp4"):
                m = PROXY_WINDOW_FILE.fullmatch(f.name)
                if m is None:
                    continue
                k = m.group(1)
                if (folder / f"{k}.mp4").is_file():
                    drop.append(f)
                elif k not in keys:
                    if not unsure or _idle(f, offline_cutoff):
                        drop.append(f)
                else:
                    with contextlib.suppress(OSError):
                        found.append((f.stat().st_mtime_ns, f, k, (int(m[2]), int(m[3]))))
            for _, f, k, span in sorted(found):  # oldest first: evicted first
                if span not in self.windows.get((project, k), {}):
                    drop += self._keep_window((project, k), span, f)
        self._delete([f for f in drop if _idle(f, cutoff)])

    def _build(
        self,
        project: Path,
        real: Path,
        proxy: Path,
        key: str,
        audio_streams: int,
        color: str = "",
        duration: float = 0.0,
    ) -> None:
        tmp = proxy.with_name(f"{key}.part.mp4")
        state = "failed"
        made: list[Path] = []  # ui folders this build created
        try:
            # ui or ui/proxy a file: a clear refusal, not a traceback
            _ui_folder(project, "proxy", made)
            # a link made while this build was queued
            _own(project, f"{UI_DIR}/proxy/{key}.part.mp4")
            gpu = self._gpu_encoder()
            track = ((project, key), duration)  # the page shows how far the whole proxy got
            code, tail = self._encode(
                proxy_args(real, tmp, audio_streams, encoder=gpu, color=color), progress=track
            )
            if code != 0 and gpu and not self.closing:
                # The probe encodes 0.2 s of black: a driver can still fail on real footage.
                logger.warning(
                    "proxy of %s: %s failed (%s); retrying with libx264", real.name, gpu.codec, tail
                )
                code, tail = self._encode(
                    proxy_args(real, tmp, audio_streams, color=color), progress=track
                )
                self.gpu_failed = code == 0
            if code == 0 and not self.closing:
                _own(project, f"{UI_DIR}/proxy/{key}.mp4")  # a link made during the encode
                tmp.replace(proxy)
                state = ""
                self._drop_windows(project, key)  # the whole proxy replaces them
            else:
                logger.warning("proxy of %s failed: %s", real.name, tail)
        except HTTPException as err:  # refused: the temporary is removed below
            logger.warning("proxy of %s failed: %s", real.name, err.detail)
        except Exception:  # in the pool, a raised error would be lost in a Future: log it
            logger.exception("proxy of %s failed", real.name)
        finally:
            with contextlib.suppress(OSError):  # e.g. NotADirectoryError when ui is a file
                tmp.unlink(missing_ok=True)
            with self.lock:
                self.progress.pop((project, key), None)
                if state:
                    self.proxies[project, key] = state
                else:
                    self.proxies.pop((project, key), None)
                # Accepted narrow race: a request writing into a just-made ui/ right now gets one
                # retryable error. Queued builds (one worker) recreate the folders themselves.
                if state:
                    for folder in made:  # failed: the folders it created, while still empty
                        with contextlib.suppress(OSError):
                            folder.rmdir()

    def _gpu_encoder(self) -> VideoEncoder | None:
        """The GPU encoder for proxies: the one exports use (``video_encoder``, probed once),
        or None for libx264, also after a proxy failed with it and worked with libx264. AMF
        proxies stay on libx264: their size was never measured (VAAPI's was, see below)."""
        if self.gpu_failed:
            return None
        try:
            enc = video_encoder()
        except (ValueError, FFmpegError):  # also EncoderUnavailableError: libx264 then
            return None
        return None if enc.kind in ("cpu", "amf") else enc

    def _encode(
        self,
        args: list[str],
        window: ProxyWindow | None = None,
        progress: tuple[tuple[Path, str], float] | None = None,
    ) -> tuple[int, str]:
        """Run one proxy encode; ``(exit code, end of its output)``.

        Spawned and registered under ``self.lock`` once ``closing`` (and the ``window``'s
        ``cancelled``) is checked, so ``shutdown`` (or :meth:`_focus`) either kills it or it
        never starts. ``window.proc`` holds the process while it runs; a window encodes at normal
        priority (the user waits on it, and the whole proxy beside it would starve it).
        ``progress``: ``((project, key), source duration)``; ffmpeg reports how far it got, kept
        in ``self.progress[(project, key)]``.
        """
        if progress is not None:
            args = [args[0], "-progress", "pipe:1", "-nostats", *args[1:]]
        with self.lock:
            if self.closing:
                return 1, "the server is shutting down"
            if window is not None and window.cancelled:
                return 1, "no longer needed"
            proc = spawn(args, low_priority=window is None)
            self.procs.add(proc)
            if window is not None:
                window.proc = proc
            if progress is not None:
                self.progress[progress[0]] = 0.0
        try:
            if progress is None:
                tail = _tail(proc.communicate()[0] or "")  # stderr is merged into it
            else:
                tail = self._read_progress(proc, *progress)
        finally:
            with self.lock:
                self.procs.discard(proc)
                if window is not None and window.proc is proc:
                    window.proc = None
        return proc.returncode, tail

    def _read_progress(
        self, proc: subprocess.Popen[str], key: tuple[Path, str], duration: float
    ) -> str:
        """Read an encode's output (``-progress pipe:1`` merged with its errors) until it exits:
        ``out_time_us`` updates ``self.progress[key]``; returns the end of the other lines."""
        rest: deque[str] = deque(maxlen=20)
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.strip()
            m = PROXY_PROGRESS_LINE.fullmatch(line)
            if m is None:
                if line:
                    rest.append(line)
            elif m[1] == "out_time_us" and m[2].isdigit() and duration > 0:  # "N/A" at first
                # Never lower: ffmpeg reports the furthest stream not yet finished, so it steps
                # back when audio ends ahead of a video encode that is still catching up.
                with self.lock:
                    done = min(1.0, int(m[2]) / 1e6 / duration)
                    self.progress[key] = max(self.progress.get(key, 0.0), done)
        proc.wait()
        return "\n".join(rest)[-500:]

    def frame(
        self, project: Path, path: str, at: float, width: int, *, key: bool = False
    ) -> Path | None:
        """A JPEG still of ``path`` at ``at`` s (clamped into the source), ``width`` px wide.

        Cached as ``ui/frames/<content key>-<centiseconds>-<width>.jpg`` and built once even when
        requested concurrently. A still goes through its source's
        :func:`~cutpoint.encode.still_filter` (HDR, BT.2020 and BT.709 sources), so it shows what
        exports show, and then its name ends in ``-<filter hash>``, so stills cached before the
        conversion (or a changed one) are not reused. ``key``: the keyframe at or before ``at``
        (fast, for the filmstrip), cached with ``-k`` after the width, so exact stills never get one.
        None if the source cannot be read.

        Raises:
            FFmpegError: If ffmpeg decodes no image.
        """
        real = remap_source(path)
        content, res = self._lookup(real)
        if res is None or content is None:
            return None
        at = min(at, FRAME_MAX_AT)  # a huge time (1e308) would overflow its centiseconds
        if res[1] > 0:
            at = min(at, max(0.0, res[1] - 0.1))  # the last instants may hold no decodable frame
        cs = round(at * 100)
        color = self.still_color.get(content, "")
        name = f"{content}-{cs}-{width}" + ("-k" if key else "")
        if color:  # a converted still never reuses one cached before (or with another filter)
            name += f"-{hashlib.sha1(color.encode()).hexdigest()[:8]}"
        out = _own(project, f"{UI_DIR}/frames/{name}.jpg")  # nor cached through one
        if out.is_file():
            with contextlib.suppress(OSError):
                os.utime(out)  # recently used: evicted last
            return out
        with self.lock:
            lock, users = self.frame_locks.get(out.name, (threading.Lock(), 0))
            self.frame_locks[out.name] = (lock, users + 1)
        try:
            with lock:
                if not out.is_file():
                    with self.frame_slots:
                        image = grab_images(
                            real,
                            cs / 100,
                            width=width,
                            quality=5,
                            timeout=FRAME_TIMEOUT_S,
                            color=color,
                            keyframe=key,
                        )[0]
                    _ui_folder(project, "frames")  # ui or ui/frames a file: a clear 409
                    tmp = _own(project, f"{UI_DIR}/frames/{out.stem}.{threading.get_ident()}.part")
                    try:
                        tmp.write_bytes(image)
                        tmp.replace(out)
                    finally:
                        with contextlib.suppress(OSError):  # keeps the write's own error
                            tmp.unlink(missing_ok=True)
                    self._evict_frames(out.parent)
        finally:
            with self.lock:
                lock, users = self.frame_locks[out.name]
                if users > 1:
                    self.frame_locks[out.name] = (lock, users - 1)
                else:
                    del self.frame_locks[out.name]
        return out

    def _drop_frames(self, files: list[Path]) -> None:
        """Delete cached stills (call with ``self.lock`` held), skipping any being built."""
        for f in files:
            if f"{f.name.split('.')[0]}.jpg" in self.frame_locks:
                continue
            try:
                f.unlink(missing_ok=True)
            except OSError as err:  # e.g. still being sent on Windows; next time
                logger.debug("could not delete %s: %s", f.name, err)

    def _evict_frames(self, folder: Path) -> None:
        """Keep at most ``FRAME_CACHE_MAX`` stills in ``folder``, dropping the least recently used.

        Each new still only counts the folder's names; the stills are stat-ed only once there are
        more than the cap, and then evicted down to ``FRAME_EVICT_SHARE`` of it below the cap, so
        a full folder pays the stats once per that many new stills, not on each one (A746). No
        count is kept in memory: other processes' stills and a restart count as well.

        The folder is listed without ``self.lock`` (1500 stats would hold up every other request);
        two evictions at once pick the same oldest stills, and ``_drop_frames`` (under the lock)
        still skips any being built."""
        try:
            with os.scandir(folder) as found:
                names = [e for e in found if e.name.endswith(".jpg")]
        except OSError:
            return
        if len(names) <= FRAME_CACHE_MAX:
            return
        stills: list[tuple[int, Path]] = []
        for e in names:
            with contextlib.suppress(OSError):  # e.stat: no second call on Windows
                stills.append((e.stat().st_mtime_ns, Path(e.path)))
        keep = FRAME_CACHE_MAX - int(FRAME_CACHE_MAX * FRAME_EVICT_SHARE)
        stills.sort()
        with self.lock:
            self._drop_frames([f for _, f in stills[: max(0, len(stills) - keep)]])

    def prune_frames(self, project: Path, paths: list[str]) -> None:
        """Once per run and project: delete stills of sources that changed or left, and leftovers."""
        folder = project / UI_DIR / "frames"
        if project in self.frames_pruned:
            return
        keys = {self.key(remap_source(path)) for path in paths} - {None}
        with self.lock:
            self.frames_pruned.add(project)
            if not folder.is_dir() or not _inside(project, f"{UI_DIR}/frames"):
                return  # never delete in a folder a link takes outside the project
            stale = [f for f in folder.glob("*.jpg") if f.name.split("-")[0] not in keys]
            self._drop_frames([*stale, *folder.glob("*.part")])

    def envelope(self, project: Path, path: str) -> dict[str, Any] | None:
        """The audio envelope of ``path``: ``{"step": seconds per value, "peaks": [0-1, ...]}``,
        ``peaks`` empty for a source without audio. None if the source cannot be read.
        ``{"pending": True}`` while its decode runs: the first call queues it, later ones
        (the page asks again) find it cached.

        Cached as ``ui/envelope/<content key>-<version>.json``; decoded one source at a time on
        its own worker, once even when requested concurrently (see :mod:`cutpoint.envelope`), at
        low priority and streamed, so a long source costs no memory. A cached source answers at
        once, whatever decodes.

        Raises:
            FFmpegError: If the decode of an earlier call failed or ran longer than
                ``ENVELOPE_TIMEOUT_S`` (reported once), or the server is shutting down.
        """
        real = remap_source(path)
        content, res = self._lookup(real)
        if res is None or content is None:
            return None
        if res[2] == 0:
            return {"step": audio_envelope.BASE_S, "peaks": []}
        out = _own(project, f"{UI_DIR}/envelope/{content}-{ENVELOPE_VERSION}.json")
        hit = cache.load_json(out)
        if isinstance(hit, dict) and isinstance(hit.get("peaks"), list):
            return hit
        with self.lock:
            if content in self.envelope_errors:
                raise FFmpegError(self.envelope_errors.pop(content))
            if self.closing:
                raise FFmpegError("the server is shutting down")
            self.envelope_asked[content] = time.monotonic()
            if content not in self.envelopes_running:
                self.envelopes_running.add(content)
                self.envelope_pool.submit(
                    self._decode_envelope, project, real, content, res[2], out
                )
        return {"pending": True}

    def _decode_envelope(self, project: Path, real: Path, content: str, tracks: int, out: Path):
        """Decode one source's envelope into ``out`` (on the envelope worker); a failure is kept
        in ``envelope_errors`` for the next ``envelope`` call. A job nobody asked for in the last
        ``ENVELOPE_STALE_S`` (the page moved on to another source) is dropped unrun."""
        try:
            with self.lock:
                asked = self.envelope_asked.get(content, 0.0)
            if time.monotonic() - asked > ENVELOPE_STALE_S:
                return
            hit = cache.load_json(out)
            if not (isinstance(hit, dict) and isinstance(hit.get("peaks"), list)):
                self._write_envelope(project, real, tracks, out)
        except Exception as err:
            logger.exception("envelope decode of %s failed", real.name)
            with self.lock:
                self.envelope_errors[content] = str(getattr(err, "detail", "") or err)
        finally:
            with self.lock:
                self.envelopes_running.discard(content)
                self.envelope_asked.pop(content, None)

    def _write_envelope(self, project: Path, real: Path, tracks: int, out: Path) -> None:
        _ui_folder(project, "envelope")
        with self.lock:  # as in ``_encode``: shutdown sees the process, or none starts
            if self.closing:
                raise FFmpegError("the server is shutting down")
            proc = spawn(audio_envelope.ffmpeg_args(str(real), tracks), binary=True)
            self.procs.add(proc)
        timed_out = threading.Event()

        def expire() -> None:
            timed_out.set()
            procs.kill_tree(proc, 0.0)

        timer = threading.Timer(ENVELOPE_TIMEOUT_S, expire)
        timer.daemon = True
        timer.start()
        try:
            assert proc.stdout is not None
            peaks = audio_envelope.read_peaks(proc.stdout)
            proc.wait()
        finally:
            timer.cancel()
            if proc.poll() is None:
                procs.kill_tree(proc, 0.0)
                proc.wait()
            if proc.stdout is not None:
                proc.stdout.close()
            with self.lock:
                self.procs.discard(proc)
        if timed_out.is_set():
            raise FFmpegError(f"decoding the audio took longer than {ENVELOPE_TIMEOUT_S:.0f} s")
        if proc.returncode != 0:
            raise FFmpegError(f"ffmpeg could not decode the audio (exit {proc.returncode})")
        step, values = audio_envelope.shrink(peaks)
        cache.save_bytes(
            out, json.dumps({"step": step, "peaks": values}, separators=(",", ":")).encode()
        )

    def prune_envelopes(self, project: Path, paths: list[str]) -> None:
        """Once per run and project: delete the envelopes of sources that changed or left."""
        folder = project / UI_DIR / "envelope"
        if project in self.envelopes_pruned:
            return
        self.envelopes_pruned.add(project)
        if not folder.is_dir() or not _inside(project, f"{UI_DIR}/envelope"):
            return
        keep = {
            f"{key}-{ENVELOPE_VERSION}.json"
            for key in {self.key(remap_source(path)) for path in paths} - {None}
        }
        for f in folder.glob("*.json"):
            if f.name not in keep:
                with contextlib.suppress(OSError):
                    f.unlink()

    def encoding_for(self, project: Path) -> bool:
        """A whole proxy or a window proxy of ``project`` is queued or encoding now."""
        with self.lock:
            return any(
                q == project and st == "preparing" for (q, _), st in self.proxies.items()
            ) or any(
                w is not None and w.project == project
                for w in (self.window_lane.running, *self.window_lane.queue)
            )

    def forget(self, project: Path) -> None:
        """Drop what is remembered of ``project`` (it was deleted): a project made later under
        the same name starts clean."""
        with self.lock:
            for table in (
                self.proxies,
                self.progress,
                self.windows,
                self.retired,
                self.window_failures,
            ):
                for k in [k for k in table if k[0] == project]:
                    del table[k]
            self.proxies_pruned.discard(project)
            self.frames_pruned.discard(project)
            self.envelopes_pruned.discard(project)

    def shutdown(self) -> None:
        with self.lock:  # as in ``_encode``: no proxy ffmpeg starts after this
            self.closing = True
            running = list(self.procs)
            self.window_lane.close()  # the window lane ends
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.envelope_pool.shutdown(wait=False, cancel_futures=True)
        for proc in running:
            # a bare ffmpeg: no soft stop of its own to wait for
            procs.kill_tree(proc, procs.SOFT_STOP_S)
