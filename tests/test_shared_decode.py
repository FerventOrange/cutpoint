"""Shared decode of ``analyze`` (#375 step 2, ``CUTPOINT_ANALYZE_SHARED_DECODE``): the split
pass gives the frames and scene proxy of the separate decodes, its spool and failure paths
leave no proxy or spool behind, the scene share's states, the proxy control mode, and the
gates (synthetic testsrc clips; CUDA rows skip without nvidia-smi)."""

from __future__ import annotations

import functools
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import needs_ffmpeg
from cutpoint import analyze, ffmpeg_utils
from cutpoint.events import ocr_prefetch
from cutpoint.events.common import Source
from cutpoint.events.killfeed import OcrParams, ocr_vf
from cutpoint.ffmpeg_utils import FFmpegError, SplitDecode

RECT = (40, 20, 120, 60)  # x, y, w, h of the OCR crop
SIZE = RECT[2:]
VF = ocr_vf(3.0, RECT)
SCENE_VF = "fps=10,scale=320:-2"
ENC = ("-c:v", "libx264", "-preset", "ultrafast", "-crf", "32")
HW = [
    False,
    pytest.param(
        True, marks=pytest.mark.skipif(not shutil.which("nvidia-smi"), reason="no NVIDIA GPU")
    ),
]


def _clip(path: Path, seconds: int = 4) -> Path:
    """Two different test patterns of ``seconds`` each: one scene cut between them."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=30:duration={seconds}",
         "-filter_complex", "[0][1]concat=n=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-g", "30", str(path)],
        check=True,
    )  # fmt: skip
    return path


def _framemd5(path: Path) -> str:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "framemd5", "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _split(clip: Path, cdir: Path, outs: list, cancel=None, hw=False) -> SplitDecode:
    return SplitDecode(
        clip,
        size=SIZE,
        raw_vf=VF,
        stamp_fps=3.0,
        scene_vf=SCENE_VF,
        scene_enc=ENC,
        proxy=cdir / "proxy.mp4",
        spool=cdir / ffmpeg_utils.SPOOL_DIR,
        cancel=cancel or threading.Event(),
        on_exit=lambda ok, peak: outs.append((ok, peak)),
        hw=hw,
    )


def _left(cdir: Path) -> set[str]:
    return {p.name for p in cdir.iterdir()}


# ----------------------------------------------------------------- byte identity
@needs_ffmpeg
@pytest.mark.parametrize("hw", HW)
def test_split_gives_the_frames_proxy_and_cuts_of_separate_decodes(tmp_path, monkeypatch, hw):
    clip = _clip(tmp_path / "clip.mp4")
    monkeypatch.setattr(ffmpeg_utils, "SPOOL_SEGMENT_FRAMES", 7)  # several spool files
    serial = tmp_path / "serial"
    serial.mkdir()
    ref = [
        (t, f.tobytes())
        for t, f in ffmpeg_utils.decode_raw(
            clip, size=SIZE, vf=VF, pix_fmt="bgr24", stamp_fps=3.0,
            hwaccel="cuda" if hw else None, fallback_cpu=False,
        )
    ]  # fmt: skip
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *(["-hwaccel", "cuda"] if hw else []), "-i", str(clip),
         "-an", "-vf", SCENE_VF, *ENC, str(serial / "proxy.mp4")],
        check=True,
    )  # fmt: skip
    cdir = tmp_path / "shared"
    cdir.mkdir()
    outs: list = []
    codes: list[int] = []
    unlinked: list[str] = []
    real_unlink = Path.unlink

    def unlink(self, *a, **k):
        if self.name.startswith("seg-"):
            unlinked.append(self.name)
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", unlink)
    got = [(t, f.tobytes()) for t, f in _split(clip, cdir, outs, hw=hw).frames(codes)]
    assert got == ref and len(ref) == 24
    assert outs[0][0] and 0 < outs[0][1] <= len(ref) * SIZE[0] * SIZE[1] * 3 and codes == []
    assert unlinked == [f"seg-{i:05d}.raw" for i in range(3)]  # each full file once read
    assert _left(cdir) == {"proxy.mp4"}  # the spool and the tmp proxy are gone
    assert _framemd5(cdir / "proxy.mp4") == _framemd5(serial / "proxy.mp4")
    a = analyze._detect(serial / "proxy.mp4", serial / "scenes.json")
    b = analyze._detect(cdir / "proxy.mp4", cdir / "scenes.json")
    assert a == b == [4.0]
    assert (serial / "scenes.json").read_bytes() == (cdir / "scenes.json").read_bytes()


# ----------------------------------------------------------------- failure paths
@needs_ffmpeg
@pytest.mark.parametrize("call", ["open", "unlink"])
def test_a_spool_file_held_by_a_scanner_never_fails_the_read(tmp_path, monkeypatch, call):
    """WinError 32 on a spool file: a failed open ends the shared read (the OCR read in its
    turn then decodes as before), a failed delete is left to close(); no spool is left."""
    clip = _clip(tmp_path / "clip.mp4")
    monkeypatch.setattr(ffmpeg_utils, "SPOOL_SEGMENT_FRAMES", 7)
    real_open, real_unlink = Path.open, Path.unlink

    def opened(self, mode="r", *a, **k):
        if call == "open" and mode == "rb" and self.name.startswith("seg-"):
            raise PermissionError(13, "used by another process")
        return real_open(self, mode, *a, **k)

    def unlink(self, *a, **k):
        if call == "unlink" and self.name.startswith("seg-"):
            raise PermissionError(13, "used by another process")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(Path, "unlink", unlink)
    cdir = tmp_path / "c"
    cdir.mkdir()
    outs: list = []
    frames = _split(clip, cdir, outs).frames([])
    if call == "open":
        with pytest.raises(FFmpegError, match="spool read failed"):
            list(frames)  # a RuntimeError: OcrCancelled in the background read
    else:
        assert len(list(frames)) == 24
    assert ffmpeg_utils.SPOOL_DIR not in _left(cdir)


@needs_ffmpeg
def test_a_paused_read_closes_its_spool_file_before_the_spool_goes(tmp_path, monkeypatch):
    clip = _clip(tmp_path / "clip.mp4")
    cdir = tmp_path / "c"
    cdir.mkdir()
    outs: list = []
    split = _split(clip, cdir, outs)
    frames = split.frames([])
    next(frames)  # paused, as on the worker's error path
    assert split._seg is not None
    split.close()
    assert split._seg is None and ffmpeg_utils.SPOOL_DIR not in _left(cdir)
    frames.close()


@needs_ffmpeg
def test_run_cancellable(tmp_path):
    clip = _clip(tmp_path / "clip.mp4", seconds=1)
    out = tmp_path / "p.mp4"
    args = ["-v", "error", "-y", "-i", str(clip), "-an", "-vf", SCENE_VF, *ENC, str(out)]
    assert ffmpeg_utils.run_cancellable(args, threading.Event()) and out.stat().st_size
    stopped = threading.Event()
    stopped.set()
    assert not ffmpeg_utils.run_cancellable(["-v", "error", "-re", *args[2:]], stopped)
    assert not ffmpeg_utils.run_cancellable(["-v", "error", "-i", str(tmp_path / "x.mp4"),
                                             str(out)], threading.Event())  # fmt: skip


@needs_ffmpeg
def test_a_spool_write_error_fails_both_sides(tmp_path, monkeypatch):
    clip = _clip(tmp_path / "clip.mp4")
    real = Path.open

    def broken(self, mode="r", *a, **k):
        if mode == "wb" and self.name.startswith("seg-"):
            raise OSError(28, "No space left on device")
        return real(self, mode, *a, **k)

    monkeypatch.setattr(Path, "open", broken)
    cdir = tmp_path / "c"
    cdir.mkdir()
    outs: list = []
    with pytest.raises(FFmpegError, match="spool write failed"):
        list(_split(clip, cdir, outs).frames([]))
    assert outs == [(False, 0)] and _left(cdir) == set()


@needs_ffmpeg
def test_a_failed_ffmpeg_gives_no_frames_and_no_proxy(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    cdir = tmp_path / "c"
    cdir.mkdir()
    outs: list = []
    codes: list[int] = []
    with pytest.raises(FFmpegError, match="no frames"):
        list(_split(bad, cdir, outs).frames(codes))
    assert outs == [(False, 0)] and codes and codes[0] != 0  # an exit it did not cause
    assert _left(cdir) == set()


@needs_ffmpeg
@pytest.mark.parametrize("how", ["cancel", "close"])
def test_a_stopped_split_keeps_a_proxy_only_when_complete(tmp_path, how):
    """A cancel or a consumer that stops early kills a running ffmpeg: no proxy then, no
    exit code reported; ffmpeg may have finished first, and then its proxy is whole."""
    clip = _clip(tmp_path / "clip.mp4", seconds=20)
    cdir = tmp_path / "c"
    cdir.mkdir()
    outs: list = []
    codes: list[int] = []
    cancel = threading.Event()
    frames = _split(clip, cdir, outs, cancel).frames(codes)
    next(frames)
    if how == "cancel":
        cancel.set()
        rest = list(frames)
        assert len(rest) < 119
    else:
        frames.close()
    ((ok, _peak),) = outs
    assert codes == []
    assert _left(cdir) == ({"proxy.mp4"} if ok else set())
    if ok:
        assert _framemd5(cdir / "proxy.mp4").count("\n") > 100


# ----------------------------------------------------------------- stream gate
@needs_ffmpeg
def test_cover_art_or_two_video_streams_refuse_the_share(tmp_path):
    clip = _clip(tmp_path / "clip.mp4", seconds=1)
    assert ffmpeg_utils.single_video_stream(clip)
    pic, art, two = tmp_path / "pic.png", tmp_path / "art.mp4", tmp_path / "two.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64", "-frames:v", "1",
         str(pic)],
        check=True,
    )  # fmt: skip
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(clip), "-i", str(pic), "-map", "0", "-map", "1",
         "-c", "copy", "-disposition:v:1", "attached_pic", str(art)],
        check=True,
    )  # fmt: skip
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(clip), "-i", str(clip), "-map", "0:v", "-map", "1:v",
         "-c", "copy", str(two)],
        check=True,
    )  # fmt: skip
    assert not ffmpeg_utils.single_video_stream(art)
    assert not ffmpeg_utils.single_video_stream(two)
    assert not ffmpeg_utils.single_video_stream(tmp_path / "missing.mp4")


# ----------------------------------------------------------------- scene share
def _plan(cdir: Path, duration: float = 60.0) -> ocr_prefetch.ScenePlan:
    cdir.mkdir(parents=True, exist_ok=True)
    return ocr_prefetch.ScenePlan(cdir / "src.mp4", cdir, duration, SCENE_VF, ENC)


def test_a_share_not_started_is_declined_by_the_scene_cuts(tmp_path):
    share = ocr_prefetch.SceneShare(_plan(tmp_path), "split")
    assert share.result() is None and share.state == "declined"
    assert not share.claim()  # the worker then decodes the OCR only
    assert "declined (not started" in share.summary()


def test_the_scene_cuts_wait_for_a_running_share(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_prefetch, "POLL_S", 0.01)
    share = ocr_prefetch.SceneShare(_plan(tmp_path), "split")
    assert share.claim() and not share.claim()
    threading.Timer(0.1, share.finish, args=(True, 3 * 2**20)).start()
    assert share.result() == tmp_path / "proxy.mp4"
    assert share.waited >= 0.05
    assert share.summary().startswith("done, waited") and "spool peak 3 MiB" in share.summary()
    failed = ocr_prefetch.SceneShare(_plan(tmp_path), "proxy")
    failed.claim()
    failed.finish(False)
    assert failed.result() is None and failed.state == "failed"


def test_shared_mode_env(monkeypatch):
    for value, want in [("split", "split"), (" Proxy ", "proxy"), ("0", None), ("x", None)]:
        monkeypatch.setenv("CUTPOINT_ANALYZE_SHARED_DECODE", value)
        assert ocr_prefetch.shared_mode() == want
    monkeypatch.delenv("CUTPOINT_ANALYZE_SHARED_DECODE")
    assert ocr_prefetch.shared_mode() is None


# ----------------------------------------------------------------- scene cuts
def test_scene_cuts_use_a_made_proxy_and_decode_otherwise(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(analyze, "run_progress", lambda args, total, label: calls.append(args))
    monkeypatch.setattr(analyze, "_detect", lambda proxy, path: [proxy.name])
    share = SimpleNamespace(result=lambda: tmp_path / "proxy.mp4")
    assert analyze.scene_cuts(tmp_path / "s.mp4", tmp_path, 60.0, shared=share) == ["proxy.mp4"]
    assert calls == []
    share = SimpleNamespace(result=lambda: None)  # declined or failed: as before
    analyze.scene_cuts(tmp_path / "s.mp4", tmp_path, 60.0, shared=share)
    analyze.scene_cuts(tmp_path / "s.mp4", tmp_path, 60.0)
    step1 = ["ffmpeg", "-y", "-v", "error", "-hwaccel", "cuda", "-i", str(tmp_path / "s.mp4"),
             "-an", "-vf", "fps=10,scale=320:-2", "-c:v", "libx264", "-preset", "ultrafast",
             "-crf", "32", str(tmp_path / "proxy.mp4")]  # fmt: skip
    assert calls == [step1, step1]  # the step 1 command, unchanged


def test_scene_plan_only_for_a_whole_short_uncached_single_stream_source(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze.cache, "source_dir", lambda src, offset=0.0: tmp_path)
    monkeypatch.setattr(analyze, "single_video_stream", lambda io: io.name != "art.mp4")
    plan = analyze._scene_plan(tmp_path / "a.mp4", 600.0)
    assert plan is not None
    assert (plan.vf, plan.enc, plan.cdir, plan.duration) == (SCENE_VF, ENC, tmp_path, 600.0)
    assert analyze._scene_plan(tmp_path / "a.mp4", analyze.LONG_FILE_S + 1) is None
    assert analyze._scene_plan(tmp_path / "art.mp4", 600.0) is None
    (tmp_path / "scenes.json").write_text("[1.0]", encoding="utf-8")
    assert analyze._scene_plan(tmp_path / "a.mp4", 600.0) is None


# ----------------------------------------------------------------- the prefetch side
@pytest.fixture
def pre(tmp_path, monkeypatch):
    """An OcrPrefetch in a shared-decode mode with a fake source and OCR read."""
    state = SimpleNamespace(decoded=None)

    def load_source(path, duration):
        return Source(Path(path), duration, tmp_path / "cache", [])

    def detect(src, params, engine, *, decode=None, **kw):
        if decode is not None:
            state.decoded = decode(RECT, VF, [], [])
        return [], {"frames": 0}

    monkeypatch.setattr(ocr_prefetch, "load_source", load_source)
    monkeypatch.setattr(ocr_prefetch, "detect_ocr_events", detect)
    monkeypatch.setattr(ocr_prefetch, "load_preset", lambda game, project: True)
    monkeypatch.setattr(ocr_prefetch, "vram_ok", lambda: True)
    src = tmp_path / "a.mp4"
    state.src = src
    (tmp_path / "cache").mkdir()  # load_source's cache.source_dir makes it

    def make(mode: str) -> ocr_prefetch.OcrPrefetch:
        monkeypatch.setenv("CUTPOINT_ANALYZE_SHARED_DECODE", mode)
        state.pre = ocr_prefetch.OcrPrefetch(tmp_path / "proj", [src], OcrParams(game="zonewar"))
        return state.pre

    state.make = make
    state.plan = lambda: ocr_prefetch.ScenePlan(src, tmp_path / "cache", 60.0, SCENE_VF, ENC)
    yield state
    if getattr(state, "pre", None) is not None:
        state.pre.close()


def _take(state) -> None:
    src = ocr_prefetch.load_source(state.src, 60.0)
    assert state.pre.take(src, state.pre.params, None) == ([], {"frames": 0})


def test_stale_spools_are_swept_when_offered(pre, tmp_path):
    cdir = tmp_path / "cache"
    (cdir / ffmpeg_utils.SPOOL_DIR).mkdir()
    (cdir / ffmpeg_utils.SPOOL_DIR / "seg-00000.raw").write_bytes(b"x")
    (cdir / ocr_prefetch.PROXY_TMP).write_bytes(b"x")
    pre.make("0").offer(pre.src, 60.0, True)  # in mode 0 too
    assert _left(cdir) == set()


def test_split_is_declined_below_the_disk_gate(pre, monkeypatch, tmp_path):
    asked: list[int] = []
    monkeypatch.setattr(ocr_prefetch, "spool_fits", lambda folder, need: asked.append(need))
    share = pre.make("split").offer(pre.src, 60.0, True, pre.plan())
    _take(pre)
    assert pre.decoded is None and share.state == "declined"
    assert asked == [120 * 60 * 3 * 181]  # W*H*3 bytes, 3 fps over 60 s (+1 frame)
    assert "GiB + 1 GiB free" in share.why


def test_split_claims_the_share_and_the_scene_cuts_get_its_proxy(pre, monkeypatch, tmp_path):
    made: list[dict] = []

    class Fake:
        def __init__(self, path, **k):
            made.append(k)
            self.closed = False
            k["proxy"].write_bytes(b"proxy")
            k["on_exit"](True, 1234)

        def frames(self, exits):
            return iter(())

        def close(self):
            self.closed = True

    monkeypatch.setattr(ocr_prefetch, "spool_fits", lambda folder, need: True)
    monkeypatch.setattr(ocr_prefetch, "SplitDecode", Fake)
    share = pre.make("split").offer(pre.src, 60.0, True, pre.plan())
    _take(pre)
    (kw,) = made
    assert (kw["raw_vf"], kw["scene_vf"], kw["scene_enc"], kw["size"]) == (VF, SCENE_VF, ENC, SIZE)
    assert kw["spool"] == tmp_path / "cache" / ffmpeg_utils.SPOOL_DIR
    assert share.result() == tmp_path / "cache" / "proxy.mp4" and share.spool_peak == 1234


def test_a_read_that_never_decodes_declines_its_share(pre, monkeypatch):
    monkeypatch.setattr(ocr_prefetch, "detect_ocr_events", lambda *a, **k: ([], {"frames": 0}))
    share = pre.make("split").offer(pre.src, 60.0, True, pre.plan())
    _take(pre)  # a cached or resumed read: no shared pass
    assert share.state == "declined" and share.result() is None


@pytest.mark.parametrize("ok", [True, False])
def test_proxy_mode_makes_the_scene_proxy_beside_the_ocr_read(pre, monkeypatch, tmp_path, ok):
    runs: list[list[str]] = []

    def run(args, cancel):
        runs.append(args)
        Path(args[-1]).write_bytes(b"part")
        return ok

    monkeypatch.setattr(ocr_prefetch, "run_cancellable", run)
    share = pre.make("proxy").offer(pre.src, 60.0, True, pre.plan())
    proxy = share.result()
    cdir = tmp_path / "cache"
    (args,) = runs
    assert args[args.index("-i") + 1 :] == [str(pre.src), "-an", "-vf", SCENE_VF, *ENC,
                                            str(cdir / ocr_prefetch.PROXY_TMP)]  # fmt: skip
    assert "-hwaccel" in args and pre.decoded is None  # the OCR decodes on its own
    assert proxy == (cdir / "proxy.mp4" if ok else None)
    assert _left(cdir) == ({"proxy.mp4"} if ok else set())
    _take(pre)


def test_split_that_cannot_start_fails_its_share(pre, monkeypatch, tmp_path):
    def broken(path, **k):
        raise RuntimeError("no thread")

    monkeypatch.setattr(ocr_prefetch, "spool_fits", lambda folder, need: True)
    monkeypatch.setattr(ocr_prefetch, "SplitDecode", broken)
    share = pre.make("split").offer(pre.src, 60.0, True, pre.plan())
    _take(pre)
    assert pre.decoded is None and share.state == "failed" and share.result() is None


def test_proxy_mode_waits_for_enough_vram(pre, monkeypatch):
    monkeypatch.setattr(ocr_prefetch, "vram_ok", lambda: False)
    monkeypatch.setattr(ocr_prefetch, "run_cancellable", lambda *a: pytest.fail("ran"))
    share = pre.make("proxy").offer(pre.src, 60.0, True, pre.plan())
    assert share.state == "declined" and "VRAM" in share.why and share.result() is None


@pytest.mark.parametrize("taken", [True, False])
def test_close_removes_a_proxy_the_scene_cuts_never_took(pre, monkeypatch, tmp_path, taken):
    def run(args, cancel):
        Path(args[-1]).write_bytes(b"proxy")
        return True

    monkeypatch.setattr(ocr_prefetch, "run_cancellable", run)
    share = pre.make("proxy").offer(pre.src, 60.0, True, pre.plan())
    pre.pre._proxies[0].join(5)
    if taken:
        assert share.result() is not None
    pre.pre.close()
    assert share.plan.proxy.exists() is taken  # a taken one is the scene cuts' to delete


def test_mode_0_offers_no_share(pre):
    assert pre.make("0").offer(pre.src, 60.0, True, pre.plan()) is None
    assert pre.pre.mode is None


# ----------------------------------------------------------------- real engine, real ffmpeg
@needs_ffmpeg
@pytest.mark.slow
def test_real_split_read_equals_the_step_1_read(home, tmp_path, monkeypatch) -> None:
    """The real OCR engine reading through the split pass (on the CPU here: both reads take
    the "CUDA" path with the decoder flag dropped) gives the events, stats and raw OCR cache
    of a step 1 read, and its scene proxy the scene cuts of the plain decode."""
    pytest.importorskip("rapidocr_onnxruntime")
    import test_discover_e2e as e2e  # noqa: PLC0415
    from cutpoint import paths  # noqa: PLC0415
    from cutpoint.events import common, killfeed  # noqa: PLC0415
    from hud_fixtures import write_preset  # noqa: PLC0415
    from test_ocr_raw import _files  # noqa: PLC0415

    real_args = ffmpeg_utils.decode_raw_args
    monkeypatch.setattr(
        ffmpeg_utils, "decode_raw_args", lambda *a, hw, **k: real_args(*a, hw=False, **k)
    )
    monkeypatch.setattr(ocr_prefetch, "SplitDecode", functools.partial(SplitDecode, hw=False))
    monkeypatch.setattr(ocr_prefetch, "spool_fits", lambda folder, need: True)
    monkeypatch.setattr(ocr_prefetch, "vram_ok", lambda: True)
    monkeypatch.setenv("CUTPOINT_ANALYZE_SHARED_DECODE", "split")
    monkeypatch.setattr(e2e, "SECONDS", 12)
    folder = tmp_path / "footage"
    folder.mkdir()
    serial_clip, shared_clip = folder / "a.mp4", folder / "b.mp4"
    e2e.make_clip(serial_clip)
    shutil.copy2(serial_clip, shared_clip)
    fx, fy, fw, fh = (round(v * 1.5) for v in e2e.FEED)
    region = {"name": "feed", "x": fx, "y": fy, "w": fw, "h": fh, "kind": "feed", "every": 1}
    feed = {"kill_words": {"downed": "kill"}}  # the clip's invented feed lines
    write_preset(paths.presets_dir(), {"game": e2e.GAME, "regions": [region], "feed": feed})
    project = tmp_path / "proj"
    params = OcrParams(game=e2e.GAME, fps=1.0, project=project)
    engine = killfeed.OcrEngine()
    serial = killfeed.detect_ocr_events(common.load_source(serial_clip, 12.0), params, engine)
    pre = ocr_prefetch.OcrPrefetch(project, [shared_clip], OcrParams(game=e2e.GAME, fps=1.0))
    pre.make_engine = lambda: engine
    src = common.load_source(shared_clip, 12.0)
    with pre:
        share = pre.offer(shared_clip, 12.0, True, analyze._scene_plan(shared_clip, 12.0))
        assert share is not None
        while share.state == "pending":  # the worker claims it (the scene cuts come later)
            time.sleep(0.01)
        events, stats = pre.take(src, params, None)
        cuts = analyze.scene_cuts(shared_clip, src.cdir, 12.0, shared=share)
    assert share.state == "done" and stats["hwaccel"] is True
    assert events == serial[0] and serial[0], "the clip's feed gave no events"
    drop = ("elapsed_s",)
    assert {k: v for k, v in stats.items() if k not in drop} == {
        k: v for k, v in serial[1].items() if k not in drop
    }
    a, b = (_files(common.load_source(c, 12.0).cdir) for c in (serial_clip, shared_clip))
    assert b.pop("scenes.json") == b"[]"  # the serial clip's scene cuts go to `plain`
    assert a == b
    plain = tmp_path / "plain"
    plain.mkdir()
    assert cuts == analyze.scene_cuts(serial_clip, plain, 12.0)
    assert not (src.cdir / ffmpeg_utils.SPOOL_DIR).exists()
