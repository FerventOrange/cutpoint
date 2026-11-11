"""Isolate every test from the real user data (CUTPOINT_HOME, env, LOCALAPPDATA)."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutpoint import analyze, clips, config, paths, reset_state
from cutpoint.ffmpeg_utils import MediaInfo


def pytest_configure(config: pytest.Config) -> None:
    # tests/scripts: the development scripts in scripts/; `-m "not script"` skips them
    config.addinivalue_line(
        "markers", "script: tests a development script in scripts/, not the product"
    )

    # Windows DLL load order (#215, see transcribe._load_torch_first): once a test has loaded the
    # transcription stack's CUDA/cuDNN DLLs (ctranslate2), the first `import torch` in that
    # process fails. A serial run happens to import torch early (test_assistant_lookfor); under
    # pytest-xdist each worker sees the files in another order, so import it first, in every
    # process that runs tests (not in the xdist controller). A lite install has no torch.
    serial_or_worker = hasattr(config, "workerinput") or not config.getoption("numprocesses", None)
    if sys.platform == "win32" and serial_or_worker and importlib.util.find_spec("torch"):
        try:
            import torch  # noqa: F401, PLC0415
        except (ImportError, OSError, RuntimeError) as err:  # as in transcribe._load_torch_first
            warnings.warn(
                f"importing torch up front failed: {err}", pytest.PytestWarning, stacklevel=1
            )


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point CUTPOINT_HOME and every platform default at ``tmp_path``; yields the home."""
    for name in list(os.environ):
        if name.startswith("CUTPOINT_"):
            monkeypatch.delenv(name)
    # A user's TORCH_HOME would point paths.detector_weights() at their real checkpoint. The
    # setenv records its value (or its absence), so a TORCH_HOME apply_torch_home sets during
    # the test is undone too.
    monkeypatch.setenv("TORCH_HOME", "")
    monkeypatch.delenv("TORCH_HOME")
    # A user's OLLAMA_HOST would move backends.ollama_url() (and the VLM cache key) off the
    # default; setenv first, as for TORCH_HOME, so one a test sets is undone too.
    monkeypatch.setenv("OLLAMA_HOST", "")
    monkeypatch.delenv("OLLAMA_HOST")
    home = tmp_path / "home"
    monkeypatch.setenv("CUTPOINT_HOME", str(home))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile"))
    monkeypatch.setenv("HOME", str(tmp_path / "profile"))
    monkeypatch.setattr(config, "_drive_remote", lambda d: None)  # no real mapped drives
    reset_state()
    assert paths.home() == home, "isolation failed: CUTPOINT_HOME not honoured"
    yield home
    reset_state()


class _NoNvidiaSmi:
    """Stands in for ``subprocess`` inside the modules that probe the GPU: an ``nvidia-smi``
    command fails as if the tool were missing; every other command goes to the real
    ``subprocess.run``, looked up at call time so tests that patch it still work."""

    def __getattr__(self, name: str):
        return getattr(subprocess, name)

    @staticmethod
    def run(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)) and cmd and Path(str(cmd[0])).stem == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        return subprocess.run(cmd, *args, **kwargs)


@pytest.fixture(autouse=True)
def no_host_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the host out of the tests: no ``nvidia-smi`` spawn (the GPU probes see no GPU) and no
    gcc/ld/ldconfig run by ``ctypes.util.find_library`` (keyring's macOS backend calls it on
    import), and no read of the host's free RAM (the memory guard is off, as on an unknown
    platform), and no ``/api/ps`` probe of the context Ollama has the vision model loaded with
    (it asks for the context it needs, #732). A test of a probe patches ``subprocess.run`` or the
    probe itself, as before."""
    import ctypes.util  # noqa: PLC0415

    from cutpoint import memguard  # noqa: PLC0415

    monkeypatch.setattr(memguard, "ram_free", lambda: None)
    monkeypatch.setattr(memguard, "ollama_context", lambda url, model: None)

    monkeypatch.setattr(memguard, "subprocess", _NoNvidiaSmi())  # every nvidia-smi query
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    # no footage question to a vision model the host runs (#898): the detection of unsure
    # footage stays as the frames decide it; tests of the question put the client back
    from cutpoint import category  # noqa: PLC0415

    monkeypatch.setattr(category.FootageVision, "client", lambda self: None)


@pytest.fixture
def mem(monkeypatch):
    """The memory guard's view (#109): free RAM and VRAM in GB (``vram`` None: no GPU known),
    and the calls of free_models / unload_ollama; ``after`` is the RAM free once they ran."""
    from cutpoint import analyze, memguard  # noqa: PLC0415
    from cutpoint.events import common  # noqa: PLC0415

    gb = memguard.GIB
    state = SimpleNamespace(ram=64.0, vram=None, after=None, freed=0, unloaded=0)

    def free_models() -> None:
        state.freed += 1

    def unload_ollama(url: str | None = None) -> list[str]:
        state.unloaded += 1
        if state.after is not None:
            state.ram = state.after
        return []

    monkeypatch.setattr(memguard, "ram_free", lambda: None if state.ram is None else state.ram * gb)
    monkeypatch.setattr(
        memguard, "vram_free", lambda: None if state.vram is None else state.vram * gb
    )
    monkeypatch.setattr(analyze, "free_models", free_models)
    monkeypatch.setattr(common, "unload_ollama", unload_ollama)
    return state


@pytest.fixture
def home(isolated_env: Path) -> Path:
    return isolated_env


def write(path: Path, text: str | bytes = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


# Invented activity profiles: Cutpoint ships only ``base`` and ``generic``, so the tests put
# these into the isolated user library (add_profile) or use them directly.
SLED = {  # every feature: lift captions, a replaced floor entry, its own wording
    "name": "glacier-sled",
    "description": "An invented test activity: sledding on a glacier.",
    "detect": ["a first-person view of a sled on a glacier"],
    "incident": {
        "pos": ["a sled tipped over on the ice", "a blurry spinning view of ice"],
        "neg": ["a sled gliding on smooth ice"],
        "words": {"tipped the sled": 1.0, "ice in my face": 0.8},
    },
    "highlight": {"pos": ["a wide glacier under a blue sky"], "neg": ["sled runners close up"]},
    "lift": {"pos": ["riding a cable car above a glacier"], "neg": ["sledding down the ice"]},
    "low_info": ["a flat white image"],
    "exclude_base": ["a dark blurry image"],
    "vl": {"video": "sledding", "gear": "sled rope"},
    "vlm": {
        "video": "first-person glacier sledding video",
        "edit": "sledding highlight edit",
        "rate_edit": "sledding highlight edit",
        "tags": {"transit": "on a cable car"},
    },
}
HOP = {  # captions only: the floor's wording, no lifts
    "name": "puddle-hop",
    "description": "An invented test activity: hopping over puddles.",
    "detect": ["a first-person view of jumping over a puddle"],
    "incident": {
        "pos": ["a boot splashing into a puddle"],
        "neg": ["walking on a dry path"],
        "words": {"soaked my boot": 0.9},
    },
    "highlight": {"pos": ["a person in mid-air over a puddle"], "neg": []},
    "low_info": ["a close-up of wet pavement"],
}


def add_profile(raw: dict, name: str | None = None) -> dict:
    """Write ``raw`` into the user's profile library (as ``name``, default its own); returns
    the library profile."""
    from cutpoint import activity  # noqa: PLC0415

    name = name or raw["name"]
    path = write(paths.profiles_dir() / f"{name}.json", json.dumps({**raw, "name": name}))
    activity._FILES.pop(str(path), None)
    return activity.library()[name]


@pytest.fixture
def pinned_prior(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], float]:
    """The vision rating prior (``vision.RATING_PRIOR``, #909, #948) fixed at 3.2 (frames) and
    5.1 (speech) in every style group, so a test's hand-computed centres hold when the shipped
    prior is re-fitted."""
    from cutpoint import vision  # noqa: PLC0415

    for group in vision.PRIOR_GROUPS:
        monkeypatch.setitem(vision.RATING_PRIOR, (group, "frames"), 3.2)
        monkeypatch.setitem(vision.RATING_PRIOR, (group, "speech"), 5.1)
    return vision.RATING_PRIOR


def set_config(home: Path, **data) -> None:
    """Write ``config.json`` under ``home`` and drop the cached copy."""
    write(home / "config.json", json.dumps(data))
    config.get_config.cache_clear()


def remap(home: Path, monkeypatch: pytest.MonkeyPatch, old, new, exclude=()) -> None:
    """``source_remap`` from ``old`` to ``new``; paths compare as plain spellings, on any OS."""
    data = {"source_remap": {str(old): str(new)}}
    if exclude:
        data["source_remap_exclude"] = list(exclude)
    set_config(home, **data)
    monkeypatch.setattr(config, "_prefix_forms", lambda p: (config.norm_path(p),))


def load_script(name: str):
    """The development script ``scripts/<name>.py`` as a module (loaded once; the scripts are
    not part of the package). Its tests are marked ``script``."""
    import importlib.util  # noqa: PLC0415

    key = f"cutpoint_script_{name}"
    if key not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(key, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module  # dataclasses look their module up by name
        spec.loader.exec_module(module)
    return sys.modules[key]


def media_info(path: Path = Path("v.mp4"), duration: float = 100.0, **over) -> MediaInfo:
    """A 1920x1080, 30 fps, H.264 + AAC 48 kHz stereo probe result; ``over`` replaces fields."""
    fields = {
        "path": path, "duration": duration, "vduration": duration, "width": 1920, "height": 1080,
        "fps": Fraction(30), "pix_fmt": "yuv420p", "vcodec": "h264", "audio_streams": 1,
        "acodec": "aac", "sample_rate": 48000, "channels": 2,
    }  # fmt: skip
    return MediaInfo(**(fields | over))


_skip_without_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not on PATH"
)


def needs_ffmpeg(test):
    """Skip without ffmpeg/ffprobe, and mark the test ``ffmpeg`` (it runs the real tools)."""
    return pytest.mark.ffmpeg(_skip_without_ffmpeg(test))


_UTIL_JS_LOOP = """
import * as U from %s;
import { createInterface } from "node:readline";
for await (const line of createInterface({ input: process.stdin })) {
  let out;
  try {
    const { code, args } = JSON.parse(line);
    out = { value: new Function("U", "args", code)(U, args) ?? null, error: null };
  } catch (err) {
    out = { value: null, error: String((err && err.stack) || err) };
  }
  process.stdout.write(JSON.stringify(out) + "\\n");
}
"""


class UtilJs:
    """One node process that runs ``web/util.js`` calls sent over stdin, one JSON line each."""

    def __init__(self, node: str, module: Path) -> None:
        self.proc = subprocess.Popen(
            [node, "--input-type=module", "-e", _UTIL_JS_LOOP % json.dumps(module.as_uri())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
        )  # fmt: skip

    def __call__(self, code: str, *args):
        """Run ``code``, a JS function body with ``U`` (util.js's exports) and ``args`` in scope;
        returns its JSON-decoded ``return`` value."""
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps({"code": code, "args": list(args)}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, f"node exited ({self.proc.poll()})"
        out = json.loads(line)
        assert out["error"] is None, out["error"]
        return out["value"]

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


@pytest.fixture(scope="session")
def util_js(tmp_path_factory: pytest.TempPathFactory):
    """``web/util.js`` under node (skip without node), started once per session; tests using it
    are ``slow``. util.js is copied to ``util.mjs`` so every node >= 18 loads it as an ES module:
    there is no package.json, and node before 22.7 would read a ``.js`` file as CommonJS. util.js
    has no imports of its own, so the copy runs on its own."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    module = tmp_path_factory.mktemp("web") / "util.mjs"
    shutil.copyfile(Path(analyze.__file__).parent / "web" / "util.js", module)  # ui.WEB_DIR
    runner = UtilJs(node, module.resolve())
    yield runner
    runner.close()


KEYS = {"ok", "command", "schema", "project", "outputs", "warnings", "error"}


def envelope(out: str) -> dict:
    """``out`` must be exactly one JSON object (and a newline)."""
    doc, end = json.JSONDecoder().raw_decode(out)
    assert out[end:] == "\n"
    assert isinstance(doc, dict) and set(doc) == KEYS
    assert doc["schema"] == 1 and isinstance(doc["ok"], bool)
    assert isinstance(doc["command"], str) and isinstance(doc["outputs"], dict)
    assert isinstance(doc["warnings"], list)
    assert doc["project"] is None or isinstance(doc["project"], str)
    if doc["ok"]:
        assert doc["error"] is None
    else:
        assert set(doc["error"]) == {"code", "message"}
    return doc


def run_json(capsys, argv: list[str]) -> tuple[int, dict]:
    """Run ``cutpoint <argv> --json``: exit code and the one envelope on stdout."""
    from cutpoint import cli  # noqa: PLC0415

    code = cli.main([*argv, "--json"])
    return code, envelope(capsys.readouterr().out)


def warning_codes(doc_or_warnings: dict | list[dict]) -> list[str]:
    """The ``code`` of each warning in an envelope (or in its ``warnings`` list)."""
    warnings = doc_or_warnings["warnings"] if isinstance(doc_or_warnings, dict) else doc_or_warnings
    return [w["code"] for w in warnings]


def link_file(link: Path, target: Path) -> None:
    """Make ``link`` a file symlink to ``target`` (skip where files cannot be linked)."""
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a file symlink here")


def link_dir(link: Path, target: Path) -> None:
    """Make ``link`` a directory link to ``target``: a symlink, or a junction on Windows."""
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            pytest.skip("cannot create a directory symlink here")
    done = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False
    )
    if done.returncode != 0:
        pytest.skip("cannot create a symlink or junction here")


def hard_link(link: Path, target: Path) -> None:
    try:
        os.link(target, link)
    except (OSError, NotImplementedError, AttributeError) as err:
        pytest.skip(f"hard links are not supported here: {err}")


def messages(caplog: pytest.LogCaptureFixture, level: str = "WARNING", logger=None) -> list[str]:
    """Messages logged at exactly ``level`` (optionally only by logger name ``logger``)."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelname == level and (logger is None or r.name == logger)
    ]


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Opt-in: ``time.sleep`` records its argument instead of waiting; returns the list."""
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    return slept


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch):
    """Returns ``set(flag)``, which makes ``clips._WINDOWS`` report Windows (or not)."""
    return lambda flag=True: monkeypatch.setattr(clips, "_WINDOWS", flag)


@pytest.fixture
def moved(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A 360 clip recorded under ``old/`` whose footage now lives in ``new/`` (``source_remap``)."""
    from pano_helpers import INSV, LRV, x3clip  # noqa: PLC0415

    old, new = tmp_path / "old", tmp_path / "new"
    for name in (*INSV, LRV):
        write(new / name)  # the old folder is gone; no stitched file anywhere
    remap(home, monkeypatch, old, new)
    return x3clip(lrv=str(old / LRV), insv=[str(old / n) for n in INSV]), old, new


VIDEO_SEGMENTS = [
    {
        "start": 3.0,
        "end": 4.5,
        "text": " ha ha that was great",
        "words": [
            {"start": 3.0, "end": 3.3, "word": " ha"},
            {"start": 3.3, "end": 3.6, "word": " ha"},
            {"start": 3.6, "end": 3.9, "word": " that"},
            {"start": 3.9, "end": 4.1, "word": " was"},
            {"start": 4.1, "end": 4.5, "word": " great"},
        ],
    }
]


class StubTranscriber:
    """No whisper: every source "says" VIDEO_SEGMENTS."""

    model_name, device = "stub", "cpu"

    def __init__(self, language: str | None = None, **_kw) -> None:
        self.language = language

    def transcribe(self, _wav: Path) -> list[dict]:
        return VIDEO_SEGMENTS

    def close(self) -> None:
        pass


@pytest.fixture
def video(home, tmp_path, monkeypatch) -> Path:
    """A 10 s 160x120 video, quiet with two loud bursts; the transcriber stubbed.

    Does not skip by itself: tests using it need ``@needs_ffmpeg``.
    """
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(analyze, "Transcriber", StubTranscriber)
    out = tmp_path / "videos" / "clip.mp4"
    out.parent.mkdir()
    loud = "volume=enable='between(t,2,4.5)+between(t,7,8.5)':volume=30"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
         "testsrc=size=160x120:rate=10:duration=10", "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=48000:duration=10", "-af", f"volume=0.02,{loud}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
        check=True,
    )  # fmt: skip
    return out


@pytest.fixture
def case_insensitive(monkeypatch):
    """Call with a folder to make it case-insensitive: ``config.folds_case`` is True for it,
    and ``Path.exists`` / ``is_file`` / ``is_symlink`` / ``replace`` find a name there in
    another case (``replace`` onto a name in another case replaces that file)."""
    folders: set[Path] = set()
    real_folds = config.folds_case
    real = {n: getattr(Path, n) for n in ("exists", "is_file", "is_symlink", "replace")}

    def actual(p: Path) -> Path:
        p = Path(p)
        if p.parent.absolute() in folders and not os.path.lexists(p) and p.parent.is_dir():
            for entry in p.parent.iterdir():
                if entry.name.lower() == p.name.lower():
                    return entry
        return p

    def replace(self: Path, target: str | os.PathLike[str]) -> Path:
        target = Path(target)
        if (variant := actual(target)) != target:
            variant.unlink()
        return real["replace"](actual(self), target)

    for name in ("exists", "is_file", "is_symlink"):
        monkeypatch.setattr(
            Path, name, lambda self, *a, _f=real[name], **k: _f(actual(self), *a, **k)
        )
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(
        config,
        "folds_case",
        lambda f: Path(f).absolute() in folders or real_folds(f),
    )
    return lambda folder: folders.add(Path(folder).absolute())
