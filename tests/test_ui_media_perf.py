"""Server-side costs of ``cutpoint ui`` (#1085): the dialogue events indexed once per moments
list, one source stat per cached view, one ffprobe per source key however many requests ask at
once, and stills evicted without stat-ing the whole frames folder on each new one."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from conftest import media_info, patch_ui
from cutpoint import ui
from cutpoint.ui import media as media_module
from cutpoint.ui.projects import bit_description, dialogue_events


# --------------------------------------------------------------------------- dialogue events
class Counted(list):
    """A list that counts how often it is iterated."""

    walks = 0

    def __iter__(self):
        Counted.walks += 1
        return super().__iter__()


def talk(source: str, start: float, end: float, text: str, conf: float = 0.5) -> dict:
    return {
        "kind": "text", "type": "dialogue", "source": source, "start": start, "end": end,
        "description": text, "confidence": conf,
    }  # fmt: skip


def bit(source: str, start: float, end: float) -> dict:
    return {"source": source, "start": start, "end": end, "dialogue_title": "A bit"}


def test_the_moments_view_walks_the_events_once_whatever_the_dialogue_moments() -> None:
    events = Counted([talk("/v/a.mp4", 10.0 * i, 10.0 * i + 5, f"a{i}") for i in range(50)])
    hl = {"candidates": [bit("/v/a.mp4", 10.0 * i, 10.0 * i + 8) for i in range(40)]}
    Counted.walks = 0
    view = ui.moments_view(hl, {"events": events})["candidates"]
    assert Counted.walks == 1
    assert [c["bit_description"] for c in view] == [f"a{i}" for i in range(40)]


def test_the_index_keeps_the_most_confident_dialogue_event_of_the_same_source() -> None:
    a, b = "/v/a.mp4", "/v/b.mp4"
    events = {
        "events": [
            talk(a, 12.0, 14.0, "first", 0.5),
            talk(a, 13.0, 15.0, "tie: the first stays", 0.5),
            talk(b, 12.0, 14.0, "other source", 0.9),
            talk(a, 5.0, 14.0, "starts before", 0.9),
            talk(a, 12.0, 21.0, "ends after", 0.9),
            {**talk(a, 12.0, 14.0, "not dialogue", 0.9), "type": "banter"},
            {**talk(a, 12.0, 14.0, "", 0.9)},  # no description
            {**talk(a, 12.0, 14.0, "no start", 0.9), "start": None},
            {**talk(a, 12.0, 14.0, "list source", 0.9), "source": [a]},
            "not an event",
        ]
    }
    dialogue = dialogue_events(events)
    assert bit_description(bit(a, 10.0, 20.0), dialogue) == "first"
    assert bit_description(bit(b, 10.0, 20.0), dialogue) == "other source"
    keeps = {**bit("/v/c.mp4", 10.0, 20.0), "dialogue_description": "kept"}
    assert bit_description(keeps, dialogue) == "kept"  # no event: highlights.json's
    assert bit_description({"source": a, "start": 10.0, "end": 20.0}, dialogue) == ""
    assert dialogue_events(None) == dialogue_events({"events": None}) == {}
    assert dialogue_events({"events": 5}) == {}  # a broken events.json: no 500


# --------------------------------------------------------------------------- cached view
@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The paths :meth:`Media.key` stats, in order."""
    asked: list[Path] = []
    real = ui.Media.key

    def key(path: Path) -> str | None:
        asked.append(Path(path))
        return real(path)

    monkeypatch.setattr(ui.Media, "key", staticmethod(key))
    return asked


def test_a_cached_view_stats_the_source_once(tmp_path: Path, keys: list[Path]) -> None:
    source = tmp_path / "a.mkv"
    source.write_bytes(b"x")
    media = ui.Media()
    media.probed[ui.Media.key(source)] = (False, 10.0, 1, "")  # needs a proxy
    keys.clear()
    assert media.cached_view(tmp_path, str(source)) == ("needs-proxy", True, None)
    assert keys == [source]
    keys.clear()
    assert media.cached_view(tmp_path, str(tmp_path / "gone.mkv")) == ("failed", False, None)
    assert keys == [tmp_path / "gone.mkv"]


# --------------------------------------------------------------------------- single-flight probe
@pytest.fixture
def slow_probe(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Path], threading.Event]:
    """``(paths probed, release)``: each probe waits for ``release``."""
    probed: list[Path] = []
    release = threading.Event()

    def probe(path: Path):
        probed.append(Path(path))
        assert release.wait(10)
        if len(probed) == 1 and Path(path).name == "raises.mkv":
            raise RuntimeError("the first probe broke")
        return media_info(Path(path), duration=10.0)

    patch_ui(monkeypatch, "probe", probe)
    return probed, release


def started(first: threading.Thread, probed: list[Path]) -> None:
    """Wait (up to 10 s) until ``first`` is inside its probe."""
    deadline = time.monotonic() + 10
    while not probed and first.is_alive() and time.monotonic() < deadline:
        first.join(0.01)
    assert probed


def test_concurrent_requests_for_one_key_run_one_probe(
    tmp_path: Path, slow_probe: tuple[list[Path], threading.Event]
) -> None:
    probed, release = slow_probe
    source = tmp_path / "a.mkv"
    source.write_bytes(b"x")
    media, key = ui.Media(), ui.Media.key(source)
    got: list[object] = []
    first = threading.Thread(target=lambda: got.append(media.info(source, key)))
    second = threading.Thread(target=lambda: got.append(media.info(source, key)))
    first.start()
    started(first, probed)
    second.start()
    second.join(0.2)
    assert second.is_alive() and probed == [source]  # it waits for the first one's result
    assert not media.lock.locked()  # no global lock held during the probe
    release.set()
    first.join(10)
    second.join(10)
    assert probed == [source]
    assert got == [(False, 10.0, 1, ""), (False, 10.0, 1, "")]  # .mkv: needs a proxy
    assert media.probing == {}


def test_a_waiter_probes_itself_when_the_first_probe_raises(
    tmp_path: Path, slow_probe: tuple[list[Path], threading.Event]
) -> None:
    probed, release = slow_probe
    source = tmp_path / "raises.mkv"
    source.write_bytes(b"x")
    media, key = ui.Media(), ui.Media.key(source)
    errors: list[BaseException] = []
    got: list[object] = []

    def first_call() -> None:
        try:
            media.info(source, key)
        except RuntimeError as err:
            errors.append(err)

    first = threading.Thread(target=first_call)
    second = threading.Thread(target=lambda: got.append(media.info(source, key)))
    first.start()
    started(first, probed)
    second.start()
    second.join(0.2)
    release.set()
    first.join(10)
    second.join(10)
    assert [str(e) for e in errors] == ["the first probe broke"]  # raised as before
    assert probed == [source, source] and got == [(False, 10.0, 1, "")]
    assert media.probing == {}


# --------------------------------------------------------------------------- frame eviction
@pytest.fixture
def stats(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Names of the stills stat-ed (through the folder listing or a path)."""
    seen: list[str] = []
    real = os.scandir

    class Entry:
        def __init__(self, entry: os.DirEntry) -> None:
            self.entry, self.name, self.path = entry, entry.name, entry.path

        def stat(self):
            seen.append(self.name)
            return self.entry.stat()

    class Listing:
        def __init__(self, path) -> None:
            self.it = real(path)

        def __enter__(self):
            return (Entry(e) for e in self.it)

        def __exit__(self, *exc) -> None:
            self.it.close()

    def path_stat(self: Path, *a, **k):
        if self.suffix == ".jpg":
            seen.append(self.name)
        return real_path_stat(self, *a, **k)

    real_path_stat = Path.stat
    monkeypatch.setattr(media_module.os, "scandir", Listing)
    monkeypatch.setattr(Path, "stat", path_stat)
    return seen


def stills(folder: Path, first: int, n: int, at: int) -> list[Path]:
    """``n`` stills named from ``first`` on, the k-th last used at ``at + k`` s."""
    made = []
    for i in range(first, first + n):
        f = folder / f"{'a' * 16}-{i}-160.jpg"
        f.write_bytes(b"\xff\xd8\xff")
        os.utime(f, (at + i, at + i))
        made.append(f)
    return made


def jpgs(folder: Path) -> set[Path]:
    return set(folder.glob("*.jpg"))


def test_stills_under_the_cap_are_counted_not_stat_ed(
    tmp_path: Path, stats: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_ui(monkeypatch, "FRAME_CACHE_MAX", 100)
    stills(tmp_path, 0, 100, 1_000_000)
    ui.Media()._evict_frames(tmp_path)
    assert stats == [] and len(jpgs(tmp_path)) == 100


def test_a_full_folder_is_evicted_below_the_cap_so_the_next_stills_stat_nothing(
    tmp_path: Path, stats: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_ui(monkeypatch, "FRAME_CACHE_MAX", 100)
    media = ui.Media()
    made = stills(tmp_path, 0, 101, 1_000_000)
    media._evict_frames(tmp_path)
    assert len(stats) == 101  # one stat pass over the cap
    assert jpgs(tmp_path) == set(made[6:])  # down to 95: the least recently used went
    stats.clear()
    for i in range(5):  # room for 5 more without a stat
        stills(tmp_path, 200 + i, 1, 2_000_000)
        media._evict_frames(tmp_path)
    assert stats == [] and len(jpgs(tmp_path)) == 100
    stills(tmp_path, 300, 1, 3_000_000)
    media._evict_frames(tmp_path)
    assert len(stats) == 101 and len(jpgs(tmp_path)) == 95


def test_the_cap_holds_for_stills_of_another_process_and_after_a_restart(
    tmp_path: Path, stats: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_ui(monkeypatch, "FRAME_CACHE_MAX", 100)
    media = ui.Media()
    stills(tmp_path, 0, 90, 1_000_000)
    media._evict_frames(tmp_path)
    newest = stills(tmp_path, 100, 40, 2_000_000)  # written by another process meanwhile
    stills(tmp_path, 200, 1, 3_000_000)
    media._evict_frames(tmp_path)  # one still of its own: the others count too
    assert len(jpgs(tmp_path)) == 95 and set(newest) <= jpgs(tmp_path)
    stills(tmp_path, 300, 30, 4_000_000)  # a full folder found after a restart
    ui.Media()._evict_frames(tmp_path)
    assert len(jpgs(tmp_path)) == 95


def test_a_new_still_keeps_the_frames_folder_within_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_ui(monkeypatch, "FRAME_CACHE_MAX", 20)
    patch_ui(monkeypatch, "probe", lambda path: media_info(Path(path), duration=1000.0))
    patch_ui(monkeypatch, "grab_images", lambda *_a, **_k: [b"\xff\xd8\xffjpeg"])
    project, source = tmp_path / "p", tmp_path / "a.mp4"
    project.mkdir()
    source.write_bytes(b"x")
    media = ui.Media()
    for i in range(50):
        assert media.frame(project, str(source), float(i), 160) is not None
        assert len(jpgs(project / "ui" / "frames")) <= 20
    left = sorted(int(f.name.split("-")[1]) for f in jpgs(project / "ui" / "frames"))
    assert left[-1] == 4900 and len(left) == 20  # the newest stays


# --------------------------------------------------------------------------- sources poll
def test_the_sources_route_lists_the_project_views_sources_alone(ui_client, tmp_path: Path) -> None:
    """A754: the page's proxy poll reads the project's ``sources`` without its moments; the
    same views as the whole project reply, from the same code."""
    folder = tmp_path / "footage"
    folder.mkdir()
    for name in ("a.mp4", "b.mkv"):
        (folder / name).write_bytes(b"x")
    res = ui_client.post("/api/projects", json={"name": "Poll", "source": str(folder)})
    assert res.status_code == 200, res.text
    pid = res.json()["id"]
    res = ui_client.get(f"/api/projects/{pid}/sources")
    assert res.status_code == 200, res.text
    light = res.json()
    assert list(light) == ["sources"]
    assert [s["name"] for s in light["sources"]] == ["a.mp4", "b.mkv"]
    assert set(light["sources"][0]) == {
        "id", "name", "path", "url", "duration", "status", "proxy", "fragmented", "progress",
        "windows", "pending", "failed",
    }  # fmt: skip
    assert light["sources"] == ui_client.get(f"/api/projects/{pid}").json()["sources"]


@pytest.mark.parametrize("pid", ["nope", ".hidden", " trip", "a\\b"])
def test_the_sources_route_refuses_what_the_project_route_refuses(ui_client, pid: str) -> None:
    assert ui_client.get(f"/api/projects/{pid}/sources").status_code == 404
    assert ui_client.get(f"/api/projects/{pid}").status_code == 404


def test_the_sources_route_needs_the_session_token(ui_client, tmp_path: Path) -> None:
    folder = tmp_path / "footage"
    folder.mkdir()
    (folder / "a.mp4").write_bytes(b"x")
    pid = ui_client.post("/api/projects", json={"name": "Tok", "source": str(folder)}).json()["id"]
    res = ui_client.get(f"/api/projects/{pid}/sources", headers={"X-Cutpoint-Token": "wrong"})
    assert res.status_code == 403
