"""VL framing check (detection / QA only): reply parsing and retry, frame sampling and
the per-view reply cache.

The vision model is mocked; no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import requests

from cutpoint import ffmpeg_utils
from cutpoint.pano import vlcheck as V


@pytest.mark.parametrize(
    ("content", "thinking", "want"),
    [
        (
            '{"blocked": true, "where": "Right", "what": "arm", "low_info": false}',
            "",
            {"blocked": True, "where": "right", "what": "arm", "low_info": False},
        ),
        # wrapped as {"frames": [...]}, string booleans, British spelling
        (
            'ok {"frames": [{"blocked": "true", "where": "centre", "what": "mitten"}]}',
            "",
            {"blocked": True, "where": "center", "what": "mitten", "low_info": False},
        ),
        # blocked without a usable where -> center
        ('{"blocked": true, "where": "everywhere"}', "", {"where": "center"}),
        # reply cut off: the last answer drafted in the reasoning
        (
            "",
            'maybe {"blocked": true, "where": "left"} no: {"blocked": false, "low_info": true}',
            {"blocked": False, "low_info": True},
        ),
        ('{"score": 3}', "", None),
    ],
)
def test_parse_reply(content: str, thinking: str, want: dict[str, Any] | None) -> None:
    got = V.parse_reply(content, thinking)
    if want is None:
        assert got is None
    else:
        assert got is not None and want.items() <= got.items()


def test_unparseable_reply_is_retried_once_with_the_short_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chk = V.VLChecker(url="http://127.0.0.1:1", model="m")
    prompts: list[str] = []
    replies = iter([{"content": ""}, {"content": '{"blocked": true, "where": "left"}'}])
    monkeypatch.setattr(
        chk, "_post", lambda prompt, image, n: prompts.append(prompt) or next(replies)
    )
    assert chk.ask(b"jpeg")["where"] == "left"
    assert prompts == [V.PROMPT, V.RETRY_PROMPT]
    monkeypatch.setattr(chk, "_post", lambda prompt, image, n: {"content": "?"})
    assert chk.ask(b"jpeg") is None


def test_a_server_that_stops_answering_is_not_asked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    chk = V.VLChecker(url="http://127.0.0.1:1", model="m")
    posts: list[str] = []

    def down(url: str, **kw: Any) -> None:
        posts.append(url)
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", down)
    assert chk.ask(b"jpeg") is None and chk.dead
    assert chk.ask(b"jpeg") is None and len(posts) == 1


def test_unload_frees_the_model_only_after_a_check(monkeypatch: pytest.MonkeyPatch) -> None:
    chk = V.VLChecker(url="http://127.0.0.1:1", model="m")
    posts: list[tuple[str, dict]] = []

    class Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"message": {"content": '{"blocked": false}'}}

    monkeypatch.setattr(
        requests, "post", lambda url, json, **kw: posts.append((url, json)) or Resp()
    )
    chk.unload()
    assert posts == []  # nothing asked yet: nothing to unload
    chk.ask(b"jpeg")
    chk.unload()
    assert posts[-1] == ("http://127.0.0.1:1/api/generate", {"model": "m", "keep_alive": 0})


def path_of(suspect: list[float]) -> dict[str, Any]:
    keys = [{"t": t / 2, "yaw": 30.0, "pitch": -8.0, "fov": 86.0} for t in range(25)]
    return {
        "clip": "c1",
        "segment": {"start": 1.0, "end": 10.0},
        "keys": keys,
        "vl_suspect": suspect,
    }


def test_sample_times_sparse_baseline_plus_thinned_suspects() -> None:
    # 9 s: baseline 1.5 / 4.5 / 7.5 s; suspects 3.0 then 8.5 (2.0 too close, 3.4 thinned);
    # no gap of 4.5 s is left for a midpoint
    got = V.sample_times(path_of([3.0, 4.0, 4.4, 9.5]), every=3.0, max_frames=8)
    assert got.tolist() == pytest.approx([1.5, 3.0, 4.5, 7.5, 8.5])
    assert V.sample_times(path_of([])).tolist() == pytest.approx([1.5, 4.5, 7.5])
    # 20 s, 4 frames, 2 suspects: baseline 5 / 15 s, then both suspects; no hole of dur/2
    long = path_of([9.0, 19.0])
    long["segment"]["end"] = 21.0
    got = V.sample_times(long, every=3.0, max_frames=4)
    assert got.tolist() == pytest.approx([5.0, 8.0, 15.0, 18.0])
    assert np.diff([0.0, *got, 20.0]).max() < 20.0 / 2


def test_check_preview_reports_bad_frames_and_reuses_cached_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ffmpeg_utils, "grab_images", lambda video, t, **kw: [f"{t:.1f}".encode()])

    class Fake:
        model = "fake-vl"
        calls = 0

        def ask(self, jpeg: bytes) -> dict[str, Any] | None:
            self.calls += 1
            if jpeg == b"4.5":
                return None  # unanswered: skipped, not cached
            bad = jpeg == b"1.5"
            return {
                "blocked": bad,
                "where": "left" if bad else "none",
                "what": "arm",
                "low_info": False,
            }

    fake = Fake()
    preview = tmp_path / "p.mp4"
    flags, rec = V.check_preview(fake, preview, path_of([]))  # type: ignore[arg-type]
    assert [f.t for f in flags] == [2.5, 8.5] and flags[0].blocked
    assert (rec["frames"], rec["bad"], rec["cached"]) == (2, 1, 0)
    assert rec["flags"][0]["where"] == "left"
    V._MEMO.clear()  # from the disk cache now; only the unanswered frame is asked again
    _, rec = V.check_preview(fake, preview, path_of([]))  # type: ignore[arg-type]
    assert (fake.calls, rec["cached"]) == (4, 2)
