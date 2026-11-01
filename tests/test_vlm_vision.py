"""The events VLM layer describes windows through the shared vision client (cutpoint.vision)."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import pytest
import requests

from conftest import set_config
from cutpoint import cache, jsonout, vision
from cutpoint.assistant import providers
from cutpoint.assistant.providers import Reply
from cutpoint.events import common, pipeline
from cutpoint.events import vlm_windows as vw
from cutpoint.events.common import Source, layer_cache_path

JPEGS = [b"\xff\xd8one", b"\xff\xd8two", b"\xff\xd8three"]
REPLY: dict[str, Any] = {
    "events": [{"t": 4.5, "what": "player gets a kill", "funny": 1, "hype": 6}],
    "summary": "A kill after some banter.",
    "funny_score": 3,
    "highlight_score": 6,
    "label": "Kill After Banter",
}


class Resp:
    def __init__(self, body: Any, status: int = 200) -> None:
        self.body, self.status_code = body, status
        self.is_redirect = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self.body


class Ollama:
    """Fake Ollama: ``/api/tags`` lists qwen3.5:9b; ``/api/chat`` answers with ``replies`` in
    turn (a dict is the reply's JSON content, an exception is raised, a Resp is returned)."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.chats: list[dict[str, Any]] = []
        self.unloads: list[dict[str, Any]] = []

    def get(self, url: str, **_kw: Any) -> Resp:
        assert url.endswith("/api/tags")
        return Resp({"models": [{"name": "qwen3.5:9b", "digest": "d1"}]})

    def post(self, url: str, json: dict[str, Any], **_kw: Any) -> Resp:
        if url.endswith("/api/generate"):  # unload
            self.unloads.append(json)
            return Resp({})
        assert url.endswith("/api/chat")
        self.chats.append(json)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, Resp):
            return reply
        content = reply if isinstance(reply, str) else _dumps(reply)
        return Resp({"message": {"content": content}, "prompt_eval_count": 1234})


def _dumps(data: Any) -> str:
    return json.dumps(data)


@pytest.fixture
def ollama(home, monkeypatch) -> Ollama:
    fake = Ollama()
    monkeypatch.setattr(vision.requests, "get", fake.get)
    monkeypatch.setattr(vision.requests, "post", fake.post)
    return fake


# --------------------------------------------------------------------------- VisionRater.describe


def test_describe_sends_frames_prompt_and_schema_to_ollama(ollama) -> None:
    ollama.replies = [REPLY]
    rater = vision.VisionRater()
    data = rater.describe(
        "describe", JPEGS, vw.RESPONSE_SCHEMA, num_ctx=9000, num_predict=500, temperature=0.0
    )
    assert data == REPLY and rater.last_prompt_tokens == 1234 and rater.used
    (body,) = ollama.chats
    assert body["model"] == "qwen3.5:9b"  # the analyze --vision default
    assert body["format"] == vw.RESPONSE_SCHEMA
    assert body["think"] is False and body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "num_ctx": 9000, "num_predict": 500}
    (msg,) = body["messages"]
    assert msg["role"] == "user" and msg["content"] == "describe"
    assert msg["images"] == [base64.b64encode(j).decode("ascii") for j in JPEGS]


@pytest.mark.parametrize(
    "reply",
    [
        requests.ConnectionError("refused"),
        Resp({"error": "model crashed"}, status=500),
        Resp({"done": True}),  # no message
        "not json at all",
        '{"events": [',  # cut off (done_reason=length)
        "[1, 2]",  # JSON, but not an object
    ],
    ids=["unreachable", "http-error", "no-message", "not-json", "truncated", "array"],
)
def test_describe_failure_gives_none(ollama, reply, caplog) -> None:
    ollama.replies = [reply]
    rater = vision.VisionRater()
    with caplog.at_level(logging.WARNING):
        assert rater.describe("describe", JPEGS, vw.RESPONSE_SCHEMA) is None
    assert rater.last_prompt_tokens in (None, 1234)  # counted when Ollama answered
    assert any("Vision" in r.getMessage() for r in caplog.records)


def test_describe_refuses_a_redirect(ollama, monkeypatch, caplog) -> None:
    sent: list[dict[str, Any]] = []

    def moved(url: str, json: dict[str, Any], **kw: Any) -> Resp:
        sent.append(kw)
        resp = Resp({}, status=307)
        resp.is_redirect = True
        return resp

    monkeypatch.setattr(vision.requests, "post", moved)
    with caplog.at_level(logging.WARNING):
        assert vision.VisionRater().describe("describe", JPEGS, vw.RESPONSE_SCHEMA) is None
    assert sent == [{"timeout": 600, "allow_redirects": False}]  # not followed
    assert any("redirect" in r.getMessage() for r in caplog.records)


class FakeSubscription:
    name, model, local, host = "codex", "gpt-x", False, "chatgpt.com"

    def __init__(self, text: str) -> None:
        self.text = text
        self.sent: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    def capabilities(self) -> None:
        pass

    def chat(self, messages, **kw) -> Reply:
        self.sent.append((messages, kw))
        return Reply(self.text)


@pytest.fixture
def hosted(home, monkeypatch) -> FakeSubscription:
    """The Vision role is a hosted subscription the user agreed to; Ollama is never asked."""
    fake = FakeSubscription("Here you go: " + json.dumps(REPLY))
    monkeypatch.setattr(providers, "make", lambda *_a, **_k: fake)

    def no_network(*_a, **_k):
        raise AssertionError("Ollama was asked")

    monkeypatch.setattr(vision.requests, "post", no_network)
    monkeypatch.setattr(vision.requests, "get", no_network)
    set_config(
        home,
        vision_assistant={"provider": "codex", "model": "gpt-x", "consent": ["codex@chatgpt.com"]},
    )
    return fake


def test_describe_goes_to_the_hosted_vision_role_with_the_schema(hosted) -> None:
    rater = vision.VisionRater()
    assert rater.available()
    assert rater.describe("describe", JPEGS, vw.RESPONSE_SCHEMA) == REPLY
    ((msgs, kw),) = hosted.sent
    assert kw["schema"] == vw.RESPONSE_SCHEMA
    (msg,) = msgs
    assert msg["content"][0] == {"type": "text", "text": "describe"}
    assert [p["data"] for p in msg["content"][1:]] == [base64.b64encode(j).decode() for j in JPEGS]


def test_describe_sends_nothing_to_a_hosted_role_without_consent(hosted, home) -> None:
    set_config(home, vision_assistant={"provider": "codex", "model": "gpt-x", "consent": []})
    rater = vision.VisionRater()
    assert not rater.available()
    assert rater.describe("describe", JPEGS, vw.RESPONSE_SCHEMA) is None
    assert hosted.sent == []


def test_describe_hosted_reply_without_json_gives_none(hosted) -> None:
    hosted.text = "I cannot help with that."
    assert vision.VisionRater().describe("describe", JPEGS, vw.RESPONSE_SCHEMA) is None


# --------------------------------------------------------------------------- vlm_windows requests


def _source(tmp_path: Path) -> Source:
    (tmp_path / "cache").mkdir(exist_ok=True)
    return Source(tmp_path / "game.mp4", 60.0, tmp_path / "cache", [])


def test_vision_request_keeps_text_and_frames_in_order(tmp_path) -> None:
    src = _source(tmp_path)
    win = vw.Window(str(src.path), 5.0, 20.0, 1.0, ["seed"])
    lines = [{"name": src.name, "start": 6.0, "end": 8.0, "text": "nice shot"}]
    events = [{"kind": "audio", "t": 7.0, "label": "Laughter", "confidence": 0.9}]
    msgs = vw.build_messages(src, win, JPEGS, vw.VlmParams(), None, lines, events)
    prompt, images = vw.vision_request(msgs)
    assert images == JPEGS
    assert [m["role"] for m in msgs] == ["user"]
    assert "Frame 1 @ 0.0s: [image 1]" in prompt and "Frame 3 @ 10.0s: [image 3]" in prompt
    assert "nice shot" in prompt  # the transcript and the detected sounds replace the audio
    assert "Laughter" in prompt
    assert prompt.rstrip().endswith('"label": "2-4 word slug"}')


def _describe(tmp_path: Path, monkeypatch, rater: vision.VisionRater, ski: bool = False):
    src = _source(tmp_path)
    monkeypatch.setattr(vw, "grab_frames", lambda *a, **k: list(JPEGS))
    win = vw.Window(str(src.path), 5.0, 20.0, 1.0, ["seed"])
    params = vw.VlmParams(model="qwen3.5:9b", ski=ski)
    results, stats = vw.describe_windows(rater, {str(src.path): src}, [win], params, {}, [], {})
    return src, params, results, stats


def test_describe_windows_with_the_vision_client(ollama, tmp_path, monkeypatch) -> None:
    ollama.replies = [REPLY]
    src, params, results, stats = _describe(tmp_path, monkeypatch, vision.VisionRater())
    (body,) = ollama.chats
    assert body["format"] == vw.RESPONSE_SCHEMA
    # Ollama indents structured output: twice the token budget of a compact reply
    assert body["options"] == {
        "temperature": 0.2,
        "num_ctx": params.ctx,
        "num_predict": 2 * params.max_tokens,
    }
    assert "input_audio" not in json.dumps(body)
    (res,) = results
    assert res["label"] == "kill-after-banter" and res["highlight_score"] == 6  # parse_reply
    assert res["prompt_tokens"] == 1234 and res["frames"] == 3
    assert stats["ok"] == 1 and stats["failed"] == 0
    # cached under the vision model's key
    stored = cache.load_json(layer_cache_path(src.cdir, "vlm", params.key()))
    assert stored["params"]["model"] == "qwen3.5:9b"
    assert stored["windows"]["5.0-20.0"]["summary"] == REPLY["summary"]
    assert not layer_cache_path(src.cdir, "vlm", vw.VlmParams(model="other").key()).exists()
    # a rerun takes the window from the cache and asks nothing
    _, _, again, stats = _describe(tmp_path, monkeypatch, vision.VisionRater())
    assert stats["cached"] == 1 and len(ollama.chats) == 1 and again[0]["label"] == res["label"]


def test_describe_windows_ski_uses_the_ski_schema(ollama, tmp_path, monkeypatch) -> None:
    tags = dict.fromkeys(vw.SKI_TAGS, 2) | {"jump_air": 8}
    ollama.replies = [REPLY | {"tags": tags}]
    _, _, (res,), _ = _describe(tmp_path, monkeypatch, vision.VisionRater(), ski=True)
    assert ollama.chats[0]["format"] == vw.RESPONSE_SCHEMA_SKI
    assert res["tags"]["jump_air"] == 8


def test_describe_windows_retries_once_then_counts_a_failure(ollama, tmp_path, monkeypatch) -> None:
    ollama.replies = [requests.ConnectionError("refused"), "garbage"]
    src, params, results, stats = _describe(tmp_path, monkeypatch, vision.VisionRater())
    assert [c["options"]["temperature"] for c in ollama.chats] == [0.2, 0.0]
    assert results == [] and stats["failed"] == 1 and stats["ok"] == 0
    assert not layer_cache_path(src.cdir, "vlm", params.key()).exists()  # nothing cached


def test_describe_windows_retry_succeeds(ollama, tmp_path, monkeypatch) -> None:
    ollama.replies = ['{"events": [', REPLY]
    _, _, results, stats = _describe(tmp_path, monkeypatch, vision.VisionRater())
    assert len(ollama.chats) == 2 and stats["ok"] == 1 and len(results) == 1


def test_vision_frames_are_capped_to_the_image_token_budget(ollama, tmp_path, monkeypatch) -> None:
    # 640 tokens of 32x32 px at 16:9: 606 px high, so 16 frames fit the 16k context
    assert vw.FRAME_HEIGHT == 606
    heights: list[int] = []
    monkeypatch.setattr(vw, "grab_frames", lambda *a, **k: heights.append(a[4]) or list(JPEGS))
    src = _source(tmp_path)
    win = vw.Window(str(src.path), 5.0, 20.0, 1.0, ["seed"])
    ollama.replies = [REPLY]
    vw.describe_windows(
        vision.VisionRater(), {str(src.path): src}, [win], vw.VlmParams(), {}, [], {}
    )
    assert heights == [606]


# --------------------------------------------------------------------------- pipeline wiring


@pytest.fixture
def extra_ok(home, monkeypatch):
    """The events extra counts as installed; no model files anywhere."""
    monkeypatch.setattr(pipeline.importlib.util, "find_spec", lambda name: object())


def test_preflight_needs_no_model_files_for_the_vlm_layer(extra_ok) -> None:
    pipeline.preflight(("vlm",))
    pipeline.preflight(("ocr", "vlm"))


def _run_vlm(tmp_path: Path, monkeypatch, **kw: Any) -> dict[str, Any]:
    """``run_events`` with the VLM layer only over one fake source (``kw``: more of its
    arguments); returns events.json."""
    src = _source(tmp_path)
    project = tmp_path / "proj"
    project.mkdir(exist_ok=True)
    cand = {"id": "h001", "source": str(src.path), "start": 10.0, "end": 14.0, "score": 70}
    monkeypatch.setattr(
        pipeline,
        "load_highlights",
        lambda p: {"sources": [{"path": str(src.path)}], "candidates": [cand]},
    )
    monkeypatch.setattr(pipeline, "load_source", lambda path, dur: src)
    monkeypatch.setattr(pipeline, "bad_ranges", lambda *a: {})
    monkeypatch.setattr(pipeline, "load_transcript_lines", lambda project: [])
    monkeypatch.setattr(pipeline, "detect_audio_events", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "detect_ocr_events", lambda *a, **k: ([], {}))
    monkeypatch.setattr(common, "gpu_memory_used_mib", lambda: None)
    monkeypatch.setattr(vw, "grab_frames", lambda *a, **k: list(JPEGS))
    jpath, _ = pipeline.run_events(project, layers=("vlm",), **kw)
    return cache.load_json(jpath)


def test_run_events_describes_windows_with_the_vision_client(
    extra_ok, ollama, tmp_path, monkeypatch
) -> None:
    ollama.replies = [REPLY]
    data = _run_vlm(tmp_path, monkeypatch)
    assert len(ollama.chats) == 1 and ollama.unloads == [{"model": "qwen3.5:9b", "keep_alive": 0}]
    assert data["settings"]["vlm"]["model"] == "qwen3.5:9b"
    assert "audio" not in data["settings"]["vlm"]
    vlm = [e for e in data["events"] if e["kind"] == "vlm"]
    assert [e["label"] for e in vlm] == ["player gets a kill"]
    assert data["stats"]["vlm"]["ok"] == 1


def test_run_events_without_the_vision_model_keeps_going(
    extra_ok, home, tmp_path, monkeypatch, caplog
) -> None:
    def down(*_a, **_k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(vision.requests, "get", down)
    monkeypatch.setattr(vision.requests, "post", down)
    with caplog.at_level(logging.WARNING), jsonout.run("events") as res:
        data = _run_vlm(tmp_path, monkeypatch)
    assert not [e for e in data["events"] if e["kind"] == "vlm"]
    assert any("not available" in r.getMessage() for r in caplog.records)
    # --json callers see that the layer was skipped
    (warning,) = res.warnings
    assert warning["code"] == "vlm_unavailable" and "not available" in warning["message"]


@pytest.mark.parametrize("force", [(), ("vlm",)], ids=["cached", "force-vlm"])
def test_run_events_without_the_vision_model_keeps_cached_descriptions(
    extra_ok, ollama, tmp_path, monkeypatch, force
) -> None:
    ollama.replies = [REPLY]
    _run_vlm(tmp_path, monkeypatch)  # describes the window and caches it
    ollama.replies = [REPLY]  # a second window for the forced run, if it were asked

    def down(*_a, **_k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(vision.requests, "get", down)
    monkeypatch.setattr(vision.requests, "post", down)
    with jsonout.run("events") as res:
        data = _run_vlm(tmp_path, monkeypatch, force=force)
    # --force vlm with the model down keeps the saved description instead of losing it
    vlm = [e for e in data["events"] if e["kind"] == "vlm"]
    assert [e["label"] for e in vlm] == ["player gets a kill"]
    assert data["stats"]["vlm"]["cached"] == 1
    assert [w["code"] for w in res.warnings] == (["vlm_unavailable"] if force else [])


def test_run_events_passes_the_ollama_options_to_the_vision_client(
    extra_ok, home, tmp_path, monkeypatch
) -> None:
    seen: list[tuple[Any, ...]] = []
    real = pipeline.VisionRater

    def rater(*a: Any, **k: Any) -> vision.VisionRater:
        seen.append((a, k))
        return real(*a, **k)

    monkeypatch.setattr(pipeline, "VisionRater", rater)
    monkeypatch.setattr(vision.VisionRater, "available", lambda self: False)
    data = _run_vlm(
        tmp_path, monkeypatch, ollama_url="http://gpu-box:11434", ollama_model="my-vl:7b"
    )
    assert seen == [(("http://gpu-box:11434", "my-vl:7b"), {})]
    assert data["settings"]["vlm"]["model"] == "my-vl:7b"
