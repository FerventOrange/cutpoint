"""``view_moment`` (#640): the timeline sheet, its cache, the frame consent, the per-turn cap,
the tool hidden for text-only models and how each provider sends the sheet.

Frames are synthetic JPEGs (no ffmpeg); every provider's HTTP is the ``Wire`` stub of
``test_assistant_byo``, so no real API is called.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cutpoint.assistant import providers, session, watch
from cutpoint.assistant.providers import (
    AnthropicProvider,
    Capabilities,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    Reply,
    XAIProvider,
)
from test_assistant import FakeProvider, _chunks, _Resp, call, durations
from test_assistant_byo import (
    KEY,
    XKEY,
    Wire,
    an_reply,
    oa_call,
    oa_reply,
    rs_call,
    rs_reply,
    rs_text,
    tool_use,
)
from test_assistant_byo import no_keys as no_keys  # the autouse fixture

cv2 = pytest.importorskip("cv2")


def jpeg(w: int = 640, h: int = 360, value: int = 128) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    return img


@pytest.fixture(autouse=True)
def fresh_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watch, "_TEXT_ONLY", set())


@pytest.fixture
def src(tmp_path: Path) -> Path:
    f = tmp_path / "videos" / "a.mp4"
    f.parent.mkdir()
    f.write_bytes(b"video")
    return f


@pytest.fixture
def proj_dir(tmp_path: Path) -> Path:
    p = tmp_path / "proj"
    (p / "ui" / "assistant").mkdir(parents=True)
    return p


class Grab:
    """A frame grabber that counts its calls."""

    def __init__(self, w: int = 640, h: int = 360) -> None:
        self.times: list[float] = []
        self.size = (w, h)

    def __call__(self, _src: Path, t: float) -> bytes:
        self.times.append(t)
        return jpeg(*self.size)


CANDS = [
    {"id": "c1", "source": "/v/a.mp4", "start": 10.0, "end": 20.0, "score": 90, "label": "dive"},
    {"id": "c2", "source": "/v/a.mp4", "start": 40.0, "end": 70.0, "score": 70, "label": "reef"},
    {"id": "c3", "source": "/v/a.mp4", "start": 80.0, "end": 88.0, "score": 60, "label": "boat"},
]
TRANSCRIPT = "[a.mp4 00:00:12.0-00:00:14.0] look at that fish\n[a.mp4 00:01:30.0-00:01:31.0] far"
EVENTS = {
    "events": [
        {
            "source": "/v/a.mp4",
            "start": 15.0,
            "end": 16.5,
            "kind": "audio",
            "label": "Laughter",
            "confidence": 0.82,
        },
        {
            "source": "/v/a.mp4",
            "start": 50.0,
            "end": 51.0,
            "kind": "audio",
            "label": "Scream",
            "confidence": 0.7,
        },
        {"source": "/v/b.mp4", "start": 15.0, "end": 16.0, "kind": "audio", "label": "Cheer"},
        {"source": "/v/a.mp4", "start": 15.0, "end": 15.0, "kind": "ocr", "label": "KILL"},
    ]
}


def proj() -> session.Project:
    return session.Project(list(CANDS), [], durations(), TRANSCRIPT, "k0")


def viewer(p: Path, provider: Any, src: Path, grab: Grab | None = None, given=()) -> watch.Viewer:
    return watch.Viewer(
        p, provider, resolve=lambda _s: src, consent=lambda: tuple(given), grab=grab or Grab()
    )


def run(provider: Any, view: watch.Viewer | None, text: str = "show me the fish") -> session.Turn:
    turn = session.Turn(provider, proj(), text, viewer=view)
    turn.start()
    assert turn.done.wait(10)
    return turn


def answer(text: str = "Done.") -> Reply:
    return Reply(text, [], {"role": "assistant", "content": text})


def images_in(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        p
        for m in messages
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if isinstance(p, dict) and p.get("type") == "image"
    ]


# -- the sheet ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "n"), [(10, 20, 6), (10, 22, 6), (10, 22.1, 9), (0, 90, 9)]
)
def test_frames_spread_evenly_inside_the_window_6_or_9_by_length(start, end, n) -> None:
    times = watch.frame_times(start, end)
    assert len(times) == n == watch.frame_count(start, end)
    assert start < times[0] and times[-1] < end
    gaps = np.diff([start, *times, end])
    assert np.allclose(gaps, gaps[0])


@pytest.mark.parametrize("aspect", [16 / 9, 4 / 3, 9 / 16, 2.0])
@pytest.mark.parametrize("n", [6, 9])
def test_the_sheet_is_a_3_wide_grid_of_about_1_5k_claude_tokens(aspect: float, n: int) -> None:
    w, h, cols, rows = watch.sheet_size(aspect, n)
    assert (cols, rows) == (3, n // 3)
    sw, sh = w * cols, h * rows
    assert max(sw, sh) <= watch.MAX_SIDE
    assert sw * sh <= watch.SHEET_PIXELS
    assert sw * sh / 750 <= 1600  # Claude's image tokens
    assert abs(w / h - aspect) / aspect < 0.02  # tiles keep the footage's shape
    assert w >= 300 or h >= 300  # legible for a local model


def test_timeline_sheet_stamps_each_tile_and_fills_a_short_row_black() -> None:
    times = [11.0, 12.5, 14.0, 15.5, 17.0, 18.5, 20.0]
    sheet = watch.timeline_sheet([jpeg(value=200) for _ in times], times)
    img = decode(sheet)
    w, h, cols, rows = watch.sheet_size(16 / 9, 7)
    assert img.shape[:2] == (h * rows, w * cols) == (h * 3, w * 3)
    band = img[: h // 8, : w // 2]
    assert band.min() < 40 and band.max() > 230  # white text on a black band
    assert img[h // 2, w // 2].mean() > 150  # the frame itself
    assert img[2 * h + h // 2, w + w // 2].max() < 30  # the 8th tile: black
    assert watch.stamp(3, 751.4) == "3  00:12:31.4"
    with pytest.raises(ValueError, match="decoded"):
        watch.timeline_sheet([b"not a jpeg"], [1.0])


# -- the cache ---------------------------------------------------------------------------------


def test_a_sheet_is_cached_per_moment_window_and_source_file(proj_dir: Path, src: Path) -> None:
    grab = Grab()
    v = viewer(proj_dir, FakeProvider([]), src, grab)
    first, cached = v.sheet("c1", "/v/a.mp4", 10.0, 20.0)
    assert not cached and len(grab.times) == 6
    folder = proj_dir / "ui" / "assistant" / "sheets"
    assert [f.name.split("-")[0] for f in folder.iterdir()] == ["c1"]
    again, cached = v.sheet("c1", "/v/a.mp4", 10.0, 20.0)
    assert cached and again == first and len(grab.times) == 6

    st = src.stat()  # the source file changed: built again, the old sheet removed
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    _, cached = v.sheet("c1", "/v/a.mp4", 10.0, 20.0)
    assert not cached and len(grab.times) == 12 and len(list(folder.iterdir())) == 1

    _, cached = v.sheet("c1", "/v/a.mp4", 10.0, 30.0)  # a trimmed window: 9 frames
    assert not cached and len(grab.times) == 21 and len(list(folder.iterdir())) == 1
    v.sheet("c2", "/v/a.mp4", 40.0, 70.0)
    assert len(list(folder.iterdir())) == 2


def test_a_cache_folder_linked_out_of_the_project_is_not_written(
    proj_dir: Path, src: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    try:
        (proj_dir / "ui" / "assistant" / "sheets").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("no symlinks here")
    v = viewer(proj_dir, FakeProvider([]), src)
    sheet, cached = v.sheet("c1", "/v/a.mp4", 10.0, 20.0)
    assert sheet and not cached and list(outside.iterdir()) == []


# -- consent -----------------------------------------------------------------------------------


class Hosted(FakeProvider):
    name = "anthropic"
    local = False
    host = "api.anthropic.com"
    context_tokens = 32768


def test_frames_keys_are_per_provider_and_host_and_none_on_this_machine() -> None:
    assert watch.frames_key(FakeProvider([])) is None
    assert watch.frames_key(Hosted([])) == "frames-v1:anthropic@api.anthropic.com"
    lan = OllamaProvider("http://192.168.1.5:11434", "qwen3.5:9b", 8192)
    assert watch.frames_key(lan) == "frames-v1:ollama@192.168.1.5:11434"
    assert watch.frames_key(OllamaProvider("http://127.0.0.1:11434", "q", 8192)) is None
    assert watch.frames_key(OllamaProvider("http://localhost:11434", "q", 8192)) is None
    # neither of the other consents is a frames consent
    hosted = Hosted([])
    assert providers.consent_key(hosted) != watch.frames_key(hosted)
    assert providers.consent_key(hosted, "vision") != watch.frames_key(hosted)


def test_frames_consent_is_per_project(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    key = "frames-v1:anthropic@api.anthropic.com"
    data = {"assistant": {"frames_consent": {watch.project_key(a): [key]}}}
    assert watch.frames_view(Hosted([]), a, data) == {
        "needed": False,
        "key": key,
        "host": "api.anthropic.com",
    }
    assert watch.frames_view(Hosted([]), b, data)["needed"] is True
    for broken in ({"assistant": {"frames_consent": []}}, {"assistant": "x"}, {}):
        assert watch.frames_consent(broken, a) == ()


def test_hosted_without_frames_consent_sends_no_frame_and_says_so(
    proj_dir: Path, src: Path
) -> None:
    grab = Grab()
    fake = Hosted([call("view_moment", id="m1"), answer("I can't see it yet.")])
    turn = run(fake, viewer(proj_dir, fake, src, grab, given=["frames-v1:anthropic@elsewhere"]))
    assert turn.status == "done"
    assert grab.times == [] and not (proj_dir / "ui" / "assistant" / "sheets").exists()
    assert all(not images_in(r["messages"]) for r in fake.requests)
    result = fake.requests[1]["messages"][-1]["content"]
    assert "not allowed video frames" in result and "nothing was sent" in result
    ev = next(e for e in turn.events if e["kind"] == "frames_consent")
    assert ev == {
        "n": ev["n"],
        "kind": "frames_consent",
        "id": "c1",
        "key": "frames-v1:anthropic@api.anthropic.com",
        "host": "api.anthropic.com",
    }


def test_two_denied_views_in_one_turn_show_one_consent_box(proj_dir: Path, src: Path) -> None:
    grab = Grab()
    fake = Hosted([call("view_moment", id="m1"), call("view_moment", id="m2"), answer("No.")])
    turn = run(fake, viewer(proj_dir, fake, src, grab))
    assert turn.status == "done" and grab.times == []
    assert len([e for e in turn.events if e["kind"] == "frames_consent"]) == 1
    denied = [e for e in turn.events if e["kind"] == "tool"]
    assert [(e["id"], e["error"]) for e in denied] == [("c2", "frames not allowed")]
    first, second = (fake.requests[1]["messages"][-1], fake.requests[2]["messages"][-1])
    assert first["content"] == second["content"] and "nothing was sent" in second["content"]


def test_a_subscription_provider_needs_its_own_frames_consent(proj_dir: Path, src: Path) -> None:
    sub = providers.SubscriptionProvider("claude-code", model="claude-x")
    key = "frames-v1:claude-code@api.anthropic.com"
    assert watch.frames_key(sub) == key
    assert not viewer(proj_dir, sub, src).allowed()
    assert viewer(proj_dir, sub, src, given=[key]).allowed()
    item = {"source": "/v/a.mp4", "start": 10.0, "end": 20.0}
    grab = Grab()
    seen = viewer(proj_dir, sub, src, grab).view("m1", "c1", item, {}, "")
    assert seen.sheet is None and seen.event["kind"] == "frames_consent"
    assert seen.event["key"] == key and grab.times == []


def test_a_second_image_refusal_fails_the_turn_after_three_requests(
    proj_dir: Path, src: Path
) -> None:
    def refuse(_messages):
        raise ProviderError("OpenAI-compatible x answered 400: image input is not supported")

    fake = FakeProvider([call("view_moment", id="m1"), refuse, refuse])
    fake.model = "local-llama"
    turn = run(fake, viewer(proj_dir, fake, src))
    assert turn.status == "failed" and len(fake.requests) == 3
    assert not images_in(fake.requests[2]["messages"])


def test_a_local_model_watches_without_consent(proj_dir: Path, src: Path) -> None:
    proj_dir.joinpath("events.json").write_text(json.dumps(EVENTS), "utf-8")
    fake = FakeProvider([call("view_moment", id="m1"), answer("A fish.")])
    turn = run(fake, viewer(proj_dir, fake, src))
    assert turn.status == "done"
    second = fake.requests[1]["messages"]
    result, image_msg = second[-2], second[-1]
    assert result["role"] == "tool" and "<<<DATA moment view>>>" in result["content"]
    text = result["content"]
    assert "3x2 grid" in text and "1 00:00:11.4" in text and "6 00:00:18.6" in text
    assert "look at that fish" in text and "far" not in text
    assert "Laughter (0.82, 1.5 s)" in text and "Scream" not in text and "Cheer" not in text
    assert "KILL" not in text
    [part] = images_in([image_msg])
    assert image_msg["role"] == "user" and part["media_type"] == "image/jpeg"
    assert decode(base64.b64decode(part["data"])).shape[1] > 1000
    tool = next(e for e in turn.events if e["kind"] == "tool")
    assert tool == {
        "n": tool["n"],
        "kind": "tool",
        "name": "view_moment",
        "id": "c1",
        "frames": 6,
        "cached": False,
    }
    # the chat record keeps no image
    assert "data" not in json.dumps(turn.record()["events"])


def test_hosted_with_frames_consent_watches(proj_dir: Path, src: Path) -> None:
    fake = Hosted([call("view_moment", id="m2"), answer()])
    turn = run(fake, viewer(proj_dir, fake, src, given=[watch.frames_key(fake)]))
    assert turn.status == "done" and len(images_in(fake.requests[1]["messages"])) == 1
    assert "3x3 grid" in fake.requests[1]["messages"][-2]["content"]


# -- the per-turn cap ----------------------------------------------------------------------------


@pytest.mark.parametrize(("provider", "cap"), [(FakeProvider, 2), (Hosted, 4)])
def test_a_turn_views_at_most_its_cap_and_a_repeat_sends_nothing_new(
    proj_dir: Path, src: Path, provider: type, cap: int
) -> None:
    ids = ["m1", "m1", "m2", "m3", "m4"]  # a repeat, then 3 more moments
    fake = provider([*(call("view_moment", id=i) for i in ids), answer()])
    p = session.Project(
        [*CANDS, {"id": "c4", "source": "/v/a.mp4", "start": 1.0, "end": 5.0, "label": "x"}],
        [],
        durations(),
        "",
        "k0",
    )
    v = viewer(proj_dir, fake, src, given=[watch.frames_key(fake)] if not fake.local else [])
    turn = session.Turn(fake, p, "look", viewer=v)
    turn.start()
    assert turn.done.wait(10)
    assert watch.view_cap(fake) == cap
    assert len(images_in(fake.requests[-1]["messages"])) == min(cap, 4)
    results = [m["content"] for m in fake.requests[-1]["messages"] if m.get("role") == "tool"]
    assert any("already viewed" in r for r in results)
    if cap == 2:
        assert any(f"at most {cap} moments" in r for r in results)
        assert any(e.get("error") == "limit" for e in turn.events)
    rules = fake.requests[0]["messages"][0]["content"]
    assert f"at most {cap} moments" in rules and "never for every moment" in rules


# -- offered or hidden ---------------------------------------------------------------------------


def names(tools: list[dict[str, Any]] | None) -> list[str]:
    return [t["function"]["name"] for t in tools or []]


@pytest.mark.parametrize(
    ("caps", "model", "offered"),
    [
        (Capabilities(True, True, True), "anything", True),
        (Capabilities(True, True, False), "qwen3.5:9b", False),  # Ollama without vision
        (Capabilities(True, True), "gpt-6-astra", True),  # unknown: offered
        (Capabilities(True, True), "grok-code-fast-1", False),  # a known text-only name
        (Capabilities(True, True), "gpt-3.5-turbo", False),
        (Capabilities(True, True), "gpt-5.3-codex", False),
    ],
)
def test_the_tool_is_hidden_for_text_only_models(
    proj_dir: Path, src: Path, caps, model, offered
) -> None:
    fake = FakeProvider([answer("hi")], caps)
    fake.model = model
    run(fake, viewer(proj_dir, fake, src))
    assert ("view_moment" in names(fake.requests[0]["tools"])) is offered
    rules = fake.requests[0]["messages"][0]["content"]
    assert ("view_moment" in rules) is offered


def test_no_viewer_no_tool_and_an_unoffered_call_is_unknown(proj_dir: Path) -> None:
    fake = FakeProvider([call("view_moment", id="m1"), answer()])
    run(fake, None)
    assert "view_moment" not in names(fake.requests[0]["tools"])
    assert "unknown tool 'view_moment'" in fake.requests[1]["messages"][-1]["content"]


def test_json_mode_offers_view_moment_in_the_schema_and_sends_the_sheet(
    proj_dir: Path, src: Path
) -> None:
    fake = FakeProvider(
        [
            Reply('{"tool": "view_moment", "arguments": {"id": "m1"}}'),
            Reply('{"tool": "reply", "text": "a fish"}'),
        ],
        Capabilities(False, True, True),
    )
    turn = run(fake, viewer(proj_dir, fake, src))
    assert turn.status == "done"
    first = fake.requests[0]
    assert "view_moment" in first["schema"]["properties"]["tool"]["enum"]
    assert first["schema"]["properties"]["tool"]["enum"][-1] == "reply"
    assert "view_moment {id}" in first["messages"][0]["content"]
    assert session.JSON_SCHEMA["properties"]["tool"]["enum"][-1] == "reply"  # unchanged
    msgs = fake.requests[1]["messages"]
    assert msgs[-2]["content"].startswith("view_moment result:")
    assert len(images_in(msgs)) == 1


def test_a_model_that_refuses_the_image_is_asked_again_without_it_and_offered_no_more(
    proj_dir: Path, src: Path
) -> None:
    def refuse(_messages):
        raise ProviderError("OpenAI-compatible x answered 400: image input is not supported")

    fake = FakeProvider([call("view_moment", id="m1"), refuse, answer("From the text: m1.")])
    fake.model = "local-llama"
    turn = run(fake, viewer(proj_dir, fake, src))
    assert turn.status == "done" and turn.events[-1]["text"] == "From the text: m1."
    retry = fake.requests[2]
    assert not images_in(retry["messages"])
    assert "takes no images" in retry["messages"][-1]["content"]
    assert "view_moment" not in names(retry["tools"])
    assert not watch.takes_images(fake, fake.caps)  # the rest of the session

    again = FakeProvider([answer()])
    again.model = "local-llama"
    run(again, viewer(proj_dir, again, src))
    assert "view_moment" not in names(again.requests[0]["tools"])


@pytest.mark.parametrize(
    ("why", "image"),
    [
        ("answered 400: this model does not support image input", True),
        ("answered 400: Invalid content type. image_url is only supported by certain models", True),
        ("answered 400: model is not multimodal", True),
        ("rejected the API key (401: invalid key for vision)", False),
        ("is rate limiting or out of credit (429: images per minute)", False),
        ("answered 400: max_tokens too large", False),
    ],
)
def test_rejected_image_tells_an_image_refusal_from_other_errors(why: str, image: bool) -> None:
    assert watch.rejected_image(ProviderError(f"X {why}")) is image


def test_another_error_after_a_sheet_is_not_retried(proj_dir: Path, src: Path) -> None:
    def down(_messages):
        raise ProviderError("X answered 500: overloaded; try again later")

    fake = FakeProvider([call("view_moment", id="m1"), down])
    turn = run(fake, viewer(proj_dir, fake, src))
    assert turn.status == "failed" and len(fake.requests) == 2
    assert watch.takes_images(fake, fake.caps)


# -- how each provider sends the sheet -------------------------------------------------------------


def _ollama_answers() -> list[Any]:
    calls = [{"function": {"name": "view_moment", "arguments": {"id": "m1"}}}]
    return [
        _Resp({"capabilities": ["completion", "tools", "vision"]}),
        _Resp(lines=_chunks("", calls)),
        _Resp(lines=_chunks("A fish.")),
    ]


def _check_ollama(body: dict[str, Any]) -> None:
    *_, result, image = body["messages"]
    assert result == {"role": "tool", "tool_name": "view_moment", "content": result["content"]}
    assert image["role"] == "user" and isinstance(image["content"], str)
    assert "timeline sheet" in image["content"] and len(image["images"]) == 1
    assert all(not isinstance(m.get("content"), list) for m in body["messages"])


def _check_anthropic(body: dict[str, Any]) -> None:
    last = body["messages"][-1]
    assert last["role"] == "user"
    kinds = [b["type"] for b in last["content"]]
    assert kinds == ["tool_result", "text", "image"]  # the result first, then the sheet
    assert last["content"][0]["tool_use_id"] == "tu1"
    src = last["content"][2]["source"]
    assert src["type"] == "base64" and src["media_type"] == "image/jpeg" and src["data"]


def _check_chat_completions(body: dict[str, Any]) -> None:
    *_, result, image = body["messages"]
    assert result["role"] == "tool" and result["tool_call_id"] == "c1"
    assert image["role"] == "user"
    part = image["content"][1]
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def _check_responses(body: dict[str, Any]) -> None:
    *_, result, image = body["input"]
    assert result["type"] == "function_call_output" and result["call_id"] == "c1"
    assert image["role"] == "user"
    assert [p["type"] for p in image["content"]] == ["input_text", "input_image"]
    assert image["content"][1]["image_url"].startswith("data:image/jpeg;base64,")


PROVIDERS = {
    "ollama": (
        lambda: OllamaProvider("http://127.0.0.1:11434", "qwen3.5:9b", 8192),
        _ollama_answers,
        _check_ollama,
    ),
    "anthropic": (
        lambda: AnthropicProvider("", "", 0, KEY),
        lambda: [
            _Resp({"id": "claude-sonnet-5-5"}),
            an_reply(tool_use("tu1", "view_moment", id="m1")),
            an_reply({"type": "text", "text": "A fish."}),
        ],
        _check_anthropic,
    ),
    "openai-compatible": (
        lambda: OpenAIProvider("https://openrouter.ai/api/v1", "some/vl-model", 0, KEY),
        lambda: [
            _Resp({"data": [{"id": "some/vl-model"}]}),
            oa_reply(None, [oa_call("c1", "view_moment", '{"id": "m1"}')]),
            oa_reply("A fish."),
        ],
        _check_chat_completions,
    ),
    "xai": (
        lambda: XAIProvider("", "grok-4.7", 0, XKEY),
        lambda: [
            _Resp({"data": [{"id": "grok-4.7"}]}),
            oa_reply(None, [oa_call("c1", "view_moment", '{"id": "m1"}')]),
            oa_reply("A fish."),
        ],
        _check_chat_completions,
    ),
    "openai": (
        lambda: OpenAIProvider("", "gpt-6-astra", 0, KEY),
        lambda: [
            _Resp({"data": [{"id": "gpt-6-astra"}]}),
            rs_reply(rs_call("c1", "view_moment", '{"id": "m1"}')),
            rs_reply(rs_text("A fish.", "final_answer")),
        ],
        _check_responses,
    ),
}


@pytest.mark.parametrize("name", list(PROVIDERS))
def test_each_provider_sends_the_sheet_after_the_tool_result(
    monkeypatch, proj_dir: Path, src: Path, name: str
) -> None:
    make, answers, check = PROVIDERS[name]
    wire = Wire(monkeypatch, *answers())
    p = make()
    turn = run(p, viewer(proj_dir, p, src, given=[watch.frames_key(p) or ""]))
    assert turn.status == "done", turn.error
    first, second = [s["json"] for s in wire.sent if s["method"] == "POST"][-2:]
    assert "view_moment" in json.dumps(first)
    check(second)
    assert wire.answers == []


def test_ollama_turns_image_parts_into_images_only_for_list_content() -> None:
    plain = {"role": "user", "content": "hi"}
    assert providers._ollama_message(plain) is plain
    got = providers._ollama_message(watch.image_message("sheet", b"\xff\xd8x"))
    assert got == {
        "role": "user",
        "content": "sheet",
        "images": [base64.b64encode(b"\xff\xd8x").decode()],
    }


@pytest.mark.parametrize(
    ("provider", "model", "want"),
    [
        ("anthropic", "claude-sonnet-5-5", True),
        ("anthropic", "some-new-model", True),  # every model of the Anthropic API
        ("claude-code", "claude-opus-5-5", True),
        ("openai", "gpt-6-astra", True),
        ("openai", "GPT-5.6-sol", True),
        ("openai", "gpt-5.3-codex", False),  # text-only parts win over the prefixes
        ("openai", "gpt-3.5-turbo", False),
        ("openai", "o3-mini", False),
        ("xai", "grok-4.7", True),
        ("xai", "grok-code-fast-1", False),
        ("xai", "grok-build-0.1", False),
        ("antigravity", "gemini-3.1-pro", True),
        ("ollama", "qwen3.5:9b", True),
        ("ollama", "llama3.1:8b", None),
        ("openai", "some/vl-model", None),
    ],
)
def test_accepts_images_by_name_only(provider: str, model: str, want: bool | None) -> None:
    assert providers.accepts_images(provider, model) is want
    hosted = FakeProvider([])
    hosted.name, hosted.model = provider, model
    assert providers.accepts_images(hosted, model) is want
