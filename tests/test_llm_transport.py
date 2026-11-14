"""LLMClient transport: httpx streaming, typed outcomes, think filter, key hygiene.

Every test drives `httpx.MockTransport`, so nothing leaves the process. MockTransport enforces no timeouts
and bypasses httpcore, which is why timeouts are asserted on the request extensions instead of by waiting.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx
import pytest

from voice_agent.llm.client import OFFLINE_LINE, LLMClient, LlmOutcome

MESSAGES = [{"role": "user", "content": "Who is the narrator?"}]
SECRET = "sk-secret-123"
BASE_URL = "http://127.0.0.1:11434/v1"


def sse(*contents: str, done: bool = True) -> bytes:
    events = [f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n" for text in contents]
    if done:
        events.append("data: [DONE]\n\n")
    return "".join(events).encode()


class ScriptedStream(httpx.AsyncByteStream):
    """Response body that plays steps in order: bytes are sent, exceptions raised, coroutine functions awaited."""

    def __init__(self, *steps: bytes | BaseException | Callable[[], Awaitable[None]]) -> None:
        self.steps = steps
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for step in self.steps:
            if isinstance(step, BaseException):
                raise step
            if callable(step):
                await step()
            else:
                yield step

    async def aclose(self) -> None:
        self.closed = True


def sse_response(body: bytes) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def make_client(handler: Callable[[httpx.Request], httpx.Response], api_key: str = "local") -> LLMClient:
    return LLMClient(base_url=BASE_URL, api_key=api_key, provider="ollama", transport=httpx.MockTransport(handler))


async def collect(client: LLMClient) -> list[str]:
    return [token async for token in client.stream_generate(MESSAGES)]


def warnings_from(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING and r.name == "voice_agent.llm.client"]


async def test_ok_streams_tokens() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return sse_response(sse("Hi", " there"))

    client = make_client(handler)

    assert await collect(client) == ["Hi", " there"]
    assert client.status == "ok"
    body = json.loads(seen[0].content)
    assert body["max_tokens"] == 300
    assert body["stream"] is True
    assert body["messages"] == MESSAGES
    assert seen[0].url == "http://127.0.0.1:11434/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer local"


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://h:1/v1", "http://h:1/v1/chat/completions"),
        ("http://h:1/v1/", "http://h:1/v1/chat/completions"),
        ("http://h:11434", "http://h:11434/v1/chat/completions"),
        ("http://h:1/v1/chat/completions", "http://h:1/v1/chat/completions"),
    ],
)
async def test_base_url_is_normalised_to_the_chat_endpoint(base_url: str, expected: str) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return sse_response(sse("ok"))

    client = LLMClient(base_url=base_url, provider="ollama", transport=httpx.MockTransport(handler))
    await collect(client)

    assert seen == [expected]


async def test_connect_refused_is_offline(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = make_client(handler)
    with caplog.at_level(logging.WARNING):
        assert await collect(client) == [OFFLINE_LINE]

    assert client.status == "offline"
    (record,) = warnings_from(caplog)
    assert BASE_URL in record.getMessage()
    assert "offline" in record.getMessage()
    assert "ConnectError" in record.getMessage()


async def test_connect_timeout_is_offline() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("no route", request=request)

    client = make_client(handler)

    assert await collect(client) == [OFFLINE_LINE]
    assert client.status == "offline"


@pytest.mark.parametrize(("code", "expected"), [(401, "auth"), (403, "auth"), (404, "error"), (500, "error")])
async def test_http_status_maps_to_outcome(code: int, expected: LlmOutcome, caplog: pytest.LogCaptureFixture) -> None:
    client = make_client(lambda request: httpx.Response(code, text="nope"))

    with caplog.at_level(logging.WARNING):
        assert await collect(client) == [OFFLINE_LINE]

    assert client.status == expected
    (record,) = warnings_from(caplog)
    assert str(code) in record.getMessage()


async def test_empty_200_is_error_and_speaks_offline_line(caplog: pytest.LogCaptureFixture) -> None:
    client = make_client(lambda request: sse_response(sse()))

    with caplog.at_level(logging.WARNING):
        assert await collect(client) == [OFFLINE_LINE]

    assert client.status == "error"
    (record,) = warnings_from(caplog)
    assert "no content" in record.getMessage()


async def test_cut_mid_stream_is_interrupted(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        stream = ScriptedStream(sse("Part", done=False), httpx.ReadError("cut", request=request))
        return httpx.Response(200, stream=stream)

    client = make_client(handler)
    with caplog.at_level(logging.WARNING):
        tokens = await collect(client)

    assert tokens == ["Part"]  # the partial text only: no canned text is appended (N3)
    assert client.status == "interrupted"
    assert len(warnings_from(caplog)) == 1


async def test_read_timeout_before_any_token_is_error_not_offline() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ScriptedStream(httpx.ReadTimeout("cold model", request=request)))

    client = make_client(handler)

    assert await collect(client) == [OFFLINE_LINE]
    assert client.status == "error"


@pytest.mark.parametrize("base_url", ["localhost:11434", "", "ftp://host/v1", "http://[::1"])
async def test_unusable_base_url_is_error_and_never_raises(base_url: str) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return sse_response(sse("never"))

    client = LLMClient(base_url=base_url, provider="ollama", transport=httpx.MockTransport(handler))

    assert await collect(client) == [OFFLINE_LINE]
    assert client.status == "error"


async def test_non_ascii_api_key_is_error_and_never_raises(caplog: pytest.LogCaptureFixture) -> None:
    client = make_client(lambda request: sse_response(sse("never")), api_key="kéy")

    with caplog.at_level(logging.DEBUG):
        assert await collect(client) == [OFFLINE_LINE]

    assert client.status == "error"
    assert "kéy" not in caplog.text


async def _parked_stream_client() -> tuple[LLMClient, ScriptedStream, asyncio.Event]:
    """A client whose response sends one token, then blocks until the consumer is parked waiting for more."""
    parked = asyncio.Event()

    async def block_forever() -> None:
        parked.set()
        await asyncio.Event().wait()

    stream = ScriptedStream(sse("first", done=False), block_forever)
    return make_client(lambda request: httpx.Response(200, stream=stream)), stream, parked


async def test_cancel_closes_response() -> None:
    client, stream, parked = await _parked_stream_client()
    got: list[str] = []

    async def consume() -> None:
        async for token in client.stream_generate(MESSAGES):
            got.append(token)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(parked.wait(), timeout=5)  # the consumer has the first token and awaits the second
    assert got == ["first"]
    assert not stream.closed

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed
    assert client.status == "interrupted"


async def test_closing_the_generator_closes_response() -> None:
    client, stream, _ = await _parked_stream_client()
    generator = client.stream_generate(MESSAGES)

    assert await anext(generator) == "first"
    assert not stream.closed
    await generator.aclose()

    assert stream.closed


async def test_timeouts_configured() -> None:
    seen: list[dict[str, float | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.extensions["timeout"]))
        return sse_response(sse("ok"))

    await collect(make_client(handler))

    # A 30 s read timeout covers a cold model load; only an unreachable port is "offline".
    assert seen == [{"connect": 2.0, "read": 30.0, "write": 10.0, "pool": 2.0}]


async def test_slow_first_token_ok() -> None:
    async def pause() -> None:
        await asyncio.sleep(0.2)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ScriptedStream(pause, sse("late")))

    client = make_client(handler)

    assert await collect(client) == ["late"]
    assert client.status == "ok"


async def test_key_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    handlers: list[Callable[[httpx.Request], httpx.Response]] = [
        lambda request: sse_response(sse("fine")),
        refuse,
        lambda request: httpx.Response(401, text="bad key"),
        lambda request: httpx.Response(500, text="boom"),
    ]
    with caplog.at_level(logging.DEBUG):
        for handler in handlers:
            await collect(make_client(handler, api_key=SECRET))

    assert SECRET not in caplog.text
    assert len(warnings_from(caplog)) == 3  # offline, auth, error: the checked paths really logged


async def test_key_is_sent_as_bearer_token() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return sse_response(sse("ok"))

    await collect(make_client(handler, api_key=SECRET))

    assert seen == [f"Bearer {SECRET}"]


@pytest.mark.parametrize("provider", ["builtin", "local-builtin", "native", "none", " Builtin "])
async def test_builtin_mode_is_offline_line(provider: str) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return sse_response(sse("never"))

    client = LLMClient(provider=provider, transport=httpx.MockTransport(handler))

    assert await collect(client) == [OFFLINE_LINE]
    assert client.status == "offline"
    assert calls == []


async def test_status_is_ok_before_any_call_and_recovers_after_failure() -> None:
    answers: list[Callable[[httpx.Request], httpx.Response]] = [
        lambda request: httpx.Response(500),
        lambda request: sse_response(sse("back")),
    ]
    client = make_client(lambda request: answers.pop(0)(request))
    assert client.status == "ok"

    await collect(client)
    assert client.status == "error"
    await collect(client)
    assert client.status == "ok"


@pytest.mark.parametrize(
    "event",
    [
        "data: not json",
        "data: [1, 2]",
        'data: {"choices": []}',
        'data: {"choices": [null]}',
        'data: {"choices": [{"delta": null}]}',
        'data: {"choices": [{"delta": {"content": null}}]}',
        'data: {"choices": [{"delta": {"content": 7}}]}',
        'data: {"choices": [{"delta": {"reasoning": "hidden"}}]}',
        ": keep-alive comment",
    ],
)
async def test_odd_events_are_skipped_not_raised(event: str) -> None:
    body = f"{event}\n\n".encode() + sse("real")
    client = make_client(lambda request: sse_response(body))

    assert await collect(client) == ["real"]
    assert client.status == "ok"
