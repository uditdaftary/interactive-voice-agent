"""LLMClient transport: httpx streaming, typed outcomes, think filter, key hygiene.

Every test drives `httpx.MockTransport`, so nothing leaves the process. MockTransport enforces no timeouts
and bypasses httpcore, which is why timeouts are asserted on the request extensions instead of by waiting.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

import httpx
import pytest

from voice_agent.llm.client import OFFLINE_LINE, LLMClient, LlmOutcome, ThinkFilter

MESSAGES = [{"role": "user", "content": "Who is the narrator?"}]
SECRET = "sk-secret-123"
BASE_URL = "http://127.0.0.1:11434/v1"
FIRST = "f" * ThinkFilter.LEAD_HOLD_CHARS  # long enough to pass the filter's hold at the start of a stream


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

    # A reply shorter than ThinkFilter.LEAD_HOLD_CHARS is held until the stream ends (see ThinkFilter).
    assert "".join(await collect(client)) == "Hi there"
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

    stream = ScriptedStream(sse(FIRST, done=False), block_forever)
    return make_client(lambda request: httpx.Response(200, stream=stream)), stream, parked


async def test_cancel_closes_response() -> None:
    client, stream, parked = await _parked_stream_client()
    got: list[str] = []

    async def consume() -> None:
        async for token in client.stream_generate(MESSAGES):
            got.append(token)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(parked.wait(), timeout=5)  # the consumer has the first token and awaits the second
    assert got == [FIRST]
    assert not stream.closed

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed
    assert client.status == "interrupted"


async def test_closing_the_generator_closes_response() -> None:
    client, stream, _ = await _parked_stream_client()
    generator = client.stream_generate(MESSAGES)

    assert await anext(generator) == FIRST
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


# --- think blocks inside the stream ---


async def test_think_block_never_reaches_the_consumer() -> None:
    pieces = ["<thi", "nk>weigh the ", "options</th", "ink>Answer", " here."]
    client = make_client(lambda request: sse_response(sse(*pieces)))

    assert "".join(await collect(client)) == "Answer here."
    assert client.status == "ok"


async def test_stream_that_starts_inside_a_think_block_is_cleaned() -> None:
    # A chat template that prefills "<think>" makes the model stream "reasoning</think>Answer" with no opening tag.
    pieces = ["I should greet. ", "</thi", "nk>\n\n", "Hello there."]
    client = make_client(lambda request: sse_response(sse(*pieces)))

    assert "".join(await collect(client)) == "Hello there."
    assert client.status == "ok"


async def test_prefilled_think_with_no_answer_is_error() -> None:
    client = make_client(lambda request: sse_response(sse("I should greet.", " </think>", "\n\n")))

    assert await collect(client) == [OFFLINE_LINE]
    assert client.status == "error"


async def test_partial_reply_held_by_the_filter_is_spoken_when_the_stream_breaks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ScriptedStream(sse("Part", done=False), httpx.ReadError("cut", request=request)))

    client = make_client(handler)

    assert await collect(client) == ["Part"]
    assert client.status == "interrupted"


async def test_think_only_reply_is_error() -> None:
    client = make_client(lambda request: sse_response(sse("<think>", "I should say hello", "</think>")))

    assert await collect(client) == [OFFLINE_LINE]
    assert client.status == "error"


async def test_unclosed_think_at_end_of_stream_is_dropped() -> None:
    client = make_client(lambda request: sse_response(sse("Ready. ", "<think>trailing thoughts")))

    assert "".join(await collect(client)) == "Ready. "
    assert client.status == "ok"


async def test_held_partial_tag_is_released_at_end_of_stream() -> None:
    client = make_client(lambda request: sse_response(sse("a <")))

    assert "".join(await collect(client)) == "a <"


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


# --- ThinkFilter ---


def run_filter(chunks: Sequence[str]) -> str:
    think_filter = ThinkFilter()
    return "".join(think_filter.feed(chunk) for chunk in chunks) + think_filter.flush()


def test_think_split_across_chunks() -> None:
    assert run_filter(["<thi", "nk>reasoning</th", "ink>Answer"]) == "Answer"


def test_no_think_passthrough_is_delayed_by_at_most_the_lead_hold() -> None:
    think_filter = ThinkFilter()
    plain = "Right, the narrator is never named in these passages. " * 8
    fed = emitted = 0
    pieces: list[str] = []
    for start in range(0, len(plain), 7):
        chunk = plain[start : start + 7]
        pieces.append(think_filter.feed(chunk))
        fed += len(chunk)
        emitted += len(pieces[-1])
        assert fed - emitted <= ThinkFilter.LEAD_HOLD_CHARS  # the documented ceiling on added latency
    pieces.append(think_filter.flush())

    assert "".join(pieces) == plain  # a plain reply is never altered, only briefly held at the start
    assert sum(1 for piece in pieces if piece) > 3  # and it still streams in pieces once past the hold


def test_short_plain_reply_is_released_at_the_end_of_the_stream() -> None:
    think_filter = ThinkFilter()
    assert think_filter.feed("  Hello there.") == ""
    assert think_filter.flush() == "  Hello there."


def test_partial_tag_is_held_only_while_it_could_still_be_a_tag() -> None:
    think_filter = ThinkFilter()
    assert think_filter.feed("x" * ThinkFilter.LEAD_HOLD_CHARS) == "x" * ThinkFilter.LEAD_HOLD_CHARS  # past the hold
    assert think_filter.feed("see <th") == "see "
    assert think_filter.feed("ings") == "<things"  # not a tag after all: released with the new text
    assert think_filter.flush() == ""


# --- a stream that starts inside a think block (the chat template put "<think>" in the prompt) ---


REPRO = "I should greet. </think>\n\nHello there."


def test_prefilled_think_is_dropped_up_to_the_lone_close_tag() -> None:
    assert run_filter([REPRO]) == "Hello there."


@pytest.mark.parametrize(
    "chunks",
    [
        list(REPRO),
        ["I should greet. </thi", "nk>\n", "\nHello there."],
        ["I should greet. </think", ">", "\n\nHello there."],
        ["I should greet. ", "</think>", "\n\n", "Hello", " there."],
    ],
)
def test_prefilled_think_split_across_chunks(chunks: list[str]) -> None:
    assert run_filter(chunks) == "Hello there."


def test_whitespace_after_a_leading_think_block_is_trimmed_but_later_whitespace_is_kept() -> None:
    assert run_filter(["<think>a</think>\n\nAnswer one. ", "<think>b</think> Two."]) == "Answer one.  Two."


def test_lone_close_tag_with_nothing_before_it() -> None:
    assert run_filter(["</think>Answer"]) == "Answer"


def test_prefilled_think_that_never_closes_is_released_at_the_end() -> None:
    # Indistinguishable from a short plain reply, so it is released rather than lost.
    assert run_filter(["Okay, the user greets me."]) == "Okay, the user greets me."


def test_reasoning_longer_than_the_lead_hold_is_not_caught() -> None:
    """The documented ceiling: past LEAD_HOLD_CHARS the stream is taken for a plain reply, so a later lone
    close tag is ordinary text. Only thinking disabled at the source fixes this; the filter cannot."""
    reasoning = "Okay, so the user wants to know who the narrator is. " * 4
    assert len(reasoning) > ThinkFilter.LEAD_HOLD_CHARS

    output = run_filter([reasoning, "</think>Answer"])

    assert output == reasoning + "</think>Answer"


def test_unclosed_think_dropped() -> None:
    assert run_filter(["Hi. <think>never closed", " and more"]) == "Hi. "


def test_lone_lt_passes() -> None:
    assert run_filter(["a ", "<", " b"]) == "a < b"
    assert run_filter(["a < b"]) == "a < b"


def test_multiple_think_blocks_and_text_between() -> None:
    assert run_filter(["<think>a</think>One<think>b</think> two"]) == "One two"


def test_filter_is_reusable_after_flush() -> None:
    think_filter = ThinkFilter()
    think_filter.feed("<think>half")
    assert think_filter.flush() == ""
    assert think_filter.feed("fresh") == ""  # held again: a reused filter starts a new stream
    assert think_filter.flush() == "fresh"


def test_close_tag_split_one_character_at_a_time() -> None:
    assert run_filter(list("<think>x</think>ok")) == "ok"


# --- an API key must not cross the network in clear text (M13) ---


def key_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in warnings_from(caplog) if "http" in r.getMessage() and "key" in r.getMessage()]


def test_http_key_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    client = LLMClient(provider="ollama")
    with caplog.at_level(logging.WARNING):
        client.update_config(base_url="http://models.example.com/v1", api_key=SECRET)
        client.update_config(base_url="http://models.example.com/v1", api_key=SECRET)
        client.update_config(model="another-model")

    (message,) = key_warnings(caplog)
    assert "models.example.com" in message
    assert SECRET not in caplog.text


def test_http_key_warns_per_url(caplog: pytest.LogCaptureFixture) -> None:
    client = LLMClient(provider="ollama", api_key=SECRET)
    with caplog.at_level(logging.WARNING):
        client.update_config(base_url="http://one.example.com/v1")
        client.update_config(base_url="http://two.example.com/v1")

    assert len(key_warnings(caplog)) == 2


def test_http_key_warns_from_the_constructor(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        LLMClient(base_url="http://models.example.com:8080/v1", api_key=SECRET, provider="openai")

    (message,) = key_warnings(caplog)
    assert "models.example.com" in message
    assert SECRET not in caplog.text


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.example.com/v1",
        "http://127.0.0.1:11434/v1",
        "http://localhost:11434/v1",
        "http://LOCALHOST:11434/v1",
        "http://[::1]:11434/v1",
        "http://[::1",  # unparseable: reported as an error outcome when used, not as a key warning
        "",
    ],
)
def test_no_key_warning_for_https_loopback_or_unparseable_urls(base_url: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        LLMClient(base_url=base_url, api_key=SECRET, provider="ollama")

    assert key_warnings(caplog) == []


@pytest.mark.parametrize("api_key", ["", "local"])
def test_no_key_warning_without_a_real_key(api_key: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        LLMClient(base_url="http://models.example.com/v1", api_key=api_key, provider="ollama")

    assert key_warnings(caplog) == []


def test_no_key_warning_in_no_model_mode(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        LLMClient(base_url="http://models.example.com/v1", api_key=SECRET, provider="builtin")

    assert key_warnings(caplog) == []
