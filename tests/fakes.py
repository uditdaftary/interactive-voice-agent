"""Deterministic stand-ins for the network- and hardware-backed collaborators.

Inject these through the `NarrationEngine` / `create_app` seams instead of
patching module globals. Each fake records its calls so tests can assert what
the engine actually asked for.
"""

import asyncio
import json
import threading
import time
from collections.abc import AsyncGenerator, Callable, Collection, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np

TTS_SAMPLE_RATE = 24000
SAMPLES_PER_WORD = 4800  # 0.2 s of audio per word


def wait_until(condition: Callable[[], bool], timeout_s: float = 5.0, interval_s: float = 0.01) -> bool:
    """Poll from a synchronous test while the app runs on TestClient's event-loop thread."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval_s)
    return condition()


class FakeTts:
    """Records `(text, voice_id, cadence)` and returns 24 kHz int16 silence sized by word count."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []
        self._called = asyncio.Event()

    async def __call__(self, text: str, voice_id: str, cadence: float) -> tuple[bytes, str]:
        self.calls.append((text, voice_id, cadence))
        self._called.set()
        silence = np.zeros(len(text.split()) * SAMPLES_PER_WORD, dtype=np.int16)
        return silence.tobytes(), "pcm_s16le"

    async def wait_for_calls(self, count: int) -> None:
        """Block until at least `count` calls were made. Wrap in `asyncio.timeout` in tests."""
        while len(self.calls) < count:
            self._called.clear()
            await self._called.wait()

    @property
    def texts(self) -> list[str]:
        return [text for text, _, _ in self.calls]


class FailingTts(FakeTts):
    """A synthesizer that raises for the texts in `fail_on` (for every text when `fail_on` is None).

    `attempts` lists every text it was asked to speak; `calls` and `texts` list only the ones that were synthesized.
    """

    def __init__(self, fail_on: Collection[str] | None = None) -> None:
        super().__init__()
        self.fail_on = None if fail_on is None else frozenset(fail_on)
        self.attempts: list[str] = []

    async def __call__(self, text: str, voice_id: str, cadence: float) -> tuple[bytes, str]:
        self.attempts.append(text)
        if self.fail_on is None or text in self.fail_on:
            raise RuntimeError("synthesis exploded")
        return await super().__call__(text, voice_id, cadence)


class FakeSearch:
    """Records `(query, max_results)` and returns a fixed web-grounding string."""

    def __init__(self, result: str = "- Lucy Kushinada: Netrunner and former Night City street kid.") -> None:
        self.result = result
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, query: str, max_results: int) -> str:
        self.calls.append((query, max_results))
        return self.result


class FakeLLM:
    """Streams preset tokens. Mirrors the slice of `LLMClient` that the server uses."""

    def __init__(self, tokens: Sequence[str] = ("Understood. ",), web_search_enabled: bool = False) -> None:
        self.tokens = list(tokens)
        self.web_search_enabled = web_search_enabled
        self.calls: list[list[dict[str, str]]] = []
        self.config: dict[str, Any] = {"provider": "fake", "web_search_enabled": web_search_enabled}

    async def stream_generate(
        self,
        messages: list[dict[str, str]],
        character_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        self.calls.append(messages)
        for token in self.tokens:
            yield token

    def get_config(self) -> dict[str, Any]:
        return dict(self.config)

    def update_config(self, **kwargs: Any) -> None:
        self.config.update({key: value for key, value in kwargs.items() if value is not None})
        if kwargs.get("web_search_enabled") is not None:
            self.web_search_enabled = bool(kwargs["web_search_enabled"])


class BlockingLLM(FakeLLM):
    """A stream that never produces a token until released."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_generate(  # type: ignore[override]  # the base yields tokens; this one waits first
        self,
        messages: list[dict[str, str]],
        character_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        self.calls.append(messages)
        self.started.set()
        await self.release.wait()
        yield "never"


class StreamingLLM(FakeLLM):
    """Streams its tokens, or raises `error` when asked for the first one, and records whether the stream was closed.

    An async generator is finalized promptly only when its consumer closes it; `closed` is how a test sees that happen.
    """

    def __init__(self, tokens: Sequence[str] = ("Hello ", "there. "), *, error: Exception | None = None) -> None:
        super().__init__(tokens)
        self.error = error
        self.closed = False

    async def stream_generate(
        self,
        messages: list[dict[str, str]],
        character_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        self.calls.append(messages)
        try:
            if self.error is not None:
                raise self.error
            for token in self.tokens:
                yield token
        finally:
            self.closed = True


class FakeWhisper:
    """Stands in for `faster_whisper.WhisperModel`: patch the class with an instance, and calling it builds the model.

    Every `transcribe` returns `text` as one segment (nothing when `text` is ""). The first `fail_loads` constructions raise,
    like a model missing from the Hugging Face cache. The fake is its own model.
    """

    def __init__(self, text: str = "", fail_loads: int = 0) -> None:
        self.text = text
        self.fail_loads = fail_loads
        self.constructions: list[dict[str, Any]] = []
        self.transcribe_kwargs: list[dict[str, Any]] = []
        self.inputs: list[np.ndarray] = []

    def __call__(self, model_size: str, **kwargs: Any) -> "FakeWhisper":
        self.constructions.append({"model_size": model_size, **kwargs})
        if len(self.constructions) <= self.fail_loads:
            raise OSError("model not in the Hugging Face cache")
        return self

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[list[SimpleNamespace], None]:
        self.inputs.append(audio)
        self.transcribe_kwargs.append(kwargs)
        return ([SimpleNamespace(text=self.text)] if self.text else []), None


class FakeStt:
    """Stands in for the STT engine: returns a fixed transcript and records the byte length of each input."""

    def __init__(self, fixed_transcript: str = "") -> None:
        self.fixed_transcript = fixed_transcript
        self.calls: list[int] = []

    def transcribe(self, audio_data: bytes | bytearray | np.ndarray, sample_rate: int = 16000) -> str:
        self.calls.append(audio_data.nbytes if isinstance(audio_data, np.ndarray) else len(audio_data))
        return self.fixed_transcript


class GatedStt:
    """An STT engine whose `transcribe` blocks until the test releases it, then returns `transcript`.

    `started` is set once a call is inside `transcribe`; `inputs` holds the audio of every call.
    """

    def __init__(self, transcript: str = "") -> None:
        self.transcript = transcript
        self.started = threading.Event()
        self.release = threading.Event()
        self.inputs: list[bytes] = []

    def transcribe(self, audio: bytes) -> str:
        self.inputs.append(bytes(audio))
        self.started.set()
        assert self.release.wait(5), "the test never released the recognizer"
        return self.transcript

    async def wait_started(self) -> None:
        """Return once a call is inside `transcribe`, without blocking the event loop."""
        async with asyncio.timeout(5):
            await asyncio.to_thread(self.started.wait, 5)


class RecordingSocket:
    """Stands in for a WebSocket: records every text frame the manager sends it, in arrival order."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self._arrived = asyncio.Event()

    async def accept(self) -> None:
        return None

    async def send_text(self, text: str) -> None:
        self.texts.append(text)
        self._arrived.set()

    async def send_bytes(self, data: bytes) -> None:
        return None

    async def close(self, code: int = 1000) -> None:
        return None

    def messages(self, message_type: str) -> list[dict[str, Any]]:
        parsed = [json.loads(text) for text in self.texts]
        return [message for message in parsed if message["type"] == message_type]

    def states(self) -> list[str]:
        return [message["state"] for message in self.messages("state_sync")]

    def seen(self) -> list[tuple[str, str]]:
        """(type, state) of each message received so far; state is "" for messages that have none."""
        parsed = [json.loads(text) for text in self.texts]
        return [(message["type"], message.get("state", "")) for message in parsed]

    async def wait_until(self, condition: Callable[[], bool], timeout_s: float = 5.0) -> None:
        """Return as soon as `condition()` holds, re-checking after every frame that arrives."""
        async with asyncio.timeout(timeout_s):
            while not condition():
                self._arrived.clear()
                await self._arrived.wait()
