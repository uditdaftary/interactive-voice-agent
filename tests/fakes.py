"""Deterministic stand-ins for the network- and hardware-backed collaborators.

Inject these through the `NarrationEngine` / `create_app` seams instead of
patching module globals. Each fake records its calls so tests can assert what
the engine actually asked for.
"""

import asyncio
import json
import threading
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Collection, Sequence
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np

from voice_agent.stt.backends import Transcript
from voice_agent.tts.backends import SpeechAudio

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
    """A synthesizer that records `(text, voice_id, cadence)` and makes 24 kHz int16 silence sized by word count.

    The engine calls `synthesize`, which turns what `__call__` returns (the bytes, or None for no audio, and a format
    name) into the chain's result. A test scripts a synthesizer by overriding `__call__`. `status` is what a real
    chain reports; a test sets it to say the chain fell back.
    """

    def __init__(self, status: str = "ok") -> None:
        self.status = status
        self.calls: list[tuple[str, str, float]] = []
        self._called = asyncio.Event()

    async def __call__(self, text: str, voice_id: str, cadence: float) -> tuple[bytes | None, str]:
        self.calls.append((text, voice_id, cadence))
        self._called.set()
        silence = np.zeros(len(text.split()) * SAMPLES_PER_WORD, dtype=np.int16)
        return silence.tobytes(), "audio/pcm"

    async def synthesize(self, text: str, voice_id: str, cadence: float) -> SpeechAudio | None:
        data, mime = await self(text, voice_id, cadence)
        return SpeechAudio(data, mime) if data else None

    async def wait_for_calls(self, count: int) -> None:
        """Block until at least `count` calls were made. Wrap in `asyncio.timeout` in tests."""
        while len(self.calls) < count:
            self._called.clear()
            await self._called.wait()

    @property
    def texts(self) -> list[str]:
        return [text for text, _, _ in self.calls]


async def wait_for_state(session: Any, state: str, timeout_s: float = 5.0) -> None:
    """Return as soon as `session` is in `state`, woken by its own listener rather than by polling."""
    if session.state == state:
        return
    reached = asyncio.Event()
    session.add_listener(lambda old, new, reason: reached.set() if new == state else None)
    async with asyncio.timeout(timeout_s):
        await reached.wait()


class FailingTts(FakeTts):
    """A synthesizer that raises for the texts in `fail_on` (for every text when `fail_on` is None).

    `attempts` lists every text it was asked to speak; `calls` and `texts` list only the ones that were synthesized.
    """

    def __init__(self, fail_on: Collection[str] | None = None) -> None:
        super().__init__()
        self.fail_on = None if fail_on is None else frozenset(fail_on)
        self.attempts: list[str] = []

    async def __call__(self, text: str, voice_id: str, cadence: float) -> tuple[bytes | None, str]:
        self.attempts.append(text)
        if self.fail_on is None or text in self.fail_on:
            raise RuntimeError("synthesis exploded")
        return await super().__call__(text, voice_id, cadence)


class FunctionTts(FakeTts):
    """A synthesizer scripted by one async function `(text, voice_id, cadence) -> (bytes or None, format)`."""

    def __init__(self, speak: Callable[[str, str, float], Awaitable[tuple[bytes | None, str]]]) -> None:
        super().__init__()
        self._speak = speak

    async def __call__(self, text: str, voice_id: str, cadence: float) -> tuple[bytes | None, str]:
        return await self._speak(text, voice_id, cadence)


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


def _transcript(text: str, rejected_reason: str | None) -> Transcript:
    """Words come with no reason; no words always come with one, as from a real backend."""
    return Transcript(text, None if text else (rejected_reason or "not-speech"))


class FakeStt:
    """Stands in for an `SttBackend`: returns a fixed transcript and records the byte length of each input."""

    def __init__(self, fixed_transcript: str = "", rejected_reason: str | None = None) -> None:
        self.fixed_transcript = fixed_transcript
        self.rejected_reason = rejected_reason
        self.calls: list[int] = []

    def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> Transcript:
        self.calls.append(len(pcm16))
        return _transcript(self.fixed_transcript, self.rejected_reason)


class GatedStt:
    """An STT backend whose `transcribe` blocks until the test releases it, then returns `transcript`.

    `started` is set once a call is inside `transcribe`; `inputs` holds the audio of every call.
    """

    def __init__(self, transcript: str = "", rejected_reason: str | None = None) -> None:
        self.transcript = transcript
        self.rejected_reason = rejected_reason
        self.started = threading.Event()
        self.release = threading.Event()
        self.inputs: list[bytes] = []

    def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> Transcript:
        self.inputs.append(bytes(pcm16))
        self.started.set()
        assert self.release.wait(5), "the test never released the recognizer"
        return _transcript(self.transcript, self.rejected_reason)

    async def wait_started(self) -> None:
        """Return once a call is inside `transcribe`, without blocking the event loop."""
        async with asyncio.timeout(5):
            await asyncio.to_thread(self.started.wait, 5)


class RecordingSocket:
    """Stands in for a WebSocket: records every text frame the manager sends it, in arrival order.

    `frames` is the type of every message and "blob" for every binary frame, in the order they arrived together.
    """

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.blobs: list[bytes] = []
        self.frames: list[str] = []
        self._arrived = asyncio.Event()

    async def accept(self) -> None:
        return None

    async def send_text(self, text: str) -> None:
        self.texts.append(text)
        self.frames.append(json.loads(text)["type"])
        self._arrived.set()

    async def send_bytes(self, data: bytes) -> None:
        self.blobs.append(data)
        self.frames.append("blob")
        self._arrived.set()

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


class PlayedClip(NamedTuple):
    segment_id: int
    generation: int
    gap_s: float
    data: bytes


class PlayerSocket(RecordingSocket):
    """A tab that plays audio: `clips` lists every clip it was sent, with the announcement that came before it.

    It reports a clip finished the way the browser does, with a `playback_done` message that goes through the real
    handler: at once for every clip when `auto_ack` is set, otherwise when the test calls `ack`.
    """

    def __init__(self, manager: Any = None, *, auto_ack: bool = False) -> None:
        super().__init__()
        self.manager = manager
        self.auto_ack = auto_ack
        self.clips: list[PlayedClip] = []
        self.unacked: list[PlayedClip] = []
        self._announced: dict[str, Any] | None = None
        self._acks: set[asyncio.Task[None]] = set()

    async def send_text(self, text: str) -> None:
        await super().send_text(text)
        message = json.loads(text)
        if message["type"] == "audio_segment":
            self._announced = message

    async def send_bytes(self, data: bytes) -> None:
        await super().send_bytes(data)
        assert self._announced is not None, "a clip arrived without the audio_segment that announces it"
        clip = PlayedClip(self._announced["segment_id"], self._announced["generation"], self._announced["gap_s"], data)
        self._announced = None
        self.clips.append(clip)
        self.unacked.append(clip)
        if self.auto_ack:
            task = asyncio.get_running_loop().create_task(self.ack(clip))
            self._acks.add(task)
            task.add_done_callback(self._acks.discard)

    async def ack(self, clip: PlayedClip | None = None, *, segment_id: int | None = None, generation: int | None = None) -> None:
        """Report a clip played: the oldest unacknowledged one, or `clip`. `segment_id` and `generation` override what
        is reported, to send a report that is wrong."""
        from voice_agent.server.websocket import handle_client_text_message

        if clip is None:
            clip = self.unacked[0]
        if clip in self.unacked:
            self.unacked.remove(clip)
        report = {
            "type": "playback_done",
            "segment_id": clip.segment_id if segment_id is None else segment_id,
            "generation": clip.generation if generation is None else generation,
        }
        await handle_client_text_message(json.dumps(report), self, self.manager)

    async def wait_for_clips(self, count: int, timeout_s: float = 5.0) -> None:
        await self.wait_until(lambda: len(self.clips) >= count, timeout_s)

