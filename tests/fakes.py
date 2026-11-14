"""Deterministic stand-ins for the network- and hardware-backed collaborators.

Inject these through the `NarrationEngine` / `create_app` seams instead of
patching module globals. Each fake records its calls so tests can assert what
the engine actually asked for.
"""

import asyncio
import time
from collections.abc import AsyncGenerator, Callable, Sequence
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
