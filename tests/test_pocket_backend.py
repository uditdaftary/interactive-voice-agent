"""The optional Pocket TTS backend, against a fake `pocket_tts` module that mirrors the public API of pocket-tts 4.0.1.

Nothing here loads a model or touches Hugging Face: the real weights are gated, and the default suite must pass without the extra.
"""

import asyncio
import logging
import sys
import threading
import time
import types
import wave
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voice_agent.settings import load_settings
from voice_agent.tts.backends import ChainedTts, PocketTtsBackend, SpeechAudio, TtsError, build_tts

POCKET_SAMPLE_RATE = 24000


class FakePocket:
    """Counts what the backend asks of pocket-tts, and shapes the next load and generation."""

    def __init__(self) -> None:
        self.loads = 0
        self.loads_on_main_thread: list[bool] = []
        self.fail_load = False
        self.has_voice_cloning = True
        self.prompts: list[str] = []
        self.prompts_were_paths = True
        self.chunk_count = 3
        self.chunk_delay_s = 0.0
        self.chunks_made = 0
        self.active_generations = 0
        self.max_active_generations = 0
        self.generations_finished = 0

    def module(self) -> types.ModuleType:
        fake = self

        class TTSModel:
            sample_rate = POCKET_SAMPLE_RATE

            def __init__(self, cloning: bool) -> None:
                self.has_voice_cloning = cloning

            @classmethod
            def load_model(cls) -> "TTSModel":
                fake.loads += 1
                fake.loads_on_main_thread.append(threading.current_thread() is threading.main_thread())
                if fake.fail_load:
                    raise OSError("repository is gated")
                return cls(fake.has_voice_cloning)

            def get_state_for_audio_prompt(self, audio_conditioning: Any, truncate: bool = False) -> dict[str, str]:
                fake.prompts_were_paths = fake.prompts_were_paths and isinstance(audio_conditioning, Path)
                fake.prompts.append(Path(audio_conditioning).name)
                return {"voice": Path(audio_conditioning).name}

            def generate_audio_stream(
                self,
                model_state: dict[str, str],
                text_to_generate: str,
                max_tokens: int = 50,
                frames_after_eos: int | None = None,
                copy_state: bool = True,
                stop: threading.Event | None = None,
            ) -> Any:
                fake.active_generations += 1
                fake.max_active_generations = max(fake.max_active_generations, fake.active_generations)
                for _ in range(fake.chunk_count):
                    if stop is not None and stop.is_set():
                        break
                    time.sleep(fake.chunk_delay_s)
                    fake.chunks_made += 1
                    yield np.full(2400, 0.25, dtype=np.float32)
                # Modelled on pocket-tts: a generation is over only once its stream runs out. An early close leaves its
                # threads running, so the count stays up.
                fake.active_generations -= 1
                fake.generations_finished += 1

        return _with_model(TTSModel)


def _with_model(model_class: type) -> types.ModuleType:
    module = types.ModuleType("pocket_tts")
    module.TTSModel = model_class  # type: ignore[attr-defined]
    return module


@pytest.fixture
def fake_pocket(monkeypatch: pytest.MonkeyPatch) -> FakePocket:
    fake = FakePocket()
    monkeypatch.setitem(sys.modules, "pocket_tts", fake.module())
    return fake


@pytest.fixture
def voices(tmp_path: Path) -> Path:
    """A voices folder with one voice file per id the tests use; the backend only reads them through pocket-tts."""
    for name in ("lucy", "v-male"):
        (tmp_path / f"{name}.wav").write_bytes(b"not read by the fake")
    return tmp_path


class FakeBackend:
    name = "edge"

    async def synthesize(self, text: str, voice_id: str, cadence: float) -> SpeechAudio:
        return SpeechAudio(b"x" * 200, "audio/mpeg")


async def wait_until(condition: Callable[[], bool], timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


async def test_model_loaded_once_off_the_loop(fake_pocket: FakePocket, voices: Path) -> None:
    backend = PocketTtsBackend(voices_dir=voices)

    await asyncio.gather(backend.synthesize("first", "lucy", 1.0), backend.synthesize("second", "lucy", 1.0))

    assert fake_pocket.loads == 1
    assert fake_pocket.loads_on_main_thread == [False]


async def test_output_is_pocket_wav_at_model_rate(fake_pocket: FakePocket, voices: Path) -> None:
    audio = await PocketTtsBackend(voices_dir=voices).synthesize("hello there", "lucy", 1.0)

    assert audio.mime == "audio/wav"
    assert audio.word_times is None
    with wave.open(BytesIO(audio.data), "rb") as wav:
        assert wav.getframerate() == POCKET_SAMPLE_RATE
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 3 * 2400


async def test_voice_state_cached_per_voice(fake_pocket: FakePocket, voices: Path) -> None:
    backend = PocketTtsBackend(voices_dir=voices)

    await backend.synthesize("one", "lucy", 1.0)
    await backend.synthesize("two", "lucy", 1.0)

    assert fake_pocket.prompts == ["lucy.wav"]
    assert fake_pocket.prompts_were_paths


async def test_missing_voice_falls_back_with_one_warning(
    fake_pocket: FakePocket, voices: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="voice_agent.tts.backends")
    backend = PocketTtsBackend(voices_dir=voices)

    await backend.synthesize("one", "nobody", 1.0)
    await backend.synthesize("two", "nobody", 1.0)

    assert fake_pocket.prompts == ["v-male.wav"]
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING and "nobody" in record.getMessage()]
    assert len(warnings) == 1


@pytest.mark.parametrize("unsafe", ["../lucy", "a/b", ""])
async def test_voice_id_outside_the_folder_is_not_read(fake_pocket: FakePocket, voices: Path, unsafe: str) -> None:
    await PocketTtsBackend(voices_dir=voices).synthesize("hi", unsafe, 1.0)

    assert fake_pocket.prompts == ["v-male.wav"]


async def test_missing_default_voice_is_tts_error(fake_pocket: FakePocket, tmp_path: Path) -> None:
    with pytest.raises(TtsError, match="v-male.wav"):
        await PocketTtsBackend(voices_dir=tmp_path).synthesize("hi", "nobody", 1.0)


async def test_cancel_stops_between_chunks(fake_pocket: FakePocket, voices: Path) -> None:
    fake_pocket.chunk_count = 10
    fake_pocket.chunk_delay_s = 0.05
    task = asyncio.create_task(PocketTtsBackend(voices_dir=voices).synthesize("a long sentence", "lucy", 1.0))
    await wait_until(lambda: fake_pocket.chunks_made >= 1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The generation has ended only when its stream ran out, so waiting on that is exact; no fixed sleep needed.
    await wait_until(lambda: fake_pocket.generations_finished == 1)

    assert fake_pocket.chunks_made < 3


async def test_next_generation_waits_for_the_stopped_one(fake_pocket: FakePocket, voices: Path) -> None:
    fake_pocket.chunk_count = 10
    fake_pocket.chunk_delay_s = 0.05
    backend = PocketTtsBackend(voices_dir=voices)
    task = asyncio.create_task(backend.synthesize("a long sentence", "lucy", 1.0))
    await wait_until(lambda: fake_pocket.chunks_made >= 1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    fake_pocket.chunk_count = 2
    fake_pocket.chunk_delay_s = 0.0
    audio = await backend.synthesize("the next sentence", "lucy", 1.0)

    assert audio.mime == "audio/wav"
    # Two generations never run on the model at once: the stopped one has been drained before the next starts.
    assert fake_pocket.max_active_generations == 1


async def test_gated_weights_raise_the_licence_error_once(fake_pocket: FakePocket, voices: Path) -> None:
    fake_pocket.has_voice_cloning = False
    backend = PocketTtsBackend(voices_dir=voices)

    with pytest.raises(TtsError, match="Hugging Face licence"):
        await backend.synthesize("hi", "lucy", 1.0)
    with pytest.raises(TtsError, match="Hugging Face licence"):
        await backend.synthesize("hi", "lucy", 1.0)

    assert fake_pocket.loads == 1


async def test_load_exception_is_tts_error(fake_pocket: FakePocket, voices: Path) -> None:
    fake_pocket.fail_load = True

    with pytest.raises(TtsError, match="Hugging Face licence"):
        await PocketTtsBackend(voices_dir=voices).synthesize("hi", "lucy", 1.0)


async def test_import_failure_is_tts_error(monkeypatch: pytest.MonkeyPatch, voices: Path) -> None:
    monkeypatch.setitem(sys.modules, "pocket_tts", None)

    with pytest.raises(TtsError, match="pocket-tts did not import"):
        await PocketTtsBackend(voices_dir=voices).synthesize("hi", "lucy", 1.0)


async def test_chain_falls_back_from_pocket(monkeypatch: pytest.MonkeyPatch, voices: Path) -> None:
    monkeypatch.setitem(sys.modules, "pocket_tts", None)
    chain = ChainedTts([PocketTtsBackend(voices_dir=voices), FakeBackend()])

    audio = await chain.synthesize("hello", "lucy", 1.0)

    assert audio is not None
    assert chain.status == "fallback:edge"


def test_build_tts_pocket_primary_order() -> None:
    assert [backend.name for backend in build_tts("pocket").backends] == ["pocket", "edge", "pyttsx3"]


def test_pocket_is_not_in_the_default_chain() -> None:
    assert [backend.name for backend in build_tts().backends] == ["edge", "pyttsx3"]


def test_primary_pocket_accepted_by_settings() -> None:
    assert load_settings({"VOICE_AGENT_TTS_PRIMARY": "Pocket"}).tts_primary == "pocket"
