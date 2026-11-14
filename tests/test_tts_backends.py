"""TTS backends: the fallback chain, the edge-tts backend, and the synthesize_sentence adapter. No real edge-tts or SAPI call runs here."""

import logging
from typing import Any

import pytest

from voice_agent.settings import load_settings
from voice_agent.tts import backends
from voice_agent.tts.backends import ChainedTts, EdgeTtsBackend, SpeechAudio, TtsError, build_tts

AUDIO = b"x" * 200


class FakeBackend:
    """A TtsBackend that returns `audio`, or raises `error`, and records each request."""

    def __init__(self, name: str, audio: SpeechAudio | None = None, error: Exception | None = None) -> None:
        self.name = name
        self.audio = audio
        self.error = error
        self.calls: list[tuple[str, str, float]] = []

    async def synthesize(self, text: str, voice_id: str, cadence: float) -> SpeechAudio:
        self.calls.append((text, voice_id, cadence))
        if self.error is not None:
            raise self.error
        assert self.audio is not None
        return self.audio


async def test_chain_falls_back_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    failing = FakeBackend("edge", error=TtsError("offline"))
    working = FakeBackend("pyttsx3", audio=SpeechAudio(AUDIO, "audio/wav"))
    chain = ChainedTts([failing, working])
    caplog.set_level(logging.WARNING, logger="voice_agent.tts.backends")

    audio = await chain.synthesize("hello there", "david-martinez", 1.0)

    assert audio == SpeechAudio(AUDIO, "audio/wav")
    assert chain.status == "fallback:pyttsx3"
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "edge" in warnings[0].getMessage()
    assert "TtsError" in warnings[0].getMessage()


async def test_chain_all_fail_returns_none() -> None:
    chain = ChainedTts([FakeBackend("edge", error=TtsError("offline")), FakeBackend("pyttsx3", error=OSError("no SAPI"))])

    assert await chain.synthesize("hello there", "david-martinez", 1.0) is None
    assert chain.status == "unavailable"


async def test_chain_primary_success_is_ok() -> None:
    chain = ChainedTts([FakeBackend("edge", audio=SpeechAudio(AUDIO, "audio/mpeg"))])

    await chain.synthesize("hello there", "david-martinez", 1.0)

    assert chain.status == "ok"


def test_build_tts_unknown_primary() -> None:
    with pytest.raises(ValueError):
        build_tts("nonexistent")


def test_build_tts_primary_first() -> None:
    assert [backend.name for backend in build_tts("pyttsx3").backends] == ["pyttsx3", "edge"]
    assert [backend.name for backend in build_tts("edge").backends] == ["edge", "pyttsx3"]


def test_tts_primary_env() -> None:
    assert load_settings({"VOICE_AGENT_TTS_PRIMARY": "pyttsx3"}).tts_primary == "pyttsx3"
    assert load_settings({}).tts_primary == "edge"


class FakeCommunicate:
    """Stands in for edge_tts.Communicate: one audio chunk between two word boundaries, at 0 s and 0.5 s."""

    created: list["FakeCommunicate"] = []

    def __init__(self, text: str, voice: str, *, rate: str, pitch: str, boundary: str) -> None:
        self.text, self.voice, self.rate, self.pitch, self.boundary = text, voice, rate, pitch, boundary
        FakeCommunicate.created.append(self)

    async def stream(self) -> Any:
        yield {"type": "WordBoundary", "offset": 0, "duration": 4_000_000, "text": "hello"}
        yield {"type": "audio", "data": AUDIO}
        yield {"type": "WordBoundary", "offset": 5_000_000, "duration": 4_000_000, "text": "there"}


async def test_edge_collects_word_times(monkeypatch: pytest.MonkeyPatch) -> None:
    import edge_tts

    FakeCommunicate.created = []
    monkeypatch.setattr(edge_tts, "Communicate", FakeCommunicate)

    audio = await EdgeTtsBackend().synthesize("hello there", "david-martinez", 1.0)

    assert audio == SpeechAudio(AUDIO, "audio/mpeg", [0.0, 0.5])
    assert FakeCommunicate.created[0].boundary == "WordBoundary"


async def test_edge_rate_and_pitch_come_from_the_character_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    import edge_tts

    FakeCommunicate.created = []
    monkeypatch.setattr(edge_tts, "Communicate", FakeCommunicate)

    await EdgeTtsBackend().synthesize("hello there", "david-martinez", 1.1)

    # David's profile: rate offset +6, pitch +4 Hz; cadence 1.1 adds +10 percent.
    assert FakeCommunicate.created[0].rate == "+16%"
    assert FakeCommunicate.created[0].pitch == "+4Hz"
    assert FakeCommunicate.created[0].voice == "en-US-BrianNeural"


async def test_edge_without_audio_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import edge_tts

    class SilentCommunicate(FakeCommunicate):
        async def stream(self) -> Any:
            yield {"type": "WordBoundary", "offset": 0, "duration": 1, "text": "hello"}

    monkeypatch.setattr(edge_tts, "Communicate", SilentCommunicate)

    with pytest.raises(TtsError):
        await EdgeTtsBackend().synthesize("hello there", "david-martinez", 1.0)


def test_character_profiles_moved_intact() -> None:
    assert backends.CHARACTER_AUDIO_PROFILES["johnny-silverhand"]["voice"] == "en-US-RogerNeural"
    assert set(backends.CHARACTER_AUDIO_PROFILES) == {"johnny-silverhand", "lucy-kushinada", "david-martinez", "v-male", "v-female"}
