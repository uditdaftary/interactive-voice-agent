"""M2: Whisper is built lazily, once, off the event loop, and never at import."""

import asyncio
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from voice_agent.server.app import create_app, ensure_stt_engine
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.settings import Settings
from voice_agent.stt import recognizer
from fakes import FakeStt

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


class ExplodingSTT:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("STTEngine must not be constructed here")


class CountingSTT:
    """Stands in for STTEngine: counts constructions and records the building thread."""

    built: list[dict[str, object]] = []
    build_threads: list[int] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        time.sleep(0.05)  # widen the race window so an unguarded double build would show
        CountingSTT.built.append(dict(kwargs))
        CountingSTT.build_threads.append(threading.get_ident())

    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        return "lazily built transcript"


class GatedSTT:
    """Stands in for STTEngine: the build blocks in its worker thread until the test releases it."""

    built = 0
    started = threading.Event()
    release = threading.Event()

    def __init__(self, *args: object, **kwargs: object) -> None:
        GatedSTT.built += 1
        GatedSTT.started.set()
        if not GatedSTT.release.wait(timeout=10):
            raise AssertionError("test never released the build")


@pytest.fixture
def gated_stt(monkeypatch: pytest.MonkeyPatch) -> type[GatedSTT]:
    GatedSTT.built = 0
    GatedSTT.started = threading.Event()
    GatedSTT.release = threading.Event()
    monkeypatch.setattr(recognizer, "STTEngine", GatedSTT)
    return GatedSTT


@pytest.fixture
def counting_stt(monkeypatch: pytest.MonkeyPatch) -> type[CountingSTT]:
    CountingSTT.built = []
    CountingSTT.build_threads = []
    monkeypatch.setattr(recognizer, "STTEngine", CountingSTT)
    return CountingSTT


def test_create_app_does_not_build_stt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recognizer, "STTEngine", ExplodingSTT)
    app = create_app(connection_manager=ConnectionManager())
    assert app.state.stt_engine is None


def test_connection_manager_stt_engine_never_constructs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recognizer, "STTEngine", ExplodingSTT)
    manager = ConnectionManager()
    assert manager.stt_engine is None
    mock = FakeStt("x")
    manager.stt_engine = mock
    assert manager.stt_engine is mock


async def test_ensure_stt_engine_builds_once_off_the_event_loop(counting_stt: type[CountingSTT]) -> None:
    app = create_app(connection_manager=ConnectionManager())

    first, second = await asyncio.gather(ensure_stt_engine(app), ensure_stt_engine(app))

    assert first is second
    assert app.state.stt_engine is first
    assert len(counting_stt.built) == 1
    assert counting_stt.built[0] == {"initial_prompt": ""}
    assert counting_stt.build_threads[0] != threading.get_ident()


async def test_build_passes_the_configured_initial_prompt(counting_stt: type[CountingSTT]) -> None:
    app = create_app(connection_manager=ConnectionManager(), settings=replace(Settings(), stt_initial_prompt="Arasaka"))

    await ensure_stt_engine(app)

    assert counting_stt.built == [{"initial_prompt": "Arasaka"}]


async def test_cancelled_caller_does_not_discard_the_build(gated_stt: type[GatedSTT]) -> None:
    app = create_app(connection_manager=ConnectionManager())

    first = asyncio.create_task(ensure_stt_engine(app))
    await asyncio.to_thread(gated_stt.started.wait, 5)  # the build is now running in its worker thread
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    # A later caller joins the build already in flight instead of starting a second Whisper load.
    second = asyncio.create_task(ensure_stt_engine(app))
    gated_stt.release.set()
    engine = await second

    assert gated_stt.built == 1
    assert app.state.stt_engine is engine
    assert await ensure_stt_engine(app) is engine


async def test_ensure_stt_engine_works_without_create_app(counting_stt: type[CountingSTT]) -> None:
    # An app that only mounts the websocket router has none of create_app's state set up.
    app = FastAPI()

    first, second = await asyncio.gather(ensure_stt_engine(app), ensure_stt_engine(app))

    assert first is second
    assert len(counting_stt.built) == 1


async def test_ensure_stt_engine_prefers_existing_engines(counting_stt: type[CountingSTT]) -> None:
    on_app = FakeStt("on app")
    app = create_app(connection_manager=ConnectionManager(), stt_engine=on_app)
    assert await ensure_stt_engine(app) is on_app

    # An engine set on the manager after create_app is honoured and nothing is built.
    manager = ConnectionManager()
    app = create_app(connection_manager=manager)
    on_manager = FakeStt("on manager")
    manager.stt_engine = on_manager
    assert await ensure_stt_engine(app) is on_manager
    assert app.state.stt_engine is on_manager

    assert counting_stt.built == []


def test_transcribe_endpoint_builds_stt_on_first_use(counting_stt: type[CountingSTT]) -> None:
    app = create_app(connection_manager=ConnectionManager())
    pcm = (np.sin(np.linspace(0, 200, 16000)) * 12000).astype(np.int16).tobytes()

    with TestClient(app) as client:
        assert counting_stt.built == []
        for _ in range(2):
            resp = client.post("/api/transcribe", files={"file": ("a.pcm", pcm, "application/octet-stream")})
            assert resp.status_code == 200
            assert resp.json()["transcript"] == "lazily built transcript"

    assert len(counting_stt.built) == 1


def test_websocket_voice_end_builds_stt_on_first_use(counting_stt: type[CountingSTT]) -> None:
    narration = MagicMock()
    narration.handle_user_interruption = AsyncMock()
    app = create_app(connection_manager=ConnectionManager(), narration_engine=narration)
    pcm = (np.sin(np.linspace(0, 200, 8000)) * 12000).astype(np.int16).tobytes()

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()  # handshake state_sync
        assert counting_stt.built == []

        ws.send_json({"type": "barge_in_trigger"})
        ws.receive_json()  # barge_in
        ws.receive_json()  # state_sync LISTENING
        ws.send_bytes(pcm)
        ws.send_json({"type": "voice_end"})

        transcript = ws.receive_json()
        assert transcript["type"] == "user_speech_transcript"
        assert transcript["transcript"] == "lazily built transcript"

    assert len(counting_stt.built) == 1
    narration.handle_user_interruption.assert_called_once_with("lazily built transcript")


def test_importing_the_app_module_does_not_load_whisper() -> None:
    """The module-level `app = create_app()` must not pull in faster_whisper (it would hit the Hub)."""
    code = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
        "start = time.perf_counter()\n"
        "import voice_agent.server.app\n"
        "elapsed = time.perf_counter() - start\n"
        "assert 'faster_whisper' not in sys.modules, 'faster_whisper was imported'\n"
        "assert 'huggingface_hub' not in sys.modules, 'huggingface_hub was imported'\n"
        "print(f'import took {elapsed:.2f}s')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


async def test_cancelled_build_task_is_replaced(counting_stt: type[CountingSTT]) -> None:
    """A build cancelled before its first step never runs its finally, so it stays on app.state; the next caller must build anew."""
    from voice_agent.server.app import _build_stt_engine

    app = create_app(connection_manager=ConnectionManager())
    stale = asyncio.create_task(_build_stt_engine(app))
    stale.cancel()
    await asyncio.sleep(0)
    assert stale.cancelled()
    app.state.stt_build = stale

    engine = await ensure_stt_engine(app)

    assert engine is app.state.stt_engine
    assert len(counting_stt.built) == 1
