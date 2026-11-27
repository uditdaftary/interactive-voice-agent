"""The statuses the HUD shows (model, speech, recognizer) ride on `state_sync`, and every tab is told once per change."""

import asyncio
import importlib
import threading
from collections.abc import AsyncGenerator
from dataclasses import replace
from typing import Any

import pytest

from fakes import FakeLLM, FakeSearch, FakeStt, FakeTts, PlayerSocket, StreamingLLM, wait_for_state
from voice_agent.server.app import create_app, ensure_stt_engine
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.narration import UNRECOGNIZED_LINE, NarrationEngine
from voice_agent.server.websocket import submit_turn
from voice_agent.settings import Settings
from voice_agent.stt.backends import STT_ERROR, STT_UNAVAILABLE, Transcript, stt_status
from voice_agent.tts.backends import ChainedTts, SpeechAudio, TtsError

# Not `from voice_agent.server import app`: the package re-exports the FastAPI instance under that name.
app_module = importlib.import_module("voice_agent.server.app")

AUDIO = SpeechAudio(b"x" * 200, "audio/wav")


def make_engine(
    manager: ConnectionManager, *, llm: Any = None, tts: Any = None, settings: Settings | None = None
) -> NarrationEngine:
    return NarrationEngine(
        manager,
        llm_client=llm if llm is not None else StreamingLLM(),
        settings=settings if settings is not None else Settings.fast(),
        tts=tts if tts is not None else FakeTts(),
        search=FakeSearch(),
    )


async def tab(manager: ConnectionManager) -> PlayerSocket:
    socket = PlayerSocket(manager, auto_ack=True)
    await manager.connect(socket)  # type: ignore[arg-type]
    return socket


async def delivered() -> None:
    """Wait until every message queued so far has reached the sockets."""
    pending = [task for task in asyncio.all_tasks() if task.get_name() == "broadcast"]
    if pending:
        await asyncio.wait(pending, timeout=5)


def seen(socket: PlayerSocket, field: str) -> list[Any]:
    """The value of `field` in each state_sync the tab received, in order."""
    return [message[field] for message in socket.messages("state_sync")]


def changes(values: list[Any], start: Any) -> list[Any]:
    """The values that differ from the one before them, the first one being compared with `start`."""
    return [now for before, now in zip([start, *values], values, strict=False) if now != before]


class SwitchableBackend:
    """A TTS backend that speaks while `working` is set and fails otherwise."""

    def __init__(self, name: str, working: bool = True) -> None:
        self.name = name
        self.working = working

    async def synthesize(self, text: str, voice_id: str, cadence: float) -> SpeechAudio:
        if not self.working:
            raise TtsError("down")
        return AUDIO


# --- what a tab is told on connecting ----------------------------------------------------------------------------------


async def test_a_new_tab_is_told_every_status() -> None:
    manager = ConnectionManager()
    make_engine(manager, llm=FakeLLM(), settings=replace(Settings.fast(), web_speech_preview=True))
    socket = await tab(manager)

    await manager.send_state_sync(socket)  # type: ignore[arg-type]

    message = socket.messages("state_sync")[0]
    assert message["llm_status"] == "ok"  # nothing has been asked yet
    assert message["llm_provider"] == "fake"
    assert message["tts_status"] == "ok"
    assert message["stt_status"] == "ok"
    assert message["web_speech_preview"] is True


async def test_the_web_speech_preview_is_off_by_default() -> None:
    manager = ConnectionManager()
    make_engine(manager, llm=FakeLLM())
    socket = await tab(manager)

    await manager.send_state_sync(socket)  # type: ignore[arg-type]

    assert socket.messages("state_sync")[0]["web_speech_preview"] is False


async def test_the_no_model_provider_reports_none() -> None:
    manager = ConnectionManager()
    llm = FakeLLM(status="offline")
    llm.provider = "none"
    make_engine(manager, llm=llm)
    socket = await tab(manager)

    await manager.send_state_sync(socket)  # type: ignore[arg-type]

    message = socket.messages("state_sync")[0]
    assert (message["llm_status"], message["llm_provider"]) == ("none", "none")


async def test_a_manager_without_an_engine_reports_the_defaults() -> None:
    manager = ConnectionManager()

    message = manager.get_state_sync_message()

    assert (message.llm_status, message.tts_status, message.stt_status, message.web_speech_preview) == ("ok", "ok", "ok", False)


# --- one broadcast per change, none without one ------------------------------------------------------------------------


async def test_a_recognizer_status_change_reaches_every_tab_once() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager)
    first, second = await tab(manager), await tab(manager)

    engine.set_stt_status("loading")
    await delivered()
    engine.set_stt_status("loading")  # nothing changed
    await delivered()

    for socket in (first, second):
        assert seen(socket, "stt_status") == ["loading"]

    engine.set_stt_status("ok")
    await delivered()

    assert seen(first, "stt_status") == ["loading", "ok"]
    assert seen(second, "stt_status") == ["loading", "ok"]


async def test_a_new_tab_does_not_make_the_others_hear_an_old_status_again() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager)
    first = await tab(manager)
    engine.set_stt_status("loading")
    await delivered()

    second = await tab(manager)
    await manager.send_state_sync(second)  # type: ignore[arg-type]
    engine.set_stt_status("loading")
    await delivered()

    assert seen(first, "stt_status") == ["loading"]
    assert seen(second, "stt_status") == ["loading"]


async def test_the_speech_status_follows_the_chain_and_each_change_is_announced_once() -> None:
    edge, pyttsx3 = SwitchableBackend("edge", working=False), SwitchableBackend("pyttsx3")
    manager = ConnectionManager()
    engine = make_engine(manager, tts=ChainedTts([edge, pyttsx3]))
    socket = await tab(manager)

    await engine._speak("First sentence here.")
    await engine._speak("Second sentence here.")  # still on the fallback
    await delivered()
    assert seen(socket, "tts_status") == ["fallback:pyttsx3"]

    edge.working = pyttsx3.working = False
    await engine._speak("Third sentence here.")
    await delivered()
    assert seen(socket, "tts_status") == ["fallback:pyttsx3", "unavailable"]

    edge.working = True
    await engine._speak("Fourth sentence here.")
    await delivered()
    assert seen(socket, "tts_status") == ["fallback:pyttsx3", "unavailable", "ok"]


async def test_narration_still_advances_when_every_backend_has_failed() -> None:
    manager = ConnectionManager()
    chain = ChainedTts([SwitchableBackend("edge", working=False), SwitchableBackend("pyttsx3", working=False)])
    engine = NarrationEngine(
        manager,
        llm_client=FakeLLM(),
        initial_sentences=["One two.", "Three four."],
        settings=Settings.fast(),
        tts=chain,
        search=FakeSearch(),
    )
    socket = await tab(manager)

    engine.start_narration()
    await socket.wait_until(lambda: any(m["sentence_idx"] == 1 for m in socket.messages("cursor_update")))
    await delivered()

    assert seen(socket, "tts_status")[-1] == "unavailable"
    assert changes(seen(socket, "tts_status"), "ok") == ["unavailable"]
    await manager.tasks.cancel_all()


async def test_a_model_status_is_announced_once_and_not_per_token() -> None:
    async def run(status: str, tokens: int) -> PlayerSocket:
        manager = ConnectionManager()
        engine = make_engine(manager, llm=FakeLLM(tokens=["word "] * tokens, status=status))
        socket = await tab(manager)
        engine.ask("who is lucy")
        await wait_for_state(manager.session, "PAUSED")  # the follow-up window is zero long: back to where it was
        await delivered()
        await manager.tasks.cancel_all()
        return socket

    ok, offline, long_offline = await run("ok", 3), await run("offline", 3), await run("offline", 60)

    assert changes(seen(offline, "llm_status"), "ok") == ["offline"]
    assert changes(seen(ok, "llm_status"), "ok") == []
    # The change costs one extra state_sync, and a longer stream costs none.
    assert len(offline.messages("state_sync")) == len(ok.messages("state_sync")) + 1
    assert len(long_offline.messages("state_sync")) == len(offline.messages("state_sync"))


class CutOffLLM(FakeLLM):
    """Streams one token and waits. Closing the stream leaves `interrupted`, as `LLMClient` does for a consumer that stops."""

    async def stream_generate(  # type: ignore[override]  # the base yields its tokens; this one waits after the first
        self, messages: list[dict[str, str]], character_id: str | None = None
    ) -> AsyncGenerator[str, None]:
        self.calls.append(messages)
        try:
            yield "partial "
            await asyncio.Event().wait()
        finally:
            self.status = "interrupted"


async def test_a_barge_in_is_not_reported_as_a_model_failure() -> None:
    """The user cutting a reply off ends its stream early; the model did nothing wrong."""
    manager = ConnectionManager()
    llm = CutOffLLM()
    engine = make_engine(manager, llm=llm)
    socket = await tab(manager)

    engine.ask("who is lucy")
    reply = manager.tasks.get("reply")
    assert reply is not None
    await socket.wait_until(lambda: any(m.get("reply") == "partial" for m in socket.messages("prompt_envelope")))
    manager.trigger_barge_in()
    await asyncio.wait([reply], timeout=5)
    await delivered()

    assert reply.cancelled()
    assert llm.status == "interrupted"  # the client did record it...
    assert "interrupted" not in seen(socket, "llm_status")  # ...and the HUD was not told
    assert engine.statuses().llm_status == "ok"
    await manager.tasks.cancel_all()


async def offline_engine(manager: ConnectionManager) -> NarrationEngine:
    """An engine whose last question found the model offline."""
    engine = make_engine(manager, llm=FakeLLM(status="offline"))
    engine.ask("who is lucy")
    await wait_for_state(manager.session, "PAUSED")
    assert engine.statuses().llm_status == "offline"
    return engine


@pytest.mark.parametrize(
    "change",
    [
        {"base_url": "http://127.0.0.1:1234/v1"},
        {"provider": "lmstudio"},
        {"model": "another-model"},
        {"api_key": "a-new-key"},
    ],
)
async def test_changing_the_model_endpoint_starts_the_status_over(change: dict[str, str]) -> None:
    manager = ConnectionManager()
    socket = await tab(manager)
    engine = await offline_engine(manager)

    engine.configure_llm(**change)
    await manager.broadcast_state()

    assert seen(socket, "llm_status")[-1] == "ok"
    await manager.tasks.cancel_all()


@pytest.mark.parametrize(
    "change",
    [
        {"web_search_enabled": False},
        {"base_url": None, "api_key": None, "provider": None, "model": None, "web_search_enabled": None},  # a body that sets only the voice
        {"provider": "fake"},  # what it already is
        {"api_key": "local"},
    ],
)
async def test_a_change_that_leaves_the_model_endpoint_alone_keeps_the_status(change: dict[str, object]) -> None:
    manager = ConnectionManager()
    socket = await tab(manager)
    engine = await offline_engine(manager)

    engine.configure_llm(**change)
    await manager.broadcast_state()

    assert seen(socket, "llm_status")[-1] == "offline"
    await manager.tasks.cancel_all()


async def test_a_websocket_config_message_that_only_changes_web_search_keeps_the_status() -> None:
    from voice_agent.server.websocket import handle_client_text_message

    manager = ConnectionManager()
    socket = await tab(manager)
    engine = await offline_engine(manager)

    await handle_client_text_message('{"type": "update_config", "web_search_enabled": false}', socket, manager, engine)  # type: ignore[arg-type]
    await delivered()

    assert seen(socket, "llm_status")[-1] == "offline"
    await manager.tasks.cancel_all()


async def test_switching_to_the_no_model_provider_is_announced() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager, llm=FakeLLM())
    socket = await tab(manager)

    engine.configure_llm(provider="none")
    await manager.broadcast_state()

    message = socket.messages("state_sync")[-1]
    assert (message["llm_status"], message["llm_provider"]) == ("none", "none")


# --- the recognizer ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        (Transcript("what is lucy"), "ok"),
        (Transcript("", "quiet"), "rejected"),
        (Transcript("", "hallucination"), "rejected"),
        (Transcript("", STT_UNAVAILABLE), "unavailable"),
        (Transcript("", STT_ERROR), "unavailable"),
    ],
)
def test_a_transcript_says_what_state_the_recognizer_is_in(transcript: Transcript, expected: str) -> None:
    assert stt_status(transcript) == expected


@pytest.mark.parametrize(
    ("stt", "expected"),
    [
        (FakeStt("", rejected_reason="quiet"), "rejected"),
        (FakeStt("", rejected_reason=STT_UNAVAILABLE), "unavailable"),
    ],
)
async def test_a_rejected_turn_is_reported_and_answered_with_the_unrecognized_line(stt: FakeStt, expected: str) -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine = make_engine(manager, tts=tts)
    socket = await tab(manager)
    manager.session.enter_listening("barge_in")

    submit_turn(socket, manager, engine, stt, b"\x01\x00" * 1600, None)  # type: ignore[arg-type]
    reply_started = asyncio.Event()
    manager.session.add_listener(lambda old, new, reason: reply_started.set() if new == "REPLYING" else None)
    async with asyncio.timeout(5):
        await reply_started.wait()
    await wait_for_state(manager.session, "PAUSED")
    await delivered()

    assert tts.texts == [UNRECOGNIZED_LINE]
    assert changes(seen(socket, "stt_status"), "ok") == [expected]
    await manager.tasks.cancel_all()


async def test_a_good_transcript_clears_the_rejection() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager)
    socket = await tab(manager)
    engine.set_stt_status("rejected")
    manager.session.enter_listening("barge_in")

    submit_turn(socket, manager, engine, FakeStt("what is lucy"), b"\x01\x00" * 1600, None)  # type: ignore[arg-type]
    await manager.tasks.get("stt")  # type: ignore[misc]
    await delivered()

    assert changes(seen(socket, "stt_status"), "ok") == ["rejected", "ok"]
    await manager.tasks.cancel_all()


async def test_a_slow_stt_load_is_reported_as_loading_and_then_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    started = threading.Event()

    class SlowBackend:
        def __init__(self, settings: Settings) -> None:
            started.set()
            assert release.wait(5), "the test never released the load"

        def load(self) -> bool:
            return True

    monkeypatch.setattr(app_module, "build_stt_backend", SlowBackend)
    manager = ConnectionManager()
    app = create_app(connection_manager=manager, settings=Settings.fast(), llm_client=FakeLLM(), tts=FakeTts(), search=FakeSearch())
    socket = await tab(manager)

    build = asyncio.create_task(ensure_stt_engine(app))
    await asyncio.to_thread(started.wait, 5)
    await delivered()
    assert seen(socket, "stt_status") == ["loading"]

    release.set()
    await build
    await delivered()

    assert seen(socket, "stt_status") == ["loading", "ok"]


@pytest.mark.parametrize("failure", ["raises", "does not load"])
async def test_a_recognizer_that_cannot_load_is_reported_unavailable(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenBackend:
        def __init__(self, settings: Settings) -> None:
            if failure == "raises":
                raise OSError("no model")

        def load(self) -> bool:
            return False

    monkeypatch.setattr(app_module, "build_stt_backend", BrokenBackend)
    manager = ConnectionManager()
    app = create_app(connection_manager=manager, settings=Settings.fast(), llm_client=FakeLLM(), tts=FakeTts(), search=FakeSearch())
    socket = await tab(manager)

    if failure == "raises":
        with pytest.raises(OSError):
            await ensure_stt_engine(app)
    else:
        await ensure_stt_engine(app)
    await delivered()

    assert seen(socket, "stt_status") == ["loading", "unavailable"]
