"""The browser's Web Speech preview sends the microphone to its vendor, so its text (`fallback_query`) is used only when
the user turned the preview on, only for a turn the server heard nothing in, and never longer than a question."""

import asyncio
import json
import logging
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

import pytest

from fakes import FakeStt, GatedStt
from voice_agent.orchestrator.state_machine import SessionState
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.schemas import MAX_QUERY_CHARS
from voice_agent.server.websocket import handle_client_text_message
from voice_agent.settings import Settings
from voice_agent.stt.accumulator import SpeechTurnAccumulator

SAMPLE_RATE = 16000
LOUD_20MS = (b"\x00\x40" * 320)  # a loud enough square wave to count as speech


class StubSocket:
    app = object()

    async def send_text(self, text: str) -> None:
        return None


def voice_end(fallback: str | None) -> str:
    return json.dumps({"type": "voice_end", **({} if fallback is None else {"fallback_query": fallback})})


def listening_manager(*, with_audio: bool) -> ConnectionManager:
    manager = ConnectionManager(session=SessionState(listen_deadline_s=30.0, followup_window_s=30.0))
    manager.accumulator = SpeechTurnAccumulator(sample_rate=SAMPLE_RATE, silence_timeout_ms=60_000)
    manager.session.enter_listening("barge_in")
    if with_audio:
        manager.accumulator.arm()
        for _ in range(5):
            manager.accumulator.feed(LOUD_20MS)
    return manager


def engine_with(preview: bool) -> Any:
    engine = MagicMock()
    engine.settings = replace(Settings.fast(), web_speech_preview=preview)
    return engine


async def finish_turn(manager: ConnectionManager) -> None:
    task = manager.tasks.get("stt")
    if task is not None:
        await asyncio.wait([task], timeout=5)


# --- no audio at all ----------------------------------------------------------------------------------------------------


async def test_a_preview_is_ignored_by_default_and_the_turn_is_given_back() -> None:
    manager = listening_manager(with_audio=False)
    engine = engine_with(preview=False)

    await handle_client_text_message(voice_end("who is lucy"), StubSocket(), manager, engine)  # type: ignore[arg-type]

    engine.ask.assert_not_called()
    engine.report_unrecognized.assert_not_called()
    assert manager.state == "PAUSED"  # nothing was heard: the turn goes back at once


async def test_a_preview_answers_a_turn_with_no_audio_when_enabled() -> None:
    manager = listening_manager(with_audio=False)
    engine = engine_with(preview=True)

    await handle_client_text_message(voice_end("  who is lucy  "), StubSocket(), manager, engine)  # type: ignore[arg-type]

    engine.ask.assert_called_once_with("who is lucy")


# --- audio the recognizer heard nothing in ------------------------------------------------------------------------------


async def test_a_preview_does_not_replace_an_empty_transcript_unless_enabled() -> None:
    manager = listening_manager(with_audio=True)
    engine = engine_with(preview=False)

    await handle_client_text_message(voice_end("who is lucy"), StubSocket(), manager, engine, FakeStt(""))  # type: ignore[arg-type]
    await finish_turn(manager)

    engine.ask.assert_not_called()
    engine.report_unrecognized.assert_called_once_with()  # the user hears that nothing was understood


async def test_a_preview_stands_in_for_an_empty_transcript_when_enabled() -> None:
    manager = listening_manager(with_audio=True)
    engine = engine_with(preview=True)

    await handle_client_text_message(voice_end("who is lucy"), StubSocket(), manager, engine, FakeStt(""))  # type: ignore[arg-type]
    await finish_turn(manager)

    engine.ask.assert_called_once_with("who is lucy")
    engine.report_unrecognized.assert_not_called()


async def test_a_preview_never_overrides_what_the_server_heard() -> None:
    manager = listening_manager(with_audio=True)
    engine = engine_with(preview=True)

    await handle_client_text_message(voice_end("something else"), StubSocket(), manager, engine, FakeStt("what is arasaka"))  # type: ignore[arg-type]
    await finish_turn(manager)

    engine.ask.assert_called_once_with("what is arasaka")


async def test_a_preview_that_is_a_rejected_transcript_still_needs_the_setting() -> None:
    """A rejection (not speech) is an empty transcript like any other."""
    manager = listening_manager(with_audio=True)
    engine = engine_with(preview=False)

    await handle_client_text_message(  # type: ignore[arg-type]
        voice_end("who is lucy"), StubSocket(), manager, engine, FakeStt("", rejected_reason="quiet")
    )
    await finish_turn(manager)

    engine.ask.assert_not_called()


# --- bounds and privacy -------------------------------------------------------------------------------------------------


async def test_a_preview_is_cut_to_the_length_of_a_question() -> None:
    manager = listening_manager(with_audio=False)
    engine = engine_with(preview=True)

    await handle_client_text_message(voice_end("w" * (MAX_QUERY_CHARS * 3)), StubSocket(), manager, engine)  # type: ignore[arg-type]

    (asked,) = engine.ask.call_args.args
    assert asked == "w" * MAX_QUERY_CHARS


async def test_an_ignored_preview_is_logged_without_its_text(caplog: pytest.LogCaptureFixture) -> None:
    manager = listening_manager(with_audio=False)

    with caplog.at_level(logging.DEBUG, logger="voice_agent.server.websocket"):
        await handle_client_text_message(voice_end("my private words"), StubSocket(), manager, engine_with(False))  # type: ignore[arg-type]

    messages = [record.getMessage() for record in caplog.records if "preview" in record.getMessage()]
    assert messages, "the dropped preview should leave a DEBUG line"
    assert all("my private words" not in message for message in messages)


async def test_a_turn_with_no_engine_ignores_the_preview() -> None:
    manager = listening_manager(with_audio=False)

    await handle_client_text_message(voice_end("who is lucy"), StubSocket(), manager, None)  # type: ignore[arg-type]

    assert manager.state == "PAUSED"


async def test_the_stt_wait_is_not_shortened_by_a_preview() -> None:
    """A preview does not skip the recognizer: with audio, the turn still waits for its transcript."""
    manager = listening_manager(with_audio=True)
    engine = engine_with(preview=True)
    stt = GatedStt("what is arasaka")

    await handle_client_text_message(voice_end("who is lucy"), StubSocket(), manager, engine, stt)  # type: ignore[arg-type]
    await stt.wait_started()
    engine.ask.assert_not_called()

    stt.release.set()
    await finish_turn(manager)

    engine.ask.assert_called_once_with("what is arasaka")


def test_the_parser_keeps_the_preview_text_and_ignores_anything_that_is_not_text() -> None:
    from voice_agent.server.schemas import VoiceEndMessage, parse_client_message

    kept = parse_client_message(voice_end("who is lucy"))
    assert isinstance(kept, VoiceEndMessage) and kept.fallback_query == "who is lucy"
    assert parse_client_message(voice_end(None)).fallback_query is None  # type: ignore[attr-defined]
    for junk in (123, ["a"], {"a": 1}, None):
        message = parse_client_message(json.dumps({"type": "voice_end", "fallback_query": junk}))
        assert message.fallback_query is None  # type: ignore[attr-defined]
