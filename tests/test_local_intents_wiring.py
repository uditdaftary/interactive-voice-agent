"""Commands and greetings are handled before retrieval and never reach the model, however the session is placed."""

import asyncio
from typing import Any

import pytest

from fakes import BlockingLLM, FakeLLM, FakeSearch, FakeTts, PlayerSocket, wait_for_state
from voice_agent.llm.intents import greeting_reply
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.narration import NarrationEngine
from voice_agent.settings import Settings

BOOK = [
    "The first sentence is about rain.",
    "The second sentence is about snow and wind.",
    "The third sentence is about fog and mist rolling in.",
    "The fourth sentence is about sun.",
]


def make_engine(manager: ConnectionManager, *, llm: Any = None, tts: Any = None) -> tuple[NarrationEngine, FakeLLM, FakeSearch]:
    llm = llm if llm is not None else FakeLLM(tokens=("An answer from the model.",))
    search = FakeSearch()
    engine = NarrationEngine(
        manager,
        llm_client=llm,
        initial_sentences=BOOK,
        settings=Settings.fast(),
        tts=tts if tts is not None else FakeTts(),
        search=search,
    )
    return engine, llm, search


async def tab(manager: ConnectionManager) -> PlayerSocket:
    socket = PlayerSocket(manager, auto_ack=True)
    await manager.connect(socket)  # type: ignore[arg-type]
    return socket


def record_changes(manager: ConnectionManager) -> list[str]:
    """The state each change of the session went to, in order."""
    states: list[str] = []
    manager.session.add_listener(lambda old, new, reason: states.append(new))
    return states


async def narrating(manager: ConnectionManager, engine: NarrationEngine, tts: FakeTts) -> None:
    engine.start_narration()
    async with asyncio.timeout(5):
        await tts.wait_for_calls(1)


# --- pause ------------------------------------------------------------------------------------------------------------


async def test_pause_while_narrating_pauses_and_asks_nothing() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine, llm, search = make_engine(manager, tts=tts)
    await tab(manager)
    await narrating(manager, engine, tts)
    states = record_changes(manager)

    engine.ask("Pause.")

    assert states == ["PAUSED"]
    assert manager.tasks.get("narration") is None
    assert llm.calls == [] and search.calls == []
    assert engine._conversation_history == []


async def test_pause_in_the_follow_up_window_is_one_change() -> None:
    manager = ConnectionManager()
    engine, llm, _ = make_engine(manager)
    await tab(manager)
    manager.session.enter_listening("followup", followup=True)
    states = record_changes(manager)

    engine.ask("hold on")

    assert states == ["PAUSED"]
    assert llm.calls == []


async def test_pause_in_the_middle_of_a_reply_cuts_it_off_with_one_change() -> None:
    manager = ConnectionManager()
    llm = BlockingLLM()
    engine, _, _ = make_engine(manager, llm=llm)
    await tab(manager)
    engine.ask("who writes this book")
    reply = manager.tasks.get("reply")
    assert reply is not None
    async with asyncio.timeout(5):
        await llm.started.wait()
    states = record_changes(manager)

    engine.ask("stop")
    await asyncio.wait([reply], timeout=5)

    assert states == ["PAUSED"]
    assert reply.cancelled()
    assert len(llm.calls) == 1  # the question was the only thing the model was ever asked
    await manager.tasks.cancel_all()


# --- resume -----------------------------------------------------------------------------------------------------------


async def test_resume_from_paused_narrates() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine, llm, _ = make_engine(manager, tts=tts)
    await tab(manager)
    states = record_changes(manager)

    engine.ask("continue")
    async with asyncio.timeout(5):
        await tts.wait_for_calls(1)

    assert states == ["NARRATING"]
    assert llm.calls == []
    await manager.tasks.cancel_all()


async def test_resume_spoken_after_a_barge_in_from_a_paused_session_narrates() -> None:
    """#90: the session came from PAUSED, so LISTENING would only go back to PAUSED; the user asked for narration."""
    manager = ConnectionManager()
    tts = FakeTts()
    engine, llm, _ = make_engine(manager, tts=tts)
    await tab(manager)
    manager.session.enter_listening("barge_in")
    assert manager.session.resume_to == "PAUSED"
    states = record_changes(manager)

    engine.ask("go on")
    async with asyncio.timeout(5):
        await tts.wait_for_calls(1)

    assert states == ["NARRATING"]
    assert llm.calls == []
    await manager.tasks.cancel_all()


# --- repeat -----------------------------------------------------------------------------------------------------------


async def test_repeat_while_narrating_starts_the_sentence_again_without_a_state_change() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine, llm, _ = make_engine(manager, tts=tts)
    socket = await tab(manager)
    await narrating(manager, engine, tts)
    engine.manager.tasks.cancel("narration")  # hold the loop still, mid sentence, while the test sets the cursor
    engine.current_sentence_idx, engine.current_word_idx = 2, 8  # far enough in that a restart would pre-roll
    states = record_changes(manager)
    spoken_before = len(tts.calls)

    engine.ask("say that again")
    async with asyncio.timeout(5):
        await tts.wait_for_calls(spoken_before + 1)

    assert tts.calls[spoken_before][0] == BOOK[2]  # the whole sentence, not the words from the stop point
    assert states == []
    assert manager.state == "NARRATING"
    assert llm.calls == []
    assert any(m["sentence_idx"] == 2 and m["word_offset"] == 0 for m in socket.messages("cursor_update"))
    await manager.tasks.cancel_all()


async def test_repeat_while_paused_narrates_the_current_sentence() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine, llm, _ = make_engine(manager, tts=tts)
    await tab(manager)
    engine.current_sentence_idx, engine.current_word_idx = 1, 6
    states = record_changes(manager)

    engine.ask("repeat")
    async with asyncio.timeout(5):
        await tts.wait_for_calls(1)

    assert states == ["NARRATING"]
    assert tts.calls[0][0] == BOOK[1]
    assert llm.calls == []
    await manager.tasks.cancel_all()


async def test_repeat_while_listening_returns_to_the_sentence_at_once() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine, llm, _ = make_engine(manager, tts=tts)
    await tab(manager)
    await narrating(manager, engine, tts)
    manager.trigger_barge_in()  # the user speaks: narration stops mid sentence
    engine.current_sentence_idx, engine.current_word_idx = 2, 8
    states = record_changes(manager)
    spoken_before = len(tts.calls)

    engine.ask("what did you say")
    async with asyncio.timeout(5):
        await tts.wait_for_calls(spoken_before + 1)

    assert states == ["NARRATING"]
    assert tts.calls[spoken_before][0] == BOOK[2]
    assert llm.calls == []
    await manager.tasks.cancel_all()


async def test_repeat_during_a_reply_drops_the_reply() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    llm = BlockingLLM()
    engine, _, _ = make_engine(manager, llm=llm, tts=tts)
    await tab(manager)
    engine.ask("who writes this book")
    reply = manager.tasks.get("reply")
    assert reply is not None
    async with asyncio.timeout(5):
        await llm.started.wait()
    states = record_changes(manager)

    engine.ask("repeat")
    await asyncio.wait([reply], timeout=5)
    async with asyncio.timeout(5):
        await tts.wait_for_calls(1)

    assert states == ["NARRATING"]
    assert reply.cancelled()
    assert tts.calls[0][0] == BOOK[0]
    await manager.tasks.cancel_all()


# --- greetings --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("voice", ["david-martinez", "lucy-kushinada", "unknown-voice"])
async def test_a_greeting_is_answered_locally_and_opens_the_follow_up_window(voice: str) -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine, llm, search = make_engine(manager, tts=tts)
    socket = await tab(manager)
    manager.set_voice(voice)
    states = record_changes(manager)

    engine.ask("Hello!")
    await wait_for_state(manager.session, "PAUSED")  # the window is zero long: the session goes back to what it was

    assert tts.texts == [greeting_reply(voice)]
    assert states == ["REPLYING", "LISTENING", "PAUSED"]
    assert llm.calls == [] and search.calls == []
    assert engine._conversation_history == []
    envelopes = socket.messages("prompt_envelope")
    assert envelopes[-1]["reply"] == greeting_reply(voice)
    await manager.tasks.cancel_all()


async def test_a_greeting_in_the_follow_up_window_is_one_change_and_no_model_call() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine, llm, _ = make_engine(manager, tts=tts)
    await tab(manager)
    manager.session.enter_listening("followup", followup=True)
    states = record_changes(manager)

    engine.ask("hey")
    reply = manager.tasks.get("reply")
    assert reply is not None
    assert states == ["REPLYING"]  # the change is made before ask() returns
    await asyncio.wait([reply], timeout=5)

    assert llm.calls == []
    assert tts.texts == [greeting_reply("david-martinez")]
    await manager.tasks.cancel_all()


# --- what is not a command --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["why did she stop", "stop the rain please", "tell me about the fog"])
async def test_a_longer_sentence_that_contains_a_command_word_is_a_question(query: str) -> None:
    manager = ConnectionManager()
    engine, llm, _ = make_engine(manager)
    await tab(manager)

    await engine.handle_user_interruption(query)

    assert len(llm.calls) == 1
    await manager.tasks.cancel_all()


async def test_the_awaited_form_carries_out_a_command_too() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine, llm, _ = make_engine(manager, tts=tts)
    await tab(manager)
    await narrating(manager, engine, tts)

    await engine.handle_user_interruption("pause")

    assert manager.state == "PAUSED"
    assert llm.calls == []
    await manager.tasks.cancel_all()
