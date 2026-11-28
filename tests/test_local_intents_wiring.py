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


async def delivered() -> None:
    """Wait until every message queued so far has reached the sockets."""
    pending = [task for task in asyncio.all_tasks() if task.get_name() == "broadcast"]
    if pending:
        await asyncio.wait(pending, timeout=5)


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


# --- a typed command that leaves the state as it was ----------------------------------------------------------------
#
# The page that sends typed text has already stopped its own audio and shows LISTENING. When the command changes
# nothing, no transition tells it where the session is, so the server must.


async def test_continue_while_narrating_tells_the_page_and_narrates_on_for_it() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine, llm, _ = make_engine(manager, tts=tts)
    socket = PlayerSocket(manager)  # never reports a clip played: the page stopped it
    await manager.connect(socket)  # type: ignore[arg-type]
    engine.start_narration()
    await socket.wait_for_clips(1)
    await delivered()
    generation = socket.clips[0].generation
    syncs_before = len(socket.messages("state_sync"))
    stops_before = len(socket.messages("audio_stop"))

    engine.ask("continue")
    # A narration that kept waiting for the report would send nothing for the next 30 seconds.
    await socket.wait_until(lambda: any(clip.generation > generation for clip in socket.clips))
    await delivered()

    new_syncs = socket.messages("state_sync")[syncs_before:]
    assert [m["state"] for m in new_syncs] == ["NARRATING"]
    assert len(socket.messages("audio_stop")) == stops_before + 1
    assert [m for m in socket.messages("error")] == []
    assert any(m.get("pre_roll_words") == 4 for m in socket.messages("prompt_envelope"))  # the page is told it is coming back
    assert manager.state == "NARRATING"
    assert llm.calls == []
    await manager.tasks.cancel_all()


async def test_repeat_while_narrating_tells_the_page() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine, llm, _ = make_engine(manager, tts=tts)
    socket = await tab(manager)
    await narrating(manager, engine, tts)
    await delivered()
    syncs_before = len(socket.messages("state_sync"))

    engine.ask("repeat")
    await delivered()

    assert [m["state"] for m in socket.messages("state_sync")[syncs_before:]] == ["NARRATING"]
    assert llm.calls == []
    await manager.tasks.cancel_all()


@pytest.mark.parametrize("command", ["continue", "repeat"])
async def test_the_page_hears_the_state_before_the_restarted_narration_sends_anything(command: str) -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine, _, _ = make_engine(manager, tts=tts)
    socket = PlayerSocket(manager)  # never reports a clip played: the page stopped it
    await manager.connect(socket)  # type: ignore[arg-type]
    engine.start_narration()
    await socket.wait_for_clips(1)
    await delivered()
    generation = socket.clips[0].generation
    frames_before = len(socket.frames)

    engine.ask(command)
    await socket.wait_until(lambda: any(clip.generation > generation for clip in socket.clips))
    await delivered()

    after = socket.frames[frames_before:]
    assert "state_sync" in after
    assert "cursor_update" in after
    assert after.index("state_sync") < after.index("cursor_update")
    await manager.tasks.cancel_all()


async def test_resume_during_a_reply_tells_the_page_and_the_reply_goes_on_without_the_stopped_clip() -> None:
    manager = ConnectionManager()
    engine, llm, _ = make_engine(manager)
    socket = PlayerSocket(manager)  # never reports a clip played: the page stopped it
    await manager.connect(socket)  # type: ignore[arg-type]
    engine.ask("who writes this book")
    reply = manager.tasks.get("reply")
    assert reply is not None
    await socket.wait_for_clips(1)  # the answer is being played
    await delivered()
    syncs_before = len(socket.messages("state_sync"))
    stops_before = len(socket.messages("audio_stop"))

    engine.ask("resume")
    # The reply stops waiting for a report of the clip the page dropped (the wait would otherwise last 30 seconds).
    done, _ = await asyncio.wait([reply], timeout=5)
    await delivered()

    assert reply in done and not reply.cancelled()
    assert [m["state"] for m in socket.messages("state_sync")[syncs_before:]][0] == "REPLYING"
    assert len(socket.messages("audio_stop")) == stops_before + 1
    assert len(llm.calls) == 1
    await wait_for_state(manager.session, "NARRATING")  # the resume was kept: narration follows the reply
    await manager.tasks.cancel_all()


async def test_pause_while_paused_tells_the_page() -> None:
    manager = ConnectionManager()
    engine, llm, _ = make_engine(manager)
    socket = await tab(manager)
    assert manager.state == "PAUSED"

    engine.ask("pause")
    await delivered()

    assert [m["state"] for m in socket.messages("state_sync")] == ["PAUSED"]
    assert llm.calls == []


@pytest.mark.parametrize("command", ["continue", "repeat", "pause"])
async def test_the_awaited_form_tells_the_page_too(command: str) -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine, _, _ = make_engine(manager, tts=tts)
    socket = await tab(manager)
    await narrating(manager, engine, tts)
    await delivered()
    syncs_before = len(socket.messages("state_sync"))

    await engine.handle_user_interruption(command)
    await delivered()

    new = [m["state"] for m in socket.messages("state_sync")[syncs_before:]]
    assert new == (["PAUSED"] if command == "pause" else ["NARRATING"])
    await manager.tasks.cancel_all()


# --- the resume cue ---------------------------------------------------------------------------------------------------


def resume_cues(socket: PlayerSocket) -> list[dict[str, Any]]:
    """The envelopes that tell the page narration is coming back from a question, a few words back."""
    return [m for m in socket.messages("prompt_envelope") if m["state"] == "NARRATING"]


@pytest.mark.parametrize("coming_from", ["LISTENING", "REPLYING", "PAUSED"])
async def test_a_repeat_sends_no_resume_cue_because_it_rewinds_nothing(coming_from: str) -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    llm = BlockingLLM()
    engine, _, _ = make_engine(manager, llm=llm, tts=tts)
    socket = await tab(manager)
    if coming_from == "LISTENING":
        await narrating(manager, engine, tts)
        manager.trigger_barge_in()
    elif coming_from == "REPLYING":
        engine.ask("who writes this book")
        async with asyncio.timeout(5):
            await llm.started.wait()
    spoken_before = len(tts.calls)

    engine.ask("repeat")
    async with asyncio.timeout(5):
        await tts.wait_for_calls(spoken_before + 1)
    await delivered()

    assert resume_cues(socket) == []
    await manager.tasks.cancel_all()


async def test_a_plain_resume_from_a_question_still_sends_the_cue() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine, _, _ = make_engine(manager, tts=tts)
    socket = await tab(manager)
    await narrating(manager, engine, tts)
    manager.trigger_barge_in()
    spoken_before = len(tts.calls)

    engine.start_narration()
    async with asyncio.timeout(5):
        await tts.wait_for_calls(spoken_before + 1)
    await delivered()

    assert [m["pre_roll_words"] for m in resume_cues(socket)] == [4]
    await manager.tasks.cancel_all()
