"""A reply is one tracked task. However it ends (answered, failed, cut short) the session is left in a sane state,
no task is orphaned, and the clients hear about it in the right order."""

import asyncio
import gc
import json
import logging
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fakes import (
    BlockingLLM,
    FailingTts,
    FakeSearch,
    FakeTts,
    PlayerSocket,
    RecordingSocket,
    StreamingLLM,
    wait_for_state,
    wait_until,
)
from voice_agent.llm.client import OFFLINE_LINE
from voice_agent.llm.prompt import HISTORY_MESSAGES
from voice_agent.server.app import create_app
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.narration import REPLY_ERROR_LINE, UNRECOGNIZED_LINE, NarrationEngine
from voice_agent.settings import Settings
from voice_agent.stt.accumulator import SpeechTurnAccumulator

ANSWER = "Hello there."  # what StreamingLLM's default tokens add up to


def make_engine(
    manager: ConnectionManager,
    *,
    llm: Any = None,
    tts: Any = None,
    settings: Settings | None = None,
) -> NarrationEngine:
    return NarrationEngine(
        manager,
        llm_client=llm if llm is not None else StreamingLLM(),
        settings=settings if settings is not None else Settings.fast(),
        tts=tts if tts is not None else FakeTts(),
        search=FakeSearch(),
    )


async def connected(manager: ConnectionManager) -> PlayerSocket:
    """A tab that plays every clip it is sent and says so at once, as a browser does."""
    socket = PlayerSocket(manager, auto_ack=True)
    await manager.connect(socket)  # type: ignore[arg-type]
    return socket


async def start_narrating(engine: NarrationEngine, tts: FakeTts) -> None:
    engine.start_narration()
    async with asyncio.timeout(5):
        await tts.wait_for_calls(1)


async def finished(manager: ConnectionManager) -> asyncio.Task[Any]:
    """The current reply task, once it has ended."""
    reply = manager.tasks.get("reply")
    assert reply is not None, "no reply task is running"
    await asyncio.wait([reply], timeout=5)
    assert reply.done()
    return reply


async def delivered() -> None:
    """Wait until every state change and event queued so far has reached the sockets."""
    pending = [task for task in asyncio.all_tasks() if task.get_name() == "broadcast"]
    if pending:
        await asyncio.wait(pending, timeout=5)


def block_after_first_token(manager: ConnectionManager, monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Make a reply wait, mid-stream, while it broadcasts its first token. The event is set when it gets there."""
    blocked = asyncio.Event()
    send = manager.broadcast_json

    async def gated(message: Any) -> None:
        if getattr(message, "type", "") == "prompt_envelope" and getattr(message, "reply", "") == "Hello":
            blocked.set()
            await asyncio.Event().wait()  # until the reply is cancelled
        await send(message)

    monkeypatch.setattr(manager, "broadcast_json", gated)
    return blocked


# --- failures: reported, spoken, and the session is given back ---------------------------------------------------------


@pytest.mark.parametrize("resting", ["NARRATING", "PAUSED"])
async def test_an_llm_error_is_reported_spoken_and_the_session_resumes(
    resting: str, caplog: pytest.LogCaptureFixture
) -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine = make_engine(manager, llm=StreamingLLM(error=RuntimeError("secret detail")), tts=tts)
    socket = await connected(manager)
    if resting == "NARRATING":
        await start_narrating(engine, tts)

    with caplog.at_level(logging.ERROR, logger="voice_agent.server.narration"):
        engine.ask("who is lucy")
        reply = await finished(manager)
    await delivered()

    assert reply.exception() is None  # handled inside the reply, not left for the registry to log
    assert [m["detail"] for m in socket.messages("reply_error")] == ["RuntimeError"]
    assert "secret detail" not in "".join(socket.texts)  # the type name only
    assert tts.texts.count(REPLY_ERROR_LINE) == 1
    assert manager.state == resting
    assert (manager.tasks.get("narration") is not None) == (resting == "NARRATING")
    assert engine._conversation_history == []
    logged = [record for record in caplog.records if record.exc_info]
    assert logged and logged[0].exc_info is not None and logged[0].exc_info[0] is RuntimeError
    await manager.tasks.cancel_all()


@pytest.mark.parametrize("resting", ["NARRATING", "PAUSED"])
async def test_a_tts_failure_on_the_answer_is_reported_and_the_neutral_line_is_spoken(resting: str) -> None:
    manager = ConnectionManager()
    tts = FailingTts(fail_on={ANSWER})
    engine = make_engine(manager, tts=tts)
    socket = await connected(manager)
    if resting == "NARRATING":
        engine.start_narration()
        async with asyncio.timeout(5):
            await tts.wait_for_calls(1)

    engine.ask("who is lucy")
    await finished(manager)
    await delivered()

    assert [m["detail"] for m in socket.messages("reply_error")] == ["RuntimeError"]
    assert tts.attempts.index(REPLY_ERROR_LINE) > tts.attempts.index(ANSWER)
    assert REPLY_ERROR_LINE in tts.texts  # spoken
    assert manager.state == resting
    await manager.tasks.cancel_all()


@pytest.mark.parametrize("resting", ["NARRATING", "PAUSED"])
async def test_a_tts_failure_on_the_neutral_line_is_logged_and_skipped(
    resting: str, caplog: pytest.LogCaptureFixture
) -> None:
    manager = ConnectionManager()
    tts = FailingTts(fail_on={ANSWER, REPLY_ERROR_LINE})
    engine = make_engine(manager, tts=tts)
    socket = await connected(manager)
    if resting == "NARRATING":
        engine.start_narration()
        async with asyncio.timeout(5):
            await tts.wait_for_calls(1)

    with caplog.at_level(logging.ERROR, logger="voice_agent.server.narration"):
        engine.ask("who is lucy")
        await finished(manager)
    await delivered()

    assert REPLY_ERROR_LINE in tts.attempts
    assert REPLY_ERROR_LINE not in tts.texts
    assert [m["detail"] for m in socket.messages("reply_error")] == ["RuntimeError"]
    assert any("reply-error line" in record.getMessage() for record in caplog.records)
    assert manager.state == resting  # the session is given back even though nothing could be said
    await manager.tasks.cancel_all()


@pytest.mark.parametrize("tokens", [[], ["", "   "]])
async def test_an_empty_reply_is_an_error(tokens: list[str]) -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine = make_engine(manager, llm=StreamingLLM(tokens=tokens), tts=tts)
    socket = await connected(manager)

    engine.ask("who is lucy")
    await finished(manager)
    await delivered()

    assert [m["detail"] for m in socket.messages("reply_error")] == ["EmptyReply"]
    assert tts.texts == [REPLY_ERROR_LINE]  # no canned persona line
    assert engine._conversation_history == []
    assert manager.state == "PAUSED"


async def test_a_real_reply_is_remembered() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager)

    engine.ask("who is lucy")
    await finished(manager)

    assert engine._conversation_history == [
        {"role": "user", "content": "who is lucy"},
        {"role": "assistant", "content": ANSWER},
    ]
    await manager.tasks.cancel_all()


async def test_the_web_search_envelope_reports_replying() -> None:
    manager = ConnectionManager()
    llm = StreamingLLM()
    llm.web_search_enabled = True
    engine = make_engine(manager, llm=llm)
    socket = await connected(manager)

    engine.ask("who is lucy")
    await finished(manager)
    await delivered()

    searching = [m for m in socket.messages("prompt_envelope") if m["context"].startswith("Searching web")]
    assert [m["state"] for m in searching] == ["REPLYING"]
    await manager.tasks.cancel_all()


# --- cancellation: never an error, always a closed stream ----------------------------------------------------------------


@pytest.mark.parametrize("cause", ["barge_in", "pause", "replacement", "shutdown"])
async def test_a_reply_cut_short_closes_its_stream_and_reports_no_error(
    cause: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = ConnectionManager()
    llm = StreamingLLM()
    engine = make_engine(manager, llm=llm)
    socket = await connected(manager)
    blocked = block_after_first_token(manager, monkeypatch)
    engine.ask("who is lucy")
    first = manager.tasks.get("reply")
    assert first is not None
    async with asyncio.timeout(5):
        await blocked.wait()

    if cause == "barge_in":
        manager.trigger_barge_in()
    elif cause == "pause":
        engine.pause_narration()
    elif cause == "replacement":
        engine.ask("and her friend")
    else:
        await manager.tasks.cancel_all()
    await asyncio.wait([first], timeout=5)

    assert first.cancelled()
    assert llm.closed  # closed by the reply itself, not left for the garbage collector
    await delivered()
    assert socket.messages("reply_error") == []
    expected_state = {"barge_in": "LISTENING", "pause": "PAUSED", "replacement": "REPLYING", "shutdown": "REPLYING"}
    assert manager.state == expected_state[cause]
    if cause != "replacement":
        assert manager.tasks.get("reply") is None
    await manager.tasks.cancel_all()


# --- starting a reply: synchronous, and the newest turn wins -------------------------------------------------------------


async def test_ask_has_taken_over_the_session_when_it_returns() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine = make_engine(manager, llm=BlockingLLM(), tts=tts)
    await start_narrating(engine, tts)

    engine.ask("who is lucy")

    assert manager.state == "REPLYING"
    assert manager.tasks.get("reply") is not None
    assert manager.tasks.get("narration") is None
    await manager.tasks.cancel_all()


async def test_ask_drops_a_transcription_that_is_still_pending() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager, llm=BlockingLLM())
    manager.session.enter_listening("barge_in")
    pending = manager.tasks.spawn(asyncio.sleep(3600), "stt")
    await asyncio.sleep(0)

    engine.ask("typed question")
    await asyncio.wait([pending], timeout=5)

    assert pending.cancelled()
    await manager.tasks.cancel_all()


async def test_ask_called_from_inside_the_stt_task_does_not_cancel_that_task() -> None:
    """The transcript is delivered by the stt task itself. Its own ask() must not stop it at the next await."""
    manager = ConnectionManager()
    engine = make_engine(manager, llm=BlockingLLM())
    manager.session.enter_listening("barge_in")

    async def deliver() -> str:
        engine.ask("spoken question")
        await asyncio.sleep(0)  # the next suspension point, where a self-cancel would land
        return "delivered"

    task = manager.tasks.spawn(deliver(), "stt")

    async with asyncio.timeout(5):
        assert await task == "delivered"
    assert manager.state == "REPLYING"
    await manager.tasks.cancel_all()


async def test_a_barge_in_drops_a_transcription_that_is_still_pending() -> None:
    manager = ConnectionManager()
    make_engine(manager)
    manager.session.enter_listening("barge_in")
    pending = manager.tasks.spawn(asyncio.sleep(3600), "stt")
    await asyncio.sleep(0)

    manager.trigger_barge_in()  # the user starts a newer turn
    await asyncio.wait([pending], timeout=5)

    assert pending.cancelled()
    await manager.tasks.cancel_all()


async def test_announcing_a_barge_in_does_not_change_the_state() -> None:
    manager = ConnectionManager()
    seen: list[tuple[int | None, float | None]] = []
    manager.register_barge_in_callback(lambda flushed, latency: seen.append((flushed, latency)))
    generation = manager.generation

    event = manager.announce_barge_in(flushed_samples=5, latency_ms=0.5)  # a caller that did measure
    unmeasured = manager.announce_barge_in()

    assert manager.state == "PAUSED"
    assert manager.generation == generation + 2
    assert event.generation == generation + 1
    assert (event.flushed_samples, event.latency_ms) == (5, 0.5)
    assert unmeasured.flushed_samples is None and unmeasured.latency_ms is None  # no constant stands in for a measurement
    assert manager.get_state_sync_message().latency_ms is None
    assert seen == [(5, 0.5), (None, None)]


# --- the order the clients see ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("resting", ["NARRATING", "PAUSED"])
async def test_a_question_reaches_clients_as_barge_in_then_replying_then_the_envelope(resting: str) -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine = make_engine(manager, tts=tts, settings=replace(Settings.fast(), followup_window_s=30.0))
    socket = await connected(manager)
    if resting == "NARRATING":
        await start_narrating(engine, tts)

    engine.ask("who is lucy")
    await finished(manager)
    await delivered()

    kinds = ("barge_in", "state_sync", "prompt_envelope")
    order = [message for message in socket.seen() if message[0] in kinds]
    order = order[order.index(("barge_in", "")):]
    assert order[:3] == [("barge_in", ""), ("state_sync", "REPLYING"), ("prompt_envelope", "REPLYING")]
    last_replying = max(i for i, message in enumerate(order) if message == ("prompt_envelope", "REPLYING"))
    followup = order.index(("state_sync", "LISTENING"))
    assert followup > last_replying  # no LISTENING flicker while the answer is being produced
    assert order.index(("prompt_envelope", "LISTENING")) > followup  # the window's envelope follows its state_sync
    await manager.tasks.cancel_all()


async def test_the_reply_ends_in_an_open_followup_window_with_an_empty_pre_roll() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager, settings=replace(Settings.fast(), followup_window_s=30.0))
    acc = SpeechTurnAccumulator()
    manager.accumulator = acc
    chunk = (np.sin(np.arange(320) / 5) * 8000).astype(np.int16).tobytes()
    acc.disarm()
    acc.feed(chunk)  # heard while narrating: it sits in the pre-roll
    manager.trigger_barge_in()
    assert acc._pre_roll_buffer  # a barge-in keeps it: the first word lives there

    engine.ask("who is lucy")
    reply = await finished(manager)

    assert reply.exception() is None
    assert manager.state == "LISTENING"  # the reply task has ended; nothing waits inside it
    assert acc._pre_roll_buffer == []  # the window starts clean: the answer's own echo is not the next question
    await manager.tasks.cancel_all()


@pytest.mark.parametrize("start", ["ask", "report_unrecognized"])
async def test_a_reply_cancelled_before_its_first_step_leaves_no_unawaited_coroutine(start: str) -> None:
    manager = ConnectionManager()
    engine = make_engine(manager)
    manager.trigger_barge_in()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if start == "ask":
            engine.ask("who is lucy")
        else:
            engine.report_unrecognized()
        engine.pause_narration()  # in the same loop pass: the reply task has not taken its first step
        await manager.tasks.cancel_all()
        await asyncio.sleep(0)  # the loop lets go of the finished task (and with it the coroutine it never started)
        gc.collect()  # an abandoned coroutine is reported when it is collected

    assert [str(w.message) for w in caught if "never awaited" in str(w.message)] == []


# --- unrecognized speech -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("resting", ["NARRATING", "PAUSED"])
async def test_unrecognized_speech_gets_the_neutral_line_and_the_session_resumes(resting: str) -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine = make_engine(manager, tts=tts)
    if resting == "NARRATING":
        await start_narrating(engine, tts)
    manager.trigger_barge_in()

    engine.report_unrecognized()
    assert manager.state == "REPLYING"
    await finished(manager)

    assert UNRECOGNIZED_LINE in tts.texts
    assert manager.state == resting
    await manager.tasks.cancel_all()


async def test_a_tts_failure_while_reporting_unrecognized_speech_still_resumes() -> None:
    manager = ConnectionManager()
    tts = FailingTts(fail_on={UNRECOGNIZED_LINE, REPLY_ERROR_LINE})
    engine = make_engine(manager, tts=tts)
    socket = await connected(manager)
    manager.trigger_barge_in()

    engine.report_unrecognized()
    await finished(manager)
    await delivered()

    assert [m["detail"] for m in socket.messages("reply_error")] == ["RuntimeError"]
    assert manager.state == "PAUSED"


# --- the resume cue ------------------------------------------------------------------------------------------------


def resumed_after(socket: RecordingSocket, mark: int) -> list[tuple[str, str]] | None:
    """What the socket saw from the NARRATING state_sync that follows `mark`, or None before it arrives."""
    seen = socket.seen()[mark:]
    if ("state_sync", "NARRATING") not in seen:
        return None
    return seen[seen.index(("state_sync", "NARRATING")):]


@pytest.mark.parametrize("how", ["answer", "unrecognized", "deadline"])
async def test_narration_restarted_after_an_interruption_cues_before_its_first_cursor_update(how: str) -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine = make_engine(manager, tts=tts, settings=replace(Settings.fast(), barge_in_listen_s=0.02))
    socket = await connected(manager)
    await start_narrating(engine, tts)
    await socket.wait_until(lambda: ("cursor_update", "") in socket.seen())
    mark = len(socket.texts)

    if how == "answer":
        engine.ask("who is lucy")
    elif how == "unrecognized":
        manager.trigger_barge_in()
        engine.report_unrecognized()
    else:
        manager.trigger_barge_in()  # nobody speaks: the deadline puts narration back
    await socket.wait_until(lambda: (after := resumed_after(socket, mark)) is not None and ("cursor_update", "") in after)

    after = resumed_after(socket, mark)
    assert after is not None
    assert after.index(("prompt_envelope", "NARRATING")) < after.index(("cursor_update", ""))
    await manager.tasks.cancel_all()


@pytest.mark.parametrize("how", ["answer", "unrecognized"])
async def test_nothing_is_cued_when_the_session_goes_back_to_paused(how: str) -> None:
    manager = ConnectionManager()
    engine = make_engine(manager)
    socket = await connected(manager)

    if how == "answer":
        engine.ask("who is lucy")
    else:
        manager.trigger_barge_in()
        engine.report_unrecognized()
    await finished(manager)
    await wait_for_state(manager.session, "PAUSED")
    await delivered()

    assert ("prompt_envelope", "NARRATING") not in socket.seen()
    await manager.tasks.cancel_all()


async def test_starting_from_paused_is_not_cued() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine = make_engine(manager, tts=tts)
    socket = await connected(manager)

    engine.start_narration()
    await socket.wait_until(lambda: ("cursor_update", "") in socket.seen())

    assert ("prompt_envelope", "NARRATING") not in socket.seen()
    await manager.tasks.cancel_all()


# --- a resume while a reply is in progress -------------------------------------------------------------------------------


@pytest.mark.parametrize("resting", ["NARRATING", "PAUSED"])
async def test_a_resume_during_a_reply_lets_it_finish_and_narration_follows_it(
    resting: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Every tab sends resume when its socket opens, so a second tab must not cut off an answer or talk over it. The
    resume is not lost either, even from a paused session: the reply ends in narration (#90)."""
    manager = ConnectionManager()
    llm = BlockingLLM()
    tts = FakeTts()
    engine = make_engine(manager, llm=llm, tts=tts)
    if resting == "NARRATING":
        await start_narrating(engine, tts)
    engine.ask("who is lucy")
    reply = manager.tasks.get("reply")
    assert reply is not None
    async with asyncio.timeout(5):
        await llm.started.wait()
    states: list[str] = []
    manager.session.add_listener(lambda old, new, why: states.append(new))

    with caplog.at_level(logging.DEBUG, logger="voice_agent.server.narration"):
        engine.start_narration()

    assert states == []  # no transition
    assert manager.state == "REPLYING"
    assert manager.session.resume_to == "NARRATING"  # whatever the session was doing before the question
    assert manager.tasks.get("narration") is None  # nothing narrates under the answer
    assert manager.tasks.get("reply") is reply  # and the answer is not cut off
    assert any("reply" in record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG)

    llm.release.set()
    await asyncio.wait([reply], timeout=5)
    assert not reply.cancelled() and reply.exception() is None
    await wait_for_state(manager.session, "NARRATING")
    assert states == ["LISTENING", "NARRATING"]  # the follow-up window, then narration
    assert manager.tasks.get("narration") is not None
    await manager.tasks.cancel_all()


async def test_a_resume_while_listening_still_narrates() -> None:
    manager = ConnectionManager()
    tts = FakeTts()
    engine = make_engine(manager, tts=tts)
    await start_narrating(engine, tts)
    manager.trigger_barge_in()
    assert manager.state == "LISTENING"

    engine.start_narration()  # how a user stops waiting

    assert manager.state == "NARRATING"
    assert manager.tasks.get("narration") is not None
    await manager.tasks.cancel_all()


def test_the_endpoints_ignore_a_resume_during_a_reply_and_report_the_state() -> None:
    manager = ConnectionManager()
    app = create_app(
        connection_manager=manager, settings=Settings.fast(), llm_client=BlockingLLM(), tts=FakeTts(), search=FakeSearch()
    )

    def state_syncs_until_cadence(ws: Any, cadence: float) -> list[dict[str, Any]]:
        """The state snapshots up to the one a set_cadence marker produces (audio frames are skipped)."""
        ws.send_json({"type": "set_cadence", "cadence": cadence})
        snapshots: list[dict[str, Any]] = []
        for _ in range(500):
            frame = ws.receive()
            if "text" not in frame:
                continue
            message = json.loads(frame["text"])
            if message["type"] == "state_sync":
                snapshots.append(message)
                if message["cadence"] == cadence:
                    return snapshots
        raise AssertionError("the marker never arrived")

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "playback_control", "action": "resume"})  # narrating, so a resume would be legal
        assert wait_until(lambda: manager.state == "NARRATING")
        assert client.post("/api/ask", json={"query": "who is lucy"}).status_code == 200
        assert wait_until(lambda: manager.state == "REPLYING")
        state_syncs_until_cadence(ws, 1.3)  # drain everything up to here

        rest = client.post("/api/narration/control", json={"action": "resume"})
        assert rest.status_code == 200
        assert rest.json()["state"] == "REPLYING"

        ws.send_json({"type": "playback_control", "action": "resume"})
        snapshots = state_syncs_until_cadence(ws, 1.7)  # everything before the marker has been answered

        assert manager.state == "REPLYING"
        assert manager.tasks.get("narration") is None
        assert len(snapshots) >= 2  # the resume was answered with the real state, before the marker's own
        assert {snapshot["state"] for snapshot in snapshots} == {"REPLYING"}


# --- no usable model answer: the book is read instead -------------------------------------------------------------------

BOOK = [
    "Arasaka is a megacorporation headquartered in Night City with towers of glass.",
    "Lucy walked home along the wet pavement of the old market street.",
    "David repaired the broken shuttle engine until the morning light came.",
    "Nobody spoke about the weather at the harbour that quiet evening.",
    "The ferry left at dawn carrying lanterns, crates, and a sleeping dog.",
    "A violin played somewhere above the stairwell as the lamps flickered.",
]
NOT_AUTHORITATIVE = "who is lucy"  # matches, but weakly: the web is searched too
AUTHORITATIVE = "megacorporation arasaka headquartered"
NO_MATCH = "zzzz qqqq"


def book_engine(llm: Any, tts: FakeTts, search: FakeSearch | None = None) -> NarrationEngine:
    return NarrationEngine(
        ConnectionManager(),
        llm_client=llm,
        initial_sentences=BOOK,
        settings=Settings.fast(),
        tts=tts,
        search=search if search is not None else FakeSearch(),
    )


async def ask_and_wait(engine: NarrationEngine, query: str) -> PlayerSocket:
    socket = await connected(engine.manager)
    engine.ask(query)
    await finished(engine.manager)
    await delivered()
    return socket


def closest_passage(engine: NarrationEngine, query: str) -> str:
    return engine.search_document_passages(query, 1, current_sentence_idx=engine.current_sentence_idx)[0]


@pytest.mark.parametrize("status", ["offline", "auth", "error"])
async def test_with_no_model_answer_the_closest_passage_is_read(status: str) -> None:
    tts = FakeTts()
    llm = StreamingLLM(tokens=())
    llm.status = status
    engine = book_engine(llm, tts)

    socket = await ask_and_wait(engine, NOT_AUTHORITATIVE)

    spoken = "From the text: " + closest_passage(engine, NOT_AUTHORITATIVE)
    assert tts.texts == [spoken]
    assert [m["reply"] for m in socket.messages("prompt_envelope") if m["state"] == "LISTENING"] == [spoken]  # and it is shown
    assert socket.messages("state_sync")[-1]["llm_status"] == status
    assert engine._conversation_history == []
    assert len(llm.calls) == 1  # the model was asked first
    await engine.manager.tasks.cancel_all()


async def test_with_no_model_answer_and_no_passage_the_offline_line_is_spoken() -> None:
    tts = FakeTts()
    llm = StreamingLLM(tokens=())
    llm.status = "offline"
    engine = book_engine(llm, tts)

    await ask_and_wait(engine, NO_MATCH)

    assert tts.texts == [OFFLINE_LINE]  # never "From the text: " with nothing after it
    assert engine._conversation_history == []
    await engine.manager.tasks.cancel_all()


async def test_the_no_model_provider_reads_the_passage_without_a_request() -> None:
    tts = FakeTts()
    search = FakeSearch()
    llm = StreamingLLM()
    llm.provider = "none"
    llm.web_search_enabled = True
    engine = book_engine(llm, tts, search)

    socket = await ask_and_wait(engine, NOT_AUTHORITATIVE)

    assert tts.texts == ["From the text: " + closest_passage(engine, NOT_AUTHORITATIVE)]
    assert llm.calls == []
    assert search.calls == []  # nothing would read the web results
    assert socket.messages("state_sync")[-1]["llm_status"] == "none"
    assert engine._conversation_history == []
    await engine.manager.tasks.cancel_all()


async def test_the_no_model_provider_with_no_passage_speaks_the_offline_line() -> None:
    tts = FakeTts()
    llm = StreamingLLM()
    llm.provider = "none"
    engine = book_engine(llm, tts)

    await ask_and_wait(engine, NO_MATCH)

    assert tts.texts == [OFFLINE_LINE]
    await engine.manager.tasks.cancel_all()


async def test_a_broken_stream_speaks_the_partial_text_and_is_not_remembered() -> None:
    tts = FakeTts()
    llm = StreamingLLM(tokens=("The ferry ", "left at"))
    llm.status = "interrupted"
    engine = book_engine(llm, tts)

    socket = await ask_and_wait(engine, NOT_AUTHORITATIVE)

    assert tts.texts == ["The ferry left at"]  # what arrived, with nothing appended
    assert engine._conversation_history == []
    assert socket.messages("state_sync")[-1]["llm_status"] == "interrupted"
    await engine.manager.tasks.cancel_all()


async def test_only_answers_the_model_finished_are_remembered() -> None:
    tts = FakeTts()
    llm = StreamingLLM(tokens=("A real answer.",))
    engine = book_engine(llm, tts)
    await ask_and_wait(engine, NOT_AUTHORITATIVE)
    assert engine._conversation_history == [
        {"role": "user", "content": NOT_AUTHORITATIVE},
        {"role": "assistant", "content": "A real answer."},
    ]

    # The model then goes away: the next question is answered from the book and leaves the history as it was.
    llm.tokens, llm.status = [], "offline"
    engine.ask(AUTHORITATIVE)
    await finished(engine.manager)

    assert len(engine._conversation_history) == 2
    assert OFFLINE_LINE not in str(engine._conversation_history)
    await engine.manager.tasks.cancel_all()


async def test_the_prompt_holds_the_fenced_passages_and_web_results_with_the_question_last() -> None:
    llm = StreamingLLM()
    llm.web_search_enabled = True
    search = FakeSearch("- A web page: Lucy is a netrunner.")
    engine = book_engine(llm, FakeTts(), search)

    await ask_and_wait(engine, NOT_AUTHORITATIVE)

    user = llm.calls[0][-1]["content"]
    assert '<book_passages authoritative="false">' in user
    assert closest_passage(engine, NOT_AUTHORITATIVE) in user
    assert "<web_results>" in user and "Lucy is a netrunner." in user
    assert user.rstrip().endswith("Question: " + NOT_AUTHORITATIVE)
    assert llm.calls[0][0]["role"] == "system"
    assert search.calls == [(NOT_AUTHORITATIVE, 3)]
    await engine.manager.tasks.cancel_all()


async def test_an_authoritative_passage_skips_the_web() -> None:
    llm = StreamingLLM()
    llm.web_search_enabled = True
    search = FakeSearch()
    engine = book_engine(llm, FakeTts(), search)

    await ask_and_wait(engine, AUTHORITATIVE)

    user = llm.calls[0][-1]["content"]
    assert '<book_passages authoritative="true">' in user
    assert "<web_results>" not in user
    assert search.calls == []
    await engine.manager.tasks.cancel_all()


async def test_the_question_reaches_the_web_search_and_the_book_unchanged() -> None:
    """No world is assumed: nothing is appended to the web query, and a misheard name is not mapped to a character."""
    llm = StreamingLLM()
    llm.web_search_enabled = True
    search = FakeSearch()
    engine = book_engine(llm, FakeTts(), search)

    await ask_and_wait(engine, "who is johnny")

    assert search.calls == [("who is johnny", 3)]
    assert engine.search_document_passages("who is in arakas tower") == []  # "arakas" is not rewritten to "arasaka"
    await engine.manager.tasks.cancel_all()


async def test_the_history_kept_is_the_history_sent() -> None:
    """One window: what the engine remembers is exactly the messages `build_messages` puts in front of the model."""
    llm = StreamingLLM(tokens=("An answer.",))
    engine = book_engine(llm, FakeTts())
    for number in range(4):
        await ask_and_wait(engine, f"question number {number}")
    remembered = list(engine._conversation_history)
    await ask_and_wait(engine, "question number 4")

    assert len(remembered) == HISTORY_MESSAGES  # four exchanges are eight messages; only the last six are kept
    assert llm.calls[-1][1:-1] == remembered  # between the system prompt and the final user message
    assert [m["content"] for m in remembered if m["role"] == "user"] == [f"question number {n}" for n in (1, 2, 3)]
    assert len(engine._conversation_history) == HISTORY_MESSAGES
    await engine.manager.tasks.cancel_all()
