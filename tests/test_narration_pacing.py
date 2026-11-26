"""Narration stays one sentence ahead of the listener, the cursor follows the browser's reports, and an interrupted
sentence starts again a few words before the stop point (D7, N16)."""

import asyncio
import dataclasses
import json
import logging
from collections.abc import Callable
from typing import Any

import pytest

from fakes import FailingTts, FakeLLM, FakeSearch, FakeTts, PlayerSocket, wait_for_state
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.narration import NarrationEngine
from voice_agent.settings import Settings

WORDS = 15


def sentence(index: int) -> str:
    return " ".join(f"s{index}w{word}" for word in range(WORDS))


def tail(index: int, first_word: int) -> str:
    return " ".join(f"s{index}w{word}" for word in range(first_word, WORDS))


BOOK = [sentence(index) for index in range(5)]


class GateTts(FakeTts):
    """A synthesizer that holds the call for `blocked_text` until released, and says when it has been reached."""

    def __init__(self, blocked_text: str) -> None:
        super().__init__()
        self.blocked_text = blocked_text
        self.reached = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, text: str, voice_id: str, cadence: float) -> tuple[bytes, str]:
        if text == self.blocked_text:
            self.reached.set()
            await self.release.wait()
        return await super().__call__(text, voice_id, cadence)


class SelectiveTts(FakeTts):
    """No audio for the texts in `silent`: every synthesizer failed for those."""

    def __init__(self, silent: set[str]) -> None:
        super().__init__()
        self.silent = silent

    async def __call__(self, text: str, voice_id: str, cadence: float) -> tuple[bytes | None, str]:  # type: ignore[override]
        if text in self.silent:
            self.calls.append((text, voice_id, cadence))
            return None, "none"
        return await super().__call__(text, voice_id, cadence)


def make_engine(
    manager: ConnectionManager,
    *,
    tts: Any = None,
    sentences: list[str] | None = None,
    settings: Settings | None = None,
) -> NarrationEngine:
    return NarrationEngine(
        manager,
        llm_client=FakeLLM(),
        settings=settings if settings is not None else Settings.fast(),
        tts=tts if tts is not None else FakeTts(),
        search=FakeSearch(),
        initial_sentences=BOOK if sentences is None else sentences,
    )


# An estimate so long that the cursor stays where the test puts it: the words step every 66 seconds.
SLOW_WORDS = dataclasses.replace(Settings.fast(), min_sentence_s=1000.0)


async def tab(manager: ConnectionManager) -> PlayerSocket:
    """A tab that plays clips and reports them only when the test says so."""
    socket = PlayerSocket(manager)
    await manager.connect(socket)  # type: ignore[arg-type]
    return socket


async def until(condition: Callable[[], bool], yields: int = 5000) -> None:
    """Let the event loop run until `condition()` holds. Counted in turns of the loop, not in seconds."""
    for _ in range(yields):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("the condition never held")


async def settle() -> None:
    for _ in range(300):
        await asyncio.sleep(0)


def cursor(socket: PlayerSocket) -> list[tuple[int, int]]:
    return [(m["sentence_idx"], m["word_offset"]) for m in socket.messages("cursor_update")]


# --- one sentence ahead, and no more (N16) --------------------------------------------------------------------------------


async def test_never_more_than_two_clips_are_out_and_a_third_waits_for_the_first_report() -> None:
    manager = ConnectionManager()
    listener = await tab(manager)
    tts = FakeTts()
    engine = make_engine(manager, tts=tts)

    engine.start_narration()
    await listener.wait_for_clips(2)
    await until(lambda: len(tts.calls) == 3)  # the third is made while the first plays
    await settle()  # and has the chance to be sent, which it must not be

    assert len(listener.clips) == 2  # the one playing and the one queued behind it
    assert len(tts.calls) == 3  # the third is already made, and held back until there is room for it

    outstanding = []
    for expected in range(3, 9):
        await listener.ack()
        await listener.wait_for_clips(expected)
        outstanding.append(len(listener.unacked))
    assert max(outstanding) == 2
    await manager.tasks.cancel_all()


async def test_the_cursor_is_on_the_sentence_that_is_playing_and_moves_on_the_report() -> None:
    manager = ConnectionManager()
    listener = await tab(manager)
    engine = make_engine(manager)

    engine.start_narration()
    await listener.wait_for_clips(2)
    await until(lambda: engine.current_word_idx == WORDS - 1)
    await settle()  # and has the chance to move on, which it must not do

    assert {sentence_idx for sentence_idx, _ in cursor(listener)} == {0}  # sentence 1 is queued, not playing
    assert (engine.current_sentence_idx, engine.current_word_idx) == (0, WORDS - 1)  # held on its last word

    await listener.ack()
    await listener.wait_until(lambda: (1, WORDS - 1) in cursor(listener))
    assert {sentence_idx for sentence_idx, _ in cursor(listener)} == {0, 1}  # not yet 2: it has not started playing
    assert engine.current_sentence_idx == 1
    await manager.tasks.cancel_all()


async def test_the_book_wraps_and_a_sentence_is_not_skipped_or_spoken_twice() -> None:
    manager = ConnectionManager()
    listener = await tab(manager)
    tts = FakeTts()
    engine = make_engine(manager, tts=tts, sentences=BOOK[:3])

    engine.start_narration()
    await listener.wait_for_clips(2)
    for expected in range(3, 8):
        await listener.ack()
        await listener.wait_for_clips(expected)

    assert tts.texts[:7] == [BOOK[0], BOOK[1], BOOK[2], BOOK[0], BOOK[1], BOOK[2], BOOK[0]]
    assert all(clip.data for clip in listener.clips)
    await manager.tasks.cancel_all()


async def test_a_one_sentence_book_goes_round_without_trouble() -> None:
    manager = ConnectionManager()
    listener = await tab(manager)
    tts = FakeTts()
    engine = make_engine(manager, tts=tts, sentences=["Only one sentence here."])

    engine.start_narration()
    await listener.wait_for_clips(2)
    for expected in range(3, 6):
        await listener.ack()
        await listener.wait_for_clips(expected)

    assert set(tts.texts) == {"Only one sentence here."}
    assert engine.current_sentence_idx == 0 and manager.state == "NARRATING"
    await manager.tasks.cancel_all()


async def test_a_voice_or_cadence_change_applies_from_the_next_sentence_not_yet_made() -> None:
    manager = ConnectionManager()
    listener = await tab(manager)
    tts = FakeTts()
    engine = make_engine(manager, tts=tts)

    engine.start_narration()
    await listener.wait_for_clips(2)
    await until(lambda: len(tts.calls) == 3)  # sentences 0 and 1 are out and sentence 2 is made
    manager.set_cadence(1.5)
    manager.set_voice("lucy-kushinada")
    await listener.ack()
    await listener.wait_for_clips(3)
    await until(lambda: len(tts.calls) == 4)  # the sentence after the one just sent is made with the new voice

    assert [(call[1], call[2]) for call in tts.calls[:4]] == [
        ("david-martinez", 1.0),
        ("david-martinez", 1.0),
        ("david-martinez", 1.0),
        ("lucy-kushinada", 1.5),
    ]
    await manager.tasks.cancel_all()


async def test_the_pause_between_sentences_is_the_browsers_to_add() -> None:
    manager = ConnectionManager()
    listener = await tab(manager)
    settings = dataclasses.replace(Settings.fast(), inter_sentence_pause_s=0.35)
    engine = make_engine(manager, settings=settings)

    engine.start_narration()
    await listener.wait_for_clips(2)

    assert [clip.gap_s for clip in listener.clips] == [0.0, 0.35]  # none before the first clip, the pause before the next
    await manager.tasks.cancel_all()


# --- resuming a sentence that was partly spoken (D7) ------------------------------------------------------------------------


async def narrating_at_word_ten() -> tuple[ConnectionManager, PlayerSocket, FakeTts, NarrationEngine]:
    manager = ConnectionManager()
    listener = await tab(manager)
    tts = FakeTts()
    engine = make_engine(manager, tts=tts, settings=SLOW_WORDS)
    engine.start_narration()
    await listener.wait_for_clips(2)
    engine.current_word_idx = 10  # the words step every 66 seconds here: this is where the listener has got to
    manager.update_cursor(sentence_idx=0, word_offset=10)
    return manager, listener, tts, engine


async def test_a_barge_in_at_word_ten_resumes_at_word_six_with_the_right_text() -> None:
    manager, listener, tts, engine = await narrating_at_word_ten()
    made = len(tts.calls)

    manager.trigger_barge_in()
    manager.session.resume("deadline")  # nobody spoke
    await tts.wait_for_calls(made + 1)

    assert tts.texts[made] == tail(0, 6)  # four words back, then the rest of the sentence
    await listener.wait_until(lambda: (0, 6) in cursor(listener)[-3:])
    assert (engine.current_sentence_idx, engine.current_word_idx) == (0, 6)
    await manager.tasks.cancel_all()


async def test_a_pause_and_resume_does_the_same() -> None:
    manager, listener, tts, engine = await narrating_at_word_ten()
    made = len(tts.calls)

    engine.pause_narration()
    engine.start_narration()
    await tts.wait_for_calls(made + 1)

    assert tts.texts[made] == tail(0, 6)
    await listener.wait_until(lambda: (0, 6) in cursor(listener)[-3:])
    await manager.tasks.cancel_all()


async def test_a_new_owner_hears_the_sentence_again_from_the_resume_point() -> None:
    manager, listener, tts, engine = await narrating_at_word_ten()
    newcomer = await tab(manager)
    made = len(tts.calls)

    manager.claim_ownership(newcomer, explicit=True)  # type: ignore[arg-type]
    await newcomer.wait_for_clips(1)

    assert tts.texts[made] == tail(0, 6)
    assert newcomer.clips[0].generation == manager.generation
    await manager.tasks.cancel_all()


class DroppedAtWord(PlayerSocket):
    """A tab whose connection fails while the narration loop is sending it the cursor at `word_offset` of sentence 0:
    the send is made inside the loop's own task."""

    def __init__(self, manager: ConnectionManager, word_offset: int) -> None:
        super().__init__(manager)
        self.word_offset = word_offset

    async def send_text(self, text: str) -> None:
        message = json.loads(text)
        if message["type"] == "cursor_update" and (message["sentence_idx"], message["word_offset"]) == (0, self.word_offset):
            raise ConnectionResetError("the tab went away")
        await super().send_text(text)


async def test_narration_goes_on_for_the_next_tab_when_a_send_inside_the_loop_drops_the_owner() -> None:
    manager = ConnectionManager()
    owner = DroppedAtWord(manager, word_offset=5)
    await manager.connect(owner)  # type: ignore[arg-type]
    other = PlayerSocket(manager, auto_ack=True)
    await manager.connect(other)  # type: ignore[arg-type]
    tts = FakeTts()
    engine = make_engine(manager, tts=tts)

    engine.start_narration()
    await other.wait_for_clips(5)  # the loop that dropped the owner carried on for the tab that took over

    assert manager.owner is other and manager.state == "NARRATING"
    resumed = tts.texts.index(tail(0, 1))  # word 5 was being sent: four words back
    assert tts.texts[resumed : resumed + 3] == [tail(0, 1), BOOK[1], BOOK[2]]  # in order, none twice
    assert other.clips[0].generation == manager.generation  # sent after the stop, so not dropped by it
    loops = [task for task in asyncio.all_tasks() if task.get_name() == "narration" and not task.done()]
    assert len(loops) == 1  # the old loop is not running beside the new one
    await manager.tasks.cancel_all()


@pytest.mark.parametrize("direction", ["next", "prev"])
async def test_next_and_prev_start_at_word_zero(direction: str) -> None:
    manager, listener, tts, engine = await narrating_at_word_ten()
    made = len(tts.calls)

    if direction == "next":
        engine.next_sentence()
        expected = 1
    else:
        engine.previous_sentence()
        expected = len(BOOK) - 1  # wraps
    await tts.wait_for_calls(made + 1)

    assert tts.texts[made] == BOOK[expected]  # the whole sentence
    await listener.wait_until(lambda: (expected, 0) in cursor(listener))
    await manager.tasks.cancel_all()


async def test_a_question_asked_at_word_ten_returns_to_word_six() -> None:
    manager, listener, tts, engine = await narrating_at_word_ten()
    made = len(tts.calls)

    engine.ask("who is lucy")
    await listener.wait_until(lambda: bool(listener.messages("barge_in")))
    reply_clip_count = len(listener.clips) + 1
    await listener.wait_for_clips(reply_clip_count)
    await listener.ack(listener.clips[-1])  # the answer has played; the follow-up window is zero long in tests
    await wait_for_state(manager.session, "NARRATING")
    await tts.wait_for_calls(made + 2)

    assert tts.texts[made + 1] == tail(0, 6)  # after the answer's own line
    await manager.tasks.cancel_all()


# --- when the browser does not report, and when there is nobody to report ---------------------------------------------------


async def test_a_missing_report_pauses_narration_with_an_error(caplog: pytest.LogCaptureFixture) -> None:
    manager = ConnectionManager()
    listener = await tab(manager)
    engine = make_engine(manager, settings=dataclasses.replace(Settings.fast(), playback_ack_timeout_s=0.05))

    with caplog.at_level(logging.WARNING):
        engine.start_narration()
        await wait_for_state(manager.session, "PAUSED")

    assert [message["detail"] for message in listener.messages("error")] == ["playback stalled"]
    assert any("No playback_done" in record.getMessage() for record in caplog.records)
    await manager.tasks.cancel_all()
    assert len(listener.clips) <= 2  # it did not carry on reading the book to a tab that is not playing


async def test_with_nobody_to_report_narration_is_paced_by_the_estimate() -> None:
    manager = ConnectionManager()  # nobody connected
    tts = FakeTts()
    engine = make_engine(manager, tts=tts)

    engine.start_narration()
    await until(lambda: engine.current_sentence_idx >= 3 and tts.texts.count(BOOK[1]) >= 1)

    assert tts.texts[:4] == BOOK[:4]  # in order, none skipped
    engine.pause_narration()
    await manager.tasks.cancel_all()


async def test_with_nobody_to_report_a_long_estimate_keeps_the_cursor_on_the_sentence() -> None:
    manager = ConnectionManager()
    engine = make_engine(manager, settings=SLOW_WORDS)

    engine.start_narration()
    await settle()

    assert (engine.current_sentence_idx, engine.current_word_idx) == (0, 0)  # it does not run ahead of its estimate
    await manager.tasks.cancel_all()


async def test_a_sentence_no_synthesizer_could_make_is_paced_by_the_estimate() -> None:
    manager = ConnectionManager()
    listener = await tab(manager)
    engine = make_engine(manager, tts=SelectiveTts({BOOK[1]}))

    engine.start_narration()
    await listener.wait_for_clips(1)
    await settle()
    assert len(listener.clips) == 1  # sentence 1 has no clip, and sentence 2 is not sent ahead of it

    await listener.ack()  # sentence 0 played
    await listener.wait_for_clips(2)  # sentence 2, once sentence 1 has been paced by its estimate

    await listener.wait_until(lambda: (2, 0) in cursor(listener))
    assert cursor(listener).index((1, WORDS - 1)) < cursor(listener).index((2, 0))  # it went through the silent sentence
    await manager.tasks.cancel_all()


async def test_the_sentence_after_a_silent_one_is_not_sent_before_the_silent_one_has_been_paced() -> None:
    manager = ConnectionManager()
    listener = await tab(manager)
    engine = make_engine(manager, tts=SelectiveTts({BOOK[1]}), settings=SLOW_WORDS)  # the estimate is long

    engine.start_narration()
    await listener.wait_for_clips(1)
    await listener.ack()  # sentence 0 played; sentence 1 is silent and is now being paced by its estimate
    await listener.wait_until(lambda: (1, 0) in cursor(listener))
    await settle()

    assert len(listener.clips) == 1  # sentence 2 would be heard while the cursor is still on sentence 1
    await manager.tasks.cancel_all()


async def test_a_synthesizer_that_fails_is_reported_when_its_turn_comes() -> None:
    manager = ConnectionManager()
    listener = await tab(manager)
    engine = make_engine(manager, tts=FailingTts(fail_on={BOOK[2]}))

    engine.start_narration()
    await listener.wait_for_clips(2)
    await settle()
    assert manager.state == "NARRATING"  # sentences 0 and 1 are still to be heard

    await listener.ack()
    await listener.ack()
    await wait_for_state(manager.session, "PAUSED")

    assert {"type": "error", "detail": "narration stopped: RuntimeError"} in listener.messages("error")
    await manager.tasks.cancel_all()


# --- nothing is left behind -------------------------------------------------------------------------------------------------


def stray_tasks() -> list[str]:
    return [task.get_name() for task in asyncio.all_tasks() if task is not asyncio.current_task()]


@pytest.mark.parametrize("where", ["making the first sentence", "playing", "making the third sentence", "sending"])
async def test_stopping_at_any_await_leaves_no_task_and_no_clip_behind(where: str) -> None:
    manager = ConnectionManager()
    release = asyncio.Event()
    entered = asyncio.Event()

    class SlowSocket(PlayerSocket):
        async def send_bytes(self, data: bytes) -> None:
            entered.set()
            await release.wait()
            await super().send_bytes(data)

    listener: PlayerSocket
    if where == "sending":
        listener = SlowSocket(manager)
        await manager.connect(listener)  # type: ignore[arg-type]
    else:
        listener = await tab(manager)
    gate = GateTts(BOOK[0] if where == "making the first sentence" else BOOK[2] if where == "making the third sentence" else "")
    engine = make_engine(manager, tts=gate)

    engine.start_narration()
    if where == "sending":
        await entered.wait()
    elif where == "playing":
        await listener.wait_for_clips(2)
        await settle()
    else:
        await gate.reached.wait()

    engine.pause_narration()
    gate.release.set()
    release.set()
    await settle()
    await manager.tasks.cancel_all()
    await settle()

    assert stray_tasks() == []
    generation = manager.generation
    assert all(clip.generation < generation for clip in listener.clips)  # whatever got out is of a stopped generation


async def test_stopping_narration_releases_the_waiter_for_the_playing_clip() -> None:
    manager = ConnectionManager()
    listener = await tab(manager)
    engine = make_engine(manager)
    engine.start_narration()
    await listener.wait_for_clips(2)
    await settle()
    assert len(manager.playback) == 2

    engine.pause_narration()
    await settle()

    assert len(manager.playback) == 0
    await manager.tasks.cancel_all()
