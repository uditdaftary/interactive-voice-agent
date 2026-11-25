"""Clips go to the owner alone, each announced before it, and the server learns a clip has played from the browser (N19)."""

import asyncio
import dataclasses
import json
import logging
from typing import Any

import pytest

from fakes import FakeLLM, FakeSearch, FakeTts, PlayerSocket
from voice_agent.document.parser import DocumentContent
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.narration import NarrationEngine
from voice_agent.server.playback import PlaybackTracker
from voice_agent.settings import Settings

CLIP = b"\x01\x00" * 100


async def listeners(manager: ConnectionManager, count: int, *, auto_ack: bool = False) -> list[PlayerSocket]:
    sockets = [PlayerSocket(manager, auto_ack=auto_ack) for _ in range(count)]
    for socket in sockets:
        await manager.connect(socket)  # type: ignore[arg-type]
    return sockets


def make_engine(manager: ConnectionManager, *, tts: Any = None, settings: Settings | None = None) -> NarrationEngine:
    return NarrationEngine(
        manager,
        llm_client=FakeLLM(),
        settings=settings if settings is not None else Settings.fast(),
        tts=tts if tts is not None else FakeTts(),
        search=FakeSearch(),
        initial_sentences=["One two three.", "Four five six.", "Seven eight nine."],
    )


# --- the tracker on its own ---------------------------------------------------------------------------------------------


async def test_an_acknowledgement_from_the_connection_that_got_the_clip_resolves_the_wait() -> None:
    tracker = PlaybackTracker()
    connection = object()
    segment_id = tracker.register(3, connection)

    assert tracker.acknowledge(connection, segment_id, 3)

    assert await tracker.wait_done(segment_id, 1.0) is True
    assert len(tracker) == 0  # consumed


@pytest.mark.parametrize("wrong", ["id", "generation", "connection"])
async def test_a_report_that_does_not_match_the_clip_is_ignored(wrong: str) -> None:
    tracker = PlaybackTracker()
    connection = object()
    segment_id = tracker.register(3, connection)

    report_connection = object() if wrong == "connection" else connection
    report_id = segment_id + 1 if wrong == "id" else segment_id
    report_generation = 2 if wrong == "generation" else 3

    assert not tracker.acknowledge(report_connection, report_id, report_generation)
    assert await tracker.wait_done(segment_id, 0.01) is False  # still waiting, so it times out


async def test_a_report_for_a_clip_that_was_stopped_is_ignored() -> None:
    tracker = PlaybackTracker()
    connection = object()
    segment_id = tracker.register(0, connection)
    tracker.drop_all()

    assert not tracker.acknowledge(connection, segment_id, 0)


async def test_a_missing_report_times_out_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    tracker = PlaybackTracker()
    segment_id = tracker.register(0, object())

    with caplog.at_level(logging.WARNING, logger="voice_agent.server.playback"):
        assert await tracker.wait_done(segment_id, 0.01) is False

    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert f"segment {segment_id}" in caplog.records[0].getMessage()


async def test_stopping_releases_every_waiter_without_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    tracker = PlaybackTracker()
    first = tracker.register(0, object())
    second = tracker.register(0, object())
    waiting = [asyncio.ensure_future(tracker.wait_done(first, 30)), asyncio.ensure_future(tracker.wait_done(second, 30))]
    await asyncio.sleep(0)

    with caplog.at_level(logging.WARNING, logger="voice_agent.server.playback"):
        tracker.drop_all()
        results = await asyncio.wait_for(asyncio.gather(*waiting), 1.0)

    assert results == [False, False]
    assert caplog.records == []
    assert len(tracker) == 0


async def test_waiting_for_a_segment_nobody_sent_is_false_at_once() -> None:
    assert await PlaybackTracker().wait_done(99, 30) is False


# --- sending a clip -----------------------------------------------------------------------------------------------------


async def test_a_clip_is_announced_right_before_it_and_only_the_owner_gets_either() -> None:
    manager = ConnectionManager()
    owner, other = await listeners(manager, 2)

    segment_id = await manager.send_audio(CLIP, gap_s=0.35)

    assert segment_id is not None
    assert owner.frames == ["audio_segment", "blob"]
    assert owner.blobs == [CLIP]
    announcement = owner.messages("audio_segment")[0]
    assert announcement == {"type": "audio_segment", "segment_id": segment_id, "generation": 0, "gap_s": 0.35}
    assert other.frames == [] and other.blobs == []


async def test_segments_are_numbered_in_the_order_they_are_sent() -> None:
    manager = ConnectionManager()
    (owner,) = await listeners(manager, 1)

    first = await manager.send_audio(CLIP)
    second = await manager.send_audio(CLIP)

    assert first is not None and second is not None and second > first
    assert [clip.segment_id for clip in owner.clips] == [first, second]


async def test_with_nobody_connected_there_is_no_segment_to_wait_for() -> None:
    assert await ConnectionManager().send_audio(CLIP) is None


async def test_a_clip_is_sent_whole_even_if_the_caller_is_cancelled_while_sending() -> None:
    """An announcement without its blob would be paired with the next clip, which would then be taken for stale."""
    manager = ConnectionManager()
    release = asyncio.Event()
    entered = asyncio.Event()

    class SlowSocket(PlayerSocket):
        async def send_text(self, text: str) -> None:
            entered.set()
            await release.wait()
            await super().send_text(text)

    owner = SlowSocket(manager)
    await manager.connect(owner)  # type: ignore[arg-type]

    sender = asyncio.ensure_future(manager.send_audio(CLIP))
    await entered.wait()
    sender.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await sender
    await owner.wait_for_clips(1)

    assert owner.frames == ["audio_segment", "blob"]  # both, in order
    await manager.tasks.cancel_all()


async def test_the_owner_acknowledging_a_clip_ends_the_wait_and_other_reports_do_not() -> None:
    manager = ConnectionManager()
    owner, other = await listeners(manager, 2)
    segment_id = await manager.send_audio(CLIP)
    assert segment_id is not None
    clip = owner.clips[0]
    waiting = asyncio.ensure_future(manager.playback.wait_done(segment_id, 30))
    await asyncio.sleep(0)

    await other.ack(clip)  # the other tab reports a clip it was never sent
    await owner.ack(clip, segment_id=segment_id + 7)
    await owner.ack(clip, generation=clip.generation + 1)
    await asyncio.sleep(0)
    assert not waiting.done()

    await owner.ack(clip)
    assert await asyncio.wait_for(waiting, 1.0) is True


async def test_a_report_from_a_tab_that_is_not_the_owner_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    manager = ConnectionManager()
    owner, other = await listeners(manager, 2)
    segment_id = await manager.send_audio(CLIP)
    assert segment_id is not None
    clip = owner.clips[0]

    with caplog.at_level(logging.INFO, logger="voice_agent.server.connection_manager"):
        for _ in range(3):
            await other.ack(clip)

    lines = [record for record in caplog.records if "not the microphone owner" in record.getMessage()]
    assert len(lines) == 1
    assert "playback_done" in lines[0].getMessage()


# --- stopping ---------------------------------------------------------------------------------------------------------


async def test_stopping_tells_every_tab_with_the_new_generation_and_frees_the_waiters() -> None:
    manager = ConnectionManager()
    owner, other = await listeners(manager, 2)
    segment_id = await manager.send_audio(CLIP)
    assert segment_id is not None
    waiting = asyncio.ensure_future(manager.playback.wait_done(segment_id, 30))
    await asyncio.sleep(0)

    manager.stop_audio()

    assert manager.generation == 1
    assert await asyncio.wait_for(waiting, 1.0) is False
    await owner.wait_until(lambda: "audio_stop" in owner.frames)
    await other.wait_until(lambda: "audio_stop" in other.frames)  # the old owner of a hand-over must stop too
    assert owner.messages("audio_stop") == [{"type": "audio_stop", "generation": 1}]
    await manager.tasks.cancel_all()


async def test_the_stop_reaches_the_owner_before_the_first_clip_of_the_new_generation() -> None:
    manager = ConnectionManager()
    (owner,) = await listeners(manager, 1)
    await manager.send_audio(CLIP)

    manager.stop_audio()
    second = await manager.send_audio(CLIP)  # sent in the same loop pass, before the stop's own task has run

    assert second is not None
    assert owner.frames == ["audio_segment", "blob", "audio_stop", "audio_segment", "blob"]
    assert owner.clips[1].generation == 1
    await manager.tasks.cancel_all()


async def test_a_clip_handed_over_just_before_a_stop_is_never_sent() -> None:
    """The caller may be cancelled by the stop, but the send is shielded from that: the clip must not go out under the
    newer generation, where the browser would play it after the stop."""
    manager = ConnectionManager()
    (owner,) = await listeners(manager, 1)

    sender = asyncio.ensure_future(manager.send_audio(CLIP))
    await asyncio.sleep(0)  # the clip is handed over
    manager.stop_audio()

    assert await sender is None
    await owner.wait_until(lambda: "audio_stop" in owner.frames)
    await manager.tasks.cancel_all()
    assert owner.frames == ["audio_stop"]
    assert len(manager.playback) == 0


async def test_every_stop_is_announced_in_order() -> None:
    manager = ConnectionManager()
    (owner,) = await listeners(manager, 1)

    manager.stop_audio()
    manager.stop_audio()
    await owner.wait_until(lambda: len(owner.messages("audio_stop")) == 2)
    await manager.tasks.cancel_all()

    assert [message["generation"] for message in owner.messages("audio_stop")] == [1, 2]


# --- what stops the old clip -------------------------------------------------------------------------------------------


async def narrating(manager: ConnectionManager, engine: NarrationEngine, owner: PlayerSocket) -> None:
    engine.start_narration()
    await owner.wait_for_clips(1)


async def test_a_pause_stops_the_clip_that_is_playing() -> None:
    manager = ConnectionManager()
    (owner,) = await listeners(manager, 1)
    engine = make_engine(manager)
    await narrating(manager, engine, owner)
    generation = manager.generation

    engine.pause_narration()

    assert manager.generation > generation
    await owner.wait_until(lambda: "audio_stop" in owner.frames)
    assert owner.messages("audio_stop")[-1]["generation"] == manager.generation
    await manager.tasks.cancel_all()


async def test_a_question_stops_the_clip_that_is_playing() -> None:
    manager = ConnectionManager()
    (owner,) = await listeners(manager, 1)
    engine = make_engine(manager)
    await narrating(manager, engine, owner)
    first_generation = owner.clips[0].generation

    engine.ask("who is lucy")

    await owner.wait_until(lambda: bool(owner.messages("barge_in")))
    barge_in = owner.messages("barge_in")[-1]
    assert barge_in["generation"] > first_generation  # the event is the stop: everything older is dead
    await manager.tasks.cancel_all()


async def test_navigating_while_narrating_stops_the_queued_clips() -> None:
    manager = ConnectionManager()
    (owner,) = await listeners(manager, 1)
    engine = make_engine(manager)
    await narrating(manager, engine, owner)
    generation = manager.generation

    engine.next_sentence()

    assert manager.generation > generation
    await owner.wait_until(lambda: "audio_stop" in owner.frames)
    await manager.tasks.cancel_all()


async def test_navigating_while_an_answer_plays_does_not_stop_it() -> None:
    manager = ConnectionManager()
    (owner,) = await listeners(manager, 1)
    engine = make_engine(manager)
    manager.session.transition("REPLYING", "question")
    generation = manager.generation

    engine.next_sentence()

    assert manager.generation == generation
    assert owner.frames == []


async def test_loading_a_document_stops_whatever_is_playing() -> None:
    manager = ConnectionManager()
    (owner,) = await listeners(manager, 1)
    engine = make_engine(manager)
    manager.session.transition("REPLYING", "question")
    generation = manager.generation

    engine.load_document(DocumentContent(title="New", filename="new.txt", sentences=["A new book."], total_sentences=1, total_words=3))

    assert manager.generation > generation
    await owner.wait_until(lambda: "audio_stop" in owner.frames)
    await manager.tasks.cancel_all()


async def test_a_new_owner_hears_the_sentence_again_and_the_old_one_is_stopped() -> None:
    manager = ConnectionManager()
    old, new = await listeners(manager, 2)
    engine = make_engine(manager)
    await narrating(manager, engine, old)
    generation = manager.generation

    manager.claim_ownership(new, explicit=True)  # type: ignore[arg-type]

    assert manager.generation > generation
    await new.wait_for_clips(1)
    await old.wait_until(lambda: "audio_stop" in old.frames)  # the tab that lost the audio is told to stop
    assert new.clips[0].generation == manager.generation  # and the new owner's clips are of the new generation
    await manager.tasks.cancel_all()


async def test_a_new_owner_during_an_answer_ends_the_wait_for_its_clip() -> None:
    manager = ConnectionManager()
    old, new = await listeners(manager, 2)
    make_engine(manager)  # registers itself for owner changes
    manager.session.transition("REPLYING", "question")
    segment_id = await manager.send_audio(CLIP)
    assert segment_id is not None
    waiting = asyncio.ensure_future(manager.playback.wait_done(segment_id, 30))
    await asyncio.sleep(0)

    manager.claim_ownership(new, explicit=True)  # type: ignore[arg-type]

    assert await asyncio.wait_for(waiting, 1.0) is False
    assert manager.state == "REPLYING"  # the turn goes on
    await manager.tasks.cancel_all()


async def test_audio_messages_are_valid_protocol_messages() -> None:
    from voice_agent.server.schemas import (
        AudioSegmentMessage,
        AudioStopMessage,
        InvalidClientMessage,
        PlaybackDoneMessage,
        parse_client_message,
    )

    assert AudioSegmentMessage(segment_id=1, generation=2).gap_s == 0.0
    assert AudioStopMessage(generation=3).model_dump() == {"type": "audio_stop", "generation": 3}
    done = parse_client_message(json.dumps({"type": "playback_done", "segment_id": 4, "generation": 5}))
    assert isinstance(done, PlaybackDoneMessage) and (done.segment_id, done.generation) == (4, 5)
    for bad in (
        {"type": "playback_done"},
        {"type": "playback_done", "segment_id": -1, "generation": 0},
        {"type": "playback_done", "segment_id": 1, "generation": -1},
        {"type": "playback_done", "segment_id": "1", "generation": 0},
        {"type": "playback_done", "segment_id": True, "generation": 0},
        {"type": "playback_done", "segment_id": 1.5, "generation": 0},
    ):
        with pytest.raises(InvalidClientMessage):
            parse_client_message(json.dumps(bad))


# --- a reply waits for the browser, not for a guess -------------------------------------------------------------------


class SilentTts(FakeTts):
    """Every synthesizer failed: no audio at all."""

    async def __call__(self, text: str, voice_id: str, cadence: float) -> tuple[bytes | None, str]:  # type: ignore[override]
        self.calls.append((text, voice_id, cadence))
        return None, "none"


async def test_a_reply_ends_when_the_browser_says_it_has_played_the_answer() -> None:
    manager = ConnectionManager()
    (owner,) = await listeners(manager, 1)
    engine = make_engine(manager)

    engine.ask("who is lucy")
    reply = manager.tasks.get("reply")
    assert reply is not None
    await owner.wait_for_clips(1)
    await asyncio.sleep(0.05)  # plenty of time for a reply that did not wait

    assert not reply.done()  # still speaking: the browser has not said it is finished
    assert manager.state == "REPLYING"

    await owner.ack()
    await asyncio.wait([reply], timeout=5)
    assert reply.done() and reply.exception() is None
    assert manager.state != "REPLYING"  # the follow-up window, then back to what it was doing
    await manager.tasks.cancel_all()


async def test_a_reply_goes_on_after_the_timeout_when_the_report_never_comes(caplog: pytest.LogCaptureFixture) -> None:
    manager = ConnectionManager()
    await listeners(manager, 1)  # a tab that never reports
    engine = make_engine(manager, settings=dataclasses.replace(Settings.fast(), playback_ack_timeout_s=0.05))

    with caplog.at_level(logging.WARNING, logger="voice_agent.server.playback"):
        engine.ask("who is lucy")
        reply = manager.tasks.get("reply")
        assert reply is not None
        await asyncio.wait([reply], timeout=5)

    assert reply.done() and reply.exception() is None
    assert manager.state != "REPLYING"
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    await manager.tasks.cancel_all()


async def test_a_reply_with_nobody_connected_is_paced_by_the_estimate() -> None:
    manager = ConnectionManager()
    settings = dataclasses.replace(Settings.fast(), min_sentence_s=30.0)  # an estimate that is not about to pass
    engine = make_engine(manager, settings=settings)

    engine.ask("who is lucy")
    reply = manager.tasks.get("reply")
    assert reply is not None
    await asyncio.sleep(0.05)

    assert not reply.done() and manager.state == "REPLYING"  # nobody can say it is finished: it waits the estimate
    await manager.tasks.cancel_all()


async def test_a_reply_whose_audio_could_not_be_made_is_paced_by_the_estimate() -> None:
    manager = ConnectionManager()
    (owner,) = await listeners(manager, 1)
    engine = make_engine(manager, tts=SilentTts(), settings=dataclasses.replace(Settings.fast(), min_sentence_s=30.0))

    engine.ask("who is lucy")
    reply = manager.tasks.get("reply")
    assert reply is not None
    await asyncio.sleep(0.05)

    assert not reply.done() and owner.clips == []  # no clip, so no report: it waits the estimate, not an ack
    await manager.tasks.cancel_all()


async def test_the_line_for_speech_that_was_not_understood_waits_for_the_browser_too() -> None:
    manager = ConnectionManager()
    (owner,) = await listeners(manager, 1)
    engine = make_engine(manager)
    manager.trigger_barge_in()  # a turn that recognized nothing

    engine.report_unrecognized()
    reply = manager.tasks.get("reply")
    assert reply is not None
    await owner.wait_for_clips(1)
    await asyncio.sleep(0.05)
    assert not reply.done()

    await owner.ack()
    await asyncio.wait([reply], timeout=5)
    assert reply.done() and reply.exception() is None
    await manager.tasks.cancel_all()


async def test_a_reply_whose_clip_went_to_the_old_owner_goes_on_when_the_owner_changes() -> None:
    manager = ConnectionManager()
    old, new = await listeners(manager, 2)
    engine = make_engine(manager)

    engine.ask("who is lucy")
    reply = manager.tasks.get("reply")
    assert reply is not None
    await old.wait_for_clips(1)
    await asyncio.sleep(0.05)
    assert not reply.done()

    manager.claim_ownership(new, explicit=True)  # type: ignore[arg-type]

    await asyncio.wait([reply], timeout=5)  # no 30 s wait for a tab that is no longer playing the answer
    assert reply.done() and reply.exception() is None
    await manager.tasks.cancel_all()
