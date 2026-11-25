"""SessionState: legal transitions, resume_to, listeners and the LISTENING deadline."""

import asyncio
import logging

import pytest

from voice_agent.orchestrator.state_machine import SessionState


def make_session(listen_deadline_s: float = 8.0, followup_window_s: float = 2.5) -> SessionState:
    return SessionState(listen_deadline_s=listen_deadline_s, followup_window_s=followup_window_s)


def session_in(state: str, resting: str = "NARRATING") -> SessionState:
    """A session in `state`, reached by legal edges, with `resting` recorded as the state to resume to."""
    session = make_session()
    if state == "PAUSED" and resting == "PAUSED":
        return session
    assert session.transition("NARRATING" if resting == "NARRATING" else "PAUSED", "setup")
    if state in ("LISTENING", "REPLYING"):
        assert session.transition(state, "setup")
    elif state != resting:
        assert session.transition(state, "setup")
    return session


def test_initial_state_is_paused() -> None:
    session = make_session()

    assert session.state == "PAUSED"
    assert session.resume_to == "PAUSED"


# (from, to) pairs that are legal whatever resume_to is.
LEGAL_EDGES = [
    ("NARRATING", "LISTENING"),
    ("PAUSED", "LISTENING"),
    ("NARRATING", "REPLYING"),
    ("PAUSED", "REPLYING"),
    ("LISTENING", "REPLYING"),
    ("REPLYING", "REPLYING"),
    ("REPLYING", "LISTENING"),
    ("NARRATING", "PAUSED"),
    ("LISTENING", "PAUSED"),
    ("REPLYING", "PAUSED"),
    ("PAUSED", "NARRATING"),
]


@pytest.mark.parametrize(("origin", "target"), LEGAL_EDGES)
def test_legal_edges(origin: str, target: str) -> None:
    session = session_in(origin)

    assert session.transition(target, "test") is True
    assert session.state == target


@pytest.mark.parametrize("origin", ["LISTENING", "REPLYING"])
@pytest.mark.parametrize("resting", ["NARRATING", "PAUSED"])
def test_leaving_listening_or_replying_to_resume_to_is_legal(origin: str, resting: str) -> None:
    session = session_in(origin, resting)
    assert session.resume_to == resting

    assert session.transition(resting, "done") is True
    assert session.state == resting


def test_illegal_edge_returns_false_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    session = session_in("LISTENING", resting="PAUSED")
    listener_calls: list[tuple[str, str, str]] = []
    session.add_listener(lambda old, new, reason: listener_calls.append((old, new, reason)))

    with caplog.at_level(logging.WARNING, logger="voice_agent.orchestrator.state_machine"):
        # resume_to is PAUSED, so LISTENING may not jump to NARRATING.
        assert session.transition("NARRATING", "sneaky") is False

    assert session.state == "LISTENING"
    assert listener_calls == []
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["illegal transition LISTENING -> NARRATING (sneaky)"]


def test_same_state_is_a_silent_noop_except_replying() -> None:
    session = session_in("NARRATING")
    seen: list[tuple[str, str, str]] = []
    session.add_listener(lambda old, new, reason: seen.append((old, new, reason)))

    assert session.transition("NARRATING", "again") is True
    assert seen == []

    assert session.transition("REPLYING", "first") is True
    assert session.transition("REPLYING", "second") is True
    assert seen == [("NARRATING", "REPLYING", "first"), ("REPLYING", "REPLYING", "second")]


def test_resume_to_tracks_last_resting_state() -> None:
    narrating = make_session()
    narrating.transition("NARRATING", "start")
    narrating.transition("LISTENING", "barge_in")
    assert narrating.resume_to == "NARRATING"
    assert narrating.resume("deadline") is True
    assert narrating.state == "NARRATING"

    paused = make_session()
    paused.transition("LISTENING", "push_to_talk")
    assert paused.resume_to == "PAUSED"
    assert paused.resume("deadline") is True
    assert paused.state == "PAUSED"


def test_resume_only_leaves_listening_or_replying() -> None:
    """A late resume (say a reply that finishes after the user paused) must not undo the pause."""
    session = session_in("NARRATING")
    session.transition("LISTENING", "barge_in")
    session.transition("PAUSED", "pause")

    assert session.resume("reply_done") is False
    assert session.state == "PAUSED"


def test_listeners_can_tell_a_resume_from_a_pause() -> None:
    session = make_session()  # resume_to is PAUSED
    seen: list[tuple[str, bool]] = []
    session.add_listener(lambda old, new, reason: seen.append((reason, session.resuming)))

    session.transition("LISTENING", "push_to_talk")
    session.resume("deadline")
    session.transition("LISTENING", "push_to_talk")
    session.transition("PAUSED", "pause")

    assert seen == [("push_to_talk", False), ("deadline", True), ("push_to_talk", False), ("pause", False)]
    assert session.resuming is False


def test_listeners_called_once_per_change_in_order() -> None:
    session = make_session()
    calls: list[tuple[str, str, str, str]] = []
    session.add_listener(lambda old, new, reason: calls.append(("first", old, new, reason)))
    session.add_listener(lambda old, new, reason: calls.append(("second", old, new, reason)))

    session.transition("NARRATING", "resume")

    assert calls == [
        ("first", "PAUSED", "NARRATING", "resume"),
        ("second", "PAUSED", "NARRATING", "resume"),
    ]


def test_listener_exception_logged_others_run(caplog: pytest.LogCaptureFixture) -> None:
    session = make_session()
    ran: list[str] = []

    def broken(old: str, new: str, reason: str) -> None:
        raise RuntimeError("listener blew up")

    session.add_listener(broken)
    session.add_listener(lambda old, new, reason: ran.append(new))

    with caplog.at_level(logging.ERROR, logger="voice_agent.orchestrator.state_machine"):
        assert session.transition("NARRATING", "resume") is True

    assert ran == ["NARRATING"]
    assert session.state == "NARRATING"
    failures = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(failures) == 1
    assert failures[0].exc_info is not None
    assert failures[0].exc_info[0] is RuntimeError


# --- LISTENING deadline -----------------------------------------------------------------------------------------------

DEADLINE = 0.05


async def settle(seconds: float) -> None:
    await asyncio.sleep(seconds)


async def test_deadline_returns_to_resume_to() -> None:
    session = make_session(listen_deadline_s=DEADLINE)
    session.transition("NARRATING", "start")

    assert session.enter_listening("barge_in") is True
    assert session.state == "LISTENING"

    await settle(DEADLINE * 5)

    assert session.state == "NARRATING"


async def test_deadline_resumes_to_paused_when_it_started_paused() -> None:
    session = make_session(listen_deadline_s=DEADLINE)

    session.enter_listening("push_to_talk")
    await settle(DEADLINE * 5)

    assert session.state == "PAUSED"


async def test_extend_keeps_listening() -> None:
    # A long deadline against short extension steps: a busy machine can delay a step a long way without
    # the deadline legitimately running out in between.
    deadline = 0.2
    session = make_session(listen_deadline_s=deadline)
    session.transition("NARRATING", "start")
    session.enter_listening("barge_in")

    for _ in range(15):  # 0.3 s in total, longer than the deadline
        await settle(0.02)
        session.extend_deadline()
        assert session.state == "LISTENING"

    await settle(deadline * 3)  # no more speech
    assert session.state == "NARRATING"


async def test_suspended_deadline_never_fires() -> None:
    session = make_session(listen_deadline_s=DEADLINE)
    session.transition("NARRATING", "start")
    session.enter_listening("barge_in")

    session.suspend_deadline()
    await settle(DEADLINE * 3)
    assert session.state == "LISTENING"

    session.extend_deadline()  # speech while suspended does not re-arm either
    await settle(DEADLINE * 3)
    assert session.state == "LISTENING"

    session.resume_deadline()  # re-arms the full duration
    await settle(DEADLINE * 5)
    assert session.state == "NARRATING"


async def test_leaving_listening_cancels_timer() -> None:
    session = make_session(listen_deadline_s=DEADLINE)
    session.transition("NARRATING", "start")
    session.enter_listening("barge_in")

    session.transition("REPLYING", "question")
    await settle(DEADLINE * 5)

    assert session.state == "REPLYING"


async def test_reentering_listening_does_not_inherit_a_stale_suspension() -> None:
    session = make_session(listen_deadline_s=DEADLINE)
    session.enter_listening("barge_in")
    session.suspend_deadline()
    session.transition("REPLYING", "question")

    session.enter_listening("barge_in")
    await settle(DEADLINE * 5)

    assert session.state == "PAUSED"  # resumed: the new LISTENING turn armed its own deadline


async def test_followup_uses_followup_window() -> None:
    session = make_session(listen_deadline_s=30.0, followup_window_s=DEADLINE)
    session.transition("NARRATING", "start")
    session.transition("REPLYING", "question")

    session.enter_listening("followup", followup=True)
    await settle(DEADLINE * 5)

    assert session.state == "NARRATING"


async def test_barge_in_uses_the_listen_deadline_not_the_followup_window() -> None:
    session = make_session(listen_deadline_s=30.0, followup_window_s=DEADLINE)
    session.transition("NARRATING", "start")

    session.enter_listening("barge_in")
    await settle(DEADLINE * 5)

    assert session.state == "LISTENING"


def test_enter_listening_without_a_running_loop_changes_nothing() -> None:
    session = make_session()
    session.transition("NARRATING", "start")

    with pytest.raises(RuntimeError):
        session.enter_listening("barge_in")

    assert session.state == "NARRATING"


async def test_extend_and_suspend_outside_listening_are_noops() -> None:
    session = make_session(listen_deadline_s=DEADLINE)
    session.transition("NARRATING", "start")

    session.extend_deadline()
    session.suspend_deadline()
    session.resume_deadline()
    session.enter_listening("barge_in")
    await settle(DEADLINE * 5)

    assert session.state == "NARRATING"  # the earlier no-op suspend did not leave the deadline suspended


# --- holding the deadline for a client that records until it is done ------------------------------------------------


async def wait_for_state(session: SessionState, state: str, timeout_s: float = 5.0) -> None:
    async with asyncio.timeout(timeout_s):
        while session.state != state:
            await asyncio.sleep(0.005)


async def test_held_deadline_waits_for_the_release() -> None:
    session = make_session(listen_deadline_s=DEADLINE)
    session.transition("NARRATING", "start")
    session.enter_listening("barge_in")

    session.hold_deadline(30.0)
    await settle(DEADLINE * 4)
    assert session.state == "LISTENING"
    session.extend_deadline()  # speech does not matter while held
    session.suspend_deadline()
    session.resume_deadline()
    await settle(DEADLINE * 4)
    assert session.state == "LISTENING"

    session.release_deadline_hold()  # the normal deadline starts now
    assert session.state == "LISTENING"
    await wait_for_state(session, "NARRATING")


async def test_hold_is_bounded() -> None:
    session = make_session(listen_deadline_s=DEADLINE)
    session.transition("NARRATING", "start")
    session.enter_listening("barge_in")

    session.hold_deadline(DEADLINE * 2)  # the cap, then the normal deadline on top of it
    await settle(DEADLINE)
    assert session.state == "LISTENING"

    await wait_for_state(session, "NARRATING")


async def test_leaving_listening_clears_a_hold() -> None:
    session = make_session(listen_deadline_s=DEADLINE, followup_window_s=DEADLINE)
    session.transition("NARRATING", "start")
    session.enter_listening("barge_in")
    session.hold_deadline(30.0)

    session.transition("REPLYING", "question")
    session.enter_listening("followup", followup=True)  # a fresh turn: not held
    await wait_for_state(session, "NARRATING")


async def test_hold_outside_listening_is_a_noop() -> None:
    session = make_session(listen_deadline_s=DEADLINE)
    session.transition("NARRATING", "start")

    session.hold_deadline(30.0)
    session.enter_listening("barge_in")  # the earlier hold did not carry over

    await wait_for_state(session, "NARRATING")
