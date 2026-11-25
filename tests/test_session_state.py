"""SessionState: legal transitions, resume_to and listeners."""

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


# (from, to) pairs that spec 5.1 allows without consulting resume_to.
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
