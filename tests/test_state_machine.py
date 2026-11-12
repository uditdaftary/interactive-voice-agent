import concurrent.futures
from voice_agent.orchestrator.state_machine import AgentState, AgentStateMachine


def test_barge_in_transition():
    drained = False

    def on_drain():
        nonlocal drained
        drained = True

    sm = AgentStateMachine(on_barge_in=on_drain)
    assert sm.state == AgentState.NARRATING

    # User begins speaking
    assert sm.trigger_speech_detected() is True
    assert sm.state == AgentState.LISTENING
    assert drained is True

    # User finishes speaking
    assert sm.trigger_speech_finished() is True
    assert sm.state == AgentState.REPLYING

    # Reply finishes
    assert sm.trigger_reply_finished() is True
    assert sm.state == AgentState.NARRATING


def test_barge_in_during_replying_state():
    barge_in_count = 0

    def on_barge():
        nonlocal barge_in_count
        barge_in_count += 1

    sm = AgentStateMachine(on_barge_in=on_barge)
    assert sm.state == AgentState.NARRATING

    # First turn: NARRATING -> LISTENING -> REPLYING
    assert sm.trigger_speech_detected() is True
    assert sm.state == AgentState.LISTENING
    assert barge_in_count == 1

    assert sm.trigger_speech_finished() is True
    assert sm.state == AgentState.REPLYING

    # User interrupts agent while it is REPLYING
    assert sm.trigger_speech_detected() is True
    assert sm.state == AgentState.LISTENING
    assert barge_in_count == 2

    # Finish user speech and then agent reply to return to narration
    assert sm.trigger_speech_finished() is True
    assert sm.state == AgentState.REPLYING
    assert sm.trigger_reply_finished() is True
    assert sm.state == AgentState.NARRATING


def test_exception_safety_in_barge_in_callback():
    def broken_callback():
        raise RuntimeError("Audio hardware failure in buffer drain")

    sm = AgentStateMachine(on_barge_in=broken_callback)
    assert sm.state == AgentState.NARRATING

    # Triggering speech detected with broken callback must NOT crash and must guarantee LISTENING state
    assert sm.trigger_speech_detected() is True
    assert sm.state == AgentState.LISTENING

    # State machine remains healthy and responsive
    assert sm.trigger_speech_finished() is True
    assert sm.state == AgentState.REPLYING
    assert sm.trigger_reply_finished() is True
    assert sm.state == AgentState.NARRATING


def test_reentrant_deadlock_immunity():
    sm = None
    inspected_state = None

    def reentrant_callback():
        nonlocal inspected_state
        # Inspect state machine property from within the callback running under lock
        inspected_state = sm.state

    sm = AgentStateMachine(on_barge_in=reentrant_callback)
    assert sm.trigger_speech_detected() is True
    assert inspected_state == AgentState.BARGE_IN_TRIGGERED
    assert sm.state == AgentState.LISTENING


def test_invalid_transitions():
    sm = AgentStateMachine()
    assert sm.state == AgentState.NARRATING

    # Cannot finish speech or finish reply while in NARRATING
    assert sm.trigger_speech_finished() is False
    assert sm.trigger_reply_finished() is False
    assert sm.state == AgentState.NARRATING

    # Move to LISTENING
    assert sm.trigger_speech_detected() is True
    assert sm.state == AgentState.LISTENING

    # Cannot trigger speech detected again or finish reply while LISTENING
    assert sm.trigger_speech_detected() is False
    assert sm.trigger_reply_finished() is False
    assert sm.state == AgentState.LISTENING


def test_thread_safety_concurrent_transitions():
    sm = AgentStateMachine()
    iterations = 200
    errors = []

    def speech_detected_worker():
        try:
            for _ in range(iterations):
                sm.trigger_speech_detected()
        except Exception as e:
            errors.append(e)

    def speech_finished_worker():
        try:
            for _ in range(iterations):
                sm.trigger_speech_finished()
        except Exception as e:
            errors.append(e)

    def reply_finished_worker():
        try:
            for _ in range(iterations):
                sm.trigger_reply_finished()
        except Exception as e:
            errors.append(e)

    def inspector_worker():
        try:
            for _ in range(iterations):
                curr = sm.state
                assert isinstance(curr, AgentState)
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(speech_detected_worker),
            executor.submit(speech_finished_worker),
            executor.submit(reply_finished_worker),
            executor.submit(inspector_worker),
            executor.submit(speech_detected_worker),
            executor.submit(speech_finished_worker),
            executor.submit(reply_finished_worker),
            executor.submit(inspector_worker),
        ]
        concurrent.futures.wait(futures)

    assert len(errors) == 0
    assert isinstance(sm.state, AgentState)
