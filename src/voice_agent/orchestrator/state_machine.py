import asyncio
import logging
import threading
from enum import Enum
from typing import Callable, Literal, Optional

logger = logging.getLogger(__name__)


class AgentState(Enum):
    NARRATING = "NARRATING"
    BARGE_IN_TRIGGERED = "BARGE_IN_TRIGGERED"
    LISTENING = "LISTENING"
    REPLYING = "REPLYING"


class AgentStateMachine:
    """Manages conversational state transitions and coordinates barge-in interrupts with thread safety."""

    def __init__(self, on_barge_in: Optional[Callable[[], None]] = None):
        self._lock = threading.RLock()
        self._state = AgentState.NARRATING
        self.on_barge_in = on_barge_in

    @property
    def state(self) -> AgentState:
        with self._lock:
            return self._state

    @state.setter
    def state(self, new_state: AgentState):
        with self._lock:
            self._state = new_state

    def trigger_speech_detected(self) -> bool:
        """Called when VAD signals active human speech.

        Allows barge-in interrupt from BOTH NARRATING and REPLYING states.
        Guarantees transition to LISTENING even if on_barge_in callback raises an exception.
        Returns True if barge-in was triggered, False otherwise.
        """
        with self._lock:
            if self._state in (AgentState.NARRATING, AgentState.REPLYING):
                self._state = AgentState.BARGE_IN_TRIGGERED
                try:
                    if self.on_barge_in:
                        self.on_barge_in()
                except Exception as exc:
                    logger.exception("Exception during on_barge_in callback: %s", exc)
                finally:
                    self._state = AgentState.LISTENING
                return True
            return False

    def trigger_speech_finished(self) -> bool:
        """Called when user silence hangover expires (turn complete).

        Transitions from LISTENING to REPLYING.
        Returns True if transition succeeded, False otherwise.
        """
        with self._lock:
            if self._state == AgentState.LISTENING:
                self._state = AgentState.REPLYING
                return True
            return False

    def trigger_reply_finished(self) -> bool:
        """Called when LLM/TTS response completes to resume narration.

        Transitions from REPLYING to NARRATING.
        Returns True if transition succeeded, False otherwise.
        """
        with self._lock:
            if self._state == AgentState.REPLYING:
                self._state = AgentState.NARRATING
                return True
            return False


SessionStateName = Literal["NARRATING", "PAUSED", "LISTENING", "REPLYING"]

# Edges that hold in every situation. LISTENING and REPLYING may also go to `resume_to`, which varies.
_FIXED_EDGES: dict[SessionStateName, frozenset[SessionStateName]] = {
    "NARRATING": frozenset({"LISTENING", "REPLYING", "PAUSED"}),
    "PAUSED": frozenset({"LISTENING", "REPLYING", "NARRATING"}),
    "LISTENING": frozenset({"REPLYING", "PAUSED"}),
    "REPLYING": frozenset({"LISTENING", "PAUSED"}),
}

StateListener = Callable[[str, str, str], None]


class SessionState:
    """The one owner of the session state: which transitions are legal, and what to resume to.

    Every call must come from the event loop thread. There is deliberately no lock: listeners run
    synchronously inside `transition`, so a lock would only invite re-entrant deadlocks. A listener
    must not call `transition` itself.

    `resume_to` is the resting state (NARRATING or PAUSED) most recently left, so a barge-in or a
    question returns to what the user had before.

    LISTENING has a deadline so the session can never wait for a user who is not there. It is
    extended while the user speaks and suspended while an utterance is being transcribed; it resumes
    to `resume_to` when it runs out. Leaving LISTENING always cancels it, so a stale timer never fires.
    A client that records until it says it is done (push-to-talk) can hold the clock for a bounded time.
    """

    def __init__(self, *, listen_deadline_s: float, followup_window_s: float) -> None:
        self._state: SessionStateName = "PAUSED"
        self._resume_to: Literal["NARRATING", "PAUSED"] = "PAUSED"
        self._listeners: list[StateListener] = []
        self.listen_deadline_s = listen_deadline_s
        self.followup_window_s = followup_window_s
        self._deadline_timer: asyncio.TimerHandle | None = None
        self._armed_for_s: float | None = None  # the full duration extend/resume re-arm; None outside LISTENING
        self._deadline_suspended = False
        self._deadline_held = False
        self._hold_timer: asyncio.TimerHandle | None = None
        self._resuming = False

    @property
    def state(self) -> SessionStateName:
        return self._state

    @property
    def resume_to(self) -> Literal["NARRATING", "PAUSED"]:
        return self._resume_to

    @property
    def resuming(self) -> bool:
        """True while listeners are being told about a `resume()`: the session is returning to what it was doing,
        as opposed to being stopped. A listener that cancels work on PAUSED must not cancel on a resume."""
        return self._resuming

    def add_listener(self, fn: StateListener) -> None:
        """Call `fn(old, new, reason)` after every change, in registration order."""
        self._listeners.append(fn)

    def transition(self, to: SessionStateName, reason: str) -> bool:
        """Move to `to`. An illegal edge logs a WARNING and returns False; callers act on the result."""
        old = self._state
        if to == old and to != "REPLYING":
            return True  # already there. A new reply replaces the old one, so REPLYING -> REPLYING is announced.
        legal = to in _FIXED_EDGES[old] or (old in ("LISTENING", "REPLYING") and to == self._resume_to) or to == old
        if not legal:
            logger.warning("illegal transition %s -> %s (%s)", old, to, reason)
            return False

        if old == "NARRATING" or old == "PAUSED":
            self._resume_to = old
        if old == "LISTENING" and to != "LISTENING":
            self._clear_deadline()
        self._state = to
        for listener in list(self._listeners):
            try:
                listener(old, to, reason)
            except Exception:  # noqa: BLE001  # one broken listener must not keep the others from seeing the change
                logger.exception("state listener failed on %s -> %s (%s)", old, to, reason)
        return True

    def resume(self, reason: str) -> bool:
        """Leave LISTENING or REPLYING for `resume_to`. From any other state it does nothing: a late
        resume (a reply that ends after the user paused) must not undo the pause."""
        if self._state not in ("LISTENING", "REPLYING"):
            logger.debug("resume ignored in %s (%s)", self._state, reason)
            return False
        self._resuming = True
        try:
            return self.transition(self._resume_to, reason)
        finally:
            self._resuming = False

    # --- LISTENING deadline ---------------------------------------------------------------------------------------

    def enter_listening(self, reason: str, *, followup: bool = False) -> bool:
        """Go to LISTENING and arm its deadline: the follow-up window after a reply, else the barge-in deadline.

        Needs a running loop for the timer. Without one this raises before changing anything, rather than
        leaving a session in LISTENING that nothing will ever end.
        """
        loop = asyncio.get_running_loop()
        if not self.transition("LISTENING", reason):
            return False
        self._armed_for_s = self.followup_window_s if followup else self.listen_deadline_s
        self._deadline_suspended = False
        self._release_hold()
        self._arm_deadline(loop)
        return True

    def extend_deadline(self) -> None:
        """Re-arm the full duration. Call it whenever the user is heard speaking."""
        if self._can_rearm():
            self._arm_deadline(asyncio.get_running_loop())

    def suspend_deadline(self) -> None:
        """Stop the clock while an utterance is with the recognizer, so a slow transcript cannot expire the turn."""
        if self._state == "LISTENING" and self._armed_for_s is not None:
            self._deadline_suspended = True
            self._cancel_timer()

    def resume_deadline(self) -> None:
        """Start the clock again with the full duration."""
        if self._deadline_suspended:
            self._deadline_suspended = False
            if self._can_rearm():
                self._arm_deadline(asyncio.get_running_loop())

    def hold_deadline(self, max_s: float) -> None:
        """Stop the clock until `release_deadline_hold`, for at most `max_s`, then run the normal deadline.

        For a client that records until it reports it is done: speech too quiet for the server to hear
        must not time the turn out while the user is still talking.
        """
        if self._state == "LISTENING" and self._armed_for_s is not None:
            self._release_hold()
            self._deadline_held = True
            self._cancel_timer()
            self._hold_timer = asyncio.get_running_loop().call_later(max_s, self.release_deadline_hold)

    def release_deadline_hold(self) -> None:
        """End a hold and start the normal deadline."""
        was_held = self._deadline_held
        self._release_hold()
        if was_held and self._can_rearm():
            self._arm_deadline(asyncio.get_running_loop())

    def _release_hold(self) -> None:
        self._deadline_held = False
        if self._hold_timer is not None:
            self._hold_timer.cancel()
            self._hold_timer = None

    def _can_rearm(self) -> bool:
        return (
            self._state == "LISTENING"
            and self._armed_for_s is not None
            and not self._deadline_suspended
            and not self._deadline_held
        )

    def _arm_deadline(self, loop: asyncio.AbstractEventLoop) -> None:
        self._cancel_timer()
        if self._armed_for_s is not None:
            self._deadline_timer = loop.call_later(self._armed_for_s, self._expire)

    def _cancel_timer(self) -> None:
        if self._deadline_timer is not None:
            self._deadline_timer.cancel()
            self._deadline_timer = None

    def _clear_deadline(self) -> None:
        self._cancel_timer()
        self._release_hold()
        self._armed_for_s = None
        self._deadline_suspended = False

    def _expire(self) -> None:
        self._deadline_timer = None
        # The timer is cancelled on every exit from LISTENING, so these guards only matter if that ever regresses.
        if self._state == "LISTENING" and not self._deadline_suspended and not self._deadline_held:
            self.resume("deadline")
