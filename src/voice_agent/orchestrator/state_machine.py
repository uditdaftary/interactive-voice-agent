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
    """

    def __init__(self, *, listen_deadline_s: float, followup_window_s: float) -> None:
        self._state: SessionStateName = "PAUSED"
        self._resume_to: Literal["NARRATING", "PAUSED"] = "PAUSED"
        self._listeners: list[StateListener] = []
        self.listen_deadline_s = listen_deadline_s
        self.followup_window_s = followup_window_s

    @property
    def state(self) -> SessionStateName:
        return self._state

    @property
    def resume_to(self) -> Literal["NARRATING", "PAUSED"]:
        return self._resume_to

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
        return self.transition(self._resume_to, reason)
