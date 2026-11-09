import logging
import threading
from enum import Enum
from typing import Callable, Optional

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
