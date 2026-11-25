"""Playback acknowledgements: the browser says when a clip has finished playing, and the server waits for that.

Every clip sent to the owner is a numbered segment of one generation. Stopping audio (a pause, a barge-in, a
navigation, a new document, a new owner) bumps the generation and drops every segment still outstanding, so a
late or stale report can never be taken for the end of the clip that is playing now.
"""

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class _Segment:
    generation: int
    connection: object  # the connection the clip went to: only that one can say it has played it
    done: "asyncio.Future[bool]"


class PlaybackTracker:
    """Segment ids, and waiting for a segment's `playback_done`. Event loop only."""

    def __init__(self) -> None:
        self._next_id = 1
        self._segments: dict[int, _Segment] = {}

    def __len__(self) -> int:
        return len(self._segments)

    def register(self, generation: int, connection: object) -> int:
        """Number a clip that is about to be sent. Registered before it is sent, so an immediate report finds it."""
        segment_id = self._next_id
        self._next_id += 1
        self._segments[segment_id] = _Segment(generation, connection, asyncio.get_running_loop().create_future())
        return segment_id

    def forget(self, segment_id: int) -> None:
        """Stop tracking a segment nobody will wait for (a send that failed, or a clip paced by an estimate)."""
        segment = self._segments.pop(segment_id, None)
        if segment is not None and not segment.done.done():
            segment.done.set_result(False)

    def acknowledge(self, connection: object, segment_id: int, generation: int) -> bool:
        """The browser reports a clip finished. True only for a clip that is outstanding, was sent to this connection
        and belongs to this generation: an unknown id, a report from another tab and a report for a clip that a stop
        already dropped all do nothing."""
        segment = self._segments.get(segment_id)
        if segment is None or segment.connection is not connection or segment.generation != generation:
            logger.debug("Ignoring playback_done for segment %d (generation %d): nothing waits for it", segment_id, generation)
            return False
        if not segment.done.done():
            segment.done.set_result(True)
        return True

    def done_future(self, segment_id: int) -> "asyncio.Future[bool] | None":
        """The future that `acknowledge` resolves, for a caller that wants to wait with timers of its own. It is
        consumed by `wait_done` (or `forget`); None for a segment that is not outstanding."""
        segment = self._segments.get(segment_id)
        return segment.done if segment is not None else None

    async def wait_done(self, segment_id: int, timeout_s: float) -> bool:
        """True when the browser reported the segment played. False after `timeout_s` (a WARNING says so) or when a
        stop dropped it. The segment is consumed either way."""
        segment = self._segments.get(segment_id)
        if segment is None:
            return False
        try:
            return await asyncio.wait_for(segment.done, timeout_s)
        except TimeoutError:
            logger.warning(
                "No playback_done for segment %d after %.1f s; going on without it. Is the tab playing audio?",
                segment_id,
                timeout_s,
            )
            return False
        finally:
            self._segments.pop(segment_id, None)

    def drop_all(self) -> None:
        """Every outstanding segment is dead: whoever waits for one is released with False."""
        segments = list(self._segments.values())
        self._segments.clear()
        for segment in segments:
            if not segment.done.done():
                segment.done.set_result(False)
