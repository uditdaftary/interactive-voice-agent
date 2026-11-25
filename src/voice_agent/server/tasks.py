"""Background task registry: every fire-and-forget task is referenced, logged on failure and cancellable.

The event loop keeps only weak references to tasks, so a bare `asyncio.create_task(...)` can be garbage
collected mid-flight, and an exception in one is reported only when the task object is finally destroyed.
"""

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Owns background tasks by name. Call it from the event loop thread only."""

    def __init__(self) -> None:
        self._tasks: list[asyncio.Task[Any]] = []

    def __len__(self) -> int:
        return len(self._tasks)

    def spawn(self, coro: Coroutine[Any, Any, Any], name: str, *, replace: bool = False) -> asyncio.Task[Any]:
        """Start `coro` as a tracked task. With `replace`, a live task of the same name is cancelled first
        (single-flight: one narration, one reply, one transcription at a time)."""
        if replace:
            self.cancel(name)
        task = asyncio.create_task(coro, name=name)
        self._tasks.append(task)
        task.add_done_callback(self._finished)
        return task

    def get(self, name: str) -> asyncio.Task[Any] | None:
        """The live task called `name`. A task that has been asked to cancel but has not stopped yet is not live:
        the caller is about to start its replacement."""
        for task in self._tasks:
            if task.get_name() == name and not task.done() and task.cancelling() == 0:
                return task
        return None

    def cancel(self, name: str) -> None:
        """Ask every live task called `name` to stop, without waiting for it. The task making this call is
        left alone: it is the one deciding, and cancelling it would stop it at its next await."""
        try:
            running = asyncio.current_task()
        except RuntimeError:  # called with no loop running (a teardown): no task is the caller
            running = None
        for task in self._named(name):
            if task is not running:
                task.cancel()

    async def cancel_and_wait(self, name: str) -> None:
        """Stop every live task called `name` and wait until it has.

        A failure of the stopped task is logged by its done-callback, not raised here. If the *caller* is
        cancelled while waiting, that cancellation propagates: it is never mistaken for the task's own.
        """
        tasks = self._named(name)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def cancel_all(self) -> None:
        """Cancel everything and wait for it. Tasks that spawn others while stopping are cancelled too."""
        while self._tasks:
            pending = list(self._tasks)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            self._tasks = [task for task in self._tasks if not task.done()]

    def _named(self, name: str) -> list[asyncio.Task[Any]]:
        return [task for task in self._tasks if task.get_name() == name and not task.done()]

    def _finished(self, task: asyncio.Task[Any]) -> None:
        if task in self._tasks:
            self._tasks.remove(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Background task %r failed: %s: %s", task.get_name(), type(exc).__name__, exc, exc_info=exc)
