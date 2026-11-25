"""TaskRegistry: strong references, logged failures, single-flight replacement, clean shutdown, cancellation."""

import asyncio
import logging

import pytest

from fakes import BlockingLLM, FakeLLM, FakeSearch, FakeTts
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.narration import NarrationEngine
from voice_agent.server.tasks import TaskRegistry
from voice_agent.settings import Settings


async def sleeper() -> None:
    await asyncio.sleep(3600)


async def test_spawn_keeps_reference_until_done() -> None:
    registry = TaskRegistry()
    release = asyncio.Event()

    async def work() -> str:
        await release.wait()
        return "done"

    task = registry.spawn(work(), "work")
    await asyncio.sleep(0)
    assert len(registry) == 1
    assert registry.get("work") is task

    release.set()
    assert await task == "done"
    await asyncio.sleep(0)  # let the done-callback run

    assert len(registry) == 0
    assert registry.get("work") is None


async def test_failed_task_logged_with_traceback(caplog: pytest.LogCaptureFixture) -> None:
    registry = TaskRegistry()

    async def boom() -> None:
        raise ValueError("kaput")

    with caplog.at_level(logging.ERROR, logger="voice_agent.server.tasks"):
        task = registry.spawn(boom(), "narration")
        await asyncio.wait([task])
        await asyncio.sleep(0)

    records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(records) == 1
    assert "narration" in records[0].getMessage()
    assert records[0].exc_info is not None
    assert records[0].exc_info[0] is ValueError
    assert len(registry) == 0


async def test_cancelled_task_is_not_logged_as_a_failure(caplog: pytest.LogCaptureFixture) -> None:
    registry = TaskRegistry()

    with caplog.at_level(logging.ERROR, logger="voice_agent.server.tasks"):
        task = registry.spawn(sleeper(), "reply")
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.wait([task])
        await asyncio.sleep(0)

    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


async def test_replace_cancels_previous() -> None:
    registry = TaskRegistry()
    first = registry.spawn(sleeper(), "reply")
    await asyncio.sleep(0)

    second = registry.spawn(sleeper(), "reply", replace=True)
    await asyncio.sleep(0)

    assert first.cancelled()
    assert registry.get("reply") is second
    await registry.cancel_all()


async def test_spawn_without_replace_keeps_both() -> None:
    registry = TaskRegistry()
    first = registry.spawn(sleeper(), "stt")
    second = registry.spawn(sleeper(), "stt")
    await asyncio.sleep(0)

    assert not first.done()
    assert not second.done()
    assert len(registry) == 2
    await registry.cancel_all()


async def test_get_ignores_a_task_that_is_being_cancelled() -> None:
    """Narration is restarted right after being cancelled: the old task must not count as live."""
    registry = TaskRegistry()
    release = asyncio.Event()

    async def slow_to_stop() -> None:
        try:
            await sleeper()
        except asyncio.CancelledError:
            await release.wait()
            raise

    task = registry.spawn(slow_to_stop(), "narration")
    await asyncio.sleep(0)
    registry.cancel("narration")
    await asyncio.sleep(0)

    assert not task.done()
    assert registry.get("narration") is None
    release.set()
    await registry.cancel_all()


async def test_cancel_all_leaves_nothing_pending() -> None:
    registry = TaskRegistry()
    tasks = [registry.spawn(sleeper(), name) for name in ("narration", "reply", "stt")]
    await asyncio.sleep(0)

    await registry.cancel_all()

    assert all(task.done() for task in tasks)
    assert len(registry) == 0


async def test_cancel_all_reaches_tasks_spawned_while_cancelling() -> None:
    registry = TaskRegistry()
    spawned: list[asyncio.Task] = []

    async def parent() -> None:
        try:
            await sleeper()
        except asyncio.CancelledError:
            spawned.append(registry.spawn(sleeper(), "late"))
            raise

    registry.spawn(parent(), "parent")
    await asyncio.sleep(0)

    await registry.cancel_all()

    assert len(spawned) == 1
    assert spawned[0].done()
    assert len(registry) == 0


async def test_cancel_and_wait_returns_after_the_task_has_stopped() -> None:
    registry = TaskRegistry()
    stopped = False

    async def work() -> None:
        nonlocal stopped
        try:
            await sleeper()
        finally:
            await asyncio.sleep(0.01)
            stopped = True

    registry.spawn(work(), "narration")
    await asyncio.sleep(0)

    await registry.cancel_and_wait("narration")

    assert stopped


async def test_cancel_and_wait_does_not_raise_for_a_task_that_failed_while_stopping() -> None:
    registry = TaskRegistry()

    async def work() -> None:
        try:
            await sleeper()
        finally:
            raise RuntimeError("cleanup failed")  # noqa: B012  # the failure is the point of the test

    registry.spawn(work(), "narration")
    await asyncio.sleep(0)

    await registry.cancel_and_wait("narration")  # logged by the done-callback, not raised at the caller


# --- L5: cancelling a caller must reach it ----------------------------------------------------------------------------


class SlowToStopTts:
    """Blocks inside synthesis; once cancelled it takes its time to finish stopping."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.stopping = asyncio.Event()
        self.finish_stopping = asyncio.Event()

    async def __call__(self, text: str, voice_id: str, cadence: float) -> tuple[bytes | None, str]:
        self.entered.set()
        try:
            await sleeper()
        except asyncio.CancelledError:
            self.stopping.set()
            await self.finish_stopping.wait()
            raise
        return None, "pcm_s16le"


def make_engine(llm: FakeLLM, tts: object) -> NarrationEngine:
    return NarrationEngine(
        ConnectionManager(),
        llm_client=llm,
        settings=Settings.fast(),
        tts=tts,  # type: ignore[arg-type]
        search=FakeSearch(),
    )


async def test_caller_cancellation_propagates_during_the_llm_stream() -> None:
    llm = BlockingLLM()
    engine = make_engine(llm, FakeTts())

    caller = asyncio.create_task(engine.handle_user_interruption("who is lucy"))
    async with asyncio.timeout(5):
        await llm.started.wait()
    caller.cancel()

    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(5):
            await caller


@pytest.mark.parametrize("method", ["handle_user_interruption", "handle_unrecognized_interruption"])
async def test_caller_cancellation_propagates_while_narration_is_stopping(method: str) -> None:
    """The caller waits for the narration task to stop. Cancelling the caller then must not be swallowed."""
    llm = BlockingLLM()
    tts = SlowToStopTts()
    engine = make_engine(llm, tts)
    engine.start_narration()
    async with asyncio.timeout(5):
        await tts.entered.wait()

    args = ("who is lucy",) if method == "handle_user_interruption" else ()
    caller = asyncio.create_task(getattr(engine, method)(*args))
    async with asyncio.timeout(5):
        await tts.stopping.wait()
    caller.cancel()
    tts.finish_stopping.set()

    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(5):
            await caller
    assert llm.calls == []  # the cancelled caller never went on to ask the model
    await engine.manager.tasks.cancel_all()


async def test_cancel_never_cancels_the_calling_task() -> None:
    """A task that triggers its own cancellation (a state listener running inside it) must not stop itself."""
    registry = TaskRegistry()
    reached_the_end = asyncio.Event()

    async def cancels_by_name() -> None:
        registry.cancel("reply")
        await asyncio.sleep(0)  # a cancelled task would raise here
        reached_the_end.set()

    task = registry.spawn(cancels_by_name(), "reply")
    other = registry.spawn(sleeper(), "reply")
    await asyncio.wait([task], timeout=5)

    assert reached_the_end.is_set()
    assert other.cancelled()  # a different task of the same name is still cancelled
    await registry.cancel_all()
