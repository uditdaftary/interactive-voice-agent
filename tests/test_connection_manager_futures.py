import concurrent.futures

from voice_agent.server.connection_manager import ConnectionManager


def test_log_future_exception_ignores_cancelled_future():
    fut = concurrent.futures.Future()
    fut.cancel()
    ConnectionManager._log_future_exception(fut)  # must not raise NameError
