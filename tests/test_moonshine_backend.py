"""The Moonshine backend: a child worker process speaking the local_stt_worker JSON-lines protocol.

The worker here is a fake script written to tmp_path that speaks the same protocol, so no sherpa-onnx and no model are needed.
"""

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from voice_agent.settings import Settings, load_settings
from voice_agent.stt.backends import (
    STT_BACKEND_NAMES,
    STT_UNAVAILABLE,
    MoonshineBackend,
    Transcript,
    WhisperBackend,
    build_stt_backend,
)

FAKE_WORKER = '''
import base64, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
config = json.load(open(os.path.join(HERE, "worker.json"), encoding="utf-8"))


def send(message):
    sys.stdout.buffer.write((json.dumps(message, ensure_ascii=False) + "\\n").encode("utf-8"))
    sys.stdout.buffer.flush()


with open(config["starts_log"], "a", encoding="utf-8") as starts:
    starts.write("start\\n")

if config.get("mode") == "load-error":
    send({"type": "error", "message": "load failed: FileNotFoundError: model file missing: tokens.txt"})
    sys.exit(3)
if config.get("mode") == "never-ready":
    time.sleep(120)
send({"type": "ready", "engine": "moonshine-tiny-en-int8", "kind": "moonshine", "threads": 1, "loadMs": 1})

audio_bytes = 0
for line in sys.stdin:
    message = json.loads(line)
    op = message.get("op")
    if op == "quit":
        break
    if op == "audio":
        audio_bytes += len(base64.b64decode(message["pcm"]))
        continue
    if op != "finish":
        continue
    request_id = message["id"]
    if config.get("mode") == "silent":
        time.sleep(120)
    die_flag = config.get("die_flag")
    if die_flag and os.path.exists(die_flag):
        os.remove(die_flag)
        os._exit(1)
    # The record is written before the reply, so the parent reads it once its request has returned.
    with open(config["audio_log"], "a", encoding="utf-8") as record:
        record.write(str(audio_bytes) + "\\n")
    audio_bytes = 0
    if config.get("mode") == "error":
        send({"type": "error", "id": request_id, "message": "finish failed: RuntimeError: decoder"})
    else:
        send({
            "type": "final", "id": request_id, "text": config.get("text", ""), "ms": 1,
            "seconds": 1.0, "speech": config.get("speech", True), "reused": False, "cpuMs": 1,
            **({"rejected": config["rejected"]} if config.get("rejected") else {}),
        })
'''

SAMPLE_RATE = 16000


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def make_worker(tmp_path: Path) -> Iterator[object]:
    """Writes the fake worker and its config, and returns a factory for backends that use it. Every backend is closed afterwards."""
    backends: list[MoonshineBackend] = []
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    starts_log = tmp_path / "starts.log"
    audio_log = tmp_path / "audio.log"

    def make(timeout_s: float = 5.0, clock: Clock | None = None, **config: object) -> MoonshineBackend:
        (tmp_path / "worker.json").write_text(
            json.dumps({"starts_log": str(starts_log), "audio_log": str(audio_log), **config}), encoding="utf-8"
        )
        backend = MoonshineBackend(
            model_dir=str(tmp_path),
            worker_path=script,
            python=sys.executable,
            timeout_s=timeout_s,
            clock=clock if clock is not None else Clock(),
        )
        backends.append(backend)
        return backend

    def starts() -> int:
        return len(starts_log.read_text(encoding="utf-8").splitlines()) if starts_log.exists() else 0

    def audio_received() -> list[int]:
        return [int(line) for line in audio_log.read_text(encoding="utf-8").splitlines()] if audio_log.exists() else []

    make.starts = starts  # type: ignore[attr-defined]
    make.audio_received = audio_received  # type: ignore[attr-defined]
    yield make
    for backend in backends:
        backend.close()


def one_second_of_speech() -> bytes:
    return (np.ones(SAMPLE_RATE, dtype=np.int16) * 500).tobytes()


def test_starts_once_and_reuses_the_process(make_worker) -> None:
    backend = make_worker(text="hello there")

    assert backend.transcribe(one_second_of_speech(), SAMPLE_RATE) == Transcript("hello there", None)
    assert backend.transcribe(one_second_of_speech(), SAMPLE_RATE) == Transcript("hello there", None)

    assert make_worker.starts() == 1
    assert backend.is_running


def test_final_text_returned_with_utf8(make_worker) -> None:
    backend = make_worker(text="Привет, choom")

    assert backend.transcribe(one_second_of_speech(), SAMPLE_RATE) == Transcript("Привет, choom", None)


def test_not_speech_is_rejected(make_worker) -> None:
    backend = make_worker(text="", speech=False)

    assert backend.transcribe(one_second_of_speech(), SAMPLE_RATE) == Transcript("", "not-speech")


def test_worker_rejection_reason_is_kept(make_worker) -> None:
    backend = make_worker(text="", speech=True, rejected="stock-phrase")

    assert backend.transcribe(one_second_of_speech(), SAMPLE_RATE) == Transcript("", "stock-phrase")


def test_worker_error_is_stt_error_and_logged(make_worker, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="voice_agent.stt.backends")
    backend = make_worker(mode="error")

    assert backend.transcribe(one_second_of_speech(), SAMPLE_RATE) == Transcript("", "stt-error")
    assert "decoder" in caplog.text
    assert backend.is_running


def test_audio_is_sent_at_16k_in_chunks(make_worker) -> None:
    backend = make_worker(text="ok")
    eight_khz_one_second = (np.ones(8000, dtype=np.int16) * 500).tobytes()

    backend.transcribe(eight_khz_one_second, 8000)

    assert make_worker.audio_received() == [32000]


def test_dead_worker_restarts_after_cooldown(make_worker, tmp_path: Path) -> None:
    clock = Clock()
    die_flag = tmp_path / "die-once"
    die_flag.write_text("", encoding="utf-8")
    backend = make_worker(text="hello there", die_flag=str(die_flag), clock=clock)

    assert backend.transcribe(one_second_of_speech(), SAMPLE_RATE) == Transcript("", STT_UNAVAILABLE)
    assert make_worker.starts() == 1

    clock.now += 10
    assert backend.transcribe(one_second_of_speech(), SAMPLE_RATE) == Transcript("", STT_UNAVAILABLE)
    assert make_worker.starts() == 1  # inside the 30 s cooldown: no restart

    clock.now += 31
    assert backend.transcribe(one_second_of_speech(), SAMPLE_RATE) == Transcript("hello there", None)
    assert make_worker.starts() == 2


def test_read_timeout_kills_worker(make_worker) -> None:
    backend = make_worker(timeout_s=0.2, mode="silent")

    assert backend.transcribe(one_second_of_speech(), SAMPLE_RATE) == Transcript("", STT_UNAVAILABLE)
    assert not backend.is_running


def test_worker_that_never_reports_ready_is_unavailable(make_worker, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="voice_agent.stt.backends")
    backend = make_worker(timeout_s=0.2, mode="never-ready")

    assert backend.transcribe(one_second_of_speech(), SAMPLE_RATE) == Transcript("", STT_UNAVAILABLE)
    assert not backend.is_running


def test_load_error_names_the_missing_file(make_worker, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="voice_agent.stt.backends")
    backend = make_worker(mode="load-error")

    assert backend.transcribe(one_second_of_speech(), SAMPLE_RATE) == Transcript("", STT_UNAVAILABLE)
    assert "tokens.txt" in caplog.text


def test_empty_audio_is_quiet_and_starts_nothing(make_worker) -> None:
    backend = make_worker(text="hello")

    assert backend.transcribe(b"", SAMPLE_RATE) == Transcript("", "quiet")
    assert make_worker.starts() == 0


def test_close_kills_the_worker(make_worker) -> None:
    backend = make_worker(text="hello there")
    backend.transcribe(one_second_of_speech(), SAMPLE_RATE)
    assert backend.is_running

    backend.close()

    assert not backend.is_running


def test_build_stt_backend_default_is_whisper() -> None:
    backend = build_stt_backend(Settings())

    assert isinstance(backend, WhisperBackend)


def test_build_stt_backend_selects_moonshine() -> None:
    backend = build_stt_backend(Settings(stt_backend="moonshine", moonshine_model_dir="models/x"))

    assert isinstance(backend, MoonshineBackend)
    assert backend.model_dir == "models/x"
    backend.close()


def test_stt_backend_names_and_env() -> None:
    assert STT_BACKEND_NAMES == ("whisper", "moonshine")
    assert load_settings({"VOICE_AGENT_STT_BACKEND": " Moonshine "}).stt_backend == "moonshine"
    assert load_settings({}).stt_backend == "whisper"
    assert load_settings({"VOICE_AGENT_MOONSHINE_MODEL_DIR": "elsewhere"}).moonshine_model_dir == "elsewhere"


def test_unknown_stt_backend_rejected_at_load() -> None:
    with pytest.raises(ValueError, match="VOICE_AGENT_STT_BACKEND"):
        load_settings({"VOICE_AGENT_STT_BACKEND": "google"})
