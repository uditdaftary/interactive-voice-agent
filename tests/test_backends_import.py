"""The backend modules import with neither optional extra present, so the base install and CI never need pocket-tts or sherpa-onnx.

Run in a fresh interpreter: blocking the extras with sys.modules must not leak into the other tests in this process.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PROBE = """
import sys
for blocked in ("pocket_tts", "sherpa_onnx"):
    sys.modules[blocked] = None  # an import of either raises ImportError
import voice_agent.tts.backends as tts_backends
import voice_agent.stt.backends as stt_backends
from voice_agent.settings import load_settings
assert tts_backends.build_tts("edge").backends
assert stt_backends.MoonshineBackend is not None
assert load_settings({}).stt_backend == "whisper"
print("imports-ok")
"""


def test_backends_import_without_extras() -> None:
    environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "imports-ok" in result.stdout
