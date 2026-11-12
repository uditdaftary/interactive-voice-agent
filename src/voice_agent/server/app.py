"""FastAPI Application Factory & Static HUD Dashboard Server.

Hosts the single-page HUD dashboard static assets (`ui/`) at `http://localhost:8000/`,
bi-directional streaming WebSocket endpoint at `ws://localhost:8000/ws`,
and REST endpoints for document ingestion, LLM configuration, and Q&A interruption.
"""

from contextlib import asynccontextmanager
import io
import logging
from pathlib import Path
from typing import Any, Optional, Union
import wave
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles

from voice_agent.document.parser import DocumentParser
from voice_agent.server.connection_manager import (
    ConnectionManager,
    default_connection_manager,
)
from voice_agent.server.narration import NarrationEngine
from voice_agent.server.websocket import router as websocket_router
import asyncio

logger = logging.getLogger(__name__)


def extract_audio_and_duration(content: bytes) -> tuple[bytes, float]:
    """Extract raw PCM bytes and duration in seconds from audio content.

    Supports RIFF WAV containers (converting stereo to mono PCM if needed)
    or falls back to raw 16kHz 16-bit mono PCM.
    """
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(content), "rb") as wf:
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                channels = wf.getnchannels()
                duration = round(n_frames / float(framerate), 3) if framerate > 0 else 0.0
                raw_frames = wf.readframes(n_frames)
                if channels > 1:
                    pcm_np = np.frombuffer(raw_frames, dtype=np.int16)
                    pcm_np = pcm_np.reshape(-1, channels).mean(axis=1).astype(np.int16)
                    return pcm_np.tobytes(), duration
                return raw_frames, duration
        except Exception as exc:
            logger.debug("Could not parse WAV container: %s", exc)

    # Fallback: assume raw PCM 16-bit mono 16kHz (32000 bytes/sec)
    duration = round(len(content) / 32000.0, 3)
    return content, duration


def get_stt_engine(app: FastAPI) -> Any:
    """Retrieve or lazily initialize the STTEngine from app state or connection manager."""
    engine = getattr(app.state, "stt_engine", None)
    if engine is None:
        manager = getattr(app.state, "connection_manager", None)
        if manager is not None and hasattr(manager, "stt_engine") and manager.stt_engine is not None:
            engine = manager.stt_engine
        else:
            from voice_agent.stt.recognizer import STTEngine
            engine = STTEngine(fallback_mock=False)
        app.state.stt_engine = engine
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for graceful startup and shutdown."""
    logger.info("Starting up Interactive Voice Agent Server.")
    yield
    logger.info("Shutting down Interactive Voice Agent Server gracefully.")
    engine: Optional[NarrationEngine] = getattr(app.state, "narration_engine", None)
    if engine is not None:
        engine.pause_narration()
    manager: Optional[ConnectionManager] = getattr(app.state, "connection_manager", None)
    if manager is not None:
        await manager.disconnect_all()


def create_app(
    ui_dir: Optional[Union[str, Path]] = None,
    connection_manager: Optional[ConnectionManager] = None,
    narration_engine: Optional[NarrationEngine] = None,
    stt_engine: Optional[Any] = None,
) -> FastAPI:
    """Create and configure the unified FastAPI application.

    Args:
        ui_dir: Path to static dashboard UI directory. Defaults to project `ui/`.
        connection_manager: Custom ConnectionManager instance. Defaults to default singleton.
        narration_engine: Custom NarrationEngine instance. Defaults to new instance.
        stt_engine: Custom STTEngine or MockSTTEngine instance. Defaults to connection manager's engine.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Interactive Voice Agent HUD Server",
        version="0.3.0",
        description="Unified local offline voice agent server and single-page HUD dashboard",
        lifespan=lifespan,
    )

    # Store connection manager and narration engine in app state
    manager = connection_manager if connection_manager is not None else default_connection_manager
    app.state.connection_manager = manager

    engine = narration_engine if narration_engine is not None else NarrationEngine(connection_manager=manager)
    app.state.narration_engine = engine

    # Store STT engine in app state if provided
    if stt_engine is not None:
        app.state.stt_engine = stt_engine
    elif hasattr(manager, "stt_engine"):
        app.state.stt_engine = manager.stt_engine
    else:
        app.state.stt_engine = None

    # Include WebSocket streaming router at /ws
    app.include_router(websocket_router)

    # =========================================================================
    # REST API Endpoints: Document Ingestion, Config, and Interruption
    # =========================================================================

    @app.post("/api/upload-document")
    async def upload_document(
        request: Request,
        file: Optional[UploadFile] = File(None),
    ):
        """Upload an ebook, PDF, EPUB, TXT, or MD document for narration."""
        eng: NarrationEngine = app.state.narration_engine

        if file is not None and file.filename:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            doc = DocumentParser.parse_bytes(content, filename=file.filename)
        else:
            try:
                data = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="No file or JSON text payload provided.")

            text = data.get("text", "").strip()
            title = data.get("title", "Custom Document").strip()
            if not text:
                raise HTTPException(status_code=400, detail="Text field cannot be empty.")
            doc = DocumentParser.parse_text(text, title=title)

        eng.load_document(doc)
        return {
            "status": "ok",
            "title": doc.title,
            "sentence_count": len(doc.sentences),
            "word_count": doc.total_words,
            "sentences": doc.sentences[:10],
        }

    @app.get("/api/document")
    async def get_document():
        """Retrieve current document metadata and sentences."""
        eng: NarrationEngine = app.state.narration_engine
        return {
            "title": eng.document_title,
            "sentences": eng.sentences,
            "sentence_count": len(eng.sentences),
            "current_sentence_idx": eng.current_sentence_idx,
        }

    @app.get("/api/config")
    async def get_config():
        """Retrieve active LLM and voice runtime configurations."""
        eng: NarrationEngine = app.state.narration_engine
        mgr: ConnectionManager = app.state.connection_manager
        return {
            "status": "ok",
            "llm": eng.llm.get_config(),
            "voice": {
                "active_voice": mgr.active_voice,
                "cadence": mgr.cadence,
                "mic_source": mgr.mic_source,
                "state": mgr.state,
            },
            "document": {
                "title": eng.document_title,
                "sentence_count": len(eng.sentences),
                "current_sentence_idx": eng.current_sentence_idx,
            },
        }

    @app.post("/api/config")
    async def update_config(request: Request):
        """Update runtime LLM endpoint, model, search grounding, or voice cadence."""
        data = await request.json()
        eng: NarrationEngine = app.state.narration_engine
        mgr: ConnectionManager = app.state.connection_manager

        # Update LLM Client
        eng.llm.update_config(
            base_url=data.get("base_url") or data.get("endpoint"),
            api_key=data.get("api_key"),
            provider=data.get("provider"),
            model=data.get("model") or data.get("model_name"),
            web_search_enabled=data.get("web_search_enabled"),
        )

        # Update Manager parameters if provided
        if "active_voice" in data:
            mgr.set_voice(data["active_voice"])
        if "cadence" in data:
            mgr.set_cadence(float(data["cadence"]))
        if "mic_source" in data:
            mgr.set_mic_source(data["mic_source"])

        # Broadcast state synchronization to clients
        await mgr.broadcast_json(mgr.get_state_sync_message())

        return {
            "status": "ok",
            "llm": eng.llm.get_config(),
            "voice": {
                "active_voice": mgr.active_voice,
                "cadence": mgr.cadence,
                "mic_source": mgr.mic_source,
            },
        }

    @app.post("/api/ask")
    async def ask_question(request: Request):
        """Trigger an ad-hoc voice question interruption with live web search grounding."""
        data = await request.json()
        query = str(data.get("query", "")).strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        eng: NarrationEngine = app.state.narration_engine
        asyncio.create_task(eng.handle_user_interruption(query))
        return {"status": "ok", "query": query}

    @app.post("/api/narration/control")
    async def narration_control(request: Request):
        """Programmatic control over narration (resume, pause, next, prev)."""
        data = await request.json()
        action = str(data.get("action", "resume")).lower().strip()
        eng: NarrationEngine = app.state.narration_engine
        mgr: ConnectionManager = app.state.connection_manager

        if action in ("resume", "start"):
            eng.start_narration()
        elif action in ("pause", "stop"):
            eng.pause_narration()
        elif action == "next":
            eng.current_sentence_idx = (eng.current_sentence_idx + 1) % max(1, len(eng.sentences))
            eng.current_word_idx = 0
            mgr.update_cursor(eng.current_sentence_idx, 0)
        elif action == "prev":
            eng.current_sentence_idx = (eng.current_sentence_idx - 1 + len(eng.sentences)) % max(1, len(eng.sentences))
            eng.current_word_idx = 0
            mgr.update_cursor(eng.current_sentence_idx, 0)

        await mgr.broadcast_json(mgr.get_state_sync_message())
        return {"status": "ok", "state": mgr.state, "current_sentence_idx": eng.current_sentence_idx}

    @app.post("/api/transcribe")
    async def transcribe_audio(
        file: UploadFile = File(...),
    ):
        """Transcribe an uploaded audio file (WAV or raw PCM).

        Returns:
            JSON with transcript, duration_sec, and status "ok".
        """
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        pcm_data, duration_sec = extract_audio_and_duration(content)
        stt = get_stt_engine(app)
        transcript = stt.transcribe(pcm_data)
        if transcript is None:
            transcript = ""

        return {
            "transcript": transcript,
            "duration_sec": duration_sec,
            "status": "ok",
        }

    # =========================================================================
    # Mount Static HUD Assets
    # =========================================================================
    if ui_dir is not None:
        resolved_ui = Path(ui_dir).resolve()
    else:
        # Default: check project root / ui
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        resolved_ui = project_root / "ui"
        if not resolved_ui.exists():
            resolved_ui = Path("ui").resolve()

    if resolved_ui.exists() and resolved_ui.is_dir():
        app.mount("/", StaticFiles(directory=str(resolved_ui), html=True), name="ui")
        logger.info(f"Mounted static dashboard assets from: {resolved_ui}")
    else:
        logger.warning(f"Static UI directory not found at: {resolved_ui}")

    return app


# Module-level default application instance for uvicorn
app = create_app()


def run(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """CLI runner helper for local execution."""
    import uvicorn
    uvicorn.run("voice_agent.server.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run()
