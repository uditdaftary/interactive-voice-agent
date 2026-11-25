"""FastAPI Application Factory & Static HUD Dashboard Server.

Hosts the single-page HUD dashboard static assets (`ui/`) at `http://localhost:8000/`,
bi-directional streaming WebSocket endpoint at `ws://localhost:8000/ws`,
and REST endpoints for document ingestion, LLM configuration, and Q&A interruption.
"""

from contextlib import asynccontextmanager
import io
import logging
from pathlib import Path
from typing import Annotated, Any, Optional, Union
import wave
import numpy as np
from scipy.signal import resample_poly
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from fastapi.staticfiles import StaticFiles

from voice_agent.document.parser import DOCUMENT_EXTENSIONS, DocumentParseError, DocumentParser
from voice_agent.server.connection_manager import (
    ConnectionManager,
    default_connection_manager,
)
from voice_agent.server import security
from voice_agent.server.narration import NarrationEngine, SearchFn, TtsFn
from voice_agent.server.schemas import (
    AskRequest,
    ConfigUpdateRequest,
    PlaybackControlRequest,
    TextDocumentRequest,
)
from voice_agent.server.websocket import router as websocket_router
from voice_agent.settings import Settings, load_settings
import asyncio

logger = logging.getLogger(__name__)


AUDIO_EXTENSIONS = frozenset({".wav", ".pcm", ".raw"})
STT_SAMPLE_RATE = 16000
# Real audio sits well inside this. A header rate outside it is garbage, and it would otherwise
# choose the resampler's filter length (memory) for us.
MIN_UPLOAD_SAMPLE_RATE = 8000
MAX_UPLOAD_SAMPLE_RATE = 192000


class UnsupportedAudio(ValueError):
    """Valid audio in a form the transcriber does not take (maps to 415)."""


class InvalidAudio(ValueError):
    """Bytes that are not the audio they claim to be (maps to 400)."""


def extract_audio(content: bytes) -> tuple[np.ndarray, int]:
    """Decode an upload to mono int16 samples and the sample rate they were recorded at.

    A RIFF/WAVE container must hold 16-bit PCM (channels are averaged). Anything else is raw
    16-bit little-endian mono PCM at 16 kHz.
    """
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(content), "rb") as wf:
                width, channels, rate = wf.getsampwidth(), wf.getnchannels(), wf.getframerate()
                if width != 2:
                    raise UnsupportedAudio(f"{width * 8}-bit WAV is not supported; send 16-bit PCM.")
                if not MIN_UPLOAD_SAMPLE_RATE <= rate <= MAX_UPLOAD_SAMPLE_RATE:
                    raise UnsupportedAudio(
                        f"Sample rate {rate} Hz is outside {MIN_UPLOAD_SAMPLE_RATE}-{MAX_UPLOAD_SAMPLE_RATE} Hz."
                    )
                frames = wf.readframes(wf.getnframes())
        except wave.Error as exc:
            # The stdlib reader only decodes PCM; any other format tag is valid audio we cannot take.
            if "unknown format" in str(exc):
                raise UnsupportedAudio("Only 16-bit PCM WAV is supported.") from exc
            raise InvalidAudio(f"Unreadable WAV file: {exc}") from exc
        except EOFError as exc:
            raise InvalidAudio("Unreadable WAV file: truncated header.") from exc
        usable = len(frames) - len(frames) % (2 * channels)  # drop a trailing partial frame
        samples = np.frombuffer(frames[:usable], dtype="<i2")
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    else:
        if len(content) % 2:
            raise InvalidAudio("Raw PCM must be 16-bit samples, but the byte count is odd.")
        samples, rate = np.frombuffer(content, dtype="<i2"), STT_SAMPLE_RATE

    if samples.size == 0:
        raise InvalidAudio("The audio contains no samples.")
    return samples, rate


def prepare_transcription_audio(content: bytes) -> tuple[bytes, float]:
    """Upload bytes to 16 kHz int16 PCM for the transcriber, plus the original duration in seconds.

    CPU-bound (decode and resample): call it through `asyncio.to_thread`.
    """
    samples, rate = extract_audio(content)
    duration_sec = round(len(samples) / rate, 3)
    if rate != STT_SAMPLE_RATE:
        resampled = resample_poly(samples.astype(np.float32), STT_SAMPLE_RATE, rate)
        samples = np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)
    return samples.astype("<i2").tobytes(), duration_sec


def _stored_stt_engine(app: FastAPI) -> Any:
    """The engine already on app.state or the connection manager (tests set it there), else None."""
    engine = getattr(app.state, "stt_engine", None)
    if engine is None:
        manager = getattr(app.state, "connection_manager", None)
        engine = manager.stt_engine if manager is not None else None
        if engine is not None:
            app.state.stt_engine = engine
    return engine


def _construct_stt_engine(initial_prompt: str) -> Any:
    """Runs in the build thread: constructs the engine and loads Whisper here, so no caller's thread pays for the model."""
    from voice_agent.stt.recognizer import STTEngine

    engine = STTEngine(initial_prompt=initial_prompt)
    # A failed load is logged by the backend and retried after its cooldown; the engine is stored either way.
    engine.load()
    return engine


async def _build_stt_engine(app: FastAPI) -> Any:
    settings: Settings | None = getattr(app.state, "settings", None)
    if settings is None:
        # An app built without create_app has no settings on its state; it takes the environment's, as create_app would.
        settings = load_settings()
    try:
        engine = await asyncio.to_thread(_construct_stt_engine, settings.stt_initial_prompt)
        app.state.stt_engine = engine
        return engine
    finally:
        # Success is now served from app.state.stt_engine; after a failure the next caller retries.
        app.state.stt_build = None


async def ensure_stt_engine(app: FastAPI) -> Any:
    """Return the app's STT engine, building the real one on first use (M2).

    Loading Whisper takes seconds and may reach the Hugging Face Hub, so it never happens at
    import. The build runs as one task in a worker thread. Callers await it through a shield, so
    concurrent first utterances share the build and a cancelled caller (a closed websocket) does
    not discard a half-loaded model that the next caller would otherwise load a second time.
    """
    engine = _stored_stt_engine(app)
    if engine is not None:
        return engine
    build = getattr(app.state, "stt_build", None)
    if build is None or build.done():
        # No await between the check and the set, so concurrent first callers share one task.
        build = app.state.stt_build = asyncio.create_task(_build_stt_engine(app))
    return await asyncio.shield(build)


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
    *,
    settings: Optional[Settings] = None,
    llm_client: Optional[Any] = None,
    tts: Optional[TtsFn] = None,
    search: Optional[SearchFn] = None,
    allowed_hosts: Optional[list[str]] = None,
) -> FastAPI:
    """Create and configure the unified FastAPI application.

    Args:
        ui_dir: Path to static dashboard UI directory. Defaults to project `ui/`.
        connection_manager: Custom ConnectionManager instance. Defaults to default singleton.
        narration_engine: Custom NarrationEngine instance. Defaults to new instance.
        stt_engine: Custom STT engine (any object with a `transcribe` method). Defaults to the connection manager's engine,
            else one is built on first use.
        settings: Timing and endpoint settings for the default NarrationEngine. Defaults to `load_settings()`.
        llm_client: LLM client for the default NarrationEngine. Defaults to one built from `settings`.
        tts: Async speech synthesizer for the default NarrationEngine. Defaults to `synthesize_sentence`.
        search: Async web search for the default NarrationEngine. Defaults to `search_web`.
        The last four are ignored when `narration_engine` is supplied.
        allowed_hosts: Host header names the server answers to. Defaults to loopback only; tests add `testserver`.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Interactive Voice Agent HUD Server",
        version="0.3.0",
        description="Unified local offline voice agent server and single-page HUD dashboard",
        lifespan=lifespan,
    )

    # Last added is outermost, so a bad Host is rejected before the Origin check runs.
    app.add_middleware(security.OriginCheckMiddleware)
    app.add_middleware(security.LoggingTrustedHostMiddleware, allowed_hosts=allowed_hosts or list(security.DEFAULT_ALLOWED_HOSTS))

    # Store connection manager and narration engine in app state
    # Resolved once, so the narration engine and the STT build read the same settings.
    resolved_settings = settings if settings is not None else load_settings()
    app.state.settings = resolved_settings

    manager = connection_manager if connection_manager is not None else default_connection_manager
    app.state.connection_manager = manager

    engine = (
        narration_engine
        if narration_engine is not None
        else NarrationEngine(
            connection_manager=manager,
            llm_client=llm_client,
            settings=resolved_settings,
            tts=tts,
            search=search,
        )
    )
    app.state.narration_engine = engine

    # STT is built on first use by ensure_stt_engine; only an injected or pre-set engine is stored here.
    app.state.stt_engine = stt_engine if stt_engine is not None else manager.stt_engine

    # Include WebSocket streaming router at /ws
    app.include_router(websocket_router)

    # =========================================================================
    # REST API Endpoints: Document Ingestion, Config, and Interruption
    # =========================================================================

    @app.post("/api/upload-document")
    async def upload_document(
        request: Request,
        file: Annotated[UploadFile | None, File()] = None,
    ):
        """Upload an ebook, PDF, EPUB, TXT, or MD document for narration."""
        eng: NarrationEngine = app.state.narration_engine

        if file is not None and file.filename:
            if Path(file.filename).suffix.lower() not in DOCUMENT_EXTENSIONS:
                raise HTTPException(
                    status_code=415, detail=f"Unsupported file type. Allowed: {', '.join(sorted(DOCUMENT_EXTENSIONS))}."
                )
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            try:
                # Parsing is CPU-bound; on the event loop it would stall every WebSocket for its duration.
                doc = await asyncio.to_thread(DocumentParser.parse_bytes, content, file.filename)
            except DocumentParseError as exc:
                logger.warning("Rejected upload %r: %s", file.filename, exc)
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        else:
            if not security.is_json_content_type(request.headers.get("content-type")):
                raise HTTPException(status_code=415, detail="Send a multipart file or a JSON body.")
            try:
                payload = TextDocumentRequest.model_validate_json(await request.body())
            except ValidationError as exc:
                raise RequestValidationError(exc.errors(include_url=False, include_context=False, include_input=False)) from None
            doc = await asyncio.to_thread(DocumentParser.parse_text, payload.text, payload.title)

        if not doc.sentences:
            # The old document stays loaded. Say what was skipped: it is usually why nothing was left.
            rejection: dict[str, Any] = {"detail": "document contains no readable text"}
            if doc.skipped_parts:
                rejection["skipped_parts"] = doc.skipped_parts
            return JSONResponse(rejection, status_code=422)

        eng.load_document(doc)
        return {
            "status": "ok",
            "title": doc.title,
            "sentence_count": len(doc.sentences),
            "word_count": doc.total_words,
            "sentences": doc.sentences[:10],
            "skipped_parts": doc.skipped_parts,
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

    @app.post("/api/config", dependencies=[Depends(security.require_json_content_type)])
    async def update_config(body: ConfigUpdateRequest):
        """Update runtime LLM endpoint, model, search grounding, or voice cadence.

        FastAPI validates the whole body before this runs, so an invalid field applies nothing.
        """
        eng: NarrationEngine = app.state.narration_engine
        mgr: ConnectionManager = app.state.connection_manager

        eng.llm.update_config(
            base_url=body.base_url,
            api_key=body.api_key,
            provider=body.provider,
            model=body.model,
            web_search_enabled=body.web_search_enabled,
        )
        if body.active_voice is not None:
            mgr.set_voice(body.active_voice)
        if body.cadence is not None:
            mgr.set_cadence(body.cadence)
        if body.mic_source is not None:
            mgr.set_mic_source(body.mic_source)

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

    @app.post("/api/ask", dependencies=[Depends(security.require_json_content_type)])
    async def ask_question(body: AskRequest):
        """Trigger an ad-hoc voice question interruption with live web search grounding."""
        eng: NarrationEngine = app.state.narration_engine
        asyncio.create_task(eng.handle_user_interruption(body.query))  # noqa: RUF006  # S2.1
        return {"status": "ok", "query": body.query}

    @app.post("/api/narration/control", dependencies=[Depends(security.require_json_content_type)])
    async def narration_control(body: PlaybackControlRequest):
        """Programmatic control over narration (resume, pause, next, prev)."""
        action = body.action
        eng: NarrationEngine = app.state.narration_engine
        mgr: ConnectionManager = app.state.connection_manager
        state_before = mgr.state

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

        if mgr.state == state_before:
            # No transition happened, so the state listener announced nothing; still answer the request.
            await mgr.broadcast_json(mgr.get_state_sync_message())
        return {"status": "ok", "state": mgr.state, "current_sentence_idx": eng.current_sentence_idx}

    @app.post("/api/transcribe")
    async def transcribe_audio(
        file: Annotated[UploadFile, File()],
    ):
        """Transcribe an uploaded audio file (WAV or raw PCM).

        Returns:
            JSON with transcript, duration_sec, and status "ok".
        """
        if Path(file.filename or "").suffix.lower() not in AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=415, detail=f"Unsupported file type. Allowed: {', '.join(sorted(AUDIO_EXTENSIONS))}."
            )
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        try:
            pcm_data, duration_sec = await asyncio.to_thread(prepare_transcription_audio, content)
        except UnsupportedAudio as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except InvalidAudio as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        stt = await ensure_stt_engine(app)
        transcript = await asyncio.to_thread(stt.transcribe, pcm_data)
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


def run(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """CLI runner helper for local execution."""
    import uvicorn

    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "Binding to %s exposes the server beyond this machine. It has no login, and the Host allowlist "
            "still only accepts loopback names.",
            host,
        )
    uvicorn.run("voice_agent.server.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run()
