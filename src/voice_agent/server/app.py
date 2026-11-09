"""FastAPI Application Factory & Static HUD Dashboard Server.

Hosts the single-page HUD dashboard static assets (`ui/`) at `http://localhost:8000/`
and bi-directional streaming WebSocket endpoint at `ws://localhost:8000/ws`.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Optional, Union
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from voice_agent.server.connection_manager import (
    ConnectionManager,
    default_connection_manager,
)
from voice_agent.server.websocket import router as websocket_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for graceful startup and shutdown."""
    logger.info("Starting up Interactive Voice Agent Server.")
    yield
    logger.info("Shutting down Interactive Voice Agent Server gracefully.")
    manager: Optional[ConnectionManager] = getattr(app.state, "connection_manager", None)
    if manager is not None:
        await manager.disconnect_all()


def create_app(
    ui_dir: Optional[Union[str, Path]] = None,
    connection_manager: Optional[ConnectionManager] = None,
) -> FastAPI:
    """Create and configure the unified FastAPI application.

    Args:
        ui_dir: Path to static dashboard UI directory. Defaults to project `ui/`.
        connection_manager: Custom ConnectionManager instance. Defaults to default singleton.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Interactive Voice Agent HUD Server",
        version="0.3.0",
        description="Unified local offline voice agent server and single-page HUD dashboard",
        lifespan=lifespan,
    )

    # Store connection manager in app state
    manager = connection_manager if connection_manager is not None else default_connection_manager
    app.state.connection_manager = manager

    # Include WebSocket streaming router at /ws
    app.include_router(websocket_router)

    # Resolve UI directory path
    if ui_dir is not None:
        resolved_ui = Path(ui_dir).resolve()
    else:
        # Default: check project root / ui
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        resolved_ui = project_root / "ui"
        if not resolved_ui.exists():
            resolved_ui = Path("ui").resolve()

    # Mount static assets at root "/"
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
