# -*- coding: utf-8 -*-
"""FastAPI application using LangGraph runner.

This is a complete rewrite to use LangGraph as the underlying
agent framework, replacing the AgentScope-based implementation.
"""
import mimetypes
import os
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from copaw.langgraph.factory import get_langgraph_runner_factory
from copaw.langgraph.adapters import ChannelMessageAdapter
from ..config import load_config
from ..config.utils import get_config_path
from ..constant import DOCS_ENABLED, LOG_LEVEL_ENV, CORS_ORIGINS, WORKING_DIR
from ..__version__ import __version__
from ..utils.logging import setup_logger, add_copaw_file_handler
from .auth import AuthMiddleware
from .routers import router as api_router, create_agent_scoped_router
from .routers.agent_scoped import AgentContextMiddleware
from .routers.voice import voice_router
from .multi_agent_manager import MultiAgentManager
from ..envs import load_envs_into_environ
from ..providers.provider_manager import ProviderManager
from ..local_models.manager import LocalModelManager
from .migration import (
    migrate_legacy_workspace_to_default_agent,
    migrate_legacy_skills_to_skill_pool,
    ensure_default_agent_exists,
    ensure_qa_agent_exists,
)
from .channels.registry import register_custom_channel_routes

# Apply log level
logger = setup_logger(os.environ.get(LOG_LEVEL_ENV, "info"))

# Ensure static assets are served with browser-compatible MIME types
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/wasm", ".wasm")

# Load persisted env vars
load_envs_into_environ()


# ========== LangGraph Runner ==========
class LangGraphRunnerManager:
    """Manages LangGraph runners for different agents."""

    def __init__(self):
        self._factory = get_langgraph_runner_factory()
        self._runner_cache = {}

    def get_runner(self, agent_id: str, workspace_dir: Path):
        """Get or create a runner for the given agent."""
        if agent_id in self._runner_cache:
            return self._runner_cache[agent_id]

        runner = self._factory.get_runner(
            agent_id=agent_id,
            workspace_dir=workspace_dir,
        )
        self._runner_cache[agent_id] = runner
        return runner

    async def handle_chat(
        self,
        agent_id: str,
        message: str,
        session_id: str,
        user_id: str,
        channel: str,
        workspace_dir: Path,
    ):
        """Handle a chat message using LangGraph."""
        runner = self.get_runner(agent_id, workspace_dir)

        # Create a mock request for the runner
        class MockRequest:
            def __init__(self):
                self.session_id = session_id
                self.user_id = user_id
                self.channel = channel

        # Execute using query_handler
        async for msg, last in runner.query_handler(
            msgs=[{"role": "user", "content": message}],
            request=MockRequest(),
        ):
            yield msg


# Global runner manager
_runner_manager = LangGraphRunnerManager()


# ========== Request Models ==========
class ChatRequest(BaseModel):
    message: str
    agent_id: str = "default"
    session_id: str | None = None
    user_id: str = "default"
    channel: str = "console"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    agent_id: str


# ========== FastAPI App ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    startup_start_time = time.time()
    add_copaw_file_handler(WORKING_DIR / "copaw.log")

    # Auto-register admin
    from .auth import auto_register_from_env
    auto_register_from_env()

    # Telemetry
    try:
        from ..utils.telemetry import (
            collect_and_upload_telemetry,
            has_telemetry_been_collected,
            is_telemetry_opted_out,
        )
        if not is_telemetry_opted_out(WORKING_DIR) and not has_telemetry_been_collected(WORKING_DIR):
            collect_and_upload_telemetry(WORKING_DIR)
    except Exception:
        logger.debug("Telemetry collection skipped", exc_info=True)

    # Migration
    logger.info("Checking for legacy config migration...")
    migrate_legacy_workspace_to_default_agent()
    ensure_default_agent_exists()
    migrate_legacy_skills_to_skill_pool()
    ensure_qa_agent_exists()

    # Model provider manager
    provider_manager = ProviderManager.get_instance()
    local_model_manager = LocalModelManager.get_instance()

    # Multi-agent manager
    multi_agent_manager = MultiAgentManager()

    app.state.provider_manager = provider_manager
    app.state.local_model_manager = local_model_manager
    app.state.runner_manager = _runner_manager
    app.state.multi_agent_manager = multi_agent_manager

    provider_manager.start_local_model_resume(local_model_manager)

    startup_elapsed = time.time() - startup_start_time
    logger.info(f"Application startup completed in {startup_elapsed:.3f} seconds")

    try:
        yield
    finally:
        # Cleanup
        local_model_mgr = getattr(app.state, "local_model_manager", None)
        if local_model_mgr is not None:
            logger.info("Stopping local model server...")
            try:
                await local_model_mgr.shutdown_server()
            except Exception as exc:
                logger.error(f"Error shutting down local model server: {exc}")
                with suppress(OSError, RuntimeError, ValueError):
                    local_model_mgr.force_shutdown_server()

        logger.info("Application shutdown complete")


app = FastAPI(
    title="CoPaw API",
    description="AI Assistant API using LangGraph",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
)

# Middleware
app.add_middleware(AgentContextMiddleware)
app.add_middleware(AuthMiddleware)

if CORS_ORIGINS:
    origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )


# ========== Console Static Files ==========
_CONSOLE_STATIC_ENV = "COPAW_CONSOLE_STATIC_DIR"


def _resolve_console_static_dir() -> str:
    if os.environ.get(_CONSOLE_STATIC_ENV):
        return os.environ[_CONSOLE_STATIC_ENV]
    pkg_dir = Path(__file__).resolve().parent.parent
    candidate = pkg_dir / "console"
    if candidate.is_dir() and (candidate / "index.html").exists():
        return str(candidate)
    repo_dir = pkg_dir.parent.parent
    candidate = repo_dir / "console" / "dist"
    if candidate.is_dir() and (candidate / "index.html").exists():
        return str(candidate)
    cwd = Path(os.getcwd())
    for subdir in ("console/dist", "console_dist"):
        candidate = cwd / subdir
        if candidate.is_dir() and (candidate / "index.html").exists():
            return str(candidate)
    fallback = cwd / "console" / "dist"
    logger.warning(f"Console static directory not found. Falling back to '{fallback}'.")
    return str(fallback)


_CONSOLE_STATIC_DIR = _resolve_console_static_dir()
_CONSOLE_INDEX = Path(_CONSOLE_STATIC_DIR) / "index.html" if _CONSOLE_STATIC_DIR else None
logger.info(f"STATIC_DIR: {_CONSOLE_STATIC_DIR}")


# ========== Routes ==========
@app.get("/")
def read_root():
    if _CONSOLE_INDEX and _CONSOLE_INDEX.exists():
        return FileResponse(_CONSOLE_INDEX)
    return {
        "message": "CoPaw Web Console is not available. Run `npm ci && npm run build` in console/ directory.",
    }


@app.get("/api/version")
def get_version():
    return {"version": __version__}


# Chat endpoint using LangGraph
@app.post("/api/agent/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Simple chat endpoint using LangGraph runner."""
    import uuid

    session_id = request.session_id or str(uuid.uuid4())
    workspace_dir = WORKING_DIR / "agents" / request.agent_id

    response_text = ""
    try:
        async for msg, last in _runner_manager.handle_chat(
            agent_id=request.agent_id,
            message=request.message,
            session_id=session_id,
            user_id=request.user_id,
            channel=request.channel,
            workspace_dir=workspace_dir,
        ):
            if hasattr(msg, "content"):
                response_text = msg.content
    except Exception as e:
        logger.error(f"Chat error: {e}")
        response_text = f"Error: {str(e)}"

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        agent_id=request.agent_id,
    )


# Streaming chat endpoint
@app.post("/api/agent/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using LangGraph runner."""
    import uuid

    session_id = request.session_id or str(uuid.uuid4())
    workspace_dir = WORKING_DIR / "agents" / request.agent_id

    async def generate():
        try:
            async for msg, last in _runner_manager.handle_chat(
                agent_id=request.agent_id,
                message=request.message,
                session_id=session_id,
                user_id=request.user_id,
                channel=request.channel,
                workspace_dir=workspace_dir,
            ):
                if hasattr(msg, "content"):
                    yield f"data: {msg.content}\n\n"
                if last:
                    break
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


# Include routers
app.include_router(api_router, prefix="/api")

agent_scoped_router = create_agent_scoped_router()
app.include_router(agent_scoped_router, prefix="/api")

app.include_router(voice_router, tags=["voice"])

# Custom channel routes
register_custom_channel_routes(app)


# Static files and SPA fallback
if os.path.isdir(_CONSOLE_STATIC_DIR):
    _console_path = Path(_CONSOLE_STATIC_DIR)

    def _serve_console_index():
        if _CONSOLE_INDEX and _CONSOLE_INDEX.exists():
            return FileResponse(_CONSOLE_INDEX)
        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/logo.png")
    def _console_logo():
        f = _console_path / "logo.png"
        if f.is_file():
            return FileResponse(f, media_type="image/png")
        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/dark-logo.png")
    def _console_dark_logo():
        f = _console_path / "dark-logo.png"
        if f.is_file():
            return FileResponse(f, media_type="image/png")
        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/copaw-symbol.svg")
    def _console_icon():
        f = _console_path / "copaw-symbol.svg"
        if f.is_file():
            return FileResponse(f, media_type="image/svg+xml")
        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/copaw-dark.png")
    def _console_dark_icon():
        f = _console_path / "copaw-dark.png"
        if f.is_file():
            return FileResponse(f, media_type="image/png")
        raise HTTPException(status_code=404, detail="Not Found")

    _assets_dir = _console_path / "assets"
    if _assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

    @app.get("/console")
    @app.get("/console/")
    @app.get("/console/{full_path:path}")
    def _console_spa_alias(full_path: str = ""):
        return _serve_console_index()

    @app.get("/{full_path:path}")
    def _console_spa(full_path: str):
        if full_path in ("docs", "redoc", "openapi.json"):
            raise HTTPException(status_code=404, detail="Not Found")
        if full_path.startswith("api/") or full_path == "api":
            raise HTTPException(status_code=404, detail="Not Found")
        return _serve_console_index()