"""PharmaRA FastAPI application — professional web frontend."""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

# Make project root importable (agent/, rag/, chat/ etc.)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from frontend.config import settings
from frontend.db.auth import init_db_sync

# ---------------------------------------------------------------------------
# Lifespan: load vectorstores + build agent graphs once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auth DB
    init_db_sync()

    # Vectorstores & graphs
    try:
        from rag.vectorstore import load_index, load_wiki_index
        from agent.graph import build_graph

        vs = load_index()
        wiki_vs = load_wiki_index()

        app.state.graph_classic = build_graph(vs, None, mode="classic")
        app.state.graph_wiki = build_graph(vs, wiki_vs, mode="wiki")
        app.state.ready = True
    except Exception as exc:
        print(f"[startup] WARNING: Could not load agent graphs: {exc}")
        app.state.graph_classic = None
        app.state.graph_wiki = None
        app.state.ready = False

    yield
    # Shutdown: nothing to clean up for Redis/LangGraph


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="PharmaRA", lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret_key,
    session_cookie="pharma_ra_session",
    max_age=None,  # session-only cookie (expires when browser closes)
    https_only=False,
    same_site="lax",
)

# Mount static files
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Templates (shared across routers via app.state)
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
app.state.templates = templates

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from frontend.routers.auth_router import router as auth_router
from frontend.routers.chat_router import router as chat_router
from frontend.routers.admin_router import router as admin_router

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(admin_router)


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------

from fastapi.responses import RedirectResponse

@app.get("/")
async def root():
    return RedirectResponse(url="/chat", status_code=302)
