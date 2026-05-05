"""Shared fixtures for frontend tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Use a stable per-session temp DB path to avoid mutation between modules
_TEST_DB_PATH = Path("/tmp/pharma_ra_test_session.db")

# ---------------------------------------------------------------------------
# App fixture — session-scoped, mocked lifespan (no Redis)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("db") / "auth.db"


@pytest.fixture(scope="session")
def app(test_db_path):
    """Create a FastAPI test app with mocked lifespan (no Redis needed)."""
    import frontend.db.auth as auth_mod

    # Point auth module at the test DB
    auth_mod._DB_PATH = test_db_path
    auth_mod.init_db_sync()

    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi.templating import Jinja2Templates
    from starlette.middleware.sessions import SessionMiddleware
    from fastapi.responses import RedirectResponse

    templates_dir = Path(__file__).parent.parent.parent / "frontend" / "templates"

    @asynccontextmanager
    async def mock_lifespan(application: FastAPI):
        application.state.templates = Jinja2Templates(directory=str(templates_dir))
        application.state.graph_classic = None
        application.state.graph_wiki = None
        application.state.ready = False
        yield

    from frontend.routers.auth_router import router as auth_router
    from frontend.routers.chat_router import router as chat_router
    from frontend.routers.admin_router import router as admin_router

    test_app = FastAPI(lifespan=mock_lifespan)
    test_app.add_middleware(
        SessionMiddleware,
        secret_key="test_secret_key",
        session_cookie="pharma_ra_session",
        max_age=None,
        https_only=False,
        same_site="lax",
    )
    test_app.include_router(auth_router)
    test_app.include_router(chat_router)
    test_app.include_router(admin_router)

    @test_app.get("/")
    async def root():
        return RedirectResponse(url="/chat", status_code=302)

    # ASGITransport does not fire lifespan events — initialize state directly
    test_app.state.templates = Jinja2Templates(directory=str(templates_dir))
    test_app.state.graph_classic = None
    test_app.state.graph_wiki = None
    test_app.state.ready = False

    return test_app


# ---------------------------------------------------------------------------
# HTTP client fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client(app):
    """Unauthenticated async test client."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def admin_client(app):
    """Authenticated admin async client."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/login", data={"username": "admin", "password": "admin123"})
        assert resp.status_code in (200, 302), f"Admin login failed: {resp.status_code}"
        yield c


@pytest_asyncio.fixture
async def researcher_client(app):
    """Authenticated researcher async client."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/login", data={"username": "researcher", "password": "researcher123"})
        assert resp.status_code in (200, 302), f"Researcher login failed: {resp.status_code}"
        yield c
