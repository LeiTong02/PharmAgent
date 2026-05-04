"""Integration tests for login/logout flow and protected routes."""
from __future__ import annotations

from httpx import ASGITransport, AsyncClient


async def test_root_redirects_to_chat(client):
    resp = await client.get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/chat"


async def test_chat_redirects_unauthenticated(client):
    resp = await client.get("/chat", follow_redirects=False)
    assert resp.status_code in (302, 307)


async def test_login_page_renders(client):
    resp = await client.get("/login")
    assert resp.status_code == 200
    assert b"Sign in" in resp.content or b"PharmaRA" in resp.content


async def test_login_valid_credentials_redirects(client):
    resp = await client.post(
        "/login",
        data={"username": "admin", "password": "admin123"},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert resp.headers["location"] == "/chat"


async def test_login_invalid_credentials_shows_error(client):
    resp = await client.post(
        "/login",
        data={"username": "admin", "password": "wrongpassword"},
        follow_redirects=True,
    )
    assert resp.status_code in (200, 401)
    assert b"Invalid" in resp.content or b"invalid" in resp.content


async def test_logout_redirects_to_login(admin_client):
    resp = await admin_client.get("/logout", follow_redirects=False)
    assert resp.status_code == 302
    assert "login" in resp.headers["location"]


async def test_admin_accessible_to_admin(admin_client):
    from unittest.mock import patch
    with patch("rag.vectorstore.list_uploaded_files", return_value=[]):
        resp = await admin_client.get("/admin", follow_redirects=True)
    assert resp.status_code == 200


async def test_admin_denied_to_researcher(researcher_client):
    resp = await researcher_client.get("/admin", follow_redirects=True)
    assert resp.status_code == 403


async def test_session_cookie_set_on_login(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        # Session cookie must be present
        assert len(resp.cookies) > 0 or "set-cookie" in {k.lower() for k in resp.headers}
