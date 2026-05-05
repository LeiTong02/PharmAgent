"""Unit tests for SQLite auth backend — uses its own isolated DB per test."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Give each test its own fresh SQLite DB; restore the module global afterwards."""
    import frontend.db.auth as auth_mod

    test_db = tmp_path / "auth.db"
    monkeypatch.setattr(auth_mod, "_DB_PATH", test_db)
    auth_mod.init_db_sync()
    yield test_db


def _get_db_path():
    import frontend.db.auth as auth_mod
    return auth_mod._DB_PATH


def test_init_db_creates_table():
    import frontend.db.auth as auth_mod
    assert auth_mod._DB_PATH.exists()
    with sqlite3.connect(auth_mod._DB_PATH) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "users" in tables


def test_seed_creates_admin_and_researcher():
    import frontend.db.auth as auth_mod
    with sqlite3.connect(auth_mod._DB_PATH) as conn:
        rows = conn.execute("SELECT username, role FROM users ORDER BY username").fetchall()
    role_map = {r[0]: r[1] for r in rows}
    assert role_map.get("admin") == "admin"
    assert role_map.get("researcher") == "researcher"


def test_passwords_are_hashed():
    import frontend.db.auth as auth_mod
    with sqlite3.connect(auth_mod._DB_PATH) as conn:
        hashes = [r[0] for r in conn.execute("SELECT password_hash FROM users")]
    for h in hashes:
        # bcrypt hashes start with $2b$ or $2a$
        assert h.startswith("$2b$") or h.startswith("$2a$"), f"Not bcrypt: {h[:20]}"
        assert "admin" not in h
        assert "researcher" not in h


def test_seed_is_idempotent():
    """Calling init_db_sync twice must not duplicate users."""
    import frontend.db.auth as auth_mod
    auth_mod.init_db_sync()  # second call
    with sqlite3.connect(auth_mod._DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    assert count == 2


async def test_get_user_found():
    from frontend.db.auth import get_user
    user = await get_user("admin")
    assert user is not None
    assert user["username"] == "admin"
    assert user["role"] == "admin"
    assert "password_hash" in user


async def test_get_user_not_found():
    from frontend.db.auth import get_user
    user = await get_user("nonexistent_user_xyz")
    assert user is None


async def test_verify_password_correct():
    from frontend.db.auth import verify_password
    user = await verify_password("admin", "admin123")
    assert user is not None
    assert user["username"] == "admin"


async def test_verify_password_wrong():
    from frontend.db.auth import verify_password
    user = await verify_password("admin", "wrongpassword")
    assert user is None


async def test_verify_password_nonexistent_user():
    from frontend.db.auth import verify_password
    user = await verify_password("ghost_user", "anything")
    assert user is None


async def test_verify_researcher_credentials():
    from frontend.db.auth import verify_password
    user = await verify_password("researcher", "researcher123")
    assert user is not None
    assert user["role"] == "researcher"
