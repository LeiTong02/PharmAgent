"""SQLite-backed user store with bcrypt password hashing."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import bcrypt as _bcrypt

from frontend.config import settings

_DB_PATH = Path(__file__).parent.parent / "auth.db"


def _hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()


def _verify_password_hash(plain: str, hashed: str) -> bool:
    try:
        return _bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Synchronous init (called once at startup, outside async context)
# ---------------------------------------------------------------------------

def init_db_sync() -> None:
    """Create users table and seed admin/researcher from settings if not present."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin','researcher'))
            )"""
        )
        conn.commit()
        _seed_user_sync(conn, settings.auth_admin_username, settings.auth_admin_password, "admin")
        _seed_user_sync(conn, settings.auth_researcher_username, settings.auth_researcher_password, "researcher")


def _seed_user_sync(conn: sqlite3.Connection, username: str, password: str, role: str) -> None:
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if not existing:
        hashed = _hash_password(password)
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, hashed, role),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Async helpers (used in request handlers)
# ---------------------------------------------------------------------------

async def get_user(username: str) -> dict | None:
    """Return {username, role, password_hash} or None if not found."""
    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT username, password_hash, role FROM users WHERE username = ?",
            (username,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)


async def verify_password(username: str, plain: str) -> dict | None:
    """Return user dict if credentials are valid, else None."""
    user = await get_user(username)
    if user is None:
        return None
    if not _verify_password_hash(plain, user["password_hash"]):
        return None
    return user
