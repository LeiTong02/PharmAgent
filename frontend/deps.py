"""FastAPI dependency functions for authentication and authorization."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status


def get_current_user(request: Request) -> dict:
    """Read user from signed session. Raises 401 if not authenticated."""
    user = request.session.get("user")
    if not user:
        raise HTTPException(
            status_code=status.HTTP_302_FOUND,
            headers={"Location": "/login"},
        )
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Allow only admin role. Raises 403 otherwise."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required.")
    return user
