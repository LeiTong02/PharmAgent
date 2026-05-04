"""Authentication routes: login page, login form handler, logout."""
from __future__ import annotations

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from frontend.db.auth import verify_password

router = APIRouter(tags=["auth"])


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = ""):
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "login.html", context={"error": error})


@router.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    user = await verify_password(username, password)
    if user is None:
        templates = request.app.state.templates
        return templates.TemplateResponse(
            request,
            "login.html",
            context={"error": "Invalid username or password."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    request.session["user"] = {"username": user["username"], "role": user["role"]}
    return RedirectResponse(url="/chat", status_code=status.HTTP_302_FOUND)


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
