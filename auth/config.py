"""Authentication helpers for PharmaRA.

Uses streamlit-authenticator 0.4.x with credentials built from .env vars.
Passwords are stored in plain text in .env; auto_hash=True hashes them at
startup so they are never stored in session state.

Important: do NOT wrap build_authenticator() with @st.cache_resource.
The Authenticate object reads cookies/session state during __init__, which
must happen on every script rerun — caching it causes CachedWidgetWarning
and stale cookie state.
"""
import os
import streamlit as st
import streamlit_authenticator as stauth


_COOKIE_NAME = "pharma_ra_auth"
_COOKIE_KEY = "pharma_ra_secret_key_2026"
_COOKIE_EXPIRY_DAYS = 1

# Keys that streamlit-authenticator 0.4.x expects to exist in session state
_REQUIRED_SESSION_KEYS = {
    "authentication_status": None,
    "username": None,
    "name": None,
    "logout": None,
}


def _build_credentials() -> dict:
    """Build stauth credentials dict from .env vars."""
    return {
        "usernames": {
            os.getenv("AUTH_ADMIN_USERNAME", "admin"): {
                "email": "admin@pharma-ra.demo",
                "first_name": "Admin",
                "last_name": "",
                "password": os.getenv("AUTH_ADMIN_PASSWORD", "admin123"),
                "roles": ["admin"],
                "logged_in": False,
                "failed_login_attempts": 0,
            },
            os.getenv("AUTH_RESEARCHER_USERNAME", "researcher"): {
                "email": "researcher@pharma-ra.demo",
                "first_name": "Researcher",
                "last_name": "",
                "password": os.getenv("AUTH_RESEARCHER_PASSWORD", "researcher123"),
                "roles": ["researcher"],
                "logged_in": False,
                "failed_login_attempts": 0,
            },
        }
    }


def build_authenticator() -> stauth.Authenticate:
    """Return a configured Authenticate instance.

    Call this at the top of each page script without @st.cache_resource.
    """
    for key, default in _REQUIRED_SESSION_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = default

    return stauth.Authenticate(
        _build_credentials(),
        cookie_name=_COOKIE_NAME,
        cookie_key=_COOKIE_KEY,
        cookie_expiry_days=_COOKIE_EXPIRY_DAYS,
        auto_hash=True,
    )


def login_gate(authenticator: stauth.Authenticate) -> str:
    """Render login form if not authenticated; return role ('admin'|'researcher') when logged in.

    Calls st.stop() if the user is not authenticated, so the rest of the page
    never executes for unauthenticated visitors.
    """
    auth_status = st.session_state.get("authentication_status")

    if not auth_status:
        st.title("🧬 PharmaRA — Login")
        authenticator.login(location="main", max_login_attempts=5)

        auth_status = st.session_state.get("authentication_status")

        if auth_status is False:
            st.error("Incorrect username or password.")
        elif auth_status is None:
            st.info("Please enter your credentials.")

        st.stop()

    # Authenticated — determine role from username
    username = st.session_state.get("username", "")
    admin_username = os.getenv("AUTH_ADMIN_USERNAME", "admin")
    role = "admin" if username == admin_username else "researcher"
    st.session_state["role"] = role
    return role
