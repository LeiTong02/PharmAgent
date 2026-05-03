"""Log per-query token usage to logs/token_usage.jsonl for cost tracking."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

_LOGS_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOGS_DIR / "token_usage.jsonl"


def _normalize(usage: dict) -> dict:
    """Normalize OpenAI and Google usage metadata to common keys."""
    return {
        "prompt_tokens": (usage.get("prompt_tokens") or usage.get("prompt_token_count") or 0),
        "completion_tokens": (
            usage.get("completion_tokens") or usage.get("candidates_token_count") or 0
        ),
        "total_tokens": (usage.get("total_tokens") or usage.get("total_token_count") or 0),
    }


def log_token_usage(username: str, query: str, usage: dict, model: str, mode: str) -> None:
    """Append one JSON line to logs/token_usage.jsonl."""
    _LOGS_DIR.mkdir(exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "username": username,
        "query_preview": query[:80],
        "model": model,
        "mode": mode,
        **_normalize(usage),
    }
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def get_session_usage(username: str, limit: int = 100) -> dict:
    """Return aggregate token usage for the most recent *limit* queries by this user."""
    if not _LOG_FILE.exists():
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "query_count": 0}

    entries = []
    with open(_LOG_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("username") == username:
                    entries.append(e)
            except json.JSONDecodeError:
                pass

    recent = entries[-limit:]
    return {
        "prompt_tokens": sum(e.get("prompt_tokens", 0) for e in recent),
        "completion_tokens": sum(e.get("completion_tokens", 0) for e in recent),
        "total_tokens": sum(e.get("total_tokens", 0) for e in recent),
        "query_count": len(recent),
    }
