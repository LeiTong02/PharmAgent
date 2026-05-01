import json
import os
import redis as _redis
from langchain_core.messages import messages_to_dict, messages_from_dict


def _client():
    return _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:26379"))


def _keys(username: str):
    return f"chat_history:{username}:messages", f"chat_history:{username}:citations"


def load_history(username: str) -> tuple[list, dict]:
    """Return (messages, citations) from Redis; empty defaults if not found."""
    r = _client()
    msg_key, cite_key = _keys(username)
    raw_msgs = r.get(msg_key)
    raw_cites = r.get(cite_key)
    msgs = messages_from_dict(json.loads(raw_msgs)) if raw_msgs else []
    cites = {int(k): v for k, v in json.loads(raw_cites).items()} if raw_cites else {}
    return msgs, cites


def save_history(username: str, messages: list, citations: dict) -> None:
    r = _client()
    msg_key, cite_key = _keys(username)
    r.set(msg_key, json.dumps(messages_to_dict(messages)))
    r.set(cite_key, json.dumps({str(k): v for k, v in citations.items()}))


def clear_history(username: str) -> None:
    r = _client()
    msg_key, cite_key = _keys(username)
    r.delete(msg_key, cite_key)
