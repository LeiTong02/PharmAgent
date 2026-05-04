"""Tests for chat history and streaming API endpoints."""
from __future__ import annotations

from unittest.mock import patch


async def test_get_history_structure(admin_client):
    with patch("frontend.routers.chat_router.load_history", return_value=([], {})):
        resp = await admin_client.get("/api/history")
    assert resp.status_code == 200
    data = resp.json()
    assert "messages" in data
    assert "citations" in data
    assert isinstance(data["messages"], list)
    assert isinstance(data["citations"], dict)


async def test_get_history_unauthenticated(client):
    resp = await client.get("/api/history", follow_redirects=False)
    assert resp.status_code in (302, 307, 401, 422)


async def test_delete_history(admin_client):
    with patch("frontend.routers.chat_router.clear_history") as mock_clear, \
         patch("frontend.routers.chat_router.load_history", return_value=([], {})):
        resp = await admin_client.delete("/api/history")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    mock_clear.assert_called_once()


async def test_delete_history_unauthenticated(client):
    resp = await client.delete("/api/history", follow_redirects=False)
    assert resp.status_code in (302, 307, 401, 422)


async def test_token_usage_endpoint(admin_client):
    mock_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "query_count": 1,
    }
    with patch("frontend.routers.chat_router.get_session_usage", return_value=mock_usage):
        resp = await admin_client.get("/api/token-usage")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tokens"] == 150
    assert data["query_count"] == 1


async def test_chat_endpoint_rejects_empty_query(admin_client):
    with patch("frontend.routers.chat_router.load_history", return_value=([], {})):
        resp = await admin_client.post(
            "/api/chat",
            json={"query": "  ", "mode": "classic"},
        )
    assert resp.status_code == 400


async def test_chat_endpoint_returns_503_when_not_ready(admin_client):
    with patch("frontend.routers.chat_router.load_history", return_value=([], {})):
        resp = await admin_client.post(
            "/api/chat",
            json={"query": "What is EGFR?", "mode": "classic"},
        )
    assert resp.status_code == 503


async def test_chat_endpoint_unauthenticated(client):
    resp = await client.post(
        "/api/chat",
        json={"query": "test", "mode": "classic"},
        follow_redirects=False,
    )
    assert resp.status_code in (302, 307, 401, 422)


async def test_history_serializes_messages(admin_client):
    from langchain_core.messages import AIMessage, HumanMessage

    mock_msgs = [
        HumanMessage(content="What is EGFR?"),
        AIMessage(content="EGFR is a receptor tyrosine kinase."),
    ]
    mock_citations = {"0": {"raw": "[Source: Paper A, 2023]"}}

    with patch("frontend.routers.chat_router.load_history", return_value=(mock_msgs, mock_citations)):
        resp = await admin_client.get("/api/history")

    data = resp.json()
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][1]["role"] == "assistant"
    assert "EGFR" in data["messages"][0]["content"]
