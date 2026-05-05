"""Chat page and streaming API routes."""
from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from langchain_core.messages import AIMessage, HumanMessage
from sse_starlette.sse import EventSourceResponse

from chat.history import clear_history, load_history, save_history
from chat.token_logger import get_session_usage, log_token_usage
from frontend.deps import get_current_user

router = APIRouter(tags=["chat"])
_executor = ThreadPoolExecutor(max_workers=4)


def _extract_text(content) -> str:
    """Extract plain text from Gemini thinking-model content (list of blocks or plain str)."""
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content) if content else ""


# ---------------------------------------------------------------------------
# SMILES map (loaded once from CSV at import time)
# ---------------------------------------------------------------------------

def _load_smiles_map() -> dict[str, str]:
    try:
        import pandas as pd
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "assay_results.csv",
        )
        df = pd.read_csv(csv_path)
        result: dict[str, str] = {}
        if "smiles" in df.columns:
            for _, row in df.iterrows():
                smi = str(row.get("smiles", "")).strip()
                cid = str(row.get("compound_id", "")).upper().strip()
                if smi and smi not in ("nan", ""):
                    result[cid] = smi
        return result
    except Exception:
        return {}


_SMILES_MAP = _load_smiles_map()


def _extract_compound_images(text: str) -> list[dict]:
    found = re.findall(r"\bSR-\d+\b", text, re.IGNORECASE)
    seen: list[dict] = []
    ids_seen: set[str] = set()
    for cid in found:
        cid_upper = cid.upper()
        if cid_upper in _SMILES_MAP and cid_upper not in ids_seen:
            encoded = urllib.parse.quote(_SMILES_MAP[cid_upper], safe="")
            # Use query-param format so SMILES with '/' (stereochemistry) don't break the path
            url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/PNG"
                f"?smiles={encoded}&image_size=200x200"
            )
            seen.append({"id": cid_upper, "url": url})
            ids_seen.add(cid_upper)
    return seen




# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def _messages_to_json(messages: list) -> list[dict]:
    result = []
    for m in messages:
        if isinstance(m, HumanMessage):
            result.append({"role": "user", "content": _extract_text(m.content)})
        elif isinstance(m, AIMessage):
            result.append({"role": "assistant", "content": _extract_text(m.content)})
    return result


# ---------------------------------------------------------------------------
# Chat page
# ---------------------------------------------------------------------------

@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, user: dict = Depends(get_current_user)):
    messages, citations = load_history(user["username"])
    usage = get_session_usage(user["username"], limit=50)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "chat.html",
        context={
            "user": user,
            "history_json": json.dumps(_messages_to_json(messages)),
            "citations_json": json.dumps(citations),
            "usage": usage,
            "app_ready": request.app.state.ready,
        },
    )


# ---------------------------------------------------------------------------
# History / token-usage API
# ---------------------------------------------------------------------------

@router.get("/api/history")
async def get_history(user: dict = Depends(get_current_user)):
    messages, citations = load_history(user["username"])
    return {"messages": _messages_to_json(messages), "citations": citations}


@router.delete("/api/history")
async def delete_history(user: dict = Depends(get_current_user)):
    clear_history(user["username"])
    return {"ok": True}


@router.get("/api/token-usage")
async def get_token_usage(user: dict = Depends(get_current_user)):
    return get_session_usage(user["username"], limit=50)


# ---------------------------------------------------------------------------
# SSE streaming chat endpoint
# ---------------------------------------------------------------------------

def _sse(t: str, content) -> dict:
    return {"data": json.dumps({"type": t, "content": content})}


@router.post("/api/chat")
async def chat_stream(request: Request, user: dict = Depends(get_current_user)):
    body = await request.json()
    query: str = body.get("query", "").strip()
    mode: str = body.get("mode", "classic")

    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Agent not ready. Check Redis and API key.")

    graph = (
        request.app.state.graph_wiki if mode == "wiki"
        else request.app.state.graph_classic
    )
    messages, citations = load_history(user["username"])
    messages.append(HumanMessage(content=query))
    state_in = {"messages": messages, "blocked": False}
    username = user["username"]
    model_name = request.app.state.templates  # just a ref check — use settings below

    async def event_stream() -> AsyncGenerator[dict, None]:
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        # Mutable state shared between thread worker and async consumer
        shared: dict = {
            "text_parts": [],
            "tool_traces": [],
            "token_usage": {},
            "new_citations": {},
            "blocked": False,
            "approved_visuals": [],  # policy-approved visual chunks from smart_retrieve
        }

        def _worker():
            try:
                for event in graph.stream(
                    state_in,
                    stream_mode="updates",
                    config={"recursion_limit": 25},
                ):
                    for node, update in event.items():
                        if update is None:
                            continue
                        msgs = update.get("messages", [])

                        if node == "guardrail_node":
                            if update.get("blocked"):
                                shared["blocked"] = True
                                loop.call_soon_threadsafe(
                                    queue.put_nowait,
                                    {"type": "blocked", "content": "⚠️ This query involves medical advice and cannot be answered here."},
                                )
                            for m in msgs:
                                if isinstance(m, AIMessage) and m.content:
                                    loop.call_soon_threadsafe(
                                        queue.put_nowait,
                                        {"type": "text_chunk", "content": _extract_text(m.content)},
                                    )

                        elif node == "agent_node":
                            for m in msgs:
                                if not isinstance(m, AIMessage):
                                    continue
                                if m.tool_calls:
                                    for tc in m.tool_calls:
                                        shared["tool_traces"].append(
                                            {"name": tc["name"], "args": tc.get("args", {})}
                                        )
                                        loop.call_soon_threadsafe(
                                            queue.put_nowait,
                                            {"type": "tool_status", "content": f"⚙️ Calling `{tc['name']}`..."},
                                        )
                                elif m.content:
                                    meta = m.response_metadata or {}
                                    usage = (
                                        meta.get("token_usage")
                                        or meta.get("usage_metadata")
                                        or {}
                                    )
                                    if usage:
                                        shared["token_usage"].update(usage)
                                    chunk = _extract_text(m.content)
                                    shared["text_parts"].append(chunk)
                                    loop.call_soon_threadsafe(
                                        queue.put_nowait,
                                        {"type": "text_chunk", "content": chunk},
                                    )

                        elif node == "tool_node":
                            loop.call_soon_threadsafe(
                                queue.put_nowait,
                                {"type": "tool_status", "content": "⚙️ Processing results..."},
                            )
                            for m in msgs:
                                content = getattr(m, "content", "") or ""
                                for line in content.split("\n"):
                                    if line.startswith("[Source:"):
                                        idx = len(shared["new_citations"])
                                        shared["new_citations"][str(idx)] = {"raw": line}
                                    elif line.startswith("__VISUAL_CHUNKS__:"):
                                        try:
                                            payload = json.loads(line[len("__VISUAL_CHUNKS__:"):])
                                            seen_urls = {v["url"] for v in shared["approved_visuals"]}
                                            for v in payload:
                                                if isinstance(v, dict) and v.get("url") and v["url"] not in seen_urls:
                                                    shared["approved_visuals"].append(v)
                                                    seen_urls.add(v["url"])
                                        except (json.JSONDecodeError, KeyError, TypeError):
                                            pass

            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "error", "content": f"An error occurred: {exc}"},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        loop.run_in_executor(_executor, _worker)

        # Relay events to the SSE client
        while True:
            item = await queue.get()
            if item is None:
                break
            yield _sse(item["type"], item["content"])

        # --- Post-stream finalization ---
        final_text = "".join(shared["text_parts"])

        compounds = _extract_compound_images(final_text)
        if compounds:
            yield _sse("compounds", json.dumps(compounds))

        if shared["tool_traces"]:
            yield _sse("tool_trace", json.dumps(shared["tool_traces"]))

        if shared["new_citations"]:
            yield _sse("citations", json.dumps(shared["new_citations"]))

        if shared["approved_visuals"]:
            yield _sse("paper_figures", json.dumps(shared["approved_visuals"]))

        # Persist history
        if not shared["blocked"] and final_text:
            messages.append(AIMessage(content=final_text))
            merged = {
                **citations,
                **{
                    str(len(citations) + int(k)): v
                    for k, v in shared["new_citations"].items()
                },
            }
            save_history(username, messages, merged)

            if shared["token_usage"]:
                from frontend.config import settings
                log_token_usage(
                    username, query, shared["token_usage"], settings.model_name, mode
                )

        yield _sse("done", "")

    return EventSourceResponse(event_stream())
