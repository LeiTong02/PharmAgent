# TODO — PharmaRA Demo

Last updated: 2026-05-04 (FastAPI complete; manual smoke test pending 2026-05-05)

---

## Must-Have Before Wednesday (May 6, 2026) — ALL COMPLETE ✅

### Session 2026-05-01 (Today)
Implemented from scratch:
- [x] Per-user chat history isolation (namespaced session state)
- [x] Redis-backed chat persistence (`chat/history.py`)
- [x] Admin PDF upload page with 2-pass semantic chunking
- [x] Clear chat history button
- [x] Fixed auth bugs: session-only cookies + proper login gate flow
- [x] Removed UI flicker (`st.rerun()` removal)
- [x] Full visual smoke test — all features working ✓

### Remaining Must-Haves

- [x] Copy `.env.example` → `.env` and set `OPENAI_API_KEY` *(done — Google Gemini API configured)*
- [x] Build Redis vector index (`python scripts/build_index.py`) — 33 chunks in `pharma_ra` index *(done — verified 2026-04-30)*
- [x] Programmatic end-to-end pipeline test — all 5 scenarios verified *(done 2026-04-30)*:
  - [x] RAG: returns answer with `[Source: ...]` citations
  - [x] CSV: returns markdown table sorted by IC50
  - [x] Guardrail: fires on "What dosage should I take for COPD?" — no LLM call
  - [x] GitHub tool: fetches `deepmind/alphafold` README correctly
  - [x] Scholar tool: returns paper metadata from Semantic Scholar
- [x] Open http://localhost:8501 in browser and do a visual smoke test: *(done 2026-05-01)*
  - [x] Verify citations appear in expandable Sources panel
  - [x] Verify guardrail shows red `st.error` banner (not just text)
  - [x] Click all 7 sidebar example queries — confirm each works without errors
  - [x] Check no crashes or unhandled exceptions (browser console clean)
- [ ] Prepare 3–4 demo talking points matching each tool to a Sanofi R&D use case — **see DEMO_NOTES.md** *(optional; script ready, can improvise from DEMO_NOTES.md during interview)*

---

## Nice-to-Have (if time permits before Wednesday)

- [x] Add a "How it works" diagram or flow description in the sidebar (text only, no image) *(done 2026-05-03)*
- [x] Show tool call trace in the UI (which tool was invoked, with what args) — `🔧 Tool calls` expander *(done 2026-05-03)*
- [x] Add 1–2 more mock papers to improve RAG coverage — PROTAC degraders + immuno-oncology *(done 2026-05-03)*
- [x] Add a "Clear chat" button to the sidebar *(done 2026-05-01)*
- [x] Improve CSV tool: add support for date-range filters and researcher-name queries *(done 2026-05-03)*
- [x] Add `st.toast` confirmation when guardrail fires (in addition to red banner) *(done 2026-05-03)*
- [x] Pin exact package versions in `requirements.txt` for reproducibility *(done 2026-05-03)*

---

## Future Production Extensions — ALL COMPLETE ✅

- [x] MCP integration: `mcp_server.py` via FastMCP — exposes assay, paper, and GitHub tools *(done 2026-05-03)*
- [x] PDF upload: `rag/loader.py` uses `pymupdf4llm` with 2-pass chunking; Admin Upload page accepts PDFs *(done 2026-05-01)*
- [x] Persistent chat memory: Redis-backed per-user history in `chat/history.py` *(done 2026-05-01)*
- [x] Authentication: `streamlit-authenticator` with YAML config in `auth/` *(done 2026-05-01)*
- [x] Molecule viewer: PubChem PNG via SMILES; 10 compounds with SMILES in `assay_results.csv`; auto-renders after assay queries *(done 2026-05-03)*
- [x] Batch embedding refresh: `scripts/watch_docs.py` polls `data/docs/` every 10s, auto-rebuilds both indexes on change *(done 2026-05-03)*
- [x] Evaluation harness: `scripts/evaluate_rag.py` — keyword-recall scoring on 7 Q&A pairs for Classic vs Wiki RAG *(done 2026-05-03)*
- [x] Real Sanofi document corpus: intentionally skipped — requires internal documents behind auth; mock papers serve as placeholders
- [x] Streaming responses: `graph.stream(stream_mode="updates")` shows real-time tool status ("⚙️ Calling `tool_name`...") while agent executes *(done 2026-05-03)*
- [x] Cost tracking: `chat/token_logger.py` logs per-query token usage to `logs/token_usage.jsonl`; sidebar "💰 Token usage" expander *(done 2026-05-03)*

---

## FastAPI Web Frontend — COMPLETE ✅ (branch: `frontend/fastapi-ui`)

Implemented 2026-05-04. Professional pharma-themed web UI replacing Streamlit's generic look.
Original `app.py` and all backend modules left untouched.

### What was built
- [x] `frontend/` FastAPI package — config, deps, lifespan (loads vectorstores + graphs once)
- [x] SQLite auth (`frontend/db/auth.py`) — bcrypt hashing, seeded from `.env`, replaces streamlit-authenticator
- [x] Session cookies via `SessionMiddleware` (itsdangerous signed, session-only)
- [x] Login/logout routes (`/login`, `/logout`) + pharma-themed login page
- [x] Chat page (`/chat`) — deep-navy UI, Alpine.js reactive, marked.js markdown rendering
- [x] SSE streaming (`POST /api/chat`) — `ThreadPoolExecutor` bridge from sync `graph.stream()` to async SSE queue
- [x] History API (`GET/DELETE /api/history`) — backed by existing `chat/history.py`
- [x] Tool call trace panel, source citations panel, molecule structure images (PubChem)
- [x] Retrieval mode toggle (Classic RAG / Wiki RAG), 7 example queries in sidebar
- [x] Token usage display, "How it works" section
- [x] Admin page (`/admin`) — drag-and-drop PDF upload, file listing, admin-only gating (403 for researcher)
- [x] 36 tests, all passing (`pytest tests/test_frontend/ -v`)

### How to run
```bash
uvicorn frontend.main:app --reload --port 8000
# Login: admin/admin123 or researcher/researcher123
# Original Streamlit still works: streamlit run app.py --server.port 8501
```

### Bugs fixed during development
- `passlib` ↔ `bcrypt==5.0.0` incompatibility → switched to `bcrypt` directly
- Starlette 1.0.0 changed `TemplateResponse` signature (`request` is now first positional arg)
- Jinja2 duplicate `{% block content %}` inside `{% if %}` → moved `{% endif %}` before `<main>`
- `ASGITransport` does not fire lifespan → initialize `app.state` directly in test fixture

---

## Pending (2026-05-05)

- [ ] Manual smoke test of FastAPI frontend (http://localhost:8000):
  - [ ] Login as admin → chat page loads correctly
  - [ ] Send a RAG query → SSE streaming works, citations appear
  - [ ] Send guardrail-triggering query → blocked banner shown
  - [ ] Toggle Classic RAG ↔ Wiki RAG → mode switch works
  - [ ] Click sidebar example queries → prefill + auto-submit
  - [ ] Admin page: drag-and-drop PDF upload → success
  - [ ] Logout → redirected to login page
  - [ ] Login as researcher → /admin returns 403
