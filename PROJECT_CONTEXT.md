# Project Context: Pharma Research Assistant (PharmaRA)

## Interview Target
- **Company**: Pharmacy Company
- **Purpose**: Live interview demo showcasing AI/agent engineering capabilities for pharmaceutical R&D
- **Deadline**: Wednesday May 6, 2026
- **Demo audience**: Technical evaluators — PhD-level researchers and/or AI/ML engineers

---

## Core Project Goal
Build a polished, self-contained **Pharma Research Assistant Agent** that demonstrates four distinct AI engineering capabilities:
1. RAG over local mock research papers with inline citations
2. Structured tool-calling over a local assay results CSV via pandas
3. Safety guardrail that blocks personal medical-advice requests
4. External literature / repository lookup (GitHub README + Semantic Scholar / arXiv)

Two frontends exist side-by-side: original **Streamlit** (`main` branch, port 8501) and a professional **FastAPI** web UI (`frontend/fastapi-ui` branch, port 8000). All backend modules are shared.

---

## Must-Have Features
- [x] RAG over 5 mock pharma research papers (Redis Stack vector store + Google embeddings), with `[Source: Title, Year]` inline citations
- [x] Assay data tool: natural-language queries → pandas filtering of `data/assay_results.csv` → markdown table
- [x] Safety guardrail: regex-based blocker fires before LLM is invoked; returns safe refusal message
- [x] LangGraph agent loop: `START → guardrail_node → agent_node ⇄ tool_node → END`
- [x] Streamlit chat UI with sidebar (example queries, model info, safety note)
- [x] Expandable "Sources" section after each RAG answer
- [x] `.env`-based configuration (API key, model, embedding model)
- [x] `scripts/build_index.py` one-shot Redis index builder

## Optional Features (implemented)
- [x] GitHub README fetch (`fetch_github_readme` tool) via GitHub API with graceful fallback
- [x] Academic paper lookup (`lookup_paper` tool) via Semantic Scholar, arXiv fallback, stub fallback

## Features Intentionally Deferred
- **Real document corpus**: requires internal documents behind auth; mock papers serve as placeholders

---

## Architecture Summary

```
Streamlit UI (app.py, port 8501)          FastAPI UI (frontend/, port 8000)
        │                                          │
        └──────────────────┬────────────────────────┘
                           ▼
                  LangGraph StateGraph
                    ├── guardrail_node   → regex check → BLOCK or PASS
                    ├── intent_node      → LLM classifies query intent + entities → QueryContext (→ tools.py state)
                    ├── agent_node       → Gemini via ChatGoogleGenerativeAI, 4 tools bound; strips __VISUAL_CHUNKS__ sentinel
                    └── tool_node        → dispatches to one of 4 tools:
                            ├── rag_search          → smart_retrieve() (policy-gated); context + __VISUAL_CHUNKS__:<json>
                            ├── query_assay_data    → pandas on assay_results.csv, returns markdown table
                            ├── fetch_github_readme → GitHub API /repos/{owner}/{repo}/readme
                            └── lookup_paper        → Semantic Scholar API → arXiv fallback → stub
```

**FastAPI-specific architecture:**
- Auth: SQLite (`frontend/auth.db`) + bcrypt; seeded from `.env`; `SessionMiddleware` signed cookies
- Streaming: `ThreadPoolExecutor` bridge — sync `graph.stream()` in thread → `asyncio.Queue` → SSE
- Templates: Jinja2 server-rendered + Alpine.js (reactivity) + Tailwind CSS + marked.js (all CDN)

**Vector store**: Redis Stack (Docker container, index name `pharma_ra`, port `26379`)
**Embeddings**: `models/gemini-embedding-2` via `GoogleGenerativeAIEmbeddings` (auto-selected when `OPENAI_BASE_URL` contains "google"; falls back to `OpenAIEmbeddings` otherwise)
**LLM**: `gemini-3.1-flash-lite-preview` via Google's OpenAI-compatible API (configurable via `MODEL_NAME` env var)
**Chunking**: `RecursiveCharacterTextSplitter`, chunk_size=800, overlap=100

---

## Current Implementation Status (as of 2026-05-05)

### Shared backend (both frontends)

| File | Status | Notes |
|---|---|---|
| `agent/graph.py` | Complete | 5-node graph (guardrail → intent → agent ↔ tool); LLM-based intent classifier in `intent_node`; `agent_node` strips `__VISUAL_CHUNKS__` sentinel; `recursion_limit=25` |
| `agent/tools.py` | Complete | `rag_search` uses `smart_retrieve()`; appends `__VISUAL_CHUNKS__:<json>` when visuals approved; 3 other tools unchanged |
| `agent/guardrails.py` | Complete | 8/8 unit tests pass |
| `agent/prompts.py` | Complete | System prompt with citation format + safety instructions |
| `rag/loader.py` | Complete | Parses headers; `load_pdf_bytes()` with 2-pass chunking for PDFs |
| `rag/vectorstore.py` | Complete | Redis Stack; dual index (`pharma_ra` + `pharma_wiki`); `_FixedGoogleEmbeddings` wrapper |
| `rag/query_parser.py` | Complete | `QueryIntent` enum + `QueryContext` dataclass; `parse_query()` regex fallback; `parse_query_from_llm_output()` for LLM path |
| `rag/retriever.py` | Complete | `smart_retrieve()` policy-gated (intent/entity/threshold/visual gate); `retrieve()` kept for wiki_search; 72/72 tests passing |
| `chat/history.py` | Complete | Redis-backed per-user history (load/save/clear) |
| `chat/token_logger.py` | Complete | Logs per-query token usage to `logs/token_usage.jsonl` |
| `mcp_server.py` | Complete | FastMCP server exposing 3 tools |
| `scripts/build_index.py` | Complete | One-shot builder for both indexes |
| `scripts/watch_docs.py` | Complete | Polls `data/docs/` every 10s, auto-rebuilds |
| `scripts/evaluate_rag.py` | Complete | Keyword-recall scoring on 7 Q&A pairs |
| `data/docs/*.txt` | Complete | 7 mock papers (EGFR, COPD, CRC, AlphaFold, HTS, PROTAC, immuno-oncology) |
| `data/assay_results.csv` | Complete | 30 rows, 10 SMILES, 3 targets |
| Redis index `pharma_ra` | Complete | Classic chunks; `localhost:26379` |
| Redis index `pharma_wiki` | Complete | LLM-compiled wiki pages |

### Streamlit frontend (`main` branch)

| File | Status | Notes |
|---|---|---|
| `app.py` | Complete | Chat UI, streaming, tool trace, molecule viewer, token usage |
| `pages/1_📤_Admin_Upload.py` | Complete | PDF upload with 2-pass chunking, role-gated |
| `auth/` | Complete | `streamlit-authenticator` YAML config, session-only cookies |

### FastAPI frontend (`frontend/fastapi-ui` branch)

| File | Status | Notes |
|---|---|---|
| `frontend/main.py` | Complete | FastAPI + lifespan + `SessionMiddleware` |
| `frontend/config.py` | Complete | Pydantic Settings, reads `.env` |
| `frontend/deps.py` | Complete | `get_current_user()`, `require_admin()` |
| `frontend/db/auth.py` | Complete | SQLite + bcrypt; seeded from `.env` |
| `frontend/routers/auth_router.py` | Complete | Login/logout |
| `frontend/routers/chat_router.py` | Complete | SSE streaming; parses `__VISUAL_CHUNKS__` sentinel; NoneType guard for LangGraph 1.1.9 |
| `frontend/routers/admin_router.py` | Complete | PDF upload, file listing, admin-gated |
| `frontend/templates/*.html` | Complete | base, login, chat, admin (Tailwind + Alpine.js) |
| `tests/test_frontend/` (36 tests) | Complete | All passing |
| `tests/test_rag/` (36 tests) | Complete | `test_query_parser.py` + `test_smart_retrieve.py`; all passing |
| Visual smoke test | Complete | Text RAG verified end-to-end |

---

## Important Design Decisions

### Why LangGraph over LangChain LCEL
LangGraph exposes the agent loop as an explicit graph (guardrail → agent ↔ tools), which is visually explainable and impressive in a live demo. LCEL would be simpler but less interesting architecturally.

### Why Redis Stack over FAISS or Chroma
Redis Stack runs as a persistent Docker container (`redis/redis-stack:latest`, mapped to ports `26379:6379` and `8001:8001`). This is more realistic than a file-based store for a production demo and allows showcasing enterprise-grade infrastructure. The index survives across app restarts without re-embedding, unlike in-memory stores.

### Why `GoogleGenerativeAIEmbeddings` over `OpenAIEmbeddings`
Google's OpenAI-compatible endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`) does not support the `/embeddings` path — it returns HTTP 501. The native `GoogleGenerativeAIEmbeddings` client uses Google's own embedding API correctly. Auto-detection is based on whether `OPENAI_BASE_URL` contains "google".

### Guardrail is regex-only (no LLM call)
A fast regex check before the LLM means: (a) zero extra latency on safe queries, (b) the guardrail fires even if the OpenAI API is down, (c) it's easy to explain to a technical audience.

### recursion_limit passed at invoke() time
LangGraph's `compile()` does not accept `recursion_limit` in v0.2+. It must be passed as `config={"recursion_limit": 10}` in `graph.invoke()` — this is done in `app.py`.

### Mock data topics chosen for demo coverage
- Paper 01: EGFR kinase inhibitors → supports both RAG and CSV queries about SR-0472
- Paper 02: COPD Phase 2 clinical trial → good for triggering guardrail (COPD dosage questions)
- Paper 03: CRC liquid biopsy biomarkers → shows oncology breadth
- Paper 04: AlphaFold / ML protein folding → shows ML/AI angle relevant to Sanofi's AI strategy
- Paper 05: HTS assay methodology → directly connects research literature to the assay CSV

---

## Run Commands

```bash
# 0. Start Redis Stack
docker start 373674e57d6c        # existing container
# or: docker run -d -p 26379:6379 -p 8001:8001 redis/redis-stack:latest

# 1. Install dependencies
pip install -r requirements.txt

# 2. Build Redis vector indexes (both classic + wiki)
python scripts/build_index.py

# 3a. Streamlit frontend (main branch)
streamlit run app.py --server.port 8501

# 3b. FastAPI frontend (frontend/fastapi-ui branch)
uvicorn frontend.main:app --reload --port 8000
# Login: admin/admin123 or researcher/researcher123

# Optional: RedisInsight UI
open http://localhost:8001

# Tests (FastAPI frontend only)
pytest tests/test_frontend/ -v
```

---

## Known Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Google API cost | `gemini-3.1-flash-lite-preview` is very cheap; embeddings billed per token; ~10 demo queries ≈ negligible |
| Redis container not running at demo time | `docker start <container_id>` or `docker run redis/redis-stack:latest`; index persists in container |
| Redis index lost (container deleted) | Rebuild with `python scripts/build_index.py` in ~30s |
| GitHub API rate limit (60 req/hr unauth) | Graceful fallback message; demo needs 1–2 calls max |
| Semantic Scholar API down | arXiv fallback → stub message; never crashes |
| LangGraph infinite tool loop | `recursion_limit=25` in `graph.stream()` config |
| Interviewer asks personal medical advice | Guardrail fires visibly — this is a demo feature, not a bug |
| Paper figure rendering in browser | `approved_visuals` correctly selected by policy; live browser display may have issues — text RAG unaffected |
| `.env` missing at demo time | `app.py` shows a clear `st.error` setup instruction and `st.stop()` |
| `tabulate` missing | Added to `requirements.txt`; required by `pandas.DataFrame.to_markdown()` |
