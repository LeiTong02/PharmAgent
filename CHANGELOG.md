# Changelog — PharmaRA

All notable changes to this project will be documented here.
Format: `[YYYY-MM-DD] — Description`

---

## [2026-04-30] — Initial project build

### Added
- **Project scaffolding**: directory structure (`agent/`, `rag/`, `data/docs/`, `scripts/`, `faiss_index/`), `requirements.txt`, `.env.example`, `.gitignore`
- **Mock research data**:
  - `data/docs/paper_01_kinase_inhibitor_discovery.txt` — EGFR L858R/T790M covalent inhibitor SR-0472 (Journal of Medicinal Chemistry, 2023)
  - `data/docs/paper_02_phase2_clinical_trial_COPD.txt` — SAN-4891 inhaled PDE4 inhibitor Phase 2 RCT (Lancet Respiratory Medicine, 2022)
  - `data/docs/paper_03_biomarker_validation_oncology.txt` — PharmaLiq-7 ctDNA liquid biopsy panel for CRC (Nature Medicine, 2023)
  - `data/docs/paper_04_protein_folding_ML.txt` — AlphaFold2 pipeline for LSD orphan enzyme druggable site identification (Cell Chemical Biology, 2024)
  - `data/docs/paper_05_hts_assay_methodology.txt` — HTS kinase assay standardization and CDK4/Cyclin D1 screen case study (SLAS Discovery, 2022)
  - `data/assay_results.csv` — 30-row mock assay database: 3 targets (EGFR L858R/T790M, BRAF V600E, CDK4/Cyclin D1), IC50/EC50/selectivity/cell viability columns
- **RAG pipeline**:
  - `rag/loader.py` — loads `.txt` papers, parses TITLE/AUTHORS/YEAR/JOURNAL header metadata, chunks with `RecursiveCharacterTextSplitter` (800/100)
  - `rag/vectorstore.py` — `build_index()` / `load_index()` using FAISS + `text-embedding-3-small`
  - `rag/retriever.py` — `retrieve(vectorstore, query, k=4)` returns `(context_str, citations_list)`
  - `scripts/build_index.py` — one-shot CLI to embed and save FAISS index
- **Agent layer**:
  - `agent/guardrails.py` — regex blocklist (`MEDICAL_ADVICE_PATTERNS`) + `is_medical_advice()` function; 8/8 unit tests pass
  - `agent/prompts.py` — `SYSTEM_PROMPT` with pharma researcher persona, `[Source: Title, Year]` citation format, safety instructions
  - `agent/tools.py` — 4 LangChain `@tool` functions: `rag_search`, `query_assay_data`, `fetch_github_readme`, `lookup_paper`
  - `agent/graph.py` — LangGraph `StateGraph` with nodes: `guardrail_node → agent_node ⇄ tool_node`; `recursion_limit=10` via `invoke()` config
- **Streamlit UI** (`app.py`):
  - Chat interface with `st.chat_input` / `st.chat_message`
  - `@st.cache_resource` graph loader (loads once per session)
  - Sidebar with project description, 7 example queries (clickable), safety warning, model info
  - Expandable "Sources" section extracted from `[Source: ...]` inline citations
  - Red `st.error` banner when guardrail fires
  - `st.error` + `st.stop()` if `.env` / API key is missing
  - `st.error` + `st.stop()` if FAISS index has not been built yet

### Fixed (during initial build)
- `langchain.schema.Document` → `langchain_core.documents.Document` (deprecated in LangChain 0.3+)
- `langchain.text_splitter` → `langchain_text_splitters` (moved to separate package)
- `compile(recursion_limit=10)` → `invoke(..., config={"recursion_limit": 10})` (LangGraph 0.2+ API change)
- Missing `tabulate` dependency added to `requirements.txt` (required by `pandas.DataFrame.to_markdown()`)
- `faiss-cpu` installed via pip wheel (`1.8.0.post1`) after conda-forge build failed due to missing `swig`
- Guardrail pattern expanded to catch "What dosage is recommended for COPD patients?" while still allowing "What dose was used in the Phase 2 trial?"

### Conda environment
- All dependencies installed into `/Users/charles_tong/miniconda3/envs/py310` (Python 3.10.18)
- Key versions: langchain 1.2.x, langgraph 0.2.x, faiss-cpu 1.8.0, streamlit 1.35+

---

## [2026-04-30] — Switch vector store from FAISS to Redis Stack

### Changed
- **`rag/vectorstore.py`**: Replaced `FAISS` with `langchain_redis.RedisVectorStore` + `RedisConfig`
  - `build_index()`: drops existing Redis index (`FT.DROPINDEX ... DD`) then calls `RedisVectorStore.from_documents()`
  - `load_index()`: connects to existing index via `RedisVectorStore.from_existing_index()`
  - `_get_embeddings()`: auto-detects Google API from `OPENAI_BASE_URL` and uses `GoogleGenerativeAIEmbeddings`; falls back to `OpenAIEmbeddings` otherwise
  - Redis index name: `pharma_ra`; default URL: `redis://localhost:26379`
- **`app.py`**: Updated error handler from `FileNotFoundError` → generic `Exception` (Redis errors are not file-based); updated error message to reference Redis URL
- **`scripts/build_index.py`**: Updated docstring and print output to reflect Redis storage
- **`requirements.txt`**: Removed `faiss-cpu`, added `langchain-redis>=0.1.0`, `langchain-google-genai>=2.0.0`, `redis>=5.0.0`
- **`.env`**: Added `REDIS_URL=redis://localhost:26379`

### Why
Google's OpenAI-compatible endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`) returns HTTP 501 on `/embeddings` calls. Native `GoogleGenerativeAIEmbeddings` required instead of `OpenAIEmbeddings`. Redis Stack chosen over FAISS for persistence and demo impressiveness.

### Verified
- Redis ping: OK (`localhost:26379`, `search` module loaded)
- `python scripts/build_index.py`: loaded 33 chunks from 5 files, stored in Redis
- `retrieve(vs, "EGFR T790M resistance")`: returned correct citation from paper_01

## [2026-04-30] — Fix tool calling for Gemini thinking models

### Changed
- **`agent/graph.py`**: `_get_llm()` now auto-detects Google API (same pattern as `_get_embeddings()`). When `OPENAI_BASE_URL` contains "google", uses `ChatGoogleGenerativeAI` from `langchain-google-genai` instead of `ChatOpenAI`. Gemini thinking models (e.g. `gemini-3.1-flash-lite-preview`) require native SDK for proper thought_signature handling in multi-turn tool calls; the OpenAI-compatible endpoint returns HTTP 400 `INVALID_ARGUMENT` on tool call responses.
- **`app.py`**: Added content extraction for `ChatGoogleGenerativeAI` structured content. Thinking models return `content` as `[{"type": "text", "text": "...", "extras": {"signature": "..."}}]` instead of a plain string. Extractor applied before `st.markdown()` and before citation regex.

### Fixed
- `openai.BadRequestError: 400 — Function call is missing a thought_signature` when using Gemini thinking models via OpenAI-compatible endpoint with tool use.
- Raw list content would have been rendered verbatim in the Streamlit UI instead of the LLM's formatted text.

### Verified (programmatic tests — all 5 scenarios green)
- Guardrail fires correctly, no LLM call
- RAG returns `[Source: ...]` citations from Redis index
- CSV tool returns markdown table sorted by IC50
- GitHub README fetch works for `deepmind/alphafold`
- Scholar lookup returns paper metadata from Semantic Scholar

## [2026-05-01] — Fix Gemini batch embedding bug + fix history rendering

### Fixed
- **`rag/vectorstore.py`**: `GoogleGenerativeAIEmbeddings.embed_documents()` returns only 1 vector regardless of batch size (confirmed API/wrapper bug). Added `_FixedGoogleEmbeddings` wrapper class that calls `embed_query()` per document instead. Redis index now stores all 33 chunks correctly (was: 1 key; now: 33 keys).
- **`app.py`**: Chat history loop was calling `st.markdown(msg.content)` directly, which rendered Gemini thinking model responses as raw Python list (`[{'type': 'text', 'text': '...', 'extras': {...}}]`) instead of plain text. Extracted `_extract_text(content)` helper at module level; applied to both history rendering and new-message rendering.

### How to apply after this fix
```bash
# Rebuild index (drops old 1-doc index, repopulates with all 33 chunks)
python scripts/build_index.py
# Restart Streamlit to pick up app.py changes
streamlit run app.py
```

## [2026-05-01] — Multi-user chat history + admin PDF upload + auth bug fixes

### Added
- **Per-user chat history isolation**: Session state keys namespaced by username (`messages_{username}`, `citations_{username}`). Admin and researcher each see only their own chat history within the same browser session.
- **Redis-backed chat persistence**: New `chat/` module with `chat/history.py`:
  - `load_history(username)` — loads chat + citations from Redis at session init
  - `save_history(username, messages, citations)` — saves after each AI reply
  - `clear_history(username)` — wipes history on demand (sidebar button)
  - Enables chat history to survive app restart and browser reload
- **Admin PDF upload page** (`pages/1_📤_Admin_Upload.py`):
  - Multi-page Streamlit app: main Chat tab + Admin Upload tab (role-gated)
  - PDF upload → `pymupdf4llm` OCR + markdown → 2-pass chunking (MarkdownHeaderSplitter → RecursiveCharacterSplitter)
  - Auto-duplicate detection by filename
  - Rebuild mock index button (clears all uploads, restores 5 mock papers)
  - Progress bar + per-file status (✅ indexed, ⏭️ skipped, ❌ error)
- **Sidebar Clear chat button** — wipes Redis history + session state for current user

### Changed
- **`auth/config.py`** — restored to streamlit-authenticator version with fixes:
  - `cookie_expiry_days=0` (was 1) — session-only cookies; deleted on browser close → prevents cross-session admin bypass
  - Fixed `login_gate()` logic: only calls `st.stop()` when `auth_status` is actually False/None after login; falls through if cookie auto-authenticates
- **`app.py`** — restored to Streamlit with per-user isolation + Redis history
- **`rag/loader.py`** — added `load_pdf_bytes(file_bytes, filename)` with 2-pass semantic chunking for digital PDFs
- **`rag/vectorstore.py`** — added:
  - `add_documents(docs)` — incremental insertion into existing Redis index (used by admin upload)
  - `list_uploaded_files()` — queries Redis for all `source_type="uploaded"` documents
  - Extended `_METADATA_SCHEMA` with `source_type` (text) and `upload_timestamp` (text)
- **`requirements.txt`** — added `pymupdf4llm>=0.0.17`, `streamlit-authenticator>=0.3.0`, `bcrypt>=4.0.0`

### Fixed
- **Bug 1 — No login form on fresh browser**: `login_gate()` was calling `st.stop()` unconditionally after `authenticator.login()`, even when the 1-day cookie had silently authenticated the user. Result: first page load showed nothing, or blank form, then had to click to proceed. Fix: only `st.stop()` if `auth_status` is False (wrong password) or None (waiting for input); if True (cookie worked), fall through to show chat.
- **Bug 2 — Admin upload bypass**: 1-day cookie persisted admin credentials across browser restart, allowing "bypass" where a logged-out admin from yesterday could access admin page without seeing login form. Fix: `cookie_expiry_days=0` → session-only, expires on browser close → fresh login required every browser session.
- **UI flicker on message send**: Removed `st.rerun()` at end of message handler — was causing full-page re-render (auth check, sidebar, all history). Messages already appended to session state, so they appear correctly without forced rerun.

### Verified
- Fresh browser (no cookie) → login form appears ✓
- Login as admin → chat works, Admin Upload tab visible, history persists ✓
- Close browser, reopen → login form again (cookie gone) ✓
- Login as researcher → Admin Upload shows "restricted" error if attempting direct access ✓
- Send 2 messages, reload page → history reloads from Redis ✓
- Click "Clear chat history" → history wiped ✓
- Admin can upload PDF → auto-parses with 2-pass chunking → queryable in chat ✓

### How to apply
```bash
# Already set up; just verify Redis is running:
docker ps  # should see redis container
# Then run:
streamlit run app.py
```

## [2026-05-04] — FastAPI web frontend (branch: `frontend/fastapi-ui`)

### Added
- **`frontend/` package** — FastAPI app alongside existing Streamlit `app.py` (Streamlit untouched)
  - `frontend/config.py` — Pydantic Settings, reads same `.env` as Streamlit app
  - `frontend/deps.py` — `get_current_user()` (session → SQLite lookup → 302 if missing), `require_admin()` (403 if role ≠ admin)
  - `frontend/main.py` — FastAPI lifespan: `init_db_sync()`, loads both vectorstores + graphs, `SessionMiddleware`, router mounts, root redirect `/ → /chat`
- **SQLite authentication** (`frontend/db/auth.py`) — replaces `streamlit-authenticator`
  - `users` table with `bcrypt` password hashing (uses `bcrypt` directly; `passlib` incompatible with `bcrypt==5.0.0`)
  - Seeded from `.env` on first startup; idempotent re-runs; DB at `frontend/auth.db` (gitignored)
- **Auth routes** (`frontend/routers/auth_router.py`) — `GET /login`, `POST /login`, `GET /logout`; itsdangerous signed session cookies, session-only expiry
- **Chat page** (`/chat`) with SSE streaming (`POST /api/chat`):
  - `ThreadPoolExecutor` bridge: sync `graph.stream()` runs in thread → `asyncio.Queue` via `loop.call_soon_threadsafe()` → async SSE consumer
  - SSE event types: `tool_status`, `text_chunk`, `blocked`, `tool_trace`, `citations`, `compounds`, `done`, `error`
  - History API (`GET/DELETE /api/history`) backed by existing `chat/history.py`
  - Token usage endpoint (`GET /api/token-usage`)
- **Admin page** (`/admin`) — drag-and-drop PDF upload, file listing; 403 for non-admin
- **Pharma-themed UI** (Jinja2 + Alpine.js + Tailwind CSS + marked.js, all CDN, no build step)
  - Deep-navy color scheme (`#0f1929` bg, `#38bdf8` accent)
  - Tool call trace panel, source citations panel, molecule images (PubChem PNG)
  - Retrieval mode toggle (Classic RAG / Wiki RAG), 7 example queries in sidebar
- **36 tests** (`tests/test_frontend/`) — all passing; `pytest.ini` sets `asyncio_mode = auto`

### Fixed (during development)
- `passlib` ↔ `bcrypt==5.0.0` incompatibility → switched to `import bcrypt as _bcrypt` directly
- Starlette 1.0.0 changed `TemplateResponse` signature — `request` is now first positional arg
- Jinja2 `block 'content' defined twice` error — moved `{% endif %}` before `<main>` so single block is always rendered
- `ASGITransport` does not fire lifespan events → initialize `app.state.*` directly in test fixtures
- Import-inside-function mock paths (`patch("rag.vectorstore.list_uploaded_files", ...)` not the router module)

### Verified
- 19/19 auth + login-flow tests passing (`test_auth.py`, `test_auth_flow.py`)
- 36/36 total frontend tests passing (`pytest tests/test_frontend/ -v`)
- SQLite 3.50.2, aiosqlite, bcrypt all confirmed installed
- Manual smoke test planned: 2026-05-05

### Run commands
```bash
# FastAPI (new)
uvicorn frontend.main:app --reload --port 8000
# Login: admin/admin123 or researcher/researcher123

# Streamlit (unchanged)
streamlit run app.py --server.port 8501
```

<!-- Future entries go above this line, newest first -->
