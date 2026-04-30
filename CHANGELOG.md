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

<!-- Future entries go above this line, newest first -->
