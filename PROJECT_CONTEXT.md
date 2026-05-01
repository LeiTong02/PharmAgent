# Project Context: Pharma Research Assistant (PharmaRA)

## Interview Target
- **Company**: Pharmacy Company
- **Purpose**: Live interview demo showcasing AI/agent engineering capabilities for pharmaceutical R&D
- **Deadline**: Wednesday May 6, 2026
- **Demo audience**: Technical evaluators — PhD-level researchers and/or AI/ML engineers

---

## Core Project Goal
Build a polished, self-contained **Pharma Research Assistant Agent** that demonstrates four distinct AI engineering capabilities in a single Streamlit interface:
1. RAG over local mock research papers with inline citations
2. Structured tool-calling over a local assay results CSV via pandas
3. Safety guardrail that blocks personal medical-advice requests
4. External literature / repository lookup (GitHub README + Semantic Scholar / arXiv)

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
- **MCP integration**: would require a separate MCP server wrapping the assay tool; too unstable to add under deadline pressure
- **PDF document upload**: real users need PyMuPDF; mock `.txt` files are sufficient for demo
- **Persistent conversation memory**: demo is stateless per browser session
- **Authentication / multi-user**: single-user demo, no auth needed
- **Molecule viewer**: RDKit structure rendering for compound IDs in Streamlit
- **Batch index refresh**: auto-reindex when new docs are added to `data/docs/`

---

## Architecture Summary

```
Streamlit UI (app.py)
    │
    ▼
LangGraph StateGraph
    ├── guardrail_node   → regex check → BLOCK or PASS
    ├── agent_node       → gpt-4o-mini with 4 tools bound
    └── tool_node        → dispatches to one of 4 tools:
            ├── rag_search          → Redis Stack retrieval, returns context + citations
            ├── query_assay_data    → pandas on assay_results.csv, returns markdown table
            ├── fetch_github_readme → GitHub API /repos/{owner}/{repo}/readme
            └── lookup_paper        → Semantic Scholar API → arXiv fallback → stub
```

**Vector store**: Redis Stack (Docker container, index name `pharma_ra`, port `26379`)
**Embeddings**: `models/gemini-embedding-2` via `GoogleGenerativeAIEmbeddings` (auto-selected when `OPENAI_BASE_URL` contains "google"; falls back to `OpenAIEmbeddings` otherwise)
**LLM**: `gemini-3.1-flash-lite-preview` via Google's OpenAI-compatible API (configurable via `MODEL_NAME` env var)
**Chunking**: `RecursiveCharacterTextSplitter`, chunk_size=800, overlap=100

---

## Current Implementation Status (as of 2026-04-30)

| File | Status | Notes |
|---|---|---|
| `app.py` | Complete | Streamlit chat UI, cache_resource graph loader, sources expander |
| `agent/graph.py` | Complete | LangGraph 4-node graph, recursion_limit=10 via invoke config |
| `agent/tools.py` | Complete | 4 @tool functions, all verified |
| `agent/guardrails.py` | Complete | 8/8 unit tests pass |
| `agent/prompts.py` | Complete | System prompt with citation format + safety instructions |
| `rag/loader.py` | Complete | Parses TITLE/AUTHORS/YEAR/JOURNAL header metadata |
| `rag/vectorstore.py` | Complete | build_index() / load_index() with Redis Stack; `_FixedGoogleEmbeddings` wrapper for batch bug |
| `rag/retriever.py` | Complete | Returns (context_str, citations_list) |
| `scripts/build_index.py` | Complete | One-shot index builder |
| `data/docs/*.txt` | Complete | 5 mock papers, 33 chunks total |
| `data/assay_results.csv` | Complete | 30 rows, 3 targets (EGFR L858R/T790M, BRAF V600E, CDK4/Cyclin D1) |
| Redis index `pharma_ra` | **Complete** | Built and verified in Redis Stack container on `localhost:26379` |
| `.env` | **Complete** | API key, base URL, model, embedding model, and REDIS_URL all configured |

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
# 0. Ensure Redis Stack is running
docker start 373674e57d6c        # start existing container
# or: docker run -d -p 26379:6379 -p 8001:8001 redis/redis-stack:latest

# 1. Install dependencies (one-time, into conda env py310)
/Users/charles_tong/miniconda3/envs/py310/bin/pip install -r requirements.txt

# 2. Build Redis vector index (one-time, or after adding new docs / changing embedding model)
/Users/charles_tong/miniconda3/envs/py310/bin/python scripts/build_index.py

# 3. Run the app
/Users/charles_tong/miniconda3/envs/py310/bin/streamlit run app.py

# Optional: inspect stored vectors in RedisInsight UI
open http://localhost:8001
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
| LangGraph infinite tool loop | `recursion_limit=10` in `graph.invoke()` config |
| Interviewer asks personal medical advice | Guardrail fires visibly — this is a demo feature, not a bug |
| `.env` missing at demo time | `app.py` shows a clear `st.error` setup instruction and `st.stop()` |
| `tabulate` missing | Added to `requirements.txt`; required by `pandas.DataFrame.to_markdown()` |
