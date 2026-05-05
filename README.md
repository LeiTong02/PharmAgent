# PharmaRA — Pharmaceutical Research Assistant

An AI-powered research assistant for pharmaceutical R&D teams. Combines **policy-gated RAG over research papers**, **structured assay data queries**, **reagent procurement**, **external literature lookup**, and **admin-controlled document management** in a single conversational interface.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)
![Redis](https://img.shields.io/badge/Redis-Stack-red)
![Tests](https://img.shields.io/badge/tests-92%20passing-brightgreen)

---

## Features

### 🤖 Intelligent Agent (LangGraph)

5-node graph: `START → guardrail_node → intent_node → agent_node ⇄ tool_node → END`

- **Guardrail**: Regex-based blocker for medical advice queries — fires before the LLM is invoked
- **Intent Classifier**: LLM classifies query intent (`entity_lookup`, `concept_definition`, `framework_or_architecture`, `figure_specific`, `table_or_result`, `general_qa`) before retrieval
- **6 Tools**: RAG search, assay data, reagent procurement, GitHub README, paper lookup, wiki search

### 📚 Policy-Gated RAG (`smart_retrieve`)

Not just vector search — a full retrieval policy with four explicit gates:

1. **Intent gate** — classified by LLM before retrieval; type-boost matrix re-ranks chunks by `(intent × chunk_type)`
2. **Entity grounding gate** — query must be grounded in the index before any context is returned
3. **Evidence threshold** — cosine similarity ≥ 0.75 required for context chunks (lenient fallback at 0.60 for conversational queries)
4. **Visual gate** — figures returned only for `framework_or_architecture`, `figure_specific`, `table_or_result` queries, and only when entity-matched + supported by caption/nearby text

### 🔬 Research Tools

| Tool | Description |
|------|-------------|
| `rag_search` | Policy-gated semantic search over indexed research papers; returns context + inline citations |
| `query_assay_data` | Natural-language queries on IC50/EC50/selectivity CSV; supports date and researcher filters |
| `search_reagents` | Search reagent procurement catalog by compound name, internal ID (SR-XXXX), or target; check stock, estimate cost |
| `fetch_github_readme` | Fetch documentation from any public GitHub repository |
| `lookup_paper` | Semantic Scholar → arXiv fallback for academic paper metadata |
| `wiki_search` | Pre-compiled wiki pages for structured knowledge retrieval (alternate mode) |

### 🛒 Reagent Procurement (MCP Server)

The reagent catalog is exposed both as a LangChain tool (for the agent) and as a standalone **FastMCP server** that can be connected to Claude Code, Cursor, or any MCP-compatible client:

```json
{
  "mcpServers": {
    "reagent_catalog": {
      "command": "python",
      "args": ["/path/to/PharmAgent/mcp_servers/reagent_server.py"]
    }
  }
}
```

Three tools: `search_reagents`, `check_stock`, `estimate_cost`.

### 🧬 Document Management

- **Admin PDF Upload** — OCR → markdown → 2-pass semantic chunking (MarkdownHeaderSplitter → RecursiveCharacterSplitter)
- **Layout-Aware Indexing** — Extracts figure crops, page screenshots, figure captions, and table captions as separate indexed chunks with metadata (`figure_url`, `caption`, `nearby_text`, `figure_index`)
- **Redis Vector Index** — Persistent embedding storage; survives app restarts
- **Auto-Rebuild** — `scripts/watch_docs.py` polls `data/docs/` every 10s and rebuilds on change

### 👥 Multi-User & Persistence

- **SQLite Auth** — bcrypt-hashed passwords, seeded from `.env`, session-only signed cookies
- **Per-User Chat Isolation** — Redis-backed history; admin and researcher see only their own sessions
- **Token Usage Tracking** — Per-query usage logged to `logs/token_usage.jsonl`
- **Streaming Responses** — Server-Sent Events (SSE) via `ThreadPoolExecutor` bridge from sync `graph.stream()`

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Web UI (primary)** | FastAPI + Jinja2 + Alpine.js + Tailwind CSS (CDN) |
| **Web UI (legacy)** | Streamlit 1.35+ (`app.py`, port 8501) |
| **Agent** | LangGraph 1.1+ — 5-node graph with intent classification |
| **LLM** | Google Gemini (native SDK) or OpenAI GPT-4o |
| **Embeddings** | `models/gemini-embedding-2` or OpenAI `text-embedding-3-small` |
| **Vector Store** | Redis Stack — `pharma_ra` (RAG) + `pharma_wiki` (wiki) |
| **Chat History** | Redis — per-user JSON-serialized LangChain messages |
| **Auth (FastAPI)** | SQLite + bcrypt + `SessionMiddleware` signed cookies |
| **Auth (Streamlit)** | `streamlit-authenticator` YAML config |
| **PDF Parsing** | `pymupdf4llm` + layout extraction (`fitz`) |
| **MCP Server** | FastMCP — reagent catalog exposed via Model Context Protocol |

---

## Quick Start

### Prerequisites

- Docker (for Redis Stack)
- Python 3.10+ (conda or venv)
- Google Gemini API key **or** OpenAI API key

### 1. Clone & Install

```bash
git clone https://github.com/LeiTong02/PharmAgent.git
cd PharmAgent

conda create -n py310 python=3.10 && conda activate py310
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set OPENAI_API_KEY and optionally OPENAI_BASE_URL
```

### 2. Start Redis Stack

```bash
docker run -d --name redis-pharma \
  -p 26379:6379 -p 8001:8001 \
  redis/redis-stack:latest

# Verify
redis-cli -p 26379 ping   # → PONG
```

### 3. Build Vector Index

```bash
python scripts/build_index.py
# Embeds mock papers + builds pharma_ra and pharma_wiki indexes in Redis
```

### 4. Run

```bash
# FastAPI (recommended)
uvicorn frontend.main:app --reload --port 8000
# → http://localhost:8000
# Login: admin / admin123  or  researcher / researcher123

# Streamlit (legacy)
streamlit run app.py --server.port 8501
```

---

## Project Structure

```
PharmAgent/
├── agent/
│   ├── graph.py              # LangGraph 5-node graph (guardrail→intent→agent⇄tool)
│   ├── tools.py              # 6 LangChain @tool functions
│   ├── guardrails.py         # Regex-based medical advice blocker
│   └── prompts.py            # System prompt + citation format
│
├── rag/
│   ├── query_parser.py       # QueryIntent enum + QueryContext; LLM + regex classifiers
│   ├── retriever.py          # smart_retrieve() policy-gated + retrieve() for wiki
│   ├── loader.py             # load_pdf_bytes() with layout-aware figure extraction
│   └── vectorstore.py        # Redis Stack wrapper; dual index support
│
├── mcp_servers/
│   ├── reagent_catalog.py    # Shared business logic (search, stock check, cost estimate)
│   └── reagent_server.py     # Standalone FastMCP server (MCP-compatible)
│
├── frontend/                 # FastAPI web UI
│   ├── main.py               # App lifespan, middleware, router mounts
│   ├── config.py             # Pydantic Settings
│   ├── deps.py               # get_current_user(), require_admin()
│   ├── db/auth.py            # SQLite + bcrypt auth
│   ├── routers/
│   │   ├── auth_router.py    # /login, /logout
│   │   ├── chat_router.py    # /chat, /api/chat (SSE), /api/history
│   │   └── admin_router.py   # /admin, /api/upload
│   └── templates/            # Jinja2: base, login, chat, admin
│
├── chat/
│   ├── history.py            # Redis-backed per-user chat persistence
│   └── token_logger.py       # Per-query token usage to logs/token_usage.jsonl
│
├── data/
│   ├── docs/                 # 7 mock research papers (txt)
│   ├── assay_results.csv     # 30 compounds, IC50/EC50/selectivity, 10 SMILES
│   └── reagent_catalog.csv   # 20 commercial reagents with pricing & stock
│
├── scripts/
│   ├── build_index.py        # One-shot index builder (classic + wiki)
│   ├── watch_docs.py         # Auto-rebuild on docs/ change
│   ├── evaluate_rag.py       # Keyword-recall scoring on 7 Q&A pairs
│   └── diagnose_rag.py       # Step-by-step smart_retrieve trace for live debugging
│
├── tests/
│   ├── test_frontend/        # 37 FastAPI tests (auth, chat API, admin)
│   ├── test_rag/             # 36 tests (query_parser + smart_retrieve)
│   └── test_reagent/         # 20 tests (catalog search, stock, cost)
│
├── mcp_server.py             # Legacy MCP server (assay + paper + GitHub tools)
├── app.py                    # Streamlit UI (legacy, port 8501)
├── requirements.txt
├── .env.example
├── pytest.ini
├── CHANGELOG.md
├── TODO.md
└── PROJECT_CONTEXT.md
```

---

## Example Queries

### RAG & Research
```
"Do you know about CLCNet?"
→ entity_lookup intent; retrieves from indexed PDF; returns text context, no images

"What is the framework architecture of CLCNet?"
→ framework_or_architecture intent; returns text + approved figure crops

"What are the IC50 results in Table 2?"
→ table_or_result intent; retrieves table chunks with type-boost
```

### Assay Data
```
"What are the most potent EGFR inhibitors in the database?"
"Show BRAF compounds tested after March 2023"
"Which compounds by Chen L. are leads?"
```

### Reagent Procurement
```
"Can I buy SR-0472?"
→ search_reagents: finds Osimertinib (HY-15772, MedChemExpress, ≥99%, $1.20/mg)

"Is HY-15772 in stock? How long does it take?"
→ check_stock: ✅ In Stock, 5 business days, est. delivery 2026-05-10

"How much does 25mg of HY-15772 cost?"
→ estimate_cost: $30.00 subtotal + $25.00 shipping = $55.00
```

### External Tools
```
"Fetch the README from https://github.com/deepmind/alphafold"
"Look up papers on PROTAC degraders for oncology"
```

### Guardrail (blocked)
```
"What dosage of this compound should I take for COPD?"
→ Blocked before LLM — safe refusal message shown immediately
```

---

## Testing

```bash
# Full test suite (92 tests)
pytest tests/ -v

# By module
pytest tests/test_frontend/ -v    # 37 FastAPI tests
pytest tests/test_rag/ -v         # 36 RAG policy tests
pytest tests/test_reagent/ -v     # 20 reagent catalog tests
```

---

## Environment Variables

```bash
# Required
OPENAI_API_KEY=<your-key>                     # Google Gemini or OpenAI key

# Optional (defaults shown)
MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
REDIS_URL=redis://localhost:26379
OPENAI_BASE_URL=                              # Set for Google Gemini (see below)

# FastAPI auth (seeded at first startup)
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
RESEARCHER_USERNAME=researcher
RESEARCHER_PASSWORD=researcher123
```

**Google Gemini:**
```bash
OPENAI_API_KEY=<your-google-api-key>
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
MODEL_NAME=gemini-3.1-flash-lite-preview
EMBEDDING_MODEL=models/gemini-embedding-2
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Redis connection failed` | `docker start <container>` or `docker run redis/redis-stack:latest` |
| `Agent not ready` (503) | Redis must be running before starting uvicorn |
| `PDF upload fails` | Check file is a valid PDF < 100MB |
| `No evidence found` for a valid query | Re-upload the relevant PDF via Admin Upload |
| `Guardrail not triggering` | Check patterns in `agent/guardrails.py` |
| `History not persisting` | Verify Redis is running and `REDIS_URL` is correct |
| `MCP server not connecting` | Run `python mcp_servers/reagent_server.py` and verify FastMCP output |

---

## Architecture: Policy-Gated Retrieval

```
Query
  → guardrail_node     regex check; BLOCK or pass through
  → intent_node        LLM classifies intent + extracts entities → QueryContext
  → agent_node         selects tool(s) based on query
  → tool_node
      └── rag_search → smart_retrieve()
            ├── similarity_search_with_score(k=12)
            ├── type-boost re-ranking (intent × chunk_type matrix)
            ├── entity grounding gate  (sim ≥ 0.60 required)
            ├── evidence threshold     (sim ≥ 0.75 for context chunks)
            ├── lenient fallback       (entity_lookup / concept_definition)
            └── visual gate            (intent + entity + score + support checks)
                    ↓                          ↓
             context_for_llm          approved_visuals (→ frontend via SSE)
```

---

## License

MIT License (2026)

---

For questions or issues: [GitHub Issues](https://github.com/LeiTong02/PharmAgent/issues) or lei.tongml01@gmail.com
