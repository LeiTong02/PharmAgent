# PharmaRA — Pharmaceutical Research Assistant

An AI-powered research assistant for pharmaceutical R&D teams. Combines **RAG over research papers**, **structured assay data queries**, **external literature lookup**, and **admin-controlled document management** in a single conversational interface.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![Redis](https://img.shields.io/badge/Redis-Stack-red)

---

## Features

### 🤖 Intelligent Agent
- **Guardrail System**: Blocks unsafe medical advice queries (regex-based for speed & reliability)
- **Multi-Tool Orchestration**: Routes queries to the best tool via LangGraph agent
- **RAG with Citations**: Retrieves from vectorized research papers with inline source attribution

### 📚 Research Tools
- **RAG Search** — Query research papers with semantic search + citation tracking
- **Assay Data Lookup** — Structured queries on IC50/EC50/selectivity data (CSV-backed)
- **GitHub README Fetch** — Retrieve documentation from public repositories
- **Semantic Scholar** — Look up and cite academic papers by keyword

### 🧬 Document Management
- **Admin PDF Upload** — Multi-file OCR → markdown → 2-pass semantic chunking
- **Redis Vector Index** — Persistent embedding storage (survives app restarts)
- **Per-Paper Tracking** — Metadata includes source type, upload timestamp, filename

### 👥 Multi-User & Persistence
- **Per-User Isolation** — Admin and researcher each see only their own chat history
- **Redis Chat Persistence** — History survives browser restarts and page reloads
- **Session-Only Auth** — Users must re-login after closing browser (security feature)
- **Clear History Button** — Wipe chat + citations on demand

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI** | Streamlit 1.35+ |
| **Agent** | LangGraph 0.2+ with `START → guardrail_node → agent_node ⇄ tool_node → END` |
| **LLM** | Google Gemini (via OpenAI-compatible API) or OpenAI GPT-4 |
| **Embeddings** | Google Gemini Embedding or OpenAI text-embedding-3-small |
| **Vector Store** | Redis Stack with RedisVectorStore |
| **Chat History** | Redis (persistent, JSON-serialized LangChain messages) |
| **Auth** | streamlit-authenticator with bcrypt |
| **PDF Parsing** | pymupdf4llm + 2-pass chunking (MarkdownHeaderSplitter → RecursiveCharacterSplitter) |

**Python**: 3.10.18 (conda `py310` env)

---

## Quick Start

### Prerequisites
- Docker (for Redis Stack)
- Python 3.10+ with conda or venv
- OpenAI API key OR Google Gemini API key (configured as OpenAI-compatible endpoint)

### 1. Setup Environment

```bash
# Clone repo
git clone https://github.com/LeiTong02/PharmAgent.git
cd PharmAgent

# Create conda env (if not already done)
conda create -n py310 python=3.10
conda activate py310

# Install dependencies
pip install -r requirements.txt

# Copy environment file and add your API key
cp .env.example .env
# Edit .env: set OPENAI_API_KEY (or GOOGLE_GENAI_API_KEY)
```

### 2. Start Redis Stack

```bash
# If you have the Redis container from before
docker ps -a | grep redis  # find container ID
docker start <container_id>

# OR create a fresh container
docker run -d --name redis-pharma \
  -p 26379:6379 \
  redis/redis-stack:latest
```

Verify Redis is running:
```bash
redis-cli -p 26379 ping
# Response: PONG
```

### 3. Build Vector Index

```bash
# Embed the 5 mock research papers + store in Redis
python scripts/build_index.py
```

### 4. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

**Demo Credentials:**
- **Admin**: `admin` / (any password, auto-generated)
- **Researcher**: `researcher` / (any password, auto-generated)

---

## Project Structure

```
PharmAgent/
├── app.py                           # Main Streamlit UI + chat loop
├── pages/
│   └── 1_📤_Admin_Upload.py        # Multi-page admin-only PDF upload
├── agent/
│   ├── guardrails.py                # Medical advice regex blocker
│   ├── prompts.py                   # System prompt + citation format
│   ├── tools.py                     # 4 LangChain @tool functions
│   └── graph.py                     # LangGraph StateGraph definition
├── rag/
│   ├── loader.py                    # Load .txt papers + load_pdf_bytes()
│   ├── vectorstore.py               # Redis vector store wrapper
│   └── retriever.py                 # RAG retrieve() with citations
├── chat/
│   ├── __init__.py
│   └── history.py                   # Redis chat persistence
├── auth/
│   ├── __init__.py
│   └── config.py                    # streamlit-authenticator setup
├── data/
│   ├── docs/                        # 5 mock research papers (txt)
│   └── assay_results.csv            # 30-row mock assay database
├── scripts/
│   └── build_index.py               # CLI to build Redis index
├── requirements.txt                 # Python dependencies
├── .env.example                     # Env template
├── README.md                        # This file
├── CHANGELOG.md                     # Development history
├── TODO.md                          # Feature roadmap
└── PROJECT_CONTEXT.md               # Architecture & quirks

```

---

## Example Queries

Try these in the chat interface:

1. **RAG**: "Summarize the key findings on EGFR T790M resistance mutations."
2. **CSV**: "What are the IC50 values for EGFR inhibitors in the database?"
3. **Guardrail (blocked)**: "What dosage should I take for COPD?" → Red error banner
4. **GitHub**: "Fetch the README from https://github.com/deepmind/alphafold"
5. **Scholar**: "Look up papers on protein structure prediction with AlphaFold."

---

## Key Features Deep Dive

### Per-User Chat History

Each user (admin/researcher) has isolated session state:
- `messages_{username}` — list of LangChain `Message` objects
- `citations_{username}` — dict mapping message index to source list
- Data persists in Redis, survives page reload and browser restart

### Admin PDF Upload

Access via sidebar **Admin Upload** tab (admin role only):
1. Select 1+ PDF files
2. Auto-parses to markdown via `pymupdf4llm`
3. 2-pass chunking:
   - **Pass 1**: Split at markdown headers (`#`, `##`, `###`) for semantic coherence
   - **Pass 2**: Further split chunks >800 chars to fit embedding limits
4. Stores in same Redis index as mock papers
5. Queryable immediately; no app restart needed

### Guardrail System

Regex-based blocker for medical advice queries:
```regex
Pattern: r'(?i)(dosage|dose|medical advice|should i take|should i use).*\bfor\b.*(disease|condition|illness|patient)'
```

Blocks: "What dosage for COPD?" ✅  
Allows: "What dose was used in the Phase 2 trial?" ✅

---

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...                      # OpenAI key (or Google Gemini key)

# Optional (defaults shown)
MODEL_NAME=gpt-4o-mini                      # LLM model ID
EMBEDDING_MODEL=text-embedding-3-small      # Embedding model
REDIS_URL=redis://localhost:26379           # Redis connection
OPENAI_BASE_URL=...                         # Custom endpoint (Google Gemini, etc.)

# Auth (generated at first run)
AUTH_ADMIN_USERNAME=admin
AUTH_ADMIN_PASSWORD=<auto-hashed>
AUTH_USERS={"researcher": "<auto-hashed>"}
```

For **Google Gemini**:
```bash
OPENAI_API_KEY=<your-google-api-key>
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

---

## Testing

### Programmatic E2E Test

```bash
python -c "
from agent.graph import build_graph
from rag.vectorstore import load_index

vs = load_index()
graph = build_graph(vs)

# Test all 5 scenarios
tests = [
    ('What is EGFR T790M?', 'RAG'),
    ('Top IC50 values?', 'CSV'),
    ('What dosage for COPD?', 'Guardrail'),
    ('Fetch https://github.com/deepmind/alphafold', 'GitHub'),
    ('Papers on AlphaFold', 'Scholar'),
]

for query, tool in tests:
    result = graph.invoke({'messages': [HumanMessage(content=query)], 'blocked': False})
    print(f'✓ {tool}')
"
```

### UI Smoke Test

1. Open http://localhost:8501
2. Login as admin/researcher
3. Click 7 sidebar example queries → all work without errors
4. Send custom message → shows sources in expandable panel
5. Admin: upload PDF → queryable after upload
6. Click "Clear chat history" → history wiped

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Redis connection failed` | Check `docker ps` → `docker start <container>` |
| `No login form on first load` | Clear browser cookies, refresh |
| `PDF upload fails` | Ensure PDF is <100MB; try smaller PDF first |
| `Guardrail not triggering` | Check regex pattern in `agent/guardrails.py` |
| `Citations missing` | Ensure `[Source: ...]` format in LLM prompt |
| `History not persisting` | Verify Redis is running and `REDIS_URL` is correct |

For more details, see [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) and [CHANGELOG.md](CHANGELOG.md).

---

## Future Roadmap

- [ ] MCP integration: wrap as Claude Code MCP server
- [ ] Real document corpus: replace mock papers with internal Sanofi docs
- [ ] Streaming responses: use LangChain streaming + `st.write_stream`
- [ ] Molecule viewer: RDKit 2D structures for compound IDs
- [ ] Evaluation harness: RAGAS/ARES for RAG quality scoring
- [ ] Cost tracking: log token usage per query

---

## License

Proprietary — Sanofi demo project (2026).

---

## Contact

**Built for**: Sanofi R&D interview demo  
**Demo Date**: May 6, 2026  
**Questions**: lei.tongml01@gmail.com

