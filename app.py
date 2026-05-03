"""Pharma Research Assistant — Streamlit UI."""
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

from chat.history import load_history, save_history, clear_history

# ---------------------------------------------------------------------------
# Page config — must be the FIRST st.* call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PharmaRA — Pharma Research Assistant",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
from auth.config import build_authenticator, login_gate

authenticator = build_authenticator()
user_role = login_gate(authenticator)  # stops page if not authenticated


def _extract_text(content) -> str:
    """Extract plain text from AIMessage content.

    Gemini thinking models return content as a list of typed blocks
    e.g. [{"type": "text", "text": "...", "extras": {"signature": "..."}}].
    This normalises both that format and plain strings to a single str.
    """
    if isinstance(content, list):
        return "\n".join(
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content or ""


def _render_molecule_structures(text: str) -> None:
    """Render 2D molecule structures for compound IDs found in assay result text.

    Looks up SMILES from the assay CSV and fetches PNG images from PubChem.
    Only called when query_assay_data tool was invoked.
    """
    import urllib.parse
    import pandas as pd

    smiles_map: dict[str, str] = {}
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "assay_results.csv")
        df = pd.read_csv(csv_path)
        if "smiles" in df.columns:
            for _, row in df.iterrows():
                smi = str(row.get("smiles", "")).strip()
                cid = str(row.get("compound_id", "")).upper().strip()
                if smi and smi not in ("nan", ""):
                    smiles_map[cid] = smi
    except Exception:
        return

    if not smiles_map:
        return

    found = re.findall(r"\bSR-\d+\b", text, re.IGNORECASE)
    found = [c.upper() for c in found if c.upper() in smiles_map]
    found = list(dict.fromkeys(found))[:4]  # deduplicate, max 4 structures

    if not found:
        return

    st.caption("🔬 Compound Structures")
    cols = st.columns(len(found))
    for col, cid in zip(cols, found):
        encoded = urllib.parse.quote(smiles_map[cid], safe="")
        img_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}"
            "/PNG?image_size=200x200"
        )
        try:
            col.image(img_url, caption=cid, width=150)
        except Exception:
            col.caption(f"{cid} (structure unavailable)")


# ---------------------------------------------------------------------------
# Env check
# ---------------------------------------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "**OPENAI_API_KEY not set.** Copy `.env.example` to `.env` and add your API key, "
        "then restart the app with `streamlit run app.py`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load vectorstore + build graph (cached per mode — runs once per mode)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading research library...")
def load_agent(mode: str = "classic"):
    from rag.vectorstore import load_index, load_wiki_index
    from agent.graph import build_graph
    try:
        vs = load_index()
    except Exception as e:
        return None, str(e)
    wiki_vs = None
    if mode == "wiki":
        try:
            wiki_vs = load_wiki_index()
        except Exception:
            pass  # wiki index not built yet — app will warn user
    graph = build_graph(vs, wiki_vs, mode)
    return graph, None


rag_mode = st.session_state.get("rag_mode", "classic")
graph, load_error = load_agent(rag_mode)

if load_error:
    st.error(
        f"**Could not connect to Redis vector index.**\n\n"
        f"Make sure Redis Stack is running on `{os.getenv('REDIS_URL', 'redis://localhost:26379')}` "
        f"and run:\n\n```bash\npython scripts/build_index.py\n```\n\nError: {load_error}"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧬 PharmaRA")
    st.caption("Pharmaceutical Research Assistant")

    username = st.session_state.get("username", "")
    st.caption(f"Logged in as **{username}** ({user_role})")
    try:
        authenticator.logout(button_name="Logout", location="sidebar")
    except Exception:
        pass
    if st.button("🗑️ Clear chat history", use_container_width=True):
        clear_history(username)
        st.session_state[_msg_key] = []
        st.session_state[_cite_key] = {}
        st.session_state[_trace_key] = {}
        st.rerun()
    st.divider()

    st.markdown("### Retrieval Mode")
    st.radio(
        "Select retrieval strategy:",
        options=["classic", "wiki"],
        format_func=lambda x: "🔍 Classic RAG" if x == "classic" else "📖 Wiki RAG",
        key="rag_mode",
        help=(
            "**Classic RAG**: searches raw document chunks via vector similarity.\n\n"
            "**Wiki RAG**: searches pre-compiled wiki pages (summary + concepts + findings) "
            "generated by LLM at index time."
        ),
    )
    if rag_mode == "wiki" and graph is not None:
        # Warn if wiki vectorstore wasn't loaded (index not built yet)
        from agent.tools import _wiki_vectorstore
        if _wiki_vectorstore is None:
            st.warning(
                "⚠️ Wiki index not found. Run:\n```\npython scripts/build_index.py\n```"
            )
    st.divider()

    st.markdown("### About")
    st.markdown(
        "PharmaRA is an AI-powered assistant for pharmaceutical R&D teams. "
        "It combines **RAG over research papers**, **structured assay data queries**, "
        "and **external literature lookup** in a single conversational interface."
    )

    with st.expander("⚙️ How it works", expanded=False):
        st.markdown(
            """
**Query → Agent → Tools → Answer**

1. **Safety guardrail** — blocks personal medical advice before any LLM call
2. **LLM planner** — decides which tool(s) to call based on your query
3. **Tools available:**
   - 🔍 **RAG search** — vector similarity over research paper chunks (`pharma_ra`)
   - 📖 **Wiki search** — pre-compiled wiki pages per document (`pharma_wiki`)
   - 📊 **Assay DB** — filtered queries on `assay_results.csv` (IC50, EC50, selectivity…)
   - 🐙 **GitHub** — fetches README from any public repo URL
   - 📄 **Scholar** — paper metadata from Semantic Scholar / arXiv fallback
4. **Answer synthesis** — LLM composes a final response with `[Source: ...]` citations

**Indexing pipeline:**
```
PDF / .txt → chunk → embed → Redis pharma_ra   (Classic RAG)
                   → LLM wiki page → embed → Redis pharma_wiki  (Wiki RAG)
```
            """
        )
    with st.expander("💰 Token usage", expanded=False):
        try:
            from chat.token_logger import get_session_usage
            usage = get_session_usage(username, limit=50)
            if usage["query_count"] > 0:
                st.metric("Total tokens (last 50 queries)", f"{usage['total_tokens']:,}")
                col1, col2 = st.columns(2)
                col1.metric("Prompt", f"{usage['prompt_tokens']:,}")
                col2.metric("Completion", f"{usage['completion_tokens']:,}")
                st.caption(f"Across {usage['query_count']} queries")
            else:
                st.caption("No queries yet this session.")
        except Exception:
            st.caption("Usage data unavailable.")
    st.divider()

    st.markdown("### Example queries")
    examples = [
        "What are the IC50 values for EGFR inhibitors in the database?",
        "Summarize the key findings on EGFR T790M resistance mutations.",
        "Which compounds have selectivity ratio above 100?",
        "What is the clinical trial outcome for SAN-4891 in COPD?",
        "Show me lead compounds for CDK4/Cyclin D1.",
        "Fetch the README from https://github.com/deepmind/alphafold",
        "Look up papers on protein structure prediction with AlphaFold.",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state.pending_input = ex

    st.divider()
    st.markdown("### Safety note")
    st.warning(
        "PharmaRA is for **research use only** and cannot provide medical advice. "
        "All queries requesting personal health guidance will be declined."
    )
    st.divider()
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    st.caption(f"Model: `{model}` · Embeddings: `{os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')}`")

# Per-user session state keys so each account has its own isolated chat history
_msg_key = f"messages_{username}"
_cite_key = f"citations_{username}"
_trace_key = f"traces_{username}"

# ---------------------------------------------------------------------------
# Chat state (namespaced per user, backed by Redis)
# ---------------------------------------------------------------------------
if _msg_key not in st.session_state:
    msgs, cites = load_history(username)
    st.session_state[_msg_key] = msgs
    st.session_state[_cite_key] = cites
if _trace_key not in st.session_state:
    st.session_state[_trace_key] = {}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧬 Pharma Research Assistant")
mode_label = "📖 Wiki RAG" if rag_mode == "wiki" else "🔍 Classic RAG"
st.caption(
    f"Ask about research findings, assay data, compounds, clinical trials, or external papers. "
    f"&nbsp;·&nbsp; Mode: **{mode_label}**"
)

# ---------------------------------------------------------------------------
# Display chat history
# ---------------------------------------------------------------------------
for i, msg in enumerate(st.session_state[_msg_key]):
    msg_role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(msg_role):
        st.markdown(_extract_text(msg.content))
        if msg_role == "assistant":
            if i in st.session_state[_trace_key]:
                traces = st.session_state[_trace_key][i]
                if traces:
                    with st.expander(f"🔧 Tool calls ({len(traces)})", expanded=False):
                        for name, args in traces:
                            args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                            st.caption(f"`{name}({args_str})`")
            if i in st.session_state[_cite_key]:
                cites = st.session_state[_cite_key][i]
                if cites:
                    with st.expander(f"📄 Sources ({len(cites)})", expanded=False):
                        for c in cites:
                            st.markdown(f"- {c}")

# ---------------------------------------------------------------------------
# Input handling (sidebar button OR chat input)
# ---------------------------------------------------------------------------
pending = st.session_state.pop("pending_input", None)
user_input = pending or st.chat_input("Ask a research question...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    human_msg = HumanMessage(content=user_input)
    st.session_state[_msg_key].append(human_msg)

    with st.chat_message("assistant"):
        _tool_status = st.empty()
        _text_ph = st.empty()

        final_text = ""
        tool_traces: list[tuple[str, dict]] = []
        blocked = False
        last_token_usage: dict = {}

        state_in = {"messages": st.session_state[_msg_key], "blocked": False}

        try:
            for event in graph.stream(
                state_in, stream_mode="updates", config={"recursion_limit": 10}
            ):
                for node, update in event.items():
                    msgs = update.get("messages", [])

                    if node == "guardrail_node":
                        if update.get("blocked"):
                            blocked = True
                        for m in msgs:
                            if isinstance(m, AIMessage):
                                final_text = _extract_text(m.content)

                    elif node == "agent_node":
                        for m in msgs:
                            if isinstance(m, AIMessage) and m.tool_calls:
                                for tc in m.tool_calls:
                                    tool_traces.append((tc["name"], tc.get("args", {})))
                                    _tool_status.caption(f"⚙️ Calling `{tc['name']}`...")
                            elif isinstance(m, AIMessage) and not m.tool_calls:
                                usage = (
                                    m.response_metadata.get("token_usage")
                                    or m.response_metadata.get("usage_metadata")
                                    or {}
                                )
                                if usage:
                                    last_token_usage = usage
                                final_text = _extract_text(m.content)
                                if not blocked:
                                    _text_ph.markdown(final_text)

                    elif node == "tool_node":
                        _tool_status.caption("⚙️ Processing results...")

        except Exception as exc:
            final_text = f"An error occurred: {exc}"
            _text_ph.error(final_text)

        _tool_status.empty()

        if blocked:
            _text_ph.empty()
            st.error(final_text)
            st.toast("Safety guardrail triggered — medical advice request blocked.", icon="🚫")
        elif not final_text:
            final_text = "No response generated."
            _text_ph.markdown(final_text)
        else:
            _text_ph.markdown(final_text)

        # Molecule viewer: render 2D structures when assay data was queried
        if not blocked and any(name == "query_assay_data" for name, _ in tool_traces):
            _render_molecule_structures(final_text)

        # Log token usage for cost tracking
        if last_token_usage:
            from chat.token_logger import log_token_usage
            log_token_usage(
                username=username,
                query=user_input,
                usage=last_token_usage,
                model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                mode=rag_mode,
            )

        citation_idx = len(st.session_state[_msg_key])

        if tool_traces and not blocked:
            with st.expander(f"🔧 Tool calls ({len(tool_traces)})", expanded=False):
                for name, args in tool_traces:
                    args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                    st.caption(f"`{name}({args_str})`")
            st.session_state[_trace_key][citation_idx] = tool_traces

        cite_matches = re.findall(r"\[Source:\s*([^\]]+)\]", final_text)
        if cite_matches and not blocked:
            with st.expander(f"📄 Sources ({len(set(cite_matches))})", expanded=False):
                for c in set(cite_matches):
                    st.markdown(f"- {c}")
            st.session_state[_cite_key][citation_idx] = list(set(cite_matches))

    ai_reply = AIMessage(content=final_text)
    st.session_state[_msg_key].append(ai_reply)
    save_history(username, st.session_state[_msg_key], st.session_state[_cite_key])
