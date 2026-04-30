"""Pharma Research Assistant — Streamlit UI."""
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PharmaRA — Pharma Research Assistant",
    page_icon="🧬",
    layout="wide",
)

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
# Load vectorstore + build graph (cached — runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading research library...")
def load_agent():
    from rag.vectorstore import load_index
    from agent.graph import build_graph
    try:
        vs = load_index()
    except Exception as e:
        return None, str(e)
    graph = build_graph(vs)
    return graph, None


graph, load_error = load_agent()

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
    st.divider()

    st.markdown("### About")
    st.markdown(
        "PharmaRA is an AI-powered assistant for pharmaceutical R&D teams. "
        "It combines **RAG over research papers**, **structured assay data queries**, "
        "and **external literature lookup** in a single conversational interface."
    )
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

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "citations" not in st.session_state:
    st.session_state.citations = {}  # message_index → list of citation strings

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧬 Pharma Research Assistant")
st.caption(
    "Ask about research findings, assay data, compounds, clinical trials, or external papers and repositories."
)

# ---------------------------------------------------------------------------
# Display chat history
# ---------------------------------------------------------------------------
for i, msg in enumerate(st.session_state.messages):
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)
        # Show citations for assistant messages
        if role == "assistant" and i in st.session_state.citations:
            cites = st.session_state.citations[i]
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
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    human_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(human_msg)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            state_in = {"messages": st.session_state.messages, "blocked": False}
            result = graph.invoke(state_in, config={"recursion_limit": 10})

        final_msgs = result["messages"]
        blocked = result.get("blocked", False)

        # Find the last AI message
        ai_reply = None
        for m in reversed(final_msgs):
            if isinstance(m, AIMessage) and not m.tool_calls:
                ai_reply = m
                break

        if ai_reply is None:
            ai_reply = AIMessage(content="No response generated.")

        # ChatGoogleGenerativeAI (thinking models) returns content as a list of blocks
        raw = ai_reply.content
        if isinstance(raw, list):
            reply_text = "\n".join(
                block["text"] for block in raw
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            reply_text = raw

        if blocked:
            st.error(reply_text)
        else:
            st.markdown(reply_text)

        # Extract citations from the reply text
        import re
        cite_matches = re.findall(r"\[Source:\s*([^\]]+)\]", reply_text)
        citation_idx = len(st.session_state.messages)  # index of the reply we're about to append

        if cite_matches and not blocked:
            with st.expander(f"📄 Sources ({len(set(cite_matches))})", expanded=False):
                for c in set(cite_matches):
                    st.markdown(f"- {c}")
            st.session_state.citations[citation_idx] = list(set(cite_matches))

    st.session_state.messages.append(ai_reply)
    st.rerun()
