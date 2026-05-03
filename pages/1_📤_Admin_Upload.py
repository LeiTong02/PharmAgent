"""Admin PDF Upload page — only accessible to admin role users."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config — must be the FIRST st.* call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PharmaRA — Admin Upload",
    page_icon="📤",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------
from auth.config import build_authenticator, login_gate

authenticator = build_authenticator()
role = login_gate(authenticator)

if role != "admin":
    st.error("This page is restricted to admin users.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: logout + navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📤 Admin Upload")
    username = st.session_state.get("username", "")
    st.caption(f"Logged in as **{username}** (admin)")
    try:
        authenticator.logout(button_name="Logout", location="sidebar")
    except Exception:
        pass
    st.divider()
    st.markdown("### Actions")
    st.markdown("Use this page to upload PDF research papers into the vector index.")
    st.divider()
    st.caption(f"Redis: `{os.getenv('REDIS_URL', 'redis://localhost:26379')}`")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("📤 Upload Research PDFs")
st.caption(
    "Uploaded PDFs are parsed, chunked, embedded, and added to the `pharma_ra` Redis index. "
    "They will be immediately searchable in the chat."
)

# ---------------------------------------------------------------------------
# Already-indexed uploaded files
# ---------------------------------------------------------------------------
from rag.vectorstore import list_uploaded_files

with st.expander("📚 Currently indexed uploaded files", expanded=False):
    uploaded_files = list_uploaded_files()
    if uploaded_files:
        for f in uploaded_files:
            st.markdown(f"- `{f}`")
    else:
        st.info("No uploaded PDFs in index yet.")

st.divider()

# ---------------------------------------------------------------------------
# File uploader
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Digital PDFs (journal articles, preprints, reports). Scanned image-only PDFs are not supported.",
)

if uploaded:
    already_indexed = set(list_uploaded_files())

    st.markdown(f"**{len(uploaded)} file(s) selected.** Click the button below to index them.")

    if st.button("Index selected PDFs", type="primary", use_container_width=True):
        from rag.loader import load_pdf_bytes
        from rag.vectorstore import add_documents

        results = []  # (filename, status, chunk_count, message)

        progress_bar = st.progress(0, text="Starting...")

        for i, file in enumerate(uploaded):
            fname = file.name
            progress_bar.progress(
                (i) / len(uploaded),
                text=f"Processing {fname} ({i + 1}/{len(uploaded)})..."
            )

            if fname in already_indexed:
                results.append((fname, "skipped", 0, "Already in index — skipped."))
                continue

            try:
                pdf_bytes = file.read()
                docs = load_pdf_bytes(pdf_bytes, fname)
                n = add_documents(docs)

                # Also generate wiki pages and add to pharma_wiki index
                from rag.wiki_generator import docs_to_wiki_documents
                from rag.vectorstore import add_wiki_documents
                with st.spinner(f"Generating wiki pages for {fname}..."):
                    wiki_docs = docs_to_wiki_documents(docs)
                    add_wiki_documents(wiki_docs)

                results.append((fname, "success", n, f"Indexed {n} chunks + {len(wiki_docs)} wiki page(s)."))
            except ValueError as e:
                results.append((fname, "error", 0, str(e)))
            except Exception as e:
                results.append((fname, "error", 0, f"Unexpected error: {e}"))

        progress_bar.progress(1.0, text="Done.")

        # Summary
        st.divider()
        st.markdown("### Results")
        for fname, status, count, msg in results:
            if status == "success":
                st.success(f"**{fname}** — {msg}")
            elif status == "skipped":
                st.warning(f"**{fname}** — {msg}")
            else:
                st.error(f"**{fname}** — {msg}")

        success_count = sum(1 for _, s, _, _ in results if s == "success")
        total_chunks = sum(c for _, s, c, _ in results if s == "success")
        if success_count:
            st.info(
                f"Successfully indexed **{success_count} PDF(s)** → "
                f"**{total_chunks} chunks** added to the `pharma_ra` index."
            )
        st.rerun()

# ---------------------------------------------------------------------------
# Danger zone: rebuild mock index
# ---------------------------------------------------------------------------
st.divider()
with st.expander("⚠️ Rebuild mock index (removes all uploaded PDFs)", expanded=False):
    st.warning(
        "This will **drop the entire `pharma_ra` index** and rebuild it with only the 5 mock papers. "
        "All uploaded PDFs will be removed from the index (the PDF files themselves are not deleted)."
    )
    if st.button("Rebuild mock index", type="secondary"):
        from rag.loader import load_documents
        from rag.vectorstore import build_index, build_wiki_index
        from rag.wiki_generator import docs_to_wiki_documents

        with st.spinner("Rebuilding classic index..."):
            docs = load_documents()
            build_index(docs)
        with st.spinner("Generating and rebuilding wiki index..."):
            wiki_docs = docs_to_wiki_documents(docs)
            build_wiki_index(wiki_docs)
        st.success(
            f"Both indexes rebuilt: {len(docs)} chunks + {len(wiki_docs)} wiki pages from 5 papers."
        )
        st.rerun()
