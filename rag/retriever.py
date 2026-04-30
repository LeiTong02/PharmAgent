from __future__ import annotations
from langchain_community.vectorstores import FAISS


def retrieve(vectorstore: FAISS, query: str, k: int = 4) -> tuple[str, list[str]]:
    """Return (context_text, citations) for the top-k chunks matching query."""
    results = vectorstore.similarity_search(query, k=k)

    context_parts: list[str] = []
    citations: list[str] = []
    seen: set[str] = set()

    for doc in results:
        m = doc.metadata
        title = m.get("title", m.get("source_file", "Unknown"))
        year = m.get("year", "")
        authors = m.get("authors", "")

        citation = f"{title} ({year})"
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
            if authors:
                citations[-1] = f"{title} — {authors} ({year})"

        context_parts.append(
            f"[Source: {title}, {year}]\n{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)
    return context, citations
