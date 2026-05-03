"""Wiki page generator: converts research documents into structured wiki pages using LLM."""
from __future__ import annotations

import os
from collections import defaultdict

from langchain_core.documents import Document

_WIKI_PROMPT = """\
You are a pharmaceutical research knowledge curator. Given a research document, \
create a concise structured wiki page.

Format your response as markdown with EXACTLY these sections:

## Summary
[2-3 sentences covering the main contribution or finding]

## Key Concepts
- **[concept name]**: [brief explanation — 1 sentence]
[list 4-6 concepts]

## Methodology
[1-2 sentences on the experimental or analytical approach]

## Key Findings
- [finding 1]
- [finding 2]
- [finding 3]

Keep each section brief. Focus on facts that help a pharmaceutical researcher."""


def _get_llm():
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    if "google" in base_url.lower():
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            max_output_tokens=1000,
            google_api_key=os.getenv("OPENAI_API_KEY"),
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=0, max_tokens=1000, base_url=base_url or None)


def _extract_text(content) -> str:
    if isinstance(content, list):
        return "\n".join(
            b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return content or ""


def generate_wiki_page(title: str, full_text: str) -> str:
    """Call LLM to generate a structured wiki markdown page for one document."""
    llm = _get_llm()
    prompt = f"{_WIKI_PROMPT}\n\nDocument title: {title}\n\nDocument content:\n{full_text[:6000]}"
    response = llm.invoke(prompt)
    return _extract_text(response.content)


def docs_to_wiki_documents(docs: list[Document]) -> list[Document]:
    """Group documents by source_file, generate one wiki page per file, return as Documents."""
    groups: dict[str, list[Document]] = defaultdict(list)
    for doc in docs:
        key = doc.metadata.get("source_file", "unknown")
        groups[key].append(doc)

    wiki_docs = []
    for source_file, group_docs in groups.items():
        meta = group_docs[0].metadata.copy()
        title = meta.get("title", source_file)
        full_text = "\n\n".join(d.page_content for d in group_docs)
        wiki_content = generate_wiki_page(title, full_text)
        wiki_docs.append(Document(
            page_content=f"# {title}\n\n{wiki_content}",
            metadata={
                "title": title,
                "authors": meta.get("authors", ""),
                "year": meta.get("year", ""),
                "journal": meta.get("journal", ""),
                "source_file": source_file,
                "source_type": meta.get("source_type", "mock"),
                "upload_timestamp": meta.get("upload_timestamp", ""),
            },
        ))

    return wiki_docs
