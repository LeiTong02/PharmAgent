import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"


def _parse_header(text: str) -> tuple[dict, str]:
    """Extract TITLE/AUTHORS/YEAR/JOURNAL header fields and return (metadata, body)."""
    metadata = {}
    lines = text.split("\n")
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            body_start = i + 1
            break
        for field in ("TITLE", "AUTHORS", "YEAR", "JOURNAL"):
            if line.startswith(f"{field}:"):
                metadata[field.lower()] = line[len(field) + 1:].strip()
    body = "\n".join(lines[body_start:]).strip()
    return metadata, body


def load_documents() -> list[Document]:
    """Load all .txt research papers from data/docs/, returning chunked Documents with metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs: list[Document] = []

    for txt_file in sorted(DOCS_DIR.glob("*.txt")):
        raw = txt_file.read_text(encoding="utf-8")
        metadata, body = _parse_header(raw)
        metadata["source_file"] = txt_file.name

        chunks = splitter.create_documents(
            texts=[body],
            metadatas=[metadata] * 1,
        )
        docs.extend(chunks)

    return docs
