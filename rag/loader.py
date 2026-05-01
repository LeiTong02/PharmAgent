import io
import os
from datetime import datetime, timezone
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"

_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_CHAR_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


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
    docs: list[Document] = []

    for txt_file in sorted(DOCS_DIR.glob("*.txt")):
        raw = txt_file.read_text(encoding="utf-8")
        metadata, body = _parse_header(raw)
        metadata["source_file"] = txt_file.name
        metadata["source_type"] = "mock"
        metadata["upload_timestamp"] = ""

        chunks = _CHAR_SPLITTER.create_documents(
            texts=[body],
            metadatas=[metadata] * 1,
        )
        docs.extend(chunks)

    return docs


def load_pdf_bytes(file_bytes: bytes, filename: str) -> list[Document]:
    """Parse a digital PDF and return semantically-chunked Documents.

    Two-pass chunking:
    1. MarkdownHeaderTextSplitter splits at #/##/### section boundaries.
    2. RecursiveCharacterTextSplitter subdivides any chunk still > 800 chars.
    """
    try:
        import pymupdf4llm
    except ImportError as e:
        raise ImportError("pymupdf4llm is required for PDF parsing: pip install pymupdf4llm") from e

    try:
        import fitz
        fitz_doc = fitz.open(stream=file_bytes, filetype="pdf")
        md_text = pymupdf4llm.to_markdown(fitz_doc)
        fitz_doc.close()
    except Exception as e:
        raise ValueError(f"Could not extract text from '{filename}'. "
                         f"The file may be image-only or corrupted: {e}") from e

    if not md_text.strip():
        raise ValueError(f"No text extracted from '{filename}'. "
                         "The file may be a scanned image without embedded text.")

    # Pass 1: split at markdown section headers
    header_splitter = MarkdownHeaderTextSplitter(
        _HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    header_chunks = header_splitter.split_text(md_text)

    # Pass 2: subdivide chunks that are still too large
    docs = _CHAR_SPLITTER.split_documents(header_chunks)

    ts = datetime.now(timezone.utc).isoformat()
    for doc in docs:
        doc.metadata.update({
            "source_file": filename,
            "source_type": "uploaded",
            "upload_timestamp": ts,
        })

    return docs
