#!/usr/bin/env python3
"""
Standalone PDF indexing test — no Redis or API key required.

Tests extraction and chunking of PDFs in test_dir/pdf/ and runs
keyword-based retrieval against three representative query types.

Usage:
    python scripts/test_pdf_indexing.py
"""
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.loader import load_pdf_bytes  # noqa: E402

PDF_DIR = Path(__file__).parent.parent / "test_dir" / "pdf"

QUERIES = [
    "What is the framework or method used in this paper?",
    "What are the main results or performance metrics on the dataset?",
    "What does Figure 2 show? What are the key tables and results?",
]


def keyword_search(docs, query: str, top_k: int = 3) -> list[tuple[int, str]]:
    """Simple keyword overlap scoring — no embeddings needed."""
    query_words = {w.lower() for w in re.split(r"\W+", query) if len(w) > 3}
    scored = []
    for doc in docs:
        text = doc.page_content.lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, doc.page_content))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


def test_pdf(path: Path) -> bool:
    print(f"\n{'=' * 65}")
    print(f"PDF: {path.name}  ({path.stat().st_size // 1024} KB)")
    print("=" * 65)

    try:
        pdf_bytes = path.read_bytes()
        docs = load_pdf_bytes(pdf_bytes, path.name)
    except Exception as exc:
        print(f"  FAIL during extraction: {exc}")
        return False

    all_text = "\n".join(d.page_content for d in docs)
    n_table_rows = len(re.findall(r"^\|", all_text, re.MULTILINE))
    n_figures = all_text.count("[Figure:")

    print(f"  Chunks: {len(docs)}")
    print(f"  Table rows in extracted text: {n_table_rows}")
    print(f"  Figure markers: {n_figures}")

    all_pass = True
    for query in QUERIES:
        print(f"\n  Query: {query!r}")
        results = keyword_search(docs, query)
        if results:
            score, preview = results[0]
            print(f"  Best match (score={score}): {preview[:250]!r}")
        else:
            print("  WARNING: no matching chunks found — query terms may not appear in text")
            all_pass = False

    return all_pass


def test_type_guard():
    """Verify that passing a string instead of bytes raises TypeError."""
    print("\n--- Type guard test ---")
    try:
        load_pdf_bytes("filename.pdf", b"ignored")
        print("FAIL: expected TypeError was not raised")
        return False
    except TypeError as e:
        print(f"OK: TypeError raised as expected: {e}")
        return True
    except Exception as e:
        print(f"FAIL: wrong exception type {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        sys.exit(1)

    results = []
    results.append(test_type_guard())

    for pdf_path in pdfs:
        results.append(test_pdf(pdf_path))

    print(f"\n{'=' * 65}")
    passed = sum(results)
    total = len(results)
    status = "ALL PASS" if all(results) else f"{passed}/{total} PASSED"
    print(f"RESULT: {status}")
    sys.exit(0 if all(results) else 1)
