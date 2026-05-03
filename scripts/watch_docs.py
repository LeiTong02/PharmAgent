#!/usr/bin/env python3
"""Watch data/docs/ for new or changed .txt files and auto-rebuild both RAG indexes.

Run with:
    python scripts/watch_docs.py

The script polls every 10 seconds. When it detects a new, removed, or modified
.txt file it rebuilds pharma_ra (classic chunks) and pharma_wiki (wiki pages)
from scratch, then prints a summary.

Ctrl-C to stop.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"
POLL_INTERVAL = 10  # seconds


def _snapshot() -> dict[str, float]:
    """Return {filename: mtime} for all .txt files in DOCS_DIR."""
    return {f.name: f.stat().st_mtime for f in sorted(DOCS_DIR.glob("*.txt"))}


def _rebuild() -> None:
    from rag.loader import load_documents
    from rag.vectorstore import build_index, build_wiki_index
    from rag.wiki_generator import docs_to_wiki_documents

    print("[watch] Loading documents...")
    docs = load_documents()
    file_count = len(set(d.metadata.get("source_file", "") for d in docs))
    print(f"[watch] {len(docs)} chunks from {file_count} files")

    print("[watch] Rebuilding pharma_ra (classic)...")
    build_index(docs)

    print("[watch] Generating wiki pages...")
    wiki_docs = docs_to_wiki_documents(docs)
    print(f"[watch] {len(wiki_docs)} wiki pages → rebuilding pharma_wiki...")
    build_wiki_index(wiki_docs)

    print(f"[watch] Done — {len(docs)} chunks + {len(wiki_docs)} wiki pages ready.\n")


def main() -> None:
    print(f"[watch] Watching {DOCS_DIR} (polling every {POLL_INTERVAL}s) — Ctrl-C to stop")
    prev = _snapshot()
    print(f"[watch] Initial state: {len(prev)} file(s): {', '.join(sorted(prev)) or '(none)'}")

    while True:
        time.sleep(POLL_INTERVAL)
        curr = _snapshot()

        new_files = sorted(set(curr) - set(prev))
        removed = sorted(set(prev) - set(curr))
        modified = sorted(
            f for f in curr if f in prev and curr[f] != prev[f]
        )

        if new_files or removed or modified:
            if new_files:
                print(f"[watch] New:      {new_files}")
            if removed:
                print(f"[watch] Removed:  {removed}")
            if modified:
                print(f"[watch] Modified: {modified}")
            try:
                _rebuild()
            except Exception as exc:
                print(f"[watch] ERROR during rebuild: {exc}")
            prev = curr


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[watch] Stopped.")
