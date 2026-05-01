#!/usr/bin/env python3
"""One-shot script to embed mock research paper documents and store them in Redis Stack.

Drops any existing index first, then re-creates it with all mock papers tagged
source_type='mock'. Run this again after changing mock paper .txt files or the
embedding model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rag.loader import load_documents
from rag.vectorstore import build_index

if __name__ == "__main__":
    print("Loading mock documents...")
    docs = load_documents()
    file_count = len(set(d.metadata.get("source_file", "") for d in docs))
    print(f"  Loaded {len(docs)} chunks from {file_count} files (source_type=mock)")

    import os
    redis_url = os.getenv("REDIS_URL", "redis://localhost:26379")
    print(f"Building Redis vector index (index: pharma_ra, url: {redis_url})...")
    build_index(docs)
    print(f"  Index stored in Redis at {redis_url}")
    print("Done. Note: any previously uploaded PDFs were removed. "
          "Use the Admin Upload page to re-index them.")
