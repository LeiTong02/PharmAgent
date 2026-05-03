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
from rag.vectorstore import build_index, build_wiki_index
from rag.wiki_generator import docs_to_wiki_documents

if __name__ == "__main__":
    import os
    redis_url = os.getenv("REDIS_URL", "redis://localhost:26379")

    print("Loading mock documents...")
    docs = load_documents()
    file_count = len(set(d.metadata.get("source_file", "") for d in docs))
    print(f"  Loaded {len(docs)} chunks from {file_count} files")

    print(f"[Classic] Building pharma_ra index ({redis_url})...")
    build_index(docs)
    print(f"  Stored {len(docs)} chunks in pharma_ra")

    print(f"[Wiki] Generating wiki pages for {file_count} documents...")
    wiki_docs = docs_to_wiki_documents(docs)
    print(f"  Generated {len(wiki_docs)} wiki pages")
    print(f"[Wiki] Building pharma_wiki index ({redis_url})...")
    build_wiki_index(wiki_docs)
    print(f"  Stored {len(wiki_docs)} wiki pages in pharma_wiki")

    print("Done. Both indexes ready.")
    print("Note: any previously uploaded PDFs were removed. "
          "Use the Admin Upload page to re-index them.")
