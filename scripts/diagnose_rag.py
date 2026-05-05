"""Diagnostic script: trace smart_retrieve step-by-step for two test queries."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from rag.query_parser import parse_query, QueryIntent
from rag.retriever import (
    EVIDENCE_THRESHOLD,
    _GROUNDING_THRESHOLD,
    _LENIENT_INTENTS,
    _IMAGE_CHUNK_TYPES,
    _VISUAL_INTENTS,
    _entity_in_doc,
    _get_type_boost,
    _has_visual_support,
    _dedupe_visuals,
    _to_visual_dict,
    _NO_EVIDENCE_MSG,
)
from rag.vectorstore import load_index


def sep(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def subsep(title: str) -> None:
    print(f"\n  --- {title} ---")


def diagnose(vs, query: str) -> None:
    sep(f"QUERY: {query!r}")

    # ── 1. Intent + entities ────────────────────────────────────────────────
    qctx = parse_query(query)
    print(f"\n1. DETECTED INTENT    : {qctx.intent.value}")
    print(f"   EXTRACTED ENTITIES : {qctx.entities}")
    print(f"   FIGURE REFS        : {qctx.figure_refs}")
    print(f"   TABLE REFS         : {qctx.table_refs}")

    # ── 2. Document scope ───────────────────────────────────────────────────
    print(f"\n2. DOCUMENT SCOPE")
    print(f"   Index: pharma_ra (all uploaded docs, no session filter)")
    print(f"   (full index — all documents in scope for this retrieval)")

    # ── 3. Raw top-k retrieval ──────────────────────────────────────────────
    K = 12
    raw_results = vs.similarity_search_with_score(query, k=K)
    subsep(f"3. TOP-{K} RETRIEVED CHUNKS BEFORE FILTERING")
    for rank, (doc, distance) in enumerate(raw_results, 1):
        sim = 1.0 - distance
        ctype = doc.metadata.get("chunk_type", "text")
        src = doc.metadata.get("source_file", "?")
        snippet = doc.page_content[:80].replace("\n", " ").strip()
        em = _entity_in_doc(doc, qctx.entities) if qctx.entities else "n/a (no entities)"
        boost = _get_type_boost(ctype, qctx.intent)
        final = sim * boost * (1.3 if (qctx.entities and em is True) else 1.0)
        print(f"   [{rank:2d}] dist={distance:.4f} sim={sim:.4f} boost={boost:.1f} final={final:.4f}"
              f" entity_match={em}")
        print(f"        type={ctype} | src={src[:40]}")
        print(f"        content: {snippet!r}")

    # ── 4. Scored + sorted ──────────────────────────────────────────────────
    scored: list = []
    for doc, distance in raw_results:
        sim = 1.0 - distance
        ctype = doc.metadata.get("chunk_type", "text")
        em = _entity_in_doc(doc, qctx.entities)
        boost = _get_type_boost(ctype, qctx.intent)
        final = sim * boost * (1.3 if em else 1.0)
        scored.append((doc, sim, em, final))
    scored.sort(key=lambda x: x[3], reverse=True)

    # ── 5. Entity grounding gate ─────────────────────────────────────────────
    subsep("4. ENTITY GROUNDING GATE")
    if not qctx.entities:
        print("   SKIP: no named entities extracted — gate not applied")
        entity_gate_passed = True
        grounded = []
    else:
        grounded = [
            (doc, sim)
            for doc, sim, em, _ in scored
            if em
            and doc.metadata.get("chunk_type", "text") not in _IMAGE_CHUNK_TYPES
            and sim >= _GROUNDING_THRESHOLD
        ]
        entity_gate_passed = bool(grounded)
        print(f"   Entities required    : {qctx.entities}")
        print(f"   _GROUNDING_THRESHOLD : {_GROUNDING_THRESHOLD}  (entity presence check)")
        print(f"   EVIDENCE_THRESHOLD   : {EVIDENCE_THRESHOLD}  (context quality check)")
        print(f"   Is lenient intent    : {qctx.intent in _LENIENT_INTENTS} ({qctx.intent.value})")
        print(f"   Grounded text chunks above grounding threshold: {len(grounded)}")
        for doc, sim in grounded[:3]:
            snippet = doc.page_content[:100].replace("\n", " ").strip()
            print(f"     sim={sim:.4f} src={doc.metadata.get('source_file','?')[:35]}")
            print(f"     {snippet!r}")
        if not entity_gate_passed:
            print("   RESULT: GATE FIRED → no reliable evidence (entity not in index)")
        else:
            print("   RESULT: gate passed (entity found in index)")

    # ── 6. Evidence threshold filtering ─────────────────────────────────────
    subsep("5. CHUNKS AFTER EVIDENCE FILTERING (text only, sim >= EVIDENCE_THRESHOLD)")
    context_chunks: list = []
    if not entity_gate_passed:
        print("   (skipped — entity gate fired)")
    else:
        context_chunks = [
            doc
            for doc, sim, _, _ in scored
            if doc.metadata.get("chunk_type", "text") not in _IMAGE_CHUNK_TYPES
            and sim >= EVIDENCE_THRESHOLD
        ][:8]
        # Lenient intent fallback
        if not context_chunks and grounded and qctx.intent in _LENIENT_INTENTS:
            context_chunks = [doc for doc, _ in grounded][:8]
            print(f"   0 chunks above EVIDENCE_THRESHOLD={EVIDENCE_THRESHOLD}")
            print(f"   LENIENT FALLBACK ({qctx.intent.value}): using {len(context_chunks)} "
                  f"entity-matched grounding chunks (sim >= {_GROUNDING_THRESHOLD})")
        else:
            print(f"   {len(context_chunks)} chunks pass EVIDENCE_THRESHOLD={EVIDENCE_THRESHOLD}:")
        for doc in context_chunks:
            sim_val = next((s for d, s, _, _ in scored if d is doc), 0.0)
            snippet = doc.page_content[:100].replace("\n", " ").strip()
            em = _entity_in_doc(doc, qctx.entities)
            print(f"     sim={sim_val:.4f} entity_match={em} type={doc.metadata.get('chunk_type')}"
                  f" src={doc.metadata.get('source_file','?')[:35]}")
            print(f"       {snippet!r}")

    # ── 7. Visual gate ───────────────────────────────────────────────────────
    subsep("6-8. VISUAL GATE (4 checks)")
    print(f"   Check 1 — intent_match: intent={qctx.intent.value} "
          f"in _VISUAL_INTENTS={[i.value for i in _VISUAL_INTENTS]} → "
          f"{'PASS' if qctx.intent in _VISUAL_INTENTS else 'FAIL (no visuals)'}")

    approved_visuals: list[dict] = []
    if entity_gate_passed and qctx.intent in _VISUAL_INTENTS:
        for doc, sim, em, _ in scored:
            ctype = doc.metadata.get("chunk_type", "")
            if ctype not in _IMAGE_CHUNK_TYPES:
                continue
            url = doc.metadata.get("figure_url", "")
            caption = (doc.metadata.get("caption") or "").strip()
            nearby = (doc.metadata.get("nearby_text") or "").strip()
            c2 = em
            c3 = sim >= EVIDENCE_THRESHOLD
            c4 = _has_visual_support(doc)
            c_url = bool(url)
            passed = c2 and c3 and c4 and c_url
            print(f"\n   Image chunk: type={ctype} fig={doc.metadata.get('figure_index','')} "
                  f"src={doc.metadata.get('source_file','?')[:30]}")
            print(f"     Check 2 entity_match={c2}  Check 3 score={sim:.4f}>={EVIDENCE_THRESHOLD}={c3}"
                  f"  Check 4 support={c4}  has_url={c_url}")
            print(f"     caption={caption[:80]!r}  nearby={nearby[:60]!r}")
            print(f"     → {'APPROVED' if passed else 'REJECTED'}")
            if passed:
                approved_visuals.append(_to_visual_dict(doc))
        approved_visuals = _dedupe_visuals(approved_visuals)
    elif not entity_gate_passed:
        print("   (skipped — entity gate fired)")
    else:
        print("   RESULT: intent not in _VISUAL_INTENTS → visual_outputs = []")

    # ── 8. Final answer context ──────────────────────────────────────────────
    subsep("7. FINAL ANSWER CONTEXT (sent to LLM)")
    if not entity_gate_passed or not context_chunks:
        print(f"   {_NO_EVIDENCE_MSG}")
    else:
        context_parts = []
        citations = []
        seen: set[str] = set()
        for doc in context_chunks:
            m = doc.metadata
            title = m.get("title", m.get("source_file", "Unknown"))
            year = m.get("year", "")
            authors = m.get("authors", "")
            ctype = m.get("chunk_type", "")
            citation_key = f"{title} ({year})"
            if citation_key not in seen:
                seen.add(citation_key)
                citations.append(f"{title} — {authors} ({year})" if authors else citation_key)
            prefix = f"[Source: {title}, {year}]"
            context_parts.append(f"{prefix}\n{doc.page_content[:200]}")
        context = "\n\n---\n\n".join(context_parts)
        print(f"   (first 400 chars)\n   {context[:400].replace(chr(10), chr(10)+'   ')}")

    # ── 9. Final visual outputs ──────────────────────────────────────────────
    subsep("8. FINAL VISUAL_OUTPUTS (sent to frontend)")
    if approved_visuals:
        for v in approved_visuals:
            print(f"   url={v['url']}")
            print(f"   figure_index={v['figure_index']}  chunk_type={v['chunk_type']}")
            print(f"   caption={v['caption'][:80]!r}")
    else:
        print("   [] — no visuals returned")

    # ── 10. Sources ──────────────────────────────────────────────────────────
    subsep("10. SOURCES RETURNED")
    if entity_gate_passed and context_chunks:
        src_set: set[str] = set()
        for doc in context_chunks:
            src_set.add(doc.metadata.get("source_file", "?"))
        for s in sorted(src_set):
            print(f"   {s}")
    else:
        print("   [] — no sources (no reliable evidence)")


def main():
    print("Loading vectorstore…")
    vs = load_index()
    print("Vectorstore loaded.\n")

    diagnose(vs, "Hello, do you know about the CLCNet")
    diagnose(vs, "do you know ABRA?")


if __name__ == "__main__":
    main()
