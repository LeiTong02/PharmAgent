from __future__ import annotations

from langchain_core.documents import Document

from rag.query_parser import QueryIntent, QueryContext, parse_query

# ---------------------------------------------------------------------------
# Evidence threshold (cosine SIMILARITY space, 0–1, higher = more similar)
#
# langchain_redis returns cosine DISTANCE (lower = more similar).
# We convert: similarity = 1.0 − distance before applying any threshold.
#
# Calibrated from live Redis data:
#   Good CLCNet match:  distance 0.156–0.202 → similarity 0.798–0.844
#   Unrelated mock doc: distance 0.264–0.271 → similarity 0.729–0.736
#   Threshold at 0.75 separates them cleanly.
# ---------------------------------------------------------------------------

EVIDENCE_THRESHOLD: float = 0.75  # minimum cosine similarity for high-quality context chunks

# Lower threshold used ONLY to check whether an entity EXISTS in the index.
# Conversational queries ("do you know X?") produce lower similarity scores than
# direct technical queries, so entity presence cannot be gated at EVIDENCE_THRESHOLD.
# Calibrated: "do you know ABRA?" → best ABRA chunk sim ≈ 0.66; unrelated content
# that lacks entity_match is filtered by entity_match check, not by this threshold.
_GROUNDING_THRESHOLD: float = 0.60  # minimum sim to count a chunk as entity-grounding evidence

# For entity_lookup and concept_definition, if no chunk clears EVIDENCE_THRESHOLD but
# entity-matched chunks clear _GROUNDING_THRESHOLD, fall back to those chunks.
# This prevents false no-evidence for valid entities queried conversationally.
_LENIENT_INTENTS: frozenset[QueryIntent] = frozenset({
    QueryIntent.ENTITY_LOOKUP,
    QueryIntent.CONCEPT_DEFINITION,
})

_VISUAL_INTENTS: frozenset[QueryIntent] = frozenset({
    QueryIntent.FRAMEWORK_OR_ARCHITECTURE,
    QueryIntent.FIGURE_SPECIFIC,
    QueryIntent.TABLE_OR_RESULT,
})

# Type-boost multipliers applied to similarity scores.
# Values > 1 promote a chunk type for this intent; < 1 demote it.
_TYPE_BOOST: dict[str, dict[str, float]] = {
    QueryIntent.ENTITY_LOOKUP: {
        "text": 1.0, "table": 1.0, "figure_caption": 0.8, "table_caption": 0.9,
        "figure_image": 0.1, "page_screenshot": 0.05,
    },
    QueryIntent.CONCEPT_DEFINITION: {
        "text": 1.0, "table": 0.9, "figure_caption": 0.9, "table_caption": 0.9,
        "figure_image": 0.1, "page_screenshot": 0.05,
    },
    QueryIntent.FRAMEWORK_OR_ARCHITECTURE: {
        "text": 1.0, "table": 0.7, "figure_caption": 1.2, "table_caption": 0.7,
        "figure_image": 1.3, "page_screenshot": 0.8,
    },
    QueryIntent.FIGURE_SPECIFIC: {
        "text": 0.8, "table": 0.5, "figure_caption": 1.3, "table_caption": 0.5,
        "figure_image": 1.4, "page_screenshot": 0.7,
    },
    QueryIntent.TABLE_OR_RESULT: {
        "text": 1.0, "table": 1.4, "figure_caption": 0.8, "table_caption": 1.3,
        "figure_image": 0.5, "page_screenshot": 0.5,
    },
    QueryIntent.GENERAL_QA: {
        "text": 1.0, "table": 1.0, "figure_caption": 1.0, "table_caption": 1.0,
        "figure_image": 0.3, "page_screenshot": 0.3,
    },
}
_DEFAULT_TYPE_BOOST = 1.0

_IMAGE_CHUNK_TYPES: frozenset[str] = frozenset({"figure_image", "page_screenshot"})

_NO_EVIDENCE_MSG = (
    "No reliable evidence found in the indexed corpus for this query. "
    "The retrieved content does not sufficiently match the query topic. "
    "Consider re-uploading relevant documents or rephrasing the query."
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _entity_in_doc(doc: Document, entities: list[str]) -> bool:
    """True if any entity appears (case-insensitive) in the document text or metadata."""
    if not entities:
        return True  # no specific entity constraint — all docs are potentially relevant
    haystack = " ".join([
        doc.page_content,
        doc.metadata.get("source_file", ""),
        doc.metadata.get("title", ""),
        doc.metadata.get("caption", ""),
        doc.metadata.get("nearby_text", ""),
        doc.metadata.get("section", ""),
    ]).lower()
    return any(e.lower() in haystack for e in entities)


def _get_type_boost(chunk_type: str, intent: QueryIntent) -> float:
    intent_map = _TYPE_BOOST.get(intent, {})
    return intent_map.get(chunk_type, _DEFAULT_TYPE_BOOST)


def _to_visual_dict(doc: Document) -> dict:
    m = doc.metadata
    figure_url = m.get("figure_url", "")
    source_file = m.get("source_file", "")
    if not source_file and "/figures/" in figure_url:
        source_file = figure_url.split("/figures/", 1)[1].rsplit("/", 1)[0]
    return {
        "url": figure_url,
        "filename": source_file,
        "figure_index": m.get("figure_index", ""),
        "page_number": m.get("page_number", 0),
        "caption": m.get("caption", ""),
        "chunk_type": m.get("chunk_type", ""),
    }


def _has_visual_support(doc: Document) -> bool:
    """Check 4 — support_check: image must have non-empty caption or nearby_text.

    page_screenshot chunks don't carry captions; they pass by default because
    entity_match + evidence_score already provide sufficient grounding.
    figure_image chunks without any caption or nearby_text are anonymous images
    with no textual evidence linking them to the query topic.
    """
    if doc.metadata.get("chunk_type") == "page_screenshot":
        return True
    caption = (doc.metadata.get("caption") or "").strip()
    nearby = (doc.metadata.get("nearby_text") or "").strip()
    return bool(caption or nearby)


def _dedupe_visuals(visuals: list[dict]) -> list[dict]:
    """Prefer figure_image crops over page_screenshot. Suppress page shots when crops exist."""
    figure_crops = [v for v in visuals if v.get("chunk_type") == "figure_image" and v.get("figure_index")]
    page_shots = [v for v in visuals if v.get("chunk_type") == "page_screenshot"]

    if figure_crops:
        seen: set[tuple] = set()
        result: list[dict] = []
        for v in figure_crops:
            key = (v.get("filename", ""), v.get("figure_index", ""))
            if key not in seen:
                seen.add(key)
                result.append(v)
        return result
    else:
        seen_urls: set[str] = set()
        result = []
        for v in page_shots:
            url = v.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                result.append(v)
        return result


# ---------------------------------------------------------------------------
# Public: smart_retrieve (policy-gated)
# ---------------------------------------------------------------------------

def smart_retrieve(
    vectorstore,
    query: str,
    k: int = 12,
    qctx: QueryContext | None = None,
) -> tuple[str, list[str], list[dict]]:
    """Policy-gated retrieval with intent, entity grounding, evidence threshold, visual gate.

    Parameters
    ----------
    vectorstore : RedisVectorStore
        The vector index to search.
    query : str
        The user query string.
    k : int
        Number of candidates to retrieve from the vector index.
    qctx : QueryContext | None
        Pre-classified query context from the graph's intent_node.
        When None (e.g. tests, wiki mode), falls back to regex-based parse_query().

    Returns
    -------
    (context_for_llm, citations, approved_visuals)

    approved_visuals is [] unless:
      - intent is FRAMEWORK_OR_ARCHITECTURE, FIGURE_SPECIFIC, or TABLE_OR_RESULT; AND
      - at least one image chunk is entity-matched and above EVIDENCE_THRESHOLD.
    """
    if qctx is None:
        qctx = parse_query(query)

    results_with_scores: list[tuple[Document, float]] = vectorstore.similarity_search_with_score(query, k=k)
    if not results_with_scores:
        return _NO_EVIDENCE_MSG, [], []

    # Convert cosine distance → similarity; apply type-boost and entity-boost.
    # All subsequent logic operates in similarity space (higher = better).
    scored: list[tuple[Document, float, bool, float]] = []
    for doc, distance in results_with_scores:
        similarity = 1.0 - distance          # cosine distance → cosine similarity
        ctype = doc.metadata.get("chunk_type", "text")
        entity_match = _entity_in_doc(doc, qctx.entities)
        type_boost = _get_type_boost(ctype, qctx.intent)
        final_score = similarity * type_boost * (1.3 if entity_match else 1.0)
        scored.append((doc, similarity, entity_match, final_score))

    scored.sort(key=lambda x: x[3], reverse=True)  # descending by boosted score

    # Entity grounding gate: if the query names specific entities, at least one text
    # chunk must (a) mention the entity AND (b) have similarity ≥ _GROUNDING_THRESHOLD.
    # Uses the lower _GROUNDING_THRESHOLD (not EVIDENCE_THRESHOLD) because conversational
    # queries ("do you know X?") produce lower similarity scores than direct technical
    # queries, even when the entity is well-covered in the indexed documents.
    grounded_text: list[tuple] = []
    if qctx.entities:
        grounded_text = [
            (doc, sim)
            for doc, sim, entity_match, _ in scored
            if entity_match
            and doc.metadata.get("chunk_type", "text") not in _IMAGE_CHUNK_TYPES
            and sim >= _GROUNDING_THRESHOLD
        ]
        if not grounded_text:
            return _NO_EVIDENCE_MSG, [], []

    # Collect high-quality text context chunks (similarity ≥ EVIDENCE_THRESHOLD).
    context_chunks = [
        doc
        for doc, sim, _, _ in scored
        if doc.metadata.get("chunk_type", "text") not in _IMAGE_CHUNK_TYPES
        and sim >= EVIDENCE_THRESHOLD
    ][:8]

    # Fallback for entity_lookup / concept_definition: if no chunk clears
    # EVIDENCE_THRESHOLD but entity-matched chunks cleared _GROUNDING_THRESHOLD,
    # use those entity-matched chunks so we can answer "do you know X?" correctly.
    if not context_chunks and grounded_text and qctx.intent in _LENIENT_INTENTS:
        context_chunks = [doc for doc, _ in grounded_text][:8]

    if not context_chunks:
        return _NO_EVIDENCE_MSG, [], []

    context_parts: list[str] = []
    citations: list[str] = []
    seen: set[str] = set()
    for doc in context_chunks:
        m = doc.metadata
        title = m.get("title", m.get("source_file", "Unknown"))
        year = m.get("year", "")
        authors = m.get("authors", "")
        ctype = m.get("chunk_type", "")
        figure_url = m.get("figure_url", "")
        fig_index = m.get("figure_index", "")

        citation_key = f"{title} ({year})"
        if citation_key not in seen:
            seen.add(citation_key)
            citations.append(f"{title} — {authors} ({year})" if authors else citation_key)

        prefix = f"[Source: {title}, {year}]"
        if ctype in ("figure_caption", "table_caption") and fig_index:
            prefix += f"\n[Figure {fig_index} caption]"
            if figure_url:
                prefix += f"\n[Figure {fig_index} image URL: {figure_url}]"

        context_parts.append(f"{prefix}\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    # Visual output gate — four explicit checks must all pass.
    # answer_context (text above) and visual_outputs are kept separate:
    # image chunks never appear in context_chunks; visuals never go to the LLM directly.
    approved_visuals: list[dict] = []
    if qctx.intent in _VISUAL_INTENTS:
        for doc, sim, entity_match, _ in scored:
            ctype = doc.metadata.get("chunk_type", "")
            if ctype not in _IMAGE_CHUNK_TYPES:
                continue
            # Check 1 — intent_match: already enforced by the outer `if qctx.intent` guard.
            # Check 2 — entity_match: image caption/nearby_text/metadata must mention the entity.
            if not entity_match:
                continue
            # Check 3 — evidence_score: raw cosine similarity must clear the threshold.
            if sim < EVIDENCE_THRESHOLD:
                continue
            # Check 4 — support_check: image must have caption or nearby_text as textual support.
            if not _has_visual_support(doc):
                continue
            if not doc.metadata.get("figure_url", ""):
                continue
            approved_visuals.append(_to_visual_dict(doc))
        approved_visuals = _dedupe_visuals(approved_visuals)

    return context, citations, approved_visuals


# ---------------------------------------------------------------------------
# Public: retrieve (unchanged — used by wiki_search and tests)
# ---------------------------------------------------------------------------

def retrieve(vectorstore, query: str, k: int = 4) -> tuple[str, list[str]]:
    """Return (context_text, citations) for the top-k chunks matching query.

    When retrieved chunks include figure_image, page_screenshot, or figure_caption chunks,
    their figure_url is embedded in the context prefix so the LLM can cite the image URL
    in its response and the frontend can render it.
    """
    results = vectorstore.similarity_search(query, k=k)

    context_parts: list[str] = []
    citations: list[str] = []
    seen: set[str] = set()

    for doc in results:
        m = doc.metadata
        title = m.get("title", m.get("source_file", "Unknown"))
        year = m.get("year", "")
        authors = m.get("authors", "")
        ctype = m.get("chunk_type", "")
        figure_url = m.get("figure_url", "")
        fig_index = m.get("figure_index", "")

        citation_key = f"{title} ({year})"
        if citation_key not in seen:
            seen.add(citation_key)
            citations.append(
                f"{title} — {authors} ({year})" if authors else citation_key
            )

        prefix = f"[Source: {title}, {year}]"

        if ctype in ("figure_image", "page_screenshot") and figure_url:
            prefix += f"\n[Image URL: {figure_url}]"
        if ctype in ("figure_caption", "table_caption") and fig_index:
            prefix += f"\n[Figure {fig_index} caption]"
            if figure_url:
                prefix += f"\n[Figure {fig_index} image URL: {figure_url}]"

        context_parts.append(f"{prefix}\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)
    return context, citations
