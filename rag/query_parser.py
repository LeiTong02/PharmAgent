"""Query intent classification and entity extraction for policy-gated retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryIntent(str, Enum):
    ENTITY_LOOKUP = "entity_lookup"
    CONCEPT_DEFINITION = "concept_definition"
    FRAMEWORK_OR_ARCHITECTURE = "framework_or_architecture"
    FIGURE_SPECIFIC = "figure_specific"
    TABLE_OR_RESULT = "table_or_result"
    GENERAL_QA = "general_qa"


@dataclass
class QueryContext:
    intent: QueryIntent
    entities: list[str]
    figure_refs: list[str]   # e.g. ["2", "3a"] from "Figure 2", "Fig 3a"
    table_refs: list[str]    # e.g. ["1", "2"] from "Table 1"
    raw_query: str


# ---------------------------------------------------------------------------
# Regex fallback patterns — used when LLM classification is unavailable
# ---------------------------------------------------------------------------

_FIGURE_PATTERNS = [
    r"\bfig(?:ure)?\.?\s+\w",
    r"\bshow\s+figure\b",
    r"\bimage\s+of\b",
    r"\bplot\s+of\b",
]
_TABLE_PATTERNS = [
    r"\btable\s+\w",
    r"\bresults?\s+table\b",
    r"\bperformance\s+table\b",
    r"\baccuracy\b",
    r"\bbenchmark\b",
]
_FRAMEWORK_PATTERNS = [
    r"\bframework\b",
    r"\barchitecture\b",
    r"\bpipeline\b",
    r"\boverview\b",
    r"\bkey\s+diagram\b",
    r"\bmain\s+model\b",
    r"\bmodel\s+structure\b",
    r"\boverall\s+structure\b",
    r"\bsystem\s+design\b",
]
_CONCEPT_PATTERNS = [
    r"^what\s+(?:is|are)\b",
    r"^define\b",
    r"^explain\b",
    r"^describe\b",
    r"^how\s+does\b",
    r"^what\s+does\b",
]
_ENTITY_LOOKUP_PATTERNS = [
    r"\bdo\s+you\s+know\b",
    r"\btell\s+me\s+about\b",
    r"\bwhat\s+do\s+you\s+know\s+about\b",
    r"\bhave\s+you\s+(?:heard|seen|indexed)\b",
    r"\bis\s+there\s+(?:any|a)\b",
]

_STOPWORDS: frozenset[str] = frozenset({
    "what", "which", "who", "when", "where", "why", "how", "can", "does",
    "did", "are", "is", "was", "were", "has", "have", "had", "the", "its",
    "and", "for", "not", "that", "this", "from", "with", "you", "your",
    "tell", "show", "give", "find", "list", "explain", "define", "describe",
    "key", "main", "primary", "top", "best", "most", "any", "all", "more",
    "some", "about", "know", "paper", "document", "source", "literature",
    "study", "research", "work", "using", "use", "used", "based", "than",
    "into", "such", "also", "both", "there",
    # Greetings and social openers that start sentences capitalized but are not entities
    "hello", "hi", "hey", "dear", "thanks", "thank", "please", "sorry",
    "excuse", "pardon", "greetings", "okay", "sure", "yes", "nope",
})

_VALID_INTENTS: frozenset[str] = frozenset(v.value for v in QueryIntent)


def _match_any(q_lower: str, patterns: list[str]) -> bool:
    return any(re.search(p, q_lower) for p in patterns)


def _classify_intent_regex(q_lower: str) -> QueryIntent:
    """Regex-based fallback intent classifier."""
    if _match_any(q_lower, _FIGURE_PATTERNS):
        return QueryIntent.FIGURE_SPECIFIC
    if _match_any(q_lower, _TABLE_PATTERNS):
        return QueryIntent.TABLE_OR_RESULT
    if _match_any(q_lower, _FRAMEWORK_PATTERNS):
        return QueryIntent.FRAMEWORK_OR_ARCHITECTURE
    # CONCEPT_DEFINITION before entity checks so "explain X method" → definition
    if _match_any(q_lower, _CONCEPT_PATTERNS):
        return QueryIntent.CONCEPT_DEFINITION
    if _match_any(q_lower, _ENTITY_LOOKUP_PATTERNS):
        return QueryIntent.ENTITY_LOOKUP
    return QueryIntent.GENERAL_QA


def _extract_entities(query: str) -> list[str]:
    """Return capitalized tokens (≥3 chars) that are not stopwords."""
    tokens = re.findall(r"\b([A-Z][A-Za-z0-9\-]*)\b", query)
    seen: set[str] = set()
    result: list[str] = []
    for tok in tokens:
        if tok.lower() in _STOPWORDS:
            continue
        if len(tok) < 3:
            continue
        if tok not in seen:
            seen.add(tok)
            result.append(tok)
    return result


def _extract_figure_refs(query: str) -> list[str]:
    return re.findall(r"(?:fig(?:ure)?\.?\s*)(\w+)", query, re.IGNORECASE)


def _extract_table_refs(query: str) -> list[str]:
    return re.findall(r"table\s+(\w+)", query, re.IGNORECASE)


def parse_query(query: str) -> QueryContext:
    """Parse a query string into a QueryContext using regex fallback."""
    q_lower = query.lower().strip()
    return QueryContext(
        intent=_classify_intent_regex(q_lower),
        entities=_extract_entities(query),
        figure_refs=_extract_figure_refs(query),
        table_refs=_extract_table_refs(query),
        raw_query=query,
    )


def parse_query_from_llm_output(data: dict, raw_query: str) -> QueryContext:
    """Build a QueryContext from a structured LLM output dict.

    Validates intent against the enum; falls back to GENERAL_QA for unknown values.
    Called by the graph's intent_node after parsing the LLM JSON response.
    """
    intent_str = data.get("intent", "general_qa")
    if intent_str not in _VALID_INTENTS:
        intent_str = "general_qa"
    entities = [str(e) for e in data.get("entities", [])]
    figure_refs = [str(r) for r in data.get("figure_refs", [])]
    table_refs = [str(r) for r in data.get("table_refs", [])]
    # Supplement entity extraction with regex if LLM missed obvious proper nouns
    regex_entities = _extract_entities(raw_query)
    for e in regex_entities:
        if e not in entities:
            entities.append(e)
    return QueryContext(
        intent=QueryIntent(intent_str),
        entities=entities,
        figure_refs=figure_refs or _extract_figure_refs(raw_query),
        table_refs=table_refs or _extract_table_refs(raw_query),
        raw_query=raw_query,
    )
