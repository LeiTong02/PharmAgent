"""Tests for smart_retrieve: re-ranking, entity grounding, evidence gate, visual policy."""
from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from rag.retriever import EVIDENCE_THRESHOLD, smart_retrieve

# ---------------------------------------------------------------------------
# Score convention
#   langchain_redis returns cosine DISTANCE (lower = more similar).
#   smart_retrieve converts: similarity = 1 - distance.
#   EVIDENCE_THRESHOLD = 0.75 (similarity space).
#   → passing distance: 1 - 0.75 = 0.25 → any distance ≤ 0.25 passes.
#   → failing distance: any distance > 0.25 (e.g. 0.30) fails.
# ---------------------------------------------------------------------------

_PASS_DIST = 0.15   # similarity = 0.85 → passes threshold
_FAIL_DIST = 0.30   # similarity = 0.70 → fails threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(
    content: str = "CLCNet is a model for cell line classification",
    chunk_type: str = "text",
    source_file: str = "clcnet_paper.pdf",
    title: str = "CLCNet",
    figure_url: str = "",
    figure_index: str = "",
    caption: str = "",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_type": chunk_type,
            "source_file": source_file,
            "title": title,
            "year": "2022",
            "authors": "Smith et al.",
            "figure_index": figure_index,
            "figure_url": figure_url,
            "caption": caption,
            "nearby_text": "",
            "page_number": 1,
            "section": "",
        },
    )


def _vs(results: list[tuple[Document, float]]) -> MagicMock:
    mock = MagicMock()
    mock.similarity_search_with_score.return_value = results
    return mock


# ---------------------------------------------------------------------------
# Entity grounding gate
# ---------------------------------------------------------------------------

def test_ungrounded_entity_returns_no_evidence():
    """Query about ABRA; all retrieved chunks mention only CLCNet → no-evidence."""
    vs = _vs([
        (_doc("CLCNet is a deep learning model"), _PASS_DIST),
        (_doc("The architecture uses convolutional layers"), _PASS_DIST),
    ])
    context, citations, visuals = smart_retrieve(vs, "do you know ABRA", k=8)
    assert "no reliable evidence" in context.lower()
    assert citations == []
    assert visuals == []


def test_grounded_entity_returns_context():
    """Query about CLCNet; chunks mention CLCNet → context returned."""
    vs = _vs([
        (_doc("CLCNet is a deep learning model for cell line classification"), _PASS_DIST),
    ])
    context, citations, visuals = smart_retrieve(vs, "do you know CLCNet", k=8)
    assert "no reliable evidence" not in context.lower()
    assert len(citations) > 0


# ---------------------------------------------------------------------------
# Visual intent gate
# ---------------------------------------------------------------------------

def test_entity_lookup_returns_no_visuals():
    """entity_lookup intent: images suppressed regardless of score."""
    vs = _vs([
        (_doc("CLCNet is a deep learning model"), _PASS_DIST),
        (_doc("", chunk_type="figure_image", figure_url="/figures/clcnet.pdf/figure_1.png", figure_index="1"), _PASS_DIST),
    ])
    _, _, visuals = smart_retrieve(vs, "do you know CLCNet", k=8)
    assert visuals == []


def test_concept_definition_returns_no_visuals():
    """concept_definition intent: images suppressed."""
    vs = _vs([
        (_doc("CLCNet is a neural network architecture"), _PASS_DIST),
        (_doc("", chunk_type="figure_image", figure_url="/figures/clcnet.pdf/figure_2.png", figure_index="2"), _PASS_DIST),
    ])
    _, _, visuals = smart_retrieve(vs, "what is CLCNet", k=8)
    assert visuals == []


def test_framework_intent_returns_figure_crops():
    """framework_or_architecture intent: image chunks returned when entity + threshold pass."""
    vs = _vs([
        (_doc("CLCNet uses a novel architecture framework"), _PASS_DIST),
        (_doc(
            "[Figure 2 image from clcnet.pdf, page 3]",
            chunk_type="figure_image",
            figure_url="/figures/clcnet.pdf/figure_2.png",
            figure_index="2",
            caption="Figure 2. Overall framework of CLCNet.",
        ), _PASS_DIST),
    ])
    _, _, visuals = smart_retrieve(vs, "what is the key framework of CLCNet", k=8)
    assert len(visuals) == 1
    assert visuals[0]["url"] == "/figures/clcnet.pdf/figure_2.png"
    assert visuals[0]["figure_index"] == "2"


def test_figure_specific_intent_returns_visuals():
    """figure_specific intent: image chunks returned."""
    vs = _vs([
        (_doc("Figure 3 shows the training pipeline", chunk_type="figure_caption",
              figure_index="3", figure_url="/figures/clcnet.pdf/figure_3.png"), _PASS_DIST),
        (_doc("", chunk_type="figure_image", figure_url="/figures/clcnet.pdf/figure_3.png",
              figure_index="3", caption="Figure 3."), _PASS_DIST),
    ])
    _, _, visuals = smart_retrieve(vs, "show me Figure 3", k=8)
    assert any(v["url"] == "/figures/clcnet.pdf/figure_3.png" for v in visuals)


# ---------------------------------------------------------------------------
# Evidence threshold gate
# ---------------------------------------------------------------------------

def test_below_threshold_returns_no_evidence():
    """All chunks below EVIDENCE_THRESHOLD → no-evidence message."""
    vs = _vs([
        (_doc("CLCNet architecture overview"), _FAIL_DIST),
        (_doc("Some other content"), _FAIL_DIST),
    ])
    context, citations, visuals = smart_retrieve(vs, "what is the CLCNet framework", k=8)
    assert "no reliable evidence" in context.lower()
    assert visuals == []


def test_empty_results_returns_no_evidence():
    """Vectorstore returns nothing → no-evidence."""
    vs = _vs([])
    context, citations, visuals = smart_retrieve(vs, "CLCNet architecture", k=8)
    assert "no reliable evidence" in context.lower()


# ---------------------------------------------------------------------------
# Visual deduplication
# ---------------------------------------------------------------------------

def test_page_screenshot_suppressed_when_figure_crop_exists():
    """When both figure_image and page_screenshot pass, only figure_image returned."""
    vs = _vs([
        (_doc("CLCNet overall framework description"), _PASS_DIST),
        (_doc("", chunk_type="figure_image",
              figure_url="/figures/clcnet.pdf/figure_2.png", figure_index="2",
              caption="Figure 2. CLCNet framework."), _PASS_DIST),
        (_doc("", chunk_type="page_screenshot",
              figure_url="/figures/clcnet.pdf/page_3.png"), _PASS_DIST),
    ])
    _, _, visuals = smart_retrieve(vs, "what is the key framework of CLCNet", k=8)
    chunk_types = {v["chunk_type"] for v in visuals}
    assert "figure_image" in chunk_types
    assert "page_screenshot" not in chunk_types


def test_page_screenshot_returned_when_no_figure_crop():
    """When no figure_image exists, page_screenshot is returned for visual intents."""
    vs = _vs([
        (_doc("CLCNet overall framework description"), _PASS_DIST),
        (_doc("", chunk_type="page_screenshot",
              figure_url="/figures/clcnet.pdf/page_3.png"), _PASS_DIST),
    ])
    _, _, visuals = smart_retrieve(vs, "what is the key framework of CLCNet", k=8)
    assert any(v["chunk_type"] == "page_screenshot" for v in visuals)


# ---------------------------------------------------------------------------
# qctx parameter — pre-classified context from intent_node
# ---------------------------------------------------------------------------

def test_smart_retrieve_accepts_prebuilt_qctx():
    """When qctx is passed explicitly, it takes precedence over internal parse_query."""
    from rag.query_parser import QueryContext, QueryIntent
    qctx = QueryContext(
        intent=QueryIntent.FRAMEWORK_OR_ARCHITECTURE,
        entities=["CLCNet"],
        figure_refs=[],
        table_refs=[],
        raw_query="CLCNet",
    )
    vs = _vs([
        (_doc("CLCNet framework description"), _PASS_DIST),
        (_doc("", chunk_type="figure_image",
              figure_url="/figures/clcnet.pdf/figure_1.png", figure_index="1",
              caption="Figure 1."), _PASS_DIST),
    ])
    _, _, visuals = smart_retrieve(vs, "CLCNet", k=8, qctx=qctx)
    assert len(visuals) == 1


# ---------------------------------------------------------------------------
# Backward compat: retrieve() is unaffected
# ---------------------------------------------------------------------------

def test_retrieve_still_works():
    """The original retrieve() function is untouched and still callable."""
    from rag.retriever import retrieve
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [
        _doc("Test content about EGFR inhibitors"),
    ]
    context, citations = retrieve(mock_vs, "EGFR inhibitor", k=4)
    assert isinstance(context, str)
    assert isinstance(citations, list)
