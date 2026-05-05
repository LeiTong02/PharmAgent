"""Tests for query intent classification and entity extraction."""
from __future__ import annotations

import pytest

from rag.query_parser import QueryContext, QueryIntent, parse_query, parse_query_from_llm_output


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def test_entity_lookup_intent():
    qctx = parse_query("do you know ABRA")
    assert qctx.intent == QueryIntent.ENTITY_LOOKUP
    assert "ABRA" in qctx.entities


def test_concept_definition_intent_what_is():
    qctx = parse_query("what is CLCNet")
    assert qctx.intent == QueryIntent.CONCEPT_DEFINITION


def test_concept_definition_intent_explain():
    qctx = parse_query("explain the cell line classification method")
    assert qctx.intent == QueryIntent.CONCEPT_DEFINITION


def test_framework_or_architecture_intent():
    qctx = parse_query("what is the key framework of CLCNet")
    assert qctx.intent == QueryIntent.FRAMEWORK_OR_ARCHITECTURE


def test_framework_intent_architecture_keyword():
    qctx = parse_query("describe the overall architecture of the model")
    assert qctx.intent == QueryIntent.FRAMEWORK_OR_ARCHITECTURE


def test_figure_specific_intent():
    qctx = parse_query("show me Figure 2 of the paper")
    assert qctx.intent == QueryIntent.FIGURE_SPECIFIC
    assert "2" in qctx.figure_refs


def test_figure_specific_intent_fig_abbrev():
    qctx = parse_query("what does fig. 3 show")
    assert qctx.intent == QueryIntent.FIGURE_SPECIFIC
    assert "3" in qctx.figure_refs


def test_table_result_intent():
    qctx = parse_query("what are the performance results in Table 1")
    assert qctx.intent == QueryIntent.TABLE_OR_RESULT
    assert "1" in qctx.table_refs


def test_table_result_intent_accuracy_keyword():
    qctx = parse_query("what is the accuracy on the test set")
    assert qctx.intent == QueryIntent.TABLE_OR_RESULT


def test_methodology_maps_to_general_qa():
    qctx = parse_query("how was the training procedure designed")
    assert qctx.intent == QueryIntent.GENERAL_QA


def test_comparison_maps_to_general_qa():
    qctx = parse_query("compare CLCNet vs baseline models")
    assert qctx.intent == QueryIntent.GENERAL_QA


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def test_entity_extraction_capitalized():
    qctx = parse_query("explain the CLCNet architecture pipeline")
    assert "CLCNet" in qctx.entities


def test_entity_extraction_multiple():
    qctx = parse_query("compare CLCNet and ResNet performance")
    assert "CLCNet" in qctx.entities
    assert "ResNet" in qctx.entities


def test_entity_extraction_excludes_stopwords():
    qctx = parse_query("What is the best method")
    assert "What" not in qctx.entities
    assert "The" not in qctx.entities


def test_no_entity_for_all_lowercase():
    qctx = parse_query("what is a neural network")
    assert qctx.entities == []


def test_figure_refs_extracted():
    qctx = parse_query("describe Figure 2a and Fig. 3")
    assert "2a" in qctx.figure_refs
    assert "3" in qctx.figure_refs


def test_table_refs_extracted():
    qctx = parse_query("the results in Table S1 and Table 2")
    assert "S1" in qctx.table_refs
    assert "2" in qctx.table_refs


# ---------------------------------------------------------------------------
# Priority order: FIGURE_SPECIFIC beats FRAMEWORK
# ---------------------------------------------------------------------------

def test_figure_intent_beats_framework():
    qctx = parse_query("show me the framework figure 2")
    assert qctx.intent == QueryIntent.FIGURE_SPECIFIC


def test_table_intent_beats_concept():
    qctx = parse_query("what is shown in Table 1")
    assert qctx.intent == QueryIntent.TABLE_OR_RESULT


# ---------------------------------------------------------------------------
# parse_query_from_llm_output — validates and supplements LLM JSON output
# ---------------------------------------------------------------------------

def test_llm_output_valid_intent():
    qctx = parse_query_from_llm_output(
        {"intent": "framework_or_architecture", "entities": ["CLCNet"], "figure_refs": [], "table_refs": []},
        "what is the key framework of CLCNet",
    )
    assert qctx.intent == QueryIntent.FRAMEWORK_OR_ARCHITECTURE
    assert "CLCNet" in qctx.entities


def test_llm_output_invalid_intent_falls_back_to_general_qa():
    qctx = parse_query_from_llm_output(
        {"intent": "unknown_intent", "entities": [], "figure_refs": [], "table_refs": []},
        "some query",
    )
    assert qctx.intent == QueryIntent.GENERAL_QA


def test_llm_output_supplements_entities_from_regex():
    """If LLM misses a capitalized entity, regex supplement catches it."""
    qctx = parse_query_from_llm_output(
        {"intent": "concept_definition", "entities": [], "figure_refs": [], "table_refs": []},
        "what is CLCNet",
    )
    assert "CLCNet" in qctx.entities


def test_general_qa_is_default():
    qctx = parse_query("some random query without clear patterns")
    assert qctx.intent == QueryIntent.GENERAL_QA
