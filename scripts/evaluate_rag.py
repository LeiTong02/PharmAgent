#!/usr/bin/env python3
"""Evaluate Classic RAG vs Wiki RAG recall quality on predefined Q&A test cases.

Runs both retrieval modes against a fixed set of pharmaceutical queries and
reports keyword-recall scores. No external LLM call needed — scoring is purely
based on whether expected domain keywords appear in the retrieved context.

Usage:
    python scripts/evaluate_rag.py           # evaluate both modes
    python scripts/evaluate_rag.py classic   # classic RAG only
    python scripts/evaluate_rag.py wiki      # wiki RAG only
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Test cases: (query, expected_keywords, description)
# ---------------------------------------------------------------------------

class TestCase(NamedTuple):
    query: str
    keywords: list[str]
    description: str


TEST_CASES: list[TestCase] = [
    TestCase(
        query="What is the mechanism of EGFR T790M resistance to erlotinib?",
        keywords=["T790M", "resistance", "EGFR", "mutation"],
        description="EGFR resistance mechanism",
    ),
    TestCase(
        query="What clinical outcomes were observed for SAN-4891 in COPD?",
        keywords=["SAN-4891", "COPD", "clinical", "Phase"],
        description="SAN-4891 COPD clinical data",
    ),
    TestCase(
        query="Explain PROTAC mechanism and BRD4 degradation",
        keywords=["PROTAC", "degradation", "BRD4", "E3"],
        description="PROTAC/BRD4 degrader mechanism",
    ),
    TestCase(
        query="How does PD-1 LAG-3 dual blockade improve outcomes in colorectal cancer?",
        keywords=["LAG-3", "PD-1", "MSS", "colorectal"],
        description="PD-1/LAG-3 combination immunotherapy",
    ),
    TestCase(
        query="What are the key biomarkers for EGFR targeted therapy response?",
        keywords=["biomarker", "EGFR", "mutation", "response"],
        description="EGFR biomarker prediction",
    ),
    TestCase(
        query="Describe BRAF V600E mutation and vemurafenib resistance",
        keywords=["BRAF", "V600E", "vemurafenib", "resistance"],
        description="BRAF inhibitor resistance",
    ),
    TestCase(
        query="What is CDK4 cyclin D1 complex inhibition mechanism?",
        keywords=["CDK4", "cyclin", "inhibition", "cell cycle"],
        description="CDK4/CyclinD1 biology",
    ),
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def recall_at_k(result_text: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in result (case-insensitive)."""
    if not keywords:
        return 0.0
    found = sum(1 for kw in keywords if kw.lower() in result_text.lower())
    return found / len(keywords)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_mode(mode: str) -> list[dict]:
    from rag.vectorstore import load_index, load_wiki_index
    from agent.tools import set_vectorstore, set_wiki_vectorstore, rag_search, wiki_search

    print(f"\nLoading {mode} index...")
    vs = load_index()
    set_vectorstore(vs)

    if mode == "wiki":
        wiki_vs = load_wiki_index()
        set_wiki_vectorstore(wiki_vs)
        search_fn = wiki_search
    else:
        search_fn = rag_search

    results = []
    for tc in TEST_CASES:
        t0 = time.time()
        try:
            result = search_fn.invoke({"query": tc.query})
            score = recall_at_k(result, tc.keywords)
            latency = time.time() - t0
            results.append({
                "description": tc.description,
                "query": tc.query[:60] + "...",
                "score": score,
                "latency_s": round(latency, 2),
                "result_chars": len(result),
                "error": None,
            })
        except Exception as exc:
            results.append({
                "description": tc.description,
                "query": tc.query[:60] + "...",
                "score": 0.0,
                "latency_s": round(time.time() - t0, 2),
                "result_chars": 0,
                "error": str(exc),
            })

    return results


def print_report(mode: str, results: list[dict]) -> None:
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    avg_latency = sum(r["latency_s"] for r in results) / len(results) if results else 0.0

    print(f"\n{'=' * 65}")
    print(f"  Mode: {mode.upper():10s}  |  Avg recall: {avg_score:.1%}  |  Avg latency: {avg_latency:.2f}s")
    print(f"{'=' * 65}")
    print(f"  {'Description':<35s}  {'Recall':>7s}  {'Latency':>8s}")
    print(f"  {'-' * 35}  {'-' * 7}  {'-' * 8}")
    for r in results:
        flag = "  ✓" if r["score"] >= 0.75 else ("  ⚠" if r["score"] >= 0.5 else "  ✗")
        err = f" [ERROR: {r['error'][:30]}]" if r["error"] else ""
        print(f"  {r['description']:<35s}  {r['score']:>6.0%}  {r['latency_s']:>7.2f}s{flag}{err}")
    print(f"{'=' * 65}")
    print(f"  TOTAL  avg recall={avg_score:.1%}  ({sum(r['score']>=0.75 for r in results)}/{len(results)} passed at ≥75%)\n")


def compare_modes(classic: list[dict], wiki: list[dict]) -> None:
    print("\n" + "=" * 65)
    print("  COMPARISON: Classic RAG vs Wiki RAG")
    print("=" * 65)
    print(f"  {'Description':<35s}  {'Classic':>8s}  {'Wiki':>8s}  {'Winner':>8s}")
    print(f"  {'-' * 35}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    classic_wins = wiki_wins = ties = 0
    for c, w in zip(classic, wiki):
        diff = w["score"] - c["score"]
        if diff > 0.05:
            winner = "Wiki +"
            wiki_wins += 1
        elif diff < -0.05:
            winner = "Classic+"
            classic_wins += 1
        else:
            winner = "tie"
            ties += 1
        print(f"  {c['description']:<35s}  {c['score']:>7.0%}  {w['score']:>7.0%}  {winner:>8s}")
    print("=" * 65)
    print(f"  Classic wins: {classic_wins}  Wiki wins: {wiki_wins}  Ties: {ties}")
    avg_c = sum(r["score"] for r in classic) / len(classic)
    avg_w = sum(r["score"] for r in wiki) / len(wiki)
    best = "Wiki" if avg_w > avg_c else "Classic" if avg_c > avg_w else "Tie"
    print(f"  Overall: Classic {avg_c:.1%} vs Wiki {avg_w:.1%} → {best} wins overall\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    modes_arg = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    if modes_arg not in ("classic", "wiki", "both"):
        print(f"Usage: python evaluate_rag.py [classic|wiki|both]")
        sys.exit(1)

    classic_results = wiki_results = None

    if modes_arg in ("classic", "both"):
        classic_results = evaluate_mode("classic")
        print_report("classic", classic_results)

    if modes_arg in ("wiki", "both"):
        wiki_results = evaluate_mode("wiki")
        print_report("wiki", wiki_results)

    if classic_results and wiki_results:
        compare_modes(classic_results, wiki_results)


if __name__ == "__main__":
    main()
