"""Agent tools: RAG search, CSV query, GitHub README fetch, paper lookup."""
from __future__ import annotations

import base64
import os
import re
from pathlib import Path

import pandas as pd
import requests
from langchain_core.tools import tool

import json as _json

from rag.query_parser import QueryContext
from rag.retriever import retrieve, smart_retrieve

_CSV_PATH = Path(__file__).parent.parent / "data" / "assay_results.csv"
_df: pd.DataFrame | None = None

# Injected by graph.py after vectorstore is loaded
_vectorstore = None
_wiki_vectorstore = None
_current_query_context: QueryContext | None = None  # set by intent_node before each tool call


def set_vectorstore(vs) -> None:
    global _vectorstore
    _vectorstore = vs


def set_wiki_vectorstore(vs) -> None:
    global _wiki_vectorstore
    _wiki_vectorstore = vs


def set_current_query_context(qctx: QueryContext | None) -> None:
    global _current_query_context
    _current_query_context = qctx


def _get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(_CSV_PATH)
    return _df


# ---------------------------------------------------------------------------
# Tool 1: RAG search over research papers
# ---------------------------------------------------------------------------

@tool
def rag_search(query: str) -> str:
    """Search the internal pharmaceutical research paper library and return relevant excerpts with citations.

    Use this tool for questions about drug mechanisms, clinical trial results, biomarkers,
    protein structures, assay methodology, and any topic covered in the research literature.
    """
    if _vectorstore is None:
        return "Vector store not initialized. Please restart the application."
    context, citations, visuals = smart_retrieve(_vectorstore, query, k=12, qctx=_current_query_context)
    citation_list = "\n".join(f"  - {c}" for c in citations)
    output = f"{context}\n\n**Sources retrieved:**\n{citation_list}"
    if visuals:
        output += f"\n__VISUAL_CHUNKS__:{_json.dumps(visuals)}"
    return output


# ---------------------------------------------------------------------------
# Tool 1b: Wiki search over pre-compiled knowledge pages
# ---------------------------------------------------------------------------

@tool
def wiki_search(query: str) -> str:
    """Search the pre-compiled pharmaceutical wiki knowledge base and return structured summaries.

    Use this tool for questions about drug mechanisms, clinical trial results, biomarkers,
    protein structures, assay methodology, and any topic covered in the research literature.
    Returns pre-organized wiki pages with summaries, key concepts, and findings.
    """
    if _wiki_vectorstore is None:
        return "Wiki store not initialized. Please restart the application."
    context, citations = retrieve(_wiki_vectorstore, query, k=3)
    citation_list = "\n".join(f"  - {c}" for c in citations)
    return f"{context}\n\n**Sources retrieved:**\n{citation_list}"


# ---------------------------------------------------------------------------
# Tool 2: Assay results CSV query
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "january": "01", "jan": "01", "february": "02", "feb": "02",
    "march": "03", "mar": "03", "april": "04", "apr": "04",
    "may": "05", "june": "06", "jun": "06", "july": "07", "jul": "07",
    "august": "08", "aug": "08", "september": "09", "sep": "09", "sept": "09",
    "october": "10", "oct": "10", "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}


def _apply_date_filter(df: pd.DataFrame, q: str) -> pd.DataFrame:
    """Return df filtered by any date constraints found in q."""
    df = df.copy()
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")

    # "after YYYY-MM" / "after Month YYYY" / "since ..."
    m = re.search(r"(?:after|since|from)\s+(\w+)\s+(\d{4})", q)
    if not m:
        m = re.search(r"(?:after|since|from)\s+(\d{4}-\d{2})", q)
    if m:
        raw = m.group(1)
        if raw in _MONTH_MAP:
            cutoff = pd.Timestamp(f"{m.group(2)}-{_MONTH_MAP[raw]}-01")
        else:
            cutoff = pd.Timestamp(raw + "-01") if len(raw) == 7 else pd.Timestamp(raw)
        df = df[df["_date"] >= cutoff]

    # "before YYYY-MM" / "before Month YYYY" / "until ..."
    m = re.search(r"(?:before|until|up to|prior to)\s+(\w+)\s+(\d{4})", q)
    if not m:
        m = re.search(r"(?:before|until|up to|prior to)\s+(\d{4}-\d{2})", q)
    if m:
        raw = m.group(1)
        if raw in _MONTH_MAP:
            cutoff = pd.Timestamp(f"{m.group(2)}-{_MONTH_MAP[raw]}-01")
        else:
            cutoff = pd.Timestamp(raw + "-01") if len(raw) == 7 else pd.Timestamp(raw)
        df = df[df["_date"] < cutoff]

    # "between Month YYYY and Month YYYY"
    m = re.search(r"between\s+(\w+)\s+(\d{4})\s+and\s+(\w+)\s+(\d{4})", q)
    if m:
        mon1 = _MONTH_MAP.get(m.group(1), m.group(1))
        mon2 = _MONTH_MAP.get(m.group(3), m.group(3))
        start = pd.Timestamp(f"{m.group(2)}-{mon1}-01")
        end = pd.Timestamp(f"{m.group(4)}-{mon2}-01") + pd.offsets.MonthEnd(0)
        df = df[(df["_date"] >= start) & (df["_date"] <= end)]

    # "in Month YYYY" / "in Q1/Q2/Q3/Q4 YYYY"
    m = re.search(r"\bin\s+(\w+)\s+(\d{4})\b", q)
    if m:
        token = m.group(1).lower()
        yr = m.group(2)
        if token in _MONTH_MAP:
            mon = _MONTH_MAP[token]
            df = df[df["_date"].dt.strftime("%Y-%m") == f"{yr}-{mon}"]
        elif token in ("q1", "q2", "q3", "q4"):
            q_start = {"q1": "01", "q2": "04", "q3": "07", "q4": "10"}[token]
            q_end_mon = {"q1": "03", "q2": "06", "q3": "09", "q4": "12"}[token]
            start = pd.Timestamp(f"{yr}-{q_start}-01")
            end = pd.Timestamp(f"{yr}-{q_end_mon}-01") + pd.offsets.MonthEnd(0)
            df = df[(df["_date"] >= start) & (df["_date"] <= end)]

    return df.drop(columns=["_date"])


def _apply_researcher_filter(df: pd.DataFrame, q: str) -> pd.DataFrame:
    """Return df filtered by researcher name if one is found in q."""
    known = df["researcher"].dropna().unique()
    for name in known:
        # Match last name or full name (case-insensitive)
        parts = name.lower().replace(".", "").split()
        if any(part in q for part in parts if len(part) > 2):
            return df[df["researcher"].str.lower() == name.lower()]
    return df


@tool
def query_assay_data(question: str) -> str:
    """Query the internal assay results database (assay_results.csv) to look up compound data.

    Use this tool for questions about IC50, EC50, selectivity ratios, cell viability,
    specific compounds (e.g. SR-0472), targets (EGFR, BRAF, CDK4/Cyclin D1),
    researchers, dates, or compound status (lead, active, deprioritized).
    Supports date-range filters ("after February 2023", "in Q1 2023", "between Jan and Mar 2023")
    and researcher filters ("compounds by Chen L.", "Patel's results").
    """
    df = _get_df()
    q = question.lower()

    # Pre-filter by date range and researcher before routing
    df = _apply_date_filter(df, q)
    df = _apply_researcher_filter(df, q)

    # Route to appropriate query based on keywords
    if any(k in q for k in ["ic50", "potency", "most potent", "lowest ic50", "best ic50"]):
        subset = df[df["IC50_nM"].notna()].sort_values("IC50_nM")
        for target_kw, target_val in [("egfr", "EGFR"), ("braf", "BRAF"), ("cdk4", "CDK4"), ("cdk", "CDK4")]:
            if target_kw in q:
                subset = subset[subset["target"].str.contains(target_val, case=False, na=False)]
                break
        result = subset.head(10)[["compound_id", "compound_name", "target", "IC50_nM", "selectivity_ratio", "researcher", "date", "status"]]

    elif any(k in q for k in ["ec50", "cellular"]):
        subset = df[df["EC50_nM"].notna()].sort_values("EC50_nM")
        for target_kw, target_val in [("egfr", "EGFR"), ("braf", "BRAF"), ("cdk4", "CDK4"), ("cdk", "CDK4")]:
            if target_kw in q:
                subset = subset[subset["target"].str.contains(target_val, case=False, na=False)]
                break
        result = subset.head(10)[["compound_id", "compound_name", "target", "EC50_nM", "selectivity_ratio", "researcher", "date", "status"]]

    elif any(k in q for k in ["selectivity", "selective"]):
        subset = df[df["selectivity_ratio"].notna()].sort_values("selectivity_ratio", ascending=False)
        result = subset.head(10)[["compound_id", "compound_name", "target", "IC50_nM", "selectivity_ratio", "researcher", "date", "status"]]

    elif any(k in q for k in ["lead", "leads"]):
        result = df[df["status"] == "lead"][["compound_id", "compound_name", "target", "IC50_nM", "EC50_nM", "selectivity_ratio", "researcher", "date"]]

    elif "deprioritized" in q:
        result = df[df["status"] == "deprioritized"][["compound_id", "compound_name", "target", "IC50_nM", "EC50_nM", "researcher", "date"]]

    elif any(k in q for k in ["egfr"]):
        result = df[df["target"].str.contains("EGFR", case=False, na=False)]

    elif any(k in q for k in ["braf"]):
        result = df[df["target"].str.contains("BRAF", case=False, na=False)]

    elif any(k in q for k in ["cdk4", "cdk"]):
        result = df[df["target"].str.contains("CDK4", case=False, na=False)]

    elif re.search(r"SR-\d+", question, re.IGNORECASE):
        match = re.search(r"SR-\d+", question, re.IGNORECASE)
        cid = match.group(0).upper()
        result = df[df["compound_id"].str.upper() == cid]
        if result.empty:
            return f"No compound found with ID {cid}."

    elif any(k in q for k in ["researcher", "who", "scientist", "by "]):
        result = df[["compound_id", "compound_name", "target", "IC50_nM", "EC50_nM", "researcher", "date", "status"]]

    else:
        result = df.head(15)

    if result.empty:
        return "No matching records found in the assay database."

    return result.to_markdown(index=False)


# ---------------------------------------------------------------------------
# Tool 3: GitHub README fetch
# ---------------------------------------------------------------------------

@tool
def fetch_github_readme(repo_url: str) -> str:
    """Fetch and summarize the README of a public GitHub repository.

    Accepts URLs like: https://github.com/owner/repo
    Use this when the user provides a GitHub link and wants a summary of what the repository does.
    """
    # Extract owner/repo from URL
    match = re.search(r"github\.com/([^/]+)/([^/?\s]+)", repo_url)
    if not match:
        return f"Could not parse a GitHub owner/repo from: {repo_url}"

    owner, repo = match.group(1), match.group(2).rstrip("/")
    api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        resp = requests.get(api_url, headers=headers, timeout=8)
        if resp.status_code == 404:
            return f"Repository {owner}/{repo} not found or has no README."
        if resp.status_code == 403:
            return f"GitHub API rate limit reached. Try again in a few minutes. (Repo: {owner}/{repo})"
        resp.raise_for_status()
        content_b64 = resp.json().get("content", "")
        readme_text = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        # Truncate to avoid overwhelming the context
        if len(readme_text) > 4000:
            readme_text = readme_text[:4000] + "\n\n... [README truncated at 4000 chars]"
        return f"**README for {owner}/{repo}:**\n\n{readme_text}"
    except requests.Timeout:
        return f"GitHub API timed out for {owner}/{repo}. Please try again."
    except Exception as exc:
        return f"Could not fetch README for {owner}/{repo}: {exc}"


# ---------------------------------------------------------------------------
# Tool 4: Academic paper lookup via Semantic Scholar / arXiv fallback
# ---------------------------------------------------------------------------

@tool
def lookup_paper(query: str) -> str:
    """Look up an academic paper by title, keywords, or DOI using Semantic Scholar.

    Returns title, authors, year, abstract, and citation count.
    Falls back to arXiv if Semantic Scholar is unavailable.
    Use this when the user wants metadata or a summary of a specific paper.
    """
    # Try Semantic Scholar first
    try:
        ss_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": 3,
            "fields": "title,authors,year,abstract,citationCount,externalIds",
        }
        resp = requests.get(ss_url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                results = []
                for paper in data[:2]:
                    authors = ", ".join(a["name"] for a in paper.get("authors", [])[:4])
                    abstract = (paper.get("abstract") or "No abstract available.")[:400]
                    doi = paper.get("externalIds", {}).get("DOI", "")
                    results.append(
                        f"**{paper.get('title', 'Unknown')}**\n"
                        f"Authors: {authors}\n"
                        f"Year: {paper.get('year', 'Unknown')}\n"
                        f"Citations: {paper.get('citationCount', 'N/A')}\n"
                        f"DOI: {doi or 'N/A'}\n"
                        f"Abstract: {abstract}..."
                    )
                return "\n\n---\n\n".join(results)
    except Exception:
        pass

    # Fallback: arXiv
    try:
        arxiv_url = "http://export.arxiv.org/api/query"
        params = {"search_query": f"all:{query}", "max_results": 2}
        resp = requests.get(arxiv_url, params=params, timeout=10)
        if resp.status_code == 200 and "<entry>" in resp.text:
            import xml.etree.ElementTree as ET
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", ns)
            results = []
            for entry in entries[:2]:
                title = (entry.findtext("atom:title", namespaces=ns) or "").strip()
                summary = (entry.findtext("atom:summary", namespaces=ns) or "").strip()[:300]
                published = (entry.findtext("atom:published", namespaces=ns) or "")[:4]
                authors = [a.findtext("atom:name", namespaces=ns) for a in entry.findall("atom:author", ns)]
                results.append(
                    f"**{title}** (arXiv, {published})\n"
                    f"Authors: {', '.join(authors[:4])}\n"
                    f"Summary: {summary}..."
                )
            if results:
                return "\n\n---\n\n".join(results)
    except Exception:
        pass

    return (
        f"Could not retrieve results for '{query}' from Semantic Scholar or arXiv. "
        "Please check your network connection or try a different query."
    )
