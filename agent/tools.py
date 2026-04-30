"""Agent tools: RAG search, CSV query, GitHub README fetch, paper lookup."""
from __future__ import annotations

import base64
import os
import re
from pathlib import Path

import pandas as pd
import requests
from langchain_core.tools import tool

from rag.retriever import retrieve

_CSV_PATH = Path(__file__).parent.parent / "data" / "assay_results.csv"
_df: pd.DataFrame | None = None

# Injected by graph.py after vectorstore is loaded
_vectorstore = None


def set_vectorstore(vs) -> None:
    global _vectorstore
    _vectorstore = vs


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
    context, citations = retrieve(_vectorstore, query, k=4)
    citation_list = "\n".join(f"  - {c}" for c in citations)
    return f"{context}\n\n**Sources retrieved:**\n{citation_list}"


# ---------------------------------------------------------------------------
# Tool 2: Assay results CSV query
# ---------------------------------------------------------------------------

@tool
def query_assay_data(question: str) -> str:
    """Query the internal assay results database (assay_results.csv) to look up compound data.

    Use this tool for questions about IC50, EC50, selectivity ratios, cell viability,
    specific compounds (e.g. SR-0472), targets (EGFR, BRAF, CDK4/Cyclin D1),
    researchers, dates, or compound status (lead, active, deprioritized).
    """
    df = _get_df()
    q = question.lower()

    # Route to appropriate query based on keywords
    if any(k in q for k in ["ic50", "potency", "most potent", "lowest ic50", "best ic50"]):
        subset = df[df["IC50_nM"].notna()].sort_values("IC50_nM")
        # Filter by target if mentioned
        for target_kw, target_val in [("egfr", "EGFR"), ("braf", "BRAF"), ("cdk4", "CDK4"), ("cdk", "CDK4")]:
            if target_kw in q:
                subset = subset[subset["target"].str.contains(target_val, case=False, na=False)]
                break
        result = subset.head(10)[["compound_id", "compound_name", "target", "IC50_nM", "selectivity_ratio", "status"]]

    elif any(k in q for k in ["ec50", "cellular"]):
        subset = df[df["EC50_nM"].notna()].sort_values("EC50_nM")
        for target_kw, target_val in [("egfr", "EGFR"), ("braf", "BRAF"), ("cdk4", "CDK4"), ("cdk", "CDK4")]:
            if target_kw in q:
                subset = subset[subset["target"].str.contains(target_val, case=False, na=False)]
                break
        result = subset.head(10)[["compound_id", "compound_name", "target", "EC50_nM", "selectivity_ratio", "status"]]

    elif any(k in q for k in ["selectivity", "selective"]):
        subset = df[df["selectivity_ratio"].notna()].sort_values("selectivity_ratio", ascending=False)
        result = subset.head(10)[["compound_id", "compound_name", "target", "IC50_nM", "selectivity_ratio", "status"]]

    elif any(k in q for k in ["lead", "leads"]):
        result = df[df["status"] == "lead"][["compound_id", "compound_name", "target", "IC50_nM", "EC50_nM", "selectivity_ratio"]]

    elif "deprioritized" in q:
        result = df[df["status"] == "deprioritized"][["compound_id", "compound_name", "target", "IC50_nM", "EC50_nM"]]

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
