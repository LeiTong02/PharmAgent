#!/usr/bin/env python3
"""MCP server exposing PharmaRA's assay query tool via the Model Context Protocol.

Run with:
    python mcp_server.py

Then add to Cursor / Claude Code MCP settings:
    {
      "mcpServers": {
        "pharma_ra": {
          "command": "python",
          "args": ["/path/to/Pharam_RA/mcp_server.py"]
        }
      }
    }
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so agent.tools can be imported
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import fastmcp
from fastmcp import FastMCP

mcp = FastMCP(
    name="PharmaRA",
    instructions=(
        "PharmaRA exposes internal pharmaceutical assay data. "
        "Use query_assay_data to look up IC50, EC50, selectivity ratios, "
        "and compound status from the internal assay database."
    ),
)


@mcp.tool()
def query_assay_data(question: str) -> str:
    """Query the PharmaRA internal assay results database.

    Supports natural-language questions about:
    - IC50 / EC50 / selectivity for specific targets (EGFR, BRAF, CDK4)
    - Compound status (lead, active, deprioritized)
    - Specific compound IDs (e.g. SR-0472)
    - Date-range filters ("after February 2023", "in Q1 2023")
    - Researcher filters ("compounds by Chen L.")
    Returns a markdown table of matching records.
    """
    from agent.tools import query_assay_data as _tool
    return _tool.invoke({"question": question})


@mcp.tool()
def lookup_paper(query: str) -> str:
    """Look up a pharmaceutical or biomedical paper via Semantic Scholar / arXiv.

    Returns title, authors, year, abstract snippet, and citation count.
    """
    from agent.tools import lookup_paper as _tool
    return _tool.invoke({"query": query})


@mcp.tool()
def fetch_github_readme(repo_url: str) -> str:
    """Fetch and return the README of a public GitHub repository.

    Accepts URLs like https://github.com/owner/repo
    """
    from agent.tools import fetch_github_readme as _tool
    return _tool.invoke({"repo_url": repo_url})


if __name__ == "__main__":
    mcp.run()
