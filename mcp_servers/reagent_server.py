"""Reagent procurement MCP server — exposes catalog search, stock check, and cost estimation.

Run standalone:
    python mcp_servers/reagent_server.py

Add to Claude Code / Cursor MCP settings:
    {
      "mcpServers": {
        "reagent_catalog": {
          "command": "python",
          "args": ["/path/to/Pharam_RA/mcp_servers/reagent_server.py"]
        }
      }
    }

The same business logic is also exposed as a LangChain @tool (search_reagents) in
agent/tools.py so the internal LangGraph agent can call it without a subprocess.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastmcp import FastMCP
from mcp_servers.reagent_catalog import search_reagents_impl, check_stock_impl, estimate_cost_impl

mcp = FastMCP(
    name="ReagentCatalog",
    instructions=(
        "ReagentCatalog provides pharmaceutical reagent procurement services. "
        "Use search_reagents to find compounds by name, ID, or target. "
        "Use check_stock to verify availability of a specific catalog item. "
        "Use estimate_cost to get a total price quote including shipping."
    ),
)


@mcp.tool()
def search_reagents(query: str, target: str = "", min_purity_pct: int = 95) -> str:
    """Search the reagent catalog by compound name, internal ID (e.g. SR-0472), or target.

    Args:
        query: Compound name, synonym, or internal compound ID (SR-XXXX).
        target: Optional target filter, e.g. 'EGFR', 'BRAF', 'CDK4'.
        min_purity_pct: Minimum purity percentage (default 95). Use 98 for high-quality biochemical assays.

    Returns a markdown table with catalog ID, supplier, purity, price/mg, stock status, and lead time.
    """
    return search_reagents_impl(query, target, min_purity_pct)


@mcp.tool()
def check_stock(catalog_id: str) -> str:
    """Check availability and lead time for a specific reagent catalog entry.

    Args:
        catalog_id: Supplier catalog ID, e.g. 'HY-15772', 'SML1277', 'PD0332991'.

    Returns stock status, minimum order quantity, lead time, and supplier contact.
    """
    return check_stock_impl(catalog_id)


@mcp.tool()
def estimate_cost(catalog_id: str, quantity_mg: float) -> str:
    """Estimate total procurement cost for a reagent, including shipping.

    Args:
        catalog_id: Supplier catalog ID, e.g. 'HY-15772'.
        quantity_mg: Desired quantity in milligrams (must be ≥ minimum order quantity).

    Returns itemised cost breakdown: unit price, subtotal, shipping, total, and estimated delivery date.
    """
    return estimate_cost_impl(catalog_id, quantity_mg)


if __name__ == "__main__":
    mcp.run()
