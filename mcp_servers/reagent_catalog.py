"""Core reagent catalog logic — shared by both the MCP server and the LangChain tool.

Keeping the implementations here (not in tools.py or reagent_server.py) means
neither side duplicates code and tests can import this module directly.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_CATALOG_PATH = Path(__file__).parent.parent / "data" / "reagent_catalog.csv"
_df: pd.DataFrame | None = None

# Flat shipping rate; surcharge on large orders
_SHIPPING_BASE_USD = 25.0
_SHIPPING_SURCHARGE_THRESHOLD_USD = 200.0
_SHIPPING_SURCHARGE_RATE = 0.08  # 8% of order value above threshold

_SUPPLIER_CONTACTS: dict[str, str] = {
    "Sigma-Aldrich":     "orders.sigma@merckgroup.com | +1-800-325-3010",
    "MedChemExpress":    "sales@medchemexpress.com    | +1-732-484-9848",
    "Cayman Chemical":   "orders@caymanchem.com       | +1-734-971-3335",
    "Selleck Chemicals": "orders@selleckchem.com      | +1-832-582-8158",
}


def _catalog() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(_CATALOG_PATH)
    return _df


def _purity_to_int(purity_str: str) -> int:
    """Extract numeric purity floor from strings like '≥98%' → 98."""
    digits = "".join(c for c in purity_str if c.isdigit())
    return int(digits) if digits else 0


# ---------------------------------------------------------------------------
# search_reagents
# ---------------------------------------------------------------------------

def search_reagents_impl(query: str, target: str = "", min_purity_pct: int = 95) -> str:
    df = _catalog().copy()
    q = query.strip().lower()

    # Match compound_name, compound_id, or notes (case-insensitive)
    name_mask = df["compound_name"].str.lower().str.contains(q, na=False)
    id_mask   = df["compound_id"].str.lower().str.contains(q, na=False)
    note_mask = df["notes"].str.lower().str.contains(q, na=False)
    df = df[name_mask | id_mask | note_mask]

    if target:
        df = df[df["target"].str.lower().str.contains(target.lower(), na=False)]

    df = df[df["purity"].apply(_purity_to_int) >= min_purity_pct]

    if df.empty:
        return (
            f"No reagents found matching '{query}'"
            + (f" with target '{target}'" if target else "")
            + f" and purity ≥{min_purity_pct}%.\n"
            "Try a broader search term or lower the purity threshold."
        )

    cols = ["compound_name", "compound_id", "catalog_id", "supplier",
            "target", "purity", "price_per_mg", "min_quantity_mg", "in_stock", "lead_time_days"]
    result = df[cols].copy()
    result["in_stock"] = result["in_stock"].map({True: "✅ Yes", False: "❌ No"})
    result["lead_time_days"] = result["lead_time_days"].astype(str) + " days"
    result["price_per_mg"] = result["price_per_mg"].map("${:.2f}".format)
    result.columns = ["Name", "Internal ID", "Catalog ID", "Supplier",
                      "Target", "Purity", "Price/mg", "Min Qty (mg)", "In Stock", "Lead Time"]
    return result.to_markdown(index=False)


# ---------------------------------------------------------------------------
# check_stock
# ---------------------------------------------------------------------------

def check_stock_impl(catalog_id: str) -> str:
    df = _catalog()
    row = df[df["catalog_id"].str.upper() == catalog_id.strip().upper()]

    if row.empty:
        return (
            f"Catalog ID '{catalog_id}' not found.\n"
            "Use search_reagents to find the correct catalog ID."
        )

    r = row.iloc[0]
    in_stock   = bool(r["in_stock"])
    lead_days  = int(r["lead_time_days"])
    eta        = date.today() + timedelta(days=lead_days + 2)  # +2 processing days
    contact    = _SUPPLIER_CONTACTS.get(r["supplier"], r["supplier"])
    stock_icon = "✅ In Stock" if in_stock else "❌ Out of Stock"

    lines = [
        f"**{r['compound_name']}** ({r['catalog_id']} — {r['supplier']})",
        f"",
        f"| Field              | Value |",
        f"|--------------------|-------|",
        f"| Status             | {stock_icon} |",
        f"| Target             | {r['target']} |",
        f"| Purity             | {r['purity']} |",
        f"| Min. order qty     | {r['min_quantity_mg']} mg |",
        f"| Price per mg       | ${r['price_per_mg']:.2f} |",
        f"| Lead time          | {lead_days} business days |",
        f"| Est. delivery      | {eta.strftime('%Y-%m-%d')} (if ordered today) |",
        f"| Supplier contact   | {contact} |",
    ]
    if not in_stock:
        lines.append(f"\n⚠️ Item currently out of stock. Lead time reflects back-order estimate.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

def estimate_cost_impl(catalog_id: str, quantity_mg: float) -> str:
    df = _catalog()
    row = df[df["catalog_id"].str.upper() == catalog_id.strip().upper()]

    if row.empty:
        return (
            f"Catalog ID '{catalog_id}' not found.\n"
            "Use search_reagents to find the correct catalog ID."
        )

    r = row.iloc[0]
    min_qty = float(r["min_quantity_mg"])

    if quantity_mg < min_qty:
        return (
            f"Requested quantity {quantity_mg} mg is below the minimum order of {min_qty} mg "
            f"for {r['compound_name']} ({catalog_id}).\n"
            f"Please request at least {min_qty} mg."
        )

    subtotal = quantity_mg * float(r["price_per_mg"])
    shipping = _SHIPPING_BASE_USD
    if subtotal > _SHIPPING_SURCHARGE_THRESHOLD_USD:
        shipping += (subtotal - _SHIPPING_SURCHARGE_THRESHOLD_USD) * _SHIPPING_SURCHARGE_RATE
    total = subtotal + shipping

    lead_days = int(r["lead_time_days"])
    eta       = date.today() + timedelta(days=lead_days + 2)

    lines = [
        f"**Cost Estimate — {r['compound_name']}** ({catalog_id}, {r['supplier']})",
        f"",
        f"| Item                | Amount |",
        f"|---------------------|--------|",
        f"| Quantity            | {quantity_mg} mg |",
        f"| Unit price          | ${r['price_per_mg']:.2f} / mg |",
        f"| Subtotal            | ${subtotal:.2f} |",
        f"| Shipping & handling | ${shipping:.2f} |",
        f"| **Total**           | **${total:.2f}** |",
        f"| Est. delivery       | {eta.strftime('%Y-%m-%d')} ({lead_days} business days + 2 processing) |",
        f"",
        f"*Prices are indicative. Contact {r['supplier']} for a formal quote and bulk discounts.*",
    ]
    return "\n".join(lines)
