"""Unit tests for reagent catalog business logic (no network, no MCP server needed)."""
from __future__ import annotations

import pytest
from mcp_servers.reagent_catalog import search_reagents_impl, check_stock_impl, estimate_cost_impl


# ---------------------------------------------------------------------------
# search_reagents_impl
# ---------------------------------------------------------------------------

class TestSearchReagents:
    def test_search_by_compound_name(self):
        result = search_reagents_impl("Osimertinib")
        assert "Osimertinib" in result
        assert "HY-15772" in result
        assert "MedChemExpress" in result

    def test_search_by_internal_id(self):
        result = search_reagents_impl("SR-0472")
        assert "SR-0472" in result
        assert "Osimertinib" in result

    def test_search_by_target(self):
        result = search_reagents_impl("inhibitor", target="BRAF")
        assert "BRAF" in result
        # Should include Vemurafenib, Dabrafenib, Encorafenib
        assert "Vemurafenib" in result or "Dabrafenib" in result

    def test_search_with_purity_filter(self):
        # ≥99% filter should exclude ≥98% entries
        result = search_reagents_impl("EGFR", min_purity_pct=99)
        # Osimertinib is ≥99% and EGFR target — should appear
        assert "Osimertinib" in result
        # Gefitinib is ≥98% — should be excluded at purity ≥99
        assert "Gefitinib" not in result

    def test_search_no_results_returns_helpful_message(self):
        result = search_reagents_impl("XYZ_NONEXISTENT_COMPOUND_12345")
        assert "No reagents found" in result
        assert "broader search" in result.lower() or "lower" in result.lower()

    def test_search_cdk4_returns_multiple(self):
        result = search_reagents_impl("CDK4")
        assert "Palbociclib" in result
        assert "Ribociclib" in result


# ---------------------------------------------------------------------------
# check_stock_impl
# ---------------------------------------------------------------------------

class TestCheckStock:
    def test_in_stock_item(self):
        result = check_stock_impl("SML1277")  # Gefitinib — in_stock=True
        assert "In Stock" in result
        assert "Gefitinib" in result
        assert "lead time" in result.lower() or "Lead time" in result

    def test_out_of_stock_item(self):
        result = check_stock_impl("HY-15840")  # Rociletinib — in_stock=False
        assert "Out of Stock" in result
        assert "Rociletinib" in result
        assert "back-order" in result.lower() or "out of stock" in result.lower()

    def test_unknown_catalog_id(self):
        result = check_stock_impl("INVALID-9999")
        assert "not found" in result.lower()
        assert "search_reagents" in result

    def test_includes_supplier_contact(self):
        result = check_stock_impl("HY-15772")  # MedChemExpress
        assert "medchemexpress" in result.lower()

    def test_includes_estimated_delivery(self):
        result = check_stock_impl("HY-15772")
        assert "delivery" in result.lower() or "Est." in result


# ---------------------------------------------------------------------------
# estimate_cost_impl
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_basic_cost_calculation(self):
        # Osimertinib HY-15772: $1.20/mg, min 5mg
        result = estimate_cost_impl("HY-15772", 10.0)
        assert "10" in result          # quantity
        assert "$12.00" in result      # subtotal: 10 * 1.20
        assert "Total" in result
        assert "Shipping" in result.lower() or "shipping" in result.lower()

    def test_below_minimum_quantity_rejected(self):
        result = estimate_cost_impl("HY-15772", 2.0)  # min is 5mg
        assert "minimum" in result.lower() or "below" in result.lower()
        assert "5" in result  # mentions minimum

    def test_shipping_surcharge_on_large_order(self):
        # Price > $200 should trigger shipping surcharge
        # SML1277 (Gefitinib): $0.45/mg → need >445mg for subtotal >$200
        result = estimate_cost_impl("SML1277", 500.0)
        subtotal = 500 * 0.45  # $225
        # Base shipping $25 + 8% of ($225 - $200) = $25 + $2 = $27
        assert "$227" in result or "227" in result or "$27" in result

    def test_unknown_catalog_id(self):
        result = estimate_cost_impl("INVALID-9999", 10.0)
        assert "not found" in result.lower()

    def test_result_includes_delivery_date(self):
        result = estimate_cost_impl("PD0332991", 10.0)  # Palbociclib
        assert "delivery" in result.lower() or "Est." in result

    def test_result_includes_disclaimer(self):
        result = estimate_cost_impl("SML1277", 10.0)
        assert "indicative" in result.lower() or "formal quote" in result.lower()


# ---------------------------------------------------------------------------
# LangChain tool routing (search_reagents tool wrapper)
# ---------------------------------------------------------------------------

class TestSearchReagentsTool:
    def test_tool_routes_to_search_by_name(self):
        from agent.tools import search_reagents
        result = search_reagents.invoke({"query": "Osimertinib"})
        assert "Osimertinib" in result

    def test_tool_routes_to_cost_estimate_when_qty_and_catalog_id_present(self):
        from agent.tools import search_reagents
        result = search_reagents.invoke({"query": "estimate cost for HY-15772 10mg"})
        assert "Total" in result
        assert "$12.00" in result

    def test_tool_routes_to_stock_check(self):
        from agent.tools import search_reagents
        result = search_reagents.invoke({"query": "is HY-15772 in stock?"})
        assert "Osimertinib" in result
        assert "Stock" in result
