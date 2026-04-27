"""Tests for src/extraction/metric_extractor.py."""

from __future__ import annotations

import pytest


@pytest.fixture
def simple_revenue_table():
    """A minimal financial table containing a revenue row."""
    return [
        ["Descrição", "2023", "2022"],
        ["Receita Líquida", "1.500.000", "1.200.000"],
        ["Custo dos Produtos Vendidos", "(900.000)", "(720.000)"],
        ["Resultado Bruto", "600.000", "480.000"],
    ]


class TestExtractFromTable:
    def test_finds_revenue(self, simple_revenue_table):
        from src.extraction.metric_extractor import extract_from_table, METRIC_PATTERNS

        result = extract_from_table(
            simple_revenue_table,
            "revenue",
            METRIC_PATTERNS["revenue"],
            page_number=1,
        )
        assert result is not None
        assert result.metric_name == "revenue"
        assert result.parsed_value == pytest.approx(1_500_000.0)

    def test_finds_gross_profit(self, simple_revenue_table):
        from src.extraction.metric_extractor import extract_from_table, METRIC_PATTERNS

        result = extract_from_table(
            simple_revenue_table,
            "gross_profit",
            METRIC_PATTERNS["gross_profit"],
            page_number=1,
        )
        assert result is not None
        assert result.parsed_value == pytest.approx(600_000.0)

    def test_returns_none_when_not_found(self, simple_revenue_table):
        from src.extraction.metric_extractor import extract_from_table, METRIC_PATTERNS

        result = extract_from_table(
            simple_revenue_table,
            "ebitda",
            METRIC_PATTERNS["ebitda"],
            page_number=1,
        )
        assert result is None
