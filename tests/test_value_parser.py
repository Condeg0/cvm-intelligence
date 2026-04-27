"""Tests for src/extraction/value_parser.py — Brazilian number format parsing."""

from __future__ import annotations

import pytest


class TestParseBrNumber:
    """Direct tests of parse_br_number (no surrounding noise)."""

    def test_basic_integer(self):
        from src.extraction.value_parser import parse_br_number
        assert parse_br_number("1234") == pytest.approx(1234.0)

    def test_thousands_separator(self):
        from src.extraction.value_parser import parse_br_number
        assert parse_br_number("1.234") == pytest.approx(1234.0)

    def test_decimal_only(self):
        from src.extraction.value_parser import parse_br_number
        assert parse_br_number("0,89") == pytest.approx(0.89)

    def test_decimal_with_thousands(self):
        from src.extraction.value_parser import parse_br_number
        assert parse_br_number("1.234,56") == pytest.approx(1234.56)

    def test_large_number(self):
        from src.extraction.value_parser import parse_br_number
        assert parse_br_number("1.234.567,89") == pytest.approx(1234567.89)

    def test_positive_sign(self):
        from src.extraction.value_parser import parse_br_number
        assert parse_br_number("+1.000") == pytest.approx(1000.0)

    def test_negative_sign(self):
        from src.extraction.value_parser import parse_br_number
        assert parse_br_number("-1.234,56") == pytest.approx(-1234.56)

    def test_zero(self):
        from src.extraction.value_parser import parse_br_number
        assert parse_br_number("0") == pytest.approx(0.0)

    def test_empty_returns_none(self):
        from src.extraction.value_parser import parse_br_number
        assert parse_br_number("") is None

    def test_non_numeric_returns_none(self):
        from src.extraction.value_parser import parse_br_number
        assert parse_br_number("abc") is None

    def test_mixed_text_returns_none(self):
        from src.extraction.value_parser import parse_br_number
        # Raw cell "1.234 mil" would be cleaned by normalize_cell first
        assert parse_br_number("1.234 mil") is None


class TestNormalizeCell:
    """Tests for normalize_cell, which handles raw pdfplumber cell strings."""

    def test_parentheses_negative(self):
        from src.extraction.value_parser import normalize_cell
        assert normalize_cell("(45.678)") == pytest.approx(-45678.0)

    def test_parentheses_negative_with_decimal(self):
        from src.extraction.value_parser import normalize_cell
        assert normalize_cell("(1.234,56)") == pytest.approx(-1234.56)

    def test_empty_cell_returns_none(self):
        from src.extraction.value_parser import normalize_cell
        assert normalize_cell("") is None
        assert normalize_cell(None) is None

    def test_dash_returns_zero(self):
        from src.extraction.value_parser import normalize_cell
        assert normalize_cell("-") == pytest.approx(0.0)
        assert normalize_cell("—") == pytest.approx(0.0)
        assert normalize_cell("–") == pytest.approx(0.0)

    def test_non_numeric_returns_none(self):
        from src.extraction.value_parser import normalize_cell
        assert normalize_cell("n/a") is None
        assert normalize_cell("Receita Líquida") is None

    def test_currency_prefix_stripped(self):
        from src.extraction.value_parser import normalize_cell
        assert normalize_cell("R$ 1.234,56") == pytest.approx(1234.56)

    def test_leading_minus(self):
        from src.extraction.value_parser import normalize_cell
        assert normalize_cell("-1.234,56") == pytest.approx(-1234.56)

    def test_leading_plus(self):
        from src.extraction.value_parser import normalize_cell
        assert normalize_cell("+1.000") == pytest.approx(1000.0)

    def test_whitespace_stripped(self):
        from src.extraction.value_parser import normalize_cell
        assert normalize_cell("  1.234,56  ") == pytest.approx(1234.56)

    def test_large_value_as_in_petrobras(self):
        from src.extraction.value_parser import normalize_cell
        # Petrobras reports in R$ millions; 613.334.000 = 613,334,000
        assert normalize_cell("613.334.000") == pytest.approx(613_334_000.0)


class TestDetectScale:
    """Tests for detect_scale (table header unit detection)."""

    def test_reais_mil(self):
        from src.extraction.value_parser import detect_scale
        assert detect_scale("(Reais Mil)") == pytest.approx(1_000.0)

    def test_em_r_mil(self):
        from src.extraction.value_parser import detect_scale
        assert detect_scale("Em R$ mil") == pytest.approx(1_000.0)

    def test_milhoes(self):
        from src.extraction.value_parser import detect_scale
        assert detect_scale("(em R$ milhões)") == pytest.approx(1_000_000.0)

    def test_bilhoes(self):
        from src.extraction.value_parser import detect_scale
        assert detect_scale("Em bilhões de reais") == pytest.approx(1_000_000_000.0)

    def test_no_indicator(self):
        from src.extraction.value_parser import detect_scale
        assert detect_scale("Descrição da Conta") == pytest.approx(1.0)
