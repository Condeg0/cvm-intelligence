# src/extraction/ — Quantitative Metric Extraction & Validation

## Module Purpose

Extract 9 target financial metrics from PDF tables using deterministic rules (regex + table position heuristics) and validate every extracted value against CVM structured CSV ground truth.

**Critical constraint:** This module is RULE-BASED. Never use an LLM to extract numbers. The entire point is deterministic, reproducible, debuggable extraction with measured accuracy.

## Architecture

- `metric_extractor.py` — Orchestrator. Takes parsed tables from a filing, returns extracted metrics as a list of `ExtractedMetric` objects.
- `value_parser.py` — Converts Brazilian-formatted number strings to Python floats. Handles edge cases (negative values in parentheses, thousands separators, currency symbols).
- `validator.py` — Joins extracted metrics with CSV ground truth by (company, period, metric) and computes accuracy metrics.

## Target Metrics and Label Patterns

Each metric has multiple possible Portuguese labels. Match case-insensitively and with fuzzy whitespace.

```python
METRIC_PATTERNS = {
    "revenue": [
        r"Receita\s+de\s+Venda",
        r"Receita\s+Líquida",
        r"Receita\s+Operacional\s+Líquida",
    ],
    "cogs": [
        r"Custo\s+dos?\s+Bens",
        r"Custo\s+dos?\s+Produtos",
        r"Custo\s+dos?\s+Serviços",
    ],
    "gross_profit": [
        r"Resultado\s+Bruto",
        r"Lucro\s+Bruto",
    ],
    "ebitda": [
        r"EBITDA",
        r"LAJIDA",
    ],
    "net_income": [
        r"Lucro\s*\(?Prejuízo\)?\s+Líquido",
        r"Resultado\s+Líquido",
    ],
    "total_assets": [
        r"Ativo\s+Total",
    ],
    "total_equity": [
        r"Patrimônio\s+Líquido",
    ],
    "net_debt": [
        r"Dívida\s+Líquida",
    ],
    "operating_cash_flow": [
        r"Caixa\s+(?:Líquido\s+)?(?:das?\s+)?Atividades\s+Operacionais",
    ],
}
```

## Value Parsing Rules

Brazilian number format: `1.234.567,89`

```
Input                    → Output
"1.234.567,89"          → 1234567.89
"(1.234,56)"            → -1234.56      # parentheses = negative
"-1.234,56"             → -1234.56
"R$ 1.234,56"           → 1234.56       # strip currency
"1.234"                 → 1234.0        # no decimal part
"0,89"                  → 0.89
"-"                     → 0.0 or None   # dash = zero or missing
""                      → None          # empty = missing
```

**Unit ambiguity:** CVM tables often express values in thousands (`em milhares de R$`) or millions. Check table headers for unit indicators and apply the appropriate multiplier. Common patterns:
- `(em R$ mil)` or `(em milhares)` → multiply by 1,000
- `(em R$ milhões)` or `(em milhões)` → multiply by 1,000,000
- `(em R$)` or no indicator → use as-is

## Extraction Strategy

1. For each target metric, scan all extracted tables for rows matching the label patterns
2. When a match is found, identify the correct column (current period vs. prior period). The current period column is typically the first numeric column after the label column.
3. Parse the value using `value_parser.py`
4. If multiple tables match the same metric, prefer the table from the expected section (e.g., revenue from DRE, total assets from Balanço Patrimonial)

## Validation Against CSV Ground Truth

The CVM structured CSVs use standardized column names. Key columns:
- `CD_CVM` — company code
- `DT_REFER` — reference date
- `DS_CONTA` — account description (maps to metric labels)
- `VL_CONTA` — account value
- `CD_CONTA` — account code (standardized numbering)

For validation:
1. Load CSV for the same company and period
2. Match `DS_CONTA` to the metric being validated
3. Compare `VL_CONTA` with `extracted_value`
4. Classify: `exact` (values equal within rounding), `close` (within 1% — likely unit/rounding difference), `mismatch` (>1% difference), `missing` (metric not found in PDF)

## Failure Taxonomy

Every extraction failure is categorized for the evaluation page:

| Code | Description |
|---|---|
| `table_not_found` | No table detected in the expected section |
| `row_not_matched` | Label pattern didn't match any row |
| `wrong_row` | Matched a similar but incorrect row (e.g., "Receita Bruta" instead of "Receita Líquida") |
| `wrong_column` | Extracted prior period instead of current period |
| `parse_error` | Value string couldn't be parsed to float |
| `unit_mismatch` | Thousands/millions multiplier applied incorrectly |
| `section_missing` | The entire section was not found in the PDF |

## Testing

- `test_value_parser.py` — exhaustive test cases for Brazilian number parsing (including edge cases above)
- `test_extractor.py` — test metric extraction on 3–5 sample tables with known correct values
- Validation results should be reproducible: same input → same accuracy metrics
