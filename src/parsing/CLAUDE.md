# src/parsing/ — PDF Parsing & Section Detection

## Module Purpose

Extract structured text and tables from CVM filing PDFs (ITR and DFP). This is the foundation — everything downstream depends on parsing quality.

## Architecture

- `pdf_parser.py` — Orchestrator. Takes a PDF path, returns a structured `FilingDocument` object with sections, text, and tables.
- `section_detector.py` — Identifies major filing sections by matching heading patterns (font size, bold, regex).
- `chunker.py` — Splits section text into semantically coherent chunks at paragraph boundaries. Falls back to sentence boundaries for chunks exceeding 384 tokens.

## CVM Filing Structure

Brazilian public company filings typically contain these sections (order varies):

| Section | Portuguese Name | Content Type |
|---|---|---|
| Balance Sheet | Balanço Patrimonial | Tables (assets, liabilities, equity) |
| Income Statement | Demonstração do Resultado (DRE) | Tables (revenue through net income) |
| Cash Flow Statement | Demonstração dos Fluxos de Caixa (DFC) | Tables |
| Management Report | Relatório da Administração | Prose (qualitative commentary) |
| Explanatory Notes | Notas Explicativas | Mixed (prose + tables) |
| Auditor's Report | Relatório do Auditor | Prose |

## Section Detection Patterns

Common heading patterns to match (case-insensitive, with accent tolerance):

```
Balanço Patrimonial
Demonstração do Resultado
Demonstração dos Fluxos de Caixa
Demonstração das Mutações do Patrimônio Líquido (DMPL)
Relatório da Administração
Relatório da Diretoria
Notas Explicativas
Parecer dos Auditores / Relatório do Auditor
Comentário de Desempenho
```

Detection heuristics:
- Headings are often in larger font size or bold
- Headings often appear on their own line with whitespace above/below
- Some filings use numbered sections ("1. Relatório da Administração")
- Watch for OCR artifacts that break heading text

## Chunking Rules

1. Split on paragraph boundaries (double newline or significant vertical gap in PDF layout)
2. Each chunk carries metadata: `{filing_id, section_name, chunk_index, token_count}`
3. If a paragraph exceeds 384 tokens, split at sentence boundaries (`. ` followed by uppercase letter)
4. Minimum chunk size: 50 tokens (skip tiny fragments like headers or page numbers)
5. Preserve paragraph integrity — never split mid-sentence unless the sentence itself exceeds 384 tokens

## Known Quirks

- **Page headers/footers:** CVM PDFs repeat company name, page number, and date on every page. Strip these before chunking.
- **Two-column layouts:** Some older filings use two-column layout. PyMuPDF's text block positions can detect this.
- **Number formatting:** Brazilian format uses `.` for thousands and `,` for decimals. Example: `R$ 1.234.567,89`.
- **Encoding:** Most CVM PDFs are UTF-8, but some older ones have Latin-1 encoding issues. Handle gracefully.
- **Empty pages:** Some filings have blank separator pages. Skip them.
- **Image-only pages:** Some pages are scanned images with no extractable text. Log and skip — don't crash.

## Testing

- Test section detection regex against at least 20 manually verified documents
- Test chunker output: verify no chunk exceeds 384 tokens, no chunk is below 50 tokens, metadata is correct
- Test on PDFs from at least 5 different companies to catch formatting variation
