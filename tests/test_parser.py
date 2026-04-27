"""Tests for src/parsing/pdf_parser.py and src/parsing/section_detector.py."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Section detector tests
# ---------------------------------------------------------------------------

class TestSectionDetector:
    """Test classify_block against all expected CVM heading patterns."""

    def test_relatorio_da_administracao(self):
        from src.parsing.section_detector import classify_block, SectionType
        assert classify_block("Relatório da Administração") == SectionType.RELATORIO_ADMINISTRACAO

    def test_relatorio_da_administracao_multiline(self):
        from src.parsing.section_detector import classify_block, SectionType
        # ITUB4 uses "Relatório da  \nAdministração \n1T23"
        assert classify_block("Relatório da  \nAdministração \n1T23") == SectionType.RELATORIO_ADMINISTRACAO

    def test_comentario_desempenho(self):
        from src.parsing.section_detector import classify_block, SectionType
        # Petrobras uses "Comentário do Desempenho"
        assert classify_block("Comentário do Desempenho") == SectionType.RELATORIO_ADMINISTRACAO

    def test_comentarios_de_desempenho(self):
        from src.parsing.section_detector import classify_block, SectionType
        assert classify_block("Comentários de Desempenho") == SectionType.RELATORIO_ADMINISTRACAO

    def test_balanco_patrimonial_ativo(self):
        from src.parsing.section_detector import classify_block, SectionType
        assert classify_block("DFs Individuais / Balanço Patrimonial Ativo") == SectionType.BALANCO_PATRIMONIAL

    def test_balanco_patrimonial_consolidado(self):
        from src.parsing.section_detector import classify_block, SectionType
        assert classify_block("Balanço Patrimonial Consolidado") == SectionType.BALANCO_PATRIMONIAL

    def test_dre_standard(self):
        from src.parsing.section_detector import classify_block, SectionType
        assert classify_block("DFs Consolidadas / Demonstração do Resultado") == SectionType.DRE

    def test_dre_abrangente_not_matched(self):
        from src.parsing.section_detector import classify_block, SectionType
        # "Demonstração do Resultado Abrangente" should NOT be classified as DRE
        result = classify_block("DFs Individuais / Demonstração do Resultado Abrangente")
        assert result != SectionType.DRE

    def test_notas_explicativas(self):
        from src.parsing.section_detector import classify_block, SectionType
        assert classify_block("Notas Explicativas") == SectionType.NOTAS_EXPLICATIVAS

    def test_notas_explicativas_full_phrase(self):
        from src.parsing.section_detector import classify_block, SectionType
        assert classify_block("Notas Explicativas às Demonstrações Financeiras") == SectionType.NOTAS_EXPLICATIVAS

    def test_notas_explicativas_uppercase(self):
        from src.parsing.section_detector import classify_block, SectionType
        assert classify_block("NOTAS EXPLICATIVAS") == SectionType.NOTAS_EXPLICATIVAS

    def test_unknown_text(self):
        from src.parsing.section_detector import classify_block, SectionType
        assert classify_block("Banco do Brasil S.A. – Demonstrações Contábeis") == SectionType.UNKNOWN

    def test_empty_string(self):
        from src.parsing.section_detector import classify_block, SectionType
        assert classify_block("") == SectionType.UNKNOWN

    def test_detect_sections_with_markers(self):
        """detect_sections correctly parses [SECTION:xxx] markers."""
        from src.parsing.section_detector import detect_sections, SectionType

        full_text = (
            "\n\n[SECTION:Relatório da Administração]\n\n"
            "A companhia teve bom desempenho.\n\n"
            "[SECTION:Notas Explicativas]\n\n"
            "Nota 1. Políticas contábeis.\n"
        )
        sections = detect_sections(full_text)
        assert len(sections) == 2
        assert sections[0].section_type == SectionType.RELATORIO_ADMINISTRACAO
        assert sections[1].section_type == SectionType.NOTAS_EXPLICATIVAS
        assert "bom desempenho" in sections[0].text
        assert "Políticas contábeis" in sections[1].text

    def test_detect_sections_merges_consecutive(self):
        """Consecutive pages of the same section are merged into one entry."""
        from src.parsing.section_detector import detect_sections, SectionType

        full_text = (
            "[SECTION:Notas Explicativas]\n\nNota 1.\n\n"
            "[SECTION:Notas Explicativas]\n\nNota 2.\n\n"
        )
        sections = detect_sections(full_text)
        assert len(sections) == 1
        assert sections[0].section_type == SectionType.NOTAS_EXPLICATIVAS
        assert "Nota 1" in sections[0].text
        assert "Nota 2" in sections[0].text

    def test_detect_sections_fallback_regex(self):
        """Falls back to regex scan when no [SECTION:] markers present."""
        from src.parsing.section_detector import detect_sections, SectionType

        plain_text = (
            "Some preamble.\n\n"
            "Relatório da Administração\n\n"
            "A empresa cresceu.\n\n"
            "Notas Explicativas\n\n"
            "Nota 1.\n"
        )
        sections = detect_sections(plain_text)
        assert len(sections) >= 2
        types = [s.section_type for s in sections]
        assert SectionType.RELATORIO_ADMINISTRACAO in types
        assert SectionType.NOTAS_EXPLICATIVAS in types


# ---------------------------------------------------------------------------
# pdf_parser tests (unit-level, no real PDF required)
# ---------------------------------------------------------------------------

class TestPdfParserHelpers:
    """Unit tests for internal helper functions in pdf_parser."""

    def test_is_metadata_block_cvm_banner(self):
        from src.parsing.pdf_parser import TextBlock, _is_metadata_block

        block = TextBlock(
            page_number=1, text="ITR - Informações Trimestrais - 31/03/2023 - EMPRESA",
            x0=0, y0=15, x1=400, y1=27, block_number=0
        )
        assert _is_metadata_block(block)

    def test_is_metadata_block_publica(self):
        from src.parsing.pdf_parser import TextBlock, _is_metadata_block

        block = TextBlock(
            page_number=1, text="PÚBLICA", x0=0, y0=809, x1=100, y1=820, block_number=1
        )
        assert _is_metadata_block(block)

    def test_is_metadata_block_versao(self):
        from src.parsing.pdf_parser import TextBlock, _is_metadata_block

        block = TextBlock(
            page_number=1, text="Versão : 1", x0=0, y0=25, x1=100, y1=35, block_number=2
        )
        assert _is_metadata_block(block)

    def test_is_metadata_block_real_content(self):
        from src.parsing.pdf_parser import TextBlock, _is_metadata_block

        block = TextBlock(
            page_number=1, text="A empresa apresentou crescimento consistente.",
            x0=0, y0=200, x1=400, y1=212, block_number=3
        )
        assert not _is_metadata_block(block)


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

class TestChunker:
    """Test estimate_token_count, split_into_sentences, and chunk_sections."""

    def test_estimate_token_count_empty(self):
        from src.parsing.chunker import estimate_token_count
        assert estimate_token_count("") == 1  # max(1, ...)

    def test_estimate_token_count_ten_words(self):
        from src.parsing.chunker import estimate_token_count
        # 10 words × 1.33 = 13.3 → 13
        result = estimate_token_count("um dois três quatro cinco seis sete oito nove dez")
        assert result == 13

    def test_split_into_sentences_basic(self):
        from src.parsing.chunker import split_into_sentences
        text = "A empresa cresceu. O resultado foi positivo. Os acionistas aprovaram."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
        assert "A empresa cresceu." in sentences[0]

    def test_split_into_sentences_preserves_sa(self):
        from src.parsing.chunker import split_into_sentences
        # "S.A." should not be a sentence boundary
        text = "A Petrobras S.A. registrou lucro. A Vale S.A. também cresceu."
        sentences = split_into_sentences(text)
        assert len(sentences) == 2
        assert "Petrobras S.A." in sentences[0]
        assert "Vale S.A." in sentences[1]

    def test_split_into_sentences_single_sentence(self):
        from src.parsing.chunker import split_into_sentences
        text = "Apenas uma frase sem ponto final"
        sentences = split_into_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == text

    def test_chunk_sections_prose_only(self):
        """Structured sections (BP, DRE) are skipped; only prose is chunked."""
        from src.parsing.section_detector import DetectedSection, SectionType
        from src.parsing.chunker import chunk_sections

        sections = [
            DetectedSection(
                section_type=SectionType.BALANCO_PATRIMONIAL,
                start_char=0, end_char=100,
                text="Ativo Total 1.000.000 Passivo Total 1.000.000",
            ),
            DetectedSection(
                section_type=SectionType.RELATORIO_ADMINISTRACAO,
                start_char=100, end_char=800,
                text=(
                    "A companhia apresentou crescimento sólido no período. "
                    "Os resultados superaram as expectativas dos analistas de mercado. "
                    "A direção reafirma o compromisso com a criação de valor sustentável. "
                    "Os indicadores operacionais estão dentro do planejado para o exercício. "
                    "A estratégia de expansão segue o roadmap aprovado pelo conselho. "
                    "Os investimentos em tecnologia e inovação continuam sendo prioritários. "
                    "O balanço patrimonial reflete solidez financeira da organização."
                ),
            ),
        ]
        chunks = chunk_sections(sections, "test_filing")
        # Only RA section produces chunks
        assert len(chunks) > 0
        assert all(c.section_type == SectionType.RELATORIO_ADMINISTRACAO for c in chunks)

    def test_chunk_sections_no_tiny_chunks(self):
        """Chunks below MIN_CHUNK_TOKENS (50) are discarded."""
        from src.parsing.section_detector import DetectedSection, SectionType
        from src.parsing.chunker import chunk_sections, MIN_CHUNK_TOKENS

        sections = [
            DetectedSection(
                section_type=SectionType.NOTAS_EXPLICATIVAS,
                start_char=0, end_char=50,
                text="Nota 1.",
            ),
        ]
        chunks = chunk_sections(sections, "test_filing")
        assert all(c.token_count >= MIN_CHUNK_TOKENS for c in chunks)

    def test_chunk_ids_are_unique(self):
        from src.parsing.section_detector import DetectedSection, SectionType
        from src.parsing.chunker import chunk_sections

        long_text = " ".join(["Frase de exemplo com conteúdo financeiro relevante."] * 50)
        sections = [
            DetectedSection(
                section_type=SectionType.RELATORIO_ADMINISTRACAO,
                start_char=0, end_char=len(long_text),
                text=long_text,
            ),
        ]
        chunks = chunk_sections(sections, "filing_xyz")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"
