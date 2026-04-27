"""Context assembly and answer generation via Gemini 2.5 Flash.

Assembles a structured prompt from the top reranked chunks and calls the
Gemini API to produce a grounded answer with citations.
"""

from __future__ import annotations

import logging
import os

from src.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

ANSWER_PROMPT_TEMPLATE = """\
Você é um analista financeiro especializado em empresas brasileiras de capital aberto.
Responda a pergunta abaixo com base exclusivamente nos trechos fornecidos.
Cite os trechos relevantes pelo número entre colchetes [1], [2], etc.
Se a resposta não puder ser encontrada nos trechos, diga "Informação não disponível nos documentos fornecidos."

Pergunta: {query}

Trechos:
{context}

Resposta:"""


def assemble_context(chunks: list[RetrievedChunk]) -> str:
    """Format reranked chunks into a numbered context block for the prompt.

    Each chunk is preceded by a header line showing its source metadata so
    the model can produce accurate citations.

    Args:
        chunks: Top-K reranked chunks from :func:`~src.rag.reranker.rerank`.

    Returns:
        Formatted string with each chunk numbered and preceded by its metadata.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] {chunk.ticker} | {chunk.reference_date} | {chunk.section}"
        parts.append(f"{header}\n{chunk.text}")
    return "\n---\n".join(parts)


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
) -> str:
    """Call Gemini 2.5 Flash to generate a grounded answer.

    Assembles the context from the reranked chunks, fills the prompt template,
    and sends the request to the Gemini API.  Falls back to the deprecated
    ``google-generativeai`` SDK if the new ``google-genai`` package is not
    importable.

    Args:
        query: The user's question.
        chunks: Reranked context chunks from the Phase 5 pipeline.

    Returns:
        Generated answer string from Gemini, with inline citations.
    """
    from src import config

    context = assemble_context(chunks)
    prompt = ANSWER_PROMPT_TEMPLATE.format(query=query, context=context)

    api_key = os.getenv("GEMINI_API_KEY") or config.GEMINI_API_KEY

    # Primary: new google-genai SDK
    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
        )
        return response.text
    except ImportError:
        logger.debug("google-genai not available, trying google-generativeai fallback")
    except Exception as exc:
        logger.error("Gemini API error (google-genai): %s", exc)
        raise

    # Fallback: deprecated google-generativeai SDK
    try:
        import google.generativeai as genai_old  # type: ignore

        genai_old.configure(api_key=api_key)
        model = genai_old.GenerativeModel(config.GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except ImportError:
        logger.error("Neither google-genai nor google-generativeai is installed.")
        return "Erro: SDK Gemini não disponível."
    except Exception as exc:
        logger.error("Gemini API error (google-generativeai): %s", exc)
        raise
