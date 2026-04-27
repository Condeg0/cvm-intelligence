# app/ — Streamlit Application

## Module Purpose

4-page Streamlit application serving as the user-facing product. Deployed to Streamlit Community Cloud.

## Pages

### 1. RAG Query Interface (`pages/1_query.py`)
- Text input for natural language queries (Portuguese or English)
- Sidebar filters: company ticker, date range, section type, sentiment
- Display: generated answer with source citations
- Expandable sections showing each retrieved chunk with metadata
- Show retrieval metadata: which retrieval method found each chunk, reranker scores

### 2. Financial Metrics Dashboard (`pages/2_dashboard.py`)
- Company selector dropdown (all 50 companies)
- Time series line charts for each metric (Plotly)
- Cross-company comparison table for any selected metric and period
- Data source: SQLite database queries
- Show extraction confidence: color-code values by `match_status` (exact = green, close = yellow, mismatch = red)

### 3. Sentiment Timeline (`pages/3_sentiment.py`)
- Company selector
- Timeline chart: sentiment score over quarters (Plotly)
- Click on a data point → show the actual management commentary text with sentiment label
- Cross-company sentiment comparison for any selected period
- Highlight trend changes (sentiment shift detection)

### 4. System Evaluation (`pages/4_evaluation.py`)
- **Extraction accuracy:** Exact match rate, MAPE, coverage. Failure taxonomy breakdown.
- **Retrieval metrics:** Ablation study results table (dense vs sparse vs hybrid vs hybrid+reranker). Recall@k, MRR, NDCG charts.
- **Embedding comparison:** Fine-tuned vs base vs multilingual — bar charts.
- **Sentiment classifier:** Confusion matrix, F1 score, Cohen's κ.
- **RAG quality:** Faithfulness/relevance/completeness scores on 20 evaluated queries.
- This page exists specifically for interviews. Make it clean, data-rich, and impressive.

## App Entry Point (`app.py`)

```python
import streamlit as st

st.set_page_config(
    page_title="CVM Filing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("CVM Filing Intelligence System")
st.markdown("Query Brazilian public company filings with semantic search and validated financial metrics.")
```

## Deployment Notes

- **Streamlit Community Cloud:** 1GB RAM limit. All models must be lightweight at inference.
- **Secrets:** Gemini API key in `.streamlit/secrets.toml` → access via `st.secrets["GEMINI_API_KEY"]`
- **Data files:** SQLite DB and ChromaDB persistence must be in the repo (Git LFS) or loaded at startup from GitHub Releases.
- **Model loading:** Use `@st.cache_resource` for sentence transformer and cross-encoder — load once, reuse across requests.

## Styling

Keep it professional and clean. No custom CSS unless necessary. Use Streamlit's native components. The evaluation page should look like a research dashboard — data-dense, well-labeled, with clear takeaways.
