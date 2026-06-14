"""Sentiment timeline — management commentary sentiment over time.

Queries the chunks table for sentiment labels and scores, then renders an
interactive timeline per company. Clicking a period reveals the actual chunk
text that drove the sentiment reading.

Requires sentiment scores to be pre-computed:
    python scripts/run_sentiment_scoring.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src import config  # noqa: E402

st.set_page_config(page_title="Sentiment Timeline", page_icon="💬", layout="wide")

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_companies() -> pd.DataFrame:
    with sqlite3.connect(config.DASHBOARD_DB_PATH) as conn:
        return pd.read_sql("SELECT ticker, name FROM companies ORDER BY ticker", conn)


@st.cache_data(ttl=300)
def load_sentiment(ticker: str, section: str | None) -> pd.DataFrame:
    """Aggregate sentiment per filing period for one company."""
    section_clause = ""
    params: list = [ticker]
    if section:
        section_clause = "AND c.section_name = ?"
        params.append(section)

    query = f"""
        SELECT
            f.reference_date,
            f.filing_type,
            c.section_name,
            c.sentiment_label,
            AVG(c.sentiment_score)   AS avg_score,
            COUNT(*)                 AS chunk_count
        FROM chunks c
        JOIN filings f ON c.filing_id = f.filing_id
        WHERE f.ticker = ?
          AND c.sentiment_label IS NOT NULL
          {section_clause}
        GROUP BY f.reference_date, f.filing_type, c.section_name, c.sentiment_label
        ORDER BY f.reference_date, c.sentiment_label
    """
    try:
        with sqlite3.connect(config.DASHBOARD_DB_PATH) as conn:
            df = pd.read_sql(query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    return df


@st.cache_data(ttl=300)
def load_chunks_for_period(ticker: str, period: str, section: str | None) -> pd.DataFrame:
    """Return individual chunks for one company + period (for drill-down)."""
    section_clause = ""
    params: list = [ticker, period]
    if section:
        section_clause = "AND c.section_name = ?"
        params.append(section)

    query = f"""
        SELECT
            c.chunk_id,
            c.section_name,
            c.chunk_text,
            c.sentiment_label,
            c.sentiment_score
        FROM chunks c
        JOIN filings f ON c.filing_id = f.filing_id
        WHERE f.ticker = ?
          AND f.reference_date = ?
          AND c.sentiment_label IS NOT NULL
          {section_clause}
        ORDER BY c.sentiment_score DESC
        LIMIT 50
    """
    try:
        with sqlite3.connect(config.DASHBOARD_DB_PATH) as conn:
            return pd.read_sql(query, conn, params=params)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def check_sentiment_populated() -> int:
    try:
        with sqlite3.connect(config.DASHBOARD_DB_PATH) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE sentiment_label IS NOT NULL"
            ).fetchone()[0]
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("Sentiment Timeline")
st.markdown("Track management commentary tone across filing periods.")

labeled_count = check_sentiment_populated()
if labeled_count == 0:
    st.warning(
        "Sentiment scores have not been computed yet. Run:\n\n"
        "```bash\npython scripts/run_sentiment_scoring.py\n```\n\n"
        "This takes ~20 min on GPU. The page will update automatically once done."
    )
    st.stop()

st.caption(f"{labeled_count:,} chunks scored")

companies_df = load_companies()
ticker_options = [f"{r.ticker} — {r.name}" for r in companies_df.itertuples()]

with st.sidebar:
    st.header("Filters")
    selected_label = st.selectbox("Company", ticker_options, index=0)
    ticker = selected_label.split(" — ")[0]

    section = st.selectbox(
        "Section",
        ["All", "Relatório da Administração", "Notas Explicativas"],
    )
    section_filter = None if section == "All" else section

# ---------------------------------------------------------------------------
# Sentiment timeline chart
# ---------------------------------------------------------------------------

df = load_sentiment(ticker, section_filter)
company_name = companies_df.loc[companies_df.ticker == ticker, "name"].iloc[0]

if df.empty:
    st.info(f"No sentiment data for {ticker} in the selected section.")
    st.stop()

st.subheader(f"{ticker} — {company_name}")

# Pivot: for each period get positive/negative chunk counts
pivot = df.pivot_table(
    index="reference_date", columns="sentiment_label", values="chunk_count",
    aggfunc="sum", fill_value=0,
).reset_index()

pos_col = "positive" if "positive" in pivot.columns else None
neg_col = "negative" if "negative" in pivot.columns else None

if pos_col and neg_col:
    pivot["positive_share"] = (
        pivot["positive"] / (pivot["positive"] + pivot["negative"]) * 100
    ).round(1)
elif pos_col:
    pivot["positive_share"] = 100.0
else:
    pivot["positive_share"] = 0.0

pivot["period_str"] = pivot["reference_date"].dt.strftime("%Y-%m-%d")

# Altair line chart — positive share over time
line = (
    alt.Chart(pivot)
    .mark_line(point=True, color="#2ecc71", strokeWidth=2)
    .encode(
        x=alt.X("reference_date:T", title="Period", axis=alt.Axis(format="%Y-%m")),
        y=alt.Y(
            "positive_share:Q",
            title="Positive chunk share (%)",
            scale=alt.Scale(domain=[0, 100]),
        ),
        tooltip=[
            alt.Tooltip("period_str:N", title="Period"),
            alt.Tooltip("positive_share:Q", title="Positive %", format=".1f"),
        ] + (
            [alt.Tooltip("positive:Q", title="Positive chunks")] if pos_col else []
        ) + (
            [alt.Tooltip("negative:Q", title="Negative chunks")] if neg_col else []
        ),
    )
    .properties(height=280, title=f"Positive sentiment share — {ticker}")
)

rule = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(
    color="#95a5a6", strokeDash=[4, 4]
).encode(y="y:Q")

st.altair_chart(line + rule, use_container_width=True)

# Stacked bar — positive vs negative counts per period
bar_data = df[["reference_date", "sentiment_label", "chunk_count"]].copy()
bar_data["period_str"] = bar_data["reference_date"].dt.strftime("%Y-%m-%d")
sorted_periods = sorted(bar_data["period_str"].unique())

color_scale = alt.Scale(
    domain=["positive", "negative"],
    range=["#2ecc71", "#e74c3c"],
)

bar = (
    alt.Chart(bar_data)
    .mark_bar()
    .encode(
        x=alt.X("period_str:N", title="Period", sort=sorted_periods),
        y=alt.Y("chunk_count:Q", title="Chunk count"),
        color=alt.Color("sentiment_label:N", scale=color_scale, title="Sentiment"),
        tooltip=[
            alt.Tooltip("period_str:N", title="Period"),
            alt.Tooltip("sentiment_label:N", title="Label"),
            alt.Tooltip("chunk_count:Q", title="Chunks"),
        ],
    )
    .properties(height=220, title="Chunk counts by period")
)

st.altair_chart(bar, use_container_width=True)

# ---------------------------------------------------------------------------
# Drill-down: show individual chunks for selected period
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Drill-down — filing text")

available_periods = sorted(
    df["reference_date"].dt.strftime("%Y-%m-%d").unique(), reverse=True
)
selected_period = st.selectbox("Select period", available_periods)

chunks_df = load_chunks_for_period(ticker, selected_period, section_filter)

if chunks_df.empty:
    st.info("No chunks for this period.")
else:
    label_filter = st.radio("Show", ["All", "positive", "negative"], horizontal=True)
    if label_filter != "All":
        chunks_df = chunks_df[chunks_df.sentiment_label == label_filter]

    for _, row in chunks_df.head(10).iterrows():
        label_color = "#2ecc71" if row["sentiment_label"] == "positive" else "#e74c3c"
        with st.expander(
            f"[{row['sentiment_label'].upper()} — {row['sentiment_score']:.2f}]  "
            f"{row['section_name']}"
        ):
            st.markdown(
                f'<div style="border-left: 4px solid {label_color}; padding-left: 12px;">'
                f"{row['chunk_text']}"
                f"</div>",
                unsafe_allow_html=True,
            )
