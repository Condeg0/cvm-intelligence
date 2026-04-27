"""Financial metrics dashboard — time-series charts for extracted metrics.

Queries the SQLite database for validated financial metrics and renders
interactive Plotly charts for the selected company and metric. Values are
colour-coded by extraction match status (exact / close / mismatch).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src import config  # noqa: E402

st.set_page_config(page_title="Financial Dashboard", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_companies() -> pd.DataFrame:
    """Return all tickers and company names from the DB."""
    with sqlite3.connect(config.DB_PATH) as conn:
        return pd.read_sql("SELECT ticker, name FROM companies ORDER BY ticker", conn)


@st.cache_data(ttl=300)
def load_metrics(ticker: str, metric_name: str) -> pd.DataFrame:
    """Return time-series metrics for one ticker and one metric."""
    query = """
        SELECT
            f.ticker,
            f.filing_type,
            f.reference_date,
            m.metric_name,
            m.extracted_value,
            m.validated_value,
            m.match_status,
            m.percentage_error
        FROM metrics m
        JOIN filings f ON m.filing_id = f.filing_id
        WHERE f.ticker = ?
          AND m.metric_name = ?
        ORDER BY f.reference_date
    """
    with sqlite3.connect(config.DB_PATH) as conn:
        df = pd.read_sql(query, conn, params=(ticker, metric_name))
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    return df


@st.cache_data(ttl=300)
def load_cross_company(metric_name: str, period: str) -> pd.DataFrame:
    """Return all companies for one metric and one period."""
    query = """
        SELECT
            f.ticker,
            c.name,
            m.extracted_value,
            m.match_status,
            m.percentage_error
        FROM metrics m
        JOIN filings f ON m.filing_id = f.filing_id
        JOIN companies c ON f.ticker = c.ticker
        WHERE m.metric_name = ?
          AND f.reference_date = ?
        ORDER BY m.extracted_value DESC
    """
    with sqlite3.connect(config.DB_PATH) as conn:
        return pd.read_sql(query, conn, params=(metric_name, period))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("Financial Dashboard")
st.markdown("Extracted financial metrics for 49 B3 companies — 2022 to 2025.")

companies_df = load_companies()
ticker_options = [f"{r.ticker} — {r.name}" for r in companies_df.itertuples()]

METRIC_LABELS = {
    "revenue":              "Revenue (Receita Líquida)",
    "cogs":                 "COGS (Custo dos Produtos)",
    "gross_profit":         "Gross Profit (Resultado Bruto)",
    "net_income":           "Net Income (Lucro Líquido)",
    "total_assets":         "Total Assets (Ativo Total)",
    "total_equity":         "Total Equity (Patrimônio Líquido)",
    "operating_cash_flow":  "Operating Cash Flow (Caixa Operacional)",
}

STATUS_COLORS = {
    "exact":    "#2ecc71",
    "close":    "#f39c12",
    "mismatch": "#e74c3c",
    "missing":  "#95a5a6",
}

with st.sidebar:
    st.header("Filters")
    selected_label = st.selectbox("Company", ticker_options)
    ticker = selected_label.split(" — ")[0]

    metric_key = st.selectbox(
        "Metric",
        list(METRIC_LABELS.keys()),
        format_func=lambda k: METRIC_LABELS[k],
    )

    st.divider()
    st.caption("Match status legend")
    for status, color in STATUS_COLORS.items():
        st.markdown(
            f'<span style="color:{color}">■</span> {status}', unsafe_allow_html=True
        )

# ---------------------------------------------------------------------------
# Time-series chart
# ---------------------------------------------------------------------------

df = load_metrics(ticker, metric_key)

if df.empty:
    st.warning(f"No {metric_key} data found for {ticker}.")
else:
    company_name = companies_df.loc[companies_df.ticker == ticker, "name"].iloc[0]
    st.subheader(f"{ticker} — {company_name}")

    col_chart, col_stats = st.columns([3, 1])

    with col_chart:
        fig = go.Figure()

        for status, color in STATUS_COLORS.items():
            subset = df[df.match_status == status]
            if subset.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset["reference_date"],
                y=subset["extracted_value"],
                mode="markers+lines",
                name=status,
                marker=dict(color=color, size=8),
                line=dict(color=color, width=1.5, dash="dot"),
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    f"{METRIC_LABELS[metric_key]}: %{{y:,.0f}}<br>"
                    f"Status: {status}"
                    "<extra></extra>"
                ),
            ))

        # Overlay CSV ground truth (validated_value)
        validated = df[df.validated_value.notna()]
        if not validated.empty:
            fig.add_trace(go.Scatter(
                x=validated["reference_date"],
                y=validated["validated_value"],
                mode="markers",
                name="CSV ground truth",
                marker=dict(symbol="x", color="#2c3e50", size=10),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Ground truth: %{y:,.0f}<extra></extra>",
            ))

        fig.update_layout(
            title=f"{METRIC_LABELS[metric_key]} — {ticker}",
            xaxis_title="Period",
            yaxis_title="Value (R$)",
            legend_title="Status",
            hovermode="x unified",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.markdown("**Extraction quality**")
        total = len(df)
        for status in ["exact", "close", "mismatch", "missing"]:
            n = (df.match_status == status).sum()
            pct = 100 * n / total if total else 0
            color = STATUS_COLORS[status]
            st.markdown(
                f'<span style="color:{color}">■</span> **{status}** {n} ({pct:.0f}%)',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("**Avg % error**")
        mape = df.loc[df.match_status.isin(["exact", "close", "mismatch"]), "percentage_error"].mean()
        st.metric("MAPE", f"{mape:.2f}%" if pd.notna(mape) else "—")

        latest = df.sort_values("reference_date").iloc[-1]
        st.divider()
        st.markdown("**Latest reading**")
        st.metric(
            label=latest["reference_date"].strftime("%Y-%m-%d"),
            value=f"R$ {latest['extracted_value']:,.0f}" if pd.notna(latest["extracted_value"]) else "—",
        )

# ---------------------------------------------------------------------------
# Raw data table
# ---------------------------------------------------------------------------

with st.expander("Raw data"):
    if not df.empty:
        display = df[["reference_date", "filing_type", "extracted_value",
                       "validated_value", "match_status", "percentage_error"]].copy()
        display["reference_date"] = display["reference_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display, use_container_width=True)

# ---------------------------------------------------------------------------
# Cross-company comparison
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Cross-company comparison")

available_periods = sorted(
    pd.read_sql(
        "SELECT DISTINCT reference_date FROM filings ORDER BY reference_date DESC",
        sqlite3.connect(config.DB_PATH),
    )["reference_date"].tolist()
)

col_metric2, col_period = st.columns(2)
with col_metric2:
    compare_metric = st.selectbox(
        "Metric", list(METRIC_LABELS.keys()), format_func=lambda k: METRIC_LABELS[k],
        key="compare_metric",
    )
with col_period:
    compare_period = st.selectbox("Period", available_periods, key="compare_period")

cross_df = load_cross_company(compare_metric, compare_period)

if cross_df.empty:
    st.info("No data for this combination.")
else:
    bar_fig = px.bar(
        cross_df.head(30),
        x="ticker",
        y="extracted_value",
        color="match_status",
        color_discrete_map=STATUS_COLORS,
        title=f"{METRIC_LABELS[compare_metric]} — {compare_period} (top 30 by value)",
        labels={"extracted_value": "Value (R$)", "ticker": "Company"},
        hover_data=["name", "percentage_error"],
    )
    bar_fig.update_layout(height=380)
    st.plotly_chart(bar_fig, use_container_width=True)

    st.dataframe(
        cross_df[["ticker", "name", "extracted_value", "match_status", "percentage_error"]],
        use_container_width=True,
    )
