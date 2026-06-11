"""Reusable Plotly and Altair chart components for the Streamlit app.

Import these functions in page modules instead of building charts inline.
"""

from __future__ import annotations

import pandas as pd
import altair as alt
import plotly.graph_objects as go


STATUS_COLORS: dict[str, str] = {
    "exact":    "#2ecc71",
    "close":    "#f39c12",
    "mismatch": "#e74c3c",
    "missing":  "#95a5a6",
}

SENTIMENT_COLORS: dict[str, str] = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
}


def metric_time_series(df: pd.DataFrame, metric: str, ticker: str) -> go.Figure:
    """Render a Plotly line chart for a financial metric over time.

    Args:
        df: DataFrame with columns: ``reference_date``, ``extracted_value``,
            ``validated_value``, ``match_status``.
        metric: Metric name for chart title (e.g. ``"revenue"``).
        ticker: Company ticker for chart title.

    Returns:
        A ``plotly.graph_objects.Figure`` instance.
    """
    fig = go.Figure()

    for status, color in STATUS_COLORS.items():
        subset = df[df["match_status"] == status]
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
                f"{metric}: %{{y:,.0f}}<br>"
                f"Status: {status}"
                "<extra></extra>"
            ),
        ))

    validated = df[df["validated_value"].notna()]
    if not validated.empty:
        fig.add_trace(go.Scatter(
            x=validated["reference_date"],
            y=validated["validated_value"],
            mode="markers",
            name="CSV ground truth",
            marker=dict(symbol="x", color="#2c3e50", size=10),
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Ground truth: %{y:,.0f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=f"{metric} — {ticker}",
        xaxis_title="Period",
        yaxis_title="Value (R$)",
        legend_title="Status",
        hovermode="x unified",
        height=420,
    )
    return fig


def sentiment_bar_chart(df: pd.DataFrame, ticker: str) -> alt.Chart:
    """Render an Altair stacked bar chart of sentiment distribution over periods.

    Args:
        df: DataFrame with columns: ``reference_date``, ``sentiment_label``,
            ``count``.
        ticker: Company ticker for chart title.

    Returns:
        An ``altair.Chart`` instance.
    """
    plot_df = df.copy()
    plot_df["period_str"] = pd.to_datetime(plot_df["reference_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    sorted_periods = sorted(plot_df["period_str"].unique())

    color_scale = alt.Scale(
        domain=list(SENTIMENT_COLORS.keys()),
        range=list(SENTIMENT_COLORS.values()),
    )

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("period_str:N", title="Period", sort=sorted_periods),
            y=alt.Y("count:Q", title="Chunk count"),
            color=alt.Color(
                "sentiment_label:N", scale=color_scale, title="Sentiment"
            ),
            tooltip=[
                alt.Tooltip("period_str:N", title="Period"),
                alt.Tooltip("sentiment_label:N", title="Label"),
                alt.Tooltip("count:Q", title="Chunks"),
            ],
        )
        .properties(height=220, title=f"Chunk counts by period — {ticker}")
    )
    return chart
