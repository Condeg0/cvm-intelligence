"""Reusable Plotly and Altair chart components for the Streamlit app.

Import these functions in page modules instead of building charts inline.
"""

from __future__ import annotations

import pandas as pd


def metric_time_series(df: pd.DataFrame, metric: str, ticker: str):
    """Render a Plotly line chart for a financial metric over time.

    Args:
        df: DataFrame with columns: ``reference_date``, ``validated_value``.
        metric: Metric name for chart title (e.g. ``"revenue"``).
        ticker: Company ticker for chart title.

    Returns:
        A ``plotly.graph_objects.Figure`` instance.
    """
    raise NotImplementedError


def sentiment_bar_chart(df: pd.DataFrame, ticker: str):
    """Render an Altair stacked bar chart of sentiment distribution over periods.

    Args:
        df: DataFrame with columns: ``reference_date``, ``sentiment_label``,
            ``count``.
        ticker: Company ticker for chart title.

    Returns:
        An ``altair.Chart`` instance.
    """
    raise NotImplementedError
