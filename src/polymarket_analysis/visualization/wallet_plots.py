"""
Wallet PnL visualization functions.

All plot functions return a ``plotly.graph_objects.Figure`` so the caller
can further customise or call ``.show(renderer="browser")``.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def plot_wallet_pnl_bars(
    comparison: pd.DataFrame,
    *,
    title: str = "Train vs Test PnL per wallet",
) -> go.Figure:
    """Grouped bar chart comparing train and test PnL per wallet.

    Parameters
    ----------
    comparison:
        DataFrame with columns ``wallet_short``, ``total_pnl_train``,
        ``total_pnl_test``.  Typically produced by merging the train and
        test wallet-metric frames.
    title:
        Figure title.
    """
    fig = go.Figure([
        go.Bar(
            name="Train PnL",
            x=comparison["wallet_short"],
            y=comparison["total_pnl_train"],
            marker_color="steelblue",
        ),
        go.Bar(
            name="Test PnL",
            x=comparison["wallet_short"],
            y=comparison["total_pnl_test"],
            marker_color="darkorange",
        ),
    ])
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Wallet",
        yaxis_title="PnL (USDC)",
        xaxis_tickangle=-45,
        legend_title="Period",
    )
    return fig


def plot_wallet_returns(
    comparison: pd.DataFrame,
    *,
    title: str = "Train vs Test return (PnL / notional) per wallet",
) -> go.Figure:
    """Grouped bar chart comparing train and test return per wallet.

    Parameters
    ----------
    comparison:
        DataFrame with columns ``wallet_short``, ``return_train``,
        ``return_test``.
    title:
        Figure title.
    """
    fig = go.Figure([
        go.Bar(
            name="Train return",
            x=comparison["wallet_short"],
            y=comparison["return_train"],
            marker_color="steelblue",
        ),
        go.Bar(
            name="Test return",
            x=comparison["wallet_short"],
            y=comparison["return_test"],
            marker_color="darkorange",
        ),
    ])
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Wallet",
        yaxis_title="Return",
        xaxis_tickangle=-45,
        legend_title="Period",
    )
    return fig


def plot_cumulative_pnl_by_wallet(
    buckets_full: pd.DataFrame,
    top_wallets: list[str],
    *,
    split_date: pd.Timestamp | None = None,
    title: str = "Cumulative PnL Over Time by Wallet (train + test)",
    time_col: str = "dt_floored",
) -> go.Figure:
    """Line chart of per-wallet cumulative PnL over time.

    Parameters
    ----------
    buckets_full:
        Hourly bucket DataFrame with columns ``wallet``, ``time_col``, ``pnl``.
    top_wallets:
        List of wallet addresses to include (e.g. top 20 by training PnL).
    split_date:
        If provided, a vertical dashed line is drawn at this timestamp.
    title:
        Figure title.
    time_col:
        Name of the time column in ``buckets_full``.
    """
    plot_df = (
        buckets_full[buckets_full["wallet"].isin(top_wallets)]
        .sort_values(["wallet", time_col])
        .copy()
    )
    plot_df["cumulative_pnl"] = plot_df.groupby("wallet")["pnl"].cumsum()

    fig = px.line(
        plot_df,
        x=time_col,
        y="cumulative_pnl",
        color="wallet",
        title=title,
        labels={
            time_col: "Time",
            "cumulative_pnl": "Cumulative PnL (USDC)",
            "wallet": "Wallet",
        },
    )
    if split_date is not None:
        fig.add_vline(
            x=split_date.timestamp() * 1000,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Train / Test split ({split_date.date()})",
            annotation_position="top left",
        )
    return fig


def plot_combined_cumulative_pnl(
    buckets_full: pd.DataFrame,
    wallet_set: set[str],
    *,
    split_date: pd.Timestamp | None = None,
    title: str = "Cumulative PnL Over Time (All Best Wallets, train + test)",
    time_col: str = "dt_floored",
) -> go.Figure:
    """Line chart of combined cumulative PnL across all wallets in *wallet_set*.

    Parameters
    ----------
    buckets_full:
        Hourly bucket DataFrame with columns ``wallet``, ``time_col``, ``pnl``.
    wallet_set:
        Set of wallet addresses to aggregate.
    split_date:
        If provided, a vertical dashed line is drawn at this timestamp.
    title:
        Figure title.
    time_col:
        Name of the time column in ``buckets_full``.
    """
    plot_df = (
        buckets_full[buckets_full["wallet"].isin(wallet_set)]
        .sort_values(time_col)
        .copy()
    )
    plot_df["cumulative_pnl"] = plot_df["pnl"].cumsum()

    fig = px.line(
        plot_df,
        x=time_col,
        y="cumulative_pnl",
        title=title,
        labels={time_col: "Time", "cumulative_pnl": "Cumulative PnL (USDC)"},
    )
    if split_date is not None:
        fig.add_vline(
            x=split_date.timestamp() * 1000,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Train / Test split ({split_date.date()})",
            annotation_position="top left",
        )
    return fig
