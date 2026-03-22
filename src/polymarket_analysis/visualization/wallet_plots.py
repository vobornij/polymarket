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


def plot_wallet_selection_pnl(
    df_fills: pd.DataFrame,
    wallet_cohorts: dict[str, pd.DataFrame],
    *,
    split_date: pd.Timestamp | None = None,
    top_n_individual: int = 20,
    title: str = "Wallet selection — cumulative PnL over time",
    bucket_freq: str = "1D",
) -> go.Figure:
    """Two-panel figure for all wallet-selection cohorts.

    Panel 1 — individual lines for the top-*top_n_individual* wallets of each
    cohort, coloured by cohort.  Useful for spotting a few stars vs. the field.

    Panel 2 — one aggregate cumulative PnL line per cohort (sum of all wallets
    in that cohort).  The train/test split is marked with a vertical dashed line
    in both panels.

    Parameters
    ----------
    df_fills:
        Fill-level trade DataFrame.  Must contain at least: ``wallet``, ``dt``,
        ``trade_pnl``, ``is_train``.
    wallet_cohorts:
        ``{cohort_name → DataFrame(wallet, wallet_quality)}`` as produced by
        :func:`~wallet_selection.selector.build_wallet_cohorts`.  An optional
        extra key ``"volatility"`` is handled the same way.
    split_date:
        Train/test boundary timestamp.  Derived from ``df_fills`` when omitted.
    top_n_individual:
        How many top wallets (by total PnL in training) to show in panel 1.
    title:
        Figure title.
    bucket_freq:
        Pandas offset alias for time bucketing (default ``'1D'`` = daily).

    Returns
    -------
    ``go.Figure`` with two sub-plots.
    """
    from plotly.subplots import make_subplots

    # ── derive split_date from data if not supplied ──────────────────────────
    if split_date is None:
        tmp = df_fills[df_fills["is_train"]]
        if not tmp.empty:
            split_date = pd.Timestamp(tmp["dt"].max()).normalize() + pd.Timedelta(days=1)

    # ── bucket fills to bucket_freq per wallet ───────────────────────────────
    df = df_fills.copy()
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    df["bucket"] = df["dt"].dt.floor(bucket_freq)

    all_wallets = list({w for c in wallet_cohorts.values() for w in c["wallet"]})
    df_sel = df[df["wallet"].isin(all_wallets)][["wallet", "bucket", "pnl"]].copy()

    daily = (
        df_sel.groupby(["wallet", "bucket"], sort=True)["pnl"]
        .sum()
        .reset_index()
    )

    # ── pre-compute per-wallet total train PnL for ranking ───────────────────
    train_pnl = (
        df[df["is_train"] & df["wallet"].isin(all_wallets)]
        .groupby("wallet")["pnl"]
        .sum()
        .rename("train_pnl")
    )

    # ── colour palette — one colour per cohort ───────────────────────────────
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    cohort_names = list(wallet_cohorts.keys())
    cohort_color = {name: palette[i % len(palette)] for i, name in enumerate(cohort_names)}

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=["Individual top wallets", "Cohort total PnL"],
    )

    for cohort_name, cohort_df in wallet_cohorts.items():
        color = cohort_color[cohort_name]
        wallets_in_cohort = set(cohort_df["wallet"])

        # rank by train PnL, pick top_n_individual
        ranked = (
            train_pnl[train_pnl.index.isin(wallets_in_cohort)]
            .sort_values(ascending=False)
        )
        top_wallets = list(ranked.head(top_n_individual).index)

        # ── panel 1: individual lines ─────────────────────────────────────
        ind_df = (
            daily[daily["wallet"].isin(top_wallets)]
            .sort_values(["wallet", "bucket"])
            .copy()
        )
        ind_df["cum_pnl"] = ind_df.groupby("wallet")["pnl"].cumsum()
        # post-split: restart cumulation from 0 (pre-split segment untouched)
        if split_date is not None:
            split_offset = (
                ind_df[ind_df["bucket"] < split_date]
                .groupby("wallet")["cum_pnl"]
                .last()
                .rename("split_offset")
            )
            ind_df = ind_df.join(split_offset, on="wallet")
            ind_df["split_offset"] = ind_df["split_offset"].fillna(0.0)
            post = ind_df["bucket"] >= split_date
            ind_df.loc[post, "cum_pnl"] = ind_df.loc[post, "cum_pnl"] - ind_df.loc[post, "split_offset"]

        show_legend = True
        for wallet in top_wallets:
            w_df = ind_df[ind_df["wallet"] == wallet]
            if w_df.empty:
                continue
            short = wallet[:6] + "…" + wallet[-4:]
            fig.add_trace(
                go.Scatter(
                    x=w_df["bucket"],
                    y=w_df["cum_pnl"],
                    mode="lines",
                    line={"color": color, "width": 1},
                    opacity=0.55,
                    name=cohort_name,
                    legendgroup=cohort_name,
                    showlegend=show_legend,
                    hovertemplate=f"{short}<br>%{{x|%Y-%m-%d}}<br>cum PnL: %{{y:.1f}} USDC<extra></extra>",
                ),
                row=1,
                col=1,
            )
            show_legend = False  # only first trace shows in legend

        # ── panel 2: cohort aggregate ─────────────────────────────────────
        agg_df = (
            daily[daily["wallet"].isin(wallets_in_cohort)]
            .groupby("bucket", sort=True)["pnl"]
            .sum()
            .reset_index()
        )
        agg_df["cum_pnl"] = agg_df["pnl"].cumsum()
        # post-split: restart cumulation from 0 (pre-split segment untouched)
        if split_date is not None and not agg_df.empty:
            pre_split = agg_df.loc[agg_df["bucket"] < split_date, "cum_pnl"]
            split_offset = pre_split.iloc[-1] if not pre_split.empty else 0.0
            post = agg_df["bucket"] >= split_date
            agg_df.loc[post, "cum_pnl"] = agg_df.loc[post, "cum_pnl"] - split_offset
        if not agg_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=agg_df["bucket"],
                    y=agg_df["cum_pnl"],
                    mode="lines",
                    line={"color": color, "width": 2},
                    name=cohort_name,
                    legendgroup=cohort_name,
                    showlegend=False,
                    hovertemplate=f"{cohort_name}<br>%{{x|%Y-%m-%d}}<br>cum PnL: %{{y:.1f}} USDC<extra></extra>",
                ),
                row=2,
                col=1,
            )

    # ── split-date vlines ────────────────────────────────────────────────────
    if split_date is not None:
        for panel_row in (1, 2):
            fig.add_vline(
                x=split_date,
                line_dash="dash",
                line_color="black",
                row=panel_row,
                col=1,
            )
        fig.add_annotation(
            x=split_date,
            y=1.01,
            yref="paper",
            text="train / test split",
            showarrow=False,
            font={"size": 11},
        )

    fig.update_layout(
        template="plotly_white",
        height=750,
        title=title,
        yaxis_title="Cumulative PnL (USDC)",
        yaxis2_title="Cumulative PnL (USDC)",
        legend_title="Cohort",
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
