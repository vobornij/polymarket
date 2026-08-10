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
    title: str = "Cumulative PnL Over Time by Wallet (train + test)",
    time_col: str = "dt_floored",
) -> go.Figure:
    """Line chart of per-wallet cumulative PnL over time.

    Parameters
    ----------
    buckets_full:
        Hourly bucket DataFrame with columns ``wallet``, ``time_col``, ``trade_pnl``.
    top_wallets:
        List of wallet addresses to include (e.g. top 20 by training PnL).
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
    plot_df["cumulative_pnl"] = plot_df.groupby("wallet")["trade_pnl"].cumsum()

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
    # Removed split_date handling as it is no longer a parameter
    return fig


def plot_wallet_selection_pnl(
    df_fills: pd.DataFrame,
    wallet_cohorts: dict[str, pd.DataFrame],
    *,
    period: str = "both",
    title: str = "Wallet selection — cohort cumulative PnL over time",
    bucket_freq: str = "1h",
    pnl_cols: list[str] | None = None,
    max_exposure_per_wallet: float = 100,
) -> go.Figure:
    """Single-panel aggregate PnL figure — one line per cohort.

    Each line shows the cumulative sum of ``trade_pnl`` across **all** wallets
    in that cohort.  For each column in ``pnl_cols`` an exposure-limited
    BUY-only cumulative line is drawn (the copyable strategy PnL capped at
    ``max_exposure_per_wallet`` per wallet).

    Parameters
    ----------
    df_fills:
        Fill-level trade DataFrame.  Must contain at least: ``wallet``, ``dt``,
        ``trade_pnl``, ``is_train``, ``side``, plus each ``pnl_col`` and a
        matching ``{col}_exposure`` column.
    wallet_cohorts:
        ``{cohort_name → DataFrame(wallet, wallet_quality)}`` as produced by
        :func:`~wallet_selection.selector.build_wallet_cohorts`.
    period:
        Which portion of the data to plot.  One of ``'train'``, ``'test'``,
        ``'both'``.
    title:
        Figure title.
    bucket_freq:
        Pandas offset alias for time bucketing (default ``'1D'`` = daily).
    pnl_cols:
        Copyable PnL columns to draw as exposure-limited cumulative lines
        (default ``['copyable_pnl']``).
    max_exposure_per_wallet:
        Capital budget per wallet used for the exposure-limited lines.

    Returns
    -------
    ``go.Figure`` with a single cohort-aggregate panel.
    """
    if period not in ("train", "test", "both"):
        raise ValueError(f"period must be 'train', 'test', or 'both'; got {period!r}")
    pnl_cols = list(pnl_cols) if pnl_cols else ["copyable_pnl"]
    exposure_cols = [f"{c}_exposure" for c in pnl_cols]

    # ── bucket fills to bucket_freq per wallet ───────────────────────────────
    df = df_fills.copy()
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    df["bucket"] = df["dt"].dt.floor(bucket_freq)

    # Filter to the requested period before building aggregates
    if period == "train":
        df = df[df["is_train"] == True]
    elif period == "test":
        df = df[df["is_train"] == False]
    # period == "both": keep all rows

    all_wallets = list({w for c in wallet_cohorts.values() for w in c["wallet"]})
    df_sel = df[df["wallet"].isin(all_wallets)][
        ["wallet", "bucket", "trade_pnl", "side", *pnl_cols, *exposure_cols]
    ].copy()

    # ── colour palette — one colour per cohort ───────────────────────────────
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    cohort_names = list(wallet_cohorts.keys())
    cohort_color = {name: palette[i % len(palette)] for i, name in enumerate(cohort_names)}

    fig = go.Figure()

    for cohort_name, cohort_df in wallet_cohorts.items():
        color = cohort_color[cohort_name]
        wallets_in_cohort = set(cohort_df["wallet"])
        cohort_sel = df_sel[df_sel["wallet"].isin(wallets_in_cohort)]

        agg_df = (
            cohort_sel.groupby("bucket", sort=True)[["trade_pnl", *pnl_cols]]
            .sum()
            .reset_index()
        )
        agg_df["cum_pnl"] = agg_df["trade_pnl"].cumsum()

        if agg_df.empty:
            continue

        # Total PnL (solid line)
        fig.add_trace(
            go.Scatter(
                x=agg_df["bucket"],
                y=agg_df["cum_pnl"],
                mode="lines",
                line={"color": color, "width": 2, "dash": "solid"},
                name=f"{cohort_name} (total)",
                hovertemplate=(
                    f"{cohort_name} (total)<br>%{{x|%Y-%m-%d %H:%M}}<br>"
                    "cum PnL: %{y:.1f} USDC<extra></extra>"
                ),
            )
        )

        # Exposure-limited BUY-only line per copyable PnL variant
        cohort_sel_buy = cohort_sel[cohort_sel["side"] == "BUY"]
        for i, c in enumerate(pnl_cols):
            wallet_df = (
                cohort_sel_buy.groupby(["wallet", "bucket"], sort=True)[[c, f"{c}_exposure"]]
                .sum()
                .reset_index()
            )
            scale = np.minimum(
                1, max_exposure_per_wallet / wallet_df[f"{c}_exposure"].replace(0, np.nan)
            )
            bucket_lim_df = (
                wallet_df.assign(**{f"{c}_limited": wallet_df[c] * scale})
                .groupby("bucket", sort=True)[f"{c}_limited"]
                .sum()
                .reset_index()
            )
            bucket_lim_df[f"cum_{c}_limited"] = bucket_lim_df[f"{c}_limited"].cumsum()

            fig.add_trace(
                go.Scatter(
                    x=bucket_lim_df["bucket"],
                    y=bucket_lim_df[f"cum_{c}_limited"],
                    mode="lines",
                    line={"color": color, "width": 2, "dash": "dot" if i == 0 else "dash"},
                    name=f"{cohort_name} ({c}, exposure-limited only BUY)",
                    hovertemplate=(
                        f"{cohort_name} ({c}, exposure-limited)<br>%{{x|%Y-%m-%d %H:%M}}<br>"
                        "cum Copyable PnL (exposure-limited): %{y:.1f} USDC<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        template="plotly_white",
        height=750,
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative PnL (USDC)",
        legend_title="Cohort",
    )
    return fig


def plot_wallet_individual_pnl(
    df_fills: pd.DataFrame,
    wallet_cohorts: dict[str, pd.DataFrame],
    *,
    top_n_individual: int = 20,
    title: str = "Individual wallet cumulative PnL (train + test)",
    bucket_freq: str = "1h",
) -> go.Figure:
    """Per-wallet cumulative PnL lines spanning train **and** test periods.

    Each wallet is shown as a thin line coloured by cohort membership.  The
    train/test boundary is marked with a vertical dashed line.  Wallet address
    labels are drawn at the right-hand end of each line.

    The test-period portion of each wallet's cumulative PnL is reset to start
    from zero at the split boundary (so train and test performance are visually
    independent).

    Parameters
    ----------
    df_fills:
        Fill-level trade DataFrame.  Must contain at least: ``wallet``, ``dt``,
        ``trade_pnl``, ``is_train``.
    wallet_cohorts:
        ``{cohort_name → DataFrame(wallet, wallet_quality)}``.
    top_n_individual:
        Number of top wallets per cohort (ranked by training PnL) to display.
    title:
        Figure title.
    bucket_freq:
        Pandas offset alias for time bucketing (default ``'1D'`` = daily).

    Returns
    -------
    ``go.Figure``.
    """
    # ── bucket all data ──────────────────────────────────────────────────────
    df = df_fills.copy()
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    df["bucket"] = df["dt"].dt.floor(bucket_freq)

    all_wallets = list({w for c in wallet_cohorts.values() for w in c["wallet"]})
    df_sel = df[df["wallet"].isin(all_wallets)][["wallet", "bucket", "trade_pnl"]].copy()

    daily = (
        df_sel.groupby(["wallet", "bucket"], sort=True)["trade_pnl"]
        .sum()
        .reset_index()
    )

    # ── rank wallets by training PnL ─────────────────────────────────────────
    train_mask = df_fills["is_train"] & df_fills["wallet"].isin(all_wallets)
    train_pnl = (
        df_fills[train_mask]
        .assign(dt=lambda d: pd.to_datetime(d["dt"], utc=True))
        .groupby("wallet")["trade_pnl"]
        .sum()
        .rename("train_pnl")
    )

    # ── colour palette ───────────────────────────────────────────────────────
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    cohort_names = list(wallet_cohorts.keys())
    cohort_color = {name: palette[i % len(palette)] for i, name in enumerate(cohort_names)}

    fig = go.Figure()
    legend_shown: set[str] = set()

    for cohort_name, cohort_df in wallet_cohorts.items():
        color = cohort_color[cohort_name]
        wallets_in_cohort = set(cohort_df["wallet"])

        ranked = (
            train_pnl[train_pnl.index.isin(wallets_in_cohort)]
            .sort_values(ascending=False)
        )
        top_wallets = list(ranked.head(top_n_individual).index)

        ind_df = (
            daily[daily["wallet"].isin(top_wallets)]
            .sort_values(["wallet", "bucket"])
            .copy()
        )
        ind_df["cum_pnl"] = ind_df.groupby("wallet")["trade_pnl"].cumsum()

        for wallet in top_wallets:
            w_df = ind_df[ind_df["wallet"] == wallet].copy()
            if w_df.empty:
                continue
            short = wallet[:6] + "…" + wallet[-4:]
            show_legend = cohort_name not in legend_shown
            if show_legend:
                legend_shown.add(cohort_name)

            # Line trace
            fig.add_trace(
                go.Scatter(
                    x=w_df["bucket"],
                    y=w_df["cum_pnl"],
                    mode="lines",
                    line={"color": color, "width": 1},
                    opacity=0.6,
                    name=cohort_name,
                    legendgroup=cohort_name,
                    showlegend=show_legend,
                    hovertemplate=(
                        f"{short} ({cohort_name})<br>%{{x|%Y-%m-%d}}<br>"
                        "cum PnL: %{y:.1f} USDC<extra></extra>"
                    ),
                )
            )

            # Label at the right end of the line
            last_row = w_df.iloc[-1]
            fig.add_annotation(
                x=last_row["bucket"],
                y=last_row["cum_pnl"],
                text=short,
                showarrow=False,
                xanchor="left",
                font={"size": 8, "color": color},
            )

    fig.update_layout(
        template="plotly_white",
        height=900,
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative PnL (USDC)",
        legend_title="Cohort",
        yaxis=dict(type="log", range=[0, None]),
    )
    return fig


def plot_combined_cumulative_pnl(
    buckets_full: pd.DataFrame,
    wallet_set: set[str],
    *,
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
    plot_df["cumulative_pnl"] = plot_df["trade_pnl"].cumsum()

    fig = px.line(
        plot_df,
        x=time_col,
        y="cumulative_pnl",
        title=title,
        labels={time_col: "Time", "cumulative_pnl": "Cumulative PnL (USDC)"},
    )
    # Removed split_date handling as it is no longer a parameter
    return fig
