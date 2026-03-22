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
        Hourly bucket DataFrame with columns ``wallet``, ``time_col``, ``trade_pnl``.
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
    period: str = "both",
    title: str = "Wallet selection — cohort cumulative PnL over time",
    bucket_freq: str = "1h",
) -> go.Figure:
    """Single-panel aggregate PnL figure — one line per cohort.

    Each line shows the cumulative sum of ``trade_pnl`` across **all** wallets
    in that cohort.  In ``"both"`` mode the test-period portion is reset to
    start from zero at the split boundary (same as the train portion).

    Parameters
    ----------
    df_fills:
        Fill-level trade DataFrame.  Must contain at least: ``wallet``, ``dt``,
        ``trade_pnl``, ``is_train``.
    wallet_cohorts:
        ``{cohort_name → DataFrame(wallet, wallet_quality)}`` as produced by
        :func:`~wallet_selection.selector.build_wallet_cohorts`.
    split_date:
        Train/test boundary timestamp.  Derived from ``df_fills`` when omitted.
    period:
        Which portion of the data to plot.  One of:

        * ``"train"``  — only rows where ``dt < split_date``; cumulative PnL
          starts from zero at the first training bucket.
        * ``"test"``   — only rows where ``dt >= split_date``; cumulative PnL
          starts from zero at the first test bucket.
        * ``"both"``   — all rows; test portion is reset to start from zero;
          a vertical dashed train/test split line is drawn.

        Defaults to ``"both"``.
    title:
        Figure title.
    bucket_freq:
        Pandas offset alias for time bucketing (default ``'1D'`` = daily).

    Returns
    -------
    ``go.Figure`` with a single cohort-aggregate panel.
    """
    if period not in ("train", "test", "both"):
        raise ValueError(f"period must be 'train', 'test', or 'both'; got {period!r}")

    # ── derive split_date from data if not supplied ──────────────────────────
    if split_date is None:
        tmp = df_fills[df_fills["is_train"]]
        if not tmp.empty:
            split_date = pd.Timestamp(tmp["dt"].max()).normalize() + pd.Timedelta(days=1)

    # ── bucket fills to bucket_freq per wallet ───────────────────────────────
    df = df_fills.copy()
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    df["bucket"] = df["dt"].dt.floor(bucket_freq)

    # Filter to the requested period before building aggregates
    if period == "train" and split_date is not None:
        df = df[df["bucket"] < split_date]
    elif period == "test" and split_date is not None:
        df = df[df["bucket"] >= split_date]

    all_wallets = list({w for c in wallet_cohorts.values() for w in c["wallet"]})
    df_sel = df[df["wallet"].isin(all_wallets)][["wallet", "bucket", "trade_pnl"]].copy()

    daily = (
        df_sel.groupby(["wallet", "bucket"], sort=True)["trade_pnl"]
        .sum()
        .reset_index()
    )

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

        agg_df = (
            daily[daily["wallet"].isin(wallets_in_cohort)]
            .groupby("bucket", sort=True)["trade_pnl"]
            .sum()
            .reset_index()
        )
        agg_df["cum_pnl"] = agg_df["trade_pnl"].cumsum()

        if period == "both" and split_date is not None and not agg_df.empty:
            # Reset test-period cumulation to start from 0
            pre_split = agg_df.loc[agg_df["bucket"] < split_date, "cum_pnl"]
            split_offset = float(pre_split.iloc[-1]) if not pre_split.empty else 0.0
            post = agg_df["bucket"] >= split_date
            agg_df.loc[post, "cum_pnl"] = agg_df.loc[post, "cum_pnl"] - split_offset

        if not agg_df.empty:
            # Prepend an explicit (anchor_time, 0) point so the line always
            # starts at zero regardless of the PnL in the first bucket.
            anchor_time = split_date if (period == "test" and split_date is not None) else agg_df["bucket"].iloc[0]
            zero_row = pd.DataFrame({"bucket": [anchor_time], "cum_pnl": [0.0]})
            plot_df = pd.concat([zero_row, agg_df[["bucket", "cum_pnl"]]], ignore_index=True)

            fig.add_trace(
                go.Scatter(
                    x=plot_df["bucket"],
                    y=plot_df["cum_pnl"],
                    mode="lines",
                    line={"color": color, "width": 2},
                    name=cohort_name,
                    hovertemplate=(
                        f"{cohort_name}<br>%{{x|%Y-%m-%d %H:%M}}<br>"
                        "cum PnL: %{y:.1f} USDC<extra></extra>"
                    ),
                )
            )

    # ── split-date vline in "both" mode ─────────────────────────────────────
    if period == "both" and split_date is not None:
        fig.add_vline(
            x=split_date,
            line_dash="dash",
            line_color="black",
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
        height=450,
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
    split_date: pd.Timestamp | None = None,
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
    split_date:
        Train/test boundary timestamp.  Derived from ``df_fills`` when omitted.
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
    # ── derive split_date ────────────────────────────────────────────────────
    if split_date is None:
        tmp = df_fills[df_fills["is_train"]]
        if not tmp.empty:
            split_date = pd.Timestamp(tmp["dt"].max()).normalize() + pd.Timedelta(days=1)

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

        # Reset test-period cumulation to start from 0 at the split boundary
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
            ind_df.loc[post, "cum_pnl"] = (
                ind_df.loc[post, "cum_pnl"] - ind_df.loc[post, "split_offset"]
            )

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

    # ── split-date vline ─────────────────────────────────────────────────────
    if split_date is not None:
        fig.add_vline(
            x=split_date,
            line_dash="dash",
            line_color="black",
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
        height=600,
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative PnL (USDC)",
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
    plot_df["cumulative_pnl"] = plot_df["trade_pnl"].cumsum()

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
