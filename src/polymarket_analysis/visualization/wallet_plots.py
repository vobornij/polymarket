"""
Wallet PnL visualization functions.

All plot functions return a ``plotly.graph_objects.Figure`` so the caller
can further customise or call ``.show(renderer="browser")``.
"""

from __future__ import annotations

import pandas as pd
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


def copyable_exposure_series(
    fills: pd.DataFrame,
    exp_col: str,
    *,
    release_delay: pd.Timedelta = pd.Timedelta(days=1),
    bucket_freq: str | None = None,
) -> pd.DataFrame:
    """Cumulative copy-exposure time series for one copyable variant.

    Exposure is the capital tied up replicating a variant's BUY fills
    (``exp_col`` = price × copyable qty).  It is added at each BUY fill time
    and released one day after the contract's last trade
    (``last_condition_trade_ts``), matching the ``trade_signals`` exposure
    model.  With an unlimited budget the exposure is unbounded and uncapped.

    Parameters
    ----------
    fills:
        Fill-level DataFrame with ``dt``, ``side``, ``last_condition_trade_ts``
        and ``exp_col``.
    exp_col:
        Per-fill exposure column (price × copyable qty).
    release_delay:
        Delay after ``last_condition_trade_ts`` before exposure is released.
    bucket_freq:
        Optional pandas offset alias to resample the exposure to (last value
        per bucket).

    Returns
    -------
    DataFrame with columns ``dt``, ``exposure``.
    """
    buy = fills[
        (fills["side"] == "BUY")
        & fills[exp_col].notna()
        & (fills[exp_col] > 0)
    ].copy()
    if buy.empty:
        return pd.DataFrame(columns=["dt", "exposure"])

    add = buy[["dt", exp_col]].rename(columns={exp_col: "exposure_delta"})
    release = buy[["last_condition_trade_ts", exp_col]].rename(
        columns={"last_condition_trade_ts": "dt", exp_col: "exposure_delta"}
    )
    release["dt"] = pd.to_datetime(release["dt"], utc=True) + release_delay
    release["exposure_delta"] = -release["exposure_delta"]

    events = pd.concat([add, release], ignore_index=True).sort_values("dt")
    events["exposure"] = events["exposure_delta"].cumsum()
    out = events[["dt", "exposure"]]

    if bucket_freq is not None:
        out = (
            out.set_index("dt")["exposure"]
            .resample(bucket_freq)
            .last()
            .dropna()
            .reset_index()
        )
    return out


def plot_wallet_selection_pnl(
    df_fills: pd.DataFrame,
    wallet_cohorts: dict[str, pd.DataFrame],
    *,
    period: str = "both",
    title: str = "Wallet selection — cohort cumulative PnL over time",
    bucket_freq: str = "1h",
    pnl_cols: list[str] | None = None,
    plot_exposure: bool = True,
) -> go.Figure:
    """Single-panel aggregate PnL figure — one line per cohort.

    Each line shows the cumulative sum of ``trade_pnl`` across **all** wallets
    in that cohort.  For each column in ``pnl_cols`` an **unlimited-budget**
    cumulative copyable-PnL line is drawn (raw, uncapped, BUY + SELL).  When
    ``plot_exposure`` is True the capital tied up copying that variant is drawn
    on a secondary y-axis (BUY exposure added at fill time, released one day
    after the contract's last trade).

    Parameters
    ----------
    df_fills:
        Fill-level trade DataFrame.  Must contain at least: ``wallet``, ``dt``,
        ``trade_pnl``, ``is_train``, ``side``, ``last_condition_trade_ts``,
        plus each ``pnl_col`` and a matching ``{col}_exposure`` column.
    wallet_cohorts:
        ``{cohort_name → DataFrame(wallet, wallet_quality)}`` as produced by
        :func:`~wallet_selection.selector.build_wallet_cohorts`.
    period:
        Which portion of the data to plot.  One of ``'train'``, ``'test'``,
        ``'both'``.
    title:
        Figure title.
    bucket_freq:
        Pandas offset alias for time bucketing (default ``'1h'``).
    pnl_cols:
        Copyable PnL columns to draw as unlimited cumulative lines
        (default ``['copyable_pnl']``).
    plot_exposure:
        Whether to draw exposure lines on a secondary y-axis.

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
    df["last_condition_trade_ts"] = pd.to_datetime(df["last_condition_trade_ts"], utc=True)
    df["bucket"] = df["dt"].dt.floor(bucket_freq)

    # Filter to the requested period before building aggregates
    if period == "train":
        df = df[df["is_train"] == True]
    elif period == "test":
        df = df[df["is_train"] == False]
    # period == "both": keep all rows

    all_wallets = list({w for c in wallet_cohorts.values() for w in c["wallet"]})
    df_sel = df[df["wallet"].isin(all_wallets)][
        ["wallet", "bucket", "dt", "trade_pnl", "side", "last_condition_trade_ts",
         *pnl_cols, *exposure_cols]
    ].copy()

    # ── colour palette — one colour per cohort ───────────────────────────────
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    cohort_names = list(wallet_cohorts.keys())
    cohort_color = {name: palette[i % len(palette)] for i, name in enumerate(cohort_names)}

    fig = go.Figure()
    fig.update_layout(
        yaxis2=dict(
            title="Exposure (USDC)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
    )

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

        # Unlimited-budget copyable PnL line per variant
        for i, c in enumerate(pnl_cols):
            bucket_c = agg_df[["bucket", c]].copy()
            bucket_c[f"cum_{c}"] = bucket_c[c].cumsum()

            fig.add_trace(
                go.Scatter(
                    x=bucket_c["bucket"],
                    y=bucket_c[f"cum_{c}"],
                    mode="lines",
                    line={"color": color, "width": 2, "dash": "dot" if i == 0 else "dash"},
                    name=f"{cohort_name} ({c}, raw)",
                    hovertemplate=(
                        f"{cohort_name} ({c})<br>%{{x|%Y-%m-%d %H:%M}}<br>"
                        "cum Copyable PnL (unlimited): %{y:.1f} USDC<extra></extra>"
                    ),
                )
            )

            if plot_exposure:
                exp = copyable_exposure_series(
                    cohort_sel, f"{c}_exposure", bucket_freq=bucket_freq
                )
                if not exp.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=exp["dt"],
                            y=exp["exposure"],
                            mode="lines",
                            line={"color": color, "width": 1.5, "dash": "dot"},
                            opacity=0.75,
                            yaxis="y2",
                            name=f"{cohort_name} ({c}, exposure)",
                            hovertemplate=(
                                f"{cohort_name} ({c}, exposure)<br>%{{x|%Y-%m-%d %H:%M}}<br>"
                                "exposure: %{y:.0f} USDC<extra></extra>"
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
