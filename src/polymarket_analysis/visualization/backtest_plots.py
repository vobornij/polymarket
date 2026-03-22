"""
Backtest result visualization functions.

All plot functions return a ``plotly.graph_objects.Figure``.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resample_hourly(trades: pd.DataFrame, starting_capital: float = 0.0) -> pd.DataFrame:
    """Resample filled trades to 1-hour buckets and compute cumulative PnL.

    Parameters
    ----------
    trades:
        Filled trades DataFrame with columns ``dt`` and ``net_pnl_usdc``.
    starting_capital:
        Baseline added to the cumulative sum for the equity curve.

    Returns
    -------
    DataFrame with columns ``trade_dt`` (hour bucket), ``net_pnl_usdc``,
    ``cum_net_pnl_usdc``.  Empty DataFrame (same columns) if *trades* is empty.
    """
    cols = ["trade_dt", "net_pnl_usdc", "cum_net_pnl_usdc"]
    if trades.empty or "net_pnl_usdc" not in trades.columns:
        return pd.DataFrame(columns=cols)

    hourly = (
        trades.assign(trade_dt=trades["dt"].dt.floor("1h"))
        .groupby("trade_dt", as_index=False)["net_pnl_usdc"]
        .sum()
        .sort_values("trade_dt")
        .reset_index(drop=True)
    )
    hourly["cum_net_pnl_usdc"] = hourly["net_pnl_usdc"].cumsum()
    return hourly


def resample_hourly_theoretical(run: dict) -> pd.DataFrame:
    """Resample the theoretical (trigger-price) curve to 1-hour buckets.

    Reconstructs candidate triggers by combining ``"trades"`` (filled) and
    ``"unfilled_triggers"`` — both carry ``dt`` and ``trigger_net_pnl_usdc``
    — and buckets them at 1h resolution.

    Returns
    -------
    DataFrame with columns ``trade_dt``, ``net_pnl_usdc``, ``cum_net_pnl_usdc``.
    Empty DataFrame (same columns) if neither frame has the required columns.
    """
    cols = ["trade_dt", "net_pnl_usdc", "cum_net_pnl_usdc"]
    parts = []
    for key in ("trades", "unfilled_triggers"):
        df = run.get(key, pd.DataFrame())
        if not df.empty and "trigger_net_pnl_usdc" in df.columns and "dt" in df.columns:
            parts.append(df[["dt", "trigger_net_pnl_usdc"]].rename(
                columns={"trigger_net_pnl_usdc": "net_pnl_usdc"}
            ))
    if not parts:
        return pd.DataFrame(columns=cols)

    combined = pd.concat(parts, ignore_index=True)
    hourly = (
        combined.assign(trade_dt=combined["dt"].dt.floor("1h"))
        .groupby("trade_dt", as_index=False)["net_pnl_usdc"]
        .sum()
        .sort_values("trade_dt")
        .reset_index(drop=True)
    )
    hourly["cum_net_pnl_usdc"] = hourly["net_pnl_usdc"].cumsum()
    return hourly


def with_zero_anchor_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Prepend a zero-PnL anchor one hour before the first bucket.

    Ensures cumulative PnL plots start from the origin.

    Parameters
    ----------
    hourly:
        Output of :func:`resample_hourly` with column ``trade_dt``.

    Returns
    -------
    pd.DataFrame with the anchor row prepended.
    """
    if hourly.empty:
        return hourly
    first = hourly["trade_dt"].min()
    anchor = pd.DataFrame({
        "trade_dt": [first - pd.Timedelta(hours=1)],
        "net_pnl_usdc": [0.0],
        "cum_net_pnl_usdc": [0.0],
    })
    return (
        pd.concat([anchor, hourly], ignore_index=True)
        .sort_values("trade_dt")
        .reset_index(drop=True)
    )


# Keep the old daily helper for backward compatibility with any callers that
# still pass a pre-bucketed ``daily`` DataFrame.
def with_zero_anchor(daily: pd.DataFrame) -> pd.DataFrame:
    """Prepend a zero-PnL anchor row one day before the first trade date."""
    if daily.empty:
        return daily
    first_date = daily["trade_date"].min()
    anchor = pd.DataFrame({
        "trade_date": [first_date - pd.Timedelta(days=1)],
        "net_pnl_usdc": [0.0],
        "cum_net_pnl_usdc": [0.0],
    })
    return (
        pd.concat([anchor, daily], ignore_index=True)
        .sort_values("trade_date")
        .reset_index(drop=True)
    )


def build_strategy_sum_daily(strategy_runs: dict) -> pd.DataFrame:
    """Aggregate daily PnL across all strategies for a quick portfolio view."""
    parts = []
    for run in strategy_runs.values():
        daily = run.get("daily", pd.DataFrame())
        if not daily.empty:
            parts.append(daily[["trade_date", "net_pnl_usdc"]])
    if not parts:
        return pd.DataFrame(columns=["trade_date", "net_pnl_usdc", "cum_net_pnl_usdc"])
    combined = (
        pd.concat(parts, ignore_index=True)
        .groupby("trade_date", as_index=False)["net_pnl_usdc"]
        .sum()
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    combined["cum_net_pnl_usdc"] = combined["net_pnl_usdc"].cumsum()
    return combined


def add_hourly_traces(
    fig: go.Figure,
    strategy_runs: dict,
    *,
    row: int = 1,
    col: int = 1,
    name_suffix: str = "",
    filled_opacity: float = 1.0,
    filled_dash: str = "solid",
) -> go.Figure:
    """Add 1-hour-resolution filled cumulative PnL traces to a subplot.

    Each run dict must have a ``"trades"`` key with a DataFrame that contains
    ``dt`` and ``net_pnl_usdc`` columns.  The theoretical (trigger-price) curve
    is intentionally omitted — it adds clutter without improving readability.

    Parameters
    ----------
    fig:
        Existing ``go.Figure`` / subplot figure.
    strategy_runs:
        Dict mapping strategy name → run dict (key ``"trades"``).
    row, col:
        Subplot position.
    name_suffix:
        String appended to every trace name.
    filled_opacity, filled_dash:
        Opacity and dash style for the filled PnL traces.
    """
    for name, run in strategy_runs.items():
        trades = run.get("trades", pd.DataFrame())
        hourly = resample_hourly(trades)
        if hourly.empty:
            continue
        plot_hourly = with_zero_anchor_hourly(hourly)
        fig.add_trace(
            go.Scatter(
                x=plot_hourly["trade_dt"],
                y=plot_hourly["cum_net_pnl_usdc"],
                mode="lines",
                line={"dash": filled_dash},
                name=f"{name}{name_suffix}",
                opacity=filled_opacity,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    "cum PnL: %{y:.2f} USDC<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
    return fig


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_strategy_comparison(
    strategy_runs: dict,
    *,
    title: str = "Strategy comparison – cumulative PnL (test)",
) -> go.Figure:
    """Single-panel 1h-resolution cumulative PnL chart for the test period.

    Parameters
    ----------
    strategy_runs:
        Test-period strategy runs dict (``{strategy_id → run_dict}``).
        Each run dict must contain a ``"trades"`` DataFrame with ``dt`` and
        ``net_pnl_usdc`` columns.
    title:
        Figure title.

    Returns
    -------
    ``go.Figure`` — one trace per strategy, 1h resolution.
    """
    fig = go.Figure()
    for name, run in strategy_runs.items():
        trades = run.get("trades", pd.DataFrame())
        hourly = resample_hourly(trades)
        if not hourly.empty:
            plot_hourly = with_zero_anchor_hourly(hourly)
            fig.add_trace(
                go.Scatter(
                    x=plot_hourly["trade_dt"],
                    y=plot_hourly["cum_net_pnl_usdc"],
                    mode="lines",
                    line={"dash": "solid"},
                    name=name,
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "%{x|%Y-%m-%d %H:%M}<br>"
                        "cum PnL: %{y:.2f} USDC<extra></extra>"
                    ),
                )
            )

        # Theoretical (trigger-price) curve — dashdot, lower opacity
        theo = resample_hourly_theoretical(run)
        if not theo.empty:
            plot_theo = with_zero_anchor_hourly(theo)
            fig.add_trace(
                go.Scatter(
                    x=plot_theo["trade_dt"],
                    y=plot_theo["cum_net_pnl_usdc"],
                    mode="lines",
                    line={"dash": "dashdot"},
                    opacity=0.5,
                    name=f"{name} [theoretical]",
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "%{x|%Y-%m-%d %H:%M}<br>"
                        "theoretical cum PnL: %{y:.2f} USDC<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        template="plotly_white",
        height=500,
        title=title,
        xaxis_title="Time (1h buckets)",
        yaxis_title="Cumulative net PnL (USDC)",
    )
    return fig


def plot_strategy_test(
    strategy_runs: dict,
    *,
    title: str = "Strategy comparison – test period (1h)",
) -> go.Figure:
    """1h-resolution cumulative PnL chart for the **test** period.

    Alias for :func:`plot_strategy_comparison` with a clearer name.
    """
    return plot_strategy_comparison(strategy_runs, title=title)


def plot_strategy_train(
    strategy_runs_train_ref: dict,
    *,
    title: str = "Strategy comparison – train-B period (1h)",
) -> go.Figure:
    """1h-resolution cumulative PnL chart for the **train-B** reference period.

    Parameters
    ----------
    strategy_runs_train_ref:
        Train-reference strategy runs dict.
    title:
        Figure title.
    """
    return plot_strategy_comparison(strategy_runs_train_ref, title=title)
