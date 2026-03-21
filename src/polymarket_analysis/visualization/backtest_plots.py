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

def with_zero_anchor(daily: pd.DataFrame) -> pd.DataFrame:
    """Prepend a zero-PnL anchor row one day before the first trade date.

    This ensures cumulative PnL plots start from the origin rather than
    jumping from an implicit zero on the first day.

    Parameters
    ----------
    daily:
        DataFrame produced by :func:`backtest_strategy` with columns
        ``trade_date``, ``net_pnl_usdc``, ``cum_net_pnl_usdc``.

    Returns
    -------
    pd.DataFrame with the anchor row prepended, sorted by ``trade_date``.
    """
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
    """Aggregate daily PnL across all strategies for a quick portfolio view.

    Parameters
    ----------
    strategy_runs:
        Dict mapping strategy name → run dict (keys: ``daily``, ``trades``, …).

    Returns
    -------
    pd.DataFrame with columns ``trade_date``, ``net_pnl_usdc``,
    ``cum_net_pnl_usdc``.
    """
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


def add_daily_traces(
    fig: go.Figure,
    strategy_runs: dict,
    *,
    row: int = 1,
    col: int = 1,
    include_theoretical: bool = True,
    name_suffix: str = "",
    filled_opacity: float = 1.0,
    theoretical_opacity: float = 0.55,
    filled_dash: str = "solid",
    theoretical_dash: str = "dashdot",
) -> go.Figure:
    """Add filled and (optionally) theoretical daily PnL traces to a subplot.

    Parameters
    ----------
    fig:
        Existing ``go.Figure`` / subplot figure to add traces to.
    strategy_runs:
        Dict mapping strategy name → run dict (keys: ``daily``,
        ``theoretical_daily``).
    row, col:
        Subplot position.
    include_theoretical:
        Whether to also draw the trigger-theoretical (pre-fill) curve.
    name_suffix:
        String appended to every trace name for disambiguation.
    filled_opacity, theoretical_opacity:
        Opacity for filled and theoretical traces respectively.
    filled_dash, theoretical_dash:
        Plotly line dash style for each trace type.
    """
    for name, run in strategy_runs.items():
        daily = run.get("daily", pd.DataFrame())
        if not daily.empty:
            plot_daily = with_zero_anchor(daily)
            fig.add_trace(
                go.Scatter(
                    x=plot_daily["trade_date"],
                    y=plot_daily["cum_net_pnl_usdc"],
                    mode="lines",
                    line={"dash": filled_dash},
                    name=f"{name}{name_suffix}",
                    opacity=filled_opacity,
                ),
                row=row,
                col=col,
            )
        if include_theoretical:
            theoretical_daily = run.get("theoretical_daily", pd.DataFrame())
            if not theoretical_daily.empty:
                plot_theoretical = with_zero_anchor(theoretical_daily)
                fig.add_trace(
                    go.Scatter(
                        x=plot_theoretical["trade_date"],
                        y=plot_theoretical["cum_net_pnl_usdc"],
                        mode="lines",
                        line={"dash": theoretical_dash},
                        name=f"{name} [trigger_theoretical]{name_suffix}",
                        opacity=theoretical_opacity,
                    ),
                    row=row,
                    col=col,
                )
    return fig


def plot_strategy_comparison(
    strategy_runs: dict,
    strategy_runs_train_ref: dict | None = None,
    *,
    split_date: pd.Timestamp | None = None,
    calibration_df: pd.DataFrame | None = None,
    title: str = "Strategy comparison – cumulative PnL",
) -> go.Figure:
    """Two-panel figure: cumulative PnL comparison + optional calibration bar.

    Panel 1 (top): train-reference (dotted) and test (solid) cumulative PnL
    curves for every strategy.

    Panel 2 (bottom): score-decile calibration bar chart, drawn only when
    *calibration_df* is supplied.

    Parameters
    ----------
    strategy_runs:
        Test-period strategy runs dict.
    strategy_runs_train_ref:
        Train-reference strategy runs dict.  Pass ``None`` to skip.
    split_date:
        If provided, a vertical dashed line is drawn at this timestamp in
        panel 1.
    calibration_df:
        DataFrame with columns ``score_decile`` and ``avg_copy_roi_capped``
        used for the calibration bar in panel 2.  Pass ``None`` to skip.
    title:
        Figure title.
    """
    n_rows = 2 if calibration_df is not None else 1
    subplot_titles = ["Cumulative PnL (train ref + test)"]
    if calibration_df is not None:
        subplot_titles.append("Train-B score calibration")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=subplot_titles,
    )

    if strategy_runs_train_ref:
        add_daily_traces(
            fig,
            strategy_runs_train_ref,
            row=1,
            col=1,
            name_suffix=" [train_ref]",
            filled_dash="dot",
            filled_opacity=0.65,
            theoretical_dash="dashdot",
            theoretical_opacity=0.45,
        )

    add_daily_traces(
        fig,
        strategy_runs,
        row=1,
        col=1,
        name_suffix=" [test]",
        filled_dash="solid",
        filled_opacity=1.0,
        theoretical_dash="dashdot",
        theoretical_opacity=0.55,
    )

    if split_date is not None:
        fig.add_vline(
            x=split_date,
            line_dash="dash",
            line_color="black",
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=split_date,
            y=1.02,
            yref="paper",
            text="train/test split",
            showarrow=False,
        )

    if calibration_df is not None and not calibration_df.empty:
        fig.add_trace(
            go.Bar(
                x=calibration_df["score_decile"],
                y=calibration_df["avg_copy_roi_capped"],
                name="train_b avg copy roi capped",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        height=900 if n_rows == 2 else 500,
        title=title,
    )
    return fig
