"""Visualization package for creating charts and dashboards."""

from polymarket_analysis.visualization.wallet_plots import (
    plot_wallet_pnl_bars,
    plot_wallet_returns,
    plot_cumulative_pnl_by_wallet,
    plot_combined_cumulative_pnl,
)
from polymarket_analysis.visualization.backtest_plots import (
    with_zero_anchor,
    build_strategy_sum_daily,
    add_daily_traces,
    plot_strategy_comparison,
)

__all__ = [
    # wallet_plots
    "plot_wallet_pnl_bars",
    "plot_wallet_returns",
    "plot_cumulative_pnl_by_wallet",
    "plot_combined_cumulative_pnl",
    # backtest_plots
    "with_zero_anchor",
    "build_strategy_sum_daily",
    "add_daily_traces",
    "plot_strategy_comparison",
]
