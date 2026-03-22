"""
Wallet selection package.

Exposes two parallel selection paths:

* **Volatility path** (``profitable_wallet_analysis`` notebook):
  :mod:`~wallet_selection.volatility` — capital-weighted PnL volatility metrics

* **Skill path** (``wallet_signal_v2`` notebook):
  :mod:`~wallet_selection.metrics` — open-buy edge / ROI skill metrics
  :mod:`~wallet_selection.selector` — cohort construction and sweep

And shared persistence:
  :mod:`~wallet_selection.persistence` — ``WalletSet``, save/load utilities
"""

from .metrics import aggregate_wallet_trades, compute_wallet_skill_workspace
from .volatility import (
    scaled_weighted_pnl_volatility,
    compute_wallet_metrics,
    filter_wallets_by_volatility,
)
from .selector import (
    CANDIDATE_METRICS,
    select_wallets,
    cohort_selection_sweep,
    build_wallet_cohorts,
    build_strategies_from_sweep,
)
from .persistence import WalletSet, save_wallet_set, load_wallet_set, wallet_set_exists

__all__ = [
    # metrics
    "aggregate_wallet_trades",
    "compute_wallet_skill_workspace",
    # volatility
    "scaled_weighted_pnl_volatility",
    "compute_wallet_metrics",
    "filter_wallets_by_volatility",
    # selector
    "CANDIDATE_METRICS",
    "select_wallets",
    "cohort_selection_sweep",
    "build_wallet_cohorts",
    "build_strategies_from_sweep",
    # persistence
    "WalletSet",
    "save_wallet_set",
    "load_wallet_set",
    "wallet_set_exists",
]
