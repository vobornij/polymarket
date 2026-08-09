"""Named wallet filters for the signal-lab workspace.

A :class:`WalletFilter` is a typed, named selector

    ``WalletFilter(wallet_metrics, hold_metrics) -> set[str]``

Module-level constants (``COPY_DEFAULT``, ``WHALE``, ``FLIPPER``, ...) are the
objects strategies reference directly -- there are no magic strings.  Filters
are pure functions of train-period metrics, so sets stay train-time-derived.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

FilterFn = Callable[[pd.DataFrame, pd.DataFrame], set[str]]

DEFAULT_COPY_RULES = {
    "min_buy_roi": 0.02,
    "min_buckets": 20,
    "min_markets": 15,
    "min_trade_count": 100,
    "max_drawdown_to_pnl": 0.6,
    "min_copyable_roi": 0.05,
}

DEFAULT_ARCHETYPE_MIN_TRADE_COUNT = 100
MIN_ARCHETYPE_WALLETS = 5

ARCHETYPE_FILTER_NAMES = [
    "whale",
    "retail",
    "gambler",
    "overseller",
    "overseller_deep",
    "overseller_thin",
    "consistent",
    "max_dd",
    "both_sides",
    "scalper",
    "flipper",
]


@dataclass(frozen=True)
class WalletFilter:
    """A named wallet selector: ``(wallet_metrics, hold_metrics) -> set[str]``."""

    name: str
    func: FilterFn

    def __call__(
        self,
        wallet_metrics: pd.DataFrame,
        hold_metrics: pd.DataFrame,
    ) -> set[str]:
        return self.func(wallet_metrics, hold_metrics)


# ---------------------------------------------------------------------------
# Copy-universe selection
# ---------------------------------------------------------------------------


def select_copy_wallets(
    wallet_metrics: pd.DataFrame,
    copy_rules: dict[str, float] | None = None,
) -> set[str]:
    """Select copyable wallets by the standard stage-1 thresholds."""
    copy_rules = {**DEFAULT_COPY_RULES, **(copy_rules or {})}
    mask = (
        (wallet_metrics["buy_roi"] >= copy_rules["min_buy_roi"])
        & (wallet_metrics["num_buckets"] >= copy_rules["min_buckets"])
        & (wallet_metrics["num_markets"] >= copy_rules["min_markets"])
        & (wallet_metrics["trade_count"] >= copy_rules["min_trade_count"])
        & (
            wallet_metrics["max_drawdown_to_pnl"].fillna(1.0)
            <= copy_rules["max_drawdown_to_pnl"]
        )
        & (wallet_metrics["copyable_roi"].fillna(0.0) >= copy_rules["min_copyable_roi"])
    )
    return set(wallet_metrics.loc[mask, "wallet"])


def _select_copy_default(
    wallet_metrics: pd.DataFrame,
    hold_metrics: pd.DataFrame,
) -> set[str]:
    """The standard copy-universe mask (see :func:`select_copy_wallets`)."""
    return select_copy_wallets(wallet_metrics, DEFAULT_COPY_RULES)


COPY_DEFAULT = WalletFilter("copy_default", _select_copy_default)


# ---------------------------------------------------------------------------
# Non-copy-trade universe: every wallet with at least one opening BUY in
# the train slice. Used by O2 (Finance/Politics) to test direct strategies
# on the broad universe without filtering on quality metrics.
# ---------------------------------------------------------------------------


def _select_all_buyers(
    wallet_metrics: pd.DataFrame,
    hold_metrics: pd.DataFrame,
) -> set[str]:
    """Every wallet that has any trade in the train slice (a passthrough)."""
    return set(wallet_metrics["wallet"])


ALL_BUYERS = WalletFilter("all_buyers", _select_all_buyers)


# ---------------------------------------------------------------------------
# Archetype masks
# ---------------------------------------------------------------------------


def _quantile_thresholds(series: pd.Series, qs=(0.6, 0.75, 0.8)) -> dict[str, float]:
    s = series.dropna()
    return {f"p{int(q * 100)}": float(s.quantile(q)) for q in qs}


def _archetype_sets(
    wallet_metrics: pd.DataFrame,
    hold_metrics: pd.DataFrame,
    min_trade_count: int,
    min_wallets: int = MIN_ARCHETYPE_WALLETS,
) -> dict[str, set[str]]:
    """Data-driven archetype wallet sets (train-time metrics).

    Definitions mirror the original stage-1 archetypes: quantiles over the
    active population (``trade_count >= min_trade_count``), with scalper /
    flipper additionally using hold-time metrics when available.
    """
    w = wallet_metrics[wallet_metrics["trade_count"] >= min_trade_count].copy()
    if w.empty:
        return {}
    w["avg_trade_usdc"] = w["total_notional"] / w["trade_count"].clip(lower=1)

    t_total = _quantile_thresholds(w["total_notional"], (0.6, 0.75, 0.8))
    t_avg = _quantile_thresholds(w["avg_trade_usdc"], (0.6, 0.75, 0.8))
    t_vol = _quantile_thresholds(w["pnl_volatility"], (0.6, 0.75, 0.8))
    t_topmkt = _quantile_thresholds(w["top_market_pnl_pct"], (0.6, 0.75, 0.8))
    t_posbuck = _quantile_thresholds(w["positive_bucket_share"], (0.2, 0.4, 0.5))

    masks: dict[str, pd.Series] = {
        "whale": (
            (w["total_notional"] >= t_total["p75"])
            & (w["avg_trade_usdc"] >= t_avg["p75"])
        ),
        "retail": (
            (w["avg_trade_usdc"] <= w["avg_trade_usdc"].quantile(0.25))
            & (w["total_notional"] <= w["total_notional"].quantile(0.6))
        ),
        "gambler": (
            (w["pnl_volatility"] >= t_vol["p75"])
            & (w["top_market_pnl_pct"] >= t_topmkt["p75"])
            & (w["positive_bucket_share"] <= t_posbuck["p50"])
            & (w["num_markets"] <= w["num_markets"].quantile(0.6))
        ),
        "overseller": (
            (w["sell_pnl"] < 0) & (w["total_pnl"] > 0)
        ),
        "overseller_deep": (
            (w["sell_pnl"] < 0) & (w["total_pnl"] > 0)
            & (w["sell_roi"] < -0.1)
        ),
        "overseller_thin": (
            (w["sell_pnl"] < 0) & (w["total_pnl"] > 0)
            & (w["buy_pnl"] < 50)
        ),
        "consistent": (
            (w["buy_roi"] >= 0.05)
            & (w["max_drawdown_to_pnl"].fillna(1.0) <= 0.3)
            & (w["num_markets"] >= 20)
            & (w["copyable_roi"].fillna(0.0) >= 0.05)
        ),
        "max_dd": (
            w["max_drawdown_to_pnl"].fillna(1.0) >= 0.6
        ),
        "both_sides": (
            (w["buy_notional"] >= w["buy_notional"].quantile(0.6))
            & (w["sell_notional"] >= w["sell_notional"].quantile(0.6))
        ),
    }

    if hold_metrics is not None and len(hold_metrics):
        h = hold_metrics.set_index("wallet")
        for c in ["median_hold_min", "median_flip_min", "p25_hold_min",
                  "n_round_trips", "round_trip_rate"]:
            if c in h.columns:
                w[c] = w["wallet"].map(h[c])
        w["median_hold_min"] = w["median_hold_min"].replace(np.inf, np.nan)
        w["median_flip_min"] = w["median_flip_min"].replace(np.inf, np.nan)
        masks["scalper"] = (
            (w["n_round_trips"] >= 20)
            & (w["median_hold_min"] <= w["median_hold_min"].quantile(0.25))
            & (w["buy_roi"] > 0) & (w["sell_roi"] > 0)
        )
        masks["flipper"] = (
            (w["median_flip_min"] <= w["median_flip_min"].quantile(0.25))
            & (w["n_round_trips"] >= 20)
        )

    out = {}
    for name, m in masks.items():
        sel = set(w.loc[m, "wallet"])
        if len(sel) >= min_wallets:
            out[name] = sel
    return out


def _make_archetype_filter(
    name: str,
    min_trade_count: int = DEFAULT_ARCHETYPE_MIN_TRADE_COUNT,
) -> WalletFilter:
    def _select(
        wallet_metrics: pd.DataFrame,
        hold_metrics: pd.DataFrame,
    ) -> set[str]:
        return _archetype_sets(
            wallet_metrics,
            hold_metrics,
            min_trade_count,
        ).get(name, set())

    return WalletFilter(name, _select)


WHALE = _make_archetype_filter("whale")
RETAIL = _make_archetype_filter("retail")
GAMBLER = _make_archetype_filter("gambler")
OVERSELLER = _make_archetype_filter("overseller")
OVERSELLER_DEEP = _make_archetype_filter("overseller_deep")
OVERSELLER_THIN = _make_archetype_filter("overseller_thin")
CONSISTENT = _make_archetype_filter("consistent")
MAX_DD = _make_archetype_filter("max_dd")
BOTH_SIDES = _make_archetype_filter("both_sides")
SCALPER = _make_archetype_filter("scalper")
FLIPPER = _make_archetype_filter("flipper")

WALLET_FILTERS: dict[str, WalletFilter] = {
    f.name: f
    for f in [
        COPY_DEFAULT,
        ALL_BUYERS,
        WHALE,
        RETAIL,
        GAMBLER,
        OVERSELLER,
        OVERSELLER_DEEP,
        OVERSELLER_THIN,
        CONSISTENT,
        MAX_DD,
        BOTH_SIDES,
        SCALPER,
        FLIPPER,
    ]
}


def archetype_sets(
    wallet_metrics: pd.DataFrame,
    hold_metrics: pd.DataFrame,
    *,
    min_trade_count: int = DEFAULT_ARCHETYPE_MIN_TRADE_COUNT,
    min_wallets: int = MIN_ARCHETYPE_WALLETS,
) -> dict[str, set[str]]:
    """All archetype signal sets (wallet addresses), excluding ``copy_default``."""
    return _archetype_sets(
        wallet_metrics,
        hold_metrics,
        min_trade_count,
        min_wallets=min_wallets,
    )
