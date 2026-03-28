"""
Volatility-based wallet selection metrics (from the profitable_wallet_analysis path).

This module computes capital-weighted PnL volatility per wallet from 5-minute
trading buckets.  It complements the skill-metric approach in ``metrics.py`` and
can be used independently (e.g. via ``profitable_wallet_analysis.ipynb``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core volatility formula
# ---------------------------------------------------------------------------

def scaled_weighted_pnl_volatility(buckets: pd.DataFrame) -> float:
    """Compute capital-weighted PnL volatility scaled by sqrt(total PnL).

    Each row of *buckets* must contain:

    * ``notional`` – total capital deployed in the bucket
    * ``pnl``      – realised PnL in the bucket

    Returns ``float('nan')`` when there are fewer than 2 buckets, when total
    capital is zero, or when total PnL is non-positive.
    """
    if len(buckets) < 2:
        return float("nan")

    w = buckets["notional"].to_numpy(dtype=float)
    pnl = buckets["pnl"].to_numpy(dtype=float)

    total_w = w.sum()
    total_pnl = pnl.sum()

    if total_w == 0 or total_pnl <= 0:
        return float("nan")

    mean = np.sum(w * pnl) / total_w
    variance = np.sum(w * (pnl - mean) ** 2) / total_w
    sigma = math.sqrt(variance)
    return sigma / math.sqrt(total_pnl)


# ---------------------------------------------------------------------------
# Per-wallet metric computation from pre-aggregated buckets
# ---------------------------------------------------------------------------

def _wallet_metrics_from_buckets(group: pd.DataFrame) -> pd.Series:
    """Compute per-wallet metrics from a pre-aggregated bucket DataFrame.

    *group* is a subset of the buckets DataFrame for a single wallet and must
    contain: ``pnl``, ``notional``, ``condition_id``.
    """
    pnl = group["pnl"].to_numpy(dtype=float)
    total_notional = group["notional"].sum()
    total_pnl = pnl.sum()
    total_qty = group["quantity"].sum()
    copyable_qty = group["copyable_qty"].sum()

    copyable_qty = np.clip(copyable_qty, 0, total_qty)

    if total_pnl <= 0:
        top5_pnl_pct = float("nan")
        top_market_pnl_pct = float("nan")
        median_roi = float("nan")
        average_roi = float("nan")
    else:
        top5_pnl = np.sort(pnl)[-5:].sum()
        top5_pnl_pct = top5_pnl / total_pnl
        top_market_pnl_pct = (
            group.groupby("condition_id")["pnl"].sum().max() / total_pnl
        )
        median_roi = (pnl / group["notional"]).median()
        average_roi = (pnl / group["notional"]).mean()
    return pd.Series(
        {
            "pnl_volatility": scaled_weighted_pnl_volatility(group),
            "num_buckets": len(group),
            "num_markets": group["condition_id"].nunique(),
            "total_notional": total_notional,
            "total_pnl": total_pnl,
            "copyable_pnl": copyable_qty / total_qty * total_pnl,
            "top5_pnl_pct": top5_pnl_pct,
            "top_market_pnl_pct": top_market_pnl_pct,
            "median_roi": median_roi,
            "average_roi": average_roi,
        }
    )


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def compute_wallet_metrics(
    df_slice: pd.DataFrame,
    bucket_freq: str = "5min",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-wallet metrics from a fills DataFrame.

    Steps:

    1. Floor ``dt`` to *bucket_freq* intervals.
    2. Aggregate into ``(wallet, dt_floored, condition_id)`` buckets, keeping
       only buckets with positive notional.
    3. Apply :func:`_wallet_metrics_from_buckets` per wallet.
    4. Compute ``return = total_pnl / total_notional``.

    Parameters
    ----------
    df_slice:
        Fill-level rows.  Must contain: ``wallet``, ``dt``, ``condition_id``,
        ``notional``, ``pnl``.
    bucket_freq:
        Pandas offset alias for the time bucket (default ``'5min'``).

    Returns
    -------
    result : pd.DataFrame
        One row per wallet with columns:
        ``wallet``, ``pnl_volatility``, ``num_buckets``, ``num_markets``,
        ``total_notional``, ``total_pnl``, ``top5_pnl_pct``,
        ``top_market_pnl_pct``, ``median_roi``, ``average_roi``, ``return``
    buckets : pd.DataFrame
        The intermediate bucket-level aggregation.
    """
    tmp = df_slice.copy()
    tmp["dt_floored"] = tmp["dt"].dt.floor(bucket_freq)

    buckets = (
        tmp.groupby(["wallet", "dt_floored", "condition_id"], sort=False)
        .agg(
            notional=("notional", "sum"), 
            pnl=("pnl", "sum"),
            quantity=("quantity", "sum"),
            copyable_qty=("copyable_qty", "sum"),
            )
        .reset_index()
    )
    buckets = buckets[buckets["notional"] > 0].copy()

    empty_cols = [
        "wallet", "pnl_volatility", "num_buckets", "num_markets",
        "total_notional", "total_pnl", "top5_pnl_pct", "top_market_pnl_pct", "median_roi", "average_roi", "return",
    ]

    if buckets.empty:
        return pd.DataFrame(columns=empty_cols), buckets

    result = (
        buckets.groupby("wallet", sort=False)
        .apply(_wallet_metrics_from_buckets, include_groups=False)
        .reset_index()
    )
    result["return"] = result["total_pnl"] / result["total_notional"]
    return result, buckets


# ---------------------------------------------------------------------------
# Volatility-based wallet filter
# ---------------------------------------------------------------------------

def filter_wallets_by_volatility(
    wallet_vol: pd.DataFrame,
    min_buckets: int = 20,
    max_top5_pnl_pct: float = 0.4,
    max_top_market_pnl_pct: float = 0.5,
    min_return: float | None = None,
    max_pnl_volatility: float | None = None,
) -> pd.DataFrame:
    """Apply volatility-based filters to a wallet metrics DataFrame.

    Parameters
    ----------
    wallet_vol:
        Output of :func:`compute_wallet_metrics`.
    min_buckets:
        Minimum number of 5-minute trading buckets required.
    max_top5_pnl_pct:
        Maximum fraction of total PnL attributable to the top 5 buckets
        (guards against a single lucky trade dominating).
    max_top_market_pnl_pct:
        Maximum fraction of total PnL from any single market.
    min_return:
        Optional minimum ``return`` (PnL / notional) filter.
    max_pnl_volatility:
        Optional upper bound on ``pnl_volatility``.

    Returns
    -------
    Filtered and sorted DataFrame.
    """
    mask = (
        (wallet_vol["num_buckets"] >= min_buckets)
        & (wallet_vol["top5_pnl_pct"] <= max_top5_pnl_pct)
        & (wallet_vol["top_market_pnl_pct"] <= max_top_market_pnl_pct)
    )
    if min_return is not None:
        mask &= wallet_vol["return"] >= min_return
    if max_pnl_volatility is not None:
        mask &= wallet_vol["pnl_volatility"] <= max_pnl_volatility

    return (
        wallet_vol[mask]
        .sort_values("total_pnl", ascending=False)
        .reset_index(drop=True)
    )
