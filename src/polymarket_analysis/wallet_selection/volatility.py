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

def _max_drawdown(cum: np.ndarray) -> float:
    """Largest peak-to-trough drop in a running PnL series (peak starts at 0)."""
    peak = 0.0
    dd = 0.0
    for x in cum:
        if x > peak:
            peak = x
        if x < peak:
            d = peak - x
            if d > dd:
                dd = d
    return dd


def _wallet_metrics_from_buckets(buckets: pd.DataFrame) -> pd.DataFrame:
    """Compute per-wallet metrics from a pre-aggregated bucket DataFrame.

    *buckets* is the wallet-bucketed frame produced by
    :func:`compute_wallet_metrics` (one row per (wallet, dt_floored,
    condition_id, side), wallets contiguous in appearance order) and must
    contain: ``pnl``, ``notional``, ``condition_id``, ``copyable_pnl``,
    ``quantity``, ``copyable_qty``, ``side``, ``dt_floored``, ``trade_count``.

    All metrics are computed in a single groupby pass (no per-wallet Python
    loop).  Row order within a wallet is preserved exactly: the drawdown and
    bucket-sum reductions depend on it, so no re-sorting is applied.
    """
    df = buckets.copy()
    g = df.groupby("wallet", sort=False)

    res = pd.DataFrame({"wallet": df["wallet"].drop_duplicates().to_numpy()}).set_index("wallet")
    res["num_buckets"] = g.size()
    res["trade_count"] = g["trade_count"].sum()
    res["total_notional"] = g["notional"].sum()
    res["total_pnl"] = g["pnl"].sum()
    res["copyable_pnl"] = g["copyable_pnl"].sum()
    res["num_markets"] = g["condition_id"].nunique()
    res["median_dt"] = g["dt_floored"].median()

    roi = df["pnl"].to_numpy(dtype=float) / df["notional"].to_numpy(dtype=float)
    df["_roi"] = roi
    g = df.groupby("wallet", sort=False)
    res["median_roi"] = g["_roi"].median()
    res["average_roi"] = g["_roi"].mean()

    df["_cpnl"] = g["pnl"].cumsum()
    df["_ccp"] = g["copyable_pnl"].cumsum()

    def drawdowns(sub: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "max_drawdown": _max_drawdown(sub["_cpnl"].to_numpy()),
            "max_copyable_drawdown": _max_drawdown(sub["_ccp"].to_numpy()),
        })

    dd = df.groupby("wallet", sort=False).apply(drawdowns, include_groups=False)
    res["max_drawdown"] = dd["max_drawdown"]
    res["max_copyable_drawdown"] = dd["max_copyable_drawdown"]

    piv = df.pivot_table(
        index="wallet", columns="side",
        values=["pnl", "quantity", "notional", "copyable_qty", "copyable_pnl"],
        aggfunc="sum", sort=False, fill_value=0.0)

    def side_sum(var: str, side: str) -> pd.Series:
        return piv[(var, side)] if (var, side) in piv.columns else 0.0

    res["buy_pnl"] = side_sum("pnl", "BUY")
    res["buy_quantity"] = side_sum("quantity", "BUY")
    res["sell_pnl"] = side_sum("pnl", "SELL")
    res["sell_quantity"] = side_sum("quantity", "SELL")
    res["buy_notional"] = side_sum("notional", "BUY")
    res["sell_notional"] = side_sum("notional", "SELL")
    res["buy_copyable_quantity"] = side_sum("copyable_qty", "BUY")
    res["buy_copyable_pnl"] = side_sum("copyable_pnl", "BUY")
    res["sell_copyable_quantity"] = side_sum("copyable_qty", "SELL")
    res["sell_copyable_pnl"] = side_sum("copyable_pnl", "SELL")

    res["buy_roi"] = np.where(res["buy_quantity"] > 0, res["buy_pnl"] / res["buy_notional"], 0.0)
    res["sell_roi"] = np.where(res["sell_quantity"] > 0, res["sell_pnl"] / res["sell_notional"], 0.0)
    res["buy_copyable_notional"] = np.where(
        res["buy_quantity"] > 0,
        res["buy_notional"] * (res["buy_copyable_quantity"] / res["buy_quantity"]),
        float("nan"))

    df["dt_1h"] = df["dt_floored"].dt.floor("1h")
    bucket_pnls = df.groupby(["wallet", "dt_1h", "condition_id"], sort=False)["pnl"].sum()
    total_pnl_s = res["total_pnl"]

    def topn_pct(bp: pd.Series, n: int, ascending: bool) -> pd.Series:
        s = bp.sort_values(ascending=ascending, kind="stable")
        top = s.groupby("wallet", sort=False).head(n).groupby("wallet", sort=False).sum()
        return top / total_pnl_s

    res["top5_pnl_pct"] = topn_pct(bucket_pnls, 5, False)
    res["top10_pnl_pct"] = topn_pct(bucket_pnls, 10, False)
    res["worst5_pnl_pct"] = topn_pct(bucket_pnls, 5, True)
    res["positive_bucket_share"] = (bucket_pnls > 0).groupby("wallet", sort=False).mean()

    market_pnls = df.groupby(["wallet", "condition_id"], sort=False)["pnl"].sum()
    res["top_market_pnl_pct"] = market_pnls.groupby("wallet", sort=False).max() / total_pnl_s
    abs_market_pnls = market_pnls.abs()
    abs_market_total = abs_market_pnls.groupby("wallet", sort=False).sum()
    nz = abs_market_total > 0
    res["top_market_abs_pnl_pct"] = np.where(
        nz, abs_market_pnls.groupby("wallet", sort=False).max() / abs_market_total, float("nan"))
    res["market_pnl_hhi"] = np.where(
        nz, ((abs_market_pnls / abs_market_total) ** 2).groupby("wallet", sort=False).sum(),
        float("nan"))

    df["_wp"] = df["notional"].to_numpy(dtype=float) * df["pnl"].to_numpy(dtype=float)
    df["_wp2"] = df["notional"].to_numpy(dtype=float) * df["pnl"].to_numpy(dtype=float) ** 2
    gw = df.groupby("wallet", sort=False)
    total_w = gw["notional"].sum()
    mean_wp = gw["_wp"].sum() / total_w
    var = np.clip(gw["_wp2"].sum() / total_w - mean_wp ** 2, 0, None)
    sigma = np.sqrt(var)
    valid = (gw.size() >= 2) & (total_w > 0) & (res["total_pnl"] > 0)
    res["pnl_volatility"] = np.where(valid, sigma / np.sqrt(res["total_pnl"]), float("nan"))

    nz2 = abs(res["total_pnl"]) <= 0
    for c in ["top5_pnl_pct", "top10_pnl_pct", "worst5_pnl_pct", "top_market_pnl_pct",
              "median_roi", "average_roi", "buy_roi", "buy_pnl", "buy_copyable_pnl",
              "buy_copyable_quantity", "sell_copyable_pnl", "sell_roi", "sell_copyable_quantity"]:
        res.loc[nz2, c] = float("nan")

    res["max_drawdown_to_pnl"] = np.where(
        res["total_pnl"] > 0, res["max_drawdown"] / res["total_pnl"], float("nan"))
    res["max_copyable_drawdown_to_copyable_pnl"] = np.where(
        res["copyable_pnl"] > 0, res["max_copyable_drawdown"] / res["copyable_pnl"], float("nan"))
    return res.reset_index()


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
    3. Compute per-wallet metrics in one vectorized groupby pass.
    4. Compute ``return = total_pnl / total_notional``.

    Parameters
    ----------
    df_slice:
        Fill-level rows.  Must contain: ``wallet``, ``dt``, ``condition_id``,
        ``notional``, ``pnl``.
    bucket_freq:
        Pandas offset alias for the time bucket (default ``'5m'``).

    Returns
    -------
    result : pd.DataFrame
        One row per wallet with columns:
        ``wallet``, ``pnl_volatility``, ``num_buckets``, ``num_markets``,
        ``total_notional``, ``total_pnl``, ``top5_pnl_pct``, ``worst5_pnl_pct``,
        ``top_market_pnl_pct``, ``median_roi``, ``average_roi``, ``return``
    buckets : pd.DataFrame
        The intermediate bucket-level aggregation.
    """
    # clips copyable qty here to avoid double counting for multiple trades in the same bucket,
    # it is not done in _wallet_metrics_from_buckets

    tmp = df_slice.copy()
    tmp["dt_floored"] = tmp["dt"].dt.floor(bucket_freq)

    buckets = (
        tmp.groupby(
            ["wallet", "dt_floored", "condition_id", "side"],
            sort=False,
            observed=True,
        )
        .agg(
            notional=("notional", "sum"),
            pnl=("pnl", "sum"),
            copyable_pnl=("copyable_pnl", "sum"),
            quantity=("quantity", "sum"),
            copyable_qty_sum=("copyable_qty_5m_100", "sum"),
            trade_count=("pnl", "size"),
            avail_copy_total_vol=("avail_copy_total_vol_5m_100", "max"),
        )
        .reset_index()
    )

    # clip copyable qty here to avoid double counting for multiple trades in the same bucket,
    # it is not done in _wallet_metrics_from_buckets

    buckets["copyable_qty"] = np.minimum(
        buckets["copyable_qty_sum"],
        buckets["avail_copy_total_vol"],
    )

    mask = buckets["copyable_qty_sum"] > 0
    buckets.loc[mask, "copyable_pnl"] *= (
        buckets.loc[mask, "copyable_qty"]
        / buckets.loc[mask, "copyable_qty_sum"]
    )

    buckets = buckets.drop(columns="copyable_qty_sum")

    buckets = buckets[buckets["notional"] > 0].copy()

    empty_cols = [
        "wallet", "pnl_volatility", "num_buckets", "num_markets",
        "total_notional", "total_pnl", "top5_pnl_pct", "top10_pnl_pct",
        "worst5_pnl_pct", "top_market_pnl_pct", "top_market_abs_pnl_pct",
        "market_pnl_hhi", "positive_bucket_share", "median_roi", "average_roi", "return", "trade_count",
    ]

    if buckets.empty:
        return pd.DataFrame(columns=empty_cols), buckets

    result = _wallet_metrics_from_buckets(buckets)
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
