"""
Wallet group selection for copy-trading.

Orchestrates the selection of four wallet groups:

* **openers** — strong copyable openers, always copied on BUY
* **leaders** — wallets that precede profitable follower BUYs
* **followers** — wallets with high copyable ROI liquidity
* **closers** — wallets with positive PnL, copy their SELLs

Entry point: :func:`build_wallet_groups`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .leader_follower import (
    aggregate_leader_follower_pairs,
    detect_leader_follower_pairs,
    preselect_followers,
    rank_leaders,
)


# ---------------------------------------------------------------------------
# Open-buy detection
# ---------------------------------------------------------------------------

def _compute_open_buys(df: pd.DataFrame) -> pd.DataFrame:
    """Identify open-buy events from a trades DataFrame.

    An open buy is a BUY trade where the wallet had no prior position in the
    same (condition_id, outcome).  This is computed via a cumulative signed-
    quantity approach grouped by (wallet, condition_id, outcome).

    Parameters
    ----------
    df:
        Trades DataFrame.  Must contain ``wallet``, ``condition_id``,
        ``outcome``, ``dt``, ``side``, ``quantity``, ``copyable_pnl``.
        Column names follow the notebook's fill-level convention
        (``quantity``, ``price``, ``usdc_amount``).

    Returns
    -------
    DataFrame of open_buy rows (subset of *df*, all columns preserved).
    """
    work = df[["wallet", "condition_id", "outcome", "dt", "side", "quantity"]].copy()
    work["_signed_qty"] = np.where(work["side"] == "BUY", work["quantity"], -work["quantity"])
    work["_prev_pos"] = (
        work.groupby(["wallet", "condition_id", "outcome"])["_signed_qty"]
        .cumsum()
        - work["_signed_qty"]
    )

    open_mask = (work["side"] == "BUY") & (work["_prev_pos"] <= 1e-9)
    return df.loc[open_mask.index[open_mask]].copy()


# ---------------------------------------------------------------------------
# Per-wallet metric computation
# ---------------------------------------------------------------------------

def _compute_wallet_trade_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """Compute per-wallet aggregate trade statistics.

    Columns include copyable metrics (copyable_pnl, copyable_roi,
    copyable_open_roi, copyable_sell_pnl) used for scoring, and wallet
    trade metrics (trade_pnl, trade_roi) used for filtering wallets by
    their own profitability.
    """
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "wallet", "copyable_pnl", "copyable_roi",
                "trade_pnl", "trade_roi", "trade_count",
                "buy_count", "sell_count", "total_trade_value",
                "market_pnl_hhi", "copyable_open_roi", "copyable_sell_pnl",
            ]
        )

    notional_col = "usdc_amount" if "usdc_amount" in trades.columns else "trade_value_usdc"

    agg_dict = {
        "total_trade_value": (notional_col, "sum"),
        "trade_count": ("wallet", "size"),
        "buy_count": ("side", lambda s: (s == "BUY").sum()),
        "sell_count": ("side", lambda s: (s == "SELL").sum()),
    }
    # Copyable PnL (what we earn by copying)
    agg_dict["copyable_pnl"] = ("copyable_pnl", "sum")
    # Wallet's own trade PnL (for filtering)
    if "trade_pnl" in trades.columns:
        agg_dict["trade_pnl"] = ("trade_pnl", "sum")

    wallet_agg = trades.groupby("wallet").agg(**agg_dict).reset_index()

    wallet_agg["copyable_roi"] = (
        wallet_agg["copyable_pnl"]
        / wallet_agg["total_trade_value"].clip(lower=1e-9)
    )
    if "trade_pnl" in wallet_agg.columns:
        wallet_agg["trade_roi"] = (
            wallet_agg["trade_pnl"]
            / wallet_agg["total_trade_value"].clip(lower=1e-9)
        )
    else:
        wallet_agg["trade_pnl"] = 0.0
        wallet_agg["trade_roi"] = 0.0

    # Copyable sell PnL: sum of copyable_pnl for SELL trades only.
    sell_trades = trades[trades["side"] == "SELL"]
    if not sell_trades.empty:
        sell_agg = (
            sell_trades.groupby("wallet")["copyable_pnl"]
            .sum()
            .reset_index()
            .rename(columns={"copyable_pnl": "copyable_sell_pnl"})
        )
        wallet_agg = wallet_agg.merge(sell_agg, on="wallet", how="left")
    wallet_agg["copyable_sell_pnl"] = wallet_agg.get("copyable_sell_pnl", 0.0).fillna(0.0)

    # Market PnL HHI: concentration of absolute copyable PnL across markets.
    market_pnl = (
        trades.groupby(["wallet", "condition_id"])["copyable_pnl"]
        .sum()
        .abs()
        .reset_index()
    )
    market_pnl.columns = ["wallet", "condition_id", "abs_pnl"]

    def _hhi(group: pd.DataFrame) -> float:
        total = group["abs_pnl"].sum()
        if total <= 0:
            return float("nan")
        weights = group["abs_pnl"] / total
        return float(np.square(weights).sum())

    hhi_series = market_pnl.groupby("wallet").apply(_hhi, include_groups=False)
    hhi_df = hhi_series.reset_index()
    hhi_df.columns = ["wallet", "market_pnl_hhi"]
    wallet_agg = wallet_agg.merge(hhi_df, on="wallet", how="left")

    # Copyable open ROI: BUY copyable_pnl / BUY notional.
    buy_trades = trades[trades["side"] == "BUY"]
    if not buy_trades.empty:
        buy_agg = (
            buy_trades.groupby("wallet")
            .agg(
                buy_copyable_pnl=("copyable_pnl", "sum"),
                buy_trade_value=(notional_col, "sum"),
            )
            .reset_index()
        )
        buy_agg["copyable_open_roi"] = (
            buy_agg["buy_copyable_pnl"]
            / buy_agg["buy_trade_value"].clip(lower=1e-9)
        )
        wallet_agg = wallet_agg.merge(
            buy_agg[["wallet", "copyable_open_roi"]], on="wallet", how="left"
        )
    else:
        wallet_agg["copyable_open_roi"] = float("nan")

    return wallet_agg


# ---------------------------------------------------------------------------
# Openers and closers selection
# ---------------------------------------------------------------------------

def select_openers(
    wallet_stats: pd.DataFrame,
    *,
    min_trade_pnl: float = 0.0,
    min_trade_roi: float = 0.0,
    max_market_pnl_hhi: float = 0.5,
    min_copyable_open_roi: float = 0.05,
    min_trade_count: int = 20,
    top_n: int = 100,
) -> pd.DataFrame:
    """Select strong copyable openers.

    Filter by wallet trade profitability (trade_pnl, trade_roi), then
    score by copyable metrics: weighted combination of copyable_pnl,
    copyable_roi, copyable_open_roi, and (1 − market_pnl_hhi).
    """
    eligible = wallet_stats[
        (wallet_stats["trade_pnl"] >= min_trade_pnl)
        & (wallet_stats["trade_roi"] >= min_trade_roi)
        & (wallet_stats["market_pnl_hhi"] <= max_market_pnl_hhi)
        & (wallet_stats["copyable_open_roi"] >= min_copyable_open_roi)
        & (wallet_stats["trade_count"] >= min_trade_count)
    ].copy()

    if eligible.empty:
        return pd.DataFrame(columns=["wallet", "opener_score"])

    for col in ["copyable_pnl", "copyable_roi", "copyable_open_roi"]:
        eligible[f"_rank_{col}"] = eligible[col].rank(method="average", pct=True)
    eligible["_rank_diversity"] = (1.0 - eligible["market_pnl_hhi"]).rank(
        method="average", pct=True
    )

    eligible["opener_score"] = (
        0.30 * eligible["_rank_copyable_pnl"]
        + 0.20 * eligible["_rank_copyable_roi"]
        + 0.30 * eligible["_rank_copyable_open_roi"]
        + 0.20 * eligible["_rank_diversity"]
    )

    return eligible.nlargest(top_n, "opener_score")[
        ["wallet", "opener_score", "copyable_pnl", "copyable_roi", "trade_pnl", "trade_roi",
         "copyable_open_roi", "market_pnl_hhi", "trade_count"]
    ].reset_index(drop=True)


def select_closers(
    wallet_stats: pd.DataFrame,
    *,
    min_copyable_sell_pnl: float = 0.0,
    min_sell_count: int = 5,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Select wallets with strong copyable sell PnL for SELL copying."""
    eligible = wallet_stats[
        (wallet_stats["copyable_sell_pnl"] >= min_copyable_sell_pnl)
        & (wallet_stats["sell_count"] >= min_sell_count)
    ].copy()

    if eligible.empty:
        return pd.DataFrame(columns=["wallet", "closer_score"])

    eligible["closer_score"] = eligible["copyable_sell_pnl"].rank(method="average", pct=True)

    result = eligible.sort_values("closer_score", ascending=False)
    if top_n is not None:
        result = result.head(top_n)

    return result[
        ["wallet", "closer_score", "copyable_sell_pnl", "trade_pnl", "trade_roi", "sell_count"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def build_wallet_groups(
    df: pd.DataFrame,
    *,
    # Follower selection params
    min_follower_trade_value: float = 100.0,
    min_follower_copyable_roi: float = 0.10,
    min_follower_open_buys: int = 10,
    # Leader-follower detection params
    time_window_minutes: int = 10,
    min_pair_interactions: int = 10,
    # Opener selection params
    min_opener_trade_pnl: float = 0.0,
    min_opener_trade_roi: float = 0.0,
    max_opener_hhi: float = 0.5,
    min_opener_copyable_open_roi: float = 0.05,
    min_opener_trade_count: int = 20,
    top_n_openers: int = 100,
    # Closer selection params
    min_closer_copyable_sell_pnl: float = 0.0,
    min_closer_sell_count: int = 5,
    top_n_closers: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Build all four wallet groups from a trades DataFrame.

    Parameters
    ----------
    df:
        Trades DataFrame matching the notebook's fill-level schema:
        ``wallet``, ``condition_id``, ``outcome``, ``dt``, ``side``,
        ``quantity``, ``price``, ``usdc_amount``, ``copyable_pnl``,
        ``token_winner``, ``final_price``, etc.
    """
    # --- Step 1: Open buys for follower/leader detection ---------------------
    open_buys = _compute_open_buys(df)

    # --- Step 2: Pre-select followers ----------------------------------------
    if not open_buys.empty:
        followers = preselect_followers(
            open_buys,
            min_trade_value=min_follower_trade_value,
            min_copyable_roi=min_follower_copyable_roi,
            min_open_buys=min_follower_open_buys,
        )
    else:
        followers = pd.DataFrame(
            columns=["wallet", "avg_copyable_roi", "total_copyable_pnl",
                      "total_trade_value", "open_buy_count"]
        )

    # --- Step 3: Detect leader-follower pairs --------------------------------
    if not open_buys.empty and not followers.empty:
        interactions = detect_leader_follower_pairs(
            open_buys,
            followers,
            time_window_minutes=time_window_minutes,
        )
        pair_stats = aggregate_leader_follower_pairs(
            interactions, min_interactions=min_pair_interactions
        )
        leaders = rank_leaders(pair_stats)
    else:
        leaders = pd.DataFrame(
            columns=["wallet", "leader_score", "num_followers",
                      "total_interactions", "total_follower_copyable_pnl",
                      "unique_tokens_traded"]
        )

    # --- Step 4: Per-wallet stats for openers/closers ------------------------
    wallet_stats = _compute_wallet_trade_stats(df)

    # --- Step 5: Select openers and closers ----------------------------------
    openers = select_openers(
        wallet_stats,
        min_trade_pnl=min_opener_trade_pnl,
        min_trade_roi=min_opener_trade_roi,
        max_market_pnl_hhi=max_opener_hhi,
        min_copyable_open_roi=min_opener_copyable_open_roi,
        min_trade_count=min_opener_trade_count,
        top_n=top_n_openers,
    )

    closers = select_closers(
        wallet_stats,
        min_copyable_sell_pnl=min_closer_copyable_sell_pnl,
        min_sell_count=min_closer_sell_count,
        top_n=top_n_closers,
    )

    return {
        "openers": openers,
        "leaders": leaders,
        "followers": followers,
        "closers": closers,
    }
