"""
Leader-follower pattern detection for copy-trading wallet groups.

Detects directed relationships where wallet B (follower) buys the same token
as wallet A (leader) within a short time window.  The detection is seeded by
pre-selecting followers with high copyable ROI liquidity, then scanning for
leader candidates that precede their buys.

Algorithm
---------
1. Pre-select follower wallets: open_buy trades with avg copyable ROI ≥ threshold.
2. For each follower open_buy on (condition_id, outcome) at time t_F:
   scan earlier open_buys on the same (condition_id, outcome) within
   [t_F − window, t_F) to find leader candidates.
3. Aggregate (leader, follower) pairs, filter by min_interactions.
4. Weight by follower's copyable_pnl.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Step 1: Pre-select followers
# ---------------------------------------------------------------------------

def preselect_followers(
    open_buys: pd.DataFrame,
    *,
    min_trade_value: float = 100.0,
    min_copyable_roi: float = 0.10,
    min_open_buys: int = 10,
) -> pd.DataFrame:
    """Select follower wallets with high copyable ROI liquidity.

    Parameters
    ----------
    open_buys:
        DataFrame of open_buy events.  Must contain ``wallet``,
        ``usdc_amount`` (or ``trade_value_usdc``), ``copyable_pnl``,
        ``total_quantity`` (or ``quantity``).
    min_trade_value:
        Minimum total trade value (USDC) to be eligible.
    min_copyable_roi:
        Minimum average copyable ROI (= sum(copyable_pnl) / sum(trade_value)).
    min_open_buys:
        Minimum number of open_buy events.

    Returns
    -------
    DataFrame with columns: ``wallet``, ``avg_copyable_roi``,
    ``total_copyable_pnl``, ``total_trade_value``, ``open_buy_count``.
    Sorted descending by ``avg_copyable_roi``.
    """
    work = open_buys.copy()

    # Normalise column names for fill-level vs grouped schemas.
    if "trade_value_usdc" not in work.columns and "usdc_amount" in work.columns:
        work["trade_value_usdc"] = work["usdc_amount"]
    if "total_quantity" not in work.columns and "quantity" in work.columns:
        work["total_quantity"] = work["quantity"]

    wallet_agg = (
        work.groupby("wallet")
        .agg(
            total_copyable_pnl=("copyable_pnl", "sum"),
            total_trade_value=("trade_value_usdc", "sum"),
            open_buy_count=("wallet", "size"),
        )
        .reset_index()
    )

    wallet_agg["avg_copyable_roi"] = (
        wallet_agg["total_copyable_pnl"]
        / wallet_agg["total_trade_value"].clip(lower=1e-9)
    )

    eligible = wallet_agg[
        (wallet_agg["total_trade_value"] >= min_trade_value)
        & (wallet_agg["avg_copyable_roi"] >= min_copyable_roi)
        & (wallet_agg["open_buy_count"] >= min_open_buys)
    ].copy()

    return eligible.sort_values("avg_copyable_roi", ascending=False).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Step 2: Detect leader-follower pairs
# ---------------------------------------------------------------------------

def detect_leader_follower_pairs(
    open_buys: pd.DataFrame,
    follower_wallets: pd.DataFrame,
    *,
    time_window_minutes: int = 10,
) -> pd.DataFrame:
    """Scan for leader→follower BUY-after-BUY patterns within a time window.

    For every open_buy by a *follower* wallet on token T at time t_F, look
    backwards within ``[t_F − window, t_F)`` for open_buys by any other wallet
    on the same ``(condition_id, outcome)``.  Each such earlier buy is a
    leader-candidate interaction.

    Parameters
    ----------
    open_buys:
        All open_buy events.  Must contain ``wallet``, ``condition_id``,
        ``outcome``, ``dt``, ``trade_value_usdc`` (or ``usdc_amount``),
        ``copyable_pnl``.
    follower_wallets:
        Output of :func:`preselect_followers`.  Must contain ``wallet``.
    time_window_minutes:
        Maximum time delta (in minutes) between leader and follower buys.

    Returns
    -------
    DataFrame with columns: ``leader``, ``follower``, ``condition_id``,
    ``outcome``, ``leader_dt``, ``follower_dt``, ``time_delta_seconds``,
    ``follower_copyable_pnl``, ``follower_trade_value``.
    One row per detected interaction.
    """
    follower_set = set(follower_wallets["wallet"])
    work = open_buys[open_buys["wallet"].isin(follower_set)].copy()

    if work.empty:
        return pd.DataFrame(
            columns=[
                "leader", "follower", "condition_id", "outcome",
                "leader_dt", "follower_dt", "time_delta_seconds",
                "follower_copyable_pnl", "follower_trade_value",
            ]
        )

    work["dt"] = pd.to_datetime(work["dt"], utc=True)

    if "trade_value_usdc" not in work.columns and "usdc_amount" in work.columns:
        work["trade_value_usdc"] = work["usdc_amount"]

    # Also need the full set of open_buys for leader candidates.
    all_ob = open_buys.copy()
    all_ob["dt"] = pd.to_datetime(all_ob["dt"], utc=True)
    if "trade_value_usdc" not in all_ob.columns and "usdc_amount" in all_ob.columns:
        all_ob["trade_value_usdc"] = all_ob["usdc_amount"]

    # Group everything by (condition_id, outcome) for windowed scans.
    interactions: list[dict] = []
    window = pd.Timedelta(minutes=time_window_minutes)

    # Process each (condition_id, outcome) group.
    groups = work.groupby(["condition_id", "outcome"], sort=False)
    all_groups = all_ob.groupby(["condition_id", "outcome"], sort=False)

    for key, fg in groups:
        if key not in all_groups.groups:
            continue
        ag = all_groups.get_group(key).sort_values("dt")

        # For each follower buy, find leader candidates via binary search.
        follower_buys = fg.sort_values("dt")
        # Convert to int64 nanoseconds for timezone-agnostic binary search.
        ag_dts = ag["dt"].astype("int64").values
        n = len(ag_dts)

        for _, frow in follower_buys.iterrows():
            f_dt = frow["dt"]
            f_wallet = frow["wallet"]
            cutoff = f_dt - window

            # Binary search for the left boundary: first index with dt >= cutoff.
            lo = int(np.searchsorted(ag_dts, np.int64(cutoff.value), side="left"))
            # All entries in [lo, hi) are within the window.
            hi = int(np.searchsorted(ag_dts, np.int64(f_dt.value), side="left"))

            for idx in range(lo, hi):
                lrow = ag.iloc[idx]
                if lrow["wallet"] == f_wallet:
                    continue  # skip self
                interactions.append(
                    {
                        "leader": lrow["wallet"],
                        "follower": f_wallet,
                        "condition_id": key[0],
                        "outcome": key[1],
                        "leader_dt": lrow["dt"],
                        "follower_dt": f_dt,
                        "time_delta_seconds": (f_dt - lrow["dt"]).total_seconds(),
                        "follower_copyable_pnl": float(frow.get("copyable_pnl", 0.0)),
                        "follower_trade_value": float(frow.get("trade_value_usdc", 0.0)),
                    }
                )

    return pd.DataFrame(interactions)


# ---------------------------------------------------------------------------
# Step 3: Aggregate and rank
# ---------------------------------------------------------------------------

def aggregate_leader_follower_pairs(
    interactions: pd.DataFrame,
    *,
    min_interactions: int = 10,
) -> pd.DataFrame:
    """Aggregate detected interactions into (leader, follower) pair stats.

    Parameters
    ----------
    interactions:
        Output of :func:`detect_leader_follower_pairs`.
    min_interactions:
        Minimum number of observed interactions to keep a pair.

    Returns
    -------
    DataFrame with columns: ``leader``, ``follower``, ``interaction_count``,
    ``total_follower_copyable_pnl``, ``total_follower_trade_value``,
    ``avg_time_delta_seconds``, ``unique_tokens``.
    Sorted descending by ``total_follower_copyable_pnl``.
    """
    if interactions.empty:
        return interactions

    grouped = (
        interactions.groupby(["leader", "follower"], sort=False)
        .agg(
            interaction_count=("follower", "size"),
            total_follower_copyable_pnl=("follower_copyable_pnl", "sum"),
            total_follower_trade_value=("follower_trade_value", "sum"),
            avg_time_delta_seconds=("time_delta_seconds", "mean"),
            unique_tokens=("condition_id", "nunique"),
        )
        .reset_index()
    )

    eligible = grouped[grouped["interaction_count"] >= min_interactions].copy()
    return eligible.sort_values(
        "total_follower_copyable_pnl", ascending=False
    ).reset_index(drop=True)


def rank_leaders(
    pair_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-leader aggregate scores from pair statistics.

    A leader's score is the sum of ``total_follower_copyable_pnl`` across all
    their follower pairs, weighted by interaction count.

    Parameters
    ----------
    pair_stats:
        Output of :func:`aggregate_leader_follower_pairs`.

    Returns
    -------
    DataFrame with columns: ``wallet``, ``leader_score``,
    ``num_followers``, ``total_interactions``, ``total_follower_copyable_pnl``,
    ``unique_tokens_traded``.
    Sorted descending by ``leader_score``.
    """
    if pair_stats.empty:
        return pd.DataFrame(
            columns=[
                "wallet", "leader_score", "num_followers",
                "total_interactions", "total_follower_copyable_pnl",
                "unique_tokens_traded",
            ]
        )

    leader_agg = (
        pair_stats.groupby("leader", sort=False)
        .agg(
            leader_score=("total_follower_copyable_pnl", "sum"),
            num_followers=("follower", "nunique"),
            total_interactions=("interaction_count", "sum"),
            total_follower_copyable_pnl=("total_follower_copyable_pnl", "sum"),
            unique_tokens_traded=("unique_tokens", "sum"),
        )
        .reset_index()
        .rename(columns={"leader": "wallet"})
    )

    return leader_agg.sort_values("leader_score", ascending=False).reset_index(
        drop=True
    )
