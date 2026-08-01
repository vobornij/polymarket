"""
Shared library for stage1/2/3 wallet-selection notebooks.

Provides:
- Data loading & preprocessing
- Wallet volatility metrics
- Copyable wallet selection/scoring
- Train/val/test splitting
- Copy-trading simulation engine
- Result printing helpers
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

# Ensure the project's src/ is importable (polymarket_analysis lives there)
_SRC = str(Path(__file__).resolve().parent.parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import pandas as pd

try:
    from _twopass_impl import _pass1_leader_scores, _pass2_follower_scores, _warmup
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
DEFAULT_TRADES_DIR = (_HERE / "../../data/polygon_trades_processed").resolve()
DEFAULT_WORKSPACE_DIR = (_HERE / "../../data/trade_signals_workspace_v2").resolve()
# DEFAULT_TAGS = {"Politics"}
DEFAULT_TAGS = {"Weather"}

RESULTS_DIR = Path(__file__).parent  # same directory as this file

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------


def load_markets(trades_dir: Path = DEFAULT_TRADES_DIR, tags: set[str] | None = DEFAULT_TAGS) -> pd.DataFrame:
    """Load and filter market metadata."""
    from polymarket_analysis.data.data_catalogue import load_markets_processed

    mdf = load_markets_processed()
    print(f"Markets: {len(mdf)}")

    mdf = mdf[
        ~mdf["primary_tag"].isin(["Sports", "Crypto"])
        & (mdf["winner_token_id"].notna())
    ]
    if tags is not None:
        mdf = mdf[
            mdf["tags"].apply(lambda t: any(tag in tags for tag in t))
        ].copy()

    print(f"Filtered markets for {tags}: {len(mdf)}")
    return mdf


def load_trades(
    trades_dir: Path = DEFAULT_TRADES_DIR,
    mdf: pd.DataFrame | None = None,
    tags: set[str] | None = DEFAULT_TAGS,
) -> pd.DataFrame:
    """Load trade shards, join with markets, clean, and compute PnL columns."""
    if mdf is None:
        mdf = load_markets(trades_dir, tags)

    trade_files = sorted(trades_dir.glob("*.parquet"))
    print(f"Loading {len(trade_files)} trade shards...")

    df_full = pd.concat(
        [pd.read_parquet(f).merge(mdf, on="condition_id", how="inner") for f in trade_files],
        ignore_index=True,
    )

    if tags is not None:
        df_full = df_full[df_full["primary_tag"].isin(tags)].copy().reset_index(drop=True)

    # Deduplicate outcome columns from merge
    if "outcome_x" in df_full.columns:
        df_full["outcome"] = df_full["outcome_x"]
        del df_full["outcome_x"], df_full["outcome_y"]

    df_full["dt"] = pd.to_datetime(df_full["dt"], utc=True)

    # Normalise grouped schema
    if "total_quantity" in df_full.columns and "quantity" not in df_full.columns:
        df_full = df_full.rename(columns={
            "total_quantity": "quantity",
            "avg_price": "price",
            "trade_value_usdc": "usdc_amount",
        })

    df_full["usdc_amount"] = df_full["usdc_amount"].astype(float)
    df_full["final_value_usdc"] = df_full["final_value_usdc"].astype(float)
    df_full["quantity"] = df_full["quantity"].astype(float)

    # PnL and notional
    if "trade_pnl" in df_full.columns:
        df_full["pnl"] = df_full["trade_pnl"]  # alias for backward compat
    else:
        df_full["pnl"] = np.where(
            df_full["side"] == "BUY",
            df_full["final_value_usdc"] - df_full["usdc_amount"],
            df_full["usdc_amount"] - df_full["final_value_usdc"],
        )
    df_full["notional"] = np.where(
        df_full["side"] == "BUY",
        df_full["usdc_amount"],
        df_full["quantity"] * (1 - df_full["price"].astype(float)),
    )

    print(f"Total trades loaded: {len(df_full):,}")
    print(f"Unique wallets: {df_full['wallet'].nunique():,}")
    print(f"Date range: {df_full['dt'].min()} -> {df_full['dt'].max()}")
    return df_full


# ---------------------------------------------------------------------------
# 2. Train / val / test split
# ---------------------------------------------------------------------------


def split_data(
    df_full: pd.DataFrame,
    method: str = "random",
    train_pct: float = 0.40,
    val_pct: float = 0.30,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split trades into train / val / test by end date buckets.

    Each unique ``end_date_iso`` is assigned to a bucket so all trades
    for that date land in the same split.

    Parameters
    ----------
    method : {"random", "chronological"}
        - "random": shuffle dates then assign to buckets.
        - "chronological": sort dates ascending, split at market-count
          percentile boundaries.
    """
    market_end_dates = df_full.groupby("condition_id")["end_date_iso"].first()
    unique_dates = market_end_dates.unique()
    n = len(unique_dates)
    n_train = int(n * train_pct)
    n_val = int(n * val_pct)

    if method == "random":
        raise RuntimeError('only for testing')
        rng = np.random.RandomState(seed)
        rng.shuffle(unique_dates)
        date_to_split = {}
        for d in unique_dates[:n_train]:
            date_to_split[d] = "train"
        for d in unique_dates[n_train:n_train + n_val]:
            date_to_split[d] = "val"
        for d in unique_dates[n_train + n_val:]:
            date_to_split[d] = "test"
    elif method == "chronological":
        sorted_dates = np.sort(unique_dates)
        train_end = sorted_dates[n_train - 1]
        val_end = sorted_dates[n_train + n_val - 1]

        print(f"Chronological split: train <= {train_end}, val <= {val_end}, test > {val_end}")
        date_to_split = {}
        for d in sorted_dates:
            if d <= train_end:
                date_to_split[d] = "train"
            elif d <= val_end:
                date_to_split[d] = "val"
            else:
                date_to_split[d] = "test"
    else:
        raise ValueError(f"Unknown method {method!r}, expected 'random' or 'chronological'")

    split_col = df_full["end_date_iso"].map(date_to_split)
    df_train = df_full[split_col == "train"].copy()
    df_val = df_full[split_col == "val"].copy()
    df_test = df_full[split_col == "test"].copy()

    print(f"Method: {method}  |  Unique end dates: {n}  (train={n_train}, val={n_val}, test={n - n_train - n_val})")
    print()
    print(f"  Train: {len(df_train):>10,} trades  ({df_train['condition_id'].nunique():>5,} markets)")
    print(f"  Val:   {len(df_val):>10,} trades  ({df_val['condition_id'].nunique():>5,} markets)")
    print(f"  Test:  {len(df_test):>10,} trades  ({df_test['condition_id'].nunique():>5,} markets)")
    print(f"  Total: {len(df_full):>10,} trades  ({df_full['condition_id'].nunique():>5,} markets)")
    return df_train, df_val, df_test


def compute_copyable_notional(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``copyable_notional``, ``copyable_roi``, ``roi`` columns.

    ``copyable_notional`` is the fraction of each trade's notional that
    corresponds to the copyable portion of PnL (matches reference notebook).
    """
    df = df.copy()
    df["copyable_notional"] = df["notional"] * (
        df["copyable_pnl"] / df["pnl"].replace(0, np.nan)
    )
    df["roi"] = df["pnl"] / df["notional"].replace(0, np.nan)
    df["copyable_roi"] = df["copyable_pnl"] / df["copyable_notional"].replace(0, np.nan)
    return df


def compute_opening_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-wallet opening-BUY metrics and return as a DataFrame.

    An opening buy is a BUY where ``position == quantity`` (wallet had no
    prior position).  Returns one row per wallet with columns:
    ``wallet``, ``opening_pnl``, ``opening_notional``, ``opening_roi``,
    ``opening_copyable_pnl``, ``opening_copyable_notional``,
    ``opening_copyable_roi``, ``opening_buys``.
    """
    buys = df[(df["side"] == "BUY") & (df["position"] == df["quantity"])].copy()
    if buys.empty:
        return pd.DataFrame(columns=[
            "wallet", "opening_pnl", "opening_notional", "opening_roi",
            "opening_copyable_pnl", "opening_copyable_notional",
            "opening_copyable_roi", "opening_buys",
        ])

    agg = buys.groupby("wallet").agg(
        opening_pnl=("pnl", "sum"),
        opening_notional=("notional", "sum"),
        opening_copyable_pnl=("copyable_pnl", "sum"),
        opening_buys=("wallet", "size"),
    ).reset_index()

    if "copyable_notional" in buys.columns:
        cn_agg = buys.groupby("wallet")["copyable_notional"].sum().reset_index()
        cn_agg = cn_agg.rename(columns={"copyable_notional": "opening_copyable_notional"})
        agg = agg.merge(cn_agg, on="wallet", how="left")
    else:
        agg["opening_copyable_notional"] = np.nan

    agg["opening_roi"] = agg["opening_pnl"] / agg["opening_notional"].clip(lower=1e-9)
    agg["opening_copyable_roi"] = (
        agg["opening_copyable_pnl"] / agg["opening_copyable_notional"].clip(lower=1e-9)
    )
    return agg


def select_copyable_group(
    wallet_vol: pd.DataFrame,
    *,
    min_buy_roi: float,
    min_num_buckets: int,
    min_num_markets: int,
    max_drawdown_to_pnl: float,
    max_top_market_pnl_pct: float,
    max_market_pnl_hhi: float,
    min_total_notional: float,
    min_opening_roi: float,
    min_opening_pnl: float,
    min_opening_copyable_roi: float,
) -> pd.DataFrame:
    """Select copyable wallets by thresholds, sorted by opening_copyable_roi."""
    c = wallet_vol.copy()

    mask = (
        (c["buy_roi"] >= min_buy_roi)
        & (c["num_buckets"] >= min_num_buckets)
        & (c["num_markets"] >= min_num_markets)
        & (c["max_drawdown_to_pnl"] <= max_drawdown_to_pnl)
        & (c["top_market_pnl_pct"] < max_top_market_pnl_pct)
        & (c["market_pnl_hhi"].fillna(0.20) < max_market_pnl_hhi)
        & (c["total_notional"] >= min_total_notional)
        & (c["copyable_pnl"] > 0)
        & (c["opening_roi"] >= min_opening_roi)
        & (c["opening_pnl"] >= min_opening_pnl)
        & (c["opening_copyable_roi"] >= min_opening_copyable_roi)
    )
    result = c[mask].copy()
    return result.sort_values("opening_copyable_roi", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3b. Implied-trade selection helpers (leader → follower BUY detection)
# ---------------------------------------------------------------------------


def select_follower_wallets(
    wallet_vol: pd.DataFrame,
    *,
    min_copyable_roi: float = 0.05,
    min_trade_value: float = 100.0,
    min_num_buckets: int = 10,
    max_market_pnl_hhi: float = 0.3,
) -> pd.DataFrame:
    """Select follower wallets with positive copyable ROI and sufficient activity.

    Returns wallets sorted by ``copyable_roi`` descending.
    """
    c = wallet_vol.copy()
    mask = (
        (c["copyable_pnl"] > 0)
        & (c["copyable_roi"] >= min_copyable_roi)
        & (c["total_notional"] >= min_trade_value)
        & (c["num_buckets"] >= min_num_buckets)
        & (c["market_pnl_hhi"].fillna(0.20) <= max_market_pnl_hhi)
    )
    result = c[mask].copy()
    return result.sort_values("copyable_roi", ascending=False).reset_index(drop=True)


def select_leader_wallets(
    wallet_vol: pd.DataFrame,
    *,
    min_trade_count: int = 10,
    min_roi: float = 0.0,
    max_market_pnl_hhi: float = 0.5,
    side: str = "BUY",
) -> pd.DataFrame:
    """Select leader wallets filtered by side-specific trade performance.

    Parameters
    ----------
    side:
        ``"BUY"`` filters by ``buy_roi``; ``"SELL"`` filters by ``sell_roi``.
    """
    c = wallet_vol.copy()
    roi_col = "buy_roi" if side == "BUY" else "sell_roi"

    if min_roi is not None and (roi_col not in c.columns or roi_col not in c.columns):
        return pd.DataFrame(columns=["wallet"])

    mask = (
        (min_roi is None or c[roi_col].fillna(-1) >= min_roi)
        & (c['trade_count'].fillna(0) >= min_trade_count)
        & (c["market_pnl_hhi"].fillna(0.20) <= max_market_pnl_hhi)
        & (c["total_pnl"] > 0)
    )
    result = c[mask].copy()
    return result.sort_values(roi_col, ascending=False).reset_index(drop=True)


def detect_implied_buys(
    df: pd.DataFrame,
    follower_wallets: set[str],
    leader_wallets: set[str],
    *,
    time_window_minutes: int = 5,
    leader_side: str = "BUY",
) -> pd.DataFrame:
    """Detect follower BUYs that follow a leader trade within a time window.

    For each BUY by a *follower* wallet on token T at time t_F, look backwards
    within ``[t_F − window, t_F)`` for trades by *leader* wallets with
    ``side == leader_side`` on the same ``(condition_id, outcome)``.

    Returns DataFrame with one row per detected implied trade.
    """
    follower_buys = df[
        (df["wallet"].isin(follower_wallets))
        & (df["side"] == "BUY")
    ][["wallet", "condition_id", "outcome", "dt", "pnl", "notional", "copyable_pnl", "copyable_notional"]].copy()

    if follower_buys.empty:
        return pd.DataFrame(columns=[
            "leader_wallet", "follower_wallet", "condition_id", "outcome",
            "follower_dt", "leader_dt", "time_delta_seconds",
            "pnl", "copyable_pnl", "notional", "copyable_notional",
        ])

    leader_trades = df[
        (df["wallet"].isin(leader_wallets))
        & (df["side"] == leader_side)
    ][["wallet", "condition_id", "outcome", "dt"]].copy()

    if leader_trades.empty:
        return pd.DataFrame(columns=[
            "leader_wallet", "follower_wallet", "condition_id", "outcome",
            "follower_dt", "leader_dt", "time_delta_seconds",
            "pnl", "copyable_pnl", "notional", "copyable_notional",
        ])

    leader_trades = leader_trades.rename(columns={"wallet": "leader_wallet"})

    leader_trades = leader_trades.assign(leader_dt=leader_trades["dt"])


    merged = pd.merge_asof(
        follower_buys.sort_values("dt"),
        leader_trades.sort_values("dt"),
        on="dt",
        by=["condition_id", "outcome"],
        direction="backward",
        tolerance=pd.Timedelta(minutes=time_window_minutes),
        allow_exact_matches=False,
        suffixes=("", "_leader"),
    )

    implied = merged.loc[merged["leader_wallet"].notna()].copy()
    if implied.empty:
        return pd.DataFrame(columns=[
            "leader_wallet", "follower_wallet", "condition_id", "outcome",
            "follower_dt", "leader_dt", "time_delta_seconds",
            "pnl", "copyable_pnl", "notional", "copyable_notional",
        ])

    implied = implied.rename(columns={"wallet": "follower_wallet", "dt": "follower_dt"})
    implied["time_delta_seconds"] = (
        implied["follower_dt"] - implied["leader_dt"]
    ).dt.total_seconds()

    implied = implied[implied["follower_wallet"] != implied["leader_wallet"]]

    return implied.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3b-i-b. Two-pass scoring (numba-accelerated, no pair materialization)
# ---------------------------------------------------------------------------


def compute_pair_scores(
    df: pd.DataFrame,
    follower_wallets: set[str],
    leader_wallets: set[str],
    *,
    time_window_minutes: int = 5,
    leader_side: str = "BUY",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-leader and per-follower aggregate scores via two-pass sliding window.

    Pass 1: for each leader trade, find all followers in [lt-tw, lt) and sum their
    copyable_pnl. Pass 2: for each follower trade, find all leaders in [ft-tw, ft)
    and sum their leader scores from pass 1.

    Returns (leader_df, follower_df) where:
      leader_df columns: leader_wallet, leader_score, leader_num_trades, leader_avg_distinct_followers
      follower_df columns: follower_wallet, follower_score, follower_num_trades, follower_avg_distinct_leaders
    """
    if not _HAS_NUMBA:
        raise ImportError("_twopass_impl module not available (numba required)")

    if not leader_wallets or not follower_wallets:
        empty_l = pd.DataFrame(columns=["leader_wallet", "leader_score", "leader_num_trades", "leader_avg_distinct_followers"])
        empty_f = pd.DataFrame(columns=["follower_wallet", "follower_score", "follower_num_trades", "follower_avg_distinct_leaders"])
        return empty_l, empty_f

    tw_ns = int(pd.Timedelta(minutes=time_window_minutes).total_seconds() * 1e9)

    fb = df[
        (df["wallet"].isin(follower_wallets))
        & (df["side"] == "BUY")
        & (df["copyable_notional"] > 0)
    ][["wallet", "condition_id", "outcome", "dt", "copyable_pnl"]].copy()

    lt = df[
        (df["wallet"].isin(leader_wallets))
        & (df["side"] == leader_side)
    ][["wallet", "condition_id", "outcome", "dt"]].copy()

    if fb.empty or lt.empty:
        empty_l = pd.DataFrame(columns=["leader_wallet", "leader_score", "leader_num_trades", "leader_avg_distinct_followers"])
        empty_f = pd.DataFrame(columns=["follower_wallet", "follower_score", "follower_num_trades", "follower_avg_distinct_leaders"])
        return empty_l, empty_f

    fb["dt_ns"] = fb["dt"].astype(np.int64)
    lt["dt_ns"] = lt["dt"].astype(np.int64)

    # Global wallet -> int mapping
    all_wallets = np.concatenate([fb["wallet"].values, lt["wallet"].values])
    _, wallet_inv = np.unique(all_wallets, return_inverse=True)
    wallet_ids = wallet_inv.astype(np.int32)
    fb["wallet_id"] = wallet_ids[: len(fb)]
    lt["wallet_id"] = wallet_ids[len(fb) :]

    # Warmup numba on first call
    if not getattr(compute_pair_scores, "_warmed_up", False):
        _warmup()
        compute_pair_scores._warmed_up = True

    # Collect raw results then aggregate with pandas (avoids per-trade dict ops)
    lw_str_parts = []
    lt_score_parts = []
    lt_nd_parts = []
    fw_str_parts = []
    ft_score_parts = []
    ft_nd_parts = []

    fb_groups: dict[tuple, tuple] = {}
    for (cid, out), fg in fb.groupby(["condition_id", "outcome"], sort=False):
        fb_groups[(cid, out)] = (
            fg["wallet_id"].values.astype(np.int32),
            fg["wallet"].values,
            fg["dt_ns"].values,
            fg["copyable_pnl"].values,
        )

    lt_groups: dict[tuple, tuple] = {}
    for (cid, out), lg in lt.groupby(["condition_id", "outcome"], sort=False):
        lt_groups[(cid, out)] = (
            lg["wallet_id"].values.astype(np.int32),
            lg["wallet"].values,
            lg["dt_ns"].values,
        )

    for key in set(fb_groups) & set(lt_groups):
        fw_id, fw_str, fi_dt, fg_cp = fb_groups[key]
        lw_id, lw_str, li_dt = lt_groups[key]

        order_f = np.argsort(fi_dt)
        order_l = np.argsort(li_dt)
        fi_s = fi_dt[order_f]
        li_s = li_dt[order_l]

        if len(fi_s) == 0 or len(li_s) == 0:
            continue

        s1, d1 = _pass1_leader_scores(fi_s, li_s, fw_id[order_f], fg_cp[order_f], tw_ns)
        s2, d2 = _pass2_follower_scores(fi_s, li_s, lw_id[order_l], s1, tw_ns)

        lw_str_parts.append(lw_str[order_l])
        lt_score_parts.append(s1)
        lt_nd_parts.append(d1)
        fw_str_parts.append(fw_str[order_f])
        ft_score_parts.append(s2)
        ft_nd_parts.append(d2)

    if not lt_score_parts:
        empty_l = pd.DataFrame(columns=["leader_wallet", "leader_score", "leader_num_trades", "leader_avg_distinct_followers"])
        empty_f = pd.DataFrame(columns=["follower_wallet", "follower_score", "follower_num_trades", "follower_avg_distinct_leaders"])
        return empty_l, empty_f

    leader_df = (
        pd.DataFrame({
            "leader_wallet": np.concatenate(lw_str_parts),
            "score": np.concatenate(lt_score_parts),
            "nd": np.concatenate(lt_nd_parts),
        })
        .groupby("leader_wallet", sort=False)
        .agg(
            leader_score=("score", "sum"),
            leader_num_trades=("score", "size"),
            leader_avg_distinct_followers=("nd", "mean"),
        )
        .reset_index()
        .sort_values("leader_score", ascending=False)
        .reset_index(drop=True)
    )

    follower_df = (
        pd.DataFrame({
            "follower_wallet": np.concatenate(fw_str_parts),
            "score": np.concatenate(ft_score_parts),
            "nd": np.concatenate(ft_nd_parts),
        })
        .groupby("follower_wallet", sort=False)
        .agg(
            follower_score=("score", "sum"),
            follower_num_trades=("score", "size"),
            follower_avg_distinct_leaders=("nd", "mean"),
        )
        .reset_index()
        .sort_values("follower_score", ascending=False)
        .reset_index(drop=True)
    )

    return leader_df, follower_df


def evaluate_follower_buy_performance(
    df: pd.DataFrame,
    follower_wallets: set[str],
) -> dict:
    """Compute total wallet and copyable PnL/ROI across ALL buy trades by follower wallets.

    Returns aggregate buy-trade performance regardless of leader presence.
    """
    if not follower_wallets:
        return {"wallet_pnl": 0.0, "wallet_roi": 0.0, "wallet_notional": 0.0,
                "followed_copyable_pnl": 0.0, "followed_copyable_roi": 0.0, "trade_count": 0}

    buys = df[
        (df["wallet"].isin(follower_wallets))
        & (df["side"] == "BUY")
    ]
    if buys.empty:
        return {"wallet_pnl": 0.0, "wallet_roi": 0.0, "wallet_notional": 0.0,
                "followed_copyable_pnl": 0.0, "followed_copyable_roi": 0.0, "trade_count": 0}

    notional = float(buys["notional"].sum())
    pnl = float(buys["pnl"].sum())
    cpnl = float(buys["copyable_pnl"].sum()) if "copyable_pnl" in buys.columns else 0.0
    cnot = float(buys["copyable_notional"].sum()) if "copyable_notional" in buys.columns else 0.0
    return {
        "wallet_pnl": pnl,
        "wallet_roi": pnl / notional if notional > 0 else 0.0,
        "wallet_notional": notional,
        "followed_copyable_pnl": cpnl,
        "followed_copyable_roi": cpnl / cnot if cnot > 0 else 0.0,
        "trade_count": len(buys),
    }


def evaluate_leader_performance(
    df: pd.DataFrame,
    leader_wallets: set[str],
    *,
    side: str = "BUY",
) -> dict:
    """Compute total PnL/ROI for leader wallets on their own trades of *side*."""
    if not leader_wallets:
        return {"pnl": 0.0, "roi": 0.0, "notional": 0.0, "trade_count": 0}

    trades = df[
        (df["wallet"].isin(leader_wallets))
        & (df["side"] == side)
    ]
    if trades.empty:
        return {"pnl": 0.0, "roi": 0.0, "notional": 0.0, "trade_count": 0}

    notional = float(trades["notional"].sum())
    pnl = float(trades["pnl"].sum())
    return {
        "pnl": pnl,
        "roi": pnl / notional if notional > 0 else 0.0,
        "notional": notional,
        "trade_count": len(trades),
    }


def evaluate_leader_followed_performance(
    df: pd.DataFrame,
    implied: pd.DataFrame,
    leader_wallets: set[str],
    *,
    leader_side: str = "BUY",
) -> dict:
    """Compute PnL/ROI for leader trades that were actually followed.

    *implied* is the output of :func:`detect_implied_buys` for the
    corresponding *leader_side*.  Leader trades are deduplicated
    (one row per leader_wallet × condition_id × outcome × leader_dt)
    and joined back to *df* to retrieve their PnL.
    """
    if not leader_wallets or implied.empty:
        return {"pnl": 0.0, "roi": 0.0, "notional": 0.0, "trade_count": 0}

    leader_trades = (
        implied[["leader_wallet", "condition_id", "outcome", "leader_dt"]]
        .drop_duplicates()
        .rename(columns={"leader_wallet": "wallet", "leader_dt": "dt"})
    )

    merged = leader_trades.merge(
        df[["wallet", "condition_id", "outcome", "dt", "pnl", "notional"]],
        on=["wallet", "condition_id", "outcome", "dt"],
        how="inner",
    )
    if merged.empty:
        return {"pnl": 0.0, "roi": 0.0, "notional": 0.0, "trade_count": 0}

    notional = float(merged["notional"].sum())
    pnl = float(merged["pnl"].sum())
    return {
        "pnl": pnl,
        "roi": pnl / notional if notional > 0 else 0.0,
        "notional": notional,
        "trade_count": len(merged),
    }


def score_leaders(implied_trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate implied trades into per-leader scores.

    Returns DataFrame with columns: ``leader_wallet``, ``num_followers``,
    ``total_follower_copyable_pnl``, ``num_followed_trades``, ``unique_tokens``.
    Sorted descending by ``total_follower_copyable_pnl``.
    """
    if implied_trades.empty:
        return pd.DataFrame(columns=[
            "leader_wallet", "num_followers", "total_follower_copyable_pnl",
            "num_followed_trades", "unique_tokens",
        ])

    agg = (
        implied_trades.groupby("leader_wallet", sort=False)
        .agg(
            num_followers=("follower_wallet", "nunique"),
            total_follower_copyable_pnl=("copyable_pnl", "sum"),
            num_followed_trades=("follower_wallet", "size"),
            unique_tokens=("condition_id", "nunique"),
        )
        .reset_index()
    )
    return agg.sort_values("total_follower_copyable_pnl", ascending=False).reset_index(drop=True)


def evaluate_implied_pnl(
    df: pd.DataFrame,
    follower_wallets: set[str],
    leader_wallets: set[str],
    *,
    time_window_minutes: int = 5,
    leader_side: str = "BUY",
) -> dict:
    """Detect implied trades on *df* and return aggregate copyable PnL stats."""
    implied = detect_implied_buys(
        df, follower_wallets, leader_wallets,
        time_window_minutes=time_window_minutes,
        leader_side=leader_side,
    )
    if implied.empty:
        return {
            "pnl": 0.0,
            "roi": 0.0,
            "followed_copyable_pnl": 0.0,
            "followed_copyable_notional": 0.0,
            "followed_copyable_roi": 0.0,
            "wallet_pnl": 0.0,
            "wallet_roi": 0.0,
            "wallet_notional": 0.0,
            "trade_count": 0,
            "wallet_count": 0,
            "leader_count": 0,
        }
    wallet_notional = float(implied["notional"].sum())
    copyable_notional = float(implied["copyable_notional"].sum())
    return {
        "pnl": float(implied["pnl"].sum()),
        "roi": float(implied["pnl"].sum() / implied["notional"].sum()),
        "followed_copyable_pnl": float(implied["copyable_pnl"].sum()),
        "followed_copyable_notional": copyable_notional,
        "followed_copyable_roi": float(implied["copyable_pnl"].sum() / implied["copyable_notional"].sum()),
        "wallet_pnl": float(implied["pnl"].sum()),
        "wallet_roi": float(implied["pnl"].sum() / wallet_notional) if wallet_notional > 0 else 0.0,
        "wallet_notional": wallet_notional,
        "trade_count": len(implied),
        "wallet_count": int(implied["follower_wallet"].nunique()),
        "leader_count": int(implied["leader_wallet"].nunique()),
    }


# ---------------------------------------------------------------------------
# 3b-ii. Stability-based leader filtering
# ---------------------------------------------------------------------------


def filter_stable_leaders(
    df: pd.DataFrame,
    leader_wallets: set[str],
    follower_wallets: set[str],
    *,
    time_window_minutes: int = 10,
    leader_side: str = "BUY",
    n_splits: int = 3,
    min_profitable_splits: int = 2,
) -> set[str]:
    """Keep leaders whose followers are consistently profitable across time slices.

    Splits *df* into *n_splits* chronological chunks, detects implied trades
    in each, and keeps leaders where at least *min_profitable_splits* chunks
    produce positive total follower copyable PnL.
    """
    if not leader_wallets or not follower_wallets:
        return set()

    dt_min, dt_max = df["dt"].min(), df["dt"].max()
    edges = pd.date_range(dt_min, dt_max, periods=n_splits + 1, tz=dt_min.tz)

    leader_scores: dict[str, int] = {w: 0 for w in leader_wallets}
    for i in range(n_splits):
        chunk = df[(df["dt"] >= edges[i]) & (df["dt"] < edges[i + 1])]
        if chunk.empty:
            continue
        imp = detect_implied_buys(
            chunk, follower_wallets, leader_wallets,
            time_window_minutes=time_window_minutes, leader_side=leader_side,
        )
        if imp.empty:
            continue
        chunk_pnl = imp.groupby("leader_wallet")["copyable_pnl"].sum()
        for leader, pnl in chunk_pnl.items():
            if leader in leader_scores and pnl > 0:
                leader_scores[leader] += 1

    return {w for w, s in leader_scores.items() if s >= min_profitable_splits}


def filter_leaders_by_drawdown(
    df: pd.DataFrame,
    leader_wallets: set[str],
    follower_wallets: set[str],
    *,
    time_window_minutes: int = 10,
    leader_side: str = "BUY",
    max_dd_pnl_ratio: float = 0.3,
) -> set[str]:
    """Keep leaders whose followers have low drawdown relative to total copyable PnL.

    For each leader, computes the cumulative follower copyable PnL over time,
    then calculates max_drawdown / total_pnl. Leaders with ratio > max_dd_pnl_ratio
    are filtered out.
    """
    if not leader_wallets or not follower_wallets:
        return set()

    imp = detect_implied_buys(
        df, follower_wallets, leader_wallets,
        time_window_minutes=time_window_minutes, leader_side=leader_side,
    )
    if imp.empty:
        return set()

    imp = imp.sort_values("follower_dt")

    passed = set()
    for leader, grp in imp.groupby("leader_wallet"):
        cum_pnl = grp.set_index("follower_dt")["copyable_pnl"].cumsum()
        total_pnl = cum_pnl.iloc[-1]
        if total_pnl <= 0:
            continue

        running_max = cum_pnl.cummax()
        max_dd = abs((cum_pnl - running_max).min())
        if max_dd / total_pnl <= max_dd_pnl_ratio:
            passed.add(leader)

    return passed


def filter_followers_by_drawdown(
    buy_implied: pd.DataFrame,
    sell_implied: pd.DataFrame,
    *,
    max_dd_pnl_ratio: float = 0.3,
) -> set[str]:
    """Keep followers whose implied copyable PnL has low drawdown relative to total.

    For each follower, computes cumulative PnL over time from combined buy+sell
    implied trades. Filters out followers where max_drawdown / total_pnl > max_dd_pnl_ratio.
    """
    combined = pd.concat([buy_implied, sell_implied], ignore_index=True)
    if combined.empty:
        return set()

    combined = combined.sort_values("follower_dt")

    passed = set()
    for follower, grp in combined.groupby("follower_wallet"):
        cum_pnl = grp.set_index("follower_dt")["copyable_pnl"].cumsum()
        total_pnl = cum_pnl.iloc[-1]
        if total_pnl <= 0:
            continue

        running_max = cum_pnl.cummax()
        max_dd = abs((cum_pnl - running_max).min())
        if max_dd / total_pnl <= max_dd_pnl_ratio:
            passed.add(follower)

    return passed


def filter_followers_by_val_roi(
    df_val: pd.DataFrame,
    follower_wallets: set[str],
    buy_leader_wallets: set[str],
    sell_leader_wallets: set[str],
    *,
    time_window_minutes: int = 15,
    min_val_roi: float = 0.0,
) -> set[str]:
    """Keep followers whose implied copyable ROI >= min_val_roi on the validation split.

    Detects implied trades on *df_val* with the given leader/follower sets,
    computes each follower's copyable ROI, and returns only those meeting
    the threshold.  This uses the validation period as a generalization check.
    """
    if not follower_wallets:
        return set()

    buy_imp = detect_implied_buys(
        df_val, follower_wallets, buy_leader_wallets,
        time_window_minutes=time_window_minutes, leader_side="BUY",
    )
    sell_imp = detect_implied_buys(
        df_val, follower_wallets, sell_leader_wallets,
        time_window_minutes=time_window_minutes, leader_side="SELL",
    )
    combined = pd.concat([buy_imp, sell_imp], ignore_index=True)
    if combined.empty:
        return set()

    agg = combined.groupby("follower_wallet").agg(
        pnl=("copyable_pnl", "sum"),
        notional=("copyable_notional", "sum"),
    )
    agg["roi"] = agg["pnl"] / agg["notional"].clip(lower=1e-9)
    profitable = set(agg[agg["roi"] >= min_val_roi].index)
    return profitable & follower_wallets


def filter_pairs_by_frequency(
    implied: pd.DataFrame,
    min_observations: int = 3,
    min_total_pnl: float = 0.0,
) -> pd.DataFrame:
    """Keep only (follower, leader) pairs with enough observed followings.

    Returns the filtered implied DataFrame restricted to pairs that appear
    at least *min_observations* times and have total copyable_pnl >= *min_total_pnl*.
    """
    if implied.empty:
        return implied

    pair_stats = implied.groupby(["follower_wallet", "leader_wallet"], sort=False).agg(
        pair_count=("copyable_pnl", "size"),
        pair_pnl=("copyable_pnl", "sum"),
    ).reset_index()

    good_pairs = pair_stats[
        (pair_stats["pair_count"] >= min_observations)
        & (pair_stats["pair_pnl"] >= min_total_pnl)
    ][["follower_wallet", "leader_wallet"]]

    return implied.merge(good_pairs, on=["follower_wallet", "leader_wallet"], how="inner")


# ---------------------------------------------------------------------------
# 3b-iii. Iterative leader–follower refinement
# ---------------------------------------------------------------------------


def iterative_leader_follower_filter(
    df: pd.DataFrame,
    follower_wallets: set[str],
    buy_leader_wallets: set[str],
    sell_leader_wallets: set[str],
    *,
    time_window_minutes: int = 10,
    n_iterations: int = 1,
    leader_min_copyable_pnl: float | None = None,
    follower_min_copyable_pnl: float = 20.0,
    follower_min_copyable_roi: float | None = None,
    follower_max_market_hhi: float | None = None,
    follower_max_copyable_dd_ratio: float | None = None,
) -> tuple[set[str], set[str], set[str], list[dict], list[tuple[set[str], set[str], set[str]]]]:
    """Iteratively refine leader and follower sets by implied-trade profitability.

    Each iteration:

    1. Detect implied trades with current leader sets.
    2. Score leaders by their followers' total implied copyable PnL.
       Drop leaders below ``leader_min_copyable_pnl`` (skipped if *None*).
    3. Re-detect implied trades with the refined leader sets.
    4. Score followers by their own implied copyable PnL (and optionally
       copyable ROI, market HHI, drawdown ratio).  Drop followers below
       ``follower_min_copyable_pnl`` and, if set, ``follower_min_copyable_roi``,
       ``follower_max_market_hhi``, ``follower_max_copyable_dd_ratio``.

    Scoring is done on *df* (typically the training split).

    Returns
    -------
    (follower_wallets, buy_leader_wallets, sell_leader_wallets, log, snapshots)
        where *log* is a list of dicts with per-iteration stats, and
        *snapshots* is a list of (followers, buy_leaders, sell_leaders) tuples
        at each iteration (index 0 = before any refinement).
    """
    cur_followers = set(follower_wallets)
    cur_buy_leaders = set(buy_leader_wallets)
    cur_sell_leaders = set(sell_leader_wallets)
    n_iterations = int(n_iterations)

    log: list[dict] = []
    snapshots: list[tuple[set[str], set[str], set[str]]] = []

    def _snapshot() -> None:
        snapshots.append((set(cur_followers), set(cur_buy_leaders), set(cur_sell_leaders)))
        log.append({
            "iteration": len(snapshots) - 1,
            "followers": len(cur_followers),
            "buy_leaders": len(cur_buy_leaders),
            "sell_leaders": len(cur_sell_leaders),
        })

    _snapshot()

    for i in range(1, n_iterations + 1):
        # --- Step 1: detect implied trades with current leaders ---
        buy_implied = detect_implied_buys(
            df, cur_followers, cur_buy_leaders,
            time_window_minutes=time_window_minutes, leader_side="BUY",
        )
        sell_implied = detect_implied_buys(
            df, cur_followers, cur_sell_leaders,
            time_window_minutes=time_window_minutes, leader_side="SELL",
        )

        # --- Step 2: score & filter leaders ---
        if leader_min_copyable_pnl is not None:
            if not buy_implied.empty:
                buy_leader_scores = (
                    buy_implied.groupby("leader_wallet", sort=False)["copyable_pnl"]
                    .sum().reset_index()
                    .rename(columns={"copyable_pnl": "total_copyable_pnl"})
                )
                good_buy_leaders = set(
                    buy_leader_scores.loc[
                        buy_leader_scores["total_copyable_pnl"] >= leader_min_copyable_pnl,
                        "leader_wallet",
                    ]
                )
                cur_buy_leaders = cur_buy_leaders & good_buy_leaders

            if not sell_implied.empty:
                sell_leader_scores = (
                    sell_implied.groupby("leader_wallet", sort=False)["copyable_pnl"]
                    .sum().reset_index()
                    .rename(columns={"copyable_pnl": "total_copyable_pnl"})
                )
                good_sell_leaders = set(
                    sell_leader_scores.loc[
                        sell_leader_scores["total_copyable_pnl"] >= leader_min_copyable_pnl,
                        "leader_wallet",
                    ]
                )
                cur_sell_leaders = cur_sell_leaders & good_sell_leaders

        # --- Step 3: re-detect with refined leaders ---
        buy_implied = detect_implied_buys(
            df, cur_followers, cur_buy_leaders,
            time_window_minutes=time_window_minutes, leader_side="BUY",
        )
        sell_implied = detect_implied_buys(
            df, cur_followers, cur_sell_leaders,
            time_window_minutes=time_window_minutes, leader_side="SELL",
        )

        # --- Step 4: score & filter followers by copyable_pnl (+ optional roi, market hhi, dd ratio) ---
        combined = pd.concat([buy_implied, sell_implied], ignore_index=True)
        if not combined.empty:
            follower_scores = (
                combined.groupby("follower_wallet", sort=False)
                .agg(total_copyable_pnl=("copyable_pnl", "sum"),
                     total_copyable_notional=("copyable_notional", "sum"))
                .reset_index()
            )
            follower_scores["copyable_roi"] = (
                follower_scores["total_copyable_pnl"]
                / follower_scores["total_copyable_notional"].clip(lower=1e-9)
            )
            mask = follower_scores["total_copyable_pnl"] >= follower_min_copyable_pnl
            if follower_min_copyable_roi is not None:
                mask = mask & (follower_scores["copyable_roi"] >= follower_min_copyable_roi)

            # Market HHI: concentration of copyable PnL across markets
            if follower_max_market_hhi is not None:
                market_pnl = (
                    combined.groupby(["follower_wallet", "condition_id"], sort=False)["copyable_pnl"]
                    .sum()
                )
                total_pnl = combined.groupby("follower_wallet", sort=False)["copyable_pnl"].sum()
                market_share = market_pnl / total_pnl
                hhi = market_share.groupby("follower_wallet", sort=False).apply(
                    lambda s: float((s ** 2).sum()), include_groups=False
                ).rename("market_hhi").reset_index()
                follower_scores = follower_scores.merge(hhi, on="follower_wallet", how="left")
                follower_scores["market_hhi"] = follower_scores["market_hhi"].fillna(0.0)
                mask = mask & (follower_scores["market_hhi"] <= follower_max_market_hhi)

            # Max copyable drawdown / total copyable PnL
            if follower_max_copyable_dd_ratio is not None:
                sorted_imp = combined.sort_values("follower_dt")
                cum_pnl = sorted_imp.groupby("follower_wallet", sort=False)["copyable_pnl"].cumsum()
                running_max = sorted_imp.groupby("follower_wallet", sort=False)["copyable_pnl"].cumsum().groupby(
                    sorted_imp["follower_wallet"], sort=False
                ).cummax()
                dd = running_max - cum_pnl
                max_dd = dd.groupby(sorted_imp["follower_wallet"], sort=False).max().rename("max_copyable_dd").reset_index()
                total_cpnl = combined.groupby("follower_wallet", sort=False)["copyable_pnl"].sum().rename("total_cpnl").reset_index()
                dd_df = max_dd.merge(total_cpnl, on="follower_wallet")
                dd_df["copyable_dd_ratio"] = dd_df["max_copyable_dd"] / dd_df["total_cpnl"].clip(lower=1e-9)
                follower_scores = follower_scores.merge(
                    dd_df[["follower_wallet", "copyable_dd_ratio"]], on="follower_wallet", how="left"
                )
                follower_scores["copyable_dd_ratio"] = follower_scores["copyable_dd_ratio"].fillna(0.0)
                mask = mask & (follower_scores["copyable_dd_ratio"] <= follower_max_copyable_dd_ratio)

            good_followers = set(follower_scores.loc[mask, "follower_wallet"])
            cur_followers = cur_followers & good_followers

        _snapshot()

    return cur_followers, cur_buy_leaders, cur_sell_leaders, log, snapshots


# ---------------------------------------------------------------------------
# 3c. Parallel grid search for implied trades
# ---------------------------------------------------------------------------

_IWV: pd.DataFrame | None = None
_IDF: pd.DataFrame | None = None  # training trades for iterative refinement
_IDV: pd.DataFrame | None = None  # validation trades for cutoff filter + scoring


def _init_worker(wallet_vol: pd.DataFrame, df_train: pd.DataFrame, df_val: pd.DataFrame) -> None:
    global _IWV, _IDF, _IDV
    _IWV = wallet_vol
    _IDF = df_train
    _IDV = df_val


def _implied_grid_eval_one(params: dict) -> dict:
    """Evaluate one param combo for implied-trade grid search. Uses module-level state."""
    t0 = time.time()
    try:
        follower_ws = set(
            select_follower_wallets(
                _IWV,
                min_copyable_roi=params["min_follower_copyable_roi"],
                min_trade_value=params["min_follower_trade_value"],
                min_num_buckets=params["min_follower_num_buckets"],
                max_market_pnl_hhi=params.get("max_follower_hhi", 0.3),
            )["wallet"]
        )
        buy_leader_ws = set(
            select_leader_wallets(
                _IWV,
                min_trade_count=params["min_buy_leader_trade_count"],
                min_roi=params["min_buy_leader_roi"],
                max_market_pnl_hhi=params["max_buy_leader_hhi"],
                side="BUY",
            )["wallet"]
        )
        sell_leader_ws = set(
            select_leader_wallets(
                _IWV,
                min_trade_count=params["min_sell_leader_trade_count"],
                min_roi=params["min_sell_leader_roi"],
                max_market_pnl_hhi=params["max_sell_leader_hhi"],
                side="SELL",
            )["wallet"]
        )

        tw = params["time_window_minutes"]

        # Iterative refinement on training data
        n_iter = int(params.get("n_iterations", 0))
        if n_iter > 0:
            follower_ws, buy_leader_ws, sell_leader_ws, _, _ = (
                iterative_leader_follower_filter(
                    _IDF,
                    follower_ws, buy_leader_ws, sell_leader_ws,
                    time_window_minutes=tw,
                    n_iterations=n_iter,
                    leader_min_copyable_pnl=params.get("leader_min_copyable_pnl"),
                    follower_min_copyable_pnl=params.get("follower_min_copyable_pnl", 20.0),
                    follower_min_copyable_roi=params.get("follower_min_copyable_roi"),
                    follower_max_market_hhi=params.get("follower_max_market_hhi"),
                    follower_max_copyable_dd_ratio=params.get("follower_max_copyable_dd_ratio"),
                )
            )

        # Final follower filter on validation data
        follower_roi_cutoff = params.get("follower_min_copyable_roi_cutoff")
        follower_pnl_cutoff = params.get("follower_min_copyable_pnl_cutoff")
        follower_hhi_cutoff = params.get("follower_max_market_hhi_cutoff")
        follower_dd_cutoff = params.get("follower_max_copyable_dd_ratio_cutoff")

        buy_imp = detect_implied_buys(
            _IDV, follower_ws, buy_leader_ws,
            time_window_minutes=tw, leader_side="BUY",
        )
        sell_imp = detect_implied_buys(
            _IDV, follower_ws, sell_leader_ws,
            time_window_minutes=tw, leader_side="SELL",
        )
        combined = pd.concat([buy_imp, sell_imp], ignore_index=True)
        if not combined.empty:
            f_scores = (
                combined.groupby("follower_wallet", sort=False)
                .agg(total_copyable_pnl=("copyable_pnl", "sum"),
                     total_copyable_notional=("copyable_notional", "sum"))
                .reset_index()
            )
            f_scores["copyable_roi"] = (
                f_scores["total_copyable_pnl"]
                / f_scores["total_copyable_notional"].clip(lower=1e-9)
            )
            mask = pd.Series(True, index=f_scores.index)

            if follower_pnl_cutoff is not None:
                mask = mask & (f_scores["total_copyable_pnl"] >= follower_pnl_cutoff)
            if follower_roi_cutoff is not None:
                mask = mask & (f_scores["copyable_roi"] >= follower_roi_cutoff)

            # Market HHI
            if follower_hhi_cutoff is not None:
                market_pnl = combined.groupby(["follower_wallet", "condition_id"], sort=False)["copyable_pnl"].sum()
                total_pnl = combined.groupby("follower_wallet", sort=False)["copyable_pnl"].sum()
                market_share = market_pnl / total_pnl
                hhi = market_share.groupby("follower_wallet", sort=False).apply(
                    lambda s: float((s ** 2).sum()), include_groups=False
                ).rename("market_hhi").reset_index()
                f_scores = f_scores.merge(hhi, on="follower_wallet", how="left")
                f_scores["market_hhi"] = f_scores["market_hhi"].fillna(0.0)
                mask = mask & (f_scores["market_hhi"] <= follower_hhi_cutoff)

            # Max copyable drawdown / total copyable PnL
            if follower_dd_cutoff is not None:
                sorted_imp = combined.sort_values("follower_dt")
                cum_pnl = sorted_imp.groupby("follower_wallet", sort=False)["copyable_pnl"].cumsum()
                running_max = cum_pnl.groupby(sorted_imp["follower_wallet"], sort=False).cummax()
                dd = running_max - cum_pnl
                max_dd = dd.groupby(sorted_imp["follower_wallet"], sort=False).max().rename("max_copyable_dd").reset_index()
                total_cpnl = combined.groupby("follower_wallet", sort=False)["copyable_pnl"].sum().rename("total_cpnl").reset_index()
                dd_df = max_dd.merge(total_cpnl, on="follower_wallet")
                dd_df["copyable_dd_ratio"] = dd_df["max_copyable_dd"] / dd_df["total_cpnl"].clip(lower=1e-9)
                f_scores = f_scores.merge(dd_df[["follower_wallet", "copyable_dd_ratio"]], on="follower_wallet", how="left")
                f_scores["copyable_dd_ratio"] = f_scores["copyable_dd_ratio"].fillna(0.0)
                mask = mask & (f_scores["copyable_dd_ratio"] <= follower_dd_cutoff)

            follower_ws = follower_ws & set(f_scores.loc[mask, "follower_wallet"])

        buy_ev = evaluate_implied_pnl(
            _IDV, follower_ws, buy_leader_ws,
            time_window_minutes=tw, leader_side="BUY",
        )
        sell_ev = evaluate_implied_pnl(
            _IDV, follower_ws, sell_leader_ws,
            time_window_minutes=tw, leader_side="SELL",
        )

        # Filter by min_pair_interactions
        min_pi = params["min_pair_interactions"]
        if buy_ev["trade_count"] < min_pi:
            buy_ev["followed_copyable_pnl"] = 0.0
        if sell_ev["trade_count"] < min_pi:
            sell_ev["followed_copyable_pnl"] = 0.0

        total_pnl = buy_ev["followed_copyable_pnl"] + sell_ev["followed_copyable_pnl"]
        elapsed = time.time() - t0

        return {
            **params,
            "implied_copyable_pnl": total_pnl,
            "buy_pnl": buy_ev["followed_copyable_pnl"],
            "sell_pnl": sell_ev["followed_copyable_pnl"],
            "buy_trades": buy_ev["trade_count"],
            "sell_trades": sell_ev["trade_count"],
            "buy_leaders": buy_ev["leader_count"],
            "sell_leaders": sell_ev["leader_count"],
            "followers": len(follower_ws),
            "elapsed": elapsed,
        }
    except Exception as e:
        return {**params, "error": str(e), "implied_copyable_pnl": -float("inf")}


def run_implied_grid_search(
    param_grid: dict[str, list],
    wallet_vol: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    n_workers: int | None = 8,
) -> pd.DataFrame:
    """Run parallel grid search over implied-trade selection parameters.

    *df_train* is used for iterative wallet refinement (if ``n_iterations > 0``).
    *df_val* is used for both the follower ROI cutoff filter and scoring configs.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    print(f"Grid: {len(combos)} combos, {n_workers} workers")

    t_start = time.time()
    results_log = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(wallet_vol, df_train, df_val),
    ) as pool:
        futures = {
            pool.submit(_implied_grid_eval_one, dict(zip(keys, v))): i
            for i, v in enumerate(combos)
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            results_log.append(fut.result())
            if done % 50 == 0 or done == len(combos):
                print(f"  [{done}/{len(combos)}] {time.time() - t_start:.1f}s elapsed")

    elapsed = time.time() - t_start
    print(f"Done: {len(results_log)} configs in {elapsed:.1f}s")
    return pd.DataFrame(results_log).sort_values("implied_copyable_pnl", ascending=False)


# ---------------------------------------------------------------------------
# 3. Wallet volatility metrics & copyable selection
# ---------------------------------------------------------------------------


def compute_wallet_volatility(df_full: pd.DataFrame) -> pd.DataFrame:
    """Compute per-wallet volatility metrics on training trades and derive copyability."""
    from polymarket_analysis.wallet_selection.volatility import compute_wallet_metrics

    df_slice = df_full[df_full.get("is_train", pd.Series(True, index=df_full.index))].copy()
    print(f"Training trades: {len(df_slice):,}")

    wallet_vol, _ = compute_wallet_metrics(df_slice)
    print(f"Wallets with metrics: {len(wallet_vol)}")

    wallet_vol["copyable_pnl_factor"] = np.clip(
        wallet_vol["copyable_pnl"] / wallet_vol["total_pnl"].replace(0, np.nan),
        0, 1.0,
    ).fillna(0.0)
    wallet_vol["copyable_roi"] = (
        wallet_vol["average_roi"] * wallet_vol["copyable_pnl_factor"]
    )
    return wallet_vol


def select_copyable_wallets(wallet_vol: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Score wallets and split into copyable + predicting cohorts."""
    candidates = wallet_vol.copy()

    # Ensure required columns exist
    for col, default in {
        "top10_pnl_pct": np.nan,
        "top_market_abs_pnl_pct": np.nan,
        "market_pnl_hhi": np.nan,
        "positive_bucket_share": np.nan,
    }.items():
        if col not in candidates.columns:
            candidates[col] = default

    for col in [
        "average_roi", "median_roi", "num_buckets", "num_markets",
        "pnl_volatility", "max_drawdown_to_pnl",
        "top_market_pnl_pct", "top_market_abs_pnl_pct", "top5_pnl_pct",
        "top10_pnl_pct", "market_pnl_hhi",
        "copyable_roi", "copyable_pnl_factor", "copyable_pnl", "total_notional",
        "max_copyable_drawdown_to_copyable_pnl", "worst5_pnl_pct",
        "positive_bucket_share",
    ]:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")

    # Base eligibility filters
    base_mask = (
        (candidates["buy_roi"] >= 0.05)
        & (candidates["num_buckets"] >= 20)
        & (candidates["num_markets"] >= 15)
        & (candidates["max_drawdown_to_pnl"] <= 0.2)
        & (candidates["top_market_pnl_pct"] < 0.25)
        & (candidates["market_pnl_hhi"].fillna(0.20) < 0.30)
        & (candidates["total_notional"] >= 5_000)
    )

    eligible_base = candidates[base_mask].copy()
    if eligible_base.empty:
        raise ValueError("No wallets passed base eligibility filters.")

    # Predictability score
    eligible_base["sample_mult"] = np.clip(
        np.log1p(eligible_base["num_buckets"]) / np.log(2000.0), 0.20, 1.00
    )
    eligible_base["downside_tail"] = eligible_base["worst5_pnl_pct"].abs().fillna(0.0)
    eligible_base["predictability_score"] = eligible_base["sample_mult"] * (
        + 1.2 * (eligible_base["positive_bucket_share"].fillna(0.5) - 0.5)
        - 1.0 * eligible_base["pnl_volatility"].fillna(0.0)
        - 1.2 * eligible_base["max_drawdown_to_pnl"].fillna(0.0)
        - 0.8 * eligible_base["top_market_abs_pnl_pct"].fillna(
            eligible_base["top_market_pnl_pct"]
        ).fillna(0.0)
        - 0.6 * eligible_base["market_pnl_hhi"].fillna(0.0)
        - 0.5 * eligible_base["downside_tail"]
    )

    # Copyable score
    copyable_mask = (
        (eligible_base["copyable_pnl"] > 0)
        & (eligible_base["average_roi"] >= 0.04)
        & (eligible_base["copyable_roi"] >= 0.05)
    )
    copyable_candidates = eligible_base[copyable_mask].copy()
    copyable_candidates["copyable_efficiency"] = (
        copyable_candidates["copyable_pnl"].fillna(0.0)
        / (copyable_candidates["total_notional"].fillna(0.0) + 1.0)
    )
    copyable_candidates["copyable_dd_ratio"] = (
        copyable_candidates["max_copyable_drawdown_to_copyable_pnl"]
        .fillna(copyable_candidates["max_drawdown_to_pnl"])
        .fillna(0.0)
    )
    copyable_candidates["copyable_score"] = (
        1.8 * copyable_candidates["copyable_roi"].fillna(0.0)
        + 1.2 * copyable_candidates["copyable_pnl_factor"].fillna(0.0)
        + 25.0 * copyable_candidates["copyable_efficiency"].clip(lower=-1.0, upper=1.0)
        - 0.8 * copyable_candidates["copyable_dd_ratio"].clip(lower=0.0)
        - 0.5 * copyable_candidates["top_market_abs_pnl_pct"].fillna(
            copyable_candidates["top_market_pnl_pct"]
        ).fillna(0.0)
    )
    copyable_candidates["final_score"] = (
        0.60 * copyable_candidates["predictability_score"]
        + 0.40 * copyable_candidates["copyable_score"]
    )

    # Split into groups
    wallet_cohorts = {}
    wallet_cohorts["copyable_group"] = (
        copyable_candidates.sort_values("final_score", ascending=False).reset_index(drop=True)
    )

    predicting_pool = eligible_base[
        ~eligible_base["wallet"].isin(wallet_cohorts["copyable_group"]["wallet"])
    ].copy()
    wallet_cohorts["predicting_group"] = (
        predicting_pool.sort_values("predictability_score", ascending=False).reset_index(drop=True)
    )

    wallet_cohorts["copyable_group"]["wallet_quality"] = wallet_cohorts["copyable_group"]["final_score"]
    wallet_cohorts["predicting_group"]["wallet_quality"] = wallet_cohorts["predicting_group"]["predictability_score"]

    print(f"Base-eligible wallets: {len(eligible_base):,}")
    print(f"Copyable candidates: {len(copyable_candidates):,}")
    print(f"Selected copyable group: {len(wallet_cohorts['copyable_group']):,}")
    print(f"Selected predicting group: {len(wallet_cohorts['predicting_group']):,}")
    return wallet_cohorts


# ---------------------------------------------------------------------------
# 4. Copy-trading simulation
# ---------------------------------------------------------------------------


def simulate_copy_pnl(
    df_val: pd.DataFrame,
    groups: dict[str, pd.DataFrame],
    time_window_minutes: int = 10,
    measure_groups: list[str] | None = None,
) -> dict[str, dict]:
    """Simulate copy-trading PnL for the given wallet groups on a data split.

    Args:
        df_val: Trade DataFrame for the evaluation split.
        groups: dict mapping group name -> wallet DataFrame (from build_wallet_groups).
        time_window_minutes: Max gap for leader->follower detection.
        measure_groups: Which groups to measure PnL for.  ``None`` means all.
            Use e.g. ``["openers"]`` in stage 1 to measure only openers.

    Returns:
        dict mapping group name -> {total_copyable_pnl, total_notional, trade_count, wallet_count}.
    """
    if measure_groups is None:
        measure_groups = list(groups.keys())

    notional_col = "usdc_amount" if "usdc_amount" in df_val.columns else "trade_value_usdc"
    results = {}

    # ── Openers ──
    if "openers" in measure_groups:
        opener_wallets = set(groups["openers"]["wallet"]) if not groups["openers"].empty else set()
        if opener_wallets:
            ob = df_val[(df_val["wallet"].isin(opener_wallets)) & (df_val["side"] == "BUY")].copy()
            # Opening buys only: position after trade == trade quantity (no prior position)
            ob = ob[ob["position"] <= ob["quantity"] + 1e-9].copy()
            results["openers"] = {
                "total_copyable_pnl": ob["copyable_pnl"].sum(),
                "total_notional": ob[notional_col].sum(),
                "trade_count": len(ob),
                "wallet_count": ob["wallet"].nunique(),
            }
        else:
            results["openers"] = {"total_copyable_pnl": 0, "total_notional": 0, "trade_count": 0, "wallet_count": 0}

    # ── Leaders (not copied directly) ──
    if "leaders" in measure_groups:
        results["leaders"] = {
            "total_copyable_pnl": 0, "total_notional": 0,
            "trade_count": 0, "wallet_count": len(groups["leaders"]),
        }

    # ── Followers: only trades that follow a leader ──
    if "followers" in measure_groups:
        leader_wallets = set(groups["leaders"]["wallet"]) if not groups["leaders"].empty else set()
        follower_wallets = set(groups["followers"]["wallet"]) if not groups["followers"].empty else set()

        following = pd.DataFrame(columns=["condition_id", "outcome", "dt", "quantity", "copyable_pnl"])

        if leader_wallets and follower_wallets:
            leader_buys = df_val[
                (df_val["wallet"].isin(leader_wallets)) & (df_val["side"] == "BUY")
            ][["condition_id", "outcome", "dt", "wallet"]].copy()

            follower_buys = df_val[
                (df_val["wallet"].isin(follower_wallets)) & (df_val["side"] == "BUY")
            ][["condition_id", "outcome", "dt", "wallet", "copyable_pnl", "quantity", notional_col]].copy()

            if not leader_buys.empty and not follower_buys.empty:
                merged = pd.merge_asof(
                    follower_buys.sort_values("dt"),
                    leader_buys.sort_values("dt"),
                    on="dt",
                    by=["condition_id", "outcome"],
                    direction="backward",
                    tolerance=pd.Timedelta(minutes=time_window_minutes),
                    suffixes=("", "_leader"),
                )
                following = merged.loc[
                    merged["wallet_leader"].notna(),
                    ["condition_id", "outcome", "dt", "quantity", "copyable_pnl", notional_col],
                ].copy()

        results["followers"] = {
            "total_copyable_pnl": following["copyable_pnl"].sum() if not following.empty else 0,
            "total_notional": following[notional_col].sum() if not following.empty else 0,
            "trade_count": len(following),
            "wallet_count": following["wallet"].nunique() if (not following.empty and "wallet" in following.columns) else 0,
        }

    # ── Closers: position-based PnL ──
    if "closers" in measure_groups:
        closer_wallets = set(groups["closers"]["wallet"]) if not groups["closers"].empty else set()
        closer_effective_pnl = 0.0
        closer_effective_notional = 0.0
        closer_trade_count = 0
        closer_wallet_count = 0

        if closer_wallets:
            closer_sells = df_val[
                (df_val["wallet"].isin(closer_wallets)) & (df_val["side"] == "SELL")
            ][["condition_id", "outcome", "dt", "quantity", "copyable_pnl", "wallet", notional_col]].copy()

            if not closer_sells.empty:
                buy_cols = ["condition_id", "outcome", "dt", "quantity"]
                ob = df_val[
                    (df_val["wallet"].isin(set(groups["openers"]["wallet"]))) & (df_val["side"] == "BUY")
                ][buy_cols] if not groups["openers"].empty else pd.DataFrame(columns=buy_cols)

                leader_wallets_all = set(groups["leaders"]["wallet"]) if not groups["leaders"].empty else set()
                follower_wallets_all = set(groups["followers"]["wallet"]) if not groups["followers"].empty else set()
                following_trades = pd.DataFrame(columns=buy_cols)
                if leader_wallets_all and follower_wallets_all:
                    lb = df_val[
                        (df_val["wallet"].isin(leader_wallets_all)) & (df_val["side"] == "BUY")
                    ][["condition_id", "outcome", "dt", "wallet"]].copy()
                    fb = df_val[
                        (df_val["wallet"].isin(follower_wallets_all)) & (df_val["side"] == "BUY")
                    ][["condition_id", "outcome", "dt", "wallet"]].copy()
                    if not lb.empty and not fb.empty:
                        m = pd.merge_asof(
                            fb.sort_values("dt"), lb.sort_values("dt"),
                            on="dt", by=["condition_id", "outcome"],
                            direction="backward",
                            tolerance=pd.Timedelta(minutes=time_window_minutes),
                            suffixes=("", "_leader"),
                        )
                        if "wallet_leader" in m.columns:
                            fw = m.loc[m["wallet_leader"].notna(), "wallet"]
                            following_trades = df_val[
                                (df_val["wallet"].isin(set(fw))) & (df_val["side"] == "BUY")
                            ][buy_cols]

                buys = pd.concat([ob, following_trades], ignore_index=True)
                sells = closer_sells[["condition_id", "outcome", "dt", "quantity", "copyable_pnl", notional_col]].copy()
                sells["_is_buy"] = False
                buys["_is_buy"] = True

                all_trades = pd.concat([buys, sells], ignore_index=True)
                all_trades = all_trades.sort_values(["condition_id", "outcome", "dt"]).reset_index(drop=True)

                all_trades["_signed_qty"] = np.where(all_trades["_is_buy"], all_trades["quantity"], -all_trades["quantity"])
                all_trades["_cum_pos"] = all_trades.groupby(["condition_id", "outcome"])["_signed_qty"].cumsum()
                all_trades["_pos_before"] = all_trades["_cum_pos"] - all_trades["_signed_qty"]

                sell_mask = ~all_trades["_is_buy"]
                all_trades.loc[sell_mask, "_effective_qty"] = np.minimum(
                    all_trades.loc[sell_mask, "quantity"],
                    np.maximum(0, all_trades.loc[sell_mask, "_pos_before"]),
                )
                all_trades.loc[sell_mask, "_effective_pnl"] = (
                    all_trades.loc[sell_mask, "_effective_qty"]
                    / all_trades.loc[sell_mask, "quantity"].clip(lower=1e-9)
                    * all_trades.loc[sell_mask, "copyable_pnl"]
                )
                all_trades.loc[sell_mask, "_effective_notional"] = (
                    all_trades.loc[sell_mask, "_effective_qty"]
                    / all_trades.loc[sell_mask, "quantity"].clip(lower=1e-9)
                    * all_trades.loc[sell_mask, notional_col]
                )

                closer_effective_pnl = all_trades.loc[sell_mask, "_effective_pnl"].sum()
                closer_effective_notional = all_trades.loc[sell_mask, "_effective_notional"].sum()
                closer_trade_count = int(sell_mask.sum())
                closer_wallet_count = closer_sells["wallet"].nunique()

        results["closers"] = {
            "total_copyable_pnl": closer_effective_pnl,
            "total_notional": closer_effective_notional,
            "trade_count": closer_trade_count,
            "wallet_count": closer_wallet_count,
        }

    return results


def group_wallet_summary(
    df: pd.DataFrame,
    groups: dict[str, pd.DataFrame],
    group_name: str,
    opening_buys_only: bool = False,
) -> dict[str, dict]:
    """Compute wallet-level stats for a group on a data split.

    Args:
        opening_buys_only: When True, filter to BUY trades where
            ``position <= quantity + 1e-9`` (opening buys only), making
            the numbers comparable to the copyable PnL from ``simulate_copy_pnl``.

    Returns a dict with:
      - per-group totals: wallet_pnl, wallet_notional, wallet_roi, active_wallets, total_trades
      - per-wallet DataFrame (key ``"wallets"``)
    """
    notional_col = "usdc_amount" if "usdc_amount" in df.columns else "trade_value_usdc"
    wallet_set = set(groups[group_name]["wallet"]) if not groups[group_name].empty else set()
    if not wallet_set:
        return {"wallet_pnl": 0, "wallet_notional": 0, "wallet_roi": 0,
                "active_wallets": 0, "total_trades": 0, "wallets": pd.DataFrame()}

    trades = df[df["wallet"].isin(wallet_set)].copy()
    if opening_buys_only:
        trades = trades[(trades["side"] == "BUY") & (trades["position"] <= trades["quantity"] + 1e-9)].copy()
    if trades.empty:
        return {"wallet_pnl": 0, "wallet_notional": 0, "wallet_roi": 0,
                "active_wallets": 0, "total_trades": 0, "wallets": pd.DataFrame()}

    wallet_agg = trades.groupby("wallet").agg(
        wallet_pnl=("pnl", "sum"),
        wallet_notional=(notional_col, "sum"),
        total_trades=("wallet", "size"),
    ).reset_index()
    wallet_agg["wallet_roi"] = wallet_agg["wallet_pnl"] / wallet_agg["wallet_notional"].clip(lower=1e-9)

    return {
        "wallet_pnl": wallet_agg["wallet_pnl"].sum(),
        "wallet_notional": wallet_agg["wallet_notional"].sum(),
        "wallet_roi": wallet_agg["wallet_pnl"].sum() / max(wallet_agg["wallet_notional"].sum(), 1e-9),
        "active_wallets": len(wallet_agg),
        "total_trades": int(wallet_agg["total_trades"].sum()),
        "wallets": wallet_agg.sort_values("wallet_pnl", ascending=False),
    }


def print_group_summary(label: str, copyable: dict, wallet: dict, wallet_all: dict | None = None) -> None:
    """Print copyable PnL alongside wallet PnL for a group."""
    print(f"{'=' * 70}")
    print(label)
    print(f"{'=' * 70}")
    cr = copyable.get("openers", copyable.get("followers", copyable.get("closers", {})))
    c_pnl = cr.get("total_copyable_pnl", 0)
    c_not = cr.get("total_notional", 0)
    c_roi = c_pnl / c_not * 100 if c_not > 0 else 0

    w_pnl = wallet["wallet_pnl"]
    w_not = wallet["wallet_notional"]
    w_roi = wallet["wallet_roi"] * 100
    print(f"  {'copyable':>12}: pnl={c_pnl:>10.2f}  notional={c_not:>10.2f}  roi={c_roi:>6.2f}%  "
          f"open_buys={cr.get('trade_count', 0):>6}  wallets={cr.get('wallet_count', 0):>4}")
    print(f"  {'wallet_open':>12}: pnl={w_pnl:>10.2f}  notional={w_not:>10.2f}  roi={w_roi:>6.2f}%  "
          f"open_buys={wallet['total_trades']:>6}  wallets={wallet['active_wallets']:>4}")
    if wallet_all is not None and wallet_all["total_trades"] > 0:
        a_pnl = wallet_all["wallet_pnl"]
        a_not = wallet_all["wallet_notional"]
        a_roi = wallet_all["wallet_roi"] * 100
        print(f"  {'wallet_all':>12}: pnl={a_pnl:>10.2f}  notional={a_not:>10.2f}  roi={a_roi:>6.2f}%  "
              f"trades={wallet_all['total_trades']:>6}  wallets={wallet_all['active_wallets']:>4}")


# ---------------------------------------------------------------------------
# 5. Wallet-group evaluation (matching reference format)
# ---------------------------------------------------------------------------


def evaluate_wallet_group(
    df: pd.DataFrame,
    wallet_set: set[str],
    label: str = "",
) -> None:
    """Print open PnL vs total PnL for a wallet set.

    Matches the reference cell ``# Copyable group total and open PnL``.
    Requires ``copyable_notional`` column on *df* (use ``compute_copyable_notional``).
    """
    df_ana = df
    df_ana = df[(df["wallet"].isin(wallet_set))
                & (df['side'] == 'BUY')].copy()

    opening = df_ana[df_ana["position"] == df_ana["quantity"]][
        ["trade_pnl", "copyable_pnl", "notional", "copyable_notional"]
    ].sum()

    total = df_ana[["trade_pnl", "copyable_pnl", "notional", "copyable_notional"]].sum()

    if label:
        print(f"\n*** {label} ***")

    o_wallet_roi = opening["trade_pnl"] / opening["notional"] if opening["notional"] else 0
    o_copy_roi = opening["copyable_pnl"] / opening["copyable_notional"] if opening["copyable_notional"] else 0
    print(f"  Open  : wallet_pnl={opening['trade_pnl']:>10.2f}  roi={o_wallet_roi:.4f}  |  "
          f"copyable_pnl={opening['copyable_pnl']:>10.2f}  roi={o_copy_roi:.4f}")

    t_wallet_roi = total["trade_pnl"] / total["notional"] if total["notional"] else 0
    t_copy_roi = total["copyable_pnl"] / total["copyable_notional"] if total["copyable_notional"] else 0
    print(f"  Total : wallet_pnl={total['trade_pnl']:>10.2f}  roi={t_wallet_roi:.4f}  |  "
          f"copyable_pnl={total['copyable_pnl']:>10.2f}  roi={t_copy_roi:.4f}")


def evaluate_wallet_group_openers(
    df: pd.DataFrame,
    wallet_set: set[str],
    label: str = "",
) -> dict:
    """Evaluate wallet set: open PnL (opening buys) + total PnL.

    Returns dict with opening and total sums for programmatic use (grid search).
    Requires ``copyable_notional`` column on *df*.
    """
    df_ana = df[df["wallet"].isin(wallet_set)].copy()

    opening = df_ana[
        (df_ana["position"] == df_ana["quantity"])
        & (df_ana["side"] == "BUY")
        ][
        ["trade_pnl", "copyable_pnl", "notional", "copyable_notional"]
    ].sum()
    total = df_ana[["trade_pnl", "copyable_pnl", "notional", "copyable_notional"]].sum()

    return {
        "open_wallet_pnl": float(opening["trade_pnl"]),
        "open_wallet_roi": float(opening["trade_pnl"] / opening["notional"]) if opening["notional"] else 0,
        "open_copyable_pnl": float(opening["copyable_pnl"]),
        "open_copyable_roi": float(opening["copyable_pnl"] / opening["copyable_notional"]) if opening["copyable_notional"] else 0,
        "open_notional": float(opening["notional"]),
        "open_wallets": int(df_ana[df_ana["position"] == df_ana["quantity"]]["wallet"].nunique()),
        "total_wallet_pnl": float(total["trade_pnl"]),
        "total_wallet_roi": float(total["trade_pnl"] / total["notional"]) if total["notional"] else 0,
        "total_copyable_pnl": float(total["copyable_pnl"]),
        "total_copyable_roi": float(total["copyable_pnl"] / total["copyable_notional"]) if total["copyable_notional"] else 0,
        "total_notional": float(total["notional"]),
        "total_wallets": int(df_ana["wallet"].nunique()),
    }


# ---------------------------------------------------------------------------
# 6. Printing helpers
# ---------------------------------------------------------------------------


def print_results(label: str, results: dict[str, dict]) -> None:
    print(f"{'=' * 60}")
    print(label)
    print(f"{'=' * 60}")
    total_pnl = 0
    total_not = 0
    for name, r in results.items():
        pnl = r["total_copyable_pnl"]
        notional = r["total_notional"]
        roi = pnl / notional * 100 if notional > 0 else 0
        total_pnl += pnl
        total_not += notional
        print(
            f"  {name:>12}: copyable_pnl={pnl:>10.2f}  notional={notional:>10.2f}  "
            f"copyable_roi={roi:>6.2f}%  trades={r['trade_count']:>6}  wallets={r['wallet_count']:>4}"
        )
    total_roi = total_pnl / total_not * 100 if total_not > 0 else 0
    print(f"  {'TOTAL':>12}: copyable_pnl={total_pnl:>10.2f}  notional={total_not:>10.2f}  copyable_roi={total_roi:>6.2f}%")


def print_wallet_stats(groups: dict[str, pd.DataFrame]) -> None:
    print("  Wallet stats (from train data):")
    for name in ["openers", "closers"]:
        gdf = groups[name]
        if not gdf.empty:
            copyable_roi = gdf["copyable_roi"].mean() if "copyable_roi" in gdf.columns else 0
            trade_roi = gdf["trade_roi"].mean() if "trade_roi" in gdf.columns else 0
            print(f"    {name:>10}: copyable_roi={copyable_roi * 100:.2f}%  trade_roi={trade_roi * 100:.2f}%  wallets={len(gdf)}")


# ---------------------------------------------------------------------------
# 7. Result persistence (stage handoff via JSON)
# ---------------------------------------------------------------------------


def save_stage_result(stage: int, params: dict, extra: dict | None = None) -> Path:
    """Save best params from a stage to JSON."""
    payload = {"stage": stage, "best_params": params}
    if extra:
        payload.update(extra)
    out = RESULTS_DIR / f"stage{stage}_result.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved stage {stage} result -> {out}")
    return out


def load_stage_result(stage: int) -> dict:
    """Load best params from a previous stage."""
    path = RESULTS_DIR / f"stage{stage}_result.json"
    with open(path) as f:
        data = json.load(f)
    return data["best_params"]


# ---------------------------------------------------------------------------
# 8. Parallel grid search helpers
# ---------------------------------------------------------------------------

# Module-level state set by run_grid_search before spawning workers
_WV: pd.DataFrame | None = None
_DV: pd.DataFrame | None = None


def grid_eval_one(params: dict) -> dict:
    """Evaluate one param combo. Uses module-level _WV, _DV."""
    t0 = time.time()
    try:
        group = select_copyable_group(_WV, **params)
        ws = set(group["wallet"])
        ev = evaluate_wallet_group_openers(_DV, ws)
        elapsed = time.time() - t0
        return {
            **params,
            "open_copyable_pnl": ev["open_copyable_pnl"],
            "open_wallet_pnl": ev["open_wallet_pnl"],
            "open_wallets": ev["open_wallets"],
            "total_copyable_pnl": ev["total_copyable_pnl"],
            "wallets": len(group),
            "elapsed": elapsed,
        }
    except Exception as e:
        return {**params, "error": str(e), "open_copyable_pnl": -float("inf")}


def run_grid_search(
    param_grid: dict[str, list],
    wallet_vol: pd.DataFrame,
    df_val: pd.DataFrame,
    n_workers: int | None = 8,
) -> pd.DataFrame:
    """Run parallel grid search over selection parameters.

    Args:
        param_grid: Dict mapping param name -> list of values to try.
        wallet_vol: Wallet metrics DataFrame (from compute_wallet_metrics + opening metrics).
        df_val: Validation trades DataFrame.
        n_workers: Number of threads (default 8).

    Returns a DataFrame of all results sorted by ``open_copyable_pnl``.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    global _WV, _DV
    _WV = wallet_vol
    _DV = df_val

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    print(f"Grid: {len(combos)} combos, {n_workers} workers")

    t_start = time.time()
    results_log = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(grid_eval_one, dict(zip(keys, v))): i
            for i, v in enumerate(combos)
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            results_log.append(fut.result())
            if done % 100 == 0 or done == len(combos):
                print(f"  [{done}/{len(combos)}] {time.time() - t_start:.1f}s elapsed")

    elapsed = time.time() - t_start
    print(f"Done: {len(results_log)} configs in {elapsed:.1f}s")
    return pd.DataFrame(results_log).sort_values("open_copyable_pnl", ascending=False)
