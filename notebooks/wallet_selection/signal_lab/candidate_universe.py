"""Rebuild the stage1 strategy-selection candidate universe from raw fills."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from polymarket_analysis.wallet_selection.exposure import wallet_capture_thresholds
from polymarket_analysis.wallet_selection.volatility import compute_wallet_metrics

from .filters import select_strategy_eligible, select_strategy_selection

_COLUMNS = [
    "wallet",
    "condition_id",
    "dt",
    "side",
    "position",
    "total_quantity",
    "avg_price",
    "trade_value_usdc",
    "final_value_usdc",
    "copyable_pnl",
    "last_condition_trade_ts",
    "copyable_qty_5m_100",
    "avail_copy_total_vol_5m_100",
]

_TRAIN_END = pd.Timestamp("2026-09-01", tz="UTC")
_EXPOSURE_FILTER_FRAC = 0.2
_EXPOSURE_PNL_COL = "copyable_pnl"


def _prepare_stream(trades_dir: Path, max_lead_days: int) -> pd.DataFrame:
    files = sorted(Path(trades_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {trades_dir}")

    df = pd.concat(
        [pd.read_parquet(f, columns=_COLUMNS) for f in files],
        ignore_index=True,
    )

    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    df["last_condition_trade_ts"] = pd.to_datetime(
        df["last_condition_trade_ts"], utc=True, errors="coerce"
    )

    lead = df["last_condition_trade_ts"] - df["dt"]
    df = df[lead <= pd.Timedelta(days=max_lead_days)].copy()

    df = df.rename(
        columns={
            "total_quantity": "quantity",
            "avg_price": "price",
            "trade_value_usdc": "usdc_amount",
        }
    )

    df["usdc_amount"] = df["usdc_amount"].astype(float)
    df["final_value_usdc"] = df["final_value_usdc"].astype(float)
    df["quantity"] = df["quantity"].astype(float)

    df["pnl"] = np.where(
        df["side"] == "BUY",
        df["final_value_usdc"] - df["usdc_amount"],
        df["usdc_amount"] - df["final_value_usdc"],
    )
    df["notional"] = np.where(
        df["side"] == "BUY",
        df["usdc_amount"],
        df["quantity"] * (1 - df["price"].astype(float)),
    )
    df["is_train"] = df["last_condition_trade_ts"] <= _TRAIN_END
    return df


def _add_all_trade_drawdown(wallet_vol: pd.DataFrame, wallet_vol_all: pd.DataFrame) -> pd.DataFrame:
    wallet_vol = wallet_vol.merge(
        wallet_vol_all[["wallet", "max_drawdown", "total_pnl"]].rename(
            columns={
                "max_drawdown": "max_drawdown_all",
                "total_pnl": "total_pnl_all",
            }
        ),
        on="wallet",
        how="left",
    )
    wallet_vol["max_drawdown_all_to_pnl_all"] = np.where(
        wallet_vol["total_pnl_all"] > 0,
        wallet_vol["max_drawdown_all"] / wallet_vol["total_pnl_all"],
        float("nan"),
    )
    return wallet_vol


def _apply_basic_filter(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["is_train"]]
    test_wallets = set(df.loc[~df["is_train"], "wallet"].unique())
    train_stats = train.groupby("wallet").agg(
        train_pnl=("pnl", "sum"),
        train_markets=("condition_id", "nunique"),
    )
    eligible = train_stats.index[
        train_stats.index.isin(test_wallets)
        & (train_stats["train_pnl"] >= 1_000)
        & (train_stats["train_markets"] >= 20)
    ]
    return df[df["wallet"].isin(eligible)].copy()


def build_sports_candidate_universe(
    trades_dir: Path,
    max_lead_days: int = 2,
    variant: str = "sports",
    exposure_filter_frac: float = _EXPOSURE_FILTER_FRAC,
    exposure_pnl_col: str = _EXPOSURE_PNL_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str], pd.DataFrame, pd.DataFrame]:
    """Replicate the stage1 strategy-selection wallet universe.

    Returns ``(copyable_group, predicting_group, candidate_wallets,
    wallet_vol_train_full, wallet_vol_train)``.
    """
    df = _prepare_stream(trades_dir, max_lead_days)
    df = _apply_basic_filter(df)

    df_all_train_full = df[df["is_train"]].copy()
    wallet_vol_all_full, _ = compute_wallet_metrics(df_all_train_full)
    del df_all_train_full

    df_buy_train_full = df[df["is_train"] & (df["side"] == "BUY")].copy()
    wallet_vol_train_full, _ = compute_wallet_metrics(df_buy_train_full)
    del df_buy_train_full
    wallet_vol_train_full = _add_all_trade_drawdown(
        wallet_vol_train_full, wallet_vol_all_full
    )

    df["deployed_capital"] = df["position"] * df["price"]
    fit_buy = df[df["is_train"] & (df["side"] == "BUY")]
    capture = wallet_capture_thresholds(
        fit_buy,
        (exposure_filter_frac,),
        exposure_pnl_col,
        "deployed_capital",
    )
    del fit_buy
    threshold = df["wallet"].map(
        capture.set_index("wallet")["threshold"]
    ).fillna(0.0)
    keep = df["deployed_capital"].notna() & (df["deployed_capital"] >= threshold)
    df = df[keep].copy()

    df_all_train = df[df["is_train"]].copy()
    wallet_vol_all, _ = compute_wallet_metrics(df_all_train)
    del df_all_train

    df = df[df["side"] == "BUY"].copy().reset_index(drop=True)
    df_slice = df[df["is_train"]].copy()
    wallet_vol_train, _ = compute_wallet_metrics(df_slice)
    del df_slice
    wallet_vol_train = _add_all_trade_drawdown(wallet_vol_train, wallet_vol_all)

    candidates = wallet_vol_train.copy()
    candidates["buy_copyable_factor"] = (
        candidates["buy_copyable_pnl"] / candidates["buy_pnl"]
    )
    test_active_wallets = set(df.loc[~df["is_train"], "wallet"].unique())
    candidates = candidates[candidates["wallet"].isin(test_active_wallets)].copy()

    for col, default in {
        "top10_pnl_pct": np.nan,
        "top_market_abs_pnl_pct": np.nan,
        "market_pnl_hhi": np.nan,
        "positive_bucket_share": np.nan,
    }.items():
        if col not in candidates.columns:
            candidates[col] = default

    for col in [
        "average_roi",
        "median_roi",
        "num_buckets",
        "num_markets",
        "pnl_volatility",
        "max_drawdown_to_pnl",
        "top_market_pnl_pct",
        "top_market_abs_pnl_pct",
        "top5_pnl_pct",
        "top10_pnl_pct",
        "market_pnl_hhi",
        "copyable_roi",
        "copyable_pnl_factor",
        "copyable_pnl",
        "total_notional",
        "max_copyable_drawdown_to_copyable_pnl",
        "worst5_pnl_pct",
        "positive_bucket_share",
    ]:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")

    eligible_wallets = select_strategy_eligible(
        wallet_vol_train_full, wallet_vol_train, variant=variant
    )
    copyable_wallets = select_strategy_selection(
        wallet_vol_train_full, wallet_vol_train, variant=variant
    )

    eligible_base = candidates[candidates["wallet"].isin(eligible_wallets)].copy()
    if eligible_base.empty:
        raise ValueError("no wallets passed base eligibility filters")

    copyable_group = eligible_base[
        eligible_base["wallet"].isin(copyable_wallets)
    ].copy()
    copyable_group["copyable_efficiency"] = (
        copyable_group["copyable_pnl"].fillna(0.0)
        / (copyable_group["total_notional"].fillna(0.0) + 1.0)
    )
    copyable_group["copyable_dd_ratio"] = (
        copyable_group["max_copyable_drawdown_to_copyable_pnl"]
        .fillna(copyable_group["max_drawdown_to_pnl"])
        .fillna(0.0)
    )

    predicting_group = eligible_base[
        ~eligible_base["wallet"].isin(copyable_group["wallet"])
    ].copy()
    predicting_group = predicting_group[
        predicting_group["wallet"] != "0x1f7105a18d9f36aef7c83e5df210e57cf487b2e4"
    ].copy()

    overlap = set(copyable_group["wallet"]) & set(predicting_group["wallet"])
    if overlap:
        raise ValueError(f"wallet groups must be disjoint; overlap={len(overlap)}")

    candidate_wallets = set(copyable_group["wallet"]) | set(predicting_group["wallet"])
    return (
        copyable_group.reset_index(drop=True),
        predicting_group.reset_index(drop=True),
        candidate_wallets,
        wallet_vol_train_full,
        wallet_vol_train,
    )
