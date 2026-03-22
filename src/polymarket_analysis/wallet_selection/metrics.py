"""
Wallet skill metric computation for the signal-v2 pipeline.

Functions here compute per-wallet ``open_buy`` statistics over a streaming
PyArrow dataset (train-a / train-b / full-train / test splits) without loading
the full dataset into memory.

Main entry-point: :func:`compute_wallet_skill_workspace`.
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _new_wallet_stat() -> dict:
    return {
        "open_buy_trades": 0,
        "volume": 0.0,
        "wins": 0,
        "sum_prob_edge": 0.0,
        "sum_prob_edge_sq": 0.0,
        "sum_weighted_edge_num": 0.0,
        "sum_copy_roi": 0.0,
        "sum_copy_roi_sq": 0.0,
        "sum_copy_roi_capped": 0.0,
        "sum_copy_roi_capped_sq": 0.0,
        "sum_copy_pnl_usdc": 0.0,
        "sum_trade_pnl": 0.0,
        "sum_brier": 0.0,
        "sum_price": 0.0,
    }


def aggregate_wallet_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trades into one row per (wallet, condition_id, outcome, dt, side).

    Accepts both fill-level data (columns ``quantity``, ``price``, ``usdc_amount``)
    and pre-grouped transaction-level data (columns ``total_quantity``, ``avg_price``,
    ``trade_value_usdc``).  The latter is normalised to the canonical names before
    aggregation so that downstream metric code is uniform.

    Computes a volume-weighted average price and running position from the
    cumulative signed quantity.  The ``prev_position`` column reflects the
    position *before* the current trade.

    Parameters
    ----------
    df:
        Trade rows.  Must contain one of the two column sets described above,
        plus ``wallet, condition_id, outcome, dt, side``.

    Returns
    -------
    Aggregated DataFrame with additional columns:
        ``price`` (vwap), ``signed_quantity``, ``position``, ``prev_position``
    """
    if df.empty:
        return df.copy()

    work = df.copy()

    # Normalise grouped-schema column names to fill-level names so the rest of
    # the function is uniform regardless of which schema the caller provides.
    if "total_quantity" in work.columns and "quantity" not in work.columns:
        work = work.rename(columns={
            "total_quantity": "quantity",
            "avg_price": "price",
            "trade_value_usdc": "usdc_amount",
        })

    required = [
        "wallet", "condition_id", "outcome", "dt", "side",
        "quantity", "price", "usdc_amount",
    ]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns for aggregation: {missing}")

    work["quantity"] = work["quantity"].astype(float)
    work["usdc_amount"] = work["usdc_amount"].astype(float)
    work["price_x_qty"] = work["price"].astype(float) * work["quantity"]

    agg_map: dict = {
        "quantity": ("quantity", "sum"),
        "usdc_amount": ("usdc_amount", "sum"),
        "price_x_qty": ("price_x_qty", "sum"),
    }
    passthrough_first = ["final_price", "token_winner", "trigger_tx_hash"]
    for col in passthrough_first:
        if col in work.columns:
            agg_map[col] = (col, "first")
    if "trade_pnl" in work.columns:
        agg_map["trade_pnl"] = ("trade_pnl", "sum")

    grouped = (
        work.groupby(
            ["wallet", "condition_id", "outcome", "dt", "side"], as_index=False
        )
        .agg(**agg_map)
        .sort_values(["wallet", "condition_id", "outcome", "dt", "side"])
        .reset_index(drop=True)
    )
    grouped["price"] = grouped["price_x_qty"] / grouped["quantity"].clip(lower=1e-9)
    grouped = grouped.drop(columns=["price_x_qty"])

    grouped["signed_quantity"] = np.where(
        grouped["side"] == "BUY", grouped["quantity"], -grouped["quantity"]
    )
    grouped["side_order"] = np.where(grouped["side"] == "BUY", 0, 1)
    grouped = grouped.sort_values(
        ["wallet", "condition_id", "outcome", "dt", "side_order"]
    ).reset_index(drop=True)
    grouped["position"] = grouped.groupby(
        ["wallet", "condition_id", "outcome"]
    )["signed_quantity"].cumsum()
    grouped["prev_position"] = grouped["position"] - grouped["signed_quantity"]
    return grouped.drop(columns=["side_order"])


# ---------------------------------------------------------------------------
# Metric frame finalisation
# ---------------------------------------------------------------------------

def _finalize_metric_frame(
    stats: dict,
    period_name: str,
    baseline_brier: float,
    market_counts: dict | None = None,
    recent_stats: dict | None = None,
    edge_prior_trades: float = 25.0,
    edge_prior_volume: float = 2500.0,
) -> pd.DataFrame:
    """Convert accumulated stat dicts into a per-wallet metric DataFrame.

    Parameters
    ----------
    stats:
        ``{wallet → {period_name → stat_dict}}`` as built by
        :func:`compute_wallet_skill_workspace`.
    period_name:
        One of ``'train_a'``, ``'train_b'``, ``'full_train'``, ``'test'``.
    baseline_brier:
        Population-average Brier score for the period (used for brier_skill).
    market_counts:
        Optional ``{wallet → distinct_markets}`` map (full-train only).
    recent_stats:
        Optional ``{wallet → {open_buy_trades, volume}}`` for recent activity.
    edge_prior_trades, edge_prior_volume:
        Laplace-like prior counts for shrinkage metrics.

    Returns
    -------
    DataFrame sorted descending by ``open_buy_trades, volume``.
    """
    rows = []
    for wallet, wallet_stats in stats.items():
        s = wallet_stats[period_name]
        n = s["open_buy_trades"]
        volume = s["volume"]

        avg_prob_edge = s["sum_prob_edge"] / n if n else np.nan
        weighted_prob_edge = s["sum_weighted_edge_num"] / volume if volume else np.nan
        avg_copy_roi_capped = s["sum_copy_roi_capped"] / n if n else np.nan
        hit_rate = s["wins"] / n if n else np.nan
        mean_brier = s["sum_brier"] / n if n else np.nan

        prob_edge_var = (
            (s["sum_prob_edge_sq"] / n - avg_prob_edge ** 2) if n else np.nan
        )
        copy_roi_var = (
            (s["sum_copy_roi_capped_sq"] / n - avg_copy_roi_capped ** 2) if n else np.nan
        )
        prob_edge_std = (
            np.sqrt(max(prob_edge_var, 0.0)) if n and not np.isnan(prob_edge_var) else np.nan
        )
        copy_roi_std = (
            np.sqrt(max(copy_roi_var, 0.0)) if n and not np.isnan(copy_roi_var) else np.nan
        )

        rows.append(
            {
                "wallet": wallet,
                "period": period_name,
                "open_buy_trades": n,
                "volume": volume,
                "wins": s["wins"],
                "hit_rate": hit_rate,
                "avg_price": s["sum_price"] / n if n else np.nan,
                "avg_prob_edge": avg_prob_edge,
                "weighted_prob_edge": weighted_prob_edge,
                "prob_edge_shrunk": (
                    s["sum_prob_edge"] / (n + edge_prior_trades)
                    if (n or edge_prior_trades) else 0.0
                ),
                "weighted_prob_edge_shrunk": (
                    s["sum_weighted_edge_num"] / (volume + edge_prior_volume)
                    if (volume or edge_prior_volume) else 0.0
                ),
                "avg_copy_roi": s["sum_copy_roi"] / n if n else np.nan,
                "avg_copy_roi_capped": avg_copy_roi_capped,
                "copy_roi_std": copy_roi_std,
                "edge_sharpe": (
                    avg_prob_edge / prob_edge_std
                    if n and prob_edge_std and prob_edge_std > 1e-12
                    else np.nan
                ),
                "roi_sharpe": (
                    avg_copy_roi_capped / copy_roi_std
                    if n and copy_roi_std and copy_roi_std > 1e-12
                    else np.nan
                ),
                "mean_brier": mean_brier,
                "brier_skill": (
                    1.0 - (mean_brier / baseline_brier)
                    if n and baseline_brier and baseline_brier > 1e-12
                    else np.nan
                ),
                "sum_prob_edge": s["sum_prob_edge"],
                "sum_weighted_edge_num": s["sum_weighted_edge_num"],
                "sum_copy_roi_capped": s["sum_copy_roi_capped"],
                "total_copy_pnl_usdc": s["sum_copy_pnl_usdc"],
                "total_trade_pnl_usdc": s["sum_trade_pnl"],
                "pnl_per_open_buy": s["sum_copy_pnl_usdc"] / n if n else np.nan,
                "pnl_per_1k_volume": (
                    s["sum_copy_pnl_usdc"] * 1000.0 / volume if volume else np.nan
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    if market_counts is not None:
        frame["distinct_markets"] = (
            frame["wallet"].map(market_counts).fillna(0).astype(int)
        )
    if recent_stats is not None:
        frame["recent_open_buy_trades"] = frame["wallet"].map(
            lambda w: recent_stats.get(w, {}).get("open_buy_trades", 0)  # type: ignore[arg-type]
        ).fillna(0).astype(int)
        frame["recent_volume"] = frame["wallet"].map(
            lambda w: recent_stats.get(w, {}).get("volume", 0.0)  # type: ignore[arg-type]
        ).fillna(0.0)

    return (
        frame.sort_values(
            ["open_buy_trades", "volume"], ascending=[False, False]
        ).reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main workspace builder
# ---------------------------------------------------------------------------

def compute_wallet_skill_workspace(
    dataset: Any,
    *,
    train_a_end_date: datetime.date,
    end_date_train: datetime.date,
    recency_days: int = 90,
    batch_size: int = 300_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute wallet skill metrics across all four data splits.

    Reads the dataset in streaming batches, accumulates per-wallet statistics
    for four named periods (``train_a``, ``train_b``, ``full_train``, ``test``),
    then returns one metric DataFrame per period.

    Parameters
    ----------
    dataset:
        PyArrow ``Dataset`` (or any object with ``.to_batches``).
    train_a_end_date:
        Last date (inclusive) of the ``train_a`` split.
    end_date_train:
        Last date (inclusive) of the full training period.
        ``train_b`` is the half-open interval ``(train_a_end_date, end_date_train]``.
        ``test`` is ``> end_date_train``.
    recency_days:
        Window (in calendar days before ``end_date_train``) used for the
        ``recent_open_buy_trades`` feature.
    batch_size:
        Arrow batch size.

    Returns
    -------
    train_a_metrics, train_b_metrics, full_train_metrics, test_metrics
        Four DataFrames, one per split.
    """
    periods = ("train_a", "train_b", "full_train", "test")
    stats: dict = defaultdict(lambda: {p: _new_wallet_stat() for p in periods})
    market_sets_full_train: dict = defaultdict(set)
    recent_stats: dict = defaultdict(lambda: {"open_buy_trades": 0, "volume": 0.0})
    baseline: dict = {p: {"sum_brier": 0.0, "n": 0} for p in periods}

    recent_cutoff = pd.Timestamp(end_date_train, tz="UTC") - pd.Timedelta(
        days=recency_days
    )

    # Accept both fill-level names (quantity/price/usdc_amount) and the
    # pre-grouped names (total_quantity/avg_price/trade_value_usdc) that
    # stage0 now writes.  aggregate_wallet_trades() normalises either set.
    _schema_names = set(dataset.schema.names)
    if "total_quantity" in _schema_names:
        columns = [
            "wallet", "condition_id", "outcome", "dt", "side",
            "total_quantity", "avg_price", "trade_value_usdc",
            "trade_pnl", "final_price", "token_winner",
        ]
    else:
        columns = [
            "wallet", "condition_id", "outcome", "dt", "side",
            "quantity", "price", "usdc_amount",
            "trade_pnl", "final_price", "token_winner",
        ]

    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        df: pd.DataFrame = batch.to_pandas()
        if df.empty:
            continue

        df["dt"] = pd.to_datetime(df["dt"], utc=True)
        df = aggregate_wallet_trades(df)
        # keep only fresh opens
        df = df[(df["side"] == "BUY") & (df["prev_position"] <= 1e-9)].copy()
        if df.empty:
            continue

        trade_date = df["dt"].dt.date
        df["period"] = np.where(
            trade_date <= train_a_end_date,
            "train_a",
            np.where(trade_date <= end_date_train, "train_b", "test"),
        )
        df["prob_edge"] = df["final_price"] - df["price"]
        df["weighted_edge_num"] = df["prob_edge"] * df["usdc_amount"]
        raw_copy_roi = np.where(
            df["token_winner"], 1.0 / df["price"].clip(lower=0.001) - 1.0, -1.0
        )
        df["copy_roi"] = raw_copy_roi
        df["copy_roi_capped"] = np.clip(raw_copy_roi, -1.0, 10.0)
        df["copy_pnl_usdc"] = df["copy_roi"] * df["usdc_amount"]
        if "trade_pnl" not in df.columns:
            df["trade_pnl"] = np.nan
        df["brier"] = (df["price"] - df["final_price"]) ** 2

        agg_cols = {
            "open_buy_trades": ("wallet", "size"),
            "volume": ("usdc_amount", "sum"),
            "wins": ("token_winner", "sum"),
            "sum_prob_edge": ("prob_edge", "sum"),
            "sum_prob_edge_sq": ("prob_edge", lambda s: float(np.square(s).sum())),
            "sum_weighted_edge_num": ("weighted_edge_num", "sum"),
            "sum_copy_roi": ("copy_roi", "sum"),
            "sum_copy_roi_sq": ("copy_roi", lambda s: float(np.square(s).sum())),
            "sum_copy_roi_capped": ("copy_roi_capped", "sum"),
            "sum_copy_roi_capped_sq": (
                "copy_roi_capped", lambda s: float(np.square(s).sum())
            ),
            "sum_copy_pnl_usdc": ("copy_pnl_usdc", "sum"),
            "sum_trade_pnl": ("trade_pnl", "sum"),
            "sum_brier": ("brier", "sum"),
            "sum_price": ("price", "sum"),
        }
        grouped = df.groupby(["wallet", "period"]).agg(**agg_cols).reset_index()

        for row in grouped.itertuples(index=False):
            d = stats[row.wallet][row.period]
            d["open_buy_trades"] += int(row.open_buy_trades)
            d["volume"] += float(row.volume)
            d["wins"] += int(row.wins)
            d["sum_prob_edge"] += float(row.sum_prob_edge)
            d["sum_prob_edge_sq"] += float(row.sum_prob_edge_sq)
            d["sum_weighted_edge_num"] += float(row.sum_weighted_edge_num)
            d["sum_copy_roi"] += float(row.sum_copy_roi)
            d["sum_copy_roi_sq"] += float(row.sum_copy_roi_sq)
            d["sum_copy_roi_capped"] += float(row.sum_copy_roi_capped)
            d["sum_copy_roi_capped_sq"] += float(row.sum_copy_roi_capped_sq)
            d["sum_copy_pnl_usdc"] += float(row.sum_copy_pnl_usdc)
            d["sum_trade_pnl"] += float(row.sum_trade_pnl)
            d["sum_brier"] += float(row.sum_brier)
            d["sum_price"] += float(row.sum_price)
            if row.period in ("train_a", "train_b"):
                full = stats[row.wallet]["full_train"]
                full["open_buy_trades"] += int(row.open_buy_trades)
                full["volume"] += float(row.volume)
                full["wins"] += int(row.wins)
                full["sum_prob_edge"] += float(row.sum_prob_edge)
                full["sum_prob_edge_sq"] += float(row.sum_prob_edge_sq)
                full["sum_weighted_edge_num"] += float(row.sum_weighted_edge_num)
                full["sum_copy_roi"] += float(row.sum_copy_roi)
                full["sum_copy_roi_sq"] += float(row.sum_copy_roi_sq)
                full["sum_copy_roi_capped"] += float(row.sum_copy_roi_capped)
                full["sum_copy_roi_capped_sq"] += float(row.sum_copy_roi_capped_sq)
                full["sum_copy_pnl_usdc"] += float(row.sum_copy_pnl_usdc)
                full["sum_trade_pnl"] += float(row.sum_trade_pnl)
                full["sum_brier"] += float(row.sum_brier)
                full["sum_price"] += float(row.sum_price)

        # Baseline Brier
        base_agg = (
            df.groupby("period")
            .agg(sum_brier=("brier", "sum"), n=("wallet", "size"))
            .reset_index()
        )
        for row in base_agg.itertuples(index=False):
            baseline[row.period]["sum_brier"] += float(row.sum_brier)
            baseline[row.period]["n"] += int(row.n)
            if row.period in ("train_a", "train_b"):
                baseline["full_train"]["sum_brier"] += float(row.sum_brier)
                baseline["full_train"]["n"] += int(row.n)

        # Market sets and recency (full-train only)
        full_train_df = df[df["period"].isin(["train_a", "train_b"])]
        if not full_train_df.empty:
            pairs = full_train_df[["wallet", "condition_id"]].drop_duplicates()
            for row in pairs.itertuples(index=False):
                market_sets_full_train[row.wallet].add(row.condition_id)
            recent_df = full_train_df[full_train_df["dt"] >= recent_cutoff]
            if not recent_df.empty:
                recent_grouped = (
                    recent_df.groupby("wallet")
                    .agg(open_buy_trades=("wallet", "size"), volume=("usdc_amount", "sum"))
                    .reset_index()
                )
                for row in recent_grouped.itertuples(index=False):
                    recent_stats[row.wallet]["open_buy_trades"] += int(row.open_buy_trades)
                    recent_stats[row.wallet]["volume"] += float(row.volume)

    baseline_brier = {
        period: (v["sum_brier"] / v["n"] if v["n"] else np.nan)
        for period, v in baseline.items()
    }
    market_counts = {w: len(m) for w, m in market_sets_full_train.items()}

    train_a = _finalize_metric_frame(
        stats, "train_a", baseline_brier["train_a"]
    )
    train_b = _finalize_metric_frame(
        stats, "train_b", baseline_brier["train_b"]
    )
    full_train = _finalize_metric_frame(
        stats,
        "full_train",
        baseline_brier["full_train"],
        market_counts=market_counts,
        recent_stats=recent_stats,
    )
    test = _finalize_metric_frame(
        stats, "test", baseline_brier["test"]
    )
    return train_a, train_b, full_train, test
