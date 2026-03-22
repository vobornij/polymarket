"""
Per-shard trade processing workers.

Two importable worker functions are provided so that ``ProcessPoolExecutor``
(which uses ``spawn`` on macOS) can pickle and call them from worker
subprocesses.  Functions defined interactively in a Jupyter notebook cannot
be pickled by spawned workers; this module solves that problem.

Phase 1 – wallet selection
--------------------------
``select_top_wallets_shard`` reads one raw shard and returns a
``{wallet: pnl_usdc}`` dict for the top-*top_pct* wallets in that shard
(training period only).  No trade data is returned, keeping memory small.

Phase 2 – grouping + filtering
-------------------------------
``enrich_and_group_shard`` reads one raw shard, inner-joins settlement data,
groups fills by ``tx_hash × wallet × side``, and filters to a caller-supplied
set of wallets.  Returns the grouped DataFrame and the per-wallet training-P&L
for that shard (so the caller can accumulate a cross-shard total P&L).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Columns read from each raw shard parquet.
_READ_COLS = [
    "tx_hash", "log_index", "block_timestamp", "trade_date", "condition_id",
    "token_id", "outcome", "price", "quantity", "usdc_amount", "position",
    "wallet", "side",
]

_GROUP_KEYS = ["tx_hash", "wallet", "side"]


# ---------------------------------------------------------------------------
# Phase 1 — select top-pct wallets per shard (no trade data returned)
# ---------------------------------------------------------------------------

def select_top_wallets_shard(
    file_path: Path,
    token_lookup_df: pd.DataFrame,
    end_train_ts: pd.Timestamp,
    top_pct: float = 0.04,
) -> tuple[dict[str, float], dict]:
    """Identify the top-*top_pct* wallets by training P&L within one shard.

    Parameters
    ----------
    file_path:
        Path to a raw-trades parquet shard.
    token_lookup_df:
        DataFrame with columns ``[token_id, token_winner, final_price]``.
    end_train_ts:
        Upper bound (exclusive) for the training period.
    top_pct:
        Fraction of wallets to keep per shard (default 4 %).

    Returns
    -------
    wallet_pnl : dict[str, float]
        ``{wallet: training_pnl_usdc}`` for wallets in the top-*top_pct* of
        this shard.  Empty dict if no in-range training rows.
    stats : dict
        Scalar diagnostics: ``raw_rows``, ``in_range_rows``,
        ``candidate_wallets``, ``selected_wallets``.
    """
    raw = pd.read_parquet(file_path, columns=_READ_COLS)
    stats: dict = {
        "raw_rows": len(raw),
        "in_range_rows": 0,
        "candidate_wallets": 0,
        "selected_wallets": 0,
    }

    if raw.empty:
        return {}, stats

    raw["token_id"] = raw["token_id"].astype(str)
    enriched = raw.merge(
        token_lookup_df[["token_id", "final_price"]],
        on="token_id",
        how="inner",
    )
    if enriched.empty:
        return {}, stats

    enriched["dt"] = pd.to_datetime(enriched["block_timestamp"], unit="s", utc=True)
    stats["in_range_rows"] = len(enriched)

    # Restrict to training period only
    train = enriched[enriched["dt"] < end_train_ts]
    if train.empty:
        return {}, stats

    # Per-fill P&L
    is_buy = train["side"] == "BUY"
    final_value = train["quantity"] * train["final_price"]
    trade_pnl = np.where(
        is_buy,
        final_value - train["usdc_amount"],
        train["usdc_amount"] - final_value,
    )

    wallet_pnl_series: pd.Series = (
        pd.Series(trade_pnl, index=train.index, dtype=float)
        .groupby(train["wallet"].to_numpy())
        .sum()
    )

    stats["candidate_wallets"] = len(wallet_pnl_series)
    threshold: float = float(wallet_pnl_series.quantile(1.0 - top_pct))
    top = wallet_pnl_series[wallet_pnl_series >= threshold]
    stats["selected_wallets"] = len(top)

    return dict(top), stats


# ---------------------------------------------------------------------------
# Phase 2 — enrich, group, and filter to a given wallet set
# ---------------------------------------------------------------------------

def enrich_and_group_shard(
    file_path: Path,
    token_lookup_df: pd.DataFrame,
    end_train_ts: pd.Timestamp,
    top_wallets: set[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Enrich one shard, group by tx×wallet×side, filter to *top_wallets*.

    Parameters
    ----------
    file_path:
        Path to a raw-trades parquet shard.
    token_lookup_df:
        DataFrame with columns ``[token_id, token_winner, final_price]``.
    end_train_ts:
        Used to label ``is_train`` and to compute per-wallet training P&L.
    top_wallets:
        Set of wallet addresses to keep.  Rows for other wallets are dropped.

    Returns
    -------
    grouped : pd.DataFrame
        One row per ``tx_hash × wallet × side`` for wallets in *top_wallets*.
        Empty DataFrame if no matching rows.
    wallet_train_pnl : dict[str, float]
        ``{wallet: training_pnl_usdc}`` accumulated from this shard (training
        rows only), restricted to *top_wallets*.
    """
    raw = pd.read_parquet(file_path, columns=_READ_COLS)

    if raw.empty:
        return pd.DataFrame(), {}

    raw["token_id"] = raw["token_id"].astype(str)
    enriched = raw.merge(
        token_lookup_df[["token_id", "token_winner", "final_price"]],
        on="token_id",
        how="inner",
    )
    if enriched.empty:
        return pd.DataFrame(), {}

    # Filter to top wallets early to reduce work
    enriched = enriched[enriched["wallet"].isin(top_wallets)]
    if enriched.empty:
        return pd.DataFrame(), {}

    enriched["dt"] = pd.to_datetime(enriched["block_timestamp"], unit="s", utc=True)
    enriched["final_value_usdc"] = enriched["quantity"] * enriched["final_price"]
    enriched["price_x_qty"] = enriched["price"] * enriched["quantity"]

    # Group fills → one row per tx_hash × wallet × side
    grouped = (
        enriched.groupby(_GROUP_KEYS, sort=False)
        .agg(
            dt               = ("dt",              "first"),
            condition_id     = ("condition_id",    "first"),
            outcome          = ("outcome",          "first"),
            token_winner     = ("token_winner",     "first"),
            final_price      = ("final_price",      "first"),
            position         = ("position",         "max"),
            total_quantity   = ("quantity",         "sum"),
            price_x_qty_sum  = ("price_x_qty",     "sum"),
            trade_value_usdc = ("usdc_amount",      "sum"),
            final_value_usdc = ("final_value_usdc", "sum"),
            num_fills        = ("log_index",        "count"),
        )
        .reset_index()
    )

    is_buy = grouped["side"] == "BUY"
    grouped["trade_pnl"] = np.where(
        is_buy,
        grouped["final_value_usdc"] - grouped["trade_value_usdc"],
        grouped["trade_value_usdc"] - grouped["final_value_usdc"],
    )

    # Per-wallet training P&L from this shard
    train_grouped = grouped[grouped["dt"] < end_train_ts]
    wallet_train_pnl: dict[str, float] = (
        train_grouped.groupby("wallet")["trade_pnl"].sum().to_dict()
        if not train_grouped.empty
        else {}
    )

    return grouped, wallet_train_pnl
