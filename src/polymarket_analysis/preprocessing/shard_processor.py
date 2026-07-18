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
    "wallet", "side", "copyable_qty", "avail_copy_total_vol", "avail_copy_count",
]

_GROUP_KEYS = ["tx_hash", "wallet", "side", "token_id"]


# ---------------------------------------------------------------------------
# Phase 1 — select top-pct wallets per shard (no trade data returned)
# ---------------------------------------------------------------------------

def select_top_wallets_shard(
    file_path: Path,
    token_lookup_df: pd.DataFrame,
    end_train_ts: pd.Timestamp,
    top_pct: float = 0.04,
    selection_pnl: str = "copyable_pnl",
) -> tuple[dict[str, float], dict]:
    """Identify the top-*top_pct* wallets by training P&L within one shard.

    Parameters
    ----------
    file_path:
        Path to a raw-trades parquet shard.
    token_lookup_df:
        DataFrame with columns ``[token_id, token_winner, final_price]``.
    end_train_ts:
        Train data has resolution < *end_train_ts*.
    top_pct:
        Fraction of wallets to keep per shard (default 4 %).

    selection_pnl:
        Wallet-ranking target computed on training rows.
        Supported values: ``"trade_pnl"`` (default), ``"copyable_pnl"``.

    Returns
    -------
    wallet_pnl : dict[str, float]
        ``{wallet: training_pnl_usdc}`` for wallets in the top-*top_pct* of
        this shard.  Empty dict if no in-range training rows.
    stats : dict
        Scalar diagnostics: ``raw_rows``, ``in_range_rows``,
        ``candidate_wallets``, ``selected_wallets``.
    """
    print(f"Processing shard {file_path.name}...")
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
    raw['ts'] = pd.to_datetime(raw['block_timestamp'], utc=True)
    enriched = raw.merge(
        token_lookup_df,
        on="token_id",
        how="inner",
    )

    if enriched.empty:
        return {}, stats

    enriched["dt"] = pd.to_datetime(enriched["block_timestamp"], unit="s", utc=True)
    stats["in_range_rows"] = len(enriched)

    # Restrict to training period only
    train = enriched[enriched["last_condition_trade_ts"] < end_train_ts].copy()
    if train.empty:
        return {}, stats

    train.loc[:, "trade_pnl"] = np.where(
        train["side"] == "BUY",
        train["quantity"] * (train["final_price"] - train["price"]),
        train["quantity"] * (train["price"] - train["final_price"]),
    )

    train.loc[:, "copyable_pnl"] = (
        train['copyable_qty'].clip(lower=0, upper=train['quantity'])
        * (train["final_price"] - train["price"])
        * np.where(train["side"] == "BUY", 1, -1)
    )

    if selection_pnl not in {"trade_pnl", "copyable_pnl"}:
        raise ValueError(
            f"Unsupported selection_pnl={selection_pnl!r}; "
            "expected 'trade_pnl' or 'copyable_pnl'."
        )

    pnl_col = selection_pnl

    wallet_pnl_series: pd.Series = (
        pd.Series(train[pnl_col], index=train.index, dtype=float)
        .groupby(train["wallet"].to_numpy())
        .sum()
    )

    stats["candidate_wallets"] = len(wallet_pnl_series)
    threshold: float = float(wallet_pnl_series.quantile(1.0 - top_pct))
    stats['threshold'] = threshold
    print(f"Shard top {top_pct:.2%} threshold: {threshold:.2f} USDC")
    top = wallet_pnl_series[wallet_pnl_series >= threshold]
    stats["selected_wallets"] = len(top)
    stats['threshold'] = threshold

    return dict(top), stats


# ---------------------------------------------------------------------------
# Phase 2 — enrich, group, and filter to a given wallet set
# ---------------------------------------------------------------------------

def enrich_and_group_shard(
    file_path: Path,
    token_lookup_df: pd.DataFrame,
    end_train_ts: pd.Timestamp,
    top_wallets: set[str],
    wallet_pnl_metric: str = "copyable_pnl",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Enrich one shard, group by tx×wallet×side, filter to *top_wallets*.

    Parameters
    ----------
    file_path:
        Path to a raw-trades parquet shard.
    token_lookup_df:
        DataFrame with columns ``[token_id, token_winner, final_price]``.
    end_train_ts:
        threshold of resolution time
    top_wallets:
        Set of wallet addresses to keep.  Rows for other wallets are dropped.
    wallet_pnl_metric:
        Which per-row PnL column to aggregate when returning ``wallet_train_pnl``.
        Supported values: ``"trade_pnl"`` (default), ``"copyable_pnl"``.

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
        token_lookup_df[["token_id", "token_winner", "final_price", "last_condition_trade_ts"]],
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

    # Group fills → one row per tx_hash × wallet × side × token_id
    grouped = (
        enriched.groupby(_GROUP_KEYS, sort=False)
        .agg(
            dt               = ("dt",              "first"),
            condition_id     = ("condition_id",    "first"),
            outcome          = ("outcome",          "first"),
            token_winner     = ("token_winner",     "first"),
            final_price      = ("final_price",      "first"),
            last_condition_trade_ts = ("last_condition_trade_ts", "first"),
            position         = ("position",         "max"),
            total_quantity   = ("quantity",         "sum"),
            price_x_qty_sum  = ("price_x_qty",     "sum"),
            trade_value_usdc = ("usdc_amount",      "sum"),
            final_value_usdc = ("final_value_usdc", "sum"),
            num_fills        = ("log_index",        "count"),
            copyable_qty    = ("copyable_qty",   "sum"),
            avail_copy_total_vol = ("avail_copy_total_vol", "sum"),
            avail_copy_count  = ("avail_copy_count", "sum"),
        )
        .reset_index()
    )

    is_buy = grouped["side"] == "BUY"
    grouped["trade_pnl"] = np.where(
        is_buy,
        grouped["final_value_usdc"] - grouped["trade_value_usdc"],
        grouped["trade_value_usdc"] - grouped["final_value_usdc"],
    )

    grouped["copyable_pnl"] = (
        (grouped['copyable_qty'].clip(lower=0, upper=grouped['total_quantity']) / grouped['total_quantity'])
        * grouped["trade_pnl"]
    )

    if wallet_pnl_metric not in {"trade_pnl", "copyable_pnl"}:
        raise ValueError(
            f"Unsupported wallet_pnl_metric={wallet_pnl_metric!r}; "
            "expected 'trade_pnl' or 'copyable_pnl'."
        )

    # Per-wallet training P&L from this shard
    train_grouped = grouped[grouped["last_condition_trade_ts"] < end_train_ts]
    wallet_train_pnl: dict[str, float] = (
        train_grouped.groupby("wallet")[wallet_pnl_metric].sum().to_dict()
        if not train_grouped.empty
        else {}
    )

    return grouped, wallet_train_pnl
