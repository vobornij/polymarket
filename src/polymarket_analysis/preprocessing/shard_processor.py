"""
Per-shard trade processing workers for the stage0 pipeline.

Importable worker functions so that ``ProcessPoolExecutor`` (which uses
``spawn`` on macOS) can pickle and call them from worker subprocesses.
Functions defined interactively in a Jupyter notebook cannot be pickled by
spawned workers; this module solves that problem.

The wallet selection is **global**:

1. ``wallet_pnl_shard`` scans each enriched shard once with a slim column set
   and returns the per-wallet **training** P&L dict for *all* wallets of that
   shard (``trade_pnl`` or ``copyable_pnl``).  The notebook sums these dicts
   across shards and cuts the global top-*top_pct* — this way every selected
   wallet is exported from *every* shard it traded in, not just the shard
   where it was selected.
2. ``group_and_export_shard`` re-reads each enriched shard, filters fills to
   the caller-supplied global wallet set, groups them to one row per
   ``tx_hash × wallet × side × token_id`` (all periods), and writes the export
   shard directly to ``OUT_DIR``.  Shards are partitioned by the first hex
   char of ``condition_id`` and a group's token pins it to exactly one shard,
   so no cross-shard re-grouping is needed.
3. Only small per-wallet / per-market aggregates come back to the notebook
   (conditions live in exactly one shard, so ``nunique`` counts sum exactly
   across shards).

The enriched input already carries ``token_winner`` and
``last_condition_trade_ts`` (embedded by
:func:`polymarket_analysis.preprocessing.fill_extender.enrich_shard`), so no
token-lookup joins are needed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from polymarket_analysis.preprocessing.fill_extender import (
    avail_copy_count_col,
    avail_copy_qty_col,
    avail_copy_total_vol_col,
    copyable_qty_col,
)

_GROUP_KEYS = ["tx_hash", "wallet", "side", "token_id"]

# String columns dictionary-encoded after reading (large memory saving).
_CAT_COLS = ["tx_hash", "wallet", "condition_id", "token_id", "side", "outcome"]


def _read_cols(variants: list[tuple[int, float]]) -> list[str]:
    """All columns needed to group and export one enriched shard."""
    cols = [
        "tx_hash", "log_index", "block_timestamp", "condition_id",
        "token_id", "outcome", "token_winner", "wallet", "side",
        "price", "quantity", "usdc_amount", "position",
        "last_condition_trade_ts",
    ]
    for window_seconds, factor in variants:
        cols.append(avail_copy_qty_col(window_seconds, factor))
        if abs(factor - 1.0) < 1e-12:
            cols.append(avail_copy_total_vol_col(window_seconds, factor))
            cols.append(avail_copy_count_col(window_seconds, factor))
    return cols


def _pnl_read_cols(variants: list[tuple[int, float]], selection_pnl: str) -> list[str]:
    """Slim column set needed only for the wallet P&L ranking scan."""
    cols = ["wallet", "side", "price", "quantity", "token_winner", "last_condition_trade_ts"]
    if selection_pnl == "copyable_pnl":
        cols.append(avail_copy_qty_col(*variants[0]))
    return cols


def _export_cols(variants: list[tuple[int, float]]) -> list[str]:
    """Export column order (matches the historical stage0 export schema).

    Per variant: ``copyable_qty_*`` and ``avail_copy_qty_*``; the *primary*
    variant (``variants[0]``, factor 1.0) additionally exports
    ``avail_copy_total_vol_*`` / ``avail_copy_count_*``.
    """
    cols = [
        "wallet", "condition_id", "token_id", "dt",
        "side", "outcome", "position",
        "total_quantity", "avg_price",
        "trade_value_usdc", "final_value_usdc", "trade_pnl", "copyable_pnl",
        "token_winner", "final_price",
        "last_condition_trade_ts",
        "tx_hash", "num_fills", "is_train",
    ]
    for k, (window_seconds, factor) in enumerate(variants):
        cols.append(copyable_qty_col(window_seconds, factor))
        cols.append(avail_copy_qty_col(window_seconds, factor))
        if k == 0 and abs(factor - 1.0) < 1e-12:
            cols.append(avail_copy_total_vol_col(window_seconds, factor))
            cols.append(avail_copy_count_col(window_seconds, factor))
    return cols


def _read_table(
    file_path: Path, read_cols: list[str], sample_rows: int | None,
) -> pa.Table:
    """Read an enriched shard as an arrow table (single disk load).

    ``sample_rows`` reads only the first rows of the file.
    """
    if sample_rows is None:
        return pq.read_table(file_path, columns=read_cols)
    pf = pq.ParquetFile(file_path)
    batches = []
    rows = 0
    for batch in pf.iter_batches(batch_size=100_000, columns=read_cols):
        if rows + batch.num_rows > sample_rows:
            batch = batch.slice(0, sample_rows - rows)
        batches.append(batch)
        rows += batch.num_rows
        if rows >= sample_rows:
            break
    return pa.Table.from_batches(batches)


def _to_pandas(tbl: pa.Table) -> pd.DataFrame:
    """Convert an enriched-shard arrow table to a memory-lean pandas frame.

    Arrow dtypes avoid materialising object strings; the id columns are then
    dictionary-encoded and the remaining arrow columns converted to numpy.
    """
    df = tbl.to_pandas(types_mapper=pd.ArrowDtype)

    for col in _CAT_COLS:
        df[col] = df[col].astype("category")
    df["last_condition_trade_ts"] = df["last_condition_trade_ts"].astype(
        "datetime64[ns, UTC]"
    )
    for col in df.columns:
        if isinstance(df[col].dtype, pd.ArrowDtype):
            df[col] = df[col].to_numpy()
    return df


# ---------------------------------------------------------------------------
# Phase 1 — per-wallet training P&L for ALL wallets of one shard
# ---------------------------------------------------------------------------

def wallet_pnl_shard(
    file_path: Path,
    end_train_ts: pd.Timestamp,
    selection_pnl: str = "copyable_pnl",
    variants: list[tuple[int, float]] | None = None,
    sample_rows: int | None = None,
) -> tuple[dict[str, float], dict]:
    """Per-wallet training P&L for **all** wallets of one enriched shard.

    Slim-column scan (only the ranking-metric columns are read).  Returns the
    full ``{wallet: pnl}`` dict — no top-pct cut — plus scalar stats; the
    caller sums these dicts across shards and applies the global cut.

    Parameters
    ----------
    file_path:
        Path to an enriched raw-trades parquet shard
        (``enriched_*.parquet`` with ``token_winner`` and
        ``last_condition_trade_ts`` columns).
    end_train_ts:
        Train data has ``last_condition_trade_ts <= end_train_ts``.
    selection_pnl:
        Wallet-ranking metric on training rows: ``"copyable_pnl"`` or
        ``"trade_pnl"``.
    variants:
        Copy variants ``(window_seconds, factor)`` used when the shard was
        enriched; the primary variant (``variants[0]``) drives
        ``copyable_pnl``.
    sample_rows:
        Read only the first rows of the shard (smoke-testing).

    Returns
    -------
    wallet_pnl : dict[str, float]
        ``{wallet: training_pnl}`` for every wallet with training rows.
    stats : dict
        Scalar diagnostics (rows_read, train_rows, candidate_wallets).
    """
    if variants is None:
        raise ValueError("variants is required (set it in the notebook parameters cell)")
    if selection_pnl not in {"trade_pnl", "copyable_pnl"}:
        raise ValueError(
            f"Unsupported selection_pnl={selection_pnl!r}; "
            "expected 'trade_pnl' or 'copyable_pnl'."
        )

    print(f"Scanning shard {file_path.name}...")
    tbl = _read_table(file_path, _pnl_read_cols(variants, selection_pnl), sample_rows)

    stats: dict = {
        "rows_read": tbl.num_rows,
        "train_rows": 0,
        "candidate_wallets": 0,
    }
    if tbl.num_rows == 0:
        return {}, stats

    end_train_ns = pd.Timestamp(end_train_ts).value
    wallet_dict = pc.dictionary_encode(tbl["wallet"]).combine_chunks()
    codes = wallet_dict.indices.to_numpy(zero_copy_only=False)
    is_buy = pc.equal(tbl["side"], "BUY").to_numpy(zero_copy_only=False)
    token_winner = tbl["token_winner"].to_numpy(zero_copy_only=False)
    price = tbl["price"].to_numpy(zero_copy_only=False)
    qty = tbl["quantity"].to_numpy(zero_copy_only=False)
    final_price = np.where(token_winner, 1.0, 0.0)
    train_mask = (
        pc.cast(tbl["last_condition_trade_ts"], pa.int64()).to_numpy(zero_copy_only=False)
        <= end_train_ns
    )
    stats["train_rows"] = int(train_mask.sum())

    codes_train = codes[train_mask]
    if codes_train.size == 0:
        return {}, stats

    if selection_pnl == "trade_pnl":
        row_pnl = np.where(
            is_buy, qty * (final_price - price), qty * (price - final_price)
        )[train_mask]
    else:
        primary_avail = tbl[avail_copy_qty_col(*variants[0])].to_numpy(zero_copy_only=False)
        copy_qty = np.minimum(qty, primary_avail).clip(min=0)
        row_pnl = (copy_qty * (final_price - price) * np.where(is_buy, 1.0, -1.0))[train_mask]

    n_codes = len(wallet_dict.dictionary)
    pnl_by_code = np.bincount(codes_train, weights=row_pnl, minlength=n_codes).astype(np.float64)
    present_codes = np.flatnonzero(np.bincount(codes_train, minlength=n_codes))
    stats["candidate_wallets"] = present_codes.size

    wallet_pnl = dict(
        zip(
            wallet_dict.dictionary.take(present_codes).to_pylist(),
            pnl_by_code[present_codes].tolist(),
        )
    )
    return wallet_pnl, stats


# ---------------------------------------------------------------------------
# Phase 2 — filter to the global wallet set, group fills, export
# ---------------------------------------------------------------------------

def group_and_export_shard(
    file_path: Path,
    out_path: Path,
    end_train_ts: pd.Timestamp,
    top_wallets: set[str],
    variants: list[tuple[int, float]] | None = None,
    sample_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Filter one enriched shard to *top_wallets*, group fills, export.

    ``top_wallets`` is the caller's global wallet selection; rows for other
    wallets are dropped.  Groups are written directly to *out_path*.

    Parameters
    ----------
    file_path:
        Path to an enriched raw-trades parquet shard.
    out_path:
        Where the grouped/export parquet for this shard is written.
    end_train_ts:
        Train data has ``last_condition_trade_ts <= end_train_ts``.
    top_wallets:
        Global wallet set to keep (from the notebook's Phase 1).
    variants:
        Copy variants ``(window_seconds, factor)`` used when the shard was
        enriched; the primary variant (``variants[0]``) drives
        ``copyable_pnl`` and the vol/count export columns.
    sample_rows:
        Read only the first rows of the shard (smoke-testing).

    Returns
    -------
    wallet_summary_part : pd.DataFrame
        Per-wallet training aggregates for the kept wallets
        (num_markets/num_trades/total_cost_usdc/trade_pnl/copyable_pnl) —
        safe to sum across shards.
    market_summary_part : pd.DataFrame
        Per-condition aggregates (num_wallets/num_trades/volume_usdc/
        final_value_usdc) for the kept wallets — safe to concat across shards
        (each condition lives in exactly one shard).
    stats : dict
        Scalar diagnostics (rows_read, grouped_rows, train_rows, test_rows,
        wallets_in_shard).
    """
    if variants is None:
        raise ValueError("variants is required (set it in the notebook parameters cell)")

    print(f"Processing shard {file_path.name}...")
    tbl = _read_table(file_path, _read_cols(variants), sample_rows)

    stats: dict = {
        "rows_read": tbl.num_rows,
        "grouped_rows": 0,
        "train_rows": 0,
        "test_rows": 0,
        "wallets_in_shard": 0,
    }
    if tbl.num_rows == 0:
        return pd.DataFrame(), pd.DataFrame(), stats

    # ── filter to the global wallet set (arrow, before pandas conversion) ───
    mask = pc.is_in(tbl["wallet"], value_set=pa.array(sorted(top_wallets))).to_numpy(
        zero_copy_only=False
    )
    tbl = tbl.filter(pa.array(mask))
    del mask

    df = _to_pandas(tbl)
    del tbl
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), stats
    stats["wallets_in_shard"] = df["wallet"].nunique()

    df["dt"] = pd.to_datetime(df["block_timestamp"], unit="s", utc=True)
    df["final_price"] = np.where(df["token_winner"], 1.0, 0.0)
    df["final_value_usdc"] = df["quantity"] * df["final_price"]
    df["price_x_qty"] = df["price"] * df["quantity"]

    agg_dict = {
        "dt":                     ("dt",                     "first"),
        "condition_id":           ("condition_id",           "first"),
        "outcome":                ("outcome",                "first"),
        "token_winner":           ("token_winner",           "first"),
        "final_price":            ("final_price",            "first"),
        "last_condition_trade_ts": ("last_condition_trade_ts", "first"),
        "position":               ("position",               "max"),
        "total_quantity":         ("quantity",               "sum"),
        "price_x_qty_sum":        ("price_x_qty",            "sum"),
        "trade_value_usdc":       ("usdc_amount",            "sum"),
        "final_value_usdc":       ("final_value_usdc",       "sum"),
        "num_fills":              ("log_index",              "count"),
    }
    for k, (window_seconds, factor) in enumerate(variants):
        qty_col = avail_copy_qty_col(window_seconds, factor)
        agg_dict[qty_col] = (qty_col, "max")
        if k == 0 and abs(factor - 1.0) < 1e-12:
            vol_col = avail_copy_total_vol_col(window_seconds, factor)
            cnt_col = avail_copy_count_col(window_seconds, factor)
            agg_dict[vol_col] = (vol_col, "sum")
            agg_dict[cnt_col] = (cnt_col, "sum")

    grouped = (
        df.groupby(_GROUP_KEYS, sort=False, observed=True)
        .agg(**agg_dict)
        .reset_index()
    )

    for window_seconds, factor in variants:
        qty_col = avail_copy_qty_col(window_seconds, factor)
        grouped[copyable_qty_col(window_seconds, factor)] = grouped["total_quantity"].clip(
            upper=grouped[qty_col]
        )

    is_buy_grouped = grouped["side"] == "BUY"
    grouped["trade_pnl"] = np.where(
        is_buy_grouped,
        grouped["final_value_usdc"] - grouped["trade_value_usdc"],
        grouped["trade_value_usdc"] - grouped["final_value_usdc"],
    )
    primary_copy_qty = copyable_qty_col(*variants[0])
    grouped["copyable_pnl"] = (
        (grouped[primary_copy_qty].clip(lower=0, upper=grouped["total_quantity"])
         / grouped["total_quantity"])
        * grouped["trade_pnl"]
    )
    grouped["avg_price"] = grouped["price_x_qty_sum"] / grouped["total_quantity"]
    grouped["is_train"] = grouped["last_condition_trade_ts"] <= end_train_ts

    stats["grouped_rows"] = len(grouped)
    stats["train_rows"] = int(grouped["is_train"].sum())
    stats["test_rows"] = len(grouped) - stats["train_rows"]

    # ── write export shard ──────────────────────────────────────────────────
    export_df = grouped[_export_cols(variants)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_parquet(out_path, index=False)

    # ── small aggregates for cross-shard summaries ──────────────────────────
    train_grouped = grouped[grouped["is_train"]]
    wallet_summary_part = (
        train_grouped.groupby("wallet", observed=True)
        .agg(
            num_markets=("condition_id", "nunique"),
            num_trades=("num_fills", "sum"),
            total_cost_usdc=("trade_value_usdc", "sum"),
            trade_pnl=("trade_pnl", "sum"),
            copyable_pnl=("copyable_pnl", "sum"),
        )
        .reset_index()
    )
    market_summary_part = (
        grouped.groupby("condition_id", observed=True)
        .agg(
            num_wallets=("wallet", "nunique"),
            num_trades=("num_fills", "sum"),
            volume_usdc=("trade_value_usdc", "sum"),
            final_value_usdc=("final_value_usdc", "sum"),
        )
        .reset_index()
    )

    return wallet_summary_part, market_summary_part, stats
