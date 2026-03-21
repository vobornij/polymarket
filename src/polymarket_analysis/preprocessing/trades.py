"""
Trade data processing utilities.

Converts raw lists of trade dicts (as loaded by ``loader.py``) into enriched
DataFrames grouped by wallet × market × timestamp, computes wallet-level P&L
summaries, and filters to the top-N% by P&L.
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Token / market enrichment
# ---------------------------------------------------------------------------

def build_token_lookup(markets: dict[str, dict]) -> dict[str, dict]:
    """Return ``{token_id → {condition_id, outcome, token_winner, final_price}}``.

    For closed/resolved markets the final price is 1.0 (winner) or 0.0 (loser).
    For open markets the last known price from the market definition is used.
    """
    lookup: dict[str, dict] = {}
    for cid, m in markets.items():
        for tok in m.get("tokens", []):
            token_id = str(tok["token_id"])
            winner = bool(tok.get("winner", False))
            if m.get("closed", False):
                final_price = 1.0 if winner else 0.0
            else:
                final_price = float(tok.get("price") or 0.0)
            lookup[token_id] = {
                "condition_id": cid,
                "outcome": tok.get("outcome", ""),
                "token_winner": winner,
                "final_price": final_price,
            }
    return lookup


# ---------------------------------------------------------------------------
# Raw DataFrame construction
# ---------------------------------------------------------------------------

def build_raw_dataframe(
    all_trades: list[dict],
    markets: dict[str, dict],
) -> pd.DataFrame:
    """Convert raw trade dicts into an enriched, sorted DataFrame.

    Steps
    -----
    1. Build a DataFrame from *all_trades*.
    2. Parse timestamps to UTC datetimes.
    3. Merge token-level resolution data (winner, final_price).
    4. Merge market-level metadata (question, end_date, market_slug).
    5. Add ``trade_value_usdc = size × price`` and
       ``final_value_usdc = size × final_price``.

    Returns a DataFrame sorted by ``(condition_id, wallet, dt)``.
    """
    df = pd.DataFrame(all_trades)

    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df.rename(
        columns={"proxyWallet": "wallet", "conditionId": "condition_id"},
        inplace=True,
    )
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df.sort_values(["condition_id", "wallet", "dt"], inplace=True, ignore_index=True)

    # --- token enrichment ---
    token_lookup = build_token_lookup(markets)
    token_df = pd.DataFrame.from_dict(token_lookup, orient="index")
    token_df.index.name = "asset"
    token_df.reset_index(inplace=True)
    df = df.merge(token_df[["asset", "token_winner", "final_price"]], on="asset", how="left")

    # --- market metadata ---
    market_meta = pd.DataFrame(
        [
            {
                "condition_id": cid,
                "question": m["question"],
                "end_date": pd.to_datetime(m["end_date_iso"], utc=True),
                "market_slug": m["market_slug"],
            }
            for cid, m in markets.items()
        ]
    )
    df = df.merge(market_meta, on="condition_id", how="left")

    df["trade_value_usdc"] = df["size"] * df["price"]
    df["final_value_usdc"] = df["size"] * df["final_price"]
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Group raw fills by ``(wallet, condition_id, dt)`` into one row per TX.

    Each row represents all fills made by one wallet in one market at one
    exact timestamp (a single on-chain transaction may have multiple fills).
    """
    group_keys = ["wallet", "condition_id", "dt"]
    grouped = (
        df.groupby(group_keys, sort=False)
        .agg(
            question=("question", "first"),
            market_slug=("market_slug", "first"),
            end_date=("end_date", "first"),
            side=("side", "first"),
            outcome=("outcome", "first"),
            token_winner=("token_winner", "first"),
            final_price=("final_price", "first"),
            total_size=("size", "sum"),
            avg_price=("price", "mean"),
            trade_value_usdc=("trade_value_usdc", "sum"),
            final_value_usdc=("final_value_usdc", "sum"),
            num_fills=("transactionHash", "count"),
        )
        .reset_index()
        .sort_values(["wallet", "condition_id", "dt"])
        .reset_index(drop=True)
    )

    # BUY costs USDC; SELL returns USDC
    grouped["signed_cost"] = np.where(
        grouped["side"] == "BUY",
        grouped["trade_value_usdc"],
        -grouped["trade_value_usdc"],
    )
    grouped["signed_final"] = np.where(
        grouped["side"] == "BUY",
        grouped["final_value_usdc"],
        -grouped["final_value_usdc"],
    )
    return grouped


# ---------------------------------------------------------------------------
# Wallet-level summaries
# ---------------------------------------------------------------------------

def compute_wallet_summary(
    grouped: pd.DataFrame,
    end_date_train: datetime.date,
) -> pd.DataFrame:
    """Compute per-wallet P&L summary using **training data only**.

    Parameters
    ----------
    grouped:
        Output of :func:`aggregate_trades`.
    end_date_train:
        Inclusive upper bound for the training period.  Rows with
        ``dt.date > end_date_train`` are excluded from the summary.

    Returns
    -------
    DataFrame sorted descending by ``pnl_usdc`` with columns:
        wallet, num_markets, num_trades, total_cost_usdc,
        total_final_usdc, pnl_usdc
    """
    cutoff = pd.Timestamp(end_date_train, tz="UTC") + pd.Timedelta(days=1)
    train = grouped[grouped["dt"] < cutoff]

    return (
        train.groupby("wallet")
        .agg(
            num_markets=("condition_id", "nunique"),
            num_trades=("num_fills", "sum"),
            total_cost_usdc=("signed_cost", "sum"),
            total_final_usdc=("signed_final", "sum"),
        )
        .assign(pnl_usdc=lambda x: x["total_final_usdc"] - x["total_cost_usdc"])
        .sort_values("pnl_usdc", ascending=False)
        .reset_index()
    )


def filter_top_wallets(
    wallet_summary: pd.DataFrame,
    df: pd.DataFrame,
    end_date_train: datetime.date,
    quantile: float = 0.95,
) -> pd.DataFrame:
    """Filter *df* to the top wallets by training-period P&L.

    Parameters
    ----------
    wallet_summary:
        Output of :func:`compute_wallet_summary`.
    df:
        Raw fill-level DataFrame (pre-aggregation) to be filtered.
    end_date_train:
        Used to set ``is_train`` flag on the returned rows.
    quantile:
        Percentile threshold, e.g. ``0.95`` for the top 5 %.

    Returns
    -------
    Filtered copy of *df* with an added boolean ``is_train`` column.
    """
    threshold = wallet_summary["pnl_usdc"].quantile(quantile)
    top_wallets = set(
        wallet_summary.loc[wallet_summary["pnl_usdc"] >= threshold, "wallet"]
    )
    result = df[df["wallet"].isin(top_wallets)].copy()
    result["is_train"] = result["dt"].dt.date <= end_date_train
    return result
