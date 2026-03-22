"""
Signal event construction for the copy-trade pipeline.

Two main building blocks:

* :func:`build_wallet_profiles`    — size and price statistics per selected wallet
* :func:`build_signal_events`      — stream the dataset, classify open/add/close/reduce
                                     events, attach wallet profiles and consensus features
* :func:`attach_consensus_features` — add prior same/opp trader counts per market
* :func:`verify_partial_fill_grouping` — sanity-check the aggregation logic
"""

from __future__ import annotations

import datetime
from collections import defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd

from ..wallet_selection.metrics import aggregate_wallet_trades

# ---------------------------------------------------------------------------
# Default price bucket configuration
# ---------------------------------------------------------------------------

DEFAULT_PRICE_BUCKET_BINS = [0.0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0]
DEFAULT_PRICE_BUCKET_LABELS = [
    "0.00-0.10", "0.10-0.25", "0.25-0.40", "0.40-0.60",
    "0.60-0.75", "0.75-0.90", "0.90-1.00",
]


# ---------------------------------------------------------------------------
# Wallet profiles
# ---------------------------------------------------------------------------

def build_wallet_profiles(
    dataset: Any,
    selected_wallets: pd.DataFrame,
    *,
    end_date_train: datetime.date,
    train_a_end_date: datetime.date,
    period: str = "full_train",
    batch_size: int = 300_000,
) -> pd.DataFrame:
    """Compute median open-buy size and mean entry price for each selected wallet.

    Parameters
    ----------
    dataset:
        PyArrow ``Dataset`` (or any object with ``.to_batches``).
    selected_wallets:
        DataFrame with at least ``wallet`` and ``wallet_quality`` columns.
    end_date_train:
        Last training date (inclusive).
    train_a_end_date:
        Last train-a date (inclusive).
    period:
        One of ``'full_train'``, ``'train_a'``, ``'train_b'``, ``'test'``.
    batch_size:
        Arrow batch size.

    Returns
    -------
    DataFrame with columns:
        ``wallet``, ``wallet_quality``,
        ``median_open_buy_usdc``, ``mean_open_buy_usdc``, ``mean_open_buy_price``
    """
    selected_wallet_set = set(selected_wallets["wallet"])
    size_lists: dict[str, list[float]] = defaultdict(list)
    entry_price_lists: dict[str, list[float]] = defaultdict(list)

    _schema_names = set(dataset.schema.names)
    if "total_quantity" in _schema_names:
        columns = [
            "wallet", "condition_id", "outcome", "dt",
            "side", "total_quantity", "trade_value_usdc", "avg_price",
        ]
    else:
        columns = [
            "wallet", "condition_id", "outcome", "dt",
            "side", "quantity", "usdc_amount", "price",
        ]

    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        df: pd.DataFrame = batch.to_pandas()
        if df.empty:
            continue

        df["dt"] = pd.to_datetime(df["dt"], utc=True)
        if period == "full_train":
            df = df[df["dt"].dt.date <= end_date_train]
        elif period == "train_a":
            df = df[df["dt"].dt.date <= train_a_end_date]
        elif period == "train_b":
            df = df[
                (df["dt"].dt.date > train_a_end_date)
                & (df["dt"].dt.date <= end_date_train)
            ]
        elif period == "test":
            df = df[df["dt"].dt.date > end_date_train]

        df = df[df["wallet"].isin(selected_wallet_set)].copy()
        if df.empty:
            continue

        df = aggregate_wallet_trades(df)
        df = df[(df["side"] == "BUY") & (df["prev_position"] <= 1e-9)].copy()
        if df.empty:
            continue

        for row in df[["wallet", "usdc_amount", "price"]].itertuples(index=False):
            size_lists[row.wallet].append(float(row.usdc_amount))
            entry_price_lists[row.wallet].append(float(row.price))

    profiles = selected_wallets[["wallet", "wallet_quality"]].copy()
    profiles["median_open_buy_usdc"] = profiles["wallet"].map(
        lambda w: float(np.median(size_lists[w])) if size_lists.get(w) else np.nan
    )
    profiles["mean_open_buy_usdc"] = profiles["wallet"].map(
        lambda w: float(np.mean(size_lists[w])) if size_lists.get(w) else np.nan
    )
    profiles["mean_open_buy_price"] = profiles["wallet"].map(
        lambda w: float(np.mean(entry_price_lists[w])) if entry_price_lists.get(w) else np.nan
    )
    return profiles


# ---------------------------------------------------------------------------
# Consensus features
# ---------------------------------------------------------------------------

def attach_consensus_features(open_buys: pd.DataFrame) -> pd.DataFrame:
    """Add prior-trader consensus counts to each ``open_buy`` event.

    For each row, computes (at the moment of the trade, using only earlier rows):

    * ``prior_same_any``  — distinct wallets that ever bought the same outcome
    * ``prior_opp_any``   — distinct wallets that ever bought the opposite outcome
    * ``prior_same_24h``  — same, restricted to the last 24 h
    * ``prior_opp_24h``   — opposite, restricted to the last 24 h
    * ``consensus_velocity_24h = prior_same_24h / 24``

    The function processes rows in chronological order and is O(n) in the
    number of rows.

    Parameters
    ----------
    open_buys:
        DataFrame of ``open_buy`` events, must contain:
        ``dt``, ``condition_id``, ``outcome``, ``wallet``.

    Returns
    -------
    Copy of *open_buys* with the five new columns appended.
    """
    open_buys = open_buys.sort_values(
        ["dt", "wallet", "condition_id", "outcome"]
    ).reset_index(drop=True)

    seen: dict = defaultdict(lambda: defaultdict(set))
    recent: dict = defaultdict(lambda: defaultdict(deque))

    same_any = []
    opp_any = []
    same_24h = []
    opp_24h = []

    for row in open_buys[["condition_id", "outcome", "wallet", "dt"]].itertuples(
        index=False
    ):
        market_recent = recent[row.condition_id]
        cutoff = row.dt - pd.Timedelta(hours=24)

        # evict stale entries
        for outcome_name in list(market_recent.keys()):
            dq = market_recent[outcome_name]
            while dq and dq[0][0] < cutoff:
                dq.popleft()

        same_any_wallets = seen[row.condition_id][row.outcome] - {row.wallet}
        opp_any_wallets: set = set()
        for outcome_name, wallets in seen[row.condition_id].items():
            if outcome_name != row.outcome:
                opp_any_wallets |= wallets - {row.wallet}

        same_recent_wallets = {
            wallet for _, wallet in market_recent[row.outcome] if wallet != row.wallet
        }
        opp_recent_wallets: set = set()
        for outcome_name, dq in market_recent.items():
            if outcome_name != row.outcome:
                opp_recent_wallets |= {wallet for _, wallet in dq if wallet != row.wallet}

        same_any.append(len(same_any_wallets))
        opp_any.append(len(opp_any_wallets))
        same_24h.append(len(same_recent_wallets))
        opp_24h.append(len(opp_recent_wallets))

        seen[row.condition_id][row.outcome].add(row.wallet)
        market_recent[row.outcome].append((row.dt, row.wallet))

    out = open_buys.copy()
    out["prior_same_any"] = same_any
    out["prior_opp_any"] = opp_any
    out["prior_same_24h"] = same_24h
    out["prior_opp_24h"] = opp_24h
    out["consensus_velocity_24h"] = out["prior_same_24h"] / 24.0
    return out


# ---------------------------------------------------------------------------
# Signal event construction
# ---------------------------------------------------------------------------

def build_signal_events(
    dataset: Any,
    wallet_profiles: pd.DataFrame,
    *,
    end_date_train: datetime.date,
    train_a_end_date: datetime.date,
    period: str = "test",
    tx_hash_column: str | None = None,
    price_bucket_bins: list[float] | None = None,
    price_bucket_labels: list[str] | None = None,
    batch_size: int = 300_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build classified trade events and ``open_buy`` signal rows for a period.

    Parameters
    ----------
    dataset:
        PyArrow ``Dataset``.
    wallet_profiles:
        Output of :func:`build_wallet_profiles`.  Must contain
        ``wallet``, ``wallet_quality``, ``median_open_buy_usdc``.
    end_date_train, train_a_end_date:
        Date boundaries for period selection.
    period:
        One of ``'test'``, ``'train_b'``, ``'full_train'``.
    tx_hash_column:
        Column name in the dataset holding the transaction hash, or ``None``
        to skip hash tracking.
    price_bucket_bins, price_bucket_labels:
        Bin edges and labels for ``pd.cut``.  Defaults to
        :data:`DEFAULT_PRICE_BUCKET_BINS` / :data:`DEFAULT_PRICE_BUCKET_LABELS`.
    batch_size:
        Arrow batch size.

    Returns
    -------
    (events, open_buys)
        *events* — all classified events (open/add/close/reduce)
        *open_buys* — only ``open_buy`` events, enriched with profiles,
                       conviction, consensus, and probability edge
    """
    if price_bucket_bins is None:
        price_bucket_bins = DEFAULT_PRICE_BUCKET_BINS
    if price_bucket_labels is None:
        price_bucket_labels = DEFAULT_PRICE_BUCKET_LABELS

    selected_wallet_set = set(wallet_profiles["wallet"])
    chunks: list[pd.DataFrame] = []

    _schema_names = set(dataset.schema.names)
    if "total_quantity" in _schema_names:
        columns = [
            "wallet", "condition_id", "dt", "side", "outcome",
            "total_quantity", "avg_price", "trade_value_usdc",
            "trade_pnl", "token_winner", "final_price",
        ]
    else:
        columns = [
            "wallet", "condition_id", "dt", "side", "outcome",
            "quantity", "price", "usdc_amount",
            "trade_pnl", "token_winner", "final_price",
        ]
    if tx_hash_column is not None:
        columns.append(tx_hash_column)

    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        df: pd.DataFrame = batch.to_pandas()
        if df.empty:
            continue

        df["dt"] = pd.to_datetime(df["dt"], utc=True)

        if period == "train_b":
            df = df[
                (df["dt"].dt.date > train_a_end_date)
                & (df["dt"].dt.date <= end_date_train)
            ]
        elif period == "test":
            df = df[df["dt"].dt.date > end_date_train]
        elif period == "full_train":
            df = df[df["dt"].dt.date <= end_date_train]

        df = df[df["wallet"].isin(selected_wallet_set)].copy()
        if df.empty:
            continue

        if tx_hash_column is not None and tx_hash_column in df.columns:
            df["trigger_tx_hash"] = df[tx_hash_column].astype(str)

        df = aggregate_wallet_trades(df)
        df["position_change"] = df["signed_quantity"]
        df["market_key"] = df["condition_id"] + "|" + df["outcome"]
        df["event_type"] = np.select(
            [
                (df["side"] == "BUY") & (df["prev_position"] <= 1e-9),
                (df["side"] == "BUY") & (df["prev_position"] > 1e-9) & (df["position_change"] > 1e-9),
                (df["side"] == "SELL") & (df["position"] <= 1e-9) & (df["prev_position"] > 1e-9),
                (df["side"] == "SELL") & (df["position_change"] < -1e-9),
            ],
            ["open_buy", "add_buy", "close_sell", "reduce_sell"],
            default="other",
        )
        chunks.append(df)

    if not chunks:
        return pd.DataFrame(), pd.DataFrame()

    events = (
        pd.concat(chunks, ignore_index=True)
        .sort_values(["dt", "wallet", "condition_id", "outcome"])
        .reset_index(drop=True)
    )

    market_first_trade = (
        events.groupby("condition_id")["dt"]
        .min()
        .rename("first_selected_trade_dt")
    )

    open_buys = events[events["event_type"] == "open_buy"].copy()
    open_buys = open_buys.merge(wallet_profiles, on="wallet", how="left")
    open_buys["conviction_ratio"] = (
        open_buys["usdc_amount"]
        / open_buys["median_open_buy_usdc"].replace({0.0: np.nan})
    )
    open_buys["conviction_ratio"] = (
        open_buys["conviction_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )
    open_buys = open_buys.merge(market_first_trade, on="condition_id", how="left")
    open_buys["hours_since_first_selected_trade"] = (
        (open_buys["dt"] - open_buys["first_selected_trade_dt"])
        .dt.total_seconds()
        / 3600.0
    ).clip(lower=0.0)

    open_buys["prob_edge"] = open_buys["final_price"] - open_buys["price"]
    open_buys["copy_roi"] = np.where(
        open_buys["token_winner"],
        1.0 / open_buys["price"].clip(lower=0.001) - 1.0,
        -1.0,
    )
    open_buys["copy_roi_capped"] = np.clip(open_buys["copy_roi"], -1.0, 10.0)
    open_buys["price_bucket"] = pd.cut(
        open_buys["price"],
        bins=price_bucket_bins,
        labels=price_bucket_labels,
        include_lowest=True,
    ).astype(str)

    open_buys = attach_consensus_features(open_buys)
    return events, open_buys


# ---------------------------------------------------------------------------
# Partial-fill grouping diagnostics
# ---------------------------------------------------------------------------

def verify_partial_fill_grouping(
    dataset: Any, batch_size: int = 300_000
) -> pd.DataFrame:
    """Sanity-check that partial-fill aggregation produces the expected open-buy counts.

    Reads the first non-empty batch from *dataset* and compares raw vs grouped
    event counts.

    Returns a small diagnostic DataFrame.
    """
    _schema_names = set(dataset.schema.names)
    _is_grouped_schema = "total_quantity" in _schema_names
    if _is_grouped_schema:
        columns = [
            "wallet", "condition_id", "outcome", "dt", "side",
            "total_quantity", "avg_price", "trade_value_usdc",
        ]
    else:
        columns = [
            "wallet", "condition_id", "outcome", "dt", "side",
            "quantity", "price", "usdc_amount", "position",
        ]
    sample = None
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        df: pd.DataFrame = batch.to_pandas()
        if not df.empty:
            sample = df
            break

    if sample is None or sample.empty:
        return pd.DataFrame({"metric": ["note"], "value": ["no rows available"]})

    sample["dt"] = pd.to_datetime(sample["dt"], utc=True)
    grouped = aggregate_wallet_trades(sample)

    keys = ["wallet", "condition_id", "outcome", "dt", "side"]
    raw_dupe_rows = int(sample.duplicated(keys, keep=False).sum())
    grouped_dupe_rows = int(grouped.duplicated(keys, keep=False).sum())

    if _is_grouped_schema:
        # Dataset is already grouped; raw == grouped, no fill-level position available
        raw_open_buys = int(
            ((grouped["side"] == "BUY") & (grouped["prev_position"] <= 1e-9)).sum()
        )
    else:
        raw_signed_qty = np.where(
            sample["side"] == "BUY", sample["quantity"], -sample["quantity"]
        )
        raw_prev_pos = sample["position"] - raw_signed_qty
        raw_open_buys = int(
            ((sample["side"] == "BUY") & (raw_prev_pos <= 1e-9)).sum()
        )
    grouped_open_buys = int(
        ((grouped["side"] == "BUY") & (grouped["prev_position"] <= 1e-9)).sum()
    )

    return pd.DataFrame(
        [
            {"metric": "raw_rows", "value": int(len(sample))},
            {"metric": "grouped_rows", "value": int(len(grouped))},
            {"metric": "raw_duplicate_key_rows", "value": raw_dupe_rows},
            {"metric": "grouped_duplicate_key_rows", "value": grouped_dupe_rows},
            {"metric": "raw_open_buy_events", "value": raw_open_buys},
            {"metric": "grouped_open_buy_events", "value": grouped_open_buys},
        ]
    )
