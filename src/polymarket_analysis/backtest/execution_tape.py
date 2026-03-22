"""
Execution tape construction and forward-fill simulation for copy-trade backtesting.

The execution tape represents observable market liquidity after a trigger event.
It is built from the raw trade dataset and used to simulate fills for copy-trades.

Key design constraints
----------------------
- A BUY copy-trade triggered at price ``p`` should only fill at tape prices ≤ ``p``.
  The trigger wallet printed a price; we wouldn't accept a worse (higher) price.
  Slippage is applied on top of the raw tape price but the result is still capped at
  the trigger price + a small absolute tolerance (``max_price_premium``).
- ``fill_tx_hash`` tracks which tape transaction provided each fill.
- The tape may be built from cached parquet files that pre-date ``fill_tx_hash``
  support; ``normalize_execution_tape`` adds the column as ``None`` when absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Tape construction
# ---------------------------------------------------------------------------

def build_execution_tape(
    dataset: Any,
    condition_ids: Iterable[str],
    *,
    tx_hash_column: str | None = "tx_hash",
    batch_size: int = 300_000,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Build an execution tape from a PyArrow dataset.

    The tape contains one row per (market_key, tape_dt, exec_price_raw,
    exec_source) combination and represents available liquidity that a
    copy-trade could consume.

    Two sources of liquidity are modelled:

    * ``same_token`` — BUY orders on the same token as the trigger.
    * ``opposite_token`` — SELL orders on the complementary token, converted
      via ``exec_price_raw = 1 - sell_price``.

    Parameters
    ----------
    dataset:
        A PyArrow ``Dataset`` (or any object with ``.to_batches``).
    condition_ids:
        The set of condition IDs to include.
    tx_hash_column:
        Column name in the dataset that holds the transaction hash.
        Pass ``None`` to omit hashes entirely.
    batch_size:
        Arrow batch size for streaming reads.
    start_date, end_date:
        Inclusive date range filter (``datetime.date`` objects or ``None``).
        If ``start_date`` is ``None`` the tape includes all dates.

    Returns
    -------
    pd.DataFrame with columns:
        market_key, tape_dt, exec_price_raw, exec_source,
        available_qty, available_usdc[, fill_tx_hash]
    """
    condition_ids = set(condition_ids)

    # Support both the old raw-fill schema (price, quantity) and the grouped
    # stage0 schema (avg_price, total_quantity).  Detect which is present and
    # normalise to price/quantity inside each batch.
    schema_names = set(dataset.schema.names)
    if "price" in schema_names:
        price_col, qty_col = "price", "quantity"
    else:
        price_col, qty_col = "avg_price", "total_quantity"

    columns = ["condition_id", "outcome", "dt", price_col, qty_col, "side"]
    if tx_hash_column is not None and tx_hash_column in schema_names:
        columns.append(tx_hash_column)

    chunks: list[pd.DataFrame] = []
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        df: pd.DataFrame = batch.to_pandas()
        if df.empty:
            continue

        # Normalise column names to price / quantity
        if price_col != "price":
            df = df.rename(columns={price_col: "price", qty_col: "quantity"})

        df["dt"] = pd.to_datetime(df["dt"], utc=True)
        date_mask = pd.Series(True, index=df.index)
        if start_date is not None:
            date_mask &= df["dt"].dt.date >= start_date
        if end_date is not None:
            date_mask &= df["dt"].dt.date <= end_date

        df = df[date_mask & df["condition_id"].isin(condition_ids)]
        if df.empty:
            continue

        df["quantity"] = df["quantity"].astype(float)
        df["price"] = df["price"].astype(float)
        df["price_x_qty"] = df["price"] * df["quantity"]

        group_cols = ["condition_id", "outcome", "dt", "side"]
        if tx_hash_column is not None and tx_hash_column in df.columns:
            df["fill_tx_hash"] = df[tx_hash_column].astype(str)
            group_cols.append("fill_tx_hash")

        grouped = (
            df.groupby(group_cols, as_index=False)
            .agg(quantity=("quantity", "sum"), price_x_qty=("price_x_qty", "sum"))
        )
        grouped["price"] = grouped["price_x_qty"] / grouped["quantity"].clip(lower=1e-9)

        keep_cols = ["condition_id", "outcome", "dt", "price", "quantity", "side"]
        if "fill_tx_hash" in grouped.columns:
            keep_cols.append("fill_tx_hash")
        chunks.append(grouped[keep_cols])

    if not chunks:
        return pd.DataFrame(columns=[
            "market_key", "tape_dt", "exec_price_raw", "exec_source",
            "available_qty", "available_usdc",
        ])

    raw_tape = pd.concat(chunks, ignore_index=True).rename(columns={"dt": "tape_dt"})
    raw_tape["market_key"] = raw_tape["condition_id"] + "|" + raw_tape["outcome"]

    # Build outcome complement map for opposite-token liquidity
    outcome_map_rows: list[dict] = []
    for condition_id, outcomes in raw_tape.groupby("condition_id")["outcome"].unique().items():
        outcomes = sorted(outcomes.tolist())
        if len(outcomes) == 2:
            outcome_map_rows.append({
                "condition_id": condition_id,
                "outcome": outcomes[0],
                "opposite_outcome": outcomes[1],
            })
            outcome_map_rows.append({
                "condition_id": condition_id,
                "outcome": outcomes[1],
                "opposite_outcome": outcomes[0],
            })

    hash_cols = ["fill_tx_hash"] if "fill_tx_hash" in raw_tape.columns else []

    # --- same-token liquidity (BUY orders on the same token) ---
    same_side_cols = ["market_key", "tape_dt", "price", "quantity"] + hash_cols
    same_side = raw_tape[raw_tape["side"] == "BUY"][same_side_cols].copy()
    same_side = same_side.rename(columns={"price": "exec_price_raw"})
    same_side["exec_source"] = "same_token"
    same_side["available_qty"] = same_side["quantity"].astype(float)
    same_side["available_usdc"] = same_side["available_qty"] * same_side["exec_price_raw"]
    keep = ["market_key", "tape_dt", "exec_price_raw", "exec_source",
            "available_qty", "available_usdc"] + hash_cols
    same_side = same_side[keep]

    # --- opposite-token liquidity (SELL orders on complementary token) ---
    if outcome_map_rows:
        outcome_map = pd.DataFrame(outcome_map_rows)
        opp = raw_tape[raw_tape["side"] == "SELL"].merge(
            outcome_map, on=["condition_id", "outcome"], how="inner"
        )
        opp["market_key"] = opp["condition_id"] + "|" + opp["opposite_outcome"]
        opp["exec_price_raw"] = 1.0 - opp["price"]
        opp["exec_source"] = "opposite_token"
        opp["available_qty"] = opp["quantity"].astype(float)
        opp["available_usdc"] = opp["available_qty"] * opp["exec_price_raw"]
        opposite_side = opp[keep]
    else:
        opposite_side = pd.DataFrame(columns=keep)

    parts = [df for df in [same_side, opposite_side] if not df.empty]
    execution_tape = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=keep)

    group_cols = ["market_key", "tape_dt", "exec_price_raw", "exec_source"]
    if hash_cols:
        group_cols.extend(hash_cols)

    execution_tape = (
        execution_tape
        .groupby(group_cols, as_index=False)
        .agg(available_qty=("available_qty", "sum"), available_usdc=("available_usdc", "sum"))
        .sort_values(["market_key", "tape_dt"])
        .reset_index(drop=True)
    )
    return execution_tape


def normalize_execution_tape(execution_tape: pd.DataFrame) -> pd.DataFrame:
    """Ensure the tape has all expected columns, adding defaults where absent.

    This handles tapes loaded from parquet caches built before ``fill_tx_hash``
    was added to the schema.
    """
    tape = execution_tape.copy()

    if "exec_price_raw" not in tape.columns:
        if "price" in tape.columns:
            tape["exec_price_raw"] = tape["price"].astype(float)
        else:
            raise ValueError("Execution tape missing exec_price_raw and price")

    if "exec_source" not in tape.columns:
        tape["exec_source"] = "same_token"

    if "available_qty" not in tape.columns:
        if "quantity" in tape.columns:
            tape["available_qty"] = tape["quantity"].astype(float)
        elif "available_usdc" in tape.columns:
            tape["available_qty"] = (
                tape["available_usdc"].astype(float)
                / tape["exec_price_raw"].clip(lower=1e-9)
            )
        else:
            raise ValueError("Execution tape missing available_qty")

    if "fill_tx_hash" not in tape.columns:
        tape["fill_tx_hash"] = None

    return tape


def build_tape_groups(market_tape: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Index tape by market_key for O(1) lookup during fill simulation."""
    return {
        k: g.reset_index(drop=True)
        for k, g in market_tape.groupby("market_key", sort=False)
    }


# ---------------------------------------------------------------------------
# Forward-fill simulation
# ---------------------------------------------------------------------------

def attach_forward_fills(
    trades: pd.DataFrame,
    tape_groups: dict[str, pd.DataFrame],
    *,
    latency_seconds: float = 0,
    fill_horizon_seconds: float = 600,
    slippage_bps: float = 50.0,
    min_fill_ratio: float = 1.0,
    max_price_premium: float = 0.0,
    max_rel_price_diff_by_bucket: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Simulate copy-trade fills against a forward execution tape.

    For each trigger trade the function scans the tape in the window
    ``[dt + latency, dt + latency + fill_horizon]`` and greedily consumes
    available liquidity subject to two price guards:

    1. **Trigger-price cap** (``max_price_premium``): the simulated exec price
       after slippage must not exceed ``trigger_price * (1 + max_price_premium)``.
       Default is ``0.0``, meaning fills are capped at exactly the trigger price
       (plus slippage friction).  Set to a small positive value (e.g. ``0.05``)
       to allow moderate adverse moves while still bounding exposure.

    2. **Bucket-level relative cap** (``max_rel_price_diff_by_bucket``): a
       per-price-bucket override of the relative price difference limit.  When
       both guards are active the stricter one wins.

    Parameters
    ----------
    trades:
        Signal rows.  Must have columns: ``market_key``, ``dt``,
        ``stake_usdc``, ``price``.  Optional: ``price_bucket``.
    tape_groups:
        Output of :func:`build_tape_groups`.
    latency_seconds:
        Seconds added to ``dt`` before searching the tape.
    fill_horizon_seconds:
        Maximum seconds after ``dt + latency`` to search.
    slippage_bps:
        Execution slippage in basis points applied on top of raw tape price.
    min_fill_ratio:
        Minimum ``filled_usdc / stake_usdc`` ratio to accept a fill (default
        ``1.0`` = full fill required).
    max_price_premium:
        Maximum allowed ratio by which the slippage-adjusted exec price may
        exceed the trigger price.  ``0.0`` means exec_price ≤ trigger_price.
        ``0.05`` allows up to 5 % above.
    max_rel_price_diff_by_bucket:
        Optional dict mapping ``price_bucket`` label → maximum relative price
        difference ``|exec_price - trigger_price| / trigger_price``.  When
        ``None`` no bucket-level cap is applied beyond ``max_price_premium``.

    Returns
    -------
    pd.DataFrame
        Subset of ``trades`` that were filled, with additional columns:
        ``filled_usdc``, ``filled_qty``, ``exec_price``, ``exec_price_raw``,
        ``fill_ratio``, ``tape_dt``, ``fill_tx_hash``,
        ``same_fill_share``, ``opposite_fill_share``, ``exec_source``.
    """
    if max_rel_price_diff_by_bucket is None:
        max_rel_price_diff_by_bucket = {}

    trades = trades.copy().sort_values(["market_key", "dt"]).reset_index(drop=True)
    trades["fill_search_dt"] = trades["dt"] + pd.to_timedelta(latency_seconds, unit="s")
    trades["fill_deadline_dt"] = trades["fill_search_dt"] + pd.to_timedelta(
        fill_horizon_seconds, unit="s"
    )
    slippage = slippage_bps / 10_000.0

    filled_parts: list[pd.DataFrame] = []

    for market_key, group in trades.groupby("market_key", sort=False):
        tape = tape_groups.get(market_key)
        if tape is None or tape.empty:
            continue

        tape = tape.sort_values("tape_dt").reset_index(drop=True).copy()
        tape_times = tape["tape_dt"].to_numpy()
        tape_prices = tape["exec_price_raw"].to_numpy(dtype=float)
        tape_qty_remaining = tape["available_qty"].to_numpy(dtype=float).copy()
        tape_source = tape["exec_source"].to_numpy()
        tape_fill_hash: np.ndarray = (
            tape["fill_tx_hash"].to_numpy()
            if "fill_tx_hash" in tape.columns
            else np.full(len(tape), None, dtype=object)
        )

        for row in group.itertuples(index=False):
            start_idx = int(tape_times.searchsorted(row.fill_search_dt, side="right"))
            if start_idx >= len(tape):
                continue

            trigger_price = float(getattr(row, "price", np.nan))
            # Absolute price cap: exec_price ≤ trigger_price * (1 + max_price_premium)
            abs_price_cap = (
                trigger_price * (1.0 + max_price_premium)
                if trigger_price > 1e-9
                else np.inf
            )
            # Per-bucket relative cap (overrides abs_price_cap when stricter)
            bucket_limit = max_rel_price_diff_by_bucket.get(
                getattr(row, "price_bucket", None), np.inf
            )

            remaining_usdc = float(row.stake_usdc)
            filled_usdc = 0.0
            filled_qty = 0.0
            same_qty = 0.0
            opp_qty = 0.0
            fill_dt = None
            fill_tx_hash = None

            j = start_idx
            while (
                j < len(tape)
                and tape_times[j] <= row.fill_deadline_dt
                and remaining_usdc > 1e-9
            ):
                avail_qty = tape_qty_remaining[j]
                if avail_qty <= 1e-12:
                    j += 1
                    continue

                exec_price = float(
                    np.clip(tape_prices[j] * (1.0 + slippage), 0.001, 0.999)
                )

                # Guard 1: trigger-price cap
                if exec_price > abs_price_cap:
                    j += 1
                    continue

                # Guard 2: bucket-level relative cap
                if trigger_price > 1e-9:
                    rel_diff = abs(exec_price - trigger_price) / trigger_price
                    if rel_diff > bucket_limit:
                        j += 1
                        continue

                max_usdc_here = avail_qty * exec_price
                take_usdc = min(remaining_usdc, max_usdc_here)
                take_qty = take_usdc / exec_price
                tape_qty_remaining[j] -= take_qty
                remaining_usdc -= take_usdc
                filled_usdc += take_usdc
                filled_qty += take_qty
                fill_dt = tape_times[j]
                fill_tx_hash = tape_fill_hash[j]
                if tape_source[j] == "same_token":
                    same_qty += take_qty
                else:
                    opp_qty += take_qty
                j += 1

            fill_ratio = (
                filled_usdc / float(row.stake_usdc) if row.stake_usdc > 0 else 0.0
            )
            if filled_usdc <= 0 or fill_ratio + 1e-12 < min_fill_ratio:
                continue

            out = pd.DataFrame([row._asdict()])
            out["filled_usdc"] = filled_usdc
            out["filled_qty"] = filled_qty
            out["exec_price"] = filled_usdc / filled_qty
            out["exec_price_raw"] = out["exec_price"]
            out["fill_ratio"] = fill_ratio
            out["tape_dt"] = fill_dt
            out["fill_tx_hash"] = fill_tx_hash
            out["same_fill_share"] = same_qty / filled_qty if filled_qty > 0 else np.nan
            out["opposite_fill_share"] = (
                opp_qty / filled_qty if filled_qty > 0 else np.nan
            )
            out["exec_source"] = (
                "same_token"
                if opp_qty <= 1e-12
                else ("opposite_token" if same_qty <= 1e-12 else "mixed")
            )
            filled_parts.append(out)

    if not filled_parts:
        return trades.iloc[0:0].copy()

    return (
        pd.concat(filled_parts, ignore_index=True)
        .sort_values("dt")
        .reset_index(drop=True)
    )
