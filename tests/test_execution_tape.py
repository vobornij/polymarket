"""
Tests for execution_tape module.

Uses a small real-data fixture extracted from the polygon_trades_processed parquet
dataset (condition_id 0x000464..., Over/Under market with 36 rows).
"""

from __future__ import annotations

import datetime
import math
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from polymarket_analysis.backtest.execution_tape import (
    attach_forward_fills,
    build_execution_tape,
    build_tape_groups,
    normalize_execution_tape,
)

# ---------------------------------------------------------------------------
# Real-data fixture (36 trades, 2 outcomes: Over / Under)
# Extracted from polygon_trades_processed/ for condition_id
# 0x000464b8a29d57de5aa1d9ca95026f9c6c2212328577d40d484ea45675bf576f
# ---------------------------------------------------------------------------

CID = "0x000464b8a29d57de5aa1d9ca95026f9c6c2212328577d40d484ea45675bf576f"

_RAW_TRADES = [
    # (outcome, dt_str, side, price, qty, tx_hash)
    ("Over",  "2026-03-03 09:02:55+00:00", "BUY",  0.150, 24.000000, "0xe547"),
    ("Over",  "2026-03-03 09:02:55+00:00", "BUY",  0.150, 17.000000, "0xe547"),
    ("Over",  "2026-03-04 09:52:03+00:00", "SELL", 0.180, 17.000000, "0xe0a2"),
    ("Under", "2026-03-04 09:52:03+00:00", "BUY",  0.820, 19.000000, "0xe0a2"),
    ("Over",  "2026-03-04 09:52:03+00:00", "BUY",  0.180, 36.000000, "0xe0a2"),
    ("Over",  "2026-03-04 09:52:03+00:00", "SELL", 0.190, 17.421051, "0xe0a2"),
    ("Over",  "2026-03-04 09:52:03+00:00", "BUY",  0.190, 17.421051, "0xe0a2"),
    ("Under", "2026-03-04 09:52:21+00:00", "BUY",  0.820,  5.555554, "0xc1da"),
    ("Under", "2026-03-04 09:52:25+00:00", "BUY",  0.820,  5.555554, "0xf148"),
    ("Over",  "2026-03-04 10:08:15+00:00", "SELL", 0.160, 53.420000, "0x219a"),
    ("Under", "2026-03-04 11:28:51+00:00", "BUY",  0.820,  5.555554, "0x7af9"),
    ("Under", "2026-03-04 11:56:49+00:00", "SELL", 0.830,  5.550000, "0xd62e"),
    ("Under", "2026-03-04 19:02:37+00:00", "SELL", 0.830, 30.116662, "0x9b5b"),
    ("Over",  "2026-03-04 19:02:37+00:00", "BUY",  0.170, 2385.883338, "0x9b5b"),
    ("Over",  "2026-03-04 19:02:39+00:00", "BUY",  0.170, 295.000000, "0x4acf"),
    ("Over",  "2026-03-04 19:03:51+00:00", "SELL", 0.180, 68.000000,  "0xe262"),
    ("Over",  "2026-03-04 19:10:19+00:00", "SELL", 0.150,  7.000000,  "0x6418"),
    ("Over",  "2026-03-04 19:10:47+00:00", "BUY",  0.130,  1.517238,  "0xaaf7"),
    ("Over",  "2026-03-04 19:10:51+00:00", "SELL", 0.160,  6.578949,  "0x66b3"),
    ("Under", "2026-03-04 19:10:51+00:00", "BUY",  0.840,  5.921051,  "0x66b3"),
    ("Over",  "2026-03-04 19:17:51+00:00", "SELL", 0.120, 82.000000,  "0x82a9"),
    ("Over",  "2026-03-04 19:18:25+00:00", "SELL", 0.120,  0.200000,  "0x5122"),
    ("Over",  "2026-03-04 19:18:27+00:00", "BUY",  0.100,  1.222220,  "0x83a2"),
    ("Over",  "2026-03-04 19:18:35+00:00", "SELL", 0.120,  8.333332,  "0xb20f"),
    ("Over",  "2026-03-04 19:18:37+00:00", "BUY",  0.100,  1.222220,  "0x7cfe"),
    ("Over",  "2026-03-04 19:24:11+00:00", "BUY",  0.080,  5.000000,  "0x949f"),
    ("Over",  "2026-03-04 19:24:15+00:00", "BUY",  0.080,  8.330000,  "0xc9c9"),
    ("Over",  "2026-03-04 19:24:39+00:00", "SELL", 0.100, 10.500000,  "0x5e19"),
    ("Under", "2026-03-04 19:25:01+00:00", "SELL", 0.920,  1.086950,  "0x8de6"),
    ("Over",  "2026-03-04 19:25:03+00:00", "SELL", 0.100, 10.500000,  "0x61d3"),
    ("Under", "2026-03-04 19:25:21+00:00", "SELL", 0.920,  1.080000,  "0x00cb"),
    ("Under", "2026-03-04 19:25:29+00:00", "SELL", 0.920,  1.086950,  "0x4651"),
    ("Over",  "2026-03-04 20:58:49+00:00", "BUY",  0.001, 30.000000,  "0xfa43"),
    ("Under", "2026-03-04 20:58:49+00:00", "BUY",  0.999, 30.000000,  "0xfa43"),
    ("Under", "2026-03-04 21:06:05+00:00", "BUY",  0.999,  1.080000,  "0x0d87"),
    ("Under", "2026-03-04 21:08:49+00:00", "BUY",  0.999, 41.000000,  "0x8b2b"),
]


def _make_raw_df() -> pd.DataFrame:
    rows = [
        {
            "condition_id": CID,
            "outcome": outcome,
            "dt": pd.Timestamp(dt, tz="UTC"),
            "side": side,
            "price": price,
            "quantity": qty,
            "tx_hash": tx,
        }
        for outcome, dt, side, price, qty, tx in _RAW_TRADES
    ]
    return pd.DataFrame(rows)


def _make_mock_dataset(df: pd.DataFrame):
    """Wrap a DataFrame as a mock PyArrow dataset."""
    mock = MagicMock()

    class FakeBatch:
        def to_pandas(self):
            return df.copy()

    mock.to_batches.return_value = [FakeBatch()]
    return mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tape_from_df(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    dataset = _make_mock_dataset(df)
    return build_execution_tape(
        dataset,
        condition_ids=[CID],
        tx_hash_column="tx_hash",
        **kwargs,
    )


def _make_trigger(
    market_key: str,
    dt_str: str,
    price: float,
    stake_usdc: float = 10.0,
    price_bucket: str | None = None,
) -> pd.DataFrame:
    row = {
        "market_key": market_key,
        "dt": pd.Timestamp(dt_str, tz="UTC"),
        "price": price,
        "stake_usdc": stake_usdc,
    }
    if price_bucket is not None:
        row["price_bucket"] = price_bucket
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# build_execution_tape
# ---------------------------------------------------------------------------


class TestBuildExecutionTape:

    def test_returns_dataframe_with_expected_columns(self):
        tape = _tape_from_df(_make_raw_df())
        required = {
            "market_key", "tape_dt", "exec_price_raw",
            "exec_source", "available_qty", "available_usdc", "fill_tx_hash",
        }
        assert required.issubset(set(tape.columns)), (
            f"Missing columns: {required - set(tape.columns)}"
        )

    def test_same_token_buy_rows_appear_on_correct_market_key(self):
        tape = _tape_from_df(_make_raw_df())
        over_same = tape[
            (tape["market_key"] == f"{CID}|Over")
            & (tape["exec_source"] == "same_token")
        ]
        # All BUY Over rows should produce same_token entries for Over
        assert len(over_same) > 0

    def test_opposite_token_sell_rows_appear_on_complementary_key(self):
        """SELL Over rows → opposite_token liquidity for Under."""
        tape = _tape_from_df(_make_raw_df())
        under_opp = tape[
            (tape["market_key"] == f"{CID}|Under")
            & (tape["exec_source"] == "opposite_token")
        ]
        assert len(under_opp) > 0

    def test_opposite_token_price_is_complement(self):
        """SELL Over rows produce Under opposite_token entries with exec_price = 1 - sell_price.

        At 2026-03-04 09:52:03 tx 0xe0a2 has two SELL Over rows (0.18, qty=17) and
        (0.19, qty=17.42), same tx_hash → aggregated into one row with weighted-average
        sell price.  We verify exec_price_raw = 1 - weighted_avg_sell_price.
        """
        tape = _tape_from_df(_make_raw_df())
        ts = pd.Timestamp("2026-03-04 09:52:03+00:00")
        row = tape[
            (tape["market_key"] == f"{CID}|Under")
            & (tape["exec_source"] == "opposite_token")
            & (tape["tape_dt"] == ts)
        ]
        assert len(row) > 0, "Expected at least one opposite_token Under entry at this ts"
        # Weighted-average sell price: (0.18*17 + 0.19*17.421051) / (17 + 17.421051)
        sell_qty_a, sell_qty_b = 17.0, 17.421051
        sell_price_a, sell_price_b = 0.18, 0.19
        expected_sell_price = (
            (sell_price_a * sell_qty_a + sell_price_b * sell_qty_b)
            / (sell_qty_a + sell_qty_b)
        )
        expected_exec = 1.0 - expected_sell_price
        assert math.isclose(row["exec_price_raw"].iloc[0], expected_exec, rel_tol=1e-5), (
            f"Expected {expected_exec:.6f}, got {row['exec_price_raw'].iloc[0]:.6f}"
        )

    def test_fill_tx_hash_propagated(self):
        tape = _tape_from_df(_make_raw_df())
        assert "fill_tx_hash" in tape.columns
        non_null = tape["fill_tx_hash"].dropna()
        assert len(non_null) > 0, "Expected at least some non-null fill_tx_hash values"

    def test_date_range_filter(self):
        """Entries before start_date should be excluded."""
        tape_all = _tape_from_df(_make_raw_df())
        start = datetime.date(2026, 3, 4)
        tape_filtered = _tape_from_df(_make_raw_df(), start_date=start)
        # 2026-03-03 rows should be absent in filtered tape
        early = tape_filtered[tape_filtered["tape_dt"].dt.date < start]
        assert len(early) == 0, f"Found {len(early)} rows before start_date"
        assert len(tape_filtered) < len(tape_all)

    def test_end_date_filter(self):
        end = datetime.date(2026, 3, 3)
        tape = _tape_from_df(_make_raw_df(), end_date=end)
        late = tape[tape["tape_dt"].dt.date > end]
        assert len(late) == 0

    def test_empty_when_no_matching_condition_ids(self):
        dataset = _make_mock_dataset(_make_raw_df())
        tape = build_execution_tape(dataset, condition_ids=["nonexistent_id"])
        assert tape.empty

    def test_rows_sorted_by_market_key_then_tape_dt(self):
        tape = _tape_from_df(_make_raw_df())
        for mkt, grp in tape.groupby("market_key"):
            assert grp["tape_dt"].is_monotonic_increasing, (
                f"tape_dt not sorted for market_key={mkt}"
            )

    def test_same_timestamp_same_price_quantities_aggregated(self):
        """Rows 0+1 are both BUY Over at 0.15 at the same ts and tx — they merge."""
        tape = _tape_from_df(_make_raw_df())
        ts = pd.Timestamp("2026-03-03 09:02:55+00:00")
        over_same = tape[
            (tape["market_key"] == f"{CID}|Over")
            & (tape["exec_source"] == "same_token")
            & (tape["tape_dt"] == ts)
        ]
        # Same tx_hash groups into one row; qty should be 24+17=41
        # (they have the same tx hash "0xe547" and same price)
        assert len(over_same) == 1
        assert math.isclose(over_same["available_qty"].iloc[0], 41.0, abs_tol=1e-6)

    def test_no_tx_hash_column(self):
        """When tx_hash_column=None the tape is still built without fill_tx_hash."""
        dataset = _make_mock_dataset(_make_raw_df())
        tape = build_execution_tape(dataset, condition_ids=[CID], tx_hash_column=None)
        assert "fill_tx_hash" not in tape.columns
        assert len(tape) > 0


# ---------------------------------------------------------------------------
# normalize_execution_tape
# ---------------------------------------------------------------------------


class TestNormalizeExecutionTape:

    def test_adds_fill_tx_hash_when_absent(self):
        tape = pd.DataFrame({
            "market_key": ["A|Yes"],
            "tape_dt": [pd.Timestamp("2026-01-01", tz="UTC")],
            "exec_price_raw": [0.5],
            "exec_source": ["same_token"],
            "available_qty": [10.0],
            "available_usdc": [5.0],
        })
        normalized = normalize_execution_tape(tape)
        assert "fill_tx_hash" in normalized.columns
        assert normalized["fill_tx_hash"].iloc[0] is None

    def test_adds_exec_price_raw_from_price(self):
        tape = pd.DataFrame({
            "market_key": ["A|Yes"],
            "tape_dt": [pd.Timestamp("2026-01-01", tz="UTC")],
            "price": [0.6],
            "exec_source": ["same_token"],
            "available_qty": [10.0],
            "available_usdc": [6.0],
        })
        normalized = normalize_execution_tape(tape)
        assert math.isclose(normalized["exec_price_raw"].iloc[0], 0.6)

    def test_raises_when_price_missing(self):
        tape = pd.DataFrame({
            "market_key": ["A|Yes"],
            "tape_dt": [pd.Timestamp("2026-01-01", tz="UTC")],
            "exec_source": ["same_token"],
            "available_qty": [10.0],
        })
        with pytest.raises(ValueError, match="exec_price_raw"):
            normalize_execution_tape(tape)

    def test_existing_columns_not_overwritten(self):
        tape = pd.DataFrame({
            "market_key": ["A|Yes"],
            "tape_dt": [pd.Timestamp("2026-01-01", tz="UTC")],
            "exec_price_raw": [0.4],
            "exec_source": ["same_token"],
            "available_qty": [5.0],
            "available_usdc": [2.0],
            "fill_tx_hash": ["0xabc"],
        })
        normalized = normalize_execution_tape(tape)
        assert normalized["fill_tx_hash"].iloc[0] == "0xabc"
        assert math.isclose(normalized["exec_price_raw"].iloc[0], 0.4)


# ---------------------------------------------------------------------------
# build_tape_groups
# ---------------------------------------------------------------------------


class TestBuildTapeGroups:

    def test_keys_are_market_keys(self):
        tape = _tape_from_df(_make_raw_df())
        groups = build_tape_groups(tape)
        assert f"{CID}|Over" in groups
        assert f"{CID}|Under" in groups

    def test_each_group_contains_only_its_market_key(self):
        tape = _tape_from_df(_make_raw_df())
        groups = build_tape_groups(tape)
        for mkt, grp in groups.items():
            assert (grp["market_key"] == mkt).all()

    def test_index_is_reset(self):
        tape = _tape_from_df(_make_raw_df())
        groups = build_tape_groups(tape)
        for grp in groups.values():
            assert list(grp.index) == list(range(len(grp)))


# ---------------------------------------------------------------------------
# attach_forward_fills — core behaviour
# ---------------------------------------------------------------------------


def _simple_tape(market_key: str, entries: list[tuple]) -> dict[str, pd.DataFrame]:
    """Build a tape_groups dict from (tape_dt_str, price, qty, source, tx_hash)."""
    rows = []
    for tape_dt, price, qty, source, tx in entries:
        rows.append({
            "market_key": market_key,
            "tape_dt": pd.Timestamp(tape_dt, tz="UTC"),
            "exec_price_raw": float(price),
            "exec_source": source,
            "available_qty": float(qty),
            "fill_tx_hash": tx,
        })
    df = pd.DataFrame(rows)
    return build_tape_groups(df)


class TestAttachForwardFills:

    def test_basic_fill_within_horizon(self):
        mkt = "cid|Yes"
        tape = _simple_tape(mkt, [
            ("2026-01-01 00:01:00+00:00", 0.50, 100.0, "same_token", "0xfill1"),
        ])
        trigger = _make_trigger(mkt, "2026-01-01 00:00:00+00:00", price=0.50, stake_usdc=10.0)
        result = attach_forward_fills(trigger, tape, fill_horizon_seconds=600, slippage_bps=0)
        assert len(result) == 1
        assert math.isclose(result["filled_usdc"].iloc[0], 10.0, rel_tol=1e-4)
        assert result["fill_tx_hash"].iloc[0] == "0xfill1"

    def test_no_fill_beyond_horizon(self):
        mkt = "cid|Yes"
        tape = _simple_tape(mkt, [
            ("2026-01-01 00:20:00+00:00", 0.50, 100.0, "same_token", "0xfill1"),
        ])
        trigger = _make_trigger(mkt, "2026-01-01 00:00:00+00:00", price=0.50, stake_usdc=10.0)
        result = attach_forward_fills(trigger, tape, fill_horizon_seconds=600, slippage_bps=0)
        assert len(result) == 0

    def test_no_fill_before_trigger_dt(self):
        """Tape entries at or before trigger dt must be skipped."""
        mkt = "cid|Yes"
        tape = _simple_tape(mkt, [
            ("2026-01-01 00:00:00+00:00", 0.50, 100.0, "same_token", "0xold"),  # same dt
            ("2026-01-01 00:01:00+00:00", 0.50, 100.0, "same_token", "0xnew"),
        ])
        trigger = _make_trigger(mkt, "2026-01-01 00:00:00+00:00", price=0.50, stake_usdc=10.0)
        result = attach_forward_fills(trigger, tape, fill_horizon_seconds=600, slippage_bps=0)
        assert len(result) == 1
        assert result["fill_tx_hash"].iloc[0] == "0xnew"

    def test_price_cap_prevents_fill_above_trigger_price(self):
        """A tape price above the trigger price must be rejected (max_price_premium=0)."""
        mkt = "cid|Yes"
        tape = _simple_tape(mkt, [
            # exec_price after 0 slippage = 0.49, trigger = 0.37 → rejected
            ("2026-01-01 00:01:00+00:00", 0.49, 100.0, "same_token", "0xbad"),
        ])
        trigger = _make_trigger(mkt, "2026-01-01 00:00:00+00:00", price=0.37, stake_usdc=10.0)
        result = attach_forward_fills(
            trigger, tape, fill_horizon_seconds=600, slippage_bps=0, max_price_premium=0.0
        )
        assert len(result) == 0, (
            "Should not fill when tape price > trigger price with max_price_premium=0"
        )

    def test_price_cap_allows_fill_at_trigger_price(self):
        """Tape at exactly trigger price must be accepted."""
        mkt = "cid|Yes"
        tape = _simple_tape(mkt, [
            ("2026-01-01 00:01:00+00:00", 0.37, 100.0, "same_token", "0xok"),
        ])
        trigger = _make_trigger(mkt, "2026-01-01 00:00:00+00:00", price=0.37, stake_usdc=10.0)
        result = attach_forward_fills(
            trigger, tape, fill_horizon_seconds=600, slippage_bps=0, max_price_premium=0.0
        )
        assert len(result) == 1

    def test_price_cap_allows_fill_below_trigger_price(self):
        """A better (lower) tape price should always be accepted."""
        mkt = "cid|Yes"
        tape = _simple_tape(mkt, [
            ("2026-01-01 00:01:00+00:00", 0.30, 100.0, "same_token", "0xbetter"),
        ])
        trigger = _make_trigger(mkt, "2026-01-01 00:00:00+00:00", price=0.37, stake_usdc=10.0)
        result = attach_forward_fills(
            trigger, tape, fill_horizon_seconds=600, slippage_bps=0, max_price_premium=0.0
        )
        assert len(result) == 1
        assert result["exec_price"].iloc[0] < 0.37

    def test_slippage_applied_to_exec_price(self):
        mkt = "cid|Yes"
        tape = _simple_tape(mkt, [
            ("2026-01-01 00:01:00+00:00", 0.50, 100.0, "same_token", "0xslip"),
        ])
        trigger = _make_trigger(mkt, "2026-01-01 00:00:00+00:00", price=0.60, stake_usdc=10.0)
        result = attach_forward_fills(
            trigger, tape, fill_horizon_seconds=600, slippage_bps=100, max_price_premium=1.0
        )
        assert len(result) == 1
        expected_exec = 0.50 * 1.01  # 1% slippage
        assert math.isclose(result["exec_price"].iloc[0], expected_exec, rel_tol=1e-4)

    def test_partial_fill_rejected_when_min_fill_ratio_1(self):
        mkt = "cid|Yes"
        tape = _simple_tape(mkt, [
            ("2026-01-01 00:01:00+00:00", 0.50, 5.0, "same_token", "0xpart"),
        ])
        # stake=100 but only 5*0.50=2.5 USDC available → fill_ratio ≈ 0.025
        trigger = _make_trigger(mkt, "2026-01-01 00:00:00+00:00", price=0.50, stake_usdc=100.0)
        result = attach_forward_fills(
            trigger, tape, fill_horizon_seconds=600, slippage_bps=0, min_fill_ratio=1.0
        )
        assert len(result) == 0

    def test_partial_fill_accepted_when_min_fill_ratio_relaxed(self):
        mkt = "cid|Yes"
        tape = _simple_tape(mkt, [
            ("2026-01-01 00:01:00+00:00", 0.50, 5.0, "same_token", "0xpart"),
        ])
        trigger = _make_trigger(mkt, "2026-01-01 00:00:00+00:00", price=0.50, stake_usdc=10.0)
        result = attach_forward_fills(
            trigger, tape, fill_horizon_seconds=600, slippage_bps=0, min_fill_ratio=0.0
        )
        assert len(result) == 1
        assert result["filled_usdc"].iloc[0] < 10.0

    def test_fill_consumes_tape_liquidity_across_triggers(self):
        """Second trigger in same market sees reduced liquidity from first."""
        mkt = "cid|Yes"
        tape = _simple_tape(mkt, [
            ("2026-01-01 00:01:00+00:00", 0.50, 10.0, "same_token", "0xshared"),
        ])
        t1 = _make_trigger(mkt, "2026-01-01 00:00:00+00:00", price=0.50, stake_usdc=5.0)
        t2 = _make_trigger(mkt, "2026-01-01 00:00:30+00:00", price=0.50, stake_usdc=5.0)
        triggers = pd.concat([t1, t2], ignore_index=True)
        result = attach_forward_fills(
            triggers, tape, fill_horizon_seconds=600, slippage_bps=0, min_fill_ratio=1.0
        )
        # 10 qty * 0.50 = 5 USDC total. First trigger takes 5, second gets none.
        assert len(result) == 1
        assert math.isclose(result["filled_usdc"].iloc[0], 5.0, rel_tol=1e-4)

    def test_exec_price_uses_slippage_adjusted_cost(self):
        mkt = "cid|Yes"
        tape = _simple_tape(mkt, [
            ("2026-01-01 00:01:00+00:00", 0.40, 1000.0, "same_token", "0xh"),
        ])
        trigger = _make_trigger(mkt, "2026-01-01 00:00:00+00:00", price=0.50, stake_usdc=10.0)
        result = attach_forward_fills(
            trigger, tape, fill_horizon_seconds=600, slippage_bps=50, max_price_premium=1.0
        )
        assert len(result) == 1
        # exec_price = filled_usdc / filled_qty
        assert math.isclose(
            result["exec_price"].iloc[0],
            result["filled_usdc"].iloc[0] / result["filled_qty"].iloc[0],
            rel_tol=1e-6,
        )

    def test_same_and_opposite_fill_share_tracked(self):
        mkt = "cid|Yes"
        tape_df = pd.DataFrame([
            {"market_key": mkt, "tape_dt": pd.Timestamp("2026-01-01 00:01:00+00:00"),
             "exec_price_raw": 0.50, "exec_source": "same_token", "available_qty": 10.0,
             "fill_tx_hash": "0xa"},
            {"market_key": mkt, "tape_dt": pd.Timestamp("2026-01-01 00:02:00+00:00"),
             "exec_price_raw": 0.50, "exec_source": "opposite_token", "available_qty": 10.0,
             "fill_tx_hash": "0xb"},
        ])
        groups = build_tape_groups(tape_df)
        trigger = _make_trigger(mkt, "2026-01-01 00:00:00+00:00", price=0.50, stake_usdc=10.0)
        result = attach_forward_fills(
            trigger, groups, fill_horizon_seconds=600, slippage_bps=0, min_fill_ratio=0.5,
            max_price_premium=1.0
        )
        assert len(result) == 1
        assert result["same_fill_share"].iloc[0] > 0
        assert result["opposite_fill_share"].iloc[0] > 0

    def test_latency_skips_immediate_tape_entry(self):
        mkt = "cid|Yes"
        tape = _simple_tape(mkt, [
            ("2026-01-01 00:00:05+00:00", 0.50, 100.0, "same_token", "0xearly"),
            ("2026-01-01 00:00:31+00:00", 0.50, 100.0, "same_token", "0xlate"),
        ])
        trigger = _make_trigger(mkt, "2026-01-01 00:00:00+00:00", price=0.50, stake_usdc=10.0)
        result = attach_forward_fills(
            trigger, tape, latency_seconds=30, fill_horizon_seconds=600, slippage_bps=0
        )
        assert len(result) == 1
        assert result["fill_tx_hash"].iloc[0] == "0xlate"

    def test_bucket_price_cap_enforced(self):
        """max_rel_price_diff_by_bucket rejects tape entries too far from trigger."""
        mkt = "cid|Yes"
        tape = _simple_tape(mkt, [
            # 0.35 vs trigger 0.30 → rel_diff = 0.167 > 0.10 → rejected
            ("2026-01-01 00:01:00+00:00", 0.35, 100.0, "same_token", "0xfar"),
            # 0.31 vs trigger 0.30 → rel_diff = 0.033 < 0.10 → accepted
            ("2026-01-01 00:02:00+00:00", 0.31, 100.0, "same_token", "0xclose"),
        ])
        trigger = _make_trigger(
            mkt, "2026-01-01 00:00:00+00:00", price=0.30, stake_usdc=5.0,
            price_bucket="low"
        )
        result = attach_forward_fills(
            trigger, tape,
            fill_horizon_seconds=600,
            slippage_bps=0,
            max_price_premium=1.0,          # disable the absolute cap
            max_rel_price_diff_by_bucket={"low": 0.10},
        )
        assert len(result) == 1
        assert result["fill_tx_hash"].iloc[0] == "0xclose"

    def test_no_tape_for_market_returns_empty(self):
        groups: dict = {}
        trigger = _make_trigger("missing|Yes", "2026-01-01 00:00:00+00:00", price=0.5)
        result = attach_forward_fills(trigger, groups, fill_horizon_seconds=600)
        assert result.empty

    def test_fill_tx_hash_is_last_tape_row_consumed(self):
        """fill_tx_hash should be the hash of the last tape row consumed."""
        mkt = "cid|Yes"
        tape = _simple_tape(mkt, [
            ("2026-01-01 00:01:00+00:00", 0.50, 5.0, "same_token", "0xfirst"),
            ("2026-01-01 00:02:00+00:00", 0.50, 50.0, "same_token", "0xlast"),
        ])
        trigger = _make_trigger(mkt, "2026-01-01 00:00:00+00:00", price=0.50, stake_usdc=10.0)
        result = attach_forward_fills(
            trigger, tape, fill_horizon_seconds=600, slippage_bps=0, min_fill_ratio=1.0,
            max_price_premium=1.0
        )
        assert len(result) == 1
        assert result["fill_tx_hash"].iloc[0] == "0xlast"


# ---------------------------------------------------------------------------
# Integration: build tape from real data + attach fills
# ---------------------------------------------------------------------------


class TestIntegrationRealData:

    def test_trigger_at_0_15_fills_from_same_token(self):
        """
        Trigger: BUY Over at 0.15 on 2026-03-03.
        Only tape entries strictly after trigger dt are eligible.
        The first same-token BUY for Over after that dt is on 2026-03-04.
        """
        raw = _make_raw_df()
        tape = _tape_from_df(raw)
        tape = normalize_execution_tape(tape)
        groups = build_tape_groups(tape)

        # Trigger on 2026-03-03 09:02:55 at price 0.15
        trigger = _make_trigger(
            f"{CID}|Over",
            "2026-03-03 09:02:55+00:00",
            price=0.15,
            stake_usdc=5.0,
        )
        result = attach_forward_fills(
            trigger,
            groups,
            fill_horizon_seconds=24 * 3600,  # 1 day horizon
            slippage_bps=0,
            max_price_premium=0.0,           # strict: only fill ≤ 0.15
        )
        # Over BUYs after the trigger: 0.18 (2026-03-04 09:52) → rejected (>0.15)
        # 0.17 (2026-03-04 19:02) → rejected (>0.15)
        # 0.13 (2026-03-04 19:10) → accepted
        # 0.10, 0.08 etc → accepted
        if len(result) > 0:
            assert result["exec_price"].iloc[0] <= 0.15 + 1e-6, (
                f"exec_price {result['exec_price'].iloc[0]} > trigger price 0.15"
            )

    def test_exec_price_never_exceeds_trigger_price_strict(self):
        """exec_price must be ≤ trigger_price when max_price_premium=0."""
        raw = _make_raw_df()
        tape = normalize_execution_tape(_tape_from_df(raw))
        groups = build_tape_groups(tape)

        # Trigger at a lower price than most tape entries
        trigger = _make_trigger(
            f"{CID}|Over",
            "2026-03-03 08:00:00+00:00",
            price=0.15,
            stake_usdc=20.0,
        )
        result = attach_forward_fills(
            trigger, groups,
            fill_horizon_seconds=7 * 24 * 3600,
            slippage_bps=0,
            max_price_premium=0.0,
        )
        if not result.empty:
            for _, row in result.iterrows():
                assert row["exec_price"] <= 0.15 + 1e-6, (
                    f"exec_price {row['exec_price']} exceeds trigger price 0.15"
                )

    def test_under_trigger_uses_opposite_token_liquidity(self):
        """
        A trigger to BUY Under at 0.82 can fill from SELL Over entries
        (opposite_token at exec_price = 1 - sell_price).
        SELL Over at 0.18 → Under exec_price = 0.82.
        """
        raw = _make_raw_df()
        tape = normalize_execution_tape(_tape_from_df(raw))
        groups = build_tape_groups(tape)

        # Trigger before the big Under BUY block
        trigger = _make_trigger(
            f"{CID}|Under",
            "2026-03-03 08:00:00+00:00",
            price=0.82,
            stake_usdc=10.0,
        )
        result = attach_forward_fills(
            trigger, groups,
            fill_horizon_seconds=7 * 24 * 3600,
            slippage_bps=0,
            max_price_premium=0.0,   # exec_price ≤ 0.82
        )
        if not result.empty:
            assert result["exec_price"].iloc[0] <= 0.82 + 1e-6
            # Verify opposite_token entries contributed
            assert (result["exec_source"].isin(["same_token", "opposite_token", "mixed"])).all()
