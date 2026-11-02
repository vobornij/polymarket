"""Synthetic end-to-end tests for the stage0 shard pipeline.

``enrich_shard`` (variants required, sample_rows, last_condition_trade_ts
derivation, copyable-qty caps), ``wallet_pnl_shard`` (full per-wallet
training-PnL dict per shard for the global selection) and
``group_and_export_shard`` (filter to the global wallet set, grouping, export,
per-wallet / per-market aggregates) are exercised on small hand-computed
fixtures — including a two-shard test proving that a globally-selected wallet
is exported from *every* shard it traded in.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polymarket_analysis.preprocessing.fill_extender import (
    avail_copy_qty_col,
    copyable_qty_col,
    enrich_shard,
)
from polymarket_analysis.preprocessing.shard_processor import (
    group_and_export_shard,
    wallet_pnl_shard,
)

VARIANTS = [(300, 1.0), (300, 0.95)]

COND1 = "0xaaa1"
COND2 = "0xbbb2"
COND3 = "0xccc3"
T1, C1 = "111", "222"  # cond1: T1 wins
T2, C2 = "333", "444"  # cond2: C2 wins (T2 loses)
T3, C3 = "555", "666"  # cond3: T3 wins
W1, W2, W3, W4, W5 = "w1", "w2", "w3", "w4", "w5"
END_TRAIN_TS = pd.Timestamp(1000, unit="s", tz="UTC")


def _raw_trades_shard0() -> pd.DataFrame:
    return pd.DataFrame({
        "tx_hash": ["h1", "h2", "h3", "h4", "h5", "h6"],
        "log_index": [0, 0, 0, 0, 0, 0],
        "block_timestamp": [100, 110, 120, 130, 1400, 1500],
        "condition_id": [COND1] * 4 + [COND2] * 2,
        "token_id": [T1, T1, T1, C1, T2, C2],
        "outcome": ["Yes"] * 4 + ["Yes", "No"],
        "token_winner": [True, True, True, False, False, True],
        "wallet": [W1, W2, W2, W3, W1, W4],
        "side": ["BUY", "BUY", "SELL", "BUY", "BUY", "BUY"],
        "price": [0.50, 0.40, 0.60, 0.30, 0.60, 0.35],
        "quantity": [10.0, 20.0, 5.0, 8.0, 10.0, 4.0],
        "usdc_amount": [5.0, 8.0, 3.0, 2.4, 6.0, 1.4],
        "position": [0.0] * 6,
    })


def _raw_trades_shard1() -> pd.DataFrame:
    # cond3 resolves before END_TRAIN_TS; W1 has copyable pnl here too
    return pd.DataFrame({
        "tx_hash": ["h7", "h8"],
        "log_index": [0, 0],
        "block_timestamp": [200, 210],
        "condition_id": [COND3, COND3],
        "token_id": [T3, T3],
        "outcome": ["Yes", "Yes"],
        "token_winner": [True, True],
        "wallet": [W1, W5],
        "side": ["BUY", "BUY"],
        "price": [0.30, 0.25],
        "quantity": [10.0, 4.0],
        "usdc_amount": [3.0, 1.0],
        "position": [0.0, 0.0],
    })


def _token_lookup() -> pd.DataFrame:
    return pd.DataFrame({
        "token_id": [T1, C1, T2, C2, T3, C3],
        "condition_id": [COND1, COND1, COND2, COND2, COND3, COND3],
        "outcome": ["Yes", "No", "Yes", "No", "Yes", "No"],
        "token_winner": [True, False, False, True, True, False],
        "final_price": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    })


@pytest.fixture()
def shard_dirs(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    _raw_trades_shard0().to_parquet(raw_dir / "0.parquet", index=False)
    _raw_trades_shard1().to_parquet(raw_dir / "1.parquet", index=False)
    lookup_path = tmp_path / "token_df.parquet"
    _token_lookup().to_parquet(lookup_path, index=False)
    return raw_dir, lookup_path, tmp_path / "enriched", tmp_path / "out"


def _enrich(shard_dirs):
    raw_dir, lookup_path, enriched_dir, _ = shard_dirs
    for name in ("0.parquet", "1.parquet"):
        enrich_shard(raw_dir / name, enriched_dir, lookup_path, variants=VARIANTS)


# ---------------------------------------------------------------------------
# enrich_shard
# ---------------------------------------------------------------------------

def test_enrich_shard_values_and_schema(shard_dirs):
    raw_dir, lookup_path, enriched_dir, _ = shard_dirs
    enrich_shard(raw_dir / "0.parquet", enriched_dir, lookup_path, variants=VARIANTS)

    enr = pd.read_parquet(enriched_dir / "enriched_0.parquet")
    assert list(enr.columns) == [
        "tx_hash", "log_index", "block_timestamp", "condition_id", "token_id",
        "outcome", "token_winner", "wallet", "side", "price", "quantity",
        "usdc_amount", "position", "ts",
        avail_copy_qty_col(300, 1.0), "avail_copy_total_vol_5m_100",
        "avail_copy_count_5m_100", avail_copy_qty_col(300, 0.95),
        copyable_qty_col(300, 1.0), copyable_qty_col(300, 0.95),
        "last_condition_trade_ts",
    ]
    assert len(enr) == 6

    by_tx = enr.set_index("tx_hash")

    # t1 (BUY T @0.50): t2 (BUY T @0.40) is a future better price within 5 min
    assert by_tx.loc["h1", "avail_copy_qty_5m_100"] == pytest.approx(20.0)
    assert by_tx.loc["h1", "avail_copy_total_vol_5m_100"] == pytest.approx(8.0)  # 20*0.40
    assert by_tx.loc["h1", "avail_copy_count_5m_100"] == pytest.approx(1.0)
    assert by_tx.loc["h1", "copyable_qty_5m_100"] == pytest.approx(10.0)  # min(qty, avail)
    # factor 0.95: adjusted buy limit floor(0.50*0.95)=0.475 > 0.40 -> still copyable
    assert by_tx.loc["h1", "avail_copy_qty_5m_095"] == pytest.approx(20.0)
    assert by_tx.loc["h1", "copyable_qty_5m_095"] == pytest.approx(10.0)

    # no future better-price trades for the other fills
    for tx in ["h2", "h3", "h4", "h5", "h6"]:
        assert by_tx.loc[tx, "avail_copy_qty_5m_100"] == pytest.approx(0.0)
        assert by_tx.loc[tx, "avail_copy_qty_5m_095"] == pytest.approx(0.0)
        assert by_tx.loc[tx, "copyable_qty_5m_100"] == pytest.approx(0.0)

    # last_condition_trade_ts = max block_timestamp of the condition (per shard)
    assert by_tx.loc["h1", "last_condition_trade_ts"] == pd.Timestamp(130, unit="s", tz="UTC")
    assert by_tx.loc["h5", "last_condition_trade_ts"] == pd.Timestamp(1500, unit="s", tz="UTC")


def test_enrich_shard_skip_if_exists(shard_dirs):
    raw_dir, lookup_path, enriched_dir, _ = shard_dirs
    enrich_shard(raw_dir / "0.parquet", enriched_dir, lookup_path, variants=VARIANTS)
    first = pd.read_parquet(enriched_dir / "enriched_0.parquet")

    # second call must be a no-op (file untouched)
    enrich_shard(raw_dir / "0.parquet", enriched_dir, lookup_path, variants=VARIANTS)
    second = pd.read_parquet(enriched_dir / "enriched_0.parquet")
    pd.testing.assert_frame_equal(first, second)


def test_enrich_shard_sample_rows(shard_dirs):
    raw_dir, lookup_path, enriched_dir, _ = shard_dirs
    enrich_shard(raw_dir / "0.parquet", enriched_dir, lookup_path, variants=VARIANTS, sample_rows=3)
    enr = pd.read_parquet(enriched_dir / "enriched_0.parquet")
    assert list(enr["tx_hash"]) == ["h1", "h2", "h3"]
    # last trade ts derived from the sampled rows only
    assert enr["last_condition_trade_ts"].iloc[0] == pd.Timestamp(120, unit="s", tz="UTC")


# ---------------------------------------------------------------------------
# wallet_pnl_shard (Phase 1 — full per-wallet dict, no cut)
# ---------------------------------------------------------------------------

def test_wallet_pnl_shard_copyable_full_dict(shard_dirs):
    _enrich(shard_dirs)
    _, _, enriched_dir, _ = shard_dirs

    wallet_pnl, stats = wallet_pnl_shard(
        enriched_dir / "enriched_0.parquet",
        end_train_ts=END_TRAIN_TS,
        selection_pnl="copyable_pnl",
        variants=VARIANTS,
    )
    # W4 only trades in the test period -> excluded; W2/W3 present with 0 pnl
    assert wallet_pnl == {W1: pytest.approx(5.0), W2: pytest.approx(0.0), W3: pytest.approx(0.0)}
    assert stats == {"rows_read": 6, "train_rows": 4, "candidate_wallets": 3}

    pnl1, stats1 = wallet_pnl_shard(
        enriched_dir / "enriched_1.parquet",
        end_train_ts=END_TRAIN_TS,
        selection_pnl="copyable_pnl",
        variants=VARIANTS,
    )
    # t7 (BUY T3 @0.30): t8 (BUY T3 @0.25) is a future better price -> cq=4
    assert pnl1 == {W1: pytest.approx(2.8), W5: pytest.approx(0.0)}
    assert stats1 == {"rows_read": 2, "train_rows": 2, "candidate_wallets": 2}


def test_wallet_pnl_shard_trade_pnl_full_dict(shard_dirs):
    _enrich(shard_dirs)
    _, _, enriched_dir, _ = shard_dirs

    wallet_pnl, _ = wallet_pnl_shard(
        enriched_dir / "enriched_0.parquet",
        end_train_ts=END_TRAIN_TS,
        selection_pnl="trade_pnl",
        variants=VARIANTS,
    )
    # W1=+5 (t1), W2=+12-2 (t2,t3), W3=-2.4 (t4); W4 test-only
    assert wallet_pnl == {
        W1: pytest.approx(5.0),
        W2: pytest.approx(10.0),
        W3: pytest.approx(-2.4),
    }


def test_wallet_pnl_shard_requires_variants(shard_dirs):
    _enrich(shard_dirs)
    _, _, enriched_dir, _ = shard_dirs
    with pytest.raises(ValueError, match="variants is required"):
        wallet_pnl_shard(
            enriched_dir / "enriched_0.parquet",
            end_train_ts=END_TRAIN_TS,
        )


def test_wallet_pnl_shard_rejects_bad_selection_pnl(shard_dirs):
    _enrich(shard_dirs)
    _, _, enriched_dir, _ = shard_dirs
    with pytest.raises(ValueError, match="selection_pnl"):
        wallet_pnl_shard(
            enriched_dir / "enriched_0.parquet",
            end_train_ts=END_TRAIN_TS,
            selection_pnl="banana",
            variants=VARIANTS,
        )


# ---------------------------------------------------------------------------
# group_and_export_shard (Phase 2 — global wallet set)
# ---------------------------------------------------------------------------

def test_group_and_export_shard_grouping_and_export(shard_dirs):
    _enrich(shard_dirs)
    _, _, enriched_dir, out_dir = shard_dirs

    wallet_part, market_part, stats = group_and_export_shard(
        enriched_dir / "enriched_0.parquet",
        out_dir / "0.parquet",
        end_train_ts=END_TRAIN_TS,
        top_wallets={W1, W2},
        variants=VARIANTS,
    )

    assert stats == {
        "rows_read": 6,
        "grouped_rows": 4,
        "train_rows": 3,
        "test_rows": 1,
        "wallets_in_shard": 2,
    }

    out = pd.read_parquet(out_dir / "0.parquet").set_index(["tx_hash", "wallet"])
    assert set(out.index.get_level_values("wallet")) == {W1, W2}

    g1 = out.loc[("h1", W1)]  # cond1, train
    assert g1["side"] == "BUY"
    assert g1["total_quantity"] == pytest.approx(10.0)
    assert g1["avg_price"] == pytest.approx(0.50)
    assert g1["trade_value_usdc"] == pytest.approx(5.0)
    assert g1["final_value_usdc"] == pytest.approx(10.0)  # T1 winner -> 1.0
    assert g1["trade_pnl"] == pytest.approx(5.0)
    assert g1["copyable_pnl"] == pytest.approx(5.0)  # full ratio (10/10 copyable)
    assert g1["num_fills"] == 1
    assert bool(g1["is_train"]) is True
    assert g1["avail_copy_qty_5m_100"] == pytest.approx(20.0)
    assert g1["copyable_qty_5m_100"] == pytest.approx(10.0)
    assert g1["copyable_qty_5m_095"] == pytest.approx(10.0)

    g5 = out.loc[("h5", W1)]  # cond2 resolves after END_TRAIN_TS -> test
    assert g5["final_price"] == pytest.approx(0.0)  # T2 loses
    assert g5["trade_pnl"] == pytest.approx(-6.0)
    assert g5["copyable_pnl"] == pytest.approx(0.0)
    assert bool(g5["is_train"]) is False

    g2 = out.loc[("h2", W2)]  # BUY T qty 20 @ 0.40, not copyable
    assert g2["trade_pnl"] == pytest.approx(12.0)
    assert g2["copyable_pnl"] == pytest.approx(0.0)

    g3 = out.loc[("h3", W2)]  # SELL T qty 5 @ 0.60
    assert g3["trade_value_usdc"] == pytest.approx(3.0)
    assert g3["final_value_usdc"] == pytest.approx(5.0)
    assert g3["trade_pnl"] == pytest.approx(-2.0)
    assert g3["copyable_pnl"] == pytest.approx(0.0)

    # ── wallet summary: train rows only ───────────────────────────────────────
    ws = wallet_part.set_index("wallet")
    assert ws.loc[W1]["num_markets"] == 1  # cond2 is test-period, not counted
    assert ws.loc[W1]["num_trades"] == 1
    assert ws.loc[W1]["total_cost_usdc"] == pytest.approx(5.0)
    assert ws.loc[W1]["trade_pnl"] == pytest.approx(5.0)
    assert ws.loc[W1]["copyable_pnl"] == pytest.approx(5.0)
    assert ws.loc[W2]["num_markets"] == 1
    assert ws.loc[W2]["num_trades"] == 2
    assert ws.loc[W2]["total_cost_usdc"] == pytest.approx(11.0)
    assert ws.loc[W2]["trade_pnl"] == pytest.approx(10.0)
    assert ws.loc[W2]["copyable_pnl"] == pytest.approx(0.0)

    # ── market summary: all periods, kept wallets only ─────────────────────────
    ms = market_part.set_index("condition_id")
    assert ms.loc[COND1]["num_wallets"] == 2
    assert ms.loc[COND1]["num_trades"] == 3
    assert ms.loc[COND1]["volume_usdc"] == pytest.approx(16.0)
    assert ms.loc[COND1]["final_value_usdc"] == pytest.approx(35.0)
    assert ms.loc[COND2]["num_wallets"] == 1
    assert ms.loc[COND2]["num_trades"] == 1
    assert ms.loc[COND2]["volume_usdc"] == pytest.approx(6.0)
    assert ms.loc[COND2]["final_value_usdc"] == pytest.approx(0.0)


def test_group_and_export_shard_requires_variants(shard_dirs):
    _enrich(shard_dirs)
    _, _, enriched_dir, out_dir = shard_dirs
    with pytest.raises(ValueError, match="variants is required"):
        group_and_export_shard(
            enriched_dir / "enriched_0.parquet",
            out_dir / "0.parquet",
            end_train_ts=END_TRAIN_TS,
            top_wallets={W1},
        )


def test_global_selection_exports_wallet_from_every_shard(shard_dirs):
    """Regression test: a globally-selected wallet keeps its trades from every
    shard it traded in (per-shard selection used to miss them)."""
    _enrich(shard_dirs)
    _, _, enriched_dir, out_dir = shard_dirs

    # Phase 1 — sum per-wallet pnl dicts across both shards, global top cut
    wallet_pnl: dict[str, float] = {}
    for name in ("enriched_0.parquet", "enriched_1.parquet"):
        shard_pnl, _ = wallet_pnl_shard(
            enriched_dir / name,
            end_train_ts=END_TRAIN_TS,
            selection_pnl="copyable_pnl",
            variants=VARIANTS,
        )
        for w, pnl in shard_pnl.items():
            wallet_pnl[w] = wallet_pnl.get(w, 0.0) + pnl

    # W1: 5.0 (shard 0) + 2.8 (shard 1) = 7.8; W2/W3/W5 = 0
    assert wallet_pnl == {
        W1: pytest.approx(7.8),
        W2: pytest.approx(0.0),
        W3: pytest.approx(0.0),
        W5: pytest.approx(0.0),
    }

    pnl_series = pd.Series(wallet_pnl)
    threshold = float(pnl_series.quantile(1.0 - 0.25))  # top 25% -> 1 of 4 wallets
    final_wallets = set(pnl_series[pnl_series >= threshold].index)
    assert final_wallets == {W1}

    # Phase 2 — export both shards with the global set
    for name in ("enriched_0.parquet", "enriched_1.parquet"):
        group_and_export_shard(
            enriched_dir / name,
            out_dir / name.removeprefix("enriched_"),
            end_train_ts=END_TRAIN_TS,
            top_wallets=final_wallets,
            variants=VARIANTS,
        )

    out0 = pd.read_parquet(out_dir / "0.parquet")
    out1 = pd.read_parquet(out_dir / "1.parquet")

    # W1's trades from BOTH shards are exported (the per-shard bug missed them)
    assert set(out0["tx_hash"]) == {"h1", "h5"}  # cond1 (train) + cond2 (test)
    assert set(out1["tx_hash"]) == {"h7"}        # cond3 (train, other shard)
    assert set(out0["wallet"]) == {W1}
    assert set(out1["wallet"]) == {W1}
