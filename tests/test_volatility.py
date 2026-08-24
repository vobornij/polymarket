import numpy as np
import pandas as pd
import pytest

from polymarket_analysis.wallet_selection.volatility import compute_wallet_metrics


def _fills(pnl, copyable_pnl, wallet="0xa", copy_frac_20m=None) -> pd.DataFrame:
    pnl = np.asarray(pnl, dtype=float)
    copyable_pnl = np.asarray(copyable_pnl, dtype=float)
    n = len(pnl)
    if copy_frac_20m is None:
        copy_frac_20m = np.where(pnl != 0, copyable_pnl / pnl, 0.0)
    return pd.DataFrame({
        "wallet": wallet,
        "dt": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
        "condition_id": "c1",
        "side": "BUY",
        "notional": 10.0,
        "quantity": 1.0,
        "pnl": pnl,
        "copyable_pnl": copyable_pnl,
        "copyable_qty_5m_100": np.where(copyable_pnl != 0, 1.0, 0.0),
        "avail_copy_total_vol_5m_100": 1e9,
        "copyable_qty_20m_100": np.asarray(copy_frac_20m, dtype=float),
        "avail_copy_qty_20m_100": 1e9,
    })


def _similarity(
    df: pd.DataFrame,
    wallet: str = "0xa",
    col: str = "copyable_pnl_similarity",
) -> float:
    res, _ = compute_wallet_metrics(df)
    return res.loc[res["wallet"] == wallet, col].iloc[0]


def test_scaled_copyable_curve_scores_one():
    pnl = np.linspace(-2.0, 6.0, 20)
    assert _similarity(_fills(pnl, 0.3 * pnl)) == pytest.approx(1.0, abs=1e-9)


def test_sign_flipped_copyable_curve_scores_minus_one():
    pnl = np.linspace(-2.0, 6.0, 20)
    assert _similarity(_fills(pnl, -0.5 * pnl)) == pytest.approx(-1.0, abs=1e-9)


def test_concentrated_copyable_pnl_scores_low():
    pnl = np.ones(100)
    copyable = np.zeros(100)
    copyable[50] = 1.0
    assert _similarity(_fills(pnl, copyable)) == pytest.approx(0.1, abs=1e-9)


def test_zero_copyable_is_nan():
    pnl = np.linspace(-2.0, 6.0, 20)
    assert np.isnan(_similarity(_fills(pnl, np.zeros(20))))


def test_few_active_buckets_is_nan():
    pnl = np.array([1.0, 2.0, 3.0])
    assert np.isnan(_similarity(_fills(pnl, 0.5 * pnl)))


def test_multiple_wallets():
    pnl = np.linspace(-2.0, 6.0, 20)
    df = pd.concat([
        _fills(pnl, 0.3 * pnl, wallet="0xa"),
        _fills(pnl, -0.5 * pnl, wallet="0xb"),
    ], ignore_index=True)
    res, _ = compute_wallet_metrics(df)
    sims = res.set_index("wallet")["copyable_pnl_similarity"]
    assert sims["0xa"] == pytest.approx(1.0, abs=1e-9)
    assert sims["0xb"] == pytest.approx(-1.0, abs=1e-9)


PROFIT_COL = "copyable_pnl_similarity_profit"


def test_profit_similarity_scaled_copy_scores_one():
    pnl = np.array([1.0, -1.0] * 10)
    assert _similarity(_fills(pnl, 0.3 * pnl), col=PROFIT_COL) == pytest.approx(1.0, abs=1e-9)


def test_profit_similarity_ignores_loss_divergence():
    pnl = np.array([1.0, -1.0] * 10)
    copyable = np.where(pnl > 0, pnl, -5.0 * pnl)
    assert _similarity(_fills(pnl, copyable), col=PROFIT_COL) == pytest.approx(1.0, abs=1e-9)
    assert _similarity(_fills(pnl, copyable)) < 0.0


def test_profit_similarity_penalizes_missed_upside():
    pnl = np.concatenate([np.ones(10), -np.ones(10)])
    copyable = np.zeros(20)
    copyable[0] = 1.0
    assert _similarity(_fills(pnl, copyable), col=PROFIT_COL) == pytest.approx(
        1.0 / np.sqrt(10), abs=1e-9)


def test_profit_similarity_flipped_win_scores_negative():
    pnl = np.concatenate([np.ones(10), -np.ones(10)])
    copyable = np.where(pnl > 0, -0.5 * pnl, 0.3 * pnl)
    assert _similarity(_fills(pnl, copyable), col=PROFIT_COL) == pytest.approx(-1.0, abs=1e-9)


def test_profit_similarity_few_winning_buckets_is_nan():
    pnl = np.concatenate([np.ones(3), -np.ones(10)])
    assert np.isnan(_similarity(_fills(pnl, 0.3 * pnl), col=PROFIT_COL))
    assert _similarity(_fills(pnl, 0.3 * pnl)) == pytest.approx(1.0, abs=1e-9)


def test_profit_similarity_no_winning_buckets_is_nan():
    pnl = -np.ones(10)
    assert np.isnan(_similarity(_fills(pnl, 0.3 * pnl), col=PROFIT_COL))
    assert _similarity(_fills(pnl, 0.3 * pnl)) == pytest.approx(1.0, abs=1e-9)


LOSS_COL = "copyable_pnl_similarity_loss"


def test_loss_similarity_scaled_copy_scores_one():
    pnl = np.array([1.0, -1.0] * 10)
    assert _similarity(_fills(pnl, 0.3 * pnl), col=LOSS_COL) == pytest.approx(1.0, abs=1e-9)


def test_loss_similarity_amplified_losses_score_one():
    pnl = np.array([1.0, -1.0] * 10)
    copyable = np.where(pnl < 0, 2.0 * pnl, pnl)
    assert _similarity(_fills(pnl, copyable), col=LOSS_COL) == pytest.approx(1.0, abs=1e-9)


def test_loss_similarity_ignores_profit_divergence():
    pnl = np.array([1.0, -1.0] * 10)
    copyable = np.where(pnl < 0, pnl, -5.0 * pnl)
    assert _similarity(_fills(pnl, copyable), col=LOSS_COL) == pytest.approx(1.0, abs=1e-9)
    assert _similarity(_fills(pnl, copyable), col=PROFIT_COL) < 0.0


def test_loss_similarity_few_losing_buckets_is_nan():
    pnl = np.concatenate([np.ones(10), -np.ones(3)])
    assert np.isnan(_similarity(_fills(pnl, 0.3 * pnl), col=LOSS_COL))
    assert _similarity(_fills(pnl, 0.3 * pnl)) == pytest.approx(1.0, abs=1e-9)


def test_loss_similarity_no_losing_buckets_is_nan():
    pnl = np.ones(10)
    assert np.isnan(_similarity(_fills(pnl, 0.3 * pnl), col=LOSS_COL))
    assert _similarity(_fills(pnl, 0.3 * pnl)) == pytest.approx(1.0, abs=1e-9)


SIM_20M = "copyable_pnl_similarity_20m_100"
SIM_20M_PROFIT = "copyable_pnl_similarity_profit_20m_100"
SIM_20M_LOSS = "copyable_pnl_similarity_loss_20m_100"


def test_similarity_20m_scaled_copy_scores_one():
    pnl = np.linspace(-2.0, 6.0, 20)
    df = _fills(pnl, 0.3 * pnl)
    assert _similarity(df, col=SIM_20M) == pytest.approx(1.0, abs=1e-9)
    assert _similarity(df, col=SIM_20M_PROFIT) == pytest.approx(1.0, abs=1e-9)
    assert _similarity(df, col=SIM_20M_LOSS) == pytest.approx(1.0, abs=1e-9)


def test_similarity_20m_independent_of_base():
    pnl = np.ones(100)
    frac = np.zeros(100)
    frac[50] = 1.0
    df = _fills(pnl, 0.3 * pnl, copy_frac_20m=frac)
    assert _similarity(df) == pytest.approx(1.0, abs=1e-9)
    assert _similarity(df, col=SIM_20M) == pytest.approx(0.1, abs=1e-9)


def test_similarity_20m_zero_copyable_is_nan():
    pnl = np.linspace(-2.0, 6.0, 20)
    df = _fills(pnl, 0.3 * pnl, copy_frac_20m=np.zeros(20))
    assert np.isnan(_similarity(df, col=SIM_20M))
    assert _similarity(df) == pytest.approx(1.0, abs=1e-9)


def test_similarity_20m_bucket_clip():
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    df = pd.DataFrame({
        "wallet": ["0xa", "0xa"],
        "dt": [ts, ts],
        "condition_id": ["c1", "c1"],
        "side": ["BUY", "BUY"],
        "notional": [10.0, 10.0],
        "quantity": [1.0, 1.0],
        "pnl": [1.0, 1.0],
        "copyable_pnl": [1.0, 1.0],
        "copyable_qty_5m_100": [1.0, 1.0],
        "avail_copy_total_vol_5m_100": [1e9, 1e9],
        "copyable_qty_20m_100": [1.0, 1.0],
        "avail_copy_qty_20m_100": [1.0, 1.0],
    })
    _, buckets = compute_wallet_metrics(df)
    assert buckets["copyable_qty_20m"].iloc[0] == pytest.approx(1.0)
    assert buckets["copyable_pnl_20m_100"].iloc[0] == pytest.approx(1.0)
