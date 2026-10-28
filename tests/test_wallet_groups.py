import numpy as np
import pandas as pd
import pytest

from polymarket_analysis.copy_groups.wallet_groups import _compute_wallet_trade_stats


def _trades(market_pnls: dict[str, list[float]], wallet: str = "0xa") -> pd.DataFrame:
    rows = []
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    for cid, pnls in market_pnls.items():
        for p in pnls:
            rows.append({
                "wallet": wallet,
                "condition_id": cid,
                "side": "BUY",
                "dt": t0 + pd.Timedelta(minutes=5 * len(rows)),
                "usdc_amount": 10.0,
                "trade_pnl": float(p),
                "copyable_pnl": float(p),
            })
    return pd.DataFrame(rows)


def _market_hhi(market_pnls: dict[str, list[float]]) -> float:
    stats = _compute_wallet_trade_stats(_trades(market_pnls))
    return stats.loc[stats["wallet"] == "0xa", "market_pnl_hhi"].iloc[0]


def test_hhi_single_winning_market_is_one():
    assert _market_hhi({"c1": [5.0]}) == pytest.approx(1.0)


def test_hhi_equal_winning_markets():
    df = {"c1": [5.0], "c2": [5.0], "c3": [5.0], "c4": [5.0]}
    assert _market_hhi(df) == pytest.approx(0.25)


def test_hhi_offsetting_losses_not_diluted():
    assert _market_hhi({"c1": [10.0], "c2": [10.0], "c3": [-19.0]}) == pytest.approx(200.0)


def test_hhi_nonpositive_copyable_pnl_is_nan():
    assert np.isnan(_market_hhi({"c1": [2.0], "c2": [-3.0]}))
    assert np.isnan(_market_hhi({"c1": [2.0], "c2": [-2.0]}))
    assert np.isnan(_market_hhi({"c1": [-1.0]}))
