"""Copy-wallet quality gradient signal family.

Attaches the candidate copy-wallet's own train-period quality metrics as signals:
``buy_roi``, ``copyable_roi``, ``copyable_pnl``, ``opening_roi``, ``trade_count``,
``num_markets``.  Tests whether candidates from higher-quality copy wallets are better
trades within the copy universe (see `ideas/copy_wallet_quality_gradient.md`).

Caveat: ``wallet_metrics`` is train-period, so train IC is inflated (a train candidate's
own outcome feeds its wallet's train metric); the honest read is val/test.
"""

from __future__ import annotations

import pandas as pd

from signal_lab.filters import COPY_DEFAULT, WalletFilter
from signal_lab.strategies.base import DeclarativeStrategy


def signal_col(name: str) -> str:
    return f"sig_wal_{name}"


_QUALITY_COLS = [
    ("buy_roi", "buy_roi"),
    ("copyable_roi", "copyable_roi"),
    ("copyable_pnl", "copyable_pnl"),
    ("opening_roi", "opening_roi"),
    ("trade_count", "trade_count"),
    ("num_markets", "num_markets"),
]


class CopyWalletQualitySignals(DeclarativeStrategy):
    copy_mask: WalletFilter = COPY_DEFAULT

    @property
    def name(self) -> str:
        return "CopyWalletQualitySignals"

    def get_signal_columns(self) -> list[str]:
        return [signal_col(src) for src, _ in _QUALITY_COLS]

    def calculate_signals(
        self,
        splits: dict[str, pd.DataFrame],
        *,
        trades: pd.DataFrame,
        wallet_metrics: pd.DataFrame,
        hold_metrics: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        wm = wallet_metrics.set_index("wallet")
        for src, _ in _QUALITY_COLS:
            if src not in wm.columns:
                continue
            for name, frame in splits.items():
                frame[signal_col(src)] = frame["wallet"].map(wm[src]).fillna(0.0)
        return splits
