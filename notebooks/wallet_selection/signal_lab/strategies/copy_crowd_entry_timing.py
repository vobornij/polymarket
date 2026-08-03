"""Copy-crowd entry-timing signal family.

For each candidate copy BUY at time ``t``, counts how many **distinct copy-set
wallets** already bought the same market before ``t`` (lifetime, per condition and
per condition+outcome), plus a binary first-mover flag.  Optionally, recency
variants count distinct copy wallets that bought the same (condition, outcome)
within the last ``tau_h`` hours.

The hypothesis (validated in `ideas/copy_crowd_entry_timing.md`): late entrants into
an already-crowded copy market underperform; first-movers carry the edge.  These are
*flow* signals (no position engine needed) — higher = worse, the copy-filter direction
is the negation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.filters import COPY_DEFAULT, WalletFilter
from signal_lab.strategies.base import DeclarativeStrategy


def signal_col(name: str) -> str:
    return f"sig_ccw_{name}"


class CopyCrowdEntryTiming(DeclarativeStrategy):
    copy_mask: WalletFilter = COPY_DEFAULT
    # --- Best parameters (kept for fast runs) ---
    # Lifetime crowding (n_cond) strongest; recency 24h beat 6h on train.
    taus_h: list[float] = [24.0]
    # --- Tried / not retained ---
    # taus_h = [6.0, 24.0]  # 6h: IC_train -0.093, kept 24h only (-0.106)

    def __init__(self, taus_h: list[float] | None = None):
        if taus_h is not None:
            self.taus_h = list(taus_h)

    @property
    def name(self) -> str:
        return "CopyCrowdEntryTiming"

    def get_signal_columns(self) -> list[str]:
        cols = [
            signal_col("n_cond"),
            signal_col("n_co"),
            signal_col("first"),
        ]
        for tau in self.taus_h:
            cols.append(signal_col(f"recent_{tau:g}h_co"))
        return cols

    def calculate_signals(
        self,
        splits: dict[str, pd.DataFrame],
        *,
        trades: pd.DataFrame,
        wallet_metrics: pd.DataFrame,
        hold_metrics: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        copy_wallets = set(self.copy_mask(wallet_metrics, hold_metrics))
        buys = trades[
            trades["wallet"].isin(copy_wallets) & (trades["side"] == "BUY")
        ][["condition_id", "outcome", "dt", "wallet"]].copy()
        if buys.empty:
            return splits
        buys = buys.sort_values("dt", kind="mergesort")

        # Lifetime distinct-copy-wallet counts (inclusive of the row's wallet).
        buys["_first_cond"] = ~buys.duplicated(subset=["condition_id", "wallet"])
        buys["_first_co"] = ~buys.duplicated(
            subset=["condition_id", "outcome", "wallet"]
        )
        buys[signal_col("n_cond")] = (
            buys.groupby("condition_id", sort=False)["_first_cond"].cumsum() - 1
        )
        buys[signal_col("n_co")] = (
            buys.groupby(["condition_id", "outcome"], sort=False)["_first_co"].cumsum()
            - 1
        )
        buys[signal_col("first")] = (buys[signal_col("n_cond")] == 0).astype(float)

        # Recency variants: distinct copy wallets on same (cond,outcome) in last tau.
        # Self-merge on (condition_id, outcome); lag in (0, tau_h].
        for tau in self.taus_h:
            win_s = tau * 3600.0
            merged = buys[["condition_id", "outcome", "dt", "wallet"]].merge(
                buys[["condition_id", "outcome", "dt", "wallet"]].rename(
                    columns={"wallet": "c_wallet", "dt": "c_dt"}
                ),
                on=["condition_id", "outcome"],
                how="inner",
            )
            lag = (merged["c_dt"] - merged["dt"]).dt.total_seconds()
            merged = merged[(lag > 0) & (lag <= win_s)]
            counts = (
                merged.groupby(["condition_id", "outcome", "c_dt", "c_wallet"])
                ["wallet"].nunique()
                .rename("cnt")
                .reset_index()
            )
            counts = counts.rename(columns={"c_dt": "dt", "c_wallet": "wallet"})
            col = signal_col(f"recent_{tau:g}h_co")
            buys = buys.merge(counts, on=["condition_id", "outcome", "dt", "wallet"], how="left")
            buys[col] = buys["cnt"].fillna(0.0)
            buys = buys.drop(columns="cnt")

        attach_cols = [
            "condition_id",
            "outcome",
            "dt",
            "wallet",
            signal_col("n_cond"),
            signal_col("n_co"),
            signal_col("first"),
        ] + [signal_col(f"recent_{tau:g}h_co") for tau in self.taus_h]
        attach = buys[attach_cols].drop_duplicates(
            subset=["condition_id", "outcome", "dt", "wallet"]
        )

        merge_keys = ["condition_id", "outcome", "dt", "wallet"]
        sig_cols = [c for c in attach.columns if c.startswith("sig_ccw")]
        for name, frame in splits.items():
            m = frame.merge(attach, on=merge_keys, how="left")
            for col in sig_cols:
                m[col] = m[col].fillna(0.0)
            splits[name] = m
        return splits
