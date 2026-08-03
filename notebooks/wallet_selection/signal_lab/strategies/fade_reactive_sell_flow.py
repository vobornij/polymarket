"""Fade-reactive-sell-flow signal family.

For each candidate copy BUY at time ``t``, counts same-outcome SELL trades in the
prior ``tau_h`` window by reactive wallet groups (BOTH_SIDES / OVERSELLER / RETAIL).
Both a raw trade count and a distinct-wallet count are attached.

The hypothesis (validated in `ideas/fade_reactive_sell_flow.md`): a candidate entering
into an active same-outcome sell-off by reactive wallets is a worse trade.  These are
*flow* signals (no position engine needed) — higher = worse, the copy-filter direction
is the negation.
"""

from __future__ import annotations

import pandas as pd

from signal_lab.filters import BOTH_SIDES, OVERSELLER, RETAIL, COPY_DEFAULT, WalletFilter
from signal_lab.strategies.base import DeclarativeStrategy


def signal_col(variant: str, tau_h: float, set_name: str) -> str:
    return f"sig_fsf_{variant}_{tau_h:g}h_{set_name}"


class FadeReactiveSellFlow(DeclarativeStrategy):
    copy_mask: WalletFilter = COPY_DEFAULT
    signal_sets: list[WalletFilter] = [BOTH_SIDES, OVERSELLER, RETAIL]
    # --- Best parameters (kept for fast runs) ---
    # Count variant only (distinct ~ identical); 6h best for overseller/retail,
    # BOTH_SIDES flat across taus. 6h keeps compute ~3x lower than 0.5h.
    taus_h: list[float] = [6.0]
    variants: list[str] = ["sell"]
    # --- Tried / not retained ---
    # taus_h = [0.5, 1.0, 6.0]  # 0.5h: BOTH_SIDES -0.196 / OVERSELLER -0.171 / RETAIL -0.091
    #                           # 1h:   BOTH_SIDES -0.197 / OVERSELLER -0.176 / RETAIL -0.102
    # variants = ["sell", "sell_distinct"]  # distinct ≈ count (rank corr ~1)

    def __init__(
        self,
        signal_sets: list[WalletFilter] | None = None,
        taus_h: list[float] | None = None,
        variants: list[str] | None = None,
    ):
        if signal_sets is not None:
            self.signal_sets = list(signal_sets)
        if taus_h is not None:
            self.taus_h = list(taus_h)
        if variants is not None:
            self.variants = list(variants)

    @property
    def name(self) -> str:
        return "FadeReactiveSellFlow"

    def get_signal_columns(self) -> list[str]:
        cols = []
        for variant in self.variants:
            for tau in self.taus_h:
                for flt in self.signal_sets:
                    cols.append(signal_col(variant, tau, flt.name))
        return cols

    def calculate_signals(
        self,
        splits: dict[str, pd.DataFrame],
        *,
        trades: pd.DataFrame,
        wallet_metrics: pd.DataFrame,
        hold_metrics: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        cands = pd.concat(
            [frame[["condition_id", "outcome", "dt", "wallet"]] for frame in splits.values()],
            ignore_index=True,
        )

        sell_frames: dict[str, pd.DataFrame] = {}
        for flt in self.signal_sets:
            wallets = set(flt(wallet_metrics, hold_metrics))
            sell_frames[flt.name] = trades[
                trades["wallet"].isin(wallets) & (trades["side"] == "SELL")
            ][["condition_id", "outcome", "dt", "wallet"]].copy()

        for name, frame in splits.items():
            for flt in self.signal_sets:
                sells = sell_frames[flt.name]
                for tau in self.taus_h:
                    win_s = tau * 3600.0
                    merged = sells[["condition_id", "outcome", "dt", "wallet"]].merge(
                        frame[["condition_id", "outcome", "dt", "wallet"]].rename(
                            columns={"dt": "c_dt", "wallet": "c_wallet"}
                        ),
                        on=["condition_id", "outcome"],
                        how="inner",
                    )
                    lag = (merged["c_dt"] - merged["dt"]).dt.total_seconds()
                    merged = merged[(lag > 0) & (lag <= win_s)]
                    counts = (
                        merged.groupby(["condition_id", "outcome", "c_dt", "c_wallet"])
                        .agg(
                            n_sell=("wallet", "size"),
                            n_distinct=("wallet", "nunique"),
                        )
                        .reset_index()
                        .rename(columns={"c_dt": "dt", "c_wallet": "wallet"})
                    )
                    keys = ["condition_id", "outcome", "dt", "wallet"]
                    frame = frame.merge(
                        counts, on=keys, how="left"
                    )
                    frame["n_sell"] = frame["n_sell"].fillna(0.0)
                    frame["n_distinct"] = frame["n_distinct"].fillna(0.0)
                    frame[signal_col("sell", tau, flt.name)] = frame["n_sell"]
                    frame[signal_col("sell_distinct", tau, flt.name)] = frame["n_distinct"]
                    frame = frame.drop(columns=["n_sell", "n_distinct"])
            splits[name] = frame
        return splits
