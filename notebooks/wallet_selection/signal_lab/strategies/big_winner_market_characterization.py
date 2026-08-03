"""Big-winner market-condition characterization signal family.

Trade-frame market-state features computed at candidate time ``t`` (no position
engine): condition age, cumulative activity/notional on the condition before ``t``,
number of outcomes, candidate price, and price drift over the last ``tau_h`` on the
same outcome (computed with ``merge_asof`` — memory-safe, no cross products).

The hypothesis (validated in `ideas/big_winner_market_characterization.md`): copy PnL
is concentrated in a few already-"discovered" markets, so market-state features should
predict residualized ROI *above and beyond* the per-trade crowd signals.
"""

from __future__ import annotations

import pandas as pd

from signal_lab.filters import COPY_DEFAULT, WalletFilter
from signal_lab.strategies.base import DeclarativeStrategy


def signal_col(name: str) -> str:
    return f"sig_mkt_{name}"


_MERGE_KEYS = ["condition_id", "outcome", "dt", "wallet"]


class BigWinnerMarketCharacterization(DeclarativeStrategy):
    copy_mask: WalletFilter = COPY_DEFAULT
    # momentum windows in hours (best retained after sweep)
    mom_taus_h: list[float] = [6.0]

    def __init__(self, mom_taus_h: list[float] | None = None):
        if mom_taus_h is not None:
            self.mom_taus_h = list(mom_taus_h)

    @property
    def name(self) -> str:
        return "BigWinnerMarketCharacterization"

    def get_signal_columns(self) -> list[str]:
        cols = [
            signal_col("age_h"),
            signal_col("trades_before"),
            signal_col("notional_before"),
            signal_col("n_outcomes"),
            signal_col("price"),
        ]
        for tau in self.mom_taus_h:
            cols.append(signal_col(f"pmom_{tau:g}h"))
        return cols

    def calculate_signals(
        self,
        splits: dict[str, pd.DataFrame],
        *,
        trades: pd.DataFrame,
        wallet_metrics: pd.DataFrame,
        hold_metrics: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        t = trades.sort_values("dt", kind="mergesort").copy()
        t["_row"] = t.groupby("condition_id", sort=False).cumcount()
        t["_cum_notional"] = t.groupby("condition_id", sort=False)["quantity"].cumsum()

        # condition first-trade time (any trade) and outcome count
        cond_first = t.groupby("condition_id")["dt"].min().rename("_cond_first")
        cond_n_outcomes = (
            t.groupby("condition_id")["outcome"].nunique().rename("_n_outcomes")
        )
        cond_meta = pd.concat([cond_first, cond_n_outcomes], axis=1)

        attach = t[["condition_id", "outcome", "dt", "wallet", "_row", "_cum_notional"]]
        attach = attach.merge(cond_meta, left_on="condition_id", right_index=True)
        attach[signal_col("trades_before")] = attach["_row"]
        attach[signal_col("notional_before")] = attach["_cum_notional"]
        attach[signal_col("age_h")] = (
            (attach["dt"] - attach["_cond_first"]).dt.total_seconds() / 3600.0
        )
        attach[signal_col("n_outcomes")] = attach["_n_outcomes"]

        # candidate price 6h ago: asof lookup at t - tau (merge_asof needs
        # left/right sorted by the time key globally, by-keys need not be sorted)
        by = ["condition_id", "outcome"]
        right = t[by + ["dt", "price"]].sort_values("dt")
        cands = pd.concat(
            [frame[by + ["dt", "wallet", "price"]] for frame in splits.values()],
            ignore_index=True,
        )
        cands = cands.rename(columns={"price": "_p_now"})
        for tau in self.mom_taus_h:
            look = cands.copy()
            look["lookup_dt"] = look["dt"] - pd.Timedelta(hours=tau)
            look = look.sort_values("lookup_dt")
            prior = pd.merge_asof(
                look,
                right,
                left_on="lookup_dt",
                right_on="dt",
                by=by,
                direction="backward",
            )
            prior = prior.rename(columns={"price": f"_p_tau{tau:g}", "dt_x": "dt"})
            prior = prior.drop(columns=[c for c in ["dt_y", "lookup_dt"] if c in prior])
            prior[signal_col(f"pmom_{tau:g}h")] = (
                prior["_p_now"] - prior[f"_p_tau{tau:g}"]
            ).fillna(0.0)
            attach = attach.merge(
                prior[_MERGE_KEYS + [signal_col(f"pmom_{tau:g}h")]],
                on=_MERGE_KEYS,
                how="left",
            )

        sig_cols = [c for c in attach.columns if c.startswith("sig_mkt")]
        for name, frame in splits.items():
            m = frame.merge(attach[_MERGE_KEYS + sig_cols], on=_MERGE_KEYS, how="left")
            for col in sig_cols:
                m[col] = m[col].fillna(0.0)
            m[signal_col("price")] = m["price"].fillna(0.0)
            splits[name] = m
        return splits
