"""Underwater-add x opposite-crowding composite (tail-drop hypothesis).

Attaches the candidate copy-wallet's OWN pre-buy position state (never
computed before — all existing strategies only aggregate archetype sets):

- ``sig_ua_held``          quantity already held on the candidate token before
                           this BUY (0 = fresh open, not an add)
- ``sig_ua_cost``          value-at-cost (USDC) of that held position
- ``sig_ua_entry``         average entry price of the held position
- ``sig_ua_premium``       entry/price - 1 (< 0 = the wallet is underwater and
                           adding to a losing position = averaging down)
- ``sig_ua_underwater_usdc`` cost - held*price  (negative = underwater amount)

Plus the standard aggregate crowd signals on the OPPOSITE outcome for the copy
set itself and the archetype sets (BOTH_SIDES / FLIPPER): ``sig_fval_opp_*``
(fresh value-at-cost), ``sig_val_opp_*``, ``sig_uwl_opp_*``.

Hypothesis: BUYs where the copy wallet doubles down underwater AND the strong
wallet crowd is on the opposite side are unusually bad trades; dropping a small
tail of them should raise raw pnl (see `ideas/pnl_gate_revaluation.md`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_lab.filters import BOTH_SIDES, COPY_DEFAULT, FLIPPER, WalletFilter
from signal_lab.signal_engines import (
    PositionSignalEngine,
    VAL_OPP,
    VAL_OWN,
    UWL_OPP,
)
from signal_lab.stage1 import attach_position_signal_panel
from signal_lab.strategies.base import DeclarativeStrategy

_UA_COLS = [
    "held",
    "cost",
    "entry",
    "premium",
    "underwater_usdc",
]


def ua_col(name: str) -> str:
    return f"sig_ua_{name}"


class UnderwaterAddCrowding(DeclarativeStrategy):
    copy_mask: WalletFilter = COPY_DEFAULT
    signal_sets = [COPY_DEFAULT, BOTH_SIDES, FLIPPER]
    kinds = [VAL_OPP, VAL_OWN, UWL_OPP]
    fresh_kinds = [VAL_OPP]
    taus_h = [6.0]

    @property
    def name(self) -> str:
        return "UnderwaterAddCrowding"

    def get_signal_columns(self) -> list[str]:
        cols = [ua_col(c) for c in _UA_COLS]
        for flt in self.signal_sets:
            for kind in self.kinds:
                cols.append(
                    f"sig_{kind.family}_{kind.var}_{flt.name}"
                )
            for kind in self.fresh_kinds:
                cols.append(
                    f"sig_{kind.fresh().family}_{kind.fresh().var}_6h_{flt.name}"
                )
        return cols

    def _attach_own_position(
        self,
        splits: dict[str, pd.DataFrame],
        trades: pd.DataFrame,
        copy_wallets: set[str],
    ) -> dict[str, pd.DataFrame]:
        engine = getattr(trades, "_position_signal_engine", None)
        if engine is None:
            engine = PositionSignalEngine(trades)
            object.__setattr__(trades, "_position_signal_engine", engine)
        conditions = set()
        for frame in splits.values():
            conditions.update(frame["condition_id"].unique())
        ck = engine.build_checkpoints(copy_wallets, conditions=conditions)
        ck = engine.compute_vac(ck)
        right = ck.sort_values("dt")[
            ["dt", "wallet", "condition_id", "outcome", "position", "vac"]
        ]
        for name, frame in splits.items():
            left = frame.sort_values("dt")
            m = pd.merge_asof(
                left[["dt", "wallet", "condition_id", "outcome"]],
                right,
                on="dt",
                by=["wallet", "condition_id", "outcome"],
                direction="backward",
                allow_exact_matches=False,
            )
            m = m.sort_index()
            held = m["position"].fillna(0.0).to_numpy()
            cost = m["vac"].fillna(0.0).to_numpy()
            price = frame["price"].to_numpy()
            frame[ua_col("held")] = held
            frame[ua_col("cost")] = cost
            with np.errstate(divide="ignore", invalid="ignore"):
                entry = np.where(held > 0, cost / held, np.nan)
                premium = np.where(held > 0, entry / price - 1.0, 0.0)
            frame[ua_col("entry")] = entry
            frame[ua_col("premium")] = premium
            frame[ua_col("underwater_usdc")] = cost - held * price
        return splits

    def calculate_signals(
        self,
        splits: dict[str, pd.DataFrame],
        *,
        trades: pd.DataFrame,
        wallet_metrics: pd.DataFrame,
        hold_metrics: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        copy_wallets = set(self.copy_mask(wallet_metrics, hold_metrics))
        splits = self._attach_own_position(splits, trades, copy_wallets)
        splits, _ = attach_position_signal_panel(
            trades,
            splits,
            self.signal_sets,
            kinds=self.kinds,
            fresh_kinds=self.fresh_kinds,
            taus_h=self.taus_h,
            wallet_metrics=wallet_metrics,
            hold_metrics=hold_metrics,
        )
        return splits
