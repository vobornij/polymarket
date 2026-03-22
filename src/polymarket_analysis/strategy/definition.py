"""
Core dataclasses for the research strategy layer.

WalletSelectionStrategy
    Full persisted artifact produced by the wallet-selection stage.

TriggerSpec
    Lightweight descriptor of a trigger rule: function reference + params + mode.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class TriggerSpec:
    """Descriptor of a trigger rule.

    Attributes
    ----------
    fn_ref:
        Fully-qualified importable path to the trigger function.
        Example: ``'polymarket_analysis.strategy.triggers.score_threshold'``
    params:
        Dict of parameters passed to the trigger function at backtest time.
        Example: ``{'threshold': 0.80, 'dynamic_sizing': True}``
    mode:
        ``'frame'`` (default) — the trigger function receives the full signals
        DataFrame and returns a boolean Series (vectorised, fast).
        ``'row'`` — the trigger function receives one ``pd.Series`` row and
        returns a bool (flexible but slower; useful for prototyping).
    """

    fn_ref: str
    params: dict[str, Any] = field(default_factory=dict)
    mode: str = "frame"  # 'frame' | 'row'

    def to_dict(self) -> dict[str, Any]:
        return {"fn_ref": self.fn_ref, "params": self.params, "mode": self.mode}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TriggerSpec":
        return cls(
            fn_ref=d["fn_ref"],
            params=dict(d.get("params", {})),
            mode=d.get("mode", "frame"),
        )


@dataclass
class WalletSelectionStrategy:
    """Complete, self-describing wallet-selection + trigger strategy.

    Attributes
    ----------
    strategy_id:
        Unique identifier string used as the file stem when persisted.
        Example: ``'skill_prob_edge_top50'``.
    name:
        Human-readable display name.
    selection_method:
        Which selection family was used.
        One of: ``'skill_sweep'``, ``'volatility'``, ``'cohort_early_entry'``,
        ``'cohort_smooth_pnl'``, ``'manual'``.
    trigger:
        :class:`TriggerSpec` describing how to convert a signal row into a
        trade decision.
    wallets:
        DataFrame output of the selection step.  Must contain at least a
        ``wallet`` column and a ``wallet_quality`` column (0–1 percentile rank).
        May contain additional metric columns for interpretability.
    params:
        Selection-step parameters (metric name, top-N, eligibility thresholds,
        etc.).  Stored verbatim in the JSON sidecar.
    metadata:
        Arbitrary key-value pairs for provenance (train window dates, sweep
        results, calibration threshold, etc.).
    created_at:
        UTC creation timestamp.  Set automatically.
    """

    strategy_id: str
    name: str
    selection_method: str
    trigger: TriggerSpec
    wallets: pd.DataFrame
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
    )
