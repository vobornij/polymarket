"""
Copy-trade research strategy layer.

This module provides the data contracts and utilities for designing and
comparing multiple wallet-selection + trigger strategies in a systematic way.

Core concepts
-------------
WalletSelectionStrategy
    A named, persisted artifact produced by the wallet-selection stage.
    It bundles:

    * ``wallets``       — selected wallet DataFrame (wallet + wallet_quality + metrics)
    * ``trigger_fn_ref`` — importable path to the trigger function
    * ``params``        — trigger parameters (threshold, sizing flags, etc.)
    * ``metadata``      — provenance (selection method, train window, sweep results)

TriggerSpec
    Lightweight descriptor of a trigger rule: function reference + params + mode.

Registry helpers
    :func:`save_strategy`, :func:`load_strategy`, :func:`load_all_strategies`
    persist/restore the full strategy set from a workspace directory.

Trigger library
    :mod:`~strategy.triggers` — canonical trigger functions used by the backtest.
"""

from .definition import WalletSelectionStrategy, TriggerSpec
from .registry import save_strategy, load_strategy, load_all_strategies, strategy_exists
from . import triggers

__all__ = [
    "WalletSelectionStrategy",
    "TriggerSpec",
    "save_strategy",
    "load_strategy",
    "load_all_strategies",
    "strategy_exists",
    "triggers",
]
