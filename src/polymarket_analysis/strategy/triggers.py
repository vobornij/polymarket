"""
Trigger functions for the copy-trade backtest.

Each trigger function is a **frame-mode** function:

    fn(signals: pd.DataFrame, params: dict) -> pd.Series[bool]

It receives the full scored signal DataFrame (output of
:func:`~signal.scorer.apply_signal_score`) and a params dict, and returns a
boolean mask of rows that should become trades.

This design is:
* Fast — operations on whole DataFrame columns, no row-by-row Python loops.
* Composable — triggers can combine multiple conditions.
* Serialisable — stored by importable reference string.

Naming convention
-----------------
Public functions use lower_snake_case names that clearly describe the rule.
The canonical import prefix is
``polymarket_analysis.strategy.triggers.<fn_name>``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Score-based triggers
# ---------------------------------------------------------------------------

def score_threshold(
    signals: pd.DataFrame,
    params: dict[str, Any],
) -> pd.Series:
    """Trigger when ``signal_score >= threshold``.

    Parameters
    ----------
    signals:
        Scored signal events DataFrame (must contain ``signal_score``).
    params:
        ``threshold`` (float, default 0.80) — minimum score to trigger.

    Returns
    -------
    Boolean Series aligned to ``signals.index``.
    """
    threshold = float(params.get("threshold", 0.80))
    return signals["signal_score"] >= threshold


def all_open_buys(
    signals: pd.DataFrame,
    params: dict[str, Any],
) -> pd.Series:
    """Trigger on every open-buy event for selected wallets (no score filter).

    Parameters
    ----------
    signals:
        Signal events DataFrame.  Must contain a ``wallet`` column.
    params:
        Unused.  Kept for uniform interface.

    Returns
    -------
    Boolean Series — ``True`` for every row that has a non-null ``wallet``.
    """
    return signals["wallet"].notna()


def score_and_price_bucket(
    signals: pd.DataFrame,
    params: dict[str, Any],
) -> pd.Series:
    """Trigger when score threshold is met AND price bucket is allowed.

    Parameters
    ----------
    signals:
        Scored signal events DataFrame (must contain ``signal_score`` and
        ``price_bucket``).
    params:
        ``threshold`` (float, default 0.80) — minimum score to trigger.
        ``allowed_buckets`` (list[str] | None) — if given, only rows whose
        ``price_bucket`` is in this list pass.

    Returns
    -------
    Boolean Series.
    """
    threshold = float(params.get("threshold", 0.80))
    allowed = params.get("allowed_buckets")

    mask = signals["signal_score"] >= threshold
    if allowed is not None:
        mask = mask & signals["price_bucket"].isin(allowed)
    return mask


def score_and_consensus(
    signals: pd.DataFrame,
    params: dict[str, Any],
) -> pd.Series:
    """Trigger when score threshold is met AND min prior-same wallets agree.

    Parameters
    ----------
    signals:
        Scored signal events DataFrame (must contain ``signal_score`` and
        ``prior_same_any``).
    params:
        ``threshold`` (float, default 0.80) — minimum score to trigger.
        ``min_prior_same`` (int, default 1) — minimum count of prior wallets
        that bought the same outcome.

    Returns
    -------
    Boolean Series.
    """
    threshold = float(params.get("threshold", 0.80))
    min_prior_same = int(params.get("min_prior_same", 1))

    mask = (signals["signal_score"] >= threshold) & (
        signals["prior_same_any"] >= min_prior_same
    )
    return mask


def wallet_quality_threshold(
    signals: pd.DataFrame,
    params: dict[str, Any],
) -> pd.Series:
    """Trigger based on wallet quality alone (no composite score required).

    Useful as a baseline for comparing composite scoring vs. raw selection
    quality.

    Parameters
    ----------
    signals:
        Signal events DataFrame.  Must contain ``wallet_quality``.
    params:
        ``min_wallet_quality`` (float, default 0.70) — minimum wallet_quality
        percentile to trigger.

    Returns
    -------
    Boolean Series.
    """
    min_q = float(params.get("min_wallet_quality", 0.70))
    return signals["wallet_quality"] >= min_q


def copy_triggers(
    signals: pd.DataFrame,
    params: dict[str, Any],
) -> pd.Series:
    """Trigger on *every* trade the watched wallets make (open, add, close, reduce).

    Unlike :func:`all_open_buys` which fires only on position-opening buys,
    this trigger fires on all event types so that the backtest can copy the
    full trading behaviour of the watched cohort — including add-to-position
    buys and, for sell events, buying the *opposite* token.

    The slippage tolerance is intentionally kept tight: pass
    ``slippage_bps`` and ``max_rel_price_diff_by_bucket`` in the
    ``BACKTEST_KWARGS`` override to enforce strict fills.

    Parameters
    ----------
    signals:
        Signal events DataFrame.  Must contain ``wallet`` and ``event_type``.
        ``event_type`` values: ``'open_buy'``, ``'add_buy'``,
        ``'close_sell'``, ``'reduce_sell'``.
    params:
        ``allowed_event_types`` (list[str] | None, default None) — when
        provided, only rows whose ``event_type`` is in this list pass.
        ``None`` means all event types pass.

    Returns
    -------
    Boolean Series — ``True`` for every row that has a non-null ``wallet``
    and (optionally) an allowed ``event_type``.
    """
    allowed = params.get("allowed_event_types")
    mask = signals["wallet"].notna()
    if allowed is not None:
        mask = mask & signals["event_type"].isin(allowed)
    return mask


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

_TRIGGER_REGISTRY: dict[str, object] = {
    "score_threshold": score_threshold,
    "all_open_buys": all_open_buys,
    "score_and_price_bucket": score_and_price_bucket,
    "score_and_consensus": score_and_consensus,
    "wallet_quality_threshold": wallet_quality_threshold,
    "copy_triggers": copy_triggers,
}


def get_trigger(fn_ref: str):
    """Resolve a trigger function by its fully-qualified reference string.

    Supports two formats:

    * Short name: ``'score_threshold'`` — looks up in the built-in registry.
    * Full dotted path: ``'polymarket_analysis.strategy.triggers.score_threshold'``
      — imports at runtime.

    Parameters
    ----------
    fn_ref:
        Function reference string.

    Returns
    -------
    Callable ``(signals, params) -> pd.Series``.

    Raises
    ------
    ValueError
        If the reference cannot be resolved.
    """
    # Short name lookup
    if fn_ref in _TRIGGER_REGISTRY:
        return _TRIGGER_REGISTRY[fn_ref]

    # Dotted import path
    if "." in fn_ref:
        parts = fn_ref.rsplit(".", maxsplit=1)
        module_path, fn_name = parts[0], parts[1]
        try:
            import importlib

            module = importlib.import_module(module_path)
            fn = getattr(module, fn_name, None)
            if fn is not None and callable(fn):
                return fn
        except ImportError:
            pass

    raise ValueError(
        f"Cannot resolve trigger function {fn_ref!r}. "
        "Register it in _TRIGGER_REGISTRY or use a fully-qualified dotted path."
    )
