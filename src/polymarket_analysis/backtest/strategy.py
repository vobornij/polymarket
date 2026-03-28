"""
Backtest strategy execution module.

Provides :func:`backtest_strategy`, :func:`summarize_backtest`,
:func:`build_trigger_ledger`, and :func:`compute_simple_kelly_fraction`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from polymarket_analysis.backtest.execution_tape import attach_forward_fills


# ---------------------------------------------------------------------------
# Kelly sizing
# ---------------------------------------------------------------------------

#: Conservative Kelly scale applied to full Kelly fraction.
KELLY_SCALE: float = 0.25

#: Hard cap on Kelly fraction (fraction of starting capital per trade).
KELLY_MAX_FRACTION: float = 0.10

#: Minimum stake allowed when Kelly sizing is active (USDC).
KELLY_MIN_STAKE_USDC: float = 25.0

#: Maximum stake allowed when Kelly sizing is active (USDC).
KELLY_MAX_STAKE_USDC: float = 750.0


def compute_simple_kelly_fraction(
    trades: pd.DataFrame,
    *,
    kelly_scale: float = KELLY_SCALE,
    max_fraction: float = KELLY_MAX_FRACTION,
) -> pd.Series:
    """Compute a scaled Kelly fraction for each trigger trade.

    The estimate uses signal score as a proxy for win probability:

        p_hat = clip(0.5 + 0.35 * (score - 0.5), 0.05, 0.95)
        b     = (1 - price) / price
        f*    = (b * p_hat - (1 - p_hat)) / b
        result = clip(f* * kelly_scale, 0, max_fraction)

    Parameters
    ----------
    trades:
        DataFrame with columns ``signal_score`` and ``price``.
    kelly_scale:
        Fraction of full Kelly to use.
    max_fraction:
        Upper cap on the returned fraction.

    Returns
    -------
    pd.Series aligned to ``trades.index``.
    """
    if trades.empty:
        return pd.Series(dtype=float)
    score = trades["signal_score"].fillna(0.5).clip(0.0, 1.0)
    p_hat = (0.5 + 0.35 * (score - 0.5)).clip(0.05, 0.95)
    price = trades["price"].clip(lower=0.01, upper=0.99)
    b = (1.0 - price) / price
    full_kelly = (
        ((b * p_hat - (1.0 - p_hat)) / b)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return (full_kelly * kelly_scale).clip(lower=0.0, upper=max_fraction)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def backtest_strategy(
    signals: pd.DataFrame,
    mask: pd.Series,
    tape_groups: dict,
    strategy_name: str,
    trigger_rule: str,
    *,
    base_stake_usdc: float = 100.0,
    dynamic_sizing: bool = False,
    latency_seconds: float = 0,
    fill_horizon_seconds: float = 600,
    slippage_bps: float = 50.0,
    fee_bps: float = 10.0,
    max_signals_per_day: int | None = 20,
    dedupe_by_market: bool = True,
    starting_capital: float = 10_000.0,
    cohort_name: str = "default",
    min_stake_usdc: float = KELLY_MIN_STAKE_USDC,
    max_stake_usdc: float = KELLY_MAX_STAKE_USDC,
    max_rel_price_diff_by_bucket: dict | None = None,
    max_price_premium: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a copy-trade backtest for one strategy specification.

    Parameters
    ----------
    signals:
        Scored signal events (output of :func:`apply_signal_score`).
    mask:
        Boolean Series with same index as *signals* selecting which rows to
        trade.
    tape_groups:
        Output of :func:`build_tape_groups`.
    strategy_name:
        Human-readable label stored in result columns.
    trigger_rule:
        Short description of the entry rule stored in result columns.
    base_stake_usdc:
        Fixed stake per trade when ``dynamic_sizing=False``.
    dynamic_sizing:
        If ``True``, use Kelly-based sizing; otherwise fixed.
    latency_seconds:
        Seconds added to trigger ``dt`` before searching the tape.
    fill_horizon_seconds:
        Maximum seconds after ``dt + latency`` to search for fills.
    slippage_bps:
        Execution slippage in basis points.
    fee_bps:
        Trading fee in basis points, applied to gross PnL.
    max_signals_per_day:
        Cap daily signal count.  ``None`` = no cap.
    dedupe_by_market:
        Drop duplicate ``market_key`` entries keeping the first trigger.
    starting_capital:
        Initial capital for equity curve and Kelly sizing denominator.
    cohort_name:
        Label stored in result columns.
    min_stake_usdc, max_stake_usdc:
        Kelly stake bounds.
    max_rel_price_diff_by_bucket:
        Per-price-bucket relative price cap passed to
        :func:`attach_forward_fills`.
    max_price_premium:
        Absolute price premium cap passed to :func:`attach_forward_fills`.

    Returns
    -------
    (filled_trades, daily, unfilled_triggers, theoretical_daily)
        Each is a ``pd.DataFrame``.  ``daily`` and ``theoretical_daily``
        have columns ``trade_date``, ``net_pnl_usdc``, ``cum_net_pnl_usdc``,
        ``equity_usdc``.
    """
    _empty_daily = pd.DataFrame(
        columns=["trade_date", "trades", "net_pnl_usdc", "cum_net_pnl_usdc", "equity_usdc"]
    )

    trades = signals[mask].copy().sort_values("dt")
    trades["strategy"] = strategy_name
    trades["trigger_rule"] = trigger_rule
    trades["cohort"] = cohort_name
    trades["sizing_mode"] = "kelly_simple" if dynamic_sizing else "fixed"

    # ── Normalise copy-trade direction columns ───────────────────────────────
    # For sell-event signals, copy_price/copy_market_key/copy_token_winner were
    # set by build_signal_events to point at the *opposite* token.  We overwrite
    # market_key / price / token_winner with those values so that the rest of
    # the backtest pipeline (fill simulation, PnL) acts on the right token.
    # Rows where copy_* columns are absent or NaN keep their original values.
    if "copy_market_key" in trades.columns:
        trades["market_key"] = trades["copy_market_key"].where(
            trades["copy_market_key"].notna(), trades["market_key"]
        )
    if "copy_price" in trades.columns:
        trades["price"] = trades["copy_price"].where(
            trades["copy_price"].notna(), trades["price"]
        )
    if "copy_token_winner" in trades.columns:
        trades["token_winner"] = trades["copy_token_winner"].where(
            trades["copy_token_winner"].notna(), trades["token_winner"]
        )

    if dedupe_by_market:
        trades = trades.drop_duplicates("market_key", keep="first")

    trades["trade_date"] = trades["dt"].dt.floor("D")

    if max_signals_per_day is not None:
        trades["daily_rank"] = trades.groupby("trade_date").cumcount() + 1
        trades = trades[trades["daily_rank"] <= max_signals_per_day].copy()

    if trades.empty:
        return trades, _empty_daily, trades, _empty_daily.copy()

    trades = trades.reset_index(drop=True)
    trades["trigger_id"] = np.arange(len(trades), dtype=int)

    # ── sizing ──────────────────────────────────────────────────────────────
    if dynamic_sizing:
        trades["kelly_fraction"] = compute_simple_kelly_fraction(trades)
        raw_stake = starting_capital * trades["kelly_fraction"]
        trades["stake_usdc"] = np.where(
            trades["kelly_fraction"] > 1e-12,
            np.clip(raw_stake, min_stake_usdc, max_stake_usdc),
            0.0,
        )
        trades = trades[trades["stake_usdc"] > 0.0].copy()
    else:
        trades["kelly_fraction"] = np.nan
        trades["stake_usdc"] = float(base_stake_usdc)

    if trades.empty:
        return trades, _empty_daily, trades, _empty_daily.copy()

    # ── theoretical (trigger-price) daily PnL ───────────────────────────────
    fee = fee_bps / 10_000.0
    candidate_triggers = trades.copy()
    candidate_triggers["trigger_gross_roi"] = np.where(
        candidate_triggers["token_winner"],
        1.0 / candidate_triggers["price"].clip(lower=0.001) - 1.0,
        -1.0,
    )
    candidate_triggers["trigger_net_roi"] = candidate_triggers["trigger_gross_roi"] - fee
    if "trade_pnl" in candidate_triggers.columns:
        candidate_triggers["trigger_net_pnl_usdc"] = candidate_triggers["trade_pnl"].astype(float)
    else:
        candidate_triggers["trigger_net_pnl_usdc"] = (
            candidate_triggers["stake_usdc"] * candidate_triggers["trigger_net_roi"]
        )

    theoretical_daily = (
        candidate_triggers.groupby("trade_date")
        .agg(trades=("market_key", "size"), net_pnl_usdc=("trigger_net_pnl_usdc", "sum"))
        .reset_index()
        .sort_values("trade_date")
    )
    theoretical_daily["cum_net_pnl_usdc"] = theoretical_daily["net_pnl_usdc"].cumsum()
    theoretical_daily["equity_usdc"] = starting_capital + theoretical_daily["cum_net_pnl_usdc"]

    # ── fill simulation ──────────────────────────────────────────────────────
    trades = attach_forward_fills(
        trades,
        tape_groups=tape_groups,
        latency_seconds=latency_seconds,
        fill_horizon_seconds=fill_horizon_seconds,
        slippage_bps=slippage_bps,
        min_fill_ratio=1.0,
        max_price_premium=max_price_premium,
        max_rel_price_diff_by_bucket=max_rel_price_diff_by_bucket,
    )

    if trades.empty:
        unfilled = candidate_triggers.copy()
        unfilled["fill_status"] = "no_fill"
        return trades, _empty_daily, unfilled, theoretical_daily

    filled_ids = set(trades["trigger_id"].astype(int).tolist())
    unfilled = candidate_triggers[~candidate_triggers["trigger_id"].isin(filled_ids)].copy()
    if not unfilled.empty:
        unfilled["fill_status"] = "no_fill"

    # ── realised PnL ─────────────────────────────────────────────────────────
    trades["gross_roi"] = np.where(
        trades["token_winner"], 1.0 / trades["exec_price"] - 1.0, -1.0
    )
    trades["net_roi"] = trades["gross_roi"] - fee
    trades["net_pnl_usdc"] = trades["filled_usdc"] * trades["net_roi"]
    trades["fill_delay_minutes"] = (
        (trades["tape_dt"] - trades["dt"]).dt.total_seconds() / 60.0
    )

    daily = (
        trades.groupby("trade_date")
        .agg(trades=("market_key", "size"), net_pnl_usdc=("net_pnl_usdc", "sum"))
        .reset_index()
        .sort_values("trade_date")
    )
    daily["cum_net_pnl_usdc"] = daily["net_pnl_usdc"].cumsum()
    daily["equity_usdc"] = starting_capital + daily["cum_net_pnl_usdc"]

    return trades, daily, unfilled, theoretical_daily


# ---------------------------------------------------------------------------
# Summary and ledger helpers
# ---------------------------------------------------------------------------

def summarize_backtest(
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    starting_capital: float = 10_000.0,
) -> pd.Series:
    """Compute scalar performance metrics from a completed backtest run.

    Parameters
    ----------
    trades:
        Filled trades DataFrame (first return value of
        :func:`backtest_strategy`).
    daily:
        Daily PnL DataFrame (second return value of
        :func:`backtest_strategy`).
    starting_capital:
        Used only for the equity-based drawdown calculation.

    Returns
    -------
    pd.Series of scalar metrics.
    """
    if trades.empty:
        return pd.Series({
            "filled_trades": 0,
            "net_pnl_usdc": 0.0,
            "net_roi_on_stake": np.nan,
            "win_rate": np.nan,
        })

    total_stake = float(trades["filled_usdc"].sum())
    max_drawdown = 0.0
    if not daily.empty:
        running_peak = daily["equity_usdc"].cummax()
        max_drawdown = float((running_peak - daily["equity_usdc"]).max())

    return pd.Series({
        "filled_trades": int(len(trades)),
        "net_pnl_usdc": float(trades["net_pnl_usdc"].sum()),
        "net_roi_on_stake": (
            float(trades["net_pnl_usdc"].sum() / total_stake)
            if total_stake > 0
            else np.nan
        ),
        "win_rate": float(trades["token_winner"].mean()),
        "avg_signal_score": (
            float(trades["signal_score"].mean())
            if "signal_score" in trades.columns
            else np.nan
        ),
        "avg_kelly_fraction": (
            float(trades["kelly_fraction"].mean())
            if "kelly_fraction" in trades.columns
            else np.nan
        ),
        "avg_fill_delay_minutes": float(trades["fill_delay_minutes"].mean()),
        "opposite_fill_share": float(trades["opposite_fill_share"].mean()),
        "max_drawdown_usdc": max_drawdown,
    })


def build_trigger_ledger(
    filled_trades: pd.DataFrame | None,
    unfilled_triggers: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine filled and unfilled triggers into a unified audit ledger.

    Parameters
    ----------
    filled_trades:
        First return value of :func:`backtest_strategy` (may be empty /
        ``None``).
    unfilled_triggers:
        Third return value of :func:`backtest_strategy` (may be empty /
        ``None``).

    Returns
    -------
    pd.DataFrame with ``fill_status`` column (``'filled'`` or ``'no_fill'``)
    and all columns present in either input.  Rows are sorted by ``dt``,
    ``strategy``, ``trigger_id``.
    """
    ledger_columns = [
        "cohort", "strategy", "trigger_rule", "sizing_mode",
        "fill_status", "trigger_id",
        "dt", "trade_date", "tape_dt", "fill_delay_minutes",
        "trigger_tx_hash", "fill_tx_hash",
        "wallet", "condition_id", "outcome", "market_key",
        "price", "quantity", "exec_price", "stake_usdc", "filled_usdc",
        "fill_ratio", "kelly_fraction",
        "wallet_quality", "signal_score",
        "wallet_component", "conviction_component",
        "price_component", "consensus_component",
        "conviction_ratio",
        "prior_same_any", "prior_opp_any", "prior_same_24h", "prior_opp_24h",
        "price_bucket",
        "token_winner", "prob_edge", "gross_roi", "net_roi", "net_pnl_usdc",
        "trigger_gross_roi", "trigger_net_roi", "trigger_net_pnl_usdc",
    ]

    parts: list[pd.DataFrame] = []

    if filled_trades is not None and not filled_trades.empty:
        filled = filled_trades.copy()
        filled["fill_status"] = "filled"
        parts.append(filled)

    if unfilled_triggers is not None and not unfilled_triggers.empty:
        unfilled = unfilled_triggers.copy()
        unfilled["fill_status"] = "no_fill"
        for col, default in [
            ("filled_usdc", 0.0),
            ("fill_ratio", 0.0),
            ("exec_price", np.nan),
            ("fill_tx_hash", np.nan),
            ("tape_dt", pd.NaT),
            ("fill_delay_minutes", np.nan),
        ]:
            if col not in unfilled.columns:
                unfilled[col] = default
        parts.append(unfilled)

    if not parts:
        return pd.DataFrame(columns=ledger_columns)

    ledger = pd.concat(parts, ignore_index=True, sort=False)
    for col in ["trigger_tx_hash", "fill_tx_hash"]:
        if col not in ledger.columns:
            ledger[col] = np.nan

    available = [c for c in ledger_columns if c in ledger.columns]
    return (
        ledger[available]
        .sort_values(["dt", "strategy", "trigger_id"])
        .reset_index(drop=True)
    )
