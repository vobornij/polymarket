"""
Wallet cohort selection functions for the signal-v2 pipeline.

Provides:

* :func:`select_wallets`              — skill-metric selector (quality_core cohort)
* :func:`cohort_selection_sweep`      — grid search over metrics × top-N
* :func:`build_wallet_cohorts`        — construct all named cohorts
* :func:`build_strategies_from_sweep` — factory: sweep + cohorts → list of
                                        :class:`~strategy.definition.WalletSelectionStrategy`
* :func:`_with_wallet_quality`        — internal helper used by :func:`build_wallet_cohorts`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Avoid circular import at runtime; only used for type annotations.
    from polymarket_analysis.strategy.definition import WalletSelectionStrategy


# ---------------------------------------------------------------------------
# Skill-metric selector
# ---------------------------------------------------------------------------

def select_wallets(
    metric_df: pd.DataFrame,
    metric_name: str,
    top_n: int,
    min_open_buys: int = 20,
    min_volume: float = 500.0,
    min_markets: int = 8,
    min_recent_trades: int = 3,
) -> pd.DataFrame:
    """Select the top-N wallets by *metric_name* after applying eligibility filters.

    Parameters
    ----------
    metric_df:
        Full-train metrics DataFrame (output of
        :func:`~wallet_selection.metrics.compute_wallet_skill_workspace`).
    metric_name:
        Column to rank by (e.g. ``'prob_edge_shrunk'``).
    top_n:
        Maximum number of wallets to return.
    min_open_buys:
        Minimum number of ``open_buy`` events in the full training period.
    min_volume:
        Minimum total USDC notional in the full training period.
    min_markets:
        Minimum number of distinct markets traded (``distinct_markets`` column,
        if present).
    min_recent_trades:
        Minimum ``recent_open_buy_trades`` (if present).

    Returns
    -------
    DataFrame with at most *top_n* rows, sorted descending by *metric_name*,
    with an added ``wallet_quality`` column (percentile rank 0–1).
    """
    distinct_markets = (
        metric_df["distinct_markets"]
        if "distinct_markets" in metric_df.columns
        else pd.Series(999_999, index=metric_df.index)
    )
    recent_trades = (
        metric_df["recent_open_buy_trades"]
        if "recent_open_buy_trades" in metric_df.columns
        else pd.Series(999_999, index=metric_df.index)
    )

    eligible = metric_df[
        (metric_df["open_buy_trades"] >= min_open_buys)
        & (metric_df["volume"] >= min_volume)
        & (distinct_markets >= min_markets)
        & (recent_trades >= min_recent_trades)
    ].copy()

    selected = (
        eligible.sort_values(metric_name, ascending=False)
        .dropna(subset=[metric_name])
        .head(top_n)
        .reset_index(drop=True)
    )

    if not selected.empty:
        selected["wallet_quality"] = selected[metric_name].rank(
            method="first", pct=True
        )

    return selected


# ---------------------------------------------------------------------------
# Cohort sweep
# ---------------------------------------------------------------------------

CANDIDATE_METRICS = [
    "prob_edge_shrunk",
    "weighted_prob_edge_shrunk",
    "avg_copy_roi_capped",
    "edge_sharpe",
    "roi_sharpe",
    "brier_skill",
    "hit_rate",
]


def cohort_selection_sweep(
    train_a_df: pd.DataFrame,
    train_b_df: pd.DataFrame,
    metrics: list[str] | None = None,
    top_ns: tuple[int, ...] = (50, 100, 200, 300, 500),
) -> pd.DataFrame:
    """Evaluate all (metric × top_n) combinations using train-a → train-b persistence.

    Wallets are selected from *train_a_df* and their aggregated performance on
    *train_b_df* is measured.  This is the meta-selection step that avoids
    lookahead bias.

    Parameters
    ----------
    train_a_df, train_b_df:
        Metric DataFrames for the two sub-periods.
    metrics:
        Metrics to sweep.  Defaults to :data:`CANDIDATE_METRICS`.
    top_ns:
        Candidate cohort sizes.

    Returns
    -------
    DataFrame with one row per (metric, top_n) combination.
    """
    if metrics is None:
        metrics = CANDIDATE_METRICS

    eligible_a = train_a_df[
        (train_a_df["open_buy_trades"] >= 10) & (train_a_df["volume"] >= 250)
    ].copy()

    rows = []
    for metric in metrics:
        ranked = (
            eligible_a.sort_values(metric, ascending=False)
            .dropna(subset=[metric])
        )
        for top_n in top_ns:
            selected = set(ranked.head(top_n)["wallet"])
            cohort = train_b_df[train_b_df["wallet"].isin(selected)].copy()
            if cohort.empty:
                continue
            trades = cohort["open_buy_trades"].sum()
            volume = cohort["volume"].sum()
            rows.append(
                {
                    "metric": metric,
                    "top_n": top_n,
                    "wallets_found_in_train_b": cohort["wallet"].nunique(),
                    "train_b_open_buy_trades": trades,
                    "train_b_weighted_prob_edge": (
                        cohort["sum_weighted_edge_num"].sum() / volume
                        if volume else np.nan
                    ),
                    "train_b_avg_prob_edge": (
                        cohort["sum_prob_edge"].sum() / trades if trades else np.nan
                    ),
                    "train_b_avg_copy_roi_capped": (
                        cohort["sum_copy_roi_capped"].sum() / trades if trades else np.nan
                    ),
                    "train_b_total_copy_pnl_usdc": cohort["total_copy_pnl_usdc"].sum(),
                    "train_b_hit_rate": (
                        cohort["wins"].sum() / trades if trades else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _with_wallet_quality(
    df: pd.DataFrame,
    score_col: str,
    ascending: bool = False,
    top_n: int = 100,
) -> pd.DataFrame:
    """Rank wallets by *score_col* and return (wallet, wallet_quality) pairs.

    *wallet_quality* is a percentile rank (0–1) with higher = better.
    """
    ranked = (
        df.sort_values(score_col, ascending=ascending)
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )
    if ranked.empty:
        ranked["wallet_quality"] = pd.Series(dtype=float)
        return ranked[["wallet", "wallet_quality"]]

    rank_values = ranked[score_col].rank(method="first", ascending=ascending, pct=True)
    ranked["wallet_quality"] = rank_values
    return ranked[["wallet", "wallet_quality"]]


# ---------------------------------------------------------------------------
# Named cohort builder
# ---------------------------------------------------------------------------

def build_wallet_cohorts(
    full_train_metrics: pd.DataFrame,
    train_b_open_buys: pd.DataFrame,
    base_selected_wallets: pd.DataFrame,
    top_n: int = 100,
) -> dict[str, pd.DataFrame]:
    """Build a dictionary of named wallet cohorts.

    Three cohorts are constructed:

    * ``quality_core`` — the base selected wallets (output of
      :func:`select_wallets`), used as-is.
    * ``early_entry`` — wallets that tend to enter markets early, scored by
      combining shrunk edge with a penalty for late median entry time.
    * ``smooth_pnl`` — wallets with high PnL relative to their copy-ROI
      standard deviation (low-variance PnL generators).

    Parameters
    ----------
    full_train_metrics:
        Full-train metric DataFrame.
    train_b_open_buys:
        ``open_buy`` signal events for the train-b period (used to compute
        median entry time).
    base_selected_wallets:
        Output of :func:`select_wallets`.
    top_n:
        Maximum cohort size for the derived cohorts.

    Returns
    -------
    ``{cohort_name → DataFrame(wallet, wallet_quality)}``
    """
    top_n = int(top_n)
    cohorts: dict[str, pd.DataFrame] = {}

    # --- quality_core: pass through base selection ---
    core = base_selected_wallets[["wallet", "wallet_quality"]].copy().reset_index(drop=True)
    cohorts["quality_core"] = core

    # --- common eligibility filter ---
    eligible = full_train_metrics[
        (full_train_metrics["open_buy_trades"] >= 20)
        & (full_train_metrics["volume"] >= 500.0)
    ][["wallet", "prob_edge_shrunk", "total_copy_pnl_usdc", "copy_roi_std"]].copy()

    # --- early_entry cohort ---
    early_stats = (
        train_b_open_buys.groupby("wallet")
        .agg(
            median_hours_since_first=("hours_since_first_selected_trade", "median"),
            train_b_open_buys=("wallet", "size"),
        )
        .reset_index()
    )
    early = eligible.merge(early_stats, on="wallet", how="left")
    early = early[
        (early["train_b_open_buys"] >= 5) & early["median_hours_since_first"].notna()
    ].copy()
    early["early_score"] = (
        early["prob_edge_shrunk"].fillna(0.0)
        - 0.03 * early["median_hours_since_first"]
    )
    cohorts["early_entry"] = _with_wallet_quality(
        early[["wallet", "early_score"]], "early_score", ascending=False, top_n=top_n
    )

    # --- smooth_pnl cohort ---
    smooth = eligible.copy()
    smooth["copy_roi_std"] = smooth["copy_roi_std"].fillna(
        smooth["copy_roi_std"].median()
    )
    smooth = smooth[smooth["total_copy_pnl_usdc"] > 0].copy()
    smooth["smooth_score"] = smooth["total_copy_pnl_usdc"] / smooth[
        "copy_roi_std"
    ].clip(lower=0.02)
    cohorts["smooth_pnl"] = _with_wallet_quality(
        smooth[["wallet", "smooth_score"]], "smooth_score", ascending=False, top_n=top_n
    )

    return {
        name: df.drop_duplicates("wallet").reset_index(drop=True)
        for name, df in cohorts.items()
        if not df.empty
    }


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def build_strategies_from_sweep(
    wallet_cohorts: dict[str, pd.DataFrame],
    signal_threshold: float,
    selection_metric: str,
    top_n: int,
    sweep_df: pd.DataFrame | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> "list[WalletSelectionStrategy]":
    """Build a :class:`~strategy.definition.WalletSelectionStrategy` per cohort.

    This is the bridge between the wallet-selection stage and the backtest
    stage.  It converts cohort DataFrames + calibration parameters into a
    list of fully-described strategy objects that the backtest can load and
    iterate over.

    Two trigger variants are created for each cohort:

    * ``{cohort}_score_threshold`` — ``signal_score >= signal_threshold``
      with Kelly dynamic sizing.
    * ``{cohort}_all_open_buys``   — all open-buy events with fixed sizing.

    Parameters
    ----------
    wallet_cohorts:
        Output of :func:`build_wallet_cohorts`.
    signal_threshold:
        Calibrated score threshold (output of
        :func:`~signal.scorer.select_signal_threshold`).
    selection_metric:
        The wallet ranking metric chosen by the sweep (e.g.
        ``'prob_edge_shrunk'``).
    top_n:
        Cohort size selected by the sweep.
    sweep_df:
        Optional sweep results DataFrame (stored in metadata for provenance).
    extra_metadata:
        Any additional key-value pairs to store in every strategy's
        ``metadata`` dict.

    Returns
    -------
    List of :class:`~strategy.definition.WalletSelectionStrategy` objects,
    one per (cohort × trigger_variant) combination.
    """
    # Deferred import to avoid circular dependency at module load time.
    from polymarket_analysis.strategy.definition import TriggerSpec, WalletSelectionStrategy

    base_meta: dict[str, Any] = {
        "selection_metric": selection_metric,
        "top_n": top_n,
        "signal_threshold": signal_threshold,
    }
    if sweep_df is not None and not sweep_df.empty:
        best_row = sweep_df.sort_values(
            ["train_b_avg_copy_roi_capped", "train_b_weighted_prob_edge"],
            ascending=False,
        ).iloc[0].to_dict()
        base_meta["sweep_best_row"] = {k: (None if (isinstance(v, float) and v != v) else v)
                                        for k, v in best_row.items()}
    if extra_metadata:
        base_meta.update(extra_metadata)

    strategies: list[WalletSelectionStrategy] = []

    # Map cohort name → selection method label
    selection_method_map = {
        "quality_core": "skill_sweep",
        "early_entry": "cohort_early_entry",
        "smooth_pnl": "cohort_smooth_pnl",
    }

    for cohort_name, cohort_wallets in wallet_cohorts.items():
        if cohort_wallets.empty:
            continue

        selection_method = selection_method_map.get(cohort_name, cohort_name)
        cohort_meta = dict(base_meta)
        cohort_meta["cohort"] = cohort_name

        # ── variant 1: score threshold + dynamic sizing ──────────────────
        strategies.append(
            WalletSelectionStrategy(
                strategy_id=f"{cohort_name}__score_threshold",
                name=f"{cohort_name} | score >= {signal_threshold:.2f} (Kelly)",
                selection_method=selection_method,
                trigger=TriggerSpec(
                    fn_ref="polymarket_analysis.strategy.triggers.score_threshold",
                    params={
                        "threshold": signal_threshold,
                        "dynamic_sizing": True,
                    },
                    mode="frame",
                ),
                wallets=cohort_wallets,
                params={
                    "selection_metric": selection_metric,
                    "top_n": top_n,
                    "signal_threshold": signal_threshold,
                },
                metadata=cohort_meta,
            )
        )

        # ── variant 2: all open-buys + fixed sizing ───────────────────────
        strategies.append(
            WalletSelectionStrategy(
                strategy_id=f"{cohort_name}__all_open_buys",
                name=f"{cohort_name} | all open-buys (fixed stake)",
                selection_method=selection_method,
                trigger=TriggerSpec(
                    fn_ref="polymarket_analysis.strategy.triggers.all_open_buys",
                    params={"dynamic_sizing": False},
                    mode="frame",
                ),
                wallets=cohort_wallets,
                params={
                    "selection_metric": selection_metric,
                    "top_n": top_n,
                    "signal_threshold": None,
                },
                metadata=cohort_meta,
            )
        )

    return strategies
