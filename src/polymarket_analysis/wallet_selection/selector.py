"""
Wallet cohort selection functions for the signal-v2 pipeline.

Provides:

* :func:`select_wallets`           — skill-metric selector (quality_core cohort)
* :func:`cohort_selection_sweep`   — grid search over metrics × top-N
* :func:`build_wallet_cohorts`     — construct all named cohorts
* :func:`_with_wallet_quality`     — internal helper used by :func:`build_wallet_cohorts`
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
