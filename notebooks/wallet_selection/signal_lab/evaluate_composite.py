"""
Combine confirmed signal families into composite scores and threshold them.

Runs all confirmed strategies on a shared candidate universe, deduplicates
redundant signal families (rank-corr greedy selection), builds equal /
IC-weighted / shrinkage-Markowitz composites with train-fit weights, and
evaluates a threshold grid on validation (selection) then test (report).

The composite is fit to a configurable target (default ``copyable_pnl``: the
dollar PnL the copy strategy actually earns).  Because raw ``copyable_roi`` is
~50% correlated with ``price``, the script also reports price-controlled views
(price-residualized pnl IC and within-price-bin IC) so the raw edge is not
confused with "buying cheap".
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

_NOTEBOOK_DIR = Path(__file__).resolve().parent.parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

from signal_lab.stage1 import (
    build_composite_scores,
    evaluate_signal_panel,
    evaluate_threshold_grid,
    load_stage1_data,
    run_strategies,
)
from signal_lab.signal_lib import compute_event_ic, spearman_rho
from signal_lab.sizing import capital_constrained_sim, select_scale
from signal_lab.strategies import (
    CopyCrowdEntryTiming,
    FadeReactiveSellFlow,
    FreshOppositeCrowdingFilter,
    GamblerCapitulationSqueeze,
    UwlOppContrarian,
)

TARGETS = ["roi_res", "copyable_roi", "copyable_pnl"]


def _pooled_rank_corr(splits, cols):
    """Rank correlation of all cols on train+val pooled events."""
    pool = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    return pool[cols].rank().corr()


# Curated composite family: one representative per confirmed signal family,
# including the orthogonal additive UWL_OPP pair that |IC|-greedy selection
# drops (small train IC but known to be genuinely additive on full data).
CURATED_CANDIDATES = [
    "sig_fsf_sell_6h_both_sides",   # fade reactive sell flow (neg)
    "sig_val_opp_retail",           # opposite crowding, weak hands (neg)
    "sig_ccw_n_cond",               # copy-crowd entry timing (neg)
    "sig_ccw_first",                # first-mover complement (pos)
    "sig_uwl_opp_retail",           # underwater weak-hand contrarian (pos, orthogonal)
    "sig_uwl_opp_gambler",          # underwater gambler contrarian (pos, orthogonal)
]


def select_non_redundant(splits, cols, target_col="roi_res", corr_threshold=0.70):
    """Greedy selection: sort by pooled |IC| desc, drop signals collinear with kept."""
    pool = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    target = pool[target_col]
    ranked = sorted(cols, key=lambda c: -abs(compute_event_ic(pool[c].fillna(0.0), target)))
    kept = []
    for col in ranked:
        if not kept:
            kept.append(col)
            continue
        sub = pool[kept + [col]].rank()
        if sub.corr().loc[col, kept].abs().max() < corr_threshold:
            kept.append(col)
    return kept


def select_curated(splits, cols, target_col="roi_res", corr_threshold=0.70):
    """Curated family selection.

    Start from :data:`CURATED_CANDIDATES` (only those present in the panel),
    then greedily append any remaining signal whose max |rank-corr| vs the
    curated set is below ``corr_threshold`` (sorted by |pooled IC|).
    """
    pool = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    target = pool[target_col]
    kept = [c for c in CURATED_CANDIDATES if c in cols]
    rest = sorted(
        [c for c in cols if c not in kept],
        key=lambda c: -abs(compute_event_ic(pool[c].fillna(0.0), target)),
    )
    for col in rest:
        sub = pool[kept + [col]].rank()
        if sub.corr().loc[col, kept].abs().max() < corr_threshold:
            kept.append(col)
    return kept


def add_price_residualized_pnl(splits, target_col="copyable_pnl", out_col="pnl_res"):
    """Price-residualize ``target_col`` (train-fit OLS in ranks) across splits.

    Mutates the splits in place, adding ``out_col`` = rank(target) minus the
    train-fit price component.  High IC of a composite vs ``pnl_res`` means the
    edge is beyond "buying cheap".
    """
    from signal_lab.signal_lib import fit_roi_residualizer, residualized_roi

    fit = fit_roi_residualizer(splits["train"][target_col], splits["train"]["price"])
    for frame in splits.values():
        frame[out_col] = residualized_roi(frame[target_col], frame["price"], fit)
    return splits


def add_price_bins(splits, n_bins=10, out_col="price_dec"):
    """Train-fit price decile bins applied to all splits (mutates in place)."""
    train_price = splits["train"]["price"]
    edges = pd.qcut(train_price, n_bins, labels=False, retbins=True)[1]
    edges[0], edges[-1] = -np.inf, np.inf
    for frame in splits.values():
        frame[out_col] = pd.cut(frame["price"], bins=edges, labels=False, include_lowest=True)
    return splits


def within_price_bin_ic(splits, score_col, target_col, bin_col="price_dec"):
    """Mean Spearman(score, target) within price bins, per split (partial-style)."""
    rows = []
    for split, frame in splits.items():
        ics = []
        for _, grp in frame.groupby(bin_col, observed=True):
            if len(grp) >= 100:
                r = spearman_rho(grp[score_col], grp[target_col])
                if np.isfinite(r):
                    ics.append(r)
        rows.append({
            "split": split,
            "within_price_bin_IC": float(np.mean(ics)) if ics else np.nan,
            "n_bins_used": len(ics),
        })
    return pd.DataFrame(rows)


def _parse_scale_grid(s: str) -> np.ndarray:
    start, stop, step = (float(x) for x in s.split(":"))
    return np.arange(start, stop, step)


def run_sizing(
    normalized_exposed: dict[str, pd.DataFrame],
    normalized_controlled: dict[str, pd.DataFrame],
    schemes: list[str],
    *,
    budget: float,
    scale_grid: np.ndarray,
    cost_bps: float,
    primary: str,
) -> pd.DataFrame:
    """Capital-constrained sizing on price-exposed vs price-controlled composites.

    Scale is selected on **validation** by the Sharpe-like objective and reported
    on **test** (never tuned on test).
    """
    from signal_lab.sizing import sizing_sharpe

    rows = []
    for scheme in schemes:
        col = f"composite_{scheme}"
        for label, nrm in (("price_exposed", normalized_exposed),
                           ("price_controlled", normalized_controlled)):
            best_scale, grid = select_scale(
                nrm["val"], col, budget, scale_grid, cost_bps, primary=primary
            )
            if grid.empty:
                print(f"  sizing {scheme}/{label}: no taken trades on val", flush=True)
                continue
            val_res = capital_constrained_sim(nrm["val"], col, budget, best_scale, cost_bps)
            test_res = capital_constrained_sim(nrm["test"], col, budget, best_scale, cost_bps)
            for split, res in (("val", val_res), ("test", test_res)):
                rows.append({
                    "scheme": scheme,
                    "variant": label,
                    "split": split,
                    "scale": best_scale,
                    "budget": budget,
                    "trades": res["trades"],
                    "net_pnl": round(res["net_pnl"], 2),
                    "notional": round(res["notional"], 2),
                    "peak_used": round(res["peak_used"], 2),
                    "mean_used": round(res["mean_used"], 2),
                    "pnl_per_peak": round(res["net_pnl"] / res["peak_used"], 4) if res["peak_used"] > 0 else 0.0,
                    "sharpe_daily": round(sizing_sharpe(res["daily_pnl"], 365.0), 3),
                    "sharpe_weekly": round(sizing_sharpe(res["daily_pnl"].resample("W").sum(), 52.0), 3),
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--corr-threshold", type=float, default=0.70,
                        help="Max |rank-corr| to keep a signal (greedy dedup).")
    parser.add_argument("--max-candidates", type=int, default=8,
                        help="Cap on composite candidates after dedup.")
    parser.add_argument("--greedy", action="store_true",
                        help="Use pure |IC|-greedy dedup instead of curated families.")
    parser.add_argument("--target", type=str, default="copyable_pnl",
                        choices=TARGETS,
                        help="Target the composite weights + ICs are fit/reported on.")
    parser.add_argument("--sizing", action="store_true",
                        help="Also run the capital-constrained sizing backtest.")
    parser.add_argument("--budget", type=float, default=10_000.0,
                        help="Global capital budget (USD) for the sizing backtest.")
    parser.add_argument("--scale-grid", type=str, default="0.1:3.01:0.1",
                        help="Scale grid as start:stop:step.")
    parser.add_argument("--cost-bps", type=float, default=0.0,
                        help="Cost in bps on notional of taken trades.")
    parser.add_argument("--sizing-primary", type=str, default="sharpe_daily",
                        choices=["sharpe_daily", "sharpe_weekly", "pnl_per_peak"],
                        help="Objective used to select scale on validation.")
    args = parser.parse_args()

    print("Loading stage-1 data...", flush=True)
    df_full, _df_train, _df_val, _df_test, wallet_metrics, hold_metrics = (
        load_stage1_data(max_shards=args.max_shards)
    )

    strategies = [
        CopyCrowdEntryTiming(),
        FadeReactiveSellFlow(),
        UwlOppContrarian(),
        FreshOppositeCrowdingFilter(),
        GamblerCapitulationSqueeze(),
    ]

    print("\nCalculating signals for all 5 strategies...", flush=True)
    splits, all_cols = run_strategies(df_full, wallet_metrics, hold_metrics, strategies)

    # 0. Baseline all-candidates per split (regime context for val/test).
    print("\n" + "=" * 60, flush=True)
    print("Baseline all-candidates per split", flush=True)
    print("=" * 60, flush=True)
    base_rows = []
    for sp in ("train", "val", "test"):
        f = splits[sp]
        base_rows.append({
            "split": sp,
            "n": len(f),
            "copyable_roi": float(f["copyable_roi"].mean()),
            "copyable_pnl": float(f["copyable_pnl"].sum()),
            "copyable_notional": float(f["copyable_notional"].sum()),
        })
    print(pd.DataFrame(base_rows).round(4), flush=True)

    # 1. Full panel (family ICs) against the chosen target.
    print("\n" + "=" * 60, flush=True)
    print(f"Full signal panel vs {args.target}", flush=True)
    print("=" * 60, flush=True)
    report, _ = evaluate_signal_panel(splits, all_cols, roi_col=args.target)
    print(report, flush=True)

    # 2. Redundancy check among confirmed top signals.
    print("\n" + "=" * 60, flush=True)
    print("Rank correlation among signals (train+val pooled)", flush=True)
    print("=" * 60, flush=True)
    cols_here = [c for c in all_cols if c in splits["train"].columns]
    corr = _pooled_rank_corr(splits, cols_here)
    print(corr.round(3), flush=True)

    # 3. Greedy dedup -> composite candidate set.
    if args.greedy:
        candidates = select_non_redundant(
            splits, cols_here, target_col=args.target,
            corr_threshold=args.corr_threshold,
        )[: args.max_candidates]
        mode = f"greedy (corr<{args.corr_threshold})"
    else:
        candidates = select_curated(
            splits, cols_here, target_col=args.target,
            corr_threshold=args.corr_threshold,
        )[: args.max_candidates]
        mode = f"curated (corr<{args.corr_threshold})"
    print("\n" + "=" * 60, flush=True)
    print(f"Composite candidates ({mode}, cap {args.max_candidates})", flush=True)
    print("=" * 60, flush=True)
    print(pd.Series(candidates), flush=True)

    # 4. Price-confound columns (train-fit): pnl_res + price deciles.
    add_price_residualized_pnl(splits, target_col=args.target, out_col="pnl_res")
    add_price_bins(splits)

    # 5. Build composite scores (train-fit weights on the target).
    normalized, schemes, _ = build_composite_scores(
        splits, candidates, roi_col=args.target, weight_split="train", shrinkage=0.5
    )
    print("\nComposite weight schemes:", flush=True)
    for name, w in schemes.items():
        print(f"  {name}: {w.round(3).to_dict()}", flush=True)

    # 6. Composite IC per split against target AND price-controlled views.
    print("\n" + "=" * 60, flush=True)
    print(f"Composite IC per split (target={args.target})", flush=True)
    print("=" * 60, flush=True)
    ic_rows = []
    for name in ("train", "val", "test"):
        for scheme in schemes:
            ic_rows.append({
                "split": name,
                "scheme": scheme,
                "IC_target": compute_event_ic(
                    normalized[name][f"composite_{scheme}"], normalized[name][args.target]),
                "IC_pnl_res": compute_event_ic(
                    normalized[name][f"composite_{scheme}"], normalized[name]["pnl_res"]),
                "spearman_price": spearman_rho(
                    normalized[name][f"composite_{scheme}"], normalized[name]["price"]),
            })
    print(pd.DataFrame(ic_rows).round(4).to_string(index=False), flush=True)

    print("\nWithin-price-bin IC (mean Spearman inside train-fit price deciles):", flush=True)
    bin_rows = []
    for scheme in schemes:
        sub = within_price_bin_ic(normalized, f"composite_{scheme}", args.target)
        sub.insert(0, "scheme", scheme)
        bin_rows.append(sub)
    print(pd.concat(bin_rows, ignore_index=True).round(4).to_string(index=False), flush=True)

    # 7. Threshold grid: select on val by max net PnL, report before/after on test.
    print("\n" + "=" * 60, flush=True)
    print("Threshold grid on validation (selection, by net PnL)", flush=True)
    print("=" * 60, flush=True)
    results = []
    for scheme in schemes:
        col = f"composite_{scheme}"
        val_grid = evaluate_threshold_grid(normalized["val"], col)
        n_val = len(normalized["val"])
        val_grid = val_grid[
            (val_grid["trades"] >= max(500, int(0.01 * n_val)))
            & (val_grid["firing_rate"] <= 0.95)
        ]
        if val_grid.empty:
            continue
        best = val_grid.sort_values("copyable_pnl_net", ascending=False).iloc[0]
        best_t = best["threshold"]

        test_row = evaluate_threshold_grid(normalized["test"], col)
        match = test_row[np.isclose(test_row["threshold"], best_t, atol=1e-9)]
        test_best = match.iloc[0] if not match.empty else test_row.iloc[0]
        all_t = test_row.iloc[0]

        test_ic = compute_event_ic(
            normalized["test"][col].where(normalized["test"][col] >= best_t),
            normalized["test"][args.target],
        )
        results.append({
            "scheme": scheme,
            "target": args.target,
            "val_threshold": best_t,
            "val_trades": int(best["trades"]),
            "val_firing_rate": float(best["firing_rate"]),
            "val_pnl_net": float(best["copyable_pnl_net"]),
            "val_roi_net": float(best["copyable_roi_net"]),
            "test_trades": int(test_best["trades"]),
            "test_firing_rate": float(test_best["firing_rate"]),
            "test_pnl_net": float(test_best["copyable_pnl_net"]),
            "test_roi_net": float(test_best["copyable_roi_net"]),
            "test_pnl_net_all": float(all_t["copyable_pnl_net"]),
            "test_roi_net_all": float(all_t["copyable_roi_net"]),
            "IC_test_selected_vs_target": test_ic,
        })
    out = pd.DataFrame(results)
    print(out.round(4), flush=True)
    out.to_csv("composite_results.csv", index=False)
    print("Saved to composite_results.csv", flush=True)

    if args.sizing:
        # 8. Capital-constrained sizing backtest: price-exposed vs price-controlled.
        print("\n" + "=" * 60, flush=True)
        print("Capital-constrained sizing (score-proportional, clipped to copyable_qty_5m_100)", flush=True)
        print(f"budget=${args.budget:,.0f}  cost_bps={args.cost_bps}  primary={args.sizing_primary}", flush=True)
        print("=" * 60, flush=True)
        scale_grid = _parse_scale_grid(args.scale_grid)
        # Price-controlled composite: weights fit on train against pnl_res.
        controlled, c_schemes, _ = build_composite_scores(
            splits, list(candidates), roi_col="pnl_res", weight_split="train", shrinkage=0.5
        )
        siz = run_sizing(
            normalized, controlled, list(schemes),
            budget=args.budget, scale_grid=scale_grid,
            cost_bps=args.cost_bps, primary=args.sizing_primary,
        )
        print("\nSizing results (scale picked on val, reported on val + test):", flush=True)
        print(siz.round(4).to_string(index=False), flush=True)
        siz.to_csv("sizing_results.csv", index=False)
        print("Saved to sizing_results.csv", flush=True)


if __name__ == "__main__":
    main()
