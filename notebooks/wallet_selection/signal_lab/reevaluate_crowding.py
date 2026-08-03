"""Re-evaluate the fresh opposite-crowding idea on the fixed pipeline.

Three questions this answers:

1. Is the opposite-side crowding IC real on the fixed pipeline (rankdata bug
   fixed, train-only rank normalization), and against which target
   (``roi_res`` vs raw ``copyable_roi`` vs dollar ``copyable_pnl``)?
2. Does selecting the low-crowding tail actually bump average copyable ROI
   out-of-sample (decile analysis + firing-rate selection)?
3. Is the edge large enough to matter under a capital cap (Sharpe, step 4 in
   ``crowding_overlay.py``)?

This script runs steps 1-3: the signal panel, the decile gate, and the
firing-rate selection test.  It writes ``crowding_reeval_panel.csv`` and
``crowding_reeval_deciles.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")

_NOTEBOOK_DIR = Path(__file__).resolve().parent.parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

from signal_lab.filters import (
    BOTH_SIDES,
    COPY_DEFAULT,
    FLIPPER,
    MAX_DD,
    OVERSELLER,
    RETAIL,
)
from signal_lab.signal_engines import VAL_OPP
from signal_lab.signal_lib import (
    apply_rank_transformer,
    bootstrap_ic,
    compute_event_ic,
    evaluate_strategy,
    fit_rank_transformer,
)
from signal_lab.stage1 import (
    evaluate_signal_panel,
    load_stage1_data,
    run_strategy,
)
from signal_lab.strategies.base import DeclarativeStrategy

from evaluate_composite import add_price_bins, within_price_bin_ic

REACTIVE_SETS = [BOTH_SIDES, FLIPPER, OVERSELLER, RETAIL, MAX_DD]
SET_NAMES = [f.name for f in REACTIVE_SETS]
TAUS_H = [1.0, 6.0, 24.0]
FIRING_FRACTIONS = [0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
N_DECILES = 10


class CrowdingReeval(DeclarativeStrategy):
    """Baseline ``val_opp`` + fresh ``fval_opp`` over the reactive sets."""

    copy_mask = COPY_DEFAULT
    signal_sets = REACTIVE_SETS
    kinds = [VAL_OPP]
    fresh_kinds = [VAL_OPP]
    taus_h = TAUS_H


def base_col(set_name: str) -> str:
    return f"sig_val_opp_{set_name}"


def fresh_col(set_name: str, tau: float) -> str:
    return f"sig_fval_opp_{tau:g}h_{set_name}"


def _split_ics(splits, col, target):
    return {
        split: compute_event_ic(splits[split][col].fillna(0.0), splits[split][target])
        for split in splits
    }


def panel_comparison(splits, cols, targets=("roi_res", "copyable_roi", "copyable_pnl")):
    """Per-signal IC across splits for every target (the ``signal vs target`` view)."""
    rows = []
    for col in cols:
        row = {"signal": col}
        for target in targets:
            ics = _split_ics(splits, col, target)
            for split, ic in ics.items():
                row[f"IC_{target}_{split}"] = ic
            pooled = pd.concat(
                [splits["train"][[col, target]], splits["val"][[col, target]]],
                ignore_index=True,
            )
            m, lo, hi = bootstrap_ic(
                pooled[col].fillna(0.0), pooled[target], n_iter=500, seed=42
            )
            row[f"boot_mean_{target}"] = m
            row[f"boot_ci_{target}"] = f"[{lo:.4f}, {hi:.4f}]"
        rows.append(row)
    out = pd.DataFrame(rows)
    out["|IC_roi_res_train|"] = out["IC_roi_res_train"].abs()
    return out.sort_values("|IC_roi_res_train|", ascending=False).reset_index(drop=True)


def build_copy_scores(splits, sets, tau=None):
    """Negated crowding score per set, rank-normalized with a train-only fit."""
    out = {name: frame.copy(deep=True) for name, frame in splits.items()}
    for set_name in sets:
        col = fresh_col(set_name, tau) if tau is not None else base_col(set_name)
        raw = {name: -out[name][col].fillna(0.0) for name in out}
        fit = fit_rank_transformer(raw["train"])
        for name, frame in out.items():
            out[name][f"copy_{set_name}"] = apply_rank_transformer(raw[name], fit)
    return out


def add_blend(frame, set_names, out_col):
    """Equal-weight mean of per-set copy scores (train-fit scale is already [-1,1])."""
    frame[out_col] = frame[[f"copy_{s}" for s in set_names]].mean(axis=1)


def decile_analysis(scored, score_col, n=N_DECILES):
    """Train-fit decile edges applied to every split; per-decile outcome means.

    The reported ``roi_w`` is PnL-weighted ROI = sum(copyable_pnl) /
    sum(copyable_notional) — the quantity the "bump average copyable roi" test
    cares about.
    """
    train = scored["train"]
    edges = pd.qcut(train[score_col], n, labels=False, retbins=True)[1]
    edges[0], edges[-1] = -np.inf, np.inf
    for split, frame in scored.items():
        frame["decile"] = pd.cut(
            frame[score_col], bins=edges, labels=False, include_lowest=True
        )
    rows = []
    for split, frame in scored.items():
        for d in range(n):
            g = frame[frame["decile"] == d]
            cnot = g["copyable_notional"].sum()
            rows.append({
                "split": split,
                "decile": d,
                "n": len(g),
                "mean_roi_res": float(g["roi_res"].mean()),
                "mean_copyable_roi": float(g["copyable_roi"].mean()),
                "roi_w": float(g["copyable_pnl"].sum() / cnot) if cnot > 0 else np.nan,
                "pnl": float(g["copyable_pnl"].sum()),
            })
    return pd.DataFrame(rows)


def firing_rate_test(scored, score_col, fractions, cost_bps=0.0):
    """Select the top ``f`` fraction by score; threshold from train+val."""
    ref = pd.concat([scored["train"][score_col], scored["val"][score_col]])
    rows = []
    for f in fractions:
        thr = float(ref.quantile(1.0 - f))
        for split in ("val", "test"):
            r = evaluate_strategy(scored[split], score_col, thr, cost_bps=cost_bps)
            rows.append({
                "fraction": f,
                "threshold": round(thr, 4),
                "split": split,
                "trades": r["trades"],
                "firing_rate": round(r["firing_rate"], 4),
                "copyable_pnl_net": round(r["copyable_pnl_net"], 2),
                "copyable_roi_net": round(r["copyable_roi_net"], 4),
                "pnl_per_trade": round(r["copyable_pnl_net"] / max(r["trades"], 1), 4),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--cost-bps", type=float, default=0.0)
    args = parser.parse_args()

    print("Loading stage-1 data...", flush=True)
    df_full, _dt, _dv, _dtest, wallet_metrics, hold_metrics = load_stage1_data(
        max_shards=args.max_shards
    )

    strategy = CrowdingReeval()
    print(f"\nRunning strategy {strategy.name}...", flush=True)
    splits, cols = run_strategy(df_full, wallet_metrics, hold_metrics, strategy)
    print(f"Signal cols: {len(cols)}", flush=True)

    # 1. IC panel vs roi_res (the framework target) + vs raw targets.
    print("\n" + "=" * 70, flush=True)
    print("Signal panel vs roi_res (framework IC protocol)", flush=True)
    print("=" * 70, flush=True)
    report, selected = evaluate_signal_panel(splits, cols, roi_col="roi_res")
    print(report.round(4).to_string(index=False), flush=True)

    print("\n" + "=" * 70, flush=True)
    print("Signal IC across targets (roi_res vs copyable_roi vs copyable_pnl)", flush=True)
    print("=" * 70, flush=True)
    comp = panel_comparison(splits, cols)
    print(comp.round(4).to_string(index=False), flush=True)
    comp.to_csv("crowding_reeval_panel.csv", index=False)
    print("Saved crowding_reeval_panel.csv", flush=True)

    # Within-price-bin IC for the baseline + best fresh variant per set.
    print("\n" + "=" * 70, flush=True)
    print("Within-price-bin IC (mean Spearman inside train-fit price deciles)", flush=True)
    print("=" * 70, flush=True)
    add_price_bins(splits)
    rows = []
    for s in SET_NAMES:
        for label, col in (("base", base_col(s)), ("24h", fresh_col(s, 24.0))):
            sub = within_price_bin_ic(splits, col, "roi_res")
            sub.insert(0, "signal", f"{label}:{col}")
            rows.append(sub)
    print(pd.concat(rows, ignore_index=True).round(4).to_string(index=False), flush=True)

    # 2. Copy scores (negated, rank-normalized train-fit) + blended.
    scored = build_copy_scores(splits, SET_NAMES)
    for name, frame in scored.items():
        add_blend(frame, SET_NAMES, "copy_blend")
    add_price_bins(scored)

    print("\n" + "=" * 70, flush=True)
    print("Copy-score IC vs targets (per split)", flush=True)
    print("=" * 70, flush=True)
    score_cols = [f"copy_{s}" for s in SET_NAMES] + ["copy_blend"]
    score_ics = []
    for col in score_cols:
        row = {"score": col}
        for target in ("roi_res", "copyable_roi", "copyable_pnl"):
            for split in ("train", "val", "test"):
                row[f"IC_{target}_{split}"] = compute_event_ic(
                    scored[split][col], scored[split][target]
                )
        score_ics.append(row)
    print(pd.DataFrame(score_ics).round(4).to_string(index=False), flush=True)

    # 3. Decile gate.
    print("\n" + "=" * 70, flush=True)
    print("Decile analysis (train-fit edges), copy_blend", flush=True)
    print("=" * 70, flush=True)
    dec = decile_analysis(scored, "copy_blend")
    print(dec.round(4).to_string(index=False), flush=True)
    dec.to_csv("crowding_reeval_deciles.csv", index=False)
    print("Saved crowding_reeval_deciles.csv", flush=True)

    # 4. Firing-rate selection test.
    print("\n" + "=" * 70, flush=True)
    print(f"Firing-rate selection test (top-k% by copy_blend, cost {args.cost_bps}bps)", flush=True)
    print("=" * 70, flush=True)
    frt = firing_rate_test(scored, "copy_blend", FIRING_FRACTIONS, cost_bps=args.cost_bps)
    print(frt.to_string(index=False), flush=True)
    frt.to_csv("crowding_reeval_firing.csv", index=False)
    print("Saved crowding_reeval_firing.csv", flush=True)

    # Baseline context per split.
    print("\nBaseline copy-all per split:", flush=True)
    for split in ("train", "val", "test"):
        f = splits[split]
        print(
            f"  {split}: n={len(f)}, copyable_pnl={f['copyable_pnl'].sum():,.0f}, "
            f"roi_w={f['copyable_pnl'].sum() / f['copyable_notional'].sum():.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
