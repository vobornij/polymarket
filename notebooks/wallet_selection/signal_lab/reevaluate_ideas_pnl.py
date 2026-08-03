"""Re-evaluate every implemented signal idea toward RAW PNL (dollars).

The crowding postmortem showed rank IC on ``roi_res`` / ``copyable_roi`` does
not predict dollars-per-capital.  This script re-runs all coded strategies on
the fixed pipeline and measures each idea by the dollars it produces relative
to the copy-all baseline (and a price-favorite sizing benchmark):

- IC panel vs ``roi_res`` (continuity) and vs ``copyable_pnl`` (dollar rank).
- Firing gate: train-fit quantile selection of the edge tail -> pnl / roi_w
  vs copy-all, on val and test.
- Sizing gate: walk-forward capital-constrained ($10k) sim for the single best
  signal per idea, benchmarked vs copy-all and price-favorite sizing.

Walk-forward discipline: fold A selects on train -> reports val; fold B selects
on train+val -> reports test.  Nothing is tuned on test.

Writes ideas_pnl_{ic,firing,sizing,summary}.csv
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

from signal_lab.sizing import (
    capital_constrained_sim,
    score_floor_for_fraction,
    select_scale,
    sizing_sharpe,
)
from signal_lab.signal_lib import (
    apply_rank_transformer,
    evaluate_strategy,
    fit_rank_transformer,
)
from signal_lab.stage1 import evaluate_signal_panel, load_stage1_data, run_strategies
from signal_lab.strategies import (
    BigWinnerMarketCharacterization,
    CopyCrowdEntryTiming,
    CopyWalletQualitySignals,
    FadeReactiveSellFlow,
    FreshOppositeCrowdingFilter,
    GamblerCapitulationSqueeze,
    UwlOppContrarian,
)

from reevaluate_crowding import panel_comparison

BUDGET = 10_000.0
SCALE_GRID = np.arange(0.25, 3.01, 0.25)
FIRING_FRACTIONS = [0.50, 0.25, 0.10]
SIZING_FRACTIONS = [None, 0.25]
TOP_N_FIRING = 2

STRATEGIES = [
    BigWinnerMarketCharacterization(),
    CopyCrowdEntryTiming(),
    CopyWalletQualitySignals(),
    FadeReactiveSellFlow(),
    GamblerCapitulationSqueeze(),
    FreshOppositeCrowdingFilter(),
    UwlOppContrarian(),
]


def sign_adjusted_score(splits, col, target="copyable_pnl"):
    """Frames with a rank-normalized, edge-direction-corrected score column.

    Sign from the pooled train+val IC on ``target`` so the selected tail is
    always the edge tail.  Score is in ~[-1, 1] after rank transform.
    """
    pooled = pd.concat(
        [splits["train"][[col, target]], splits["val"][[col, target]]],
        ignore_index=True,
    )
    sign = np.sign(np.corrcoef(pooled[col].fillna(0.0), pooled[target])[0, 1])
    raw = {name: sign * frame[col].fillna(0.0) for name, frame in splits.items()}
    fit = fit_rank_transformer(raw["train"])
    out = {}
    for name, frame in splits.items():
        f = frame.copy(deep=True)
        f[col] = apply_rank_transformer(raw[name], fit)
        out[name] = f
    return out


def firing_gate(splits, score_col, fractions):
    """Train-fit quantile selection; pnl / roi_w vs copy-all on val + test."""
    rows = []
    for split in ("val", "test"):
        f = splits[split]
        cnot = f["copyable_notional"].sum()
        rows.append({
            "split": split, "fraction": None, "threshold": np.nan,
            "trades": len(f), "firing_rate": 1.0,
            "pnl": round(float(f["copyable_pnl"].sum()), 2),
            "roi_w": round(float(f["copyable_pnl"].sum() / cnot), 4) if cnot > 0 else np.nan,
            "pnl_per_trade": round(float(f["copyable_pnl"].sum()) / len(f), 4),
        })
    for fraction in fractions:
        thr = float(splits["train"][score_col].quantile(1.0 - fraction))
        for split in ("val", "test"):
            r = evaluate_strategy(splits[split], score_col, thr)
            rows.append({
                "split": split, "fraction": fraction, "threshold": round(thr, 4),
                "trades": r["trades"], "firing_rate": round(r["firing_rate"], 4),
                "pnl": round(r["copyable_pnl"], 2),
                "roi_w": round(r["copyable_roi"], 4),
                "pnl_per_trade": round(r["copyable_pnl"] / max(r["trades"], 1), 4),
            })
    return pd.DataFrame(rows)


def select_best_design(sel, score_col, fractions, scale_grid):
    """Pick (fraction, floor, scale) on ``sel`` by daily Sharpe; return row + params."""
    best = None
    for f in fractions:
        floor = score_floor_for_fraction(sel, score_col, f) if f is not None else None
        scale, grid = select_scale(sel, score_col, BUDGET, scale_grid, score_floor=floor)
        if grid.empty:
            continue
        row = grid.sort_values("sharpe_daily", ascending=False).iloc[0]
        if best is None or row["sharpe_daily"] > best["sharpe_daily"]:
            best = row
            best_params = (f, floor, scale)
    if best is None:
        return None, None
    return best, best_params


def sizing_gate(splits, score_col, fractions):
    """Walk-forward capital sim: select on train / train+val, report val / test."""
    frames = {name: frame.copy(deep=True) for name, frame in splits.items()}
    price_fit = fit_rank_transformer(frames["train"]["price"])
    for name in frames:
        frames[name]["score_all"] = 1.0
        frames[name]["score_price"] = apply_rank_transformer(
            frames[name]["price"], price_fit
        )
    sel_a = frames["train"]
    sel_b = pd.concat([frames["train"], frames["val"]], ignore_index=True)

    rows = []
    for label, col in [
        ("signal", score_col),
        ("bench_score_all", "score_all"),
        ("bench_score_price", "score_price"),
    ]:
        fracs = fractions if label == "signal" else [None]
        for fold, sel, rep_name in (
            ("A", sel_a, "val"),
            ("B", sel_b, "test"),
        ):
            best, params = select_best_design(sel, col, fracs, SCALE_GRID)
            if best is None:
                continue
            f, floor, scale = params
            res = capital_constrained_sim(
                frames[rep_name], col, BUDGET, scale, score_floor=floor
            )
            rows.append({
                "fold": fold, "split": rep_name, "design": label,
                "fraction": f, "floor": round(floor, 4) if floor is not None else np.nan,
                "scale": scale,
                "trades": res["trades"],
                "pnl": round(res["net_pnl"], 2),
                "roi_w": round(res["net_pnl"] / res["notional"], 4) if res["notional"] > 0 else np.nan,
                "pnl_per_peak": round(res["net_pnl"] / res["peak_used"], 4) if res["peak_used"] > 0 else 0.0,
                "sharpe_daily": round(sizing_sharpe(res["daily_pnl"], 365.0), 3),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()

    print("Loading stage-1 data...", flush=True)
    df_full, _dt, _dv, _dtest, wallet_metrics, hold_metrics = load_stage1_data(
        max_shards=args.max_shards
    )

    print("Running all strategies on the shared candidate universe...", flush=True)
    splits, all_cols = run_strategies(df_full, wallet_metrics, hold_metrics, STRATEGIES)
    print(f"Total signal cols: {len(all_cols)}", flush=True)

    ic_rows = []
    firing_rows = []
    sizing_rows = []
    summary_rows = []
    for strategy in STRATEGIES:
        strat_cols = [c for c in strategy.get_signal_columns() if c in splits["train"].columns]
        print(f"\n{'=' * 70}\nIdea: {strategy.name} ({len(strat_cols)} signals)\n{'=' * 70}", flush=True)
        if not strat_cols:
            continue

        ic = panel_comparison(splits, strat_cols)
        ic.insert(0, "idea", strategy.name)
        ic_rows.append(ic)

        # Pick edge direction and top signals by |pooled IC on copyable_pnl|.
        pooled_ic = ic.sort_values("boot_mean_copyable_pnl", key=abs, ascending=False)
        top_cols = pooled_ic["signal"].head(TOP_N_FIRING).tolist()
        print(f"Top signals by |IC copyable_pnl|: {top_cols}", flush=True)

        for col in top_cols:
            scored = sign_adjusted_score(splits, col)
            fg = firing_gate(scored, col, FIRING_FRACTIONS)
            fg.insert(0, "idea", strategy.name)
            fg.insert(1, "signal", col)
            firing_rows.append(fg)
            print(f"\n  Firing gate: {col}", flush=True)
            print(fg.to_string(index=False), flush=True)

        best_col = pooled_ic["signal"].iloc[0]
        scored = sign_adjusted_score(splits, best_col)
        sg = sizing_gate(scored, best_col, SIZING_FRACTIONS)
        sg.insert(0, "idea", strategy.name)
        sg.insert(1, "signal", best_col)
        sizing_rows.append(sg)
        print(f"\n  Sizing gate (best signal {best_col}):", flush=True)
        print(sg.to_string(index=False), flush=True)

        fold_b = sg[sg["fold"] == "B"]
        sig_b = fold_b[fold_b["design"] == "signal"]
        all_b = fold_b[fold_b["design"] == "bench_score_all"]
        sig_row = sig_b.sort_values("sharpe_daily", ascending=False).iloc[0] if not sig_b.empty else None
        all_row = all_b.iloc[0] if not all_b.empty else None
        summary_rows.append({
            "idea": strategy.name,
            "best_signal": best_col,
            "test_pnl_signal": sig_row["pnl"] if sig_row is not None else np.nan,
            "test_pnl_copy_all": all_row["pnl"] if all_row is not None else np.nan,
            "pnl_ratio_vs_copyall": round(sig_row["pnl"] / all_row["pnl"], 3)
            if sig_row is not None and all_row is not None and all_row["pnl"] != 0 else np.nan,
            "test_sharpe_signal": sig_row["sharpe_daily"] if sig_row is not None else np.nan,
            "test_sharpe_copy_all": all_row["sharpe_daily"] if all_row is not None else np.nan,
            "test_roi_w_signal": sig_row["roi_w"] if sig_row is not None else np.nan,
            "test_roi_w_copy_all": all_row["roi_w"] if all_row is not None else np.nan,
        })

    ic_all = pd.concat(ic_rows, ignore_index=True)
    firing_all = pd.concat(firing_rows, ignore_index=True)
    sizing_all = pd.concat(sizing_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    ic_all.to_csv("ideas_pnl_ic.csv", index=False)
    firing_all.to_csv("ideas_pnl_firing.csv", index=False)
    sizing_all.to_csv("ideas_pnl_sizing.csv", index=False)
    summary.to_csv("ideas_pnl_summary.csv", index=False)

    print("\n" + "=" * 70, flush=True)
    print("Deployment-fold summary (select train+val -> test)", flush=True)
    print("=" * 70, flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nSaved ideas_pnl_{ic,firing,sizing,summary}.csv", flush=True)


if __name__ == "__main__":
    main()
