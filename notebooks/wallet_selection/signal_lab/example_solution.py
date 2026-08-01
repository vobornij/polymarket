"""Minimal working example for signal-lab exploration.

This is intentionally small and opinionated. It shows the full path:

1. load the stage-1 data,
2. attach one known sample signal,
3. convert it into a copy-trade-friendly direction,
4. evaluate it on the copy universe,
5. run a rough threshold check.

The example uses opposite-side flipper crowding because prior stage-1 work
suggests this family can contain real signal after price control.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

_NOTEBOOK_DIR = Path(__file__).resolve().parent.parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

try:
    from .filters import COPY_DEFAULT, FLIPPER
    from .signal_engines import VAL_OPP
    from .stage1 import (
        attach_position_signal_panel,
        build_composite_scores,
        candidate_splits_for,
        evaluate_signal_panel,
        evaluate_threshold_grid,
        load_stage1_data,
        restrict_trades,
    )
except ImportError:
    from filters import COPY_DEFAULT, FLIPPER  # type: ignore
    from signal_engines import VAL_OPP  # type: ignore
    from stage1 import (  # type: ignore
        attach_position_signal_panel,
        build_composite_scores,
        candidate_splits_for,
        evaluate_signal_panel,
        evaluate_threshold_grid,
        load_stage1_data,
        restrict_trades,
    )


SAMPLE_BASE_SIGNAL = "sig_val_opp_flipper"
SAMPLE_COPY_SIGNAL = "sig_copy_anti_crowding_flipper"


def run_example_solution(
    cost_bps: float = 0.0,
    data=None,
) -> dict[str, object]:
    """Run one complete lightweight signal-lab example."""
    if data is None:
        df_full, df_train, df_val, df_test, wallet_metrics, hold_metrics = (
            load_stage1_data()
        )
    else:
        df_full, df_train, df_val, df_test, wallet_metrics, hold_metrics = data

    copy_wallets = set(COPY_DEFAULT(wallet_metrics, hold_metrics))
    splits = candidate_splits_for(df_full, copy_wallets)
    conditions: set[str] = set()
    for frame in splits.values():
        conditions.update(frame["condition_id"].unique())
    trades = restrict_trades(df_full, conditions)
    splits, cols = attach_position_signal_panel(
        trades,
        splits,
        [FLIPPER],
        kinds=[VAL_OPP],
        wallet_metrics=wallet_metrics,
        hold_metrics=hold_metrics,
    )

    if SAMPLE_BASE_SIGNAL not in cols:
        raise RuntimeError(f"Expected sample signal {SAMPLE_BASE_SIGNAL!r} not found")

    # More opposite-side crowding was previously associated with worse future
    # copyable ROI, so copying prefers the *negative* of that crowding measure.
    for frame in splits.values():
        frame[SAMPLE_COPY_SIGNAL] = -frame[SAMPLE_BASE_SIGNAL].fillna(0.0)

    report, selected = evaluate_signal_panel(splits, [SAMPLE_COPY_SIGNAL])

    result: dict[str, object] = {
        "workspace_summary": pd.DataFrame(
            [
                {
                    "copy_wallets": len(copy_wallets),
                    "candidate_trades": sum(len(f) for f in splits.values()),
                    "train_candidates": len(splits["train"]),
                    "val_candidates": len(splits["val"]),
                    "test_candidates": len(splits["test"]),
                    "candidate_conditions": len(conditions),
                }
            ]
        ),
        "signal_report": report,
        "selected": selected,
        "splits": splits,
    }
    if not selected:
        return result

    scored_splits, weights, _ = build_composite_scores(splits, selected)
    val_grid = evaluate_threshold_grid(
        scored_splits["val"],
        "composite_shrinkage_markowitz",
        cost_bps=cost_bps,
    )
    best_row = val_grid[val_grid["trades"] >= 20]
    if best_row.empty:
        best_row = val_grid
    best_row = best_row.sort_values("copyable_pnl_net", ascending=False).iloc[0]
    best_threshold = float(best_row["threshold"])
    test_row = evaluate_threshold_grid(
        scored_splits["test"],
        "composite_shrinkage_markowitz",
        thresholds=pd.Index([best_threshold]).to_numpy(dtype=float),
        cost_bps=cost_bps,
    )

    result.update(
        {
            "scored_splits": scored_splits,
            "weights": weights,
            "validation_grid": val_grid,
            "best_threshold": best_threshold,
            "test_evaluation": test_row,
        }
    )
    return result


if __name__ == "__main__":
    out = run_example_solution()
    print("Workspace:")
    print(out["workspace_summary"].to_string(index=False))
    print("\nSignal report:")
    print(out["signal_report"].round(4).to_string(index=False))
    if out["selected"]:
        print("\nBest threshold:", out["best_threshold"])
        print("\nTest evaluation:")
        print(out["test_evaluation"].round(4).to_string(index=False))
    else:
        print("\nNo signal selected on train/validation.")
