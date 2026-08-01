"""Minimal working example for signal-lab exploration.

This is intentionally small and opinionated. It shows the full path:

1. load the stage-1 workspace,
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
    from .stage1 import (
        attach_position_signal_panel,
        build_composite_scores,
        build_stage1_workspace,
        evaluate_signal_panel,
        evaluate_threshold_grid,
        summarize_workspace,
    )
except ImportError:
    from stage1 import (  # type: ignore
        attach_position_signal_panel,
        build_composite_scores,
        build_stage1_workspace,
        evaluate_signal_panel,
        evaluate_threshold_grid,
        summarize_workspace,
    )


SAMPLE_BASE_SIGNAL = "sig_val_opp_flipper"
SAMPLE_COPY_SIGNAL = "sig_copy_anti_crowding_flipper"


def run_example_solution(
    cost_bps: float = 0.0,
    workspace=None,
) -> dict[str, object]:
    """Run one complete lightweight signal-lab example."""
    ws = build_stage1_workspace() if workspace is None else workspace
    splits, cols = attach_position_signal_panel(
        ws,
        signal_sets={"flipper": ws.signal_sets["flipper"]},
        kinds=[("val", "opp")],
    )

    if SAMPLE_BASE_SIGNAL not in cols:
        raise RuntimeError(f"Expected sample signal {SAMPLE_BASE_SIGNAL!r} not found")

    # More opposite-side crowding was previously associated with worse future
    # copyable ROI, so copying prefers the *negative* of that crowding measure.
    for frame in splits.values():
        frame[SAMPLE_COPY_SIGNAL] = -frame[SAMPLE_BASE_SIGNAL].fillna(0.0)

    report, selected = evaluate_signal_panel(splits, [SAMPLE_COPY_SIGNAL])

    result: dict[str, object] = {
        "workspace_summary": summarize_workspace(ws),
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
