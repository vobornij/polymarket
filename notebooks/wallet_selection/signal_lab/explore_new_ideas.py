"""
Explore new signal ideas sequentially using the functional stage1 pipeline.
"""

import sys
from pathlib import Path
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

_NOTEBOOK_DIR = Path(__file__).resolve().parent.parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

from signal_lab.stage1 import (
    evaluate_signal_panel,
    load_stage1_data,
    run_strategy,
)
from signal_lab.strategies import (
    GamblerCapitulationSqueeze,
    FreshOppositeCrowdingFilter
)

def main():
    print("Loading stage-1 data...")
    df_full, _df_train, _df_val, _df_test, wallet_metrics, hold_metrics = (
        load_stage1_data()
    )

    # 1. Instantiate the strategies you want to run
    strategies = [
        GamblerCapitulationSqueeze(),
        FreshOppositeCrowdingFilter(),
        # Add more strategies here, or comment them out!
    ]

    # 2. Run each strategy end-to-end: it rebuilds the candidate universe from
    # its own copy-wallet filter, re-splits chronologically, re-residualizes
    # ROI, and attaches its declared signal panel.
    for strategy in strategies:
        print(f"\nCalculating signals for strategy: {strategy.name}...")
        splits, cols = run_strategy(df_full, wallet_metrics, hold_metrics, strategy)

        # Some columns may be skipped if the archetypes don't exist.
        actual_cols = [c for c in cols if c in splits["train"].columns]
        if not actual_cols:
            print(f"No signals attached for {strategy.name} (empty wallet sets?).")
            continue

        print(f"Evaluating individual strategy: {strategy.name}")
        report, _ = evaluate_signal_panel(splits, actual_cols, roi_col="roi_res")
        print(report)

    # 3. Optional: Evaluate a combined model or thresholding across all generated signals
    print("\n" + "="*60)
    print("Combined Signal Evaluation (All Strategies)")
    print("="*60)

    # Example: you could run compute_optimal_weights on all generated signals here
    # to see if the combined framework improves!

if __name__ == "__main__":
    main()
