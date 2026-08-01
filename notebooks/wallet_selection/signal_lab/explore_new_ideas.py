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
    run_strategies,
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

    # 2. Run all strategies onto the same shared candidate universe
    print("\nCalculating signals for all strategies combined...")
    splits, all_cols = run_strategies(df_full, wallet_metrics, hold_metrics, strategies)
    
    # 3. Evaluate each strategy individually
    for strategy in strategies:
        strat_cols = [c for c in strategy.get_signal_columns() if c in splits["train"].columns]
        if not strat_cols:
            print(f"No signals attached for {strategy.name} (empty wallet sets?).")
            continue
            
        print(f"\nEvaluating individual strategy: {strategy.name}")
        report, _ = evaluate_signal_panel(splits, strat_cols, roi_col="roi_res")
        print(report)

    # 4. Evaluate combined model
    print("\n" + "="*60)
    print("Combined Signal Evaluation (All Strategies)")
    print("="*60)
    actual_cols = [c for c in all_cols if c in splits["train"].columns]
    report, _ = evaluate_signal_panel(splits, actual_cols, roi_col="roi_res")
    print(report)

if __name__ == "__main__":
    main()
