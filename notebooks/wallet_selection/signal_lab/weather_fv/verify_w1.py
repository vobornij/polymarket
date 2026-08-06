"""W1 verify — sanity checks for the market self-calibration baseline.

Reads weather_fv/w1_calibration.csv and asserts:
- global Brier in (0.0, 0.20) (well below the always-0.5 baseline of 0.25)
- at least 10 buckets with n >= 100
- reliability gap bounded in [-0.10, +0.10]
- output CSV has expected columns
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "w1_calibration.csv"
SUMMARY = HERE / "w1_summary.json"

EXPECTED_COLS = {
    "lead_bin", "price_bin", "n", "mean_price", "outcome_rate",
    "brier", "reliability_gap", "lead_label",
}


def main() -> int:
    summary = json.loads(SUMMARY.read_text())
    assert 0.0 < summary["global_brier"] < 0.20, (
        f"global Brier out of range: {summary['global_brier']}"
    )
    df = pd.read_csv(CSV)
    assert EXPECTED_COLS.issubset(df.columns), (
        f"missing columns: {EXPECTED_COLS - set(df.columns)}"
    )
    big_buckets = (df["n"] >= 100).sum()
    assert big_buckets >= 10, f"only {big_buckets} buckets with n>=100"
    assert df["reliability_gap"].abs().max() < 0.10, (
        f"reliability gap exceeded 10% on {(df['reliability_gap'].abs() > 0.10).sum()} rows"
    )
    print(
        f"global_brier={summary['global_brier']:.4f} "
        f"n_trades={summary['n_trades']:,} "
        f"n_buckets={summary['n_buckets']} "
        f"n_big_buckets={big_buckets}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
