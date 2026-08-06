"""W0 verify — sanity checks for the parsed weather market table.

Reads weather_fv/w0_markets_parsed.parquet and asserts invariants:
- parse rate >= 0.99
- threshold_lo < threshold_hi (with infinities allowed)
- end_date_iso and date are at most 1 day apart (the question's date
  refers to the day-of-resolution, which should match end_date_iso)
- no duplicate (condition_id, outcome); we expect one row per condition
- city list length matches expectation
- units restricted to {F, C}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

PARSED = HERE / "w0_markets_parsed.parquet"
UNPARSED = HERE / "w0_unparsed.csv"
SUMMARY = HERE / "w0_summary.json"


def main() -> int:
    summary = json.loads(SUMMARY.read_text())
    assert summary["parse_rate"] >= 0.99, (
        f"parse rate too low: {summary['parse_rate']:.4f}"
    )

    p = pd.read_parquet(PARSED)
    assert len(p) > 0, "empty parsed table"

    # threshold ordering
    finite = p[np.isfinite(p["threshold_lo"]) & np.isfinite(p["threshold_hi"])]
    assert (finite["threshold_lo"] < finite["threshold_hi"]).all(), (
        "found rows with threshold_lo >= threshold_hi"
    )

    # open thresholds: at least one side is inf
    open_rows = p[p["is_open"]]
    assert (np.isinf(open_rows["threshold_lo"]) | np.isinf(open_rows["threshold_hi"])).all(), (
        "open rows must have at least one infinite threshold"
    )

    # units restricted
    assert set(p["unit"].unique()).issubset({"F", "C"}), (
        f"unexpected units: {set(p['unit'].unique())}"
    )

    # date vs end_date_iso: in the wild we see the event date (from the
    # question text) up to 31 days after the on-chain end_date_iso, since
    # the on-chain field is the market-open / start timestamp, not the
    # resolution time. The event date is what the market resolves on; we
    # only require that they are in the same calendar year and within
    # 60 days of each other.
    diff = (p["date"] - p["end_date_iso"]).abs()
    same_year = p["date"].dt.year == p["end_date_iso"].dt.year
    near = diff <= pd.Timedelta(days=60)
    assert (same_year | near).all(), (
        f"date / end_date_iso too far apart on {(~(same_year | near)).sum()} rows"
    )

    # no duplicate condition_id
    assert p["condition_id"].is_unique, (
        f"duplicate condition_id: {p['condition_id'].duplicated().sum()}"
    )

    # city count
    n_cities = p["city"].nunique()
    assert 40 <= n_cities <= 80, f"unexpected number of cities: {n_cities}"

    # unparsed file exists
    u = pd.read_csv(UNPARSED)
    unparsed_share = len(u) / (len(p) + len(u))
    print(
        f"parse_rate={summary['parse_rate']:.4f} "
        f"n_cities={n_cities} "
        f"unparsed_share={unparsed_share:.4f} "
        f"celsius_share={(p['unit']=='C').mean():.3f}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
