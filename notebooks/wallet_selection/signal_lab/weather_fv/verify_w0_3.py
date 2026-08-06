"""W0.3 verify — sanity checks for the combined enriched market table.

Reads weather_fv/markets_enriched.parquet and asserts:
- n_markets >= 95,000
- n_with_resolution_source >= 90% of n_markets
- n_with_station_icao + n_with_station + n_with_station_other_id >= 95% of n_markets
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PARQUET = HERE / "markets_enriched.parquet"
SUMMARY = HERE / "w0_enriched_summary.json"


def main() -> int:
    s = json.loads(SUMMARY.read_text())
    n = s["n_markets"]
    assert n >= 95_000, f"too few markets: {n}"
    assert s["n_with_resolution_source"] / n >= 0.90, (
        f"only {s['n_with_resolution_source']/n:.1%} have resolution_source"
    )
    have_any = (
        s["n_with_station_icao"] + s["n_with_station"]
    )
    assert have_any / n >= 0.95, (
        f"only {have_any/n:.1%} have any station identifier"
    )
    df = pd.read_parquet(PARQUET, columns=["condition_id", "city", "metric", "unit"])
    assert df["condition_id"].is_unique, "duplicate condition_id"
    assert df["metric"].isin(["max", "min"]).all(), "metric restricted to max/min"
    assert df["unit"].isin(["F", "C"]).all(), "unit restricted to F/C"
    print(json.dumps(s, indent=2))
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
