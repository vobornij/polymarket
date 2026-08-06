"""W0.3 — combine W0.1 (parsed markets) + W0.2 (resolution sources).

Outputs:
    weather_fv/markets_enriched.parquet
    weather_fv/w0_enriched_summary.json

Each row: condition_id, city, date, metric, threshold_lo, threshold_hi,
unit, is_open, resolution_source, station, station_icao, station_other_id.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PARSED = HERE / "w0_markets_parsed.parquet"
SOURCES = HERE / "w0_resolution_sources.json"
OUT_PARQUET = HERE / "markets_enriched.parquet"
OUT_SUMMARY = HERE / "w0_enriched_summary.json"


def main() -> int:
    parsed = pd.read_parquet(PARSED)
    sources = json.loads(SOURCES.read_text())
    src_df = pd.DataFrame([
        {
            "city": c,
            "resolution_source": v.get("resolution_source"),
            "station": v.get("station"),
            "station_icao": v.get("station_icao"),
            "station_other_id": v.get("station_other_id"),
        }
        for c, v in sources.items()
    ])
    enriched = parsed.merge(src_df, on="city", how="left")
    enriched.to_parquet(OUT_PARQUET, index=False)

    n = len(enriched)
    n_with_icao = enriched["station_icao"].notna().sum()
    n_with_station = enriched["station"].notna().sum()
    n_with_source = enriched["resolution_source"].notna().sum()
    summary = {
        "n_markets": int(n),
        "n_with_resolution_source": int(n_with_source),
        "n_with_station_icao": int(n_with_icao),
        "n_with_station": int(n_with_station),
        "city_count": int(enriched["city"].nunique()),
        "src_distribution": (
            enriched["resolution_source"].value_counts(dropna=False).to_dict()
        ),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {OUT_PARQUET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
