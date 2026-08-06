"""W0.2 verify — sanity checks for the resolution source mapping.

Reads weather_fv/w0_resolution_sources.json and asserts:
- at least 50 cities mapped
- at least 90% have a station_icao, station, or other_id
- at most 5 cities with no resolution_source
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE / "w0_resolution_sources.json"


def main() -> int:
    d = json.loads(JSON_PATH.read_text())
    n = len(d)
    assert n >= 50, f"too few cities mapped: {n}"
    have_any = sum(
        1 for v in d.values()
        if v.get("station_icao") or v.get("station") or v.get("station_other_id")
    )
    assert have_any / n >= 0.90, (
        f"only {have_any}/{n} have any station info"
    )
    no_source = sum(1 for v in d.values() if not v.get("resolution_source"))
    assert no_source <= 5, f"too many cities without source: {no_source}"
    src_dist: dict[str, int] = {}
    for v in d.values():
        s = v.get("resolution_source") or "None"
        src_dist[s] = src_dist.get(s, 0) + 1
    print(f"cities={n} with_station={have_any} no_source={no_source}")
    print("source distribution:")
    for s, c in sorted(src_dist.items(), key=lambda x: -x[1]):
        print(f"  {s:>20s}: {c}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
