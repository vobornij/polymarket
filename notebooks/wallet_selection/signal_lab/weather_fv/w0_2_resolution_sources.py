"""W0.2 — fetch resolution source from market descriptions.

For each city in W0.1's parsed table, fetch one example market's description
from the CLOB API and parse out the resolution source (typically Wunderground)
and the station name (e.g., ``KLGA`` for LaGuardia). One call per city is
sufficient because the resolution source and station are consistent across a
city's markets.

Inputs:
    weather_fv/w0_markets_parsed.parquet  (for the city list)
Outputs:
    weather_fv/w0_resolution_sources.json
    weather_fv/w0_resolution_sources.csv
    weather_fv/w0_unfetched_cities.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PARSED = HERE / "w0_markets_parsed.parquet"
OUT_JSON = HERE / "w0_resolution_sources.json"
OUT_CSV = HERE / "w0_resolution_sources.csv"
OUT_UNFETCHED = HERE / "w0_unfetched_cities.csv"

CLOB_URL = "https://clob.polymarket.com/markets/{cid}"
USER_AGENT = "Mozilla/5.0 (signal_lab research)"

STATION_PATTERNS = [
    # Wunderground URL embeds the ICAO 4-letter code
    re.compile(r"wunderground\.com/history/daily/[^\s/]+/[^\s/]+/([A-Z]{4})"),
    # NOAA URL embeds the site/ICAO code
    re.compile(r"weather\.gov/wrh/timeseries\?site=([A-Z]{4})"),
    # Taipei CWA URL embeds a station ID
    re.compile(r"cwa\.gov\.tw[^\s]*?ID=(\d+)"),
    # explicit station phrase
    re.compile(
        r"recorded (?:at|by)\s+(?:the\s+)?([A-Z][A-Za-zÀ-ſ'\-]+(?:\s+[A-Z][A-Za-zÀ-ſ'\-\.]+){0,4})"
        r"(?:\s+(?:International|Intl)\s+Airport|\s+Airport|\s+Air Base|\s+Base|\s+Field)?\s+Station"
    ),
    re.compile(r"\b([A-Z]{4})\s+Station"),
]
SOURCE_PATTERNS = [
    re.compile(r"resolution source[^.]*?be\s+information from\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})"),
]


def fetch_description(condition_id: str, max_attempts: int = 3) -> str | None:
    url = CLOB_URL.format(cid=condition_id)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                d = json.loads(resp.read())
                return d.get("description", "")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
                continue
            return None
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(1)
                continue
            return None
    return None


def parse_source_station(description: str) -> dict:
    station = None
    station_icao = None
    station_other_id = None
    for pat in STATION_PATTERNS:
        m = pat.search(description or "")
        if m:
            val = m.group(1).strip()
            if pat.pattern.startswith("cwa"):
                station_other_id = val
            elif len(val) == 4 and val.isupper():
                station_icao = val
            else:
                station = val
            break
    source = None
    for pat in SOURCE_PATTERNS:
        m = pat.search(description or "")
        if m:
            source = m.group(1).strip()
            break
    return {
        "resolution_source": source,
        "station": station,
        "station_icao": station_icao,
        "station_other_id": station_other_id,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate-delay", type=float, default=0.1)
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()

    parsed = pd.read_parquet(PARSED, columns=["condition_id", "city"])
    one_per_city = parsed.drop_duplicates("city").reset_index(drop=True)
    if args.sample:
        one_per_city = one_per_city.head(args.sample)
    print(f"cities to fetch: {len(one_per_city):,}")

    sources = {}
    unfetched = []
    for _, r in one_per_city.iterrows():
        cid = r["condition_id"]
        city = r["city"]
        desc = fetch_description(cid)
        if desc is None:
            unfetched.append(city)
            print(f"  [skip] {city}")
            continue
        parsed_d = parse_source_station(desc)
        sources[city] = {
            "condition_id": cid,
            "resolution_source": parsed_d["resolution_source"],
            "station": parsed_d["station"],
            "station_icao": parsed_d["station_icao"],
            "station_other_id": parsed_d["station_other_id"],
        }
        print(f"  {city:18s} src={str(parsed_d['resolution_source']):>14s} station={str(parsed_d['station'])[:35]:35s} icao={str(parsed_d['station_icao']):5s} other_id={parsed_d['station_other_id']}")
        time.sleep(args.rate_delay)

    OUT_JSON.write_text(json.dumps(sources, indent=2))
    pd.DataFrame([
        {"city": c, "condition_id": v["condition_id"],
         "resolution_source": v["resolution_source"],
         "station": v["station"],
         "station_icao": v["station_icao"],
         "station_other_id": v["station_other_id"]}
        for c, v in sources.items()
    ]).to_csv(OUT_CSV, index=False)
    pd.DataFrame({"city": unfetched}).to_csv(OUT_UNFETCHED, index=False)
    print(f"\nWrote {OUT_JSON} ({len(sources)} cities)")
    print(f"Unfetched: {len(unfetched)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
