"""W0.1 — local parser for weather-market question text.

Extracts: city, date, metric (max/min), threshold_lo, threshold_hi, unit, is_open.
Does NOT call any external API. Resolution source is filled by W0.2.

Run modes:
    python w0_1_parse_markets.py --sample 500
    python w0_1_parse_markets.py --full

Output:
    weather_fv/w0_markets_parsed.parquet
    weather_fv/w0_unparsed.csv     (rows that did not match the regex)
    weather_fv/w0_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SIGNAL_LAB = HERE.parent
NOTEBOOKS = SIGNAL_LAB.parent
PROJECT = NOTEBOOKS.parent.parent
sys.path.insert(0, str(PROJECT))

MARKETS_PATH = PROJECT / "data/markets_processed/markets.parquet"
OUT_PARQUET = HERE / "w0_markets_parsed.parquet"
OUT_UNPARSED = HERE / "w0_unparsed.csv"
OUT_SUMMARY = HERE / "w0_summary.json"

CITY_ALIASES = {
    "NYC": "New York City",
    "Washington DC": "Washington",
    "Washington D.C.": "Washington",
    "DC": "Washington",
    "LA": "Los Angeles",
}


def _strip_arc_prefix(q: str) -> str:
    """Some questions have an ``arc`` prefix from upstream pipelines."""
    if q.startswith("arcWill "):
        return q[len("arc") :]
    return q


# Two main regex flavours:
#   (a) "Will the highest/lowest temperature in {city} be {rhs} on {date}?"
#   (b) "Will the high/low in {city} be {rhs} on {date}?"  (abbreviated)
PAT_A = re.compile(
    r"^Will the (highest|lowest) temperature in (.+?) be (.+?) on (.+?)\?$"
)
PAT_B = re.compile(
    r"^Will the (high|low) in (.+?) be (.+?) on (.+?)\?$"
)
PAT_RANGE = re.compile(r"^between\s*(-?\d+)\s*[\-\u2013\u2014]\s*(-?\d+)(°?[FC])$")
PAT_OPEN_HIGH = re.compile(r"^(-?\d+)(°?[FC])\s+or\s+higher$")
PAT_OPEN_LOW = re.compile(r"^(-?\d+)(°?[FC])\s+or\s+(?:lower|below)$")
PAT_EXACT = re.compile(r"^(-?\d+)(°?[FC])$")
PAT_RANGE_NEG = re.compile(r"^(-?\d+)°\s*[\-\u2013\u2014]\s*(-?\d+)(°?[FC])$")

UNIT_MAP = {"F": "F", "°F": "F", "C": "C", "°C": "C"}

MONTH_LOOKUP = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11,
    "December": 12,
}


def normalise_unit(u: str) -> str:
    return UNIT_MAP[u]


def normalise_city(c: str) -> str:
    c = c.strip()
    if c.endswith(" be"):
        c = c[: -len(" be")]
    return CITY_ALIASES.get(c, c)


def normalise_metric(m: str) -> str:
    return {"highest": "max", "high": "max", "lowest": "min", "low": "min"}[m]


def parse_rhs(rhs: str) -> dict | None:
    """Return {threshold_lo, threshold_hi, is_open, unit} or None."""
    rhs = rhs.strip()

    m = PAT_EXACT.match(rhs)
    if m:
        val = int(m.group(1))
        unit = normalise_unit(m.group(2))
        return {
            "threshold_lo": float(val),
            "threshold_hi": float(val + 1),  # +1 unit means "equal to val"
            "is_open": False,
            "unit": unit,
        }

    m = PAT_RANGE.match(rhs)
    if not m:
        m = PAT_RANGE_NEG.match(rhs)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2))
        unit = normalise_unit(m.group(3))
        return {
            "threshold_lo": float(lo),
            "threshold_hi": float(hi + 1),
            "is_open": False,
            "unit": unit,
        }

    m = PAT_OPEN_HIGH.match(rhs)
    if m:
        val = int(m.group(1))
        unit = normalise_unit(m.group(2))
        return {
            "threshold_lo": float(val),
            "threshold_hi": float("inf"),
            "is_open": True,
            "unit": unit,
        }

    m = PAT_OPEN_LOW.match(rhs)
    if m:
        val = int(m.group(1))
        unit = normalise_unit(m.group(2))
        return {
            "threshold_lo": float("-inf"),
            "threshold_hi": float(val + 1),
            "is_open": True,
            "unit": unit,
        }

    return None


def parse_question(q: str) -> dict | None:
    if not isinstance(q, str):
        return None
    q = _strip_arc_prefix(q).strip()
    for pat in (PAT_A, PAT_B):
        m = pat.match(q)
        if m:
            metric = normalise_metric(m.group(1))
            city = normalise_city(m.group(2))
            rhs = parse_rhs(m.group(3))
            if rhs is None:
                return None
            date_str = m.group(4).strip()
            return {
                "city": city,
                "metric": metric,
                **rhs,
                "date_str": date_str,
            }
    return None


def build_date(date_str: str, fallback: pd.Timestamp | None) -> pd.Timestamp | None:
    """Parse 'February 2', 'February 24', etc.; fall back to end_date_iso."""
    m = re.match(r"^([A-Za-z]+)\s+(\d+)$", date_str)
    if m and fallback is not None and pd.notna(fallback):
        month = MONTH_LOOKUP.get(m.group(1))
        day = int(m.group(2))
        if month:
            try:
                year = int(fallback.year) if hasattr(fallback, "year") else None
                if year is None:
                    return None
                return pd.Timestamp(year=year, month=month, day=day, tz="UTC")
            except (ValueError, TypeError):
                return fallback
    if fallback is not None and pd.notna(fallback):
        return fallback
    return None


def run(sample: int | None, full: bool) -> dict:
    cols = ["condition_id", "question", "primary_tag", "end_date_iso", "tags"]
    markets = pd.read_parquet(MARKETS_PATH, columns=cols)
    w = markets[markets["primary_tag"] == "Weather"].copy()
    if not full and sample is not None:
        w = w.sample(n=min(sample, len(w)), random_state=0)
    print(f"weather markets to parse: {len(w):,}")

    parsed_rows = []
    unparsed_rows = []
    for _, r in w.iterrows():
        out = parse_question(r["question"])
        if out is None:
            unparsed_rows.append(
                {
                    "condition_id": r["condition_id"],
                    "question": r["question"],
                    "end_date_iso": r["end_date_iso"],
                }
            )
            continue
        end_ts = pd.to_datetime(r["end_date_iso"], utc=True, errors="coerce")
        date_ts = build_date(out.pop("date_str"), end_ts)
        if date_ts is None or pd.isna(date_ts):
            unparsed_rows.append(
                {
                    "condition_id": r["condition_id"],
                    "question": r["question"],
                    "end_date_iso": r["end_date_iso"],
                }
            )
            continue
        out["condition_id"] = r["condition_id"]
        out["end_date_iso"] = end_ts
        out["date"] = date_ts
        out["tags"] = list(r["tags"]) if hasattr(r["tags"], "__iter__") else []
        parsed_rows.append(out)

    parsed = pd.DataFrame(parsed_rows)
    unparsed = pd.DataFrame(unparsed_rows)
    print(f"parsed: {len(parsed):,}, unparsed: {len(unparsed):,}")
    parsed.to_parquet(OUT_PARQUET, index=False)
    unparsed.to_csv(OUT_UNPARSED, index=False)

    summary = {
        "n_input": int(len(w)),
        "n_parsed": int(len(parsed)),
        "n_unparsed": int(len(unparsed)),
        "parse_rate": float(len(parsed) / max(len(w), 1)),
        "n_cities": int(parsed["city"].nunique()) if len(parsed) else 0,
        "cities": (
            parsed["city"].value_counts().to_dict() if len(parsed) else {}
        ),
        "metrics": (
            parsed["metric"].value_counts().to_dict() if len(parsed) else {}
        ),
        "units": (
            parsed["unit"].value_counts().to_dict() if len(parsed) else {}
        ),
        "is_open_share": (
            float(parsed["is_open"].mean()) if len(parsed) else 0.0
        ),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(f"summary: {OUT_SUMMARY}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sample", type=int, default=500)
    g.add_argument("--full", action="store_true")
    args = ap.parse_args()
    sample = None if args.full else args.sample
    summary = run(sample=sample, full=args.full)
    print(json.dumps({k: v for k, v in summary.items() if k != "cities"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
