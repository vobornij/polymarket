"""Reconstruct the per-bucket share-depth cap ``bucket_avail_copy_qty``.

``shard_processor.py`` computes ``avail_copy_qty = max`` over the fills of a
bucket (tx_hash x wallet x side x token_id) but drops it from the output
processed files, keeping only the *sum*med ``avail_copy_total_vol``.  A summed
dollar depth over-counts multi-fill buckets and is unusable as an honest cap
for copy scale > 1.  The per-fill ``avail_copy_qty`` still exists in the
enriched shards, so this script rebuilds the bucket-level max into a lookup
parquet ``data/bucket_depth.parquet``.  ``build_lookup`` loads the cached
parquet when it exists and only rebuilds from shards on the first run (or
after ``clear_cache``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_ENRICHED_DIR = Path(__file__).resolve().parents[3] / "data" / "trades_polygon_enriched"
_OUT = Path(__file__).resolve().parents[3] / "data" / "bucket_depth.parquet"
_KEYS = ["tx_hash", "wallet", "side", "token_id"]


def build_lookup(enriched_dir: Path = _ENRICHED_DIR) -> pd.DataFrame:
    if _OUT.exists():
        print(f"loading cached depth lookup: {_OUT}", flush=True)
        return pd.read_parquet(_OUT)
    files = sorted(enriched_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No enriched shards in {enriched_dir}")
    parts = []
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=_KEYS + ["avail_copy_qty"])
        parts.append(df.groupby(_KEYS, dropna=False)["avail_copy_qty"].max())
        print(f"  [{i+1}/{len(files)}] {f.name}: {len(df):,} rows", flush=True)
        del df
    out = pd.concat(parts).groupby(level=_KEYS).max().rename("bucket_avail_copy_qty").reset_index()
    out.to_parquet(_OUT, index=False)
    print(f"wrote {_OUT}: {len(out):,} buckets", flush=True)
    return out


def clear_cache() -> None:
    """Delete the cached depth lookup so the next ``build_lookup`` rebuilds it."""
    if _OUT.exists():
        _OUT.unlink()
        print(f"cleared depth lookup cache: {_OUT}", flush=True)
    else:
        print(f"no depth lookup cache at {_OUT}", flush=True)


if __name__ == "__main__":
    lookup = build_lookup()
    print(lookup["bucket_avail_copy_qty"].describe(percentiles=[.5, .9, .99]).to_string())
