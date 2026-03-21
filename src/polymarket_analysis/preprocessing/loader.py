"""
Raw data loading utilities for Polymarket trade data.

Reads JSON market definitions and JSONL trade files from dated folder trees
(``data/trades_raw/YYYY-MM-DD/``).  Parallel I/O is used by default.
"""

from __future__ import annotations

import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

# Fast JSON parser when available
try:
    import orjson as _orjson

    def _json_loads(s: bytes) -> dict:
        return _orjson.loads(s)

    def _json_load(fh) -> dict:
        return _orjson.loads(fh.read())

except ImportError:
    _orjson = None  # type: ignore[assignment]

    def _json_loads(s: bytes) -> dict:  # type: ignore[misc]
        return json.loads(s)

    def _json_load(fh) -> dict:  # type: ignore[misc]
        return json.load(fh)


# ---------------------------------------------------------------------------
# Folder enumeration
# ---------------------------------------------------------------------------

def day_folders(
    root: Path,
    start: datetime.date,
    end: datetime.date,
) -> list[Path]:
    """Return sorted list of ``YYYY-MM-DD`` folders in *root* within [start, end].

    Folders whose names cannot be parsed as ISO dates are silently skipped.
    """
    folders: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        try:
            d = datetime.date.fromisoformat(p.name)
        except ValueError:
            continue
        if start <= d <= end:
            folders.append(p)
    return folders


# ---------------------------------------------------------------------------
# Single-file loaders
# ---------------------------------------------------------------------------

def load_market(json_path: Path) -> dict:
    """Load a single market definition from its ``.json`` file."""
    with json_path.open("rb") as fh:
        return _json_load(fh)


def load_trades(jsonl_path: Path) -> list[dict]:
    """Load all trades from a ``.jsonl`` file into a list of dicts.

    The ``asset`` field is coerced to ``str`` for safe downstream use.
    """
    rows: list[dict] = []
    with jsonl_path.open("rb") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            obj = _json_loads(stripped)
            obj["asset"] = str(obj["asset"])
            rows.append(obj)
    return rows


# ---------------------------------------------------------------------------
# Paired loader
# ---------------------------------------------------------------------------

def load_market_and_trades(json_path: Path) -> tuple[dict, list[dict]] | None:
    """Load a market definition and its trades from a single JSON/JSONL pair.

    Returns ``None`` when the ``.jsonl`` counterpart does not exist.
    """
    jsonl_path = json_path.with_suffix(".jsonl")
    if not jsonl_path.exists():
        return None
    market = load_market(json_path)
    trades = load_trades(jsonl_path)
    return market, trades


# ---------------------------------------------------------------------------
# Bulk parallel loader
# ---------------------------------------------------------------------------

def load_all_markets_and_trades(
    folders: Iterable[Path],
    num_workers: int | None = None,
) -> tuple[dict[str, dict], list[dict]]:
    """Load all markets and trades from a list of day-folders in parallel.

    Parameters
    ----------
    folders:
        Day-folder paths as returned by :func:`day_folders`.
    num_workers:
        Thread-pool size.  Defaults to ``min(32, cpu_count * 4)``.

    Returns
    -------
    markets : dict[condition_id → market_dict]
        Only the first occurrence of each ``condition_id`` is kept.
    all_trades : list[dict]
        Every trade record loaded across all folders and markets.
    """
    folders = list(folders)
    all_json_paths: list[Path] = [
        json_path
        for folder in folders
        for json_path in sorted(folder.glob("*.json"))
    ]

    if num_workers is None:
        num_workers = min(32, (os.cpu_count() or 4) * 4)

    markets: dict[str, dict] = {}
    all_trades: list[dict] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(load_market_and_trades, p): p
            for p in all_json_paths
        }
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            market, trades = result
            condition_id = market["condition_id"]
            if condition_id not in markets:
                markets[condition_id] = market
            all_trades.extend(trades)

    return markets, all_trades
