"""
Preprocessing package: raw data loading and trade enrichment.

Public API
----------
From ``loader``:
    day_folders, load_market, load_trades, load_market_and_trades,
    load_all_markets_and_trades

From ``trades``:
    build_token_lookup, build_raw_dataframe, aggregate_trades,
    compute_wallet_summary, filter_top_wallets
"""

from .loader import (
    day_folders,
    load_market,
    load_trades,
    load_market_and_trades,
    load_all_markets_and_trades,
)
from .trades import (
    build_token_lookup,
    build_raw_dataframe,
    aggregate_trades,
    compute_wallet_summary,
    filter_top_wallets,
)

__all__ = [
    "day_folders",
    "load_market",
    "load_trades",
    "load_market_and_trades",
    "load_all_markets_and_trades",
    "build_token_lookup",
    "build_raw_dataframe",
    "aggregate_trades",
    "compute_wallet_summary",
    "filter_top_wallets",
]
