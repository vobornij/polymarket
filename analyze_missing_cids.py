import pandas as pd
import os
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

STAGE2_DIR = "/Users/vobornij/projects/polymarket/data/polygon_trades_processed"
BACKTEST_DIR = "/Users/vobornij/projects/polymarket/data/polybot_backtest/vwap_fix_test"
WALLET = "0xb9012e0d9b60d3920286309328b935cdfa609fc4"
CUTOFF = pd.Timestamp("2026-08-01", tz=timezone.utc)

print("=== Loading stage2 parquet files ===")
dfs = []
for f in sorted(Path(STAGE2_DIR).glob("*.parquet")):
    df = pd.read_parquet(f)
    dfs.append(df)
    print(f"  {f.name}: {len(df):,} rows")

stage2 = pd.concat(dfs, ignore_index=True)
print(f"\nTotal stage2 rows: {len(stage2):,}")
print(f"Columns: {list(stage2.columns)}")
print(f"\nStage2 unique condition_ids: {stage2['condition_id'].nunique():,}")

# Filter for our wallet
wallet_stage2 = stage2[stage2["wallet"] == WALLET].copy()
print(f"\n=== Wallet {WALLET} ===")
print(f"Total rows: {len(wallet_stage2):,}")
print(f"Unique condition_ids: {wallet_stage2['condition_id'].nunique():,}")

# Understand train/test split
if "is_train" in wallet_stage2.columns:
    print(f"\nIs train distribution:")
    print(wallet_stage2["is_train"].value_counts().to_string())

# Understand side distribution
print(f"\nSide distribution:")
print(wallet_stage2["side"].value_counts().to_string())

# Filter for BUY trades
wallet_buys = wallet_stage2[wallet_stage2["side"] == "BUY"]
print(f"\nTotal BUY rows: {len(wallet_buys):,}")
print(f"Unique BUY condition_ids: {wallet_buys['condition_id'].nunique():,}")

# Test set (not train)
if "is_train" in wallet_buys.columns:
    test_buys = wallet_buys[wallet_buys["is_train"] == False]
    print(f"\nTest set BUY rows: {len(test_buys):,}")
    print(f"Test set BUY unique condition_ids: {test_buys['condition_id'].nunique():,}")
else:
    test_buys = wallet_buys
    print("No is_train column, using all buys")

# Now check last_condition_trade_ts
if "last_condition_trade_ts" in stage2.columns:
    # Get unique condition_ids with their last_condition_trade_ts
    last_ts = stage2.groupby("condition_id")["last_condition_trade_ts"].first()
    print(f"\n=== last_condition_trade_ts analysis ===")
    print(f"Conditions with null last_condition_trade_ts: {last_ts.isna().sum()}")
    print(f"Conditions with non-null last_condition_trade_ts: {last_ts.notna().sum()}")
    
    # For test set CIDs, check their last_condition_trade_ts
    test_cids = test_buys["condition_id"].unique()
    print(f"\nTest set CIDs: {len(test_cids)}")
    
    cid_last_ts = stage2[stage2["condition_id"].isin(test_cids)].groupby("condition_id")["last_condition_trade_ts"].first()
    print(f"Test CIDs with last_condition_trade_ts: {cid_last_ts.notna().sum()}")
    print(f"Test CIDs without last_condition_trade_ts: {cid_last_ts.isna().sum()}")
    
    # Check how many are before cutoff
    if cid_last_ts.notna().any():
        ts_values = pd.to_datetime(cid_last_ts.dropna(), utc=True)
        print(f"\nTest CIDs last_condition_trade_ts distribution:")
        print(f"  Before cutoff ({CUTOFF}): {(ts_values < CUTOFF).sum()}")
        print(f"  After cutoff ({CUTOFF}): {(ts_values >= CUTOFF).sum()}")
        
        # Show some examples
        before_cutoff = ts_values[ts_values < CUTOFF].sort_values()
        if len(before_cutoff) > 0:
            print(f"\nExamples of test CIDs before cutoff:")
            for cid, ts in before_cutoff.head(10).items():
                print(f"  {cid}: {ts}")

# Load backtest output
print(f"\n=== Backtest output ===")
contracts_dir = Path(BACKTEST_DIR) / "contracts"
backtest_cids = set()
if contracts_dir.exists():
    backtest_cids = {d.name for d in contracts_dir.iterdir() if d.is_dir()}
    print(f"Backtest condition_ids: {len(backtest_cids)}")

# Load summaries.json
summaries_path = Path(BACKTEST_DIR) / "summaries.json"
backtest_summary_cids = set()
if summaries_path.exists():
    with open(summaries_path) as f:
        summaries = json.load(f)
    backtest_summary_cids = {s["conditionId"] for s in summaries}
    print(f"Summaries condition_ids: {len(backtest_summary_cids)}")
    
    # Check fill counts
    fill_counts = [s.get("fillCount", 0) for s in summaries]
    print(f"Conditions with fills > 0: {sum(1 for c in fill_counts if c > 0)}")
    print(f"Conditions with fills = 0: {sum(1 for c in fill_counts if c == 0)}")

# Compare test CIDs with backtest CIDs
test_cid_set = set(test_cids)
in_backtest = test_cid_set & backtest_cids
not_in_backtest = test_cid_set - backtest_cids
print(f"\n=== Test CIDs in backtest: {len(in_backtest)} ===")
print(f"=== Test CIDs NOT in backtest: {len(not_in_backtest)} ===")

# For CIDs not in backtest, check last_condition_trade_ts
if len(not_in_backtest) > 0 and "last_condition_trade_ts" in stage2.columns:
    print(f"\n=== CIDs NOT in backtest - last_condition_trade_ts analysis ===")
    missing_ts = stage2[stage2["condition_id"].isin(not_in_backtest)].groupby("condition_id")["last_condition_trade_ts"].first()
    
    # Check if all are null or before cutoff
    null_count = missing_ts.isna().sum()
    non_null = missing_ts.dropna()
    if len(non_null) > 0:
        ts_values = pd.to_datetime(non_null, utc=True)
        before_count = (ts_values < CUTOFF).sum()
        after_count = (ts_values >= CUTOFF).sum()
        print(f"Missing CIDs with null last_condition_trade_ts: {null_count}")
        print(f"Missing CIDs with ts before cutoff: {before_count}")
        print(f"Missing CIDs with ts after cutoff: {after_count}")
        
        if before_count > 0:
            print(f"\nThese {before_count} CIDs are filtered out by lastConditionTradeTsCutoff!")
            print("Examples:")
            before = ts_values[ts_values < CUTOFF].sort_values()
            for cid, ts in before.head(10).items():
                # Get total trades for this CID across all wallets
                total_trades = len(stage2[stage2["condition_id"] == cid])
                wallet_trades = len(stage2[(stage2["condition_id"] == cid) & (stage2["wallet"] == WALLET)])
                print(f"  {cid}: ts={ts}, total_trades={total_trades}, wallet_trades={wallet_trades}")
        
        if after_count > 0:
            print(f"\nThese {after_count} CIDs have ts after cutoff but are NOT in backtest (other filter?):")
            after = ts_values[ts_values >= CUTOFF].sort_values()
            for cid, ts in after.head(20).items():
                total_trades = len(stage2[stage2["condition_id"] == cid])
                wallet_trades = len(stage2[(stage2["condition_id"] == cid) & (stage2["wallet"] == WALLET)])
                # Check primary tag - this would need markets.parquet
                print(f"  {cid}: ts={ts}, total_trades={total_trades}, wallet_trades={wallet_trades}")
    else:
        print(f"All missing CIDs have null last_condition_trade_ts: {null_count}")
        print("These might be filtered by primaryTagFilter (Weather) or other criteria")
else:
    print("No missing CIDs or no last_condition_trade_ts column")

# Show what IS in backtest vs test CIDs
print(f"\n=== Summary ===")
print(f"Test set CIDs: {len(test_cid_set)}")
print(f"CIDs in backtest: {len(backtest_cids)}")
print(f"Test CIDs also in backtest: {len(in_backtest)}")
print(f"Test CIDs NOT in backtest: {len(not_in_backtest)}")
print(f"\nBreakdown of why CIDs are missing:")
if len(not_in_backtest) > 0 and "last_condition_trade_ts" in stage2.columns:
    missing_ts = stage2[stage2["condition_id"].isin(not_in_backtest)].groupby("condition_id")["last_condition_trade_ts"].first()
    null_cids = missing_ts[missing_ts.isna()].index
    before_cutoff_cids = []
    after_cutoff_cids = []
    for cid in missing_ts.dropna().index:
        ts = pd.to_datetime(missing_ts[cid], utc=True)
        if ts < CUTOFF:
            before_cutoff_cids.append(cid)
        else:
            after_cutoff_cids.append(cid)
    
    print(f"  1. Null last_condition_trade_ts: {len(null_cids)} CIDs")
    print(f"  2. Before cutoff (2026-08-01): {len(before_cutoff_cids)} CIDs")
    print(f"  3. After cutoff but not in backtest (primaryTagFilter?): {len(after_cutoff_cids)} CIDs")
