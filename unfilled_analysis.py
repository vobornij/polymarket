import pandas as pd
import glob
import numpy as np

WALLETS = {
    '0x04b3': '0x04b3f873b003eba3774df1e5e47a370d27ee1b7e',
    '0x9196': '0x919698b19427cbe6945b0dc823f2d9e126a4d934',
    '0x6011': '0x6011655c4afb76f36dd1b08a137a1ba73466b31e',
    '0xb901': '0xb9012e0d9b60d3920286309328b935cdfa609fc4',
    '0xa2ed': '0xa2eded4da718244771d4f2b7810da45533de9d98',
}
SHORT_TO_FULL = {k: v.lower() for k, v in WALLETS.items()}
FULL_TO_SHORT = {v.lower(): k for k, v in SHORT_TO_FULL.items()}

# 1. Load enriched data
print("Loading enriched data...")
enriched_files = sorted(glob.glob('/Users/vobornij/projects/polymarket/data/trades_polygon_enriched/enriched_*.parquet'))
all_full_wallets = set(SHORT_TO_FULL.values())
dfs = []
for f in enriched_files:
    df = pd.read_parquet(f)
    df = df[df['wallet'].str.lower().isin(all_full_wallets)]
    if len(df) > 0:
        dfs.append(df)
enriched = pd.concat(dfs, ignore_index=True)
print(f"Enriched rows for target wallets: {len(enriched)}")

# Filter BUY side
enriched = enriched[enriched['side'] == 'BUY'].copy()
print(f"BUY-side rows: {len(enriched)}")

# Compute copyable_pnl
enriched['final_price'] = enriched['token_winner'].map({True: 1.0, False: 0.0})
enriched['copyable_pnl'] = enriched['copyable_qty_5m_100'] * (enriched['final_price'] - enriched['price'])
enriched['short_wallet'] = enriched['wallet'].str.lower().map(FULL_TO_SHORT)

# 2. Load backtest fills
print("Loading backtest fills...")
fill_files = glob.glob('/Users/vobornij/projects/polymarket/data/polybot_backtest/vwap_fix_test/contracts/*/fills.parquet')
fills = pd.concat([pd.read_parquet(f) for f in fill_files], ignore_index=True)
fills['trigger_wallet_lower'] = fills['trigger_wallet'].str.lower()
fills['short_wallet'] = fills['trigger_wallet_lower'].map(FULL_TO_SHORT)
fills = fills[fills['short_wallet'].isin(WALLETS.keys())].copy()
print(f"Fills for target wallets: {len(fills)}")

# Aggregate fills per (wallet, trigger_tx_hash, condition_id)
fill_agg = fills.groupby(['short_wallet', 'trigger_tx_hash', 'condition_id']).agg(
    total_fill_quantity=('fill_quantity', 'sum'),
    total_implied_pnl=('implied_pnl', 'sum'),
    fill_count=('fill_quantity', 'count'),
).reset_index()
print(f"Unique (wallet, tx_hash, condition_id) fill groups: {len(fill_agg)}")

# 3. Join
enriched['tx_hash_lower'] = enriched['tx_hash'].str.lower()
fill_agg['trigger_tx_hash_lower'] = fill_agg['trigger_tx_hash'].str.lower()

merged = enriched.merge(
    fill_agg[['short_wallet', 'trigger_tx_hash_lower', 'condition_id', 'total_fill_quantity', 'total_implied_pnl', 'fill_count']],
    left_on=['short_wallet', 'tx_hash_lower', 'condition_id'],
    right_on=['short_wallet', 'trigger_tx_hash_lower', 'condition_id'],
    how='left',
)
merged['has_fill'] = merged['total_fill_quantity'].notna()

# 4. Classify signals
# avail_copy_qty_5m_100 <= 0 (effectively 0) => truly unfillable
merged['is_unfillable'] = merged['avail_copy_qty_5m_100'] <= 1e-10
# avail > 0 but no fill => missed opportunity
merged['is_missed'] = (~merged['is_unfillable']) & (~merged['has_fill'])
# avail > 0 and has fill => filled
merged['is_filled'] = (~merged['is_unfillable']) & (merged['has_fill'])

print(f"\nSignal classification totals:")
print(f"  Truly unfillable (avail<=0): {merged['is_unfillable'].sum()}")
print(f"  Missed (avail>0, no fill):   {merged['is_missed'].sum()}")
print(f"  Filled (avail>0, has fill):   {merged['is_filled'].sum()}")
print()

# 5. Per-wallet report
print("=" * 120)
print(f"{'METRIC':<55} {'0x04b3':>12} {'0x9196':>12} {'0x6011':>12} {'0xb901':>12} {'0xa2ed':>12} {'TOTAL':>12}")
print("=" * 120)

def wallet_stat(wallet_short, metric_series):
    return metric_series[merged['short_wallet'] == wallet_short]

results = []
labels = list(WALLETS.keys())

# Total copyable_pnl
vals = [wallet_stat(w, merged['copyable_pnl']).sum() for w in labels]
print(f"{'Total copyable_pnl':<55} {vals[0]:>12.2f} {vals[1]:>12.2f} {vals[2]:>12.2f} {vals[3]:>12.2f} {vals[4]:>12.2f} {sum(vals):>12.2f}")

# Total implied_pnl (from fills)
vals = [wallet_stat(w, merged['total_implied_pnl'].fillna(0)).sum() for w in labels]
print(f"{'Total implied_pnl (backtest fills)':<55} {vals[0]:>12.2f} {vals[1]:>12.2f} {vals[2]:>12.2f} {vals[3]:>12.2f} {vals[4]:>12.2f} {sum(vals):>12.2f}")

# Gap
cp_vals = [wallet_stat(w, merged['copyable_pnl']).sum() for w in labels]
ip_vals = [wallet_stat(w, merged['total_implied_pnl'].fillna(0)).sum() for w in labels]
gap = [cp_vals[i] - ip_vals[i] for i in range(5)]
print(f"{'Gap (copyable - implied)':<55} {gap[0]:>12.2f} {gap[1]:>12.2f} {gap[2]:>12.2f} {gap[3]:>12.2f} {gap[4]:>12.2f} {sum(gap):>12.2f}")
print("-" * 120)

# Count: truly unfillable
vals = [wallet_stat(w, merged['is_unfillable']).sum() for w in labels]
print(f"{'Signals: truly unfillable (avail<=0)':<55} {vals[0]:>12d} {vals[1]:>12d} {vals[2]:>12d} {vals[3]:>12d} {vals[4]:>12d} {sum(vals):>12d}")

# Count: missed
vals = [wallet_stat(w, merged['is_missed']).sum() for w in labels]
print(f"{'Signals: missed (avail>0, no fill)':<55} {vals[0]:>12d} {vals[1]:>12d} {vals[2]:>12d} {vals[3]:>12d} {vals[4]:>12d} {sum(vals):>12d}")

# Count: filled
vals = [wallet_stat(w, merged['is_filled']).sum() for w in labels]
print(f"{'Signals: filled (avail>0, has fill)':<55} {vals[0]:>12d} {vals[1]:>12d} {vals[2]:>12d} {vals[3]:>12d} {vals[4]:>12d} {sum(vals):>12d}")

# Total signals
vals = [len(wallet_stat(w, merged)) for w in labels]
print(f"{'Total BUY signals':<55} {vals[0]:>12d} {vals[1]:>12d} {vals[2]:>12d} {vals[3]:>12d} {vals[4]:>12d} {sum(vals):>12d}")
print("-" * 120)

# copyable_pnl from each category
vals = [wallet_stat(w, merged[merged['is_unfillable']]['copyable_pnl']).sum() for w in labels]
print(f"{'copyable_pnl from truly unfillable':<55} {vals[0]:>12.2f} {vals[1]:>12.2f} {vals[2]:>12.2f} {vals[3]:>12.2f} {vals[4]:>12.2f} {sum(vals):>12.2f}")

vals = [wallet_stat(w, merged[merged['is_missed']]['copyable_pnl']).sum() for w in labels]
print(f"{'copyable_pnl from missed opportunities':<55} {vals[0]:>12.2f} {vals[1]:>12.2f} {vals[2]:>12.2f} {vals[3]:>12.2f} {vals[4]:>12.2f} {sum(vals):>12.2f}")

vals = [wallet_stat(w, merged[merged['is_filled']]['copyable_pnl']).sum() for w in labels]
print(f"{'copyable_pnl from filled signals':<55} {vals[0]:>12.2f} {vals[1]:>12.2f} {vals[2]:>12.2f} {vals[3]:>12.2f} {vals[4]:>12.2f} {sum(vals):>12.2f}")

print("=" * 120)

# Check: do the 3 categories sum to total?
print("\nVerification (categories sum to total):")
for w in labels:
    sub = merged[merged['short_wallet'] == w]
    cat_sum = (sub[sub['is_unfillable']]['copyable_pnl'].sum() +
               sub[sub['is_missed']]['copyable_pnl'].sum() +
               sub[sub['is_filled']]['copyable_pnl'].sum())
    total = sub['copyable_pnl'].sum()
    print(f"  {w}: categories={cat_sum:.2f}, total={total:.2f}, match={abs(cat_sum - total) < 0.01}")

# Also check: unfilled opportunity = unfillable + missed
print("\nUnfilled signals (unfillable + missed):")
for w in labels:
    sub = merged[merged['short_wallet'] == w]
    n_unfil = sub['is_unfillable'].sum()
    n_miss = sub['is_missed'].sum()
    pnl_unfil = sub[sub['is_unfillable']]['copyable_pnl'].sum()
    pnl_miss = sub[sub['is_missed']]['copyable_pnl'].sum()
    print(f"  {w}: {n_unfil + n_miss} signals, copyable_pnl={pnl_unfil + pnl_miss:.2f}")
