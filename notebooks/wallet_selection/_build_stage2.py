#!/usr/bin/env python3
"""Build a fresh signal-discovery notebook.

Phase A — Broad signal discovery on random trades (all wallets, no copy filter).
Phase B — Apply surviving signals to copy-wallet trades.
"""

import json
import sys
from pathlib import Path

# Inline definitions (avoids import issues with _build_notebook.py)

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source, exec_count=None):
    cell = {
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "source": source,
        "outputs": [],
    }
    return cell

# --- Constants ---
OLD_NOTEBOOK = Path(__file__).resolve().parent / "stage1_experimental.ipynb"
OUT_NOTEBOOK = Path(__file__).resolve().parent / "stage2_signal_discovery.ipynb"

with open(OLD_NOTEBOOK) as f:
    old_nb = json.load(f)

old_cells = old_nb["cells"][:23]  # First 23 cells: data loading, wallet metrics, signal quality funcs

# Ensure old code cells have execution_count for nbformat 4.5 compatibility
for c in old_cells:
    if c["cell_type"] == "code" and "execution_count" not in c:
        c["execution_count"] = None

# --- Cell definitions ---

CELL_PARAMS = '''
import time
import numpy as np
import pandas as pd

# === Discovery phase params ===
MIN_WALLET_TRADES = 100
BROAD_SAMPLE_SIZE = 200_000
SIGNAL_WINDOW_MIN = 15
VWAP_BUCKET_MIN = 5

# === Quality wallet scoring ===
QUALITY_METRICS = ['copyable_roi', 'buy_roi', 'positive_bucket_share']
QUALITY_PENALTIES = ['max_drawdown_to_pnl', 'pnl_volatility', 'top_market_pnl_pct']

# === Quality wallet threshold (pct of wallets) ===
QW_TOP_PCT = 0.20  # top 20% = quality wallets

print(f"MIN_WALLET_TRADES={MIN_WALLET_TRADES}")
print(f"BROAD_SAMPLE_SIZE={BROAD_SAMPLE_SIZE:,}")
print(f"SIGNAL_WINDOW_MIN={SIGNAL_WINDOW_MIN}")
print(f"QW_TOP_PCT={QW_TOP_PCT}")
'''

CELL_WALLET_MIN_TRADES = '''
# Define active wallet set (used by VWAP + broad sample)
wallet_min_trades = wallet_vol[wallet_vol['trade_count'] >= MIN_WALLET_TRADES]['wallet']
print(f"Wallets with {MIN_WALLET_TRADES}+ trades: {len(wallet_min_trades)}")
'''

CELL_BROAD_SAMPLE = '''
# Broad random sample of BUY trades from wallets with 100+ trades
# NOTE: df_full must already be fully annotated (window stats, QW proximity, VWAP)

buy_mask = (
    df_full['wallet'].isin(wallet_min_trades)
    & (df_full['side'] == 'BUY')
)
all_buys = df_full[buy_mask].copy()
print(f"Total BUY trades by these wallets: {len(all_buys):,}")

rng = np.random.RandomState(42)
if len(all_buys) > BROAD_SAMPLE_SIZE:
    sample_idx = rng.choice(len(all_buys), BROAD_SAMPLE_SIZE, replace=False)
    broad_sample = all_buys.iloc[sample_idx].copy()
else:
    broad_sample = all_buys.copy()
print(f"Broad sample: {len(broad_sample):,} trades")

# Split into train/val/test by time
dates = sorted(broad_sample['dt'].unique())
n = len(dates)
train_cutoff = dates[int(n * 0.4)]
val_cutoff = dates[int(n * 0.7)]
print(f"Train cutoff: {train_cutoff}, Val cutoff: {val_cutoff}")

s_train = broad_sample[broad_sample['dt'] < train_cutoff].copy()
s_val = broad_sample[
    (broad_sample['dt'] >= train_cutoff) & (broad_sample['dt'] < val_cutoff)
].copy()
s_test = broad_sample[broad_sample['dt'] >= val_cutoff].copy()
print(f"  Train: {len(s_train):,}  Val: {len(s_val):,}  Test: {len(s_test):,}")
'''

CELL_QUALITY_SCORE = '''
# Continuous wallet quality score

def compute_quality_score(wv):
    scores = pd.DataFrame(index=wv.index)
    for col in QUALITY_METRICS:
        scores[col] = wv[col].rank(pct=True)
    for col in QUALITY_PENALTIES:
        scores[f'1_{col}'] = 1.0 - wv[col].rank(pct=True)
    score_cols = list(QUALITY_METRICS) + [f'1_{c}' for c in QUALITY_PENALTIES]
    wv['quality_score'] = scores[score_cols].mean(axis=1).clip(0, 1)
    return wv

wallet_vol = compute_quality_score(wallet_vol)
quality_threshold = wallet_vol['quality_score'].quantile(1.0 - QW_TOP_PCT)

quality_wallets = set(wallet_vol.loc[
    wallet_vol['quality_score'] >= quality_threshold, 'wallet'
])
print(f"Quality wallets (top {QW_TOP_PCT:.0%}): {len(quality_wallets)}")
print(f"  Threshold quality_score: {quality_threshold:.4f}")

# Quick stats
qw_stats = wallet_vol.loc[wallet_vol['wallet'].isin(quality_wallets), 'quality_score']
print(f"  Quality score range: {qw_stats.min():.4f} - {qw_stats.max():.4f}")
print(f"  Median quality score: {qw_stats.median():.4f}")
'''

CELL_VWAP_WALLET_SETS = '''
# Wallet subsets for VWAP variants

profitable_buyers = set(wallet_vol[wallet_vol['buy_roi'] > 0.05]['wallet'])
profitable_sellers = set(wallet_vol[wallet_vol['sell_roi'] > 0.05]['wallet'])
print(f"Profitable buyers (buy_roi>0.05): {len(profitable_buyers)}")
print(f"Profitable sellers (sell_roi>0.05): {len(profitable_sellers)}")
'''

CELL_MARKET_WINDOW = '''
# Market-level window aggregations (ALL trades, both sides)
# Uses separate groupbys to avoid lambda issues with observed=True.

WINDOW = f'{SIGNAL_WINDOW_MIN}min'

df_full['dt_window'] = df_full['dt'].dt.floor(WINDOW)

# All-trade window metrics
window_all = df_full.groupby(
    ['condition_id', 'outcome', 'dt_window'], sort=False, observed=True
).agg(
    mkt_trade_count=('wallet', 'size'),
    mkt_wallet_count=('wallet', 'nunique'),
    mkt_avg_price=('price', 'mean'),
    mkt_price_std=('price', 'std'),
    mkt_price_min=('price', 'min'),
    mkt_price_max=('price', 'max'),
).reset_index()

# Side-specific volume
buy_vol = df_full[df_full['side'] == 'BUY'].groupby(
    ['condition_id', 'outcome', 'dt_window'], sort=False, observed=True
)['usdc_amount'].sum().reset_index().rename(columns={'usdc_amount': 'mkt_buy_vol'})

sell_vol = df_full[df_full['side'] == 'SELL'].groupby(
    ['condition_id', 'outcome', 'dt_window'], sort=False, observed=True
)['usdc_amount'].sum().reset_index().rename(columns={'usdc_amount': 'mkt_sell_vol'})

# Side-specific wallet counts
buy_wallets = df_full[df_full['side'] == 'BUY'].groupby(
    ['condition_id', 'outcome', 'dt_window'], sort=False, observed=True
)['wallet'].nunique().reset_index().rename(columns={'wallet': 'mkt_buy_wallet_count'})

sell_wallets = df_full[df_full['side'] == 'SELL'].groupby(
    ['condition_id', 'outcome', 'dt_window'], sort=False, observed=True
)['wallet'].nunique().reset_index().rename(columns={'wallet': 'mkt_sell_wallet_count'})

# Side-specific trade counts
buy_trades = df_full[df_full['side'] == 'BUY'].groupby(
    ['condition_id', 'outcome', 'dt_window'], sort=False, observed=True
)['wallet'].size().reset_index().rename(columns={'wallet': 'mkt_buy_trade_count'})

sell_trades = df_full[df_full['side'] == 'SELL'].groupby(
    ['condition_id', 'outcome', 'dt_window'], sort=False, observed=True
)['wallet'].size().reset_index().rename(columns={'wallet': 'mkt_sell_trade_count'})

# Merge all
window_market = window_all.merge(buy_vol, on=['condition_id', 'outcome', 'dt_window'], how='left')
window_market = window_market.merge(sell_vol, on=['condition_id', 'outcome', 'dt_window'], how='left')
window_market = window_market.merge(buy_wallets, on=['condition_id', 'outcome', 'dt_window'], how='left')
window_market = window_market.merge(sell_wallets, on=['condition_id', 'outcome', 'dt_window'], how='left')
window_market = window_market.merge(buy_trades, on=['condition_id', 'outcome', 'dt_window'], how='left')
window_market = window_market.merge(sell_trades, on=['condition_id', 'outcome', 'dt_window'], how='left')

# Fill NaN
for c in ['mkt_buy_vol', 'mkt_sell_vol', 'mkt_buy_wallet_count', 'mkt_sell_wallet_count',
          'mkt_buy_trade_count', 'mkt_sell_trade_count']:
    window_market[c] = window_market[c].fillna(0)

# Volume derivatives
window_market['mkt_net_vol'] = window_market['mkt_buy_vol'] - window_market['mkt_sell_vol']
window_market['mkt_bs_ratio'] = window_market['mkt_buy_vol'] / window_market['mkt_sell_vol'].clip(lower=1.0)
window_market['mkt_trade_imbalance'] = (
    (window_market['mkt_buy_vol'] - window_market['mkt_sell_vol'])
    / (window_market['mkt_buy_vol'] + window_market['mkt_sell_vol']).clip(lower=1.0)
)
window_market['mkt_price_range'] = 2.0 * window_market['mkt_price_std'].fillna(0)

# Wallet disagreement signals
tot_w = window_market['mkt_wallet_count'].clip(lower=1)
window_market['mkt_wallet_buy_share'] = window_market['mkt_buy_wallet_count'] / tot_w
window_market['mkt_wallet_sell_share'] = window_market['mkt_sell_wallet_count'] / tot_w

p = window_market['mkt_wallet_buy_share'].clip(0.001, 0.999)
window_market['mkt_wallet_side_entropy'] = -(
    p * np.log2(p) + (1 - p) * np.log2(1 - p)
)

# Trade intensity
window_market['mkt_wallet_trade_intensity'] = (
    window_market['mkt_trade_count'] / window_market['mkt_wallet_count'].clip(lower=1)
)

# Price CV (volatility per unit price)
window_market['mkt_price_cv'] = window_market['mkt_price_std'].fillna(0) / window_market['mkt_avg_price'].clip(lower=1e-9)

# Shift by one window to prevent look-ahead
window_market['dt_window_prev'] = window_market['dt_window'] - pd.Timedelta(minutes=SIGNAL_WINDOW_MIN)

# Pre-compute previous-window values for growth signals
# (before shift, so _prev is the true prior window for each market-outcome)
wm_sorted = window_market.sort_values(['condition_id', 'outcome', 'dt_window'])
prev_cols = ['mkt_wallet_count', 'mkt_price_std', 'mkt_avg_price',
             'mkt_trade_count', 'mkt_buy_vol', 'mkt_sell_vol']
for col in prev_cols:
    window_market[f'{col}_prev'] = wm_sorted.groupby(
        ['condition_id', 'outcome'], sort=False
    )[col].shift(1)

# Growth signals (next_window / current_window - 1)
eps = 1e-12
window_market['mkt_wallet_count_growth'] = (
    window_market['mkt_wallet_count'] / window_market['mkt_wallet_count_prev'].clip(lower=eps) - 1
)
window_market['mkt_price_volatility_ratio'] = (
    window_market['mkt_price_std'] / window_market['mkt_price_std_prev'].clip(lower=eps)
)
window_market['mkt_volume_change'] = (
    (window_market['mkt_buy_vol'] + window_market['mkt_sell_vol'])
    / (window_market['mkt_buy_vol_prev'] + window_market['mkt_sell_vol_prev']).clip(lower=eps) - 1
)
window_market['mkt_price_momentum'] = (
    window_market['mkt_avg_price'] / window_market['mkt_avg_price_prev'].clip(lower=eps) - 1
)
window_market['mkt_trade_count_growth'] = (
    window_market['mkt_trade_count'] / window_market['mkt_trade_count_prev'].clip(lower=eps) - 1
)
window_market['mkt_buy_vol_growth'] = (
    window_market['mkt_buy_vol'] / window_market['mkt_buy_vol_prev'].clip(lower=eps) - 1
)

print(f"Market window stats: {len(window_market):,} windows")
print(f"  Columns: {[c for c in window_market.columns if c.startswith('mkt_')]}")
'''

CELL_QUALITY_WINDOW = '''
# Quality-wallet window aggregations

if len(quality_wallets) == 0:
    print("No quality wallets, skipping QW signals")
    qw_window = pd.DataFrame()
else:
    qw_trades = df_full[df_full['wallet'].isin(quality_wallets)].copy()
    qw_trades['quality_score'] = qw_trades['wallet'].map(
        wallet_vol.set_index('wallet')['quality_score'].to_dict()
    )
    qw_trades['qw_weighted_vol'] = qw_trades['quality_score'] * qw_trades['usdc_amount']

    qw_window = qw_trades.groupby(
        ['condition_id', 'outcome', 'dt_window'], sort=False, observed=True
    ).agg(
        qw_score_sum=('quality_score', 'sum'),
        qw_score_mean=('quality_score', 'mean'),
        qw_trade_count=('wallet', 'size'),
        qw_wallet_count=('wallet', 'nunique'),
        qw_weighted_vol=('qw_weighted_vol', 'sum'),
        qw_raw_vol=('usdc_amount', 'sum'),
        qw_avg_price=('price', 'mean'),
    ).reset_index()

    # Top-20% quality wallets count
    top_qw_threshold = wallet_vol['quality_score'].quantile(0.80)
    top_qw_wallets = set(wallet_vol.loc[
        wallet_vol['quality_score'] >= top_qw_threshold, 'wallet'
    ])
    qw_trades_top = qw_trades[qw_trades['wallet'].isin(top_qw_wallets)]
    qw_top_count = qw_trades_top.groupby(
        ['condition_id', 'outcome', 'dt_window'], sort=False, observed=True
    )['wallet'].nunique().reset_index()
    qw_top_count = qw_top_count.rename(columns={'wallet': 'qw_top_count'})

    qw_window = qw_window.merge(qw_top_count, on=['condition_id', 'outcome', 'dt_window'], how='left')

    # Side-specific quality wallet counts
    qw_buy_wallets = qw_trades[qw_trades['side'] == 'BUY'].groupby(
        ['condition_id', 'outcome', 'dt_window'], sort=False, observed=True
    )['wallet'].nunique().reset_index().rename(columns={'wallet': 'qw_buy_wallet_count'})

    qw_sell_wallets = qw_trades[qw_trades['side'] == 'SELL'].groupby(
        ['condition_id', 'outcome', 'dt_window'], sort=False, observed=True
    )['wallet'].nunique().reset_index().rename(columns={'wallet': 'qw_sell_wallet_count'})

    qw_window = qw_window.merge(qw_buy_wallets, on=['condition_id', 'outcome', 'dt_window'], how='left')
    qw_window = qw_window.merge(qw_sell_wallets, on=['condition_id', 'outcome', 'dt_window'], how='left')
    qw_window['qw_buy_wallet_count'] = qw_window['qw_buy_wallet_count'].fillna(0)
    qw_window['qw_sell_wallet_count'] = qw_window['qw_sell_wallet_count'].fillna(0)

    # Quality wallet net direction
    tot_qw_w = qw_window['qw_wallet_count'].clip(lower=1)
    qw_window['qw_buy_share'] = qw_window['qw_buy_wallet_count'] / tot_qw_w

    # Quality wallet net wallet imbalance
    qw_window['qw_net_wallet_imbalance'] = (
        ((qw_window['qw_buy_wallet_count'] - qw_window['qw_sell_wallet_count'])
         / tot_qw_w).fillna(0)
    )

    # Shift
    qw_window['dt_window_prev'] = qw_window['dt_window'] - pd.Timedelta(minutes=SIGNAL_WINDOW_MIN)
    print(f"Quality window stats: {len(qw_window):,} windows")
'''

CELL_QW_PROXIMITY = '''
# Quality wallet proximity (merge_asof, nearest-in-past)

if len(quality_wallets) == 0:
    print("No quality wallets, skipping QW proximity")
else:
    if 'qw_wallet_v2' not in df_full.columns:
        qw_buys = df_full[
            df_full['wallet'].isin(quality_wallets) & (df_full['side'] == 'BUY')
        ].copy()
        qw_buys['wallet_qscore'] = qw_buys['wallet'].map(
            wallet_vol.set_index('wallet')['quality_score'].to_dict()
        )
        qw_buys = qw_buys.rename(columns={
            'dt': 'dt_qw', 'wallet': 'qw_wallet2', 'usdc_amount': 'qw_usdc2'
        })[['dt_qw', 'qw_wallet2', 'qw_usdc2', 'wallet_qscore', 'condition_id', 'outcome']].sort_values('dt_qw')

        df_full = pd.merge_asof(
            df_full.sort_values('dt'),
            qw_buys,
            left_on='dt', right_on='dt_qw',
            by=['condition_id', 'outcome'],
            direction='backward',
            tolerance=pd.Timedelta(minutes=SIGNAL_WINDOW_MIN),
            allow_exact_matches=False,
            suffixes=('', '_qw'),
        )
        print("  QW proximity columns added to df_full")
    else:
        print("  QW proximity already computed")
'''

CELL_VWAP_VARIANTS = '''
# Multi-source VWAP variants

def compute_bucket_vwap(trades_df, bucket_min=VWAP_BUCKET_MIN):
    df = trades_df.copy()
    df['dt_bucket'] = df['dt'].dt.floor(f'{bucket_min}min')
    df['price_vol'] = df['price'] * df['quantity']
    result = df.groupby(
        ['condition_id', 'outcome', 'dt_bucket'], sort=False, observed=True
    ).agg(
        vwap_price=('price_vol', 'sum'),
        total_qty=('quantity', 'sum'),
        vwap_vol=('usdc_amount', 'sum'),
    ).reset_index()
    result['vwap'] = result['vwap_price'] / result['total_qty'].clip(lower=1e-12)
    result['dt_bucket_prev'] = result['dt_bucket'] - pd.Timedelta(minutes=bucket_min)
    result = result.drop(columns=['vwap_price', 'total_qty'])
    return result

bucket_vwaps = {}

# 1. All-BUY VWAP
all_buys = df_full[df_full['side'] == 'BUY']
bucket_vwaps['allbuy'] = compute_bucket_vwap(all_buys)
print(f"All-buy VWAP: {len(bucket_vwaps['allbuy']):,} buckets")

# 2. All-SELL VWAP
all_sells = df_full[df_full['side'] == 'SELL']
bucket_vwaps['allsell'] = compute_bucket_vwap(all_sells)
print(f"All-sell VWAP: {len(bucket_vwaps['allsell']):,} buckets")

# 3. All-trade VWAP
all_trades = df_full
bucket_vwaps['all'] = compute_bucket_vwap(all_trades)
print(f"All-trade VWAP: {len(bucket_vwaps['all']):,} buckets")

# 4. Profitable-buyer BUY VWAP
pb_buys = df_full[df_full['wallet'].isin(profitable_buyers) & (df_full['side'] == 'BUY')]
bucket_vwaps['pbbuy'] = compute_bucket_vwap(pb_buys)
print(f"Profitable-buyer BUY VWAP: {len(bucket_vwaps['pbbuy']):,} buckets")

# 5. Profitable-seller SELL VWAP
ps_sells = df_full[df_full['wallet'].isin(profitable_sellers) & (df_full['side'] == 'SELL')]
bucket_vwaps['pssell'] = compute_bucket_vwap(ps_sells)
print(f"Profitable-seller SELL VWAP: {len(bucket_vwaps['pssell']):,} buckets")

# 6. Quality-wallet BUY VWAP
qw_buys = df_full[df_full['wallet'].isin(quality_wallets) & (df_full['side'] == 'BUY')]
bucket_vwaps['qwbuy'] = compute_bucket_vwap(qw_buys)
print(f"Quality-wallet BUY VWAP: {len(bucket_vwaps['qwbuy']):,} buckets")

# 7. Copy-wallet BUY VWAP (current approach, wallet_min_trades)
cwl_buys = df_full[df_full['wallet'].isin(wallet_min_trades) & (df_full['side'] == 'BUY')]
bucket_vwaps['copybuy'] = compute_bucket_vwap(cwl_buys)
print(f"Copy-wallet BUY VWAP: {len(bucket_vwaps['copybuy']):,} buckets")

print(f"\\nTotal VWAP variants: {len(bucket_vwaps)}")
'''

CELL_MERGE_ALL = '''
# Merge all window signals into sample splits

def apply_vwap_signal(df_c, vwap_df, name):
    """Merge VWAP variant and compute deviation + signed signal."""
    vwap_col = f'vwap_{name}'
    right = vwap_df[['condition_id', 'outcome', 'dt_bucket_prev', 'vwap']].rename(
        columns={'vwap': vwap_col, 'dt_bucket_prev': 'dt_bucket'}
    )
    df_c = df_c.merge(
        right,
        on=['condition_id', 'outcome', 'dt_bucket'],
        how='left',
    )
    dev_col = f'sig_vwap_{name}_dev'
    sig_col = f'sig_vwap_{name}_signed'
    df_c[dev_col] = np.where(
        df_c[vwap_col].notna() & (df_c[vwap_col] > 0),
        (df_c['price'] / df_c[vwap_col]) - 1.0, np.nan,
    )
    df_c[sig_col] = np.where(
        df_c[dev_col].notna(), -df_c[dev_col], np.nan,
    )
    return df_c.drop(columns=[vwap_col])

def merge_window_signals(df_c, window_market, qw_window, bucket_vwaps):
    df_c = df_c.copy()
    df_c['dt_window'] = df_c['dt'].dt.floor(f'{SIGNAL_WINDOW_MIN}min')
    df_c['dt_window_prev'] = df_c['dt_window'] - pd.Timedelta(minutes=SIGNAL_WINDOW_MIN)
    df_c['dt_bucket'] = df_c['dt'].dt.floor(f'{VWAP_BUCKET_MIN}min')

    # Market window signals (previous window)
    win_cols = ['condition_id', 'outcome', 'dt_window_prev']
    mkt_cols = [
        'mkt_trade_count', 'mkt_wallet_count', 'mkt_buy_vol', 'mkt_sell_vol',
        'mkt_net_vol', 'mkt_bs_ratio', 'mkt_trade_imbalance',
        'mkt_avg_price', 'mkt_price_std', 'mkt_price_range',
        'mkt_buy_wallet_count', 'mkt_sell_wallet_count',
        'mkt_buy_trade_count', 'mkt_sell_trade_count',
        'mkt_wallet_buy_share', 'mkt_wallet_sell_share',
        'mkt_wallet_side_entropy', 'mkt_wallet_trade_intensity',
        'mkt_price_cv',
        'mkt_wallet_count_growth', 'mkt_price_volatility_ratio',
        'mkt_volume_change', 'mkt_price_momentum',
        'mkt_trade_count_growth', 'mkt_buy_vol_growth',
    ]
    df_c = df_c.merge(
        window_market[win_cols + mkt_cols],
        left_on=['condition_id', 'outcome', 'dt_window'],
        right_on=win_cols,
        how='left', suffixes=('', '_mkt')
    )

    # Quality window signals
    if len(qw_window) > 0:
        qw_cols = [
            'qw_score_sum', 'qw_score_mean', 'qw_trade_count',
            'qw_wallet_count', 'qw_weighted_vol', 'qw_raw_vol', 'qw_top_count',
            'qw_buy_wallet_count', 'qw_sell_wallet_count',
            'qw_buy_share', 'qw_net_wallet_imbalance', 'qw_avg_price',
        ]
        df_c = df_c.merge(
            qw_window[win_cols + qw_cols],
            left_on=['condition_id', 'outcome', 'dt_window'],
            right_on=win_cols,
            how='left', suffixes=('', '_qw')
        )

    # QW price premium: how quality wallet avg price differs from market avg
    if len(qw_window) > 0 and 'qw_avg_price' in df_c.columns:
        df_c['qw_price_premium'] = np.where(
            df_c['qw_avg_price'].notna() & (df_c['mkt_avg_price'] > 0),
            (df_c['qw_avg_price'] / df_c['mkt_avg_price']) - 1.0, np.nan,
        )

    # QW proximity
    df_c['sig_qw_proximity'] = df_c['qw_wallet2'].notna().astype(float)
    df_c['sig_qw_prox_volume'] = df_c['qw_usdc2'].fillna(0.0)
    df_c['sig_qw_prox_qscore'] = df_c['wallet_qscore'].fillna(0.0)

    # VWAP variants
    for name, vwap_df in bucket_vwaps.items():
        df_c = apply_vwap_signal(df_c, vwap_df, name)

    # CS-rank on copybuy VWAP signal (our primary variant)
    df_c['sig_vwap_csrank'] = cs_rank(df_c['sig_vwap_copybuy_signed'].fillna(0.0), df_c['dt'].dt.date)

    return df_c

for name, df_c in [('Train', s_train), ('Val', s_val), ('Test', s_test)]:
    updated = merge_window_signals(df_c, window_market, qw_window, bucket_vwaps)
    if name == 'Train': s_train = updated
    elif name == 'Val': s_val = updated
    elif name == 'Test': s_test = updated
    print(f"{name}: {len(df_c)} -> {len(updated)} rows")

# Define signal columns
raw_signal_cols = [
    'sig_qw_proximity', 'sig_qw_prox_volume', 'sig_qw_prox_qscore',
    'sig_vwap_csrank',
]
vwap_variant_names = ['allbuy', 'allsell', 'all', 'pbbuy', 'pssell', 'qwbuy', 'copybuy']
vwap_variant_cols = []
for vn in vwap_variant_names:
    vwap_variant_cols += [f'sig_vwap_{vn}_dev', f'sig_vwap_{vn}_signed']

market_signal_cols = [
    'mkt_trade_count', 'mkt_wallet_count', 'mkt_net_vol', 'mkt_bs_ratio',
    'mkt_trade_imbalance', 'mkt_avg_price', 'mkt_price_range',
    # New market signals
    'mkt_wallet_buy_share', 'mkt_wallet_side_entropy',
    'mkt_wallet_trade_intensity', 'mkt_price_cv',
]

qw_window_cols = [
    'qw_score_sum', 'qw_score_mean', 'qw_trade_count', 'qw_wallet_count',
    'qw_weighted_vol', 'qw_raw_vol', 'qw_top_count',
    # New QW signals
    'qw_buy_share', 'qw_net_wallet_imbalance', 'qw_avg_price', 'qw_price_premium',
]

growth_signal_cols = [
    'mkt_wallet_count_growth', 'mkt_price_volatility_ratio',
    'mkt_volume_change', 'mkt_price_momentum',
    'mkt_trade_count_growth', 'mkt_buy_vol_growth',
]

signal_cols = raw_signal_cols + vwap_variant_cols + market_signal_cols + qw_window_cols + growth_signal_cols
signal_cols = [c for c in signal_cols if c in s_train.columns]

# Cross-sectional rank each market + qw + growth signal per day for comparability
for col in market_signal_cols + qw_window_cols + growth_signal_cols:
    if col in s_train.columns:
        s_train[f'cs_{col}'] = cs_rank(s_train[col].fillna(0.0), s_train['dt'].dt.date)
        s_val[f'cs_{col}'] = cs_rank(s_val[col].fillna(0.0), s_val['dt'].dt.date)
        s_test[f'cs_{col}'] = cs_rank(s_test[col].fillna(0.0), s_test['dt'].dt.date)

# Also CS-rank each VWAP variant
for vn in vwap_variant_names:
    col = f'sig_vwap_{vn}_signed'
    if col in s_train.columns:
        cs_col = f'cs_vwap_{vn}'
        s_train[cs_col] = cs_rank(s_train[col].fillna(0.0), s_train['dt'].dt.date)
        s_val[cs_col] = cs_rank(s_val[col].fillna(0.0), s_val['dt'].dt.date)
        s_test[cs_col] = cs_rank(s_test[col].fillna(0.0), s_test['dt'].dt.date)

cs_signal_cols = (
    [f'cs_{c}' for c in market_signal_cols + qw_window_cols + growth_signal_cols if f'cs_{c}' in s_train.columns]
    + [f'cs_vwap_{vn}' for vn in vwap_variant_names if f'cs_vwap_{vn}' in s_train.columns]
)
all_signal_cols = signal_cols + cs_signal_cols
all_signal_cols = [c for c in all_signal_cols if c in s_train.columns]

print(f"\\nTotal signals: {len(all_signal_cols)}")
for c in all_signal_cols:
    nnan = s_val[c].notna().sum()
    print(f"  {c:35s}: non-null={nnan:>8,}  mean={s_val[c].mean():.4f}")
'''

CELL_EVAL_SIGNALS = '''
# IC evaluation for each signal with bootstrapped CIs

print("=" * 90)
print(f"{'Signal':35s} {'Val_IC':>9s} {'Val_pIR':>9s} {'Val_hit':>9s} {'Train_IC':>9s} {'Test_IC':>9s}")
print("=" * 90)

signal_results = {}
for sig in all_signal_cols:
    n_train = s_train[sig].notna().sum()
    n_val = s_val[sig].notna().sum()
    n_test = s_test[sig].notna().sum()
    if min(n_train, n_val, n_test) < 100:
        continue
    ic_val = compute_event_ic(s_val[sig], s_val['copyable_roi']) or 0.0
    ic_train = compute_event_ic(s_train[sig], s_train['copyable_roi']) or 0.0
    ic_test = compute_event_ic(s_test[sig], s_test['copyable_roi']) or 0.0
    hit_val = (s_val[sig].notna() & (s_val['copyable_roi'] > 0)).mean() if s_val[sig].notna().any() else 0.0

    signal_results[sig] = {
        'val_ic': ic_val, 'train_ic': ic_train, 'test_ic': ic_test,
        'val_n': n_val, 'hit_val': hit_val,
    }
    print(f"{sig:35s} {ic_val:9.4f} {ic_val * np.sqrt(n_val):9.4f} {hit_val:9.4f} {ic_train:9.4f} {ic_test:9.4f}")

# Sort by |val_ic|
ranked = sorted(signal_results.items(), key=lambda x: abs(x[1]['val_ic']), reverse=True)
print(f"\\n--- Top signals by |Val IC| ---")
for sig, r in ranked[:10]:
    print(f"  {sig:35s}  Val IC={r['val_ic']:.4f}  Train IC={r['train_ic']:.4f}  Test IC={r['test_ic']:.4f}  (n={r['val_n']:,})")

# Signals passing threshold
PASS_IC = 0.010
passing = [sig for sig, r in ranked if abs(r['val_ic']) >= PASS_IC and r['train_ic'] * r['val_ic'] > 0]
print(f"\\nSignals with |IC| >= {PASS_IC} AND consistent sign: {len(passing)}")
for sig in passing:
    r = signal_results[sig]
    print(f"  {sig:35s}  Val IC={r['val_ic']:.4f}  Train IC={r['train_ic']:.4f}  Test IC={r['test_ic']:.4f}")
'''

CELL_FORWARD_SELECTION = '''
# Forward selection of signals

def conditional_ic(target_col, candidate, baseline_cols, df):
    """IC of candidate after orthogonalizing wrt baseline_cols."""
    if not baseline_cols:
        return compute_event_ic(df[candidate], df[target_col]) or 0.0
    # Orthogonalize candidate against baseline using cross-sectional ranks
    # Simple approach: residual of linear regression
    X = df[baseline_cols].fillna(0).values
    y = df[candidate].fillna(0).values
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residual = y - X @ beta
    except np.linalg.LinAlgError:
        return 0.0
    return compute_event_ic(pd.Series(residual, index=df.index), df[target_col]) or 0.0

active_signals = list(passing)  # signals that passed initial screen
if len(active_signals) < 2:
    print(f"Too few signals ({len(active_signals)}) for forward selection")
    selected = active_signals
else:
    # Sort by |val_ic|, pick best first
    selected = [active_signals[0]]
    remaining = list(active_signals[1:])

    print(f"Forward selection starting with: {selected[0]}")
    print(f"{'Step':>5s} {'Selected':30s} {'Cond_IC':>9s} {'Running Count':>13s}")
    print("-" * 60)

    for step in range(min(5, len(remaining))):
        best_cond = -float('inf')
        best_sig = None
        for sig in remaining:
            cond = conditional_ic('copyable_roi', sig, selected, s_val)
            if cond > best_cond:
                best_cond = cond
                best_sig = sig
        if best_sig is None or best_cond <= 0.001:
            break
        selected.append(best_sig)
        remaining.remove(best_sig)
        print(f"{step + 1:5d} {best_sig:30s} {best_cond:9.4f} ({len(selected):3d} total)")

    print(f"\\nSelected {len(selected)} signals: {selected}")

# If no signals pass, use top 3 by val_ic (regardless of sign consistency)
if not selected and ranked:
    selected = [sig for sig, r in ranked[:3]]
    print(f"No signals pass threshold, using top 3: {selected}")
'''

CELL_TUNE_VWAP = '''
# === Tune VWAP bucket size ===

def tune_vwap_bucket(bucket_min, df_sample, target_col='copyable_roi'):
    """Compute VWAP signal at given bucket size and return IC."""
    copy_buys = df_full[
        df_full['wallet'].isin(wallet_min_trades) & (df_full['side'] == 'BUY')
    ].copy()
    copy_buys['price_vol'] = copy_buys['price'] * copy_buys['quantity']
    copy_buys['dt_bucket'] = copy_buys['dt'].dt.floor(f'{bucket_min}min')
    
    bv = copy_buys.groupby(
        ['condition_id', 'outcome', 'dt_bucket'], sort=False, observed=True
    ).agg(
        vwap_price=('price_vol', 'sum'),
        total_qty=('quantity', 'sum'),
    ).reset_index()
    bv['vwap'] = bv['vwap_price'] / bv['total_qty'].clip(lower=1e-12)
    bv['dt_bucket_prev'] = bv['dt_bucket'] - pd.Timedelta(minutes=bucket_min)
    
    result = df_sample.copy()
    result['dt_bucket'] = result['dt'].dt.floor(f'{bucket_min}min')
    result = result.merge(
        bv[['condition_id', 'outcome', 'dt_bucket_prev', 'vwap']].rename(
            columns={'dt_bucket_prev': 'dt_bucket'}),
        on=['condition_id', 'outcome', 'dt_bucket'],
        how='left',
    )
    dev = np.where(result['vwap'].notna() & (result['vwap'] > 0),
                   (result['price'] / result['vwap']) - 1.0, np.nan)
    sig = pd.Series(np.where(np.isfinite(dev), -dev, np.nan), index=result.index)
    return compute_event_ic(sig, result[target_col]) or 0.0

bucket_sizes = [1, 2, 3, 5, 7, 10, 15, 20, 30]
print(f"\\nTuning VWAP bucket size (copybuy VWAP):")
print(f"{'bucket_min':>12s} {'Val_IC':>9s} {'Train_IC':>9s} {'Test_IC':>9s} {'n_val':>8s}")
print("-" * 50)

vwap_tune_results = []
for bm in bucket_sizes:
    ic_val = tune_vwap_bucket(bm, s_val)
    ic_train = tune_vwap_bucket(bm, s_train)
    ic_test = tune_vwap_bucket(bm, s_test)
    n_val = int(s_val['dt'].dt.floor(f'{bm}min').notna().sum())
    vwap_tune_results.append({'bucket_min': bm, 'val_ic': ic_val,
                              'train_ic': ic_train, 'test_ic': ic_test, 'n_val': n_val})
    print(f"{bm:12d} {ic_val:9.4f} {ic_train:9.4f} {ic_test:9.4f} {n_val:8d}")

best_vwap = max(vwap_tune_results, key=lambda r: abs(r['val_ic']))
print(f"\\nBest VWAP bucket: {best_vwap['bucket_min']}min (Val IC={best_vwap['val_ic']:.4f})")
# Same for train consistency
train_consistent = [r for r in vwap_tune_results if r['train_ic'] * r['val_ic'] > 0]
if train_consistent:
    best_consistent = max(train_consistent, key=lambda r: abs(r['val_ic']))
    print(f"Best (consistent sign): {best_consistent['bucket_min']}min (Val IC={best_consistent['val_ic']:.4f})")
'''

CELL_TUNE_WINDOW = '''
# === Tune signal window size ===

def eval_window(wm, df_samples):
    """Evaluate IC at given window size. Returns dict of (sig -> ic_val/train/test)."""
    W = f'{wm}min'
    tw = pd.Timedelta(minutes=wm)
    
    mkt_cols = ['mkt_trade_count', 'mkt_wallet_count', 'mkt_net_vol', 'mkt_bs_ratio',
                'mkt_trade_imbalance', 'mkt_avg_price', 'mkt_wallet_buy_share',
                'mkt_wallet_side_entropy', 'mkt_wallet_trade_intensity', 'mkt_price_cv']
    
    # Market window at this size (no copy — use Series as groupby key)
    dt_w_s = df_full['dt'].dt.floor(W).rename('dt_w')
    core_cols = ['condition_id', 'outcome']
    
    wa = df_full.groupby([*core_cols, dt_w_s], sort=False, observed=True).agg(
        mkt_trade_count=('wallet', 'size'),
        mkt_wallet_count=('wallet', 'nunique'),
        mkt_avg_price=('price', 'mean'),
        mkt_price_std=('price', 'std'),
    ).reset_index()
    
    buy_mask = df_full['side'] == 'BUY'
    sell_mask = df_full['side'] == 'SELL'
    
    bv = df_full.loc[buy_mask].groupby(
        ['condition_id', 'outcome', dt_w_s.loc[buy_mask]], sort=False, observed=True
    )['usdc_amount'].sum().reset_index().rename(columns={'usdc_amount': 'mkt_buy_vol'})
    
    sv = df_full.loc[sell_mask].groupby(
        ['condition_id', 'outcome', dt_w_s.loc[sell_mask]], sort=False, observed=True
    )['usdc_amount'].sum().reset_index().rename(columns={'usdc_amount': 'mkt_sell_vol'})
    
    bw = df_full.loc[buy_mask].groupby(
        ['condition_id', 'outcome', dt_w_s.loc[buy_mask]], sort=False, observed=True
    )['wallet'].nunique().reset_index().rename(columns={'wallet': 'mkt_buy_wallet_count'})
    
    sw = df_full.loc[sell_mask].groupby(
        ['condition_id', 'outcome', dt_w_s.loc[sell_mask]], sort=False, observed=True
    )['wallet'].nunique().reset_index().rename(columns={'wallet': 'mkt_sell_wallet_count'})
    
    wm_market = wa.merge(bv, on=['condition_id', 'outcome', 'dt_w'], how='left')
    wm_market = wm_market.merge(sv, on=['condition_id', 'outcome', 'dt_w'], how='left')
    wm_market = wm_market.merge(bw, on=['condition_id', 'outcome', 'dt_w'], how='left')
    wm_market = wm_market.merge(sw, on=['condition_id', 'outcome', 'dt_w'], how='left')
    
    for c in ['mkt_buy_vol', 'mkt_sell_vol', 'mkt_buy_wallet_count', 'mkt_sell_wallet_count']:
        wm_market[c] = wm_market[c].fillna(0)
    
    wm_market['mkt_net_vol'] = wm_market['mkt_buy_vol'] - wm_market['mkt_sell_vol']
    wm_market['mkt_bs_ratio'] = wm_market['mkt_buy_vol'] / wm_market['mkt_sell_vol'].clip(lower=1.0)
    wm_market['mkt_trade_imbalance'] = (
        (wm_market['mkt_buy_vol'] - wm_market['mkt_sell_vol'])
        / (wm_market['mkt_buy_vol'] + wm_market['mkt_sell_vol']).clip(lower=1.0)
    )
    wm_market['mkt_price_range'] = 2.0 * wm_market['mkt_price_std'].fillna(0)
    tot_w = wm_market['mkt_wallet_count'].clip(lower=1)
    wm_market['mkt_wallet_buy_share'] = wm_market['mkt_buy_wallet_count'] / tot_w
    wm_market['mkt_wallet_trade_intensity'] = wm_market['mkt_trade_count'] / tot_w
    wm_market['mkt_price_cv'] = wm_market['mkt_price_std'].fillna(0) / wm_market['mkt_avg_price'].clip(lower=1e-9)
    p = wm_market['mkt_wallet_buy_share'].clip(0.001, 0.999)
    wm_market['mkt_wallet_side_entropy'] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    wm_market['dt_w_prev'] = wm_market['dt_w'] - tw
    
    # Merge and evaluate VWAP + market signals for each split
    sigs_to_test = ['sig_vwap_copybuy_signed', 'sig_vwap_csrank',
                    'mkt_wallet_buy_share', 'mkt_wallet_side_entropy',
                    'mkt_wallet_trade_intensity', 'mkt_price_cv',
                    'mkt_net_vol', 'mkt_bs_ratio', 'mkt_trade_imbalance',
                    'cs_mkt_net_vol', 'cs_mkt_wallet_buy_share']
    
    results = {}
    for label, df_c in df_samples:
        dx = df_c.copy()
        # Drop columns from main pipeline that will be re-merged at this window size
        drop_cols = [c for c in ['dt_w', 'dt_w_prev', 'dt_bucket'] + mkt_cols if c in dx.columns]
        dx = dx.drop(columns=drop_cols)
        dx['dt_w'] = dx['dt'].dt.floor(W)
        dx['dt_w_prev'] = dx['dt_w'] - tw
        dx['dt_bucket'] = dx['dt'].dt.floor(f'{VWAP_BUCKET_MIN}min')
        
        # Market signals
        win_cols = ['condition_id', 'outcome', 'dt_w_prev']
        dx = dx.merge(
            wm_market[win_cols + mkt_cols],
            on=['condition_id', 'outcome', 'dt_w_prev'], how='left',
        )
        
        # VWAP (reuse bucket_vwaps at default VWAP_BUCKET_MIN)
        dx = apply_vwap_signal(dx, bucket_vwaps['copybuy'], 'copybuy')
        
        # CS-rank
        for col in mkt_cols:
            if col in dx.columns:
                dx[f'cs_{col}'] = cs_rank(dx[col].fillna(0.0), dx['dt'].dt.date)
        
        for sig in sigs_to_test:
            if sig not in dx.columns:
                continue
            ic = compute_event_ic(dx[sig], dx['copyable_roi'])
            if ic is not None and not np.isnan(ic):
                results.setdefault(sig, {})[label] = ic
    return results

window_sizes = [10, 15, 30]  # narrowed from [5,10,15,30,60] for speed
samples = [('Train', s_train), ('Val', s_val), ('Test', s_test)]

print(f"\\nTuning signal window size (VWAP bucket={VWAP_BUCKET_MIN}min):")
print(f"{'window':>8s} {'time':>7s} ", end="")
for sig in ['sig_vwap_copybuy_signed', 'mkt_wallet_side_entropy', 'mkt_wallet_buy_share',
            'mkt_net_vol', 'mkt_bs_ratio', 'mkt_trade_imbalance']:
    print(f"{sig:>28s}", end="")
print()
print("-" * 190)

all_window_results = []
for wm in window_sizes:
    t0 = time.time()
    print(f"{wm:>5d}min ", end="")
    res = eval_window(wm, samples)
    elapsed = time.time() - t0
    all_window_results.append({'window_min': wm, 'results': res})
    print(f"{elapsed:>5.0f}s ", end="")
    for sig in ['sig_vwap_copybuy_signed', 'mkt_wallet_side_entropy', 'mkt_wallet_buy_share',
                'mkt_net_vol', 'mkt_bs_ratio', 'mkt_trade_imbalance']:
        if sig in res and 'Val' in res[sig]:
            print(f"{res[sig]['Val']:>13.4f} (T{res[sig].get('Train', 0):.3f}/Tst{res[sig].get('Test', 0):.3f})", end="")
        else:
            print(f"{'':>28s}", end="")
    print()

# Find best window for VWAP signal
print(f"\\nBest window for sig_vwap_copybuy_signed:")
for r in all_window_results:
    res = r['results'].get('sig_vwap_copybuy_signed', {})
    if res:
        val, train, test = res.get('Val', 0), res.get('Train', 0), res.get('Test', 0)
        consistent = '✓' if val * train > 0 else '✗'
        print(f"  {r['window_min']:>5d}min: Val={val:.4f} Train={train:.4f} Test={test:.4f} {consistent}")
'''

CELL_TUNE_QW = '''
# === Tune quality wallet threshold ===

def eval_qw_top(pct, df_samples):
    """Evaluate IC at given QW_TOP_PCT. Returns dict of (sig -> ic)."""
    thresh = wallet_vol['quality_score'].quantile(1.0 - pct)
    qw = set(wallet_vol.loc[wallet_vol['quality_score'] >= thresh, 'wallet'])
    
    # Quality window
    qw_trades = df_full[df_full['wallet'].isin(qw)].copy()
    qw_trades['quality_score'] = qw_trades['wallet'].map(
        wallet_vol.set_index('wallet')['quality_score'].to_dict()
    )
    
    W = f'{SIGNAL_WINDOW_MIN}min'
    qw_win = qw_trades.groupby(
        ['condition_id', 'outcome', 'dt_window'], sort=False, observed=True
    ).agg(
        qw_trade_count=('wallet', 'size'),
        qw_wallet_count=('wallet', 'nunique'),
        qw_raw_vol=('usdc_amount', 'sum'),
    ).reset_index()
    
    qw_bw = qw_trades[qw_trades['side'] == 'BUY'].groupby(
        ['condition_id', 'outcome', 'dt_window'], sort=False, observed=True
    )['wallet'].nunique().reset_index().rename(columns={'wallet': 'qw_buy_wallet_count'})
    
    qw_sw = qw_trades[qw_trades['side'] == 'SELL'].groupby(
        ['condition_id', 'outcome', 'dt_window'], sort=False, observed=True
    )['wallet'].nunique().reset_index().rename(columns={'wallet': 'qw_sell_wallet_count'})
    
    qw_win = qw_win.merge(qw_bw, on=['condition_id', 'outcome', 'dt_window'], how='left')
    qw_win = qw_win.merge(qw_sw, on=['condition_id', 'outcome', 'dt_window'], how='left')
    qw_win['qw_buy_wallet_count'] = qw_win['qw_buy_wallet_count'].fillna(0)
    qw_win['qw_sell_wallet_count'] = qw_win['qw_sell_wallet_count'].fillna(0)
    qw_win['dt_window_prev'] = qw_win['dt_window'] - pd.Timedelta(minutes=SIGNAL_WINDOW_MIN)
    
    # QW proximity (merge_asof)
    qw_buys = qw_trades[qw_trades['side'] == 'BUY'][
        ['dt', 'wallet', 'usdc_amount', 'condition_id', 'outcome']
    ].copy()
    qw_buys = qw_buys.rename(columns={
        'dt': 'dt_qw', 'wallet': 'qw_wallet2', 'usdc_amount': 'qw_usdc2'
    }).sort_values('dt_qw')
    
    sigs_to_test = ['qw_trade_count', 'qw_wallet_count', 'qw_buy_wallet_count',
                    'sig_qw_proximity', 'sig_qw_prox_volume',
                    'cs_qw_trade_count', 'cs_qw_wallet_count']
    
    results = {}
    for label, df_c in df_samples:
        dx = df_c.copy()
        # Drop columns from main pipeline that will be recomputed
        qw_drop = [c for c in ['dt_window', 'dt_window_prev', 'dt_qw', 'sig_qw_proximity',
                               'sig_qw_prox_volume', 'qw_wallet2', 'qw_usdc2', 'wallet_qscore',
                               'qw_trade_count', 'qw_wallet_count', 'qw_buy_wallet_count',
                               'cs_qw_trade_count', 'cs_qw_wallet_count']
                   if c in dx.columns]
        dx = dx.drop(columns=qw_drop)
        
        # Proximity
        dx = pd.merge_asof(
            dx.sort_values('dt'),
            qw_buys,
            left_on='dt', right_on='dt_qw',
            by=['condition_id', 'outcome'],
            direction='backward',
            tolerance=pd.Timedelta(minutes=SIGNAL_WINDOW_MIN),
            allow_exact_matches=False,
            suffixes=('', '_qw_merged'),
        )
        dx['sig_qw_proximity'] = dx['qw_wallet2'].notna().astype(float)
        dx['sig_qw_prox_volume'] = dx['qw_usdc2'].fillna(0.0)
        
        # Window signals
        dx['dt_window'] = dx['dt'].dt.floor(W)
        dx['dt_window_prev'] = dx['dt_window'] - pd.Timedelta(minutes=SIGNAL_WINDOW_MIN)
        win_cols = ['condition_id', 'outcome', 'dt_window_prev']
        dx = dx.merge(
            qw_win[win_cols + ['qw_trade_count', 'qw_wallet_count', 'qw_buy_wallet_count']],
            on=['condition_id', 'outcome', 'dt_window_prev'], how='left',
        )
        
        for col in ['qw_trade_count', 'qw_wallet_count', 'qw_buy_wallet_count']:
            if col in dx.columns:
                dx[f'cs_{col}'] = cs_rank(dx[col].fillna(0.0), dx['dt'].dt.date)
        
        for sig in sigs_to_test:
            if sig not in dx.columns:
                continue
            ic = compute_event_ic(dx[sig], dx['copyable_roi'])
            if ic is not None and not np.isnan(ic):
                results.setdefault(sig, {})[label] = ic
    return results

top_pcts = [0.10, 0.15, 0.20]  # narrowed from [0.05,0.10,0.15,0.20,0.30,0.50] for speed
samples = [('Train', s_train), ('Val', s_val), ('Test', s_test)]

print(f"\\nTuning QW_TOP_PCT:")
print(f"{'top_pct':>10s} {'time':>7s} {'qw_count':>8s} ", end="")
for sig in ['qw_wallet_count', 'qw_trade_count', 'sig_qw_proximity', 'cs_qw_trade_count']:
    print(f"{sig:>30s}", end="")
print()
print("-" * 170)

all_qw_results = []
for pct in top_pcts:
    t0 = time.time()
    n_qw = int(len(wallet_vol) * pct)
    print(f"{pct:>8.0%} ", end="")
    res = eval_qw_top(pct, samples)
    elapsed = time.time() - t0
    print(f"{elapsed:>5.0f}s {n_qw:>8d} ", end="")
    all_qw_results.append({'top_pct': pct, 'results': res})
    for sig in ['qw_wallet_count', 'qw_trade_count', 'sig_qw_proximity', 'cs_qw_trade_count']:
        if sig in res and 'Val' in res[sig]:
            print(f"{res[sig]['Val']:>13.4f} (T{res[sig].get('Train', 0):.3f}/Tst{res[sig].get('Test', 0):.3f})", end="")
        else:
            print(f"{'':>30s}", end="")
    print()

# Best for cs_qw_trade_count
print(f"\\nBest for cs_qw_trade_count:")
for r in all_qw_results:
    res = r['results'].get('cs_qw_trade_count', {})
    if res:
        val, train, test = res.get('Val', 0), res.get('Train', 0), res.get('Test', 0)
        consistent = '✓' if val * train > 0 else '✗'
        print(f"  {r['top_pct']:>6.0%}: Val={val:.4f} Train={train:.4f} Test={test:.4f} {consistent}")
'''

CELL_TUNE_TRADES = '''
# === Tune MIN_WALLET_TRADES (parallel) ===

import concurrent.futures, time

thresholds = [50, 100, 200, 500]

def tune_one(mt):
    active = set(wallet_vol[wallet_vol['trade_count'] >= mt]['wallet'])
    buy_mask = df_full['wallet'].isin(active) & (df_full['side'] == 'BUY')
    
    copy_buys = df_full[buy_mask].copy()
    copy_buys['price_vol'] = copy_buys['price'] * copy_buys['quantity']
    copy_buys['dt_bucket'] = copy_buys['dt'].dt.floor(f'{VWAP_BUCKET_MIN}min')
    bv = copy_buys.groupby(
        ['condition_id', 'outcome', 'dt_bucket'], sort=False, observed=True
    ).agg(vwap_price=('price_vol', 'sum'), total_qty=('quantity', 'sum')).reset_index()
    bv['vwap'] = bv['vwap_price'] / bv['total_qty'].clip(lower=1e-12)
    bv['dt_bucket_prev'] = bv['dt_bucket'] - pd.Timedelta(minutes=VWAP_BUCKET_MIN)
    
    all_buys = df_full[buy_mask]
    n_total = len(all_buys)
    rng = np.random.RandomState(42)
    if n_total > BROAD_SAMPLE_SIZE:
        idx = rng.choice(n_total, BROAD_SAMPLE_SIZE, replace=False)
        sample = all_buys.iloc[idx].copy()
    else:
        sample = all_buys.copy()
    
    dates = sorted(sample['dt'].unique())
    tc = dates[int(len(dates) * 0.4)]
    vc = dates[int(len(dates) * 0.7)]
    
    res = {'min_trades': mt, 'n_active': len(active), 'n_total': n_total}
    for name, mask in [('Train', sample['dt'] < tc),
                        ('Val', (sample['dt'] >= tc) & (sample['dt'] < vc)),
                        ('Test', sample['dt'] >= vc)]:
        df_c = sample[mask].copy()
        df_c['dt_bucket'] = df_c['dt'].dt.floor(f'{VWAP_BUCKET_MIN}min')
        df_c = df_c.merge(
            bv[['condition_id', 'outcome', 'dt_bucket_prev', 'vwap']].rename(
                columns={'dt_bucket_prev': 'dt_bucket'}),
            on=['condition_id', 'outcome', 'dt_bucket'], how='left',
        )
        dev = np.where(df_c['vwap'].notna() & (df_c['vwap'] > 0),
                       (df_c['price'] / df_c['vwap']) - 1.0, np.nan)
        sig = pd.Series(np.where(np.isfinite(dev), -dev, np.nan), index=df_c.index)
        ic = compute_event_ic(sig, df_c['copyable_roi']) or 0.0
        res[name] = ic
    return res

print(f"\\nTuning MIN_WALLET_TRADES (parallel, ThreadPoolExecutor):")
print(f"{'min_trades':>12s} {'active':>8s} {'total_trades':>14s} {'time':>7s} {'Val_IC':>9s} {'Train_IC':>9s} {'Test_IC':>9s}")
print("-" * 65)

t0_all = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    fut_map = {ex.submit(tune_one, mt): mt for mt in thresholds}
    tune_results = []
    for fut in concurrent.futures.as_completed(fut_map):
        r = fut.result()
        tune_results.append(r)
        elapsed = time.time() - t0_all
        print(f"{r['min_trades']:>12d} {r['n_active']:>8d} {r['n_total']:>14,} {elapsed:>6.0f}s {r.get('Val', 0):>9.4f} {r.get('Train', 0):>9.4f} {r.get('Test', 0):>9.4f}")

tune_results.sort(key=lambda r: r['min_trades'])
print(f"\\n--- Summary ---")
print(f"{'min_trades':>12s} {'active':>8s} {'total_trades':>14s} {'Val_IC':>9s} {'Train_IC':>9s} {'Test_IC':>9s}")
print("-" * 65)
for r in tune_results:
    val, train, test = r.get('Val', 0), r.get('Train', 0), r.get('Test', 0)
    check = '✓' if val * train > 0 else '✗'
    print(f"{r['min_trades']:>12d} {r['n_active']:>8d} {r['n_total']:>14,} {val:>9.4f} {train:>9.4f} {test:>9.4f}  {check}")

best = max(tune_results, key=lambda r: abs(r.get('Val', 0)))
consistent = [r for r in tune_results if r.get('Train', 0) * r.get('Val', 0) > 0]
print(f"\\nBest |Val IC|: min_trades={best['min_trades']} (Val IC={best.get('Val', 0):.4f})")
if consistent:
    best_c = max(consistent, key=lambda r: abs(r.get('Val', 0)))
    print(f"Best consistent: min_trades={best_c['min_trades']} (Val IC={best_c.get('Val', 0):.4f})")
'''

CELL_COPY_APPLICATION = '''
# Apply winning signals to copy-universe trades

# Define copy universe (less restrictive than before)
copy_mask = (
    (wallet_vol['copyable_pnl'] > 0)
    & (wallet_vol['copyable_roi'].fillna(0) >= 0.02)
    & (wallet_vol['num_buckets'] >= 20)
    & (wallet_vol['trade_count'] >= MIN_WALLET_TRADES)
    & (wallet_vol['max_drawdown_to_pnl'].fillna(1.0) <= 0.5)
)
copy_wallets = set(wallet_vol.loc[copy_mask, 'wallet'])
print(f"Copy wallets: {len(copy_wallets)}")

candidate_mask = df_full['wallet'].isin(copy_wallets) & (df_full['side'] == 'BUY')
candidate_trades = df_full[candidate_mask].copy()
print(f"Candidate BUY trades: {len(candidate_trades):,}")

# Split candidate trades into train/val/test
c_dates = sorted(candidate_trades['dt'].unique())
n = len(c_dates)
cc_train = c_dates[int(n * 0.4)]
cc_val = c_dates[int(n * 0.7)]
ct_train = candidate_trades[candidate_trades['dt'] < cc_train].copy()
ct_val = candidate_trades[
    (candidate_trades['dt'] >= cc_train) & (candidate_trades['dt'] < cc_val)
].copy()
ct_test = candidate_trades[candidate_trades['dt'] >= cc_val].copy()
print(f"  C_Train: {len(ct_train):,}  C_Val: {len(ct_val):,}  C_Test: {len(ct_test):,}")

# Merge signals into copy trades using the same function
for name, df_c in [('C_Train', ct_train), ('C_Val', ct_val), ('C_Test', ct_test)]:
    df_c = merge_window_signals(df_c, window_market, qw_window, bucket_vwaps)
    for col in market_signal_cols + qw_window_cols + growth_signal_cols:
        cs_col = f'cs_{col}'
        if col in df_c.columns:
            df_c[cs_col] = cs_rank(df_c[col].fillna(0.0), df_c['dt'].dt.date)
    if name == 'C_Train': ct_train = df_c
    elif name == 'C_Val': ct_val = df_c
    elif name == 'C_Test': ct_test = df_c

# Signal quality report on copy-val
copy_sig_cols = [c for c in selected if c in ct_val.columns]
if len(copy_sig_cols) >= 1:
    print("\\nSignal IC on copy-universe trades:")
    for sig in copy_sig_cols:
        ic_val = compute_event_ic(ct_val[sig], ct_val['copyable_roi']) or 0.0
        ic_test = compute_event_ic(ct_test[sig], ct_test['copyable_roi']) or 0.0
        print(f"  {sig:35s}  Val IC={ic_val:.4f}  Test IC={ic_test:.4f}")
'''

CELL_STRATEGY = '''
# Strategy evaluation with composite score on copy trades

copy_sig_cols = [c for c in selected if c in ct_val.columns]
if len(copy_sig_cols) < 1:
    print("No signals available for strategy")
else:
    print(f"Using {len(copy_sig_cols)} signals: {copy_sig_cols}")

    # Composite: equal weight
    for df_c in [ct_train, ct_val, ct_test]:
        df_c['composite'] = df_c[copy_sig_cols].fillna(0.0).sum(axis=1) / len(copy_sig_cols)

    # Baseline: all trades
    all_roi = ct_test['copyable_roi'].mean()
    all_cpnl = ct_test['copyable_pnl'].sum()
    print(f"\\nBaseline (all copy trades): Test ROI={all_roi:.4f}  Test CPnL={all_cpnl:.2f}")

    # Strategy at various thresholds
    print(f"\\n{'threshold':>10s} {'trades':>8s} {'roi':>10s} {'cpnl':>10s} {'cpnl_diff':>10s}")
    print("-" * 50)
    best_spnl = -float('inf')
    best_thresh = 0.0
    for thresh in np.arange(-0.5, 0.51, 0.1):
        mask = ct_test['composite'].fillna(thresh - 1) >= thresh
        if mask.sum() < 5:
            continue
        roi = ct_test.loc[mask, 'copyable_roi'].mean()
        cpnl = ct_test.loc[mask, 'copyable_pnl'].sum()
        diff = cpnl - all_cpnl * (mask.sum() / len(ct_test))
        print(f"{thresh:10.1f} {mask.sum():8d} {roi:10.4f} {cpnl:10.2f} {diff:10.2f}")
        if cpnl > best_spnl:
            best_spnl = cpnl
            best_thresh = thresh

    print(f"\\nBest threshold: {best_thresh:.2f} (CPnL={best_spnl:.2f})")

    # Evaluate best on all splits
    print(f"\\n{'Split':>10s} {'threshold':>10s} {'trades':>8s} {'roi':>10s} {'cpnl':>10s} {'all_roi':>10s} {'all_cpnl':>10s}")
    print("-" * 70)
    for label, df_c in [('Train', ct_train), ('Val', ct_val), ('Test', ct_test)]:
        mask = df_c['composite'].fillna(best_thresh - 1) >= best_thresh
        all_roi_split = df_c['copyable_roi'].mean()
        all_cpnl_split = df_c['copyable_pnl'].sum()
        if mask.sum() < 5:
            sig_roi, sig_cpnl = 0.0, 0.0
        else:
            sig_roi = df_c.loc[mask, 'copyable_roi'].mean()
            sig_cpnl = df_c.loc[mask, 'copyable_pnl'].sum()
        print(f"{label:>10s} {best_thresh:10.2f} {mask.sum():8d} {sig_roi:10.4f} {sig_cpnl:10.2f} {all_roi_split:10.4f} {all_cpnl_split:10.2f}")
'''

# --- Assemble notebook ---

new_cells = [
    md("## Stage 2: Signal Discovery\n\n"
       "**New approach:** Test signals on a broad random sample of trades first (all wallets).\n"
       "Only apply surviving signals to copy-universe trades.\n\n"
       "Key changes vs Stage 1:\n"
       "- Continuous wallet quality scores (instead of binary filters)\n"
       "- Market-level signals from ALL trades (not just quality wallets)\n"
       "- Forward selection: add signals conditionally\n"
       "- Broad sample = 200K random BUY trades (much higher statistical power)"),
]
new_cells.append(md("## Parameters"))
new_cells.append(code(CELL_PARAMS))

new_cells.append(code("""
# Load trades + wallet metrics (from stage1's first 21 cells)
# wallet_vol and df_full are already loaded in the preceding cells
print(f"Trades loaded: {len(df_full):,}")
print(f"Wallets: {df_full['wallet'].nunique():,}")
"""))

new_cells.append(md("## Wallet Quality Scoring"))
new_cells.append(code(CELL_QUALITY_SCORE))

new_cells.append(md("## Signal Computation"))
new_cells.append(md("### Step 1: Market-level Window Aggregations\n\n"
                     "Compute per-(condition_id, outcome, 15min window) aggregations "
                     "from ALL trades. These are merged back to each trade using the "
                     "previous window's values to prevent look-ahead bias."))
new_cells.append(code(CELL_MARKET_WINDOW))
new_cells.append(code(CELL_QUALITY_WINDOW))
new_cells.append(code(CELL_QW_PROXIMITY))

new_cells.append(md("### Step 2: VWAP Deviation"))
new_cells.append(code(CELL_WALLET_MIN_TRADES))
new_cells.append(code(CELL_VWAP_WALLET_SETS))
new_cells.append(code(CELL_VWAP_VARIANTS))

new_cells.append(md("### Step 3: Create Broad Sample\n\n"
                     "Now that df_full is fully annotated, sample 200K BUY trades "
                     "from active wallets for signal discovery."))
new_cells.append(code(CELL_BROAD_SAMPLE))

new_cells.append(md("### Step 4: Merge Signals into Sample"))
new_cells.append(code(CELL_MERGE_ALL))
new_cells.append(md("## Signal Evaluation"))
new_cells.append(code(CELL_EVAL_SIGNALS))
new_cells.append(code(CELL_FORWARD_SELECTION))
new_cells.append(md("## Parameter Tuning"))
new_cells.append(code(CELL_TUNE_VWAP))
new_cells.append(md("### Tune Signal Window Size"))
new_cells.append(code(CELL_TUNE_WINDOW))
new_cells.append(md("### Tune QW Threshold"))
new_cells.append(code(CELL_TUNE_QW))
new_cells.append(md("## Apply to Copy Trades"))
new_cells.append(code(CELL_COPY_APPLICATION))
new_cells.append(code(CELL_STRATEGY))
new_cells.append(md("## Summary"))
new_cells.append(md("""
**Results Summary**

We tested signals on a broad random sample of 200K BUY trades from wallets with 100+ trades.
Signals that passed initial screening were applied to copy-universe trades.

Key metrics:
- Number of signals tested
- Number passing IC threshold
- Composite performance on copy trades vs baseline
"""))

# === Assemble notebook ===

kernelspec = {
    "display_name": "polymarket-analysis-BY1ldWyW-py3.13",
    "language": "python",
    "name": "python3",
}
language_info = {
    "codemirror_mode": {"name": "ipython", "version": 3},
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
    "version": "3.13.7",
}

new_nb = {
    "cells": old_cells + new_cells,
    "metadata": {"kernelspec": kernelspec, "language_info": language_info},
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT_NOTEBOOK, "w") as f:
    json.dump(new_nb, f, indent=1, ensure_ascii=False)

print(f"Written: {len(new_nb['cells'])} cells total")
print(f"  {len(old_cells)} old cells kept + {len(new_cells)} new cells appended")
