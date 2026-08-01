"""
Positioning-signal exploration for the copy-wallet universe.

For each wallet archetype (gambler, whale, retail, scalper, overseller, ...),
compute the aggregate open-position state of that archetype on the candidate
token at trade time, and the distance of the current price from their average
entry (avgc), then evaluate IC on forward copyable ROI.

Selection is train+val sign-consistency (test = diagnostics), matching the
notebook's protocol.  Output is printed; cache the slow prep steps.
"""
import os
import sys
import time
import pickle

sys.path.insert(0, '/Users/vobornij/projects/polymarket/notebooks/wallet_selection')

import numpy as np
import pandas as pd

from lib import load_trades, split_data, compute_copyable_notional, compute_opening_metrics
from polymarket_analysis.wallet_selection.volatility import compute_wallet_metrics
from signal_lab.signal_lib import compute_event_ic
from signal_lab.signal_engines import PositionSignalEngine, compute_hold_time_metrics, archetype_sets

CACHE = '/tmp/pos_explore_cache'
os.makedirs(CACHE, exist_ok=True)

COPY = dict(COPY_MIN_BUY_ROI=0.02, COPY_MIN_BUCKETS=20, COPY_MIN_MARKETS=15,
            COPY_MIN_TRADE_COUNT=100, COPY_MAX_DD_TO_PNL=0.6, COPY_MIN_COPYABLE_ROI=0.05)


def load_prep(force=False):
    """Load data once and cache wallet_vol + candidate frames + hold metrics."""
    wp = os.path.join(CACHE, 'wallet_vol.parquet')
    hp = os.path.join(CACHE, 'hold.parquet')
    cp = os.path.join(CACHE, 'cand.parquet')
    if not force and all(os.path.exists(p) for p in (wp, hp, cp)):
        wallet_vol = pd.read_parquet(wp)
        hold = pd.read_parquet(hp)
        cands = pd.read_parquet(cp)
        print(f'cached: wallet_vol={len(wallet_vol)} hold={len(hold)} cands={len(cands)}')
        return wallet_vol, hold, cands

    df_full = load_trades()
    df_full = compute_copyable_notional(df_full)
    df_train, df_val, df_test = split_data(df_full, method='chronological')

    wallet_vol, _ = compute_wallet_metrics(df_train)
    wallet_vol["copyable_pnl_factor"] = np.clip(
        wallet_vol["copyable_pnl"] / wallet_vol["total_pnl"].replace(0, np.nan),
        0, 1.0).fillna(0.0)
    wallet_vol["copyable_roi"] = wallet_vol["average_roi"] * wallet_vol["copyable_pnl_factor"]
    om = compute_opening_metrics(df_train)
    wallet_vol = wallet_vol.merge(om, on="wallet", how="left")
    for c in ["opening_roi", "opening_pnl", "opening_copyable_roi", "opening_copyable_pnl"]:
        wallet_vol[c] = wallet_vol[c].fillna(0.0)

    hold = compute_hold_time_metrics(df_train)
    wallet_vol = wallet_vol.merge(hold, on='wallet', how='left')

    cm = ((wallet_vol['buy_roi'] >= COPY['COPY_MIN_BUY_ROI'])
          & (wallet_vol['num_buckets'] >= COPY['COPY_MIN_BUCKETS'])
          & (wallet_vol['num_markets'] >= COPY['COPY_MIN_MARKETS'])
          & (wallet_vol['trade_count'] >= COPY['COPY_MIN_TRADE_COUNT'])
          & (wallet_vol['max_drawdown_to_pnl'].fillna(1.0) <= COPY['COPY_MAX_DD_TO_PNL'])
          & (wallet_vol['copyable_roi'].fillna(0.0) >= COPY['COPY_MIN_COPYABLE_ROI']))
    copy_wallets = set(wallet_vol.loc[cm, 'wallet'])
    cmask = df_full['wallet'].isin(copy_wallets) & (df_full['side'] == 'BUY')
    cands = df_full[cmask][['wallet', 'dt', 'condition_id', 'outcome', 'price',
                            'copyable_roi', 'copyable_pnl', 'copyable_notional']].copy()
    del df_full, df_train, df_val, df_test

    wallet_vol.to_parquet(wp)
    hold.to_parquet(hp)
    cands.to_parquet(cp)
    print(f'cached to {CACHE}')
    return wallet_vol, hold, cands


def split_candidates(cands):
    t0 = pd.Timestamp('2026-05-21', tz='UTC')
    v0 = pd.Timestamp('2026-06-23', tz='UTC')
    c_train = cands[cands['dt'] < t0].copy()
    c_val = cands[(cands['dt'] >= t0) & (cands['dt'] < v0)].copy()
    c_test = cands[cands['dt'] >= v0].copy()
    print(f'candidates: train={len(c_train):,} val={len(c_val):,} test={len(c_test):,}')
    return c_train, c_val, c_test


def main(step='all'):
    wallet_vol, hold, cands = load_prep(force=False)
    c_train, c_val, c_test = split_candidates(cands)
    conditions = set(cands['condition_id'].unique())
    print(f'candidate conditions: {len(conditions):,}')

    if step in ('all', 'archetypes'):
        sets = archetype_sets(wallet_vol, hold, min_trade_count=100)
        print('\n=== Archetype sets (train metrics) ===')
        for name, sel in sets.items():
            w = wallet_vol[wallet_vol['wallet'].isin(sel['wallet'])]
            print(f'  {name:18s} n_wallets={len(sel):5d}  '
                  f'total_pnl=${w["total_pnl"].sum():>12,.0f}  '
                  f'trades={int(w["trade_count"].sum()):>9,}  '
                  f'copyable_roi={w["copyable_roi"].mean():.4f}')
        with open(os.path.join(CACHE, 'archetype_sets.pkl'), 'wb') as f:
            pickle.dump({k: v['wallet'].tolist() for k, v in sets.items()}, f)
    else:
        with open(os.path.join(CACHE, 'archetype_sets.pkl'), 'rb') as f:
            sets = {k: wallet_vol[wallet_vol['wallet'].isin(v)] for k, v in pickle.load(f).items()}

    if step in ('all', 'pos'):
        engine = PositionSignalEngine(load_restricted_df(conditions))
        print('\n=== Position signal sweep (train+val sign-consistent selection) ===')
        all_rows = []
        selected = []
        for name, sel in sets.items():
            t0 = time.time()
            rep, sel_cols = position_report_one(engine, c_train, c_val, c_test,
                                                set(sel['wallet']), name,
                                                conditions=conditions)
            dt = time.time() - t0
            all_rows.append(rep)
            selected.extend(sel_cols)
            print(f'  {name}: {dt:.0f}s  selected={sel_cols}')
        rep = pd.concat(all_rows, ignore_index=True)
        rep['|IC_train|'] = rep['IC_train'].abs()
        rep = rep.sort_values('|IC_train|', ascending=False).reset_index(drop=True)
        print('\n=== Position signal ICs (all archetypes) ===')
        print(rep.round(4).to_string(index=False))
        rep.to_csv(os.path.join(CACHE, 'position_report.csv'), index=False)
        print('\nSelected:', selected)


def position_report_one(engine, c_train, c_val, c_test, wallets, name, conditions=None,
                        min_ic=0.005, presence_min=0.005):
    from signal_lab.signal_engines import ALL_POSITION_KINDS, position_report
    return position_report(engine, c_train, c_val, c_test, wallets, name,
                           kinds=ALL_POSITION_KINDS, min_ic=min_ic,
                           presence_min=presence_min, conditions=conditions)


_restricted = None
_MIN_COLS = ['wallet', 'condition_id', 'outcome', 'dt', 'side', 'position',
             'quantity', 'price']


def load_restricted_df(conditions):
    """df_full restricted to candidate conditions (enough for position signals).

    Uses a minimal column-schema copy of the cache to keep memory light under
    swap pressure.
    """
    global _restricted
    if _restricted is not None:
        return _restricted
    p = os.path.join(CACHE, 'df_restricted_min.parquet')
    if os.path.exists(p):
        _restricted = pd.read_parquet(p)
        print(f'loaded restricted df: {len(_restricted):,} rows')
        return _restricted
    df_full = load_trades()
    df_full = compute_copyable_notional(df_full)
    cond_arr = np.asarray(list(conditions), dtype=object)
    sub = df_full[df_full['condition_id'].isin(cond_arr)][_MIN_COLS].copy()
    del df_full
    _restricted = sub
    sub.to_parquet(p)
    print(f'saved restricted df: {len(sub):,} rows')
    return sub


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else 'all'
    main(step)
