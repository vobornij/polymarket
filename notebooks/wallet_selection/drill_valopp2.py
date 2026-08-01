"""
val_opp follow-up: confounder + monotonicity checks.

Questions:
1. Is val_opp just candidate price level in disguise?  (IC of price alone;
   IC of val_opp within price terciles.)
2. Corrected monotonicity: zero-bucket + quartiles of fired, using MEDIAN
   forward copyable_roi (robust to outliers).
3. Daily IC consistency: share of days with negative IC, mean/std daily IC.
"""
import os
import sys
import time

sys.path.insert(0, '/Users/vobornij/projects/polymarket/notebooks/wallet_selection')

import numpy as np
import pandas as pd

from signal_lab.signal_lib import compute_event_ic
from signal_lab.signal_engines import PositionSignalEngine

CACHE = '/tmp/pos_explore_cache'
t0 = pd.Timestamp('2026-05-21', tz='UTC')
v0 = pd.Timestamp('2026-06-23', tz='UTC')

FOCUS = ['gambler', 'flipper', 'retail']


def split_candidates(cands):
    c_train = cands[cands['dt'] < t0].copy()
    c_val = cands[(cands['dt'] >= t0) & (cands['dt'] < v0)].copy()
    c_test = cands[cands['dt'] >= v0].copy()
    return c_train, c_val, c_test


def daily_ic_consistency(s, roi, dt):
    df = pd.DataFrame({'s': s, 'roi': roi, 'dt': dt})
    by_day = df.groupby(pd.Grouper(key='dt', freq='D'))
    ics = []
    for _, g in by_day:
        if len(g) < 20:
            continue
        ic = compute_event_ic(g['s'], g['roi'])
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.array(ics)
    if len(ics) < 5:
        return np.nan, np.nan, np.nan
    return ics.mean(), ics.std(ddof=1), float((ics < 0).mean())


def main():
    cands = pd.read_parquet(os.path.join(CACHE, 'cand.parquet'))
    c_train, c_val, c_test = split_candidates(cands)
    conditions = set(cands['condition_id'].unique())
    print(f'candidates: train={len(c_train):,} val={len(c_val):,} test={len(c_test):,}')

    df_restricted = pd.read_parquet(os.path.join(CACHE, 'df_restricted_min.parquet'))
    t_eng = time.time()
    engine = PositionSignalEngine(df_restricted)
    print(f'engine init: {time.time() - t_eng:.1f}s')

    import pickle
    with open(os.path.join(CACHE, 'archetype_sets.pkl'), 'rb') as f:
        sets = pickle.load(f)
    for name in FOCUS:
        t0_ = time.time()
        A_tbl, B_tbl = engine.build_set(set(sets[name]), conditions)
        for df_c in (c_train, c_val, c_test):
            engine.attach_position_signals(df_c, name, A_tbl, B_tbl)
        print(f'  {name}: build+attach {time.time() - t0_:.1f}s')

    # ---- 1. price-level confounder ----
    print('\n=== confounder: candidate BUY price ===')
    for lbl, df_c in [('train', c_train), ('val', c_val), ('test', c_test)]:
        ic_p = compute_event_ic(df_c['price'], df_c['copyable_roi'])
        print(f'  IC(price, roi) {lbl}: {ic_p:+.4f}')

    for name in FOCUS:
        col = f'sig_val_opp_{name}'
        print(f'\n  val_opp IC within price terciles ({name}):')
        prices = np.quantile(c_train['price'], [1 / 3, 2 / 3])
        print(f'    price edges: {prices[0]:.3f} / {prices[1]:.3f}')
        for lbl, df_c in [('train', c_train), ('val', c_val), ('test', c_test)]:
            cells = []
            for lo, hi, tag in [(0.0, prices[0], 'low'), (prices[0], prices[1], 'mid'),
                                (prices[1], 1.0, 'high')]:
                m = (df_c['price'] >= lo) & (df_c['price'] < hi)
                ic = compute_event_ic(df_c.loc[m, col].fillna(0.0),
                                      df_c.loc[m, 'copyable_roi']) if m.sum() > 50 else np.nan
                cells.append(f'{tag}={ic:+.4f}')
            print(f'    {lbl:5s}  ' + '  '.join(cells))

    # ---- 2. corrected monotonicity (zero bucket + quartiles, median roi) ----
    print('\n=== val_opp quantiles (zero + fired quartiles; MEDIAN roi) ===')
    for name in FOCUS:
        col = f'sig_val_opp_{name}'
        fired = c_train[col] > 0
        edges = np.quantile(c_train.loc[fired, col], [0.25, 0.5, 0.75])
        print(f'\n  {name}: fired edges={np.round(edges, 1)}  fired_share={fired.mean():.3f}')
        print('    bucket   train          val            test')
        for lbl, df_c in [('train', c_train), ('val', c_val), ('test', c_test)]:
            s = df_c[col]
            roi = df_c['copyable_roi']
            labels = np.select(
                [s <= 0,
                 s <= edges[0], s <= edges[1], s <= edges[2]],
                ['zero', 'Q1', 'Q2', 'Q3'], default='Q4')
            g = pd.DataFrame({'b': labels, 'roi': roi, 'pnl': df_c['copyable_pnl']})
            cells = []
            for b in ['zero', 'Q1', 'Q2', 'Q3', 'Q4']:
                sub = g[g['b'] == b]['roi'].dropna()
                if len(sub) == 0:
                    cells.append(f'{b:6s}=   --   ({int((g["b"] == b).sum()):>7,})')
                    continue
                cells.append(f'{b:6s}={np.median(sub):+.4f} ({int((g["b"] == b).sum()):>7,})')
            print(f'    {lbl:5s}  ' + '  '.join(cells))

    # ---- 3. daily IC consistency (train+val) ----
    print('\n=== val_opp daily IC consistency (train+val combined) ===')
    c_tv = pd.concat([c_train, c_val], ignore_index=True)
    for name in FOCUS:
        col = f'sig_val_opp_{name}'
        mu, sd, neg = daily_ic_consistency(c_tv[col].fillna(0.0),
                                           c_tv['copyable_roi'], c_tv['dt'])
        print(f'  {name:8s} mean_daily_IC={mu:+.4f}  sd={sd:.4f}  frac_neg_days={neg:.2f}  '
              f'IR~{mu / sd:+.2f}')

    # persist attached frames for cheap follow-ups
    out = {lbl: df for lbl, df in [('train', c_train), ('val', c_val), ('test', c_test)]}
    for lbl, df in out.items():
        df.to_parquet(os.path.join(CACHE, f'valopp_attached_{lbl}.parquet'))
    print('\ncached attached frames -> /tmp/pos_explore_cache/valopp_attached_*.parquet')


if __name__ == '__main__':
    main()
