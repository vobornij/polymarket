"""
Fresh-family sweep: recency-weighted (recent-entry) position signals.

For each focus archetype x tau in {1h, 6h, 1d}, build_set with fresh_tau_ns,
attach own+opp fresh variants, and report IC on train/val/test + daily IR.

Run per-archetype in small steps:  python fresh_sweep.py gambler  [tau list]
"""
import os
import sys
import time
import pickle

sys.path.insert(0, '/Users/vobornij/projects/polymarket/notebooks/wallet_selection')

import numpy as np
import pandas as pd

from signal_lab.signal_lib import compute_event_ic
from signal_lab.signal_engines import PositionSignalEngine

CACHE = '/tmp/pos_explore_cache'
T0 = pd.Timestamp('2026-05-21', tz='UTC')
V0 = pd.Timestamp('2026-06-23', tz='UTC')
HOUR_NS = int(3600 * 1e9)
DEFAULT_TAUS = [1 * HOUR_NS, 6 * HOUR_NS, 24 * HOUR_NS]

FRESH_COLS = ['sig_fpos_own', 'sig_fpos_opp', 'sig_fval_own', 'sig_fval_opp',
              'sig_favgc_own', 'sig_favgc_opp', 'sig_fuwl_own', 'sig_fuwl_opp']
PLAIN_COLS = ['sig_pos_own', 'sig_pos_opp', 'sig_val_own', 'sig_val_opp',
              'sig_uwl_own', 'sig_uwl_opp']


def split_candidates(cands):
    c_train = cands[cands['dt'] < T0].copy()
    c_val = cands[(cands['dt'] >= T0) & (cands['dt'] < V0)].copy()
    c_test = cands[cands['dt'] >= V0].copy()
    return c_train, c_val, c_test


def daily_ic_consistency(s, roi, dt):
    df = pd.DataFrame({'s': s, 'roi': roi, 'dt': dt})
    ics = []
    for _, g in df.groupby(pd.Grouper(key='dt', freq='D')):
        if len(g) < 20:
            continue
        ic = compute_event_ic(g['s'], g['roi'])
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.array(ics)
    if len(ics) < 5:
        return np.nan, np.nan, np.nan
    return ics.mean(), ics.std(ddof=1), float((ics < 0).mean())


def run_archetype(engine, sets, cands_splits, name, taus):
    c_train, c_val, c_test = cands_splits
    conditions = set(c_train['condition_id'].unique()) | set(c_val['condition_id'].unique()) | set(c_test['condition_id'].unique())
    wallets = set(sets[name])
    rows = []
    for tau in taus:
        tag = f'{tau // HOUR_NS:>3d}h'
        t0 = time.time()
        A, B = engine.build_set(wallets, conditions, fresh_tau_ns=tau)
        for df_c in (c_train, c_val, c_test):
            engine.attach_position_signals(df_c, name, A, B)
        print(f'  [{name} tau={tau // HOUR_NS}h] build+attach {time.time() - t0:.1f}s')
        for col in FRESH_COLS + PLAIN_COLS:
            col = f'{col}_{name}'
            ics = {lbl: compute_event_ic(df_c[col].fillna(0.0), df_c['copyable_roi'])
                   for lbl, df_c in [('train', c_train), ('val', c_val), ('test', c_test)]}
            mu, sd, neg = daily_ic_consistency(
                pd.concat([c_train[col], c_val[col]], ignore_index=True).fillna(0.0),
                pd.concat([c_train['copyable_roi'], c_val['copyable_roi']], ignore_index=True),
                pd.concat([c_train['dt'], c_val['dt']], ignore_index=True))
            rows.append({
                'signal': col.replace(f'_{name}', ''), 'tau_h': tau // HOUR_NS,
                'IC_train': ics['train'], 'IC_val': ics['val'], 'IC_test': ics['test'],
                'daily_mu': mu, 'daily_sd': sd, 'frac_neg_days': neg,
                'IR_daily': mu / sd if np.isfinite(sd) and sd > 0 else np.nan,
            })
    rep = pd.DataFrame(rows)
    rep['|IC_train|'] = rep['IC_train'].abs()
    rep = rep.sort_values('|IC_train|', ascending=False).reset_index(drop=True)
    print(f'\n=== {name}: fresh-family IC (sorted by |IC_train|) ===')
    print(rep.round(4).to_string(index=False))
    rep.to_csv(os.path.join(CACHE, f'fresh_report_{name}.csv'), index=False)


def main():
    focus = [a for a in sys.argv[1].split(',') if a] if len(sys.argv) > 1 else ['gambler']
    taus = [int(t) for t in sys.argv[2].split(',')] if len(sys.argv) > 2 else DEFAULT_TAUS

    cands = pd.read_parquet(os.path.join(CACHE, 'cand.parquet'))
    cands_splits = split_candidates(cands)
    print(f'candidates: train={len(cands_splits[0]):,} val={len(cands_splits[1]):,} '
          f'test={len(cands_splits[2]):,}')

    df_restricted = pd.read_parquet(os.path.join(CACHE, 'df_restricted_min.parquet'))
    t_eng = time.time()
    engine = PositionSignalEngine(df_restricted)
    print(f'engine init: {time.time() - t_eng:.1f}s')

    with open(os.path.join(CACHE, 'archetype_sets.pkl'), 'rb') as f:
        sets = pickle.load(f)
    print(f'available archetypes: {sorted(sets.keys())}')

    for name in focus:
        run_archetype(engine, sets, cands_splits, name, taus)


if __name__ == '__main__':
    main()
