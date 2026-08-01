"""
Drill-down on the `val_opp_*` signal family.

Finding so far: when an archetype holds high value-at-cost on the OPPOSITE
outcome, the candidate BUY underperforms (negative IC, most coherent on
gambler / flipper / retail).  This script checks:

1. presence + IC train/val/test per archetype (and the union of the three)
2. binary (any exposure) vs magnitude decomposition
3. quantile monotonicity of forward copyable ROI vs val_opp level
4. does `val_total` (whole-market value-at-cost) explain it away?
5. day-level IR (consistency over time)
6. wallet overlap between the three archetypes

Uses the cached prep from explore_positioning.  Small, fast test.
"""
import os
import sys
import pickle
import time

sys.path.insert(0, '/Users/vobornij/projects/polymarket/notebooks/wallet_selection')

import numpy as np
import pandas as pd

from signal_lab.signal_lib import compute_event_ic, compute_event_ir
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


def main():
    cands = pd.read_parquet(os.path.join(CACHE, 'cand.parquet'))
    c_train, c_val, c_test = split_candidates(cands)
    conditions = set(cands['condition_id'].unique())
    print(f'candidates: train={len(c_train):,} val={len(c_val):,} test={len(c_test):,}')

    with open(os.path.join(CACHE, 'archetype_sets.pkl'), 'rb') as f:
        sets = pickle.load(f)
    union = sorted(set().union(*(sets[n] for n in FOCUS)))
    print(f'union({FOCUS}): {len(union)} wallets')

    print('\nloading restricted df...')
    df_restricted = pd.read_parquet(os.path.join(CACHE, 'df_restricted_min.parquet'))
    print(f'  {len(df_restricted):,} rows')

    t_eng = time.time()
    engine = PositionSignalEngine(df_restricted)
    print(f'  engine init: {time.time() - t_eng:.1f}s')

    sets_to_run = FOCUS + ['union']
    A = {}
    for name in sets_to_run:
        wallets = set(sets[name]) if name != 'union' else set(union)
        t0_ = time.time()
        A_tbl, B_tbl = engine.build_set(wallets, conditions)
        A[name] = (A_tbl, B_tbl)
        for df_c in (c_train, c_val, c_test):
            engine.attach_position_signals(df_c, name, A_tbl, B_tbl)
        print(f'  {name}: build+attach {time.time() - t0_:.1f}s')

    # ---- 1. presence + IC ----
    print('\n=== val_opp IC (presence / train / val / test) ===')
    for name in sets_to_run:
        col = f'sig_val_opp_{name}'
        pres = float((c_train[col] > 0).mean())
        ics = [compute_event_ic(df_c[col].fillna(0.0), df_c['copyable_roi'])
               for df_c in (c_train, c_val, c_test)]
        print(f'  {name:9s} pres={pres:.3f}  IC={ics[0]:+.4f} / {ics[1]:+.4f} / {ics[2]:+.4f}')

    # ---- 2. binary vs magnitude decomposition ----
    print('\n=== val_opp: binary vs magnitude decomposition (train/val/test) ===')
    for name in sets_to_run:
        col = f'sig_val_opp_{name}'
        row = {'set': name}
        for lbl, df_c in [('tr', c_train), ('va', c_val), ('te', c_test)]:
            s = df_c[col]
            roi = df_c['copyable_roi']
            bin_ic = compute_event_ic((s > 0).astype(float), roi)
            fired = s > 0
            fired_ic = compute_event_ic(s[fired], roi[fired]) if fired.sum() > 20 else np.nan
            row[f'bin_{lbl}'] = bin_ic
            row[f'mag_fired_{lbl}'] = fired_ic
        print(f"  {name:9s} " + "  ".join(f"{k}={v:+.4f}" for k, v in row.items() if k != 'set'))

    # ---- 3. quantile monotonicity ----
    print('\n=== val_opp quantiles: mean forward copyable_roi (bin edges on train) ===')
    for name in sets_to_run:
        col = f'sig_val_opp_{name}'
        fired = c_train[col] > 0
        edges = np.quantile(c_train.loc[fired, col], [0.25, 0.5, 0.75])
        print(f'\n  {name}: edges={np.round(edges, 1)}')
        print(f'    bin        train    val     test    (n train)')
        for lbl, df_c in [('train', c_train), ('val', c_val), ('test', c_test)]:
            pass
        out = {}
        for lbl, df_c in [('train', c_train), ('val', c_val), ('test', c_test)]:
            s = df_c[col]
            roi = df_c['copyable_roi']
            labels = pd.cut(s, [-np.inf, edges[0], edges[1], edges[2], np.inf],
                            labels=['Q1', 'Q2', 'Q3', 'Q4'])
            labels = labels.cat.add_categories(['none']).fillna('none')
            out[lbl] = roi.groupby(labels).agg(['mean', 'count'])
        for bin_ in ['none', 'Q1', 'Q2', 'Q3', 'Q4']:
            cells = []
            for lbl in ['train', 'val', 'test']:
                m, n = out[lbl].loc[bin_]
                cells.append(f'{m:+.4f} ({int(n):,})')
            print(f'    {bin_:4s}  {"   ".join(cells)}')

    # ---- 4. val_total cross-check ----
    print('\n=== val_total IC (does the opposite-outcome effect survive?) ===')
    for name in sets_to_run:
        col = f'sig_val_total_{name}'
        ics = [compute_event_ic(df_c[col].fillna(0.0), df_c['copyable_roi'])
               for df_c in (c_train, c_val, c_test)]
        print(f'  {name:9s} IC={ics[0]:+.4f} / {ics[1]:+.4f} / {ics[2]:+.4f}')

    # ---- 5. day-level IR ----
    print('\n=== val_opp daily IR (train / val) ===')
    for name in sets_to_run:
        col = f'sig_val_opp_{name}'
        irs = [compute_event_ir(df_c[col].fillna(0.0), df_c['copyable_roi'], df_c['dt'], freq='D')
               for df_c in (c_train, c_val)]
        print(f'  {name:9s} IR={irs[0]:+.2f} / {irs[1]:+.2f}')

    # ---- 6. wallet overlap ----
    print('\n=== wallet overlap (Jaccard) ===')
    names = FOCUS
    jac = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                jac[i, j] = 1.0
                continue
            sa, sb = set(sets[a]), set(sets[b])
            jac[i, j] = len(sa & sb) / len(sa | sb)
    jac_df = pd.DataFrame(jac, index=names, columns=names).round(3)
    print(jac_df.to_string())


if __name__ == '__main__':
    main()
