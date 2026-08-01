"""
Price-residualized evaluation of position signals.

Why: forward copyable_roi is dominated by a favorite effect (buying at high
price => more likely to win), so IC(price, roi) ~ +0.5 and any signal that
correlates with price inherits spurious strength.  To measure a signal's
incremental information beyond price we residualize ROI against price:

  1. Van der Waerden (rank -> standard normal) scores of ROI and price.
  2. OLS  rank_roi ~ beta * rank_price  fitted on TRAIN only.
  3. Fixed beta/intercept applied to val and test (no refit / no leakage).
  4. Signal IC = Spearman(signal, residual) per split; daily IR likewise.

Selection stays train+val sign-consistency with |IC| >= min_ic on residuals.
"""
import os
import sys
import time
import pickle

sys.path.insert(0, '/Users/vobornij/projects/polymarket/notebooks/wallet_selection')

import numpy as np
import pandas as pd

from signal_lib import compute_event_ic, fit_roi_residualizer, residualized_roi
from signal_engines import PositionSignalEngine

CACHE = '/tmp/pos_explore_cache'
T0 = pd.Timestamp('2026-05-21', tz='UTC')
V0 = pd.Timestamp('2026-06-23', tz='UTC')

KINDS = ['pos', 'val', 'avgc', 'uwl']
VARS = ['own', 'opp', 'total']


def split_candidates(cands):
    c_train = cands[cands['dt'] < T0].copy()
    c_val = cands[(cands['dt'] >= T0) & (cands['dt'] < V0)].copy()
    c_test = cands[cands['dt'] >= V0].copy()
    return c_train, c_val, c_test


def daily_ic_consistency(s, roi_res, dt):
    df = pd.DataFrame({'s': s, 'roi': roi_res, 'dt': dt})
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


def main():
    cands = pd.read_parquet(os.path.join(CACHE, 'cand.parquet'))
    c_train, c_val, c_test = split_candidates(cands)
    conditions = set(cands['condition_id'].unique())
    print(f'candidates: train={len(c_train):,} val={len(c_val):,} test={len(c_test):,}')

    fit = fit_roi_residualizer(c_train['copyable_roi'], c_train['price'])
    print(f'residualizer fit (train): beta={fit["beta"]:+.4f} intercept={fit["intercept"]:+.4f}')
    for lbl, df_c in [('train', c_train), ('val', c_val), ('test', c_test)]:
        df_c['roi_res'] = residualized_roi(df_c['copyable_roi'], df_c['price'], fit)
        ic_price = compute_event_ic(df_c['price'], df_c['roi_res'])
        ic_roi = compute_event_ic(df_c['copyable_roi'], df_c['roi_res'])
        print(f'  {lbl}: IC(price, roi_res)={ic_price:+.4f}  IC(roi, roi_res)={ic_roi:+.4f}')

    df_restricted = pd.read_parquet(os.path.join(CACHE, 'df_restricted_min.parquet'))
    t_eng = time.time()
    engine = PositionSignalEngine(df_restricted)
    print(f'engine init: {time.time() - t_eng:.1f}s')

    with open(os.path.join(CACHE, 'archetype_sets.pkl'), 'rb') as f:
        sets = pickle.load(f)

    rows, selected = [], []
    for name, sel in sorted(sets.items()):
        t0 = time.time()
        A, B = engine.build_set(set(sel), conditions)
        for df_c in (c_train, c_val, c_test):
            engine.attach_position_signals(df_c, name, A, B)
        print(f'  {name}: build+attach {time.time() - t0:.0f}s', flush=True)

        for kind in KINDS:
            for var in VARS:
                if var == 'total' and kind in ('avgc', 'uwl'):
                    continue
                col = f'sig_{kind}_{var}_{name}'
                pres = float((c_train[col] > 0).mean())
                ics = {lbl: compute_event_ic(df_c[col].fillna(0.0), df_c['roi_res'])
                       for lbl, df_c in [('train', c_train), ('val', c_val), ('test', c_test)]}
                rows.append({'signal': col, 'kind': f'{kind}_{var}',
                             'presence_train': pres,
                             'IC_train': ics['train'], 'IC_val': ics['val'],
                             'IC_test': ics['test']})
                if (np.isfinite(ics['train']) and np.isfinite(ics['val'])
                        and np.sign(ics['train']) == np.sign(ics['val'])
                        and abs(ics['train']) >= 0.005 and abs(ics['val']) >= 0.005
                        and pres >= 0.005):
                    selected.append(col)
        pd.DataFrame(rows).to_csv(os.path.join(CACHE, 'position_report_resid.csv'), index=False)

    rep = pd.DataFrame(rows)
    rep['|IC_train|'] = rep['IC_train'].abs()
    rep = rep.sort_values('|IC_train|', ascending=False).reset_index(drop=True)
    print('\n=== Position signal IC on price-residualized ROI (sorted by |IC_train|) ===')
    print(rep.round(4).to_string(index=False))
    rep.to_csv(os.path.join(CACHE, 'position_report_resid.csv'), index=False)

    print('\n=== Selected (sign-consistent train+val, |IC|>=0.005) ===')
    print(' ', selected)

    print('\n=== Daily IR on residuals for selected signals (train+val) ===')
    c_tv = pd.concat([c_train, c_val], ignore_index=True)
    for col in selected:
        mu, sd, neg = daily_ic_consistency(c_tv[col].fillna(0.0), c_tv['roi_res'], c_tv['dt'])
        print(f'  {col:30s} daily_mu={mu:+.4f} sd={sd:.4f} frac_neg={neg:.2f} '
              f'IR={mu / sd:+.2f}' if np.isfinite(sd) and sd > 0
              else f'  {col:30s} daily_mu={mu:+.4f} sd={sd:.4f} frac_neg={neg:.2f}')

    # persist residualized frames for follow-ups
    for lbl, df_c in [('train', c_train), ('val', c_val), ('test', c_test)]:
        df_c.drop(columns='roi_res', errors='ignore').to_parquet(
            os.path.join(CACHE, f'resid_attached_{lbl}.parquet'))
    print('\ncached attached frames -> /tmp/pos_explore_cache/resid_attached_*.parquet')


if __name__ == '__main__':
    main()
