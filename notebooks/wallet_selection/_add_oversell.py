import json
import re

NB = 'stage1_experimental.ipynb'
nb = json.load(open(NB))

cells = nb['cells']
code_cells = [i for i, c in enumerate(cells) if c['cell_type'] == 'code']

# ---------------------------------------------------------------------------
# Cell A: oversell position signals (insert after the VWAP cell, before cell 38)
# ---------------------------------------------------------------------------
cell_a = """\
# ===== Oversell position signals =====
# Wallets that are net-negative on SELLs yet profitable overall (train-only
# wallet_vol selection). Signal = the EXACT aggregate position these wallets
# hold on the candidate's token at time t (own / opposite outcome), plus the
# market total (own + opposite). Exact aggregate via two-pass cumsum over
# post-trade position checkpoints (validated against a per-wallet brute force):
#   A(t) = cumsum of checkpoint positions at nearest checkpoint <= t
#   B(t) = cumsum of positions at each wallet's NEXT checkpoint <= t
#   signal = A(t) - B(t)  (0 where the set holds nothing at that time)

OS_POS_SIG = True

os_set_defs = {
    'os_pnl': (wallet_vol['sell_pnl'] < 0) & (wallet_vol['total_pnl'] > 0),
    'os_roi': (wallet_vol['sell_roi'] < 0) & (wallet_vol['total_pnl'] > 0),
    'os_active': (wallet_vol['sell_pnl'] < 0) & (wallet_vol['total_pnl'] > 0)
                 & (wallet_vol['sell_notional'] > 500),
    'os_buyprof': (wallet_vol['sell_pnl'] < 0) & (wallet_vol['total_pnl'] > 0)
                  & (wallet_vol['buy_pnl'] > 0),
}

OS_PRESENCE_MIN = 0.005  # min fraction of candidates where the set holds > 0

oversell_sets_info = {}
oversell_filter_names = []
oversell_ic_report = pd.DataFrame()

if OS_POS_SIG:
    def build_os_checkpoints(wallets):
        \"\"\"All trades (BUY+SELL) by `wallets` with post-trade position checkpoints.\"\"\"
        wallet_arr = np.asarray(list(wallets), dtype=object)
        present = np.isin(wallet_arr, _WALLET_CATEGORIES)
        codes_sub = np.searchsorted(_WALLET_CATEGORIES, wallet_arr[present])
        idx = np.flatnonzero(np.isin(_WALLET_CODES, codes_sub))
        return df_full.iloc[idx][['dt', 'wallet', 'condition_id', 'outcome', 'position']].copy()

    def _os_cumsum_tables(ck, by_cols):
        \"\"\"(A, B) asof tables keyed by by_cols for the exact aggregate position.\"\"\"
        key = ['wallet'] + by_cols
        B = ck.sort_values(key + ['dt'])
        B = B.assign(next_dt=B.groupby(key, sort=False)['dt'].shift(-1))
        B = B[B['next_dt'].notna()].drop(columns='dt').rename(columns={'next_dt': 'dt'})
        A = ck.sort_values(by_cols + ['dt'])
        A['cum'] = A.groupby(by_cols, sort=False)['position'].cumsum()
        A = A[by_cols + ['dt', 'cum']].sort_values('dt', kind='stable')
        B = B.sort_values(by_cols + ['dt'])
        B['cum'] = B.groupby(by_cols, sort=False)['position'].cumsum()
        B = B[by_cols + ['dt', 'cum']].sort_values('dt', kind='stable')
        return A, B

    def os_aggregate_position(cand, A, B, by_cols):
        \"\"\"Exact aggregate position of the set on cand's key at cand.dt.\"\"\"
        left = cand.sort_values('dt')[['dt'] + by_cols]
        idx = left.index
        a = pd.merge_asof(left, A, on='dt', by=by_cols, direction='backward',
                          allow_exact_matches=False)
        b = pd.merge_asof(left, B, on='dt', by=by_cols, direction='backward',
                          allow_exact_matches=False)
        res = a['cum'].fillna(0.0) - b['cum'].fillna(0.0)
        res.index = idx
        return res.sort_index()

    _os_swap = {'Yes': 'No', 'No': 'Yes'}
    _os_rows = []
    for _os_name, _os_mask in os_set_defs.items():
        _os_wallets = set(wallet_vol.loc[_os_mask, 'wallet'])
        if len(_os_wallets) < MIN_SET_SIZE:
            oversell_sets_info[_os_name] = {'n_wallets': len(_os_wallets), 'status': 'skipped'}
            continue
        _os_ck = build_os_checkpoints(_os_wallets)
        _A_own, _B_own = _os_cumsum_tables(_os_ck, ['condition_id', 'outcome'])
        _A_opp, _B_opp = _os_cumsum_tables(
            _os_ck.assign(outcome=_os_ck['outcome'].map(_os_swap)),
            ['condition_id', 'outcome'])
        for _var, (_A, _B) in [('own', (_A_own, _B_own)), ('opp', (_A_opp, _B_opp))]:
            _col = f'sig_os_{_var}_{_os_name}'
            for _df_c in [c_train, c_val, c_test]:
                _df_c[_col] = os_aggregate_position(_df_c, _A, _B, ['condition_id', 'outcome'])
        _col_total = f'sig_os_total_{_os_name}'
        for _df_c in [c_train, c_val, c_test]:
            _df_c[_col_total] = (_df_c[f'sig_os_own_{_os_name}']
                                 + _df_c[f'sig_os_opp_{_os_name}'])
        oversell_sets_info[_os_name] = {'n_wallets': len(_os_wallets), 'status': 'ok'}

    # Selection: sign-consistent IC on train AND val, |IC| >= IC_MIN, presence >= OS_PRESENCE_MIN
    for _os_name, _info in oversell_sets_info.items():
        if _info['status'] != 'ok':
            continue
        for _var in ('own', 'opp', 'total'):
            _col = f'sig_os_{_var}_{_os_name}'
            if _col not in c_train.columns:
                continue
            _pres = float((c_train[_col] > 0).mean())
            _ics = {lbl: compute_event_ic(df_c[_col].fillna(0.0), df_c['copyable_roi'])
                    for lbl, df_c in [('train', c_train), ('val', c_val), ('test', c_test)]}
            _os_rows.append({'signal': _col, 'n_wallets': _info['n_wallets'],
                             'presence_train': _pres,
                             'IC_train': _ics['train'], 'IC_val': _ics['val'],
                             'IC_test': _ics['test']})
            if (np.isfinite(_ics['train']) and np.isfinite(_ics['val'])
                    and np.sign(_ics['train']) == np.sign(_ics['val'])
                    and abs(_ics['train']) >= IC_MIN and abs(_ics['val']) >= IC_MIN
                    and _pres >= OS_PRESENCE_MIN):
                oversell_filter_names.append(_col)

    oversell_ic_report = pd.DataFrame(_os_rows)
    if not oversell_ic_report.empty:
        oversell_ic_report['|IC_train|'] = oversell_ic_report['IC_train'].abs()
        oversell_ic_report = (oversell_ic_report
                              .sort_values('|IC_train|', ascending=False)
                              .reset_index(drop=True))
    print("Oversell sets (train-only selection):")
    for _os_name, _info in oversell_sets_info.items():
        print(f"  {_os_name}: {_info['n_wallets']} wallets ({_info['status']})")
    print("\\nOversell position signal ICs (test = diagnostics only):")
    display(oversell_ic_report.round(4))
    print(f"\\nSelected oversell signals (sign-consistent train+val, |IC|>=IC_MIN, "
          f"presence>={OS_PRESENCE_MIN:.3f}): {oversell_filter_names}")
else:
    print("OS_POS_SIG = False; oversell position signals skipped.")
"""

# ---------------------------------------------------------------------------
# Cell B: counterparty BUY group (insert after the tier-signals cell, before save)
# ---------------------------------------------------------------------------
cell_b = """\
# ===== Overseller counterparty BUY group (standalone evaluation) =====
# Wallets that profit from BUYing the other side of the primary overseller
# set's SELLs (matched per tx_hash|condition_id|outcome).
# SELECTION: train-only, sign-consistency on val; test = diagnostics.
# Reported like the tier test; does not alter the composite.

_os_primary = 'os_pnl'
_os_primary_wallets = set(wallet_vol.loc[os_set_defs[_os_primary], 'wallet'])

_os_sell = df_full[df_full['wallet'].isin(_os_primary_wallets) & (df_full['side'] == 'SELL')]
_os_sell = _os_sell.assign(oskey=_os_sell['tx_hash'].astype(str) + '|'
                           + _os_sell['condition_id'].astype(str) + '|'
                           + _os_sell['outcome'].astype(str))
_oskeys = set(_os_sell['oskey'])

_cp_cols = ['wallet', 'tx_hash', 'condition_id', 'outcome', 'dt', 'side',
            'end_date_iso', 'copyable_pnl', 'copyable_notional', 'copyable_roi',
            'pnl', 'notional']
_cp_all = df_full[df_full['side'] == 'BUY'][_cp_cols].copy()
_cp_all['oskey'] = (_cp_all['tx_hash'].astype(str) + '|'
                    + _cp_all['condition_id'].astype(str) + '|'
                    + _cp_all['outcome'].astype(str))
_cp_matched = _cp_all[_cp_all['oskey'].isin(_oskeys)
                      & ~_cp_all['wallet'].isin(_os_primary_wallets)].copy()
del _cp_all, _os_sell

print(f"Primary overseller set '{_os_primary}': {len(_os_primary_wallets)} wallets, "
      f"{len(_oskeys):,} unique SELL oskey(s)")
print(f"Matched counterparty BUY trades (excl. oversellers): {len(_cp_matched):,}")

_cp_tr, _cp_va, _cp_te = split_data(_cp_matched, method='chronological')


def _cp_stats(f):
    _cnot = float(f['copyable_notional'].sum())
    _cpnl = float(f['copyable_pnl'].sum())
    return {'n_trades': int(len(f)),
            'n_wallets': int(f['wallet'].nunique()),
            'copyable_pnl': _cpnl,
            'copyable_notional': _cnot,
            'copyable_roi': _cpnl / _cnot if _cnot > 0 else 0.0,
            'total_pnl': float(f['pnl'].sum()),
            'notional': float(f['notional'].sum())}


counterparty_all = {l: _cp_stats(f) for l, f in
                    [('train', _cp_tr), ('val', _cp_va), ('test', _cp_te)]}

# Per-wallet TRAIN profitability; sign-consistency on VAL
_cpw = _cp_tr.groupby('wallet').agg(n_trades=('copyable_roi', 'size'),
                                    copyable_pnl=('copyable_pnl', 'sum'),
                                    copyable_notional=('copyable_notional', 'sum'))
_cpw['copyable_roi'] = _cpw['copyable_pnl'] / _cpw['copyable_notional']
_cpw = _cpw[(_cpw['n_trades'] >= 10) & (_cpw['copyable_roi'] > 0)].copy()

_cpvw = _cp_va.groupby('wallet').agg(n_trades=('copyable_roi', 'size'),
                                     copyable_pnl=('copyable_pnl', 'sum'),
                                     copyable_notional=('copyable_notional', 'sum'))
_cpvw['copyable_roi'] = _cpvw['copyable_pnl'] / _cpvw['copyable_notional']
_cpvw = _cpvw[(_cpvw['n_trades'] >= 10) & (_cpvw['copyable_roi'] > 0)]

_cp_group_wallets = sorted(set(_cpw.index) & set(_cpvw.index))

counterparty_group = {l: _cp_stats(f[f['wallet'].isin(_cp_group_wallets)])
                      for l, f in [('train', _cp_tr), ('val', _cp_va), ('test', _cp_te)]}

# Baseline: ALL BUY trades of the copy universe per split
_copy_all_stats = {}
for _l, _f in [('train', c_train), ('val', c_val), ('test', c_test)]:
    _cnot = float(_f['copyable_notional'].sum())
    _cpnl = float(_f['copyable_pnl'].sum())
    _copy_all_stats[_l] = {'n_trades': int(len(_f)),
                           'copyable_pnl': _cpnl,
                           'copyable_notional': _cnot,
                           'copyable_roi': _cpnl / _cnot if _cnot > 0 else 0.0}

print(f"\\nCounterparty group: {len(_cp_group_wallets)} wallets "
      f"(train copyable_roi > 0, n >= 10; val sign-consistent)")
print("\\n=== Copyable ROI by split (BUY fills matched to overseller SELLs) ===")
_comp_tab = pd.DataFrame({
    'all_counterparties': {l: counterparty_all[l]['copyable_roi'] for l in ('train', 'val', 'test')},
    'group': {l: counterparty_group[l]['copyable_roi'] for l in ('train', 'val', 'test')},
    'copy_universe_all': {l: _copy_all_stats[l]['copyable_roi'] for l in ('train', 'val', 'test')},
})
display(_comp_tab.round(4))
print("Matched BUY counts (all counterparties / group):",
      {l: (counterparty_all[l]['n_trades'], counterparty_group[l]['n_trades'])
       for l in ('train', 'val', 'test')})

# Top-20 counterparty wallets by TRAIN copyable_pnl (val/test diagnostic)
_cptw = _cp_te.groupby('wallet').agg(copyable_pnl_test=('copyable_pnl', 'sum'),
                                     copyable_notional_test=('copyable_notional', 'sum'))
_cptw['copyable_roi_test'] = _cptw['copyable_pnl_test'] / _cptw['copyable_notional_test']
_cp_top = _cpw.copy()
_cp_top['copyable_roi_val'] = _cpvw['copyable_roi']
_cp_top['copyable_roi_test'] = _cptw['copyable_roi_test']
_cp_top = _cp_top.sort_values('copyable_pnl', ascending=False).head(20)
print("\\n=== Top-20 counterparty wallets by TRAIN copyable_pnl (val/test diagnostic) ===")
display(_cp_top.round(4))

oversell_result = {
    'sets': {k: v for k, v in oversell_sets_info.items()},
    'signal_selection': {'IC_MIN': float(IC_MIN),
                         'presence_min': float(OS_PRESENCE_MIN),
                         'selected': oversell_filter_names},
    'signal_ics': (oversell_ic_report.round(6).to_dict(orient='records')
                   if not oversell_ic_report.empty else []),
    'counterparty': {
        'primary_set': _os_primary,
        'n_os_wallets': int(len(_os_primary_wallets)),
        'n_os_sell_oskeys': int(len(_oskeys)),
        'n_matched_buys': int(len(_cp_matched)),
        'all_counterparties': counterparty_all,
        'group': {'n_wallets': len(_cp_group_wallets),
                  'wallets': _cp_group_wallets,
                  'results': counterparty_group},
        'top_wallets': _cp_top.round(6).to_dict(orient='index'),
        'copy_universe_all_buys': _copy_all_stats,
    },
}
"""

# ---------------------------------------------------------------------------
# Locate anchors
# ---------------------------------------------------------------------------
idx_vwap = next(i for i, c in enumerate(cells)
                if c['cell_type'] == 'code' and '# Signal 3: VWAP deviation' in ''.join(c['source']))
idx_tier = next(i for i, c in enumerate(cells)
                if c['cell_type'] == 'code' and 'Signal behavior per tier + elite-3 individually' in ''.join(c['source']))
idx_quality = next(i for i, c in enumerate(cells)
                   if c['cell_type'] == 'code' and 'Signal quality report on validation set' in ''.join(c['source']))
idx_save = next(i for i, c in enumerate(cells)
                if c['cell_type'] == 'code' and '# Persist results' in ''.join(c['source']))
print(f"anchors: vwap={idx_vwap} quality={idx_quality} tier={idx_tier} save={idx_save}")
assert idx_vwap < idx_quality < idx_tier < idx_save

# ---------------------------------------------------------------------------
# Insert Cell A after VWAP cell, Cell B after tier cell (insert before save)
# ---------------------------------------------------------------------------
def make_code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

# insert Cell A after idx_vwap (everything else shifts +1)
cells.insert(idx_vwap + 1, make_code(cell_a))
# insert Cell B before the save cell (save now at idx_save + 2)
cells.insert(idx_save + 1, make_code(cell_b))

# ---------------------------------------------------------------------------
# Patch cell 38 (signal_cols) to include oversell_filter_names
# quality cell now at idx_quality + 1
# ---------------------------------------------------------------------------
q_src = ''.join(cells[idx_quality + 1]['source'])
assert '+ list(selected_filter_names)' in q_src, q_src
new_q = q_src.replace('] + list(selected_filter_names)',
                      '] + list(selected_filter_names) + list(oversell_filter_names)')
cells[idx_quality + 1]['source'] = new_q

# ---------------------------------------------------------------------------
# Patch save cell to persist the oversell section
# save cell now at idx_save + 2
# ---------------------------------------------------------------------------
s_src = ''.join(cells[idx_save + 2]['source'])
assert '"wallet_tiers": {' in s_src
new_s = s_src.replace('    "wallet_tiers": {', '    "oversell": oversell_result,\n    "wallet_tiers": {', 1)
cells[idx_save + 2]['source'] = new_s

json.dump(nb, open(NB, 'w'), indent=1, ensure_ascii=False)
print("saved", NB)
print("new cell count:", len(cells))
