"""Rewrite the oversell module cell (cell 37) with valued-position signals."""
import json

NB = 'stage1_experimental.ipynb'
nb = json.load(open(NB))

cell_src = """# ===== Oversell position + valued-position signals =====
# Wallets that are net-negative on SELLs yet profitable overall (train-only
# wallet_vol selection). Signals = the EXACT aggregate position these wallets
# hold on the candidate's token at time t (own / opposite outcome), plus the
# market total (own + opposite), plus VALUED variants that combine the price
# each wallet PAID for its open position (average-cost basis, "value-at-cost")
# with the candidate trade's own current price:
#   sig_os_*            : aggregate position (quantity)
#   sig_os_val_*        : aggregate value-at-cost (USDC) of that position
#   sig_os_uwl_*        : value-at-cost minus position x current price
#                         (positive = set is underwater on its holding)
#   sig_os_avgc_*       : value-at-cost / position / current price - 1
#                         (position-weighted entry premium vs current price)
# Exact aggregate via two-pass cumsum over post-trade checkpoints (validated
# against a per-wallet brute force):
#   A(t) = cumsum of checkpoint values at nearest checkpoint <= t
#   B(t) = cumsum of values at each wallet's NEXT checkpoint <= t
#   signal = A(t) - B(t)  (0 where the set holds nothing at that time)
# Value-at-cost uses average-cost accounting, so checkpoints are ordered by
# (wallet, condition, outcome, dt, -position) so that same-timestamp trades are
# consumed in true execution order (post-trade position as ground truth).

OS_POS_SIG = True

os_set_defs = {
    'os_pnl': (wallet_vol['sell_pnl'] < 0) & (wallet_vol['total_pnl'] > 0),
    'os_roi': (wallet_vol['sell_roi'] < 0) & (wallet_vol['total_pnl'] > 0),
    'os_active': (wallet_vol['sell_pnl'] < 0) & (wallet_vol['total_pnl'] > 0)
                 & (wallet_vol['sell_notional'] > 500),
    'os_buyprof': (wallet_vol['sell_pnl'] < 0) & (wallet_vol['total_pnl'] > 0)
                  & (wallet_vol['buy_pnl'] > 0),
    'os_deep': (wallet_vol['sell_pnl'] < 0) & (wallet_vol['total_pnl'] > 0)
               & (wallet_vol['sell_roi'] < -0.1),
    'os_thin': (wallet_vol['sell_pnl'] < 0) & (wallet_vol['total_pnl'] > 0)
               & (wallet_vol['buy_pnl'] < 50),
}

OS_PRESENCE_MIN = 0.005  # min fraction of candidates where the set holds > 0

oversell_sets_info = {}
oversell_filter_names = []
oversell_ic_report = pd.DataFrame()
_os_variants = []  # (col, kind, set_name) to evaluate

if OS_POS_SIG:
    from numba import njit

    def build_os_checkpoints(wallets):
        \"\"\"All trades (BUY+SELL) by `wallets` with post-trade position checkpoints.\"\"\"
        wallet_arr = np.asarray(list(wallets), dtype=object)
        present = np.isin(wallet_arr, _WALLET_CATEGORIES)
        codes_sub = np.searchsorted(_WALLET_CATEGORIES, wallet_arr[present])
        idx = np.flatnonzero(np.isin(_WALLET_CODES, codes_sub))
        return df_full.iloc[idx][['dt', 'wallet', 'condition_id', 'outcome',
                                  'position', 'price', 'quantity', 'side']].copy()

    @njit(nogil=True)
    def _os_vac_pass(wallet_code, key_code, is_buy, qty, price, position):
        \"\"\"Running average-cost value-at-cost per (wallet, key) checkpoint.\"\"\"
        n = len(wallet_code)
        vac = np.zeros(n, dtype=np.float64)
        w_prev = -1
        k_prev = -1
        pos_old = 0.0
        cost = 0.0
        for i in range(n):
            w = wallet_code[i]
            k = key_code[i]
            if w != w_prev or k != k_prev:
                w_prev = w
                k_prev = k
                pos_old = 0.0
                cost = 0.0
            pos_new = position[i]
            if is_buy[i]:
                cost += qty[i] * price[i]
            elif pos_old > 1e-12:
                cost *= (pos_new / pos_old)
            else:
                cost = 0.0
            pos_old = pos_new
            vac[i] = cost if pos_old > 0 else 0.0
        return vac

    def os_compute_vac(ck):
        \"\"\"Attach per-checkpoint value-at-cost (execution-order-aware).\"\"\"
        wc, _wu = pd.factorize(ck['wallet'])
        kc, _ku = pd.factorize(ck['condition_id'] + '|' + ck['outcome'])
        ck = ck.assign(_wcode=wc, _kcode=kc, _negpos=-ck['position'].to_numpy())
        ck = ck.sort_values(['_wcode', '_kcode', 'dt', '_negpos'], kind='stable')
        ck['vac'] = _os_vac_pass(ck['_wcode'].to_numpy(), ck['_kcode'].to_numpy(),
                                 (ck['side'] == 'BUY').to_numpy(),
                                 ck['quantity'].to_numpy(), ck['price'].to_numpy(),
                                 ck['position'].to_numpy())
        return ck.drop(columns=['_wcode', '_kcode', '_negpos'])

    def _os_cumsum_tables(ck, by_cols):
        \"\"\"(A, B) asof tables keyed by by_cols; cum columns for position AND vac.\"\"\"
        key = ['wallet'] + by_cols
        B = ck.sort_values(key + ['dt'], kind='stable')
        B = B.assign(next_dt=B.groupby(key, sort=False)['dt'].shift(-1))
        B = B[B['next_dt'].notna()].drop(columns='dt').rename(columns={'next_dt': 'dt'})
        A = ck.sort_values(by_cols + ['dt'], kind='stable')
        A['cum_pos'] = A.groupby(by_cols, sort=False)['position'].cumsum()
        A['cum_vac'] = A.groupby(by_cols, sort=False)['vac'].cumsum()
        A = A[by_cols + ['dt', 'cum_pos', 'cum_vac']].sort_values('dt', kind='stable')
        B = B.sort_values(by_cols + ['dt'], kind='stable')
        B['cum_pos'] = B.groupby(by_cols, sort=False)['position'].cumsum()
        B['cum_vac'] = B.groupby(by_cols, sort=False)['vac'].cumsum()
        B = B[by_cols + ['dt', 'cum_pos', 'cum_vac']].sort_values('dt', kind='stable')
        return A, B

    def os_aggregate_value(cand, A, B, by_cols):
        \"\"\"Exact aggregate position+vac of the set on cand's key at cand.dt.\"\"\"
        left = cand.sort_values('dt')[['dt'] + by_cols]
        idx = left.index
        a = pd.merge_asof(left, A, on='dt', by=by_cols, direction='backward',
                          allow_exact_matches=False)
        b = pd.merge_asof(left, B, on='dt', by=by_cols, direction='backward',
                          allow_exact_matches=False)
        out = pd.DataFrame({
            'pos': a['cum_pos'].fillna(0.0) - b['cum_pos'].fillna(0.0),
            'vac': a['cum_vac'].fillna(0.0) - b['cum_vac'].fillna(0.0),
        }, index=left.index)
        return out.sort_index()

    _os_swap = {'Yes': 'No', 'No': 'Yes'}
    _os_asof = {}  # (set_name, 'own'/'opp') -> {'tbl': (A, B)} (same tables, opp via swap)
    _os_rows = []

    def os_attach_signals(df_c, os_name):
        \"\"\"Attach all 11 oversell signal columns for one set to a candidate frame.\"\"\"
        A, B = _os_asof[(os_name, 'own')]['tbl']
        by_cols = ['condition_id', 'outcome']
        own = os_aggregate_value(df_c, A, B, by_cols)
        opp = os_aggregate_value(df_c.assign(outcome=df_c['outcome'].map(_os_swap)), A, B, by_cols)
        _p_own, _p_opp = own['pos'], opp['pos']
        _v_own, _v_opp = own['vac'], opp['vac']
        _cand_p = df_c['price']
        df_c[f'sig_os_own_{os_name}'] = _p_own
        df_c[f'sig_os_opp_{os_name}'] = _p_opp
        df_c[f'sig_os_total_{os_name}'] = _p_own + _p_opp
        df_c[f'sig_os_val_own_{os_name}'] = _v_own
        df_c[f'sig_os_val_opp_{os_name}'] = _v_opp
        df_c[f'sig_os_val_total_{os_name}'] = _v_own + _v_opp
        df_c[f'sig_os_uwl_own_{os_name}'] = _v_own - _p_own * _cand_p
        df_c[f'sig_os_uwl_opp_{os_name}'] = _v_opp - _p_opp * (1.0 - _cand_p)
        df_c[f'sig_os_uwl_total_{os_name}'] = (
            _v_own - _p_own * _cand_p) + (_v_opp - _p_opp * (1.0 - _cand_p))
        df_c[f'sig_os_avgc_own_{os_name}'] = np.where(
            _p_own > 0, _v_own / _p_own / _cand_p - 1.0, 0.0)
        df_c[f'sig_os_avgc_opp_{os_name}'] = np.where(
            _p_opp > 0, _v_opp / _p_opp / (1.0 - _cand_p) - 1.0, 0.0)

    for _os_name, _os_mask in os_set_defs.items():
        _os_wallets = set(wallet_vol.loc[_os_mask, 'wallet'])
        if len(_os_wallets) < MIN_SET_SIZE:
            oversell_sets_info[_os_name] = {'n_wallets': len(_os_wallets), 'status': 'skipped'}
            continue
        _os_ck = os_compute_vac(build_os_checkpoints(_os_wallets))
        _A, _B = _os_cumsum_tables(_os_ck, ['condition_id', 'outcome'])
        _os_asof[(_os_name, 'own')] = {'tbl': (_A, _B)}
        _os_asof[(_os_name, 'opp')] = {'tbl': (_A, _B)}
        for _df_c in [c_train, c_val, c_test]:
            os_attach_signals(_df_c, _os_name)
        for _kind, _var in [('pos', 'own'), ('pos', 'opp'), ('pos', 'total'),
                            ('val', 'own'), ('val', 'opp'), ('val', 'total'),
                            ('uwl', 'own'), ('uwl', 'opp'), ('uwl', 'total'),
                            ('avgc', 'own'), ('avgc', 'opp')]:
            _pfx = 'sig_os' if _kind == 'pos' else f'sig_os_{_kind}'
            _os_variants.append((f'{_pfx}_{_var}_{_os_name}', _kind, _os_name))
        oversell_sets_info[_os_name] = {'n_wallets': len(_os_wallets), 'status': 'ok'}

    # Selection: sign-consistent IC on train AND val, |IC| >= IC_MIN, presence >= OS_PRESENCE_MIN
    for _col, _kind, _os_name in _os_variants:
        _pres = float((c_train[_col] > 0).mean())
        _ics = {lbl: compute_event_ic(df_c[_col].fillna(0.0), df_c['copyable_roi'])
                for lbl, df_c in [('train', c_train), ('val', c_val), ('test', c_test)]}
        _os_rows.append({'signal': _col, 'kind': _kind,
                         'n_wallets': oversell_sets_info[_os_name]['n_wallets'],
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
    print("\\nOversell signal ICs by kind (test = diagnostics only):")
    display(oversell_ic_report.round(4))
    print(f"\\nSelected oversell signals (sign-consistent train+val, |IC|>=IC_MIN, "
          f"presence>={OS_PRESENCE_MIN:.3f}): {oversell_filter_names}")
else:
    print("OS_POS_SIG = False; oversell position signals skipped.")"""

# find the oversell cell and replace its source
target = None
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if c['cell_type'] == 'code' and 'Oversell position' in src:
        target = i
        break

if target is None:
    raise SystemExit("oversell cell not found")

nb['cells'][target]['source'] = cell_src.splitlines(keepends=True)
nb['cells'][target]['outputs'] = []
nb['cells'][target]['execution_count'] = None

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"replaced cell {target} in {NB}")
