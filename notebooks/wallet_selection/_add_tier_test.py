import json

PATH = 'notebooks/wallet_selection/stage1_experimental.ipynb'
nb = json.load(open(PATH))

tier_src = '''# ===== Wallet-tier robustness test =====
# Evaluate the EXISTING framework on candidate trades restricted to four
# wallet tiers: random (control), wide, careful (copy universe), elite-3.
# SELECTION: all wallet sets defined from TRAIN metrics only (no leakage).


def _parse_set_col(col):
    s = col[len('sig_set_'):]
    for var in ('buy_opp', 'sell_opp', 'buy', 'sell'):
        if s.endswith('_' + var):
            return s[:-(len(var) + 1)], var
    raise ValueError(f"cannot parse set column: {col}")


_set_wallet_lookup = {ws.name: ws.wallets for ws in _wallet_sets}


def build_tier_frames(tier_wallets):
    """Candidate BUY trades of `tier_wallets` per split, with all signals
    (5 originals + finalist set signals) recomputed for this universe."""
    _tr, _va, _te = split_data(df_full, method='chronological')
    out = {}
    for label, df_split in [('train', _tr), ('val', _va), ('test', _te)]:
        df_c = df_split[
            df_split['wallet'].isin(tier_wallets) & (df_split['side'] == 'BUY')
        ].copy()
        df_c['sig_bad_leader'] = df_c['bad_leader_wallet'].notna().astype(float)
        df_c['sig_qw_any'] = df_c['qw_wallet'].notna().astype(float)
        df_c['dt_bucket'] = df_c['dt'].dt.floor(f"{VWAP_BUCKET_MINUTES}min")
        df_c = df_c.merge(
            bucket_vwap[['condition_id', 'outcome', 'dt_bucket_prev', 'vwap', 'vwap_vol']],
            left_on=['condition_id', 'outcome', 'dt_bucket'],
            right_on=['condition_id', 'outcome', 'dt_bucket_prev'],
            how='left', suffixes=('', '_vwap'),
        )
        df_c['sig_vwap_dev'] = np.where(
            df_c['vwap'].notna() & (df_c['vwap'] > 0),
            (df_c['price'] / df_c['vwap']) - 1.0, np.nan,
        )
        df_c['sig_vwap_signed'] = np.where(
            df_c['sig_vwap_dev'].notna(), -df_c['sig_vwap_dev'], np.nan,
        )
        df_c['sig_vwap_strength'] = df_c['sig_vwap_dev'].abs().fillna(0.0)
        df_c['sig_vwap_csrank'] = cs_rank(
            df_c['sig_vwap_signed'].fillna(0.0), df_c['dt'].dt.date,
        )
        for col in selected_filter_names:
            name, var = _parse_set_col(col)
            events = build_union_events(
                _set_wallet_lookup[name],
                'BUY' if var.startswith('buy') else 'SELL',
                key=name,
            )
            df_c = merge_set_signal(df_c, events, opposite=var.endswith('_opp'), col_name=col)
            df_c[col] = df_c[col].fillna(0.0)
        out[label] = df_c
    return out


def _tune_threshold(df_val, score_col, min_trades=20):
    res = [evaluate_strategy(df_val, score_col, t) for t in np.arange(0.0, 1.05, 0.05)]
    rdf = pd.DataFrame(res)
    cand = rdf[rdf['trades'] >= min_trades]
    row = (cand if not cand.empty else rdf).sort_values('copyable_pnl', ascending=False).iloc[0]
    return float(row['threshold'])


# --- Tier 1: random control (active wallets EXCLUDING copy universe, same size) ---
_rng = np.random.default_rng(42)
_pool = [w for w in wallet_vol[wallet_vol['trade_count'] >= 100]['wallet'].tolist()
         if w not in copy_wallets]
_n_rand = min(len(copy_wallets), len(_pool))
random_control_wallets = set(_rng.choice(_pool, size=_n_rand, replace=False))

# --- Tier 2: wide (copyable_pnl >= $200 and copyable_roi >= 0.02) ---
wide_wallets = set(wallet_vol.loc[
    (wallet_vol['buy_copyable_pnl'].fillna(0.0) >= 200)
    & (wallet_vol['copyable_roi'].fillna(0.0) >= 0.02),
    'wallet'])

# --- Tier 3: careful = current copy universe ---
careful_wallets = copy_wallets

# --- Tier 4: elite 3 (stable drawdown, copyable, roi, enough copyable trades) ---
for _em in [
    ((wallet_vol['max_drawdown_to_pnl'].fillna(1.0) <= 0.2)
     & (wallet_vol['copyable_roi'].fillna(0.0) >= 0.10)
     & (wallet_vol['buy_roi'].fillna(0.0) >= 0.10)
     & (wallet_vol['trade_count'].fillna(0) >= 200)),
    ((wallet_vol['max_drawdown_to_pnl'].fillna(1.0) <= 0.3)
     & (wallet_vol['copyable_roi'].fillna(0.0) >= 0.05)
     & (wallet_vol['buy_roi'].fillna(0.0) >= 0.05)
     & (wallet_vol['trade_count'].fillna(0) >= 100)),
]:
    _elite_cand = wallet_vol[_em].sort_values('buy_copyable_pnl', ascending=False)
    if len(_elite_cand) >= 3:
        break
elite_wallets = _elite_cand.head(3)['wallet'].tolist()

for _lbl, _ws in [('random_control', random_control_wallets), ('wide', wide_wallets),
                  ('careful', careful_wallets), ('elite_3', elite_wallets)]:
    print(f"{_lbl}: {len(_ws)} wallets")

tiers = [
    ('random_control', random_control_wallets),
    ('wide', wide_wallets),
    ('careful_copy_universe', careful_wallets),
    ('elite_3', elite_wallets),
]

cols_main = [c for c in w_shrink.index if c in c_val.columns]
cols_ctrl = [c for c in w_control.index if c in c_val.columns]

tier_rows = []
for label, ws in tiers:
    frames = build_tier_frames(ws)
    tr, va, te = frames['train'], frames['val'], frames['test']
    if te.empty:
        print(f"{label}: no candidate trades, skipping")
        continue
    for df_c in frames.values():
        df_c['composite_main'] = apply_composite_score(df_c, cols_main, w_shrink)
        df_c['composite_control5'] = apply_composite_score(df_c, cols_ctrl, w_control)

    th_main = _tune_threshold(va, 'composite_main')
    th_ctrl = _tune_threshold(va, 'composite_control5')
    r_main = evaluate_strategy(te, 'composite_main', th_main)
    r_ctrl = evaluate_strategy(te, 'composite_control5', th_ctrl)
    r_all = evaluate_strategy(te, 'composite_main', -np.inf)

    tier_rows.append({
        'tier': label,
        'n_wallets': len(ws),
        'n_train': int(len(tr)), 'n_val': int(len(va)), 'n_test': int(len(te)),
        'threshold_main': th_main,
        'threshold_control': th_ctrl,
        'copyable_pnl': r_main['copyable_pnl'],
        'copyable_roi': r_main['copyable_roi'],
        'trades': r_main['trades'],
        'firing_rate': r_main['firing_rate'],
        'ctrl_copyable_pnl': r_ctrl['copyable_pnl'],
        'ctrl_copyable_roi': r_ctrl['copyable_roi'],
        'all_copyable_pnl': r_all['copyable_pnl'],
        'all_copyable_roi': r_all['copyable_roi'],
    })
    print(f"{label}: wallets={len(ws)}  train={len(tr):,}  val={len(va):,}  test={len(te):,}")

tier_df = pd.DataFrame(tier_rows)
print("\\n=== Wallet-tier robustness (test set) ===")
display(tier_df.round(4))

tier_result = tier_df.round(6).to_dict(orient="records")
tier_wallet_sets = {label: sorted(ws) for label, ws in tiers}
'''

insert_at = None
for i, c in enumerate(nb['cells']):
    if c.get('id') == 'control-cell':
        insert_at = i + 1
        break
assert insert_at is not None, 'control cell not found'

new_cell = {
    "cell_type": "code",
    "id": "wallet-tier-test",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": tier_src.splitlines(keepends=True),
}
nb['cells'].insert(insert_at, new_cell)
print(f'tier cell inserted after cell {insert_at}')

# patch save cell: add wallet_tier_results
c = None
for cell in nb['cells']:
    if cell.get('id') == 'fa9704846147':
        c = cell
        break
assert c is not None, 'save cell not found'
src = ''.join(c['source'])
old = '''        "best_variant_per_set": best_variant_per_set.head(50).round(4).to_dict(orient="records"),
    },
}'''
new = '''        "best_variant_per_set": best_variant_per_set.head(50).round(4).to_dict(orient="records"),
    },
    "wallet_tiers": {
        "wallet_sets": {k: {"n_wallets": len(v), "wallets": v} for k, v in tier_wallet_sets.items()},
        "results": tier_result,
    },
}'''
assert old in src, 'save cell tail not found'
c['source'] = src.replace(old, new).splitlines(keepends=True)
c['outputs'] = []
c['execution_count'] = None
print('save cell patched')

json.dump(nb, open(PATH, 'w'), indent=1, ensure_ascii=False)
print('notebook written')
