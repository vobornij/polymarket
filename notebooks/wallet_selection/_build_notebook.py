"""
Build the expanded stage1_experimental notebook.
Keeps existing cells 0-20, appends signal framework cells.
"""
import json
import uuid

def cid():
    return uuid.uuid4().hex[:12]

def md(source):
    return {
        "cell_type": "markdown",
        "id": cid(),
        "metadata": {},
        "source": source,
    }

def code(source, exec_count=None):
    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "id": cid(),
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# === Load the existing notebook ===
with open("/Users/vobornij/projects/polymarket/notebooks/wallet_selection/stage1_experimental.ipynb") as f:
    nb = json.load(f)

# Keep first 21 cells (0-20): title through re-split after bad_leader
old_cells = nb["cells"][:21]

# === New cells ===

CELL_SIGNAL_QUALITY_FUNCS = '''
# Signal quality: IC, IR, bootstrap, overlap, combination

import numpy as np
import pandas as pd


def _rankdata(v):
    """Fractional ranking (scipy.stats.rankdata, method='average')."""
    n = len(v)
    sorter = np.argsort(v, kind="mergesort")
    ordinal = np.empty(n, dtype=np.intp)
    ordinal[sorter] = np.arange(n)
    inv = np.argsort(sorter, kind="mergesort")
    rank = ordinal + 1.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and v[sorter[j]] == v[sorter[i]]:
            j += 1
        if j > i + 1:
            avg_rank = (i + j + 1) / 2.0
            for k in range(i, j):
                rank[sorter[k]] = avg_rank
        i = j
    return rank[inv]


def spearman_rho(x, y):
    """Spearman rank correlation (numpy-only)."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum()
    if n < 10:
        return np.nan
    rx = _rankdata(x[mask].values if hasattr(x, 'values') else x[mask])
    ry = _rankdata(y[mask].values if hasattr(y, 'values') else y[mask])
    rx_m = rx.mean()
    ry_m = ry.mean()
    num = np.sum((rx - rx_m) * (ry - ry_m))
    den = np.sqrt(np.sum((rx - rx_m)**2) * np.sum((ry - ry_m)**2))
    return num / den if den != 0 else np.nan


def compute_event_ic(signal, forward_roi):
    """IC: rank correlation between signal and forward copyable ROI."""
    return spearman_rho(signal, forward_roi)


def compute_event_ir(signal, forward_roi, timestamps, freq="D"):
    """IR = mean(IC_chunk) / std(IC_chunk) across time chunks.
    
    Higher IR means predictive power is consistent (Grinold & Kahn Ch. 7).
    """
    ts = timestamps
    chunks = pd.Series(index=pd.DatetimeIndex(ts), data=np.arange(len(signal))).groupby(
        pd.Grouper(freq=freq)
    )
    ics = []
    for _, idx in chunks:
        if len(idx) < 5:
            continue
        rho = compute_event_ic(signal.iloc[idx], forward_roi.iloc[idx])
        if not np.isnan(rho):
            ics.append(rho)
    if len(ics) < 3:
        return np.nan
    arr = np.array(ics)
    return float(arr.mean() / arr.std(ddof=1)) if arr.std(ddof=1) > 0 else np.nan


def bootstrap_ic(signal, forward_roi, n_iter=10_000, alpha=0.05, seed=42):
    """Bootstrap CI for IC (Efron & Tibshirani 1993).
    
    Returns (mean_ic, ci_lower, ci_upper).
    """
    mask = signal.notna() & forward_roi.notna()
    s = signal[mask].values
    p = forward_roi[mask].values
    n = len(s)
    if n < 10:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot_ics = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        boot_ics[i] = spearman_rho(s[idx], p[idx])
    mean_ic = float(np.nanmean(boot_ics))
    ci_lo = float(np.nanpercentile(boot_ics, 100 * alpha / 2))
    ci_hi = float(np.nanpercentile(boot_ics, 100 * (1 - alpha / 2)))
    return mean_ic, ci_lo, ci_hi


def hit_rate(signal, forward_roi):
    """Fraction of events where signal sign matches PnL sign."""
    mask = signal.notna() & forward_roi.notna() & (forward_roi != 0)
    if mask.sum() < 10:
        return np.nan
    sgn_sig = np.sign(signal[mask])
    sgn_pnl = np.sign(forward_roi[mask])
    return float((sgn_sig == sgn_pnl).mean())


def signal_quality_report(signals_df, signal_cols, roi_col="copyable_roi",
                           dt_col="dt", ir_freq="D", n_bootstrap=5_000):
    """Compute IC, IR, hit rate, and bootstrap CI for each signal.
    
    Returns DataFrame with one row per signal.
    """
    rows = []
    for col in signal_cols:
        ic = compute_event_ic(signals_df[col], signals_df[roi_col])
        ir = compute_event_ir(signals_df[col], signals_df[roi_col],
                               signals_df[dt_col], freq=ir_freq)
        hr = hit_rate(signals_df[col], signals_df[roi_col])
        m, clo, chi = bootstrap_ic(signals_df[col], signals_df[roi_col],
                                    n_iter=n_bootstrap)
        rows.append({
            "signal": col,
            "IC": ic,
            "IR": ir,
            "hit_rate": hr,
            "bootstrap_mean_ic": m,
            "bootstrap_ci_lo": clo,
            "bootstrap_ci_hi": chi,
            "n_events": int(signals_df[col].notna().sum()),
        })
    return pd.DataFrame(rows).sort_values("IC", ascending=False, key=abs)


# === Signal overlap ===

def coincidence_rate(s1, s2):
    """Jaccard-like coincidence: P(both non-zero | either non-zero)."""
    both = ((s1.notna() & (s1 != 0)) & (s2.notna() & (s2 != 0))).sum()
    either = ((s1.notna() & (s1 != 0)) | (s2.notna() & (s2 != 0))).sum()
    return both / either if either > 0 else 0.0


def ic_correlation_matrix(signals_df, signal_cols, roi_col="copyable_roi"):
    """Pairwise IC of signal values on overlapping events."""
    n = len(signal_cols)
    mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
                continue
            both = signals_df[signal_cols[i]].notna() & signals_df[signal_cols[j]].notna()
            if both.sum() < 10:
                continue
            mat[i, j] = compute_event_ic(
                signals_df.loc[both, signal_cols[i]],
                signals_df.loc[both, signal_cols[j]],
            )
    return pd.DataFrame(mat, index=signal_cols, columns=signal_cols)


# === Signal combination ===

def compute_optimal_weights(
    signals_df, signal_cols, roi_col="copyable_roi",
    shrinkage=0.5,
):
    """Markowitz-optimal signal weights with shrinkage (Grinold & Kahn Ch. 13).
    
    w = (1-lambda) * inv(Sigma) * IC + lambda * (1/n)
    
    Parameters
    ----------
    shrinkage : float
        0 = full Markowitz, 1 = equal weight.
    
    Returns
    -------
    pd.Series of weights indexed by signal_cols.
    """
    n = len(signal_cols)
    ic_vec = np.array([
        compute_event_ic(signals_df[c], signals_df[roi_col]) or 0.0
        for c in signal_cols
    ])
    
    valid = signals_df[signal_cols].notna().all(axis=1)
    if valid.sum() < 10 or n <= 1:
        return pd.Series(np.ones(n) / n, index=signal_cols)
    
    sig_vals = signals_df.loc[valid, signal_cols].values
    cov = np.cov(sig_vals, rowvar=False)
    avg_var = np.trace(cov) / n
    shrunk_cov = (1 - shrinkage) * cov + shrinkage * np.eye(n) * avg_var
    
    try:
        inv_cov = np.linalg.solve(shrunk_cov, np.eye(n))
        w = inv_cov @ ic_vec
        w_abs_sum = np.sum(np.abs(w))
        if w_abs_sum > 1e-12:
            w = w / w_abs_sum
        else:
            w = np.ones(n) / n
    except np.linalg.LinAlgError:
        w = np.ones(n) / n
    
    return pd.Series(w, index=signal_cols)


def apply_composite_score(signals_df, signal_cols, weights):
    """Composite signal = sum w_i * signal_i."""
    result = np.zeros(len(signals_df))
    for col in signal_cols:
        result += weights[col] * signals_df[col].fillna(0.0).values
    return pd.Series(result, index=signals_df.index)


def cs_rank(s, grouper=None):
    """Cross-sectional rank transform. Maps values to [-1, 1] within groups.
    
    If grouper is provided, ranks within each group independently.
    Standard Grinold & Kahn normalization.
    """
    if grouper is not None:
        result = s.groupby(grouper, sort=False).transform(
            lambda x: 2.0 * (_rankdata(x.values) - 1.0) / max(len(x) - 1, 1) - 1.0
        )
    else:
        n = len(s)
        r = _rankdata(s.values) if hasattr(s, 'values') else _rankdata(np.asarray(s))
        result = 2.0 * (r - 1.0) / max(n - 1, 1) - 1.0
    return result
'''


CELL_PARAMS = '''
# Test mode
TEST_MODE = False
MAX_CANDIDATE_WALLETS = 20
MAX_CANDIDATE_TRADES = 5000

# Signal windows (minutes)
BAD_LEADER_WINDOW = 5
QUALITY_WALLET_WINDOW = 15
VWAP_WINDOW = 15

# Quality wallet definition
QW_MIN_BUY_ROI = 0.05
QW_MIN_BUCKETS = 50
QW_MIN_TRADE_COUNT = 5000
QW_MAX_DD_TO_PNL = 0.3
QW_MIN_COPYABLE_ROI = 0.02

# Copy universe
COPY_MIN_BUY_ROI = 0.03
COPY_MIN_BUCKETS = 20
COPY_MIN_MARKETS = 15
COPY_MIN_TRADE_COUNT = 5000
COPY_MAX_DD_TO_PNL = 0.2
COPY_MIN_COPYABLE_ROI = 0.05

print(f"TEST_MODE: {TEST_MODE}")
print(f"Signals: bad_leader ({BAD_LEADER_WINDOW}min), "
      f"quality_wallet ({QUALITY_WALLET_WINDOW}min), "
      f"vwap_deviation ({VWAP_WINDOW}min)")
'''


CELL_COPY_UNIVERSE = '''
# Copy universe: wallets that pass quality + stability filter
copy_mask = (
    (wallet_vol['buy_roi'] >= COPY_MIN_BUY_ROI)
    & (wallet_vol['num_buckets'] >= COPY_MIN_BUCKETS)
    & (wallet_vol['num_markets'] >= COPY_MIN_MARKETS)
    & (wallet_vol['trade_count'] >= COPY_MIN_TRADE_COUNT)
    & (wallet_vol['max_drawdown_to_pnl'].fillna(1.0) <= COPY_MAX_DD_TO_PNL)
    & (wallet_vol['copyable_roi'].fillna(0.0) >= COPY_MIN_COPYABLE_ROI)
)
copy_wallets = set(wallet_vol.loc[copy_mask, 'wallet'])
print(f"Copy universe: {len(copy_wallets)} wallets")

if TEST_MODE and len(copy_wallets) > MAX_CANDIDATE_WALLETS:
    copy_wallets = set(
        wallet_vol.loc[copy_mask].sort_values('buy_roi', ascending=False)
        .head(MAX_CANDIDATE_WALLETS)['wallet']
    )
    print(f"  (test mode: {len(copy_wallets)} wallets)")

# BUY trades by copy-universe wallets
candidate_mask = df_full['wallet'].isin(copy_wallets) & (df_full['side'] == 'BUY')
candidate_trades = df_full[candidate_mask].copy()
print(f"Candidate BUY trades: {len(candidate_trades):,}")

if TEST_MODE and len(candidate_trades) > MAX_CANDIDATE_TRADES:
    candidate_trades = candidate_trades.sample(MAX_CANDIDATE_TRADES, random_state=42)
    print(f"  (test mode: sampled {len(candidate_trades)})")

c_train = candidate_trades[candidate_trades['dt'] < train_cutoff].copy()
c_val = candidate_trades[
    (candidate_trades['dt'] >= train_cutoff) & (candidate_trades['dt'] < val_cutoff)
].copy()
c_test = candidate_trades[candidate_trades['dt'] >= val_cutoff].copy()
print(f"  Train: {len(c_train):,}  Val: {len(c_val):,}  Test: {len(c_test):,}")
'''


CELL_SIG_BAD_LEADER = '''
# Signal 1: bad_leader_buy (binary)
# Already computed in earlier cells (bad_leader_wallet column on df_full)

c_train['sig_bad_leader'] = c_train['bad_leader_wallet'].notna().astype(float)
c_val['sig_bad_leader'] = c_val['bad_leader_wallet'].notna().astype(float)
c_test['sig_bad_leader'] = c_test['bad_leader_wallet'].notna().astype(float)

print("Signal 1: bad_leader_buy (binary)")
for label, df_c in [("Train", c_train), ("Val", c_val), ("Test", c_test)]:
    rate = df_c['sig_bad_leader'].mean()
    ic_v = compute_event_ic(df_c['sig_bad_leader'], df_c['copyable_roi'])
    print(f"  {label}: firing_rate={rate:.4f}  IC={ic_v:.4f}")
'''


CELL_SIG_QUALITY_WALLET = '''
# Signal 2: quality_wallet_proximity

# Define quality wallets from training-period metrics
qw_mask = (
    (wallet_vol['buy_roi'] >= QW_MIN_BUY_ROI)
    & (wallet_vol['num_buckets'] >= QW_MIN_BUCKETS)
    & (wallet_vol['trade_count'] >= QW_MIN_TRADE_COUNT)
    & (wallet_vol['max_drawdown_to_pnl'].fillna(1.0) <= QW_MAX_DD_TO_PNL)
    & (wallet_vol['copyable_roi'].fillna(0.0) >= QW_MIN_COPYABLE_ROI)
)
quality_wallets = set(wallet_vol.loc[qw_mask, 'wallet'])
print(f"Quality wallets: {len(quality_wallets)}")

# merge_asof: for each trade, find nearest quality-wallet buy on same
# (condition_id, outcome) in last QUALITY_WALLET_WINDOW minutes
# NOTE: old candidate slices (c_train/c_val/c_test) do NOT have this column yet;
# we will re-slice them from df_full in the next cell.
if 'qw_wallet' not in df_full.columns:
    qw_buys = df_full[
        df_full['wallet'].isin(quality_wallets) & (df_full['side'] == 'BUY')
    ].copy()
    qw_buys = qw_buys.rename(columns={
        'dt': 'dt_qw', 'wallet': 'qw_wallet', 'usdc_amount': 'qw_usdc'
    })[['dt_qw', 'qw_wallet', 'qw_usdc', 'condition_id', 'outcome']].sort_values('dt_qw')
    print(f"  Quality wallet BUY trades: {len(qw_buys):,}")

    df_full = pd.merge_asof(
        df_full.sort_values('dt'),
        qw_buys,
        left_on='dt', right_on='dt_qw',
        by=['condition_id', 'outcome'],
        direction='backward',
        tolerance=pd.Timedelta(minutes=QUALITY_WALLET_WINDOW),
        allow_exact_matches=False,
        suffixes=('', '_qw'),
    )
    print("  Signal 2 column 'qw_wallet' added to df_full")
else:
    print("  Signal 2 already computed, skipping merge_asof")

# Window stats: quality-wallet trading activity per (condition_id, outcome, time window)
qw_buys_for_stats = df_full[
    df_full['wallet'].isin(quality_wallets) & (df_full['side'] == 'BUY')
].copy()
qw_buys_for_stats['dt_window'] = qw_buys_for_stats['dt'].dt.floor('15min')

qw_roi_map = wallet_vol.set_index('wallet')['copyable_roi'].to_dict()
qw_buys_for_stats['wallet_roi'] = qw_buys_for_stats['wallet'].map(qw_roi_map)

qw_window_stats = qw_buys_for_stats.groupby(
    ['condition_id', 'outcome', 'dt_window'], sort=False, observed=True
).agg(
    qw_trade_count=('wallet', 'size'),
    qw_unique_wallets=('wallet', 'nunique'),
    qw_total_volume=('usdc_amount', 'sum'),
    qw_roi_sum=('wallet_roi', 'sum'),
).reset_index()

df_full['dt_window'] = df_full['dt'].dt.floor('15min')
df_full = df_full.merge(
    qw_window_stats,
    on=['condition_id', 'outcome', 'dt_window'],
    how='left',
)
print(f"  Window stats added: {len(qw_window_stats)} (condition_id, outcome, 15min) windows")
print("  New columns: qw_trade_count, qw_unique_wallets, qw_total_volume, qw_roi_sum")

# Candidate sets will be refreshed from df_full in the next cell,
# at which point they will have qw_wallet + qw_usdc + window stats columns.
print("  (candidate trades will be re-sliced next)")
'''

CELL_SIG_VWAP = '''
# Signal 3: VWAP deviation (simplified for TEST_MODE)

VWAP_BUCKET_MINUTES = 5

if TEST_MODE:
    print("TEST_MODE: bucketed VWAP approximation")

    def floor_dt(s, freq=f"{VWAP_BUCKET_MINUTES}min"):
        return s.dt.floor(freq)

    # All BUY trades by copy-universe wallets
    copy_buys = df_full[
        df_full['wallet'].isin(copy_wallets) & (df_full['side'] == 'BUY')
    ].copy()
    copy_buys['dt_bucket'] = floor_dt(copy_buys['dt'])

    # Per (condition_id, outcome, bucket): VWAP
    bucket_vwap = copy_buys.groupby(
        ['condition_id', 'outcome', 'dt_bucket'], sort=False
    ).apply(
        lambda g: pd.Series({
            'vwap': (g['price'] * g['quantity']).sum() / g['quantity'].sum(),
            'vwap_vol': g['usdc_amount'].sum(),
        }), include_groups=False
    ).reset_index()

    # Shift VWAP one bucket forward to avoid look-ahead
    bucket_vwap['dt_bucket_prev'] = bucket_vwap['dt_bucket'] - pd.Timedelta(minutes=VWAP_BUCKET_MINUTES)

    # For each candidate trade, merge on previous bucket
    _vwap_dfs = []
    for name, df_c in [('train', c_train), ('val', c_val), ('test', c_test)]:
        df_c['dt_bucket'] = floor_dt(df_c['dt'])
        df_c = df_c.merge(
            bucket_vwap,
            left_on=['condition_id', 'outcome', 'dt_bucket'],
            right_on=['condition_id', 'outcome', 'dt_bucket_prev'],
            how='left',
            suffixes=('', '_vwap'),
        )
        df_c['sig_vwap_dev'] = np.where(
            df_c['vwap'].notna() & (df_c['vwap'] > 0),
            (df_c['price'] / df_c['vwap']) - 1.0,
            np.nan,
        )
        # Signed: negative z-score = buying below VWAP = good entry
        df_c['sig_vwap_signed'] = np.where(
            df_c['sig_vwap_dev'].notna(),
            -df_c['sig_vwap_dev'],
            np.nan,
        )
        # Also keep magnitude for comparison
        df_c['sig_vwap_strength'] = df_c['sig_vwap_dev'].abs().fillna(0.0)
        _vwap_dfs.append((name, df_c))
    for name, df_c in _vwap_dfs:
        if name == 'train': c_train = df_c
        elif name == 'val': c_val = df_c
        elif name == 'test': c_test = df_c

else:
    print("Non-TEST_MODE: vectorized bucketed VWAP (previous bucket)")

    copy_buys = df_full[
        df_full['wallet'].isin(copy_wallets) & (df_full['side'] == 'BUY')
    ].copy()
    copy_buys['dt_bucket'] = copy_buys['dt'].dt.floor(f"{VWAP_BUCKET_MINUTES}min")

    copy_buys['price_vol'] = copy_buys['price'] * copy_buys['quantity']
    bucket_vwap = copy_buys.groupby(
        ['condition_id', 'outcome', 'dt_bucket'], sort=False, observed=True
    ).agg(
        vwap_price=('price_vol', 'sum'),
        total_qty=('quantity', 'sum'),
        vwap_vol=('usdc_amount', 'sum'),
    ).reset_index()
    bucket_vwap['vwap'] = bucket_vwap['vwap_price'] / bucket_vwap['total_qty']

    bucket_vwap['dt_bucket_prev'] = bucket_vwap['dt_bucket'] - pd.Timedelta(minutes=VWAP_BUCKET_MINUTES)

    _vwap_dfs = []
    for name, df_c in [('train', c_train), ('val', c_val), ('test', c_test)]:
        df_c['dt_bucket'] = df_c['dt'].dt.floor(f"{VWAP_BUCKET_MINUTES}min")
        df_c = df_c.merge(
            bucket_vwap[['condition_id', 'outcome', 'dt_bucket_prev', 'vwap', 'vwap_vol']],
            left_on=['condition_id', 'outcome', 'dt_bucket'],
            right_on=['condition_id', 'outcome', 'dt_bucket_prev'],
            how='left',
            suffixes=('', '_vwap'),
        )
        df_c['sig_vwap_dev'] = np.where(
            df_c['vwap'].notna() & (df_c['vwap'] > 0),
            (df_c['price'] / df_c['vwap']) - 1.0,
            np.nan,
        )
        df_c['sig_vwap_signed'] = np.where(
            df_c['sig_vwap_dev'].notna(),
            -df_c['sig_vwap_dev'],
            np.nan,
        )
        df_c['sig_vwap_strength'] = df_c['sig_vwap_dev'].abs().fillna(0.0)
        _vwap_dfs.append((name, df_c))
    for name, df_c in _vwap_dfs:
        if name == 'train': c_train = df_c
        elif name == 'val': c_val = df_c
        elif name == 'test': c_test = df_c

# CS-rank VWAP signed deviation for comparability across days
for df_c in [c_train, c_val, c_test]:
    df_c['sig_vwap_csrank'] = cs_rank(df_c['sig_vwap_signed'].fillna(0.0), df_c['dt'].dt.date)

print()
for label, df_c in [("Train", c_train), ("Val", c_val), ("Test", c_test)]:
    cov = df_c['sig_vwap_dev'].notna().mean()
    ic_signed = compute_event_ic(df_c['sig_vwap_signed'], df_c['copyable_roi'])
    ic_strength = compute_event_ic(df_c['sig_vwap_strength'], df_c['copyable_roi'])
    ic_csrank = compute_event_ic(df_c['sig_vwap_csrank'], df_c['copyable_roi'])
    print(f"  {label}: coverage={cov:.3f}  IC(signed)={ic_signed:.4f}  IC(strength)={ic_strength:.4f}  IC(csrank)={ic_csrank:.4f}")
'''


CELL_QUALITY_REPORT = '''
# Signal quality report on validation set
signal_cols = ['sig_bad_leader', 'sig_qw_any',
               'sig_qw_consensus', 'sig_qw_freq', 'sig_qw_reputation',
               'sig_vwap_signed', 'sig_vwap_strength', 'sig_vwap_csrank']
active_cols = [c for c in signal_cols if c in c_val.columns and c_val[c].notna().sum() > 10]

quality_report = signal_quality_report(
    c_val, active_cols,
    roi_col='copyable_roi', dt_col='dt',
    ir_freq='D', n_bootstrap=5_000,
)

print("Signal quality report (validation set):")
display(quality_report.round(4))
'''


CELL_OVERLAP = '''
# Overlap analysis
active_cols = [c for c in signal_cols if c in c_val.columns and c_val[c].notna().sum() > 10]
n_sig = len(active_cols)

if n_sig < 2:
    print("Need at least 2 active signals for overlap analysis")
else:
    # 1. Coincidence rate
    print("1. Coincidence rate (P(both fire | either fires)):")
    coin_mat = np.full((n_sig, n_sig), np.nan)
    for i, s1 in enumerate(active_cols):
        for j, s2 in enumerate(active_cols):
            coin_mat[i, j] = 1.0 if i == j else coincidence_rate(c_val[s1], c_val[s2])
    coin_df = pd.DataFrame(coin_mat, index=active_cols, columns=active_cols)
    display(coin_df.round(3))

    # 2. IC correlation
    print("\\n2. IC correlation (signal value correlation):")
    ic_corr = ic_correlation_matrix(c_val, active_cols)
    display(ic_corr.round(3))

    # 3. Conditional IC
    print("\\n3. Conditional IC (unique contribution):")
    for s in active_cols:
        other = [c for c in active_cols if c != s]
        neutral = np.ones(len(c_val), dtype=bool)
        for o in other:
            neutral &= (c_val[o].abs() < 0.01) | c_val[o].isna()
        if neutral.sum() < 20:
            continue
        ic_cond = compute_event_ic(c_val.loc[neutral, s], c_val.loc[neutral, 'copyable_roi'])
        ic_full = compute_event_ic(c_val[s], c_val['copyable_roi'])
        print(f"    {s:25s}: full_IC={ic_full:.4f}  conditional_IC={ic_cond:.4f}  "
              f"(n={neutral.sum()})")
'''


CELL_COMBINATION = '''
# Signal combination methods
active_cols = [c for c in signal_cols if c in c_val.columns and c_val[c].notna().sum() > 10]

if not active_cols:
    print("No active signals found")
else:
    print(f"Combining {len(active_cols)} signals: {active_cols}")

    # 1. Equal weight
    w_equal = pd.Series(1.0 / len(active_cols), index=active_cols)

    # 2. IC weight
    ic_vals = {c: compute_event_ic(c_val[c], c_val['copyable_roi']) or 0.0
               for c in active_cols}
    ic_sum = sum(abs(v) for v in ic_vals.values())
    w_ic = pd.Series({c: ic_vals[c] / ic_sum if ic_sum > 0 else 1.0/len(active_cols)
                       for c in active_cols})

    # 3. Shrinkage Markowitz
    w_shrink = compute_optimal_weights(c_val, active_cols, 'copyable_roi', shrinkage=0.5)

    schemes = {
        'equal': w_equal,
        'ic_weighted': w_ic,
        'shrinkage_markowitz': w_shrink,
    }

    for name, w in schemes.items():
        print(f"\\n  {name}:")
        for c, wt in w.items():
            print(f"    {c:25s} = {wt:.4f}")

    # Apply composite scores
    for name, w in schemes.items():
        for df_c in [c_train, c_val, c_test]:
            df_c[f'composite_{name}'] = apply_composite_score(df_c, active_cols, w)

    # Compare on validation
    comp_cols = [f'composite_{k}' for k in schemes]
    comp_results = []
    for cc in comp_cols:
        ic_c = compute_event_ic(c_val[cc], c_val['copyable_roi'])
        ir_c = compute_event_ir(c_val[cc], c_val['copyable_roi'], c_val['dt'], freq='D')
        comp_results.append({'composite': cc, 'IC': ic_c, 'IR': ir_c})
    comp_df = pd.DataFrame(comp_results)
    print("\\n\\nComposite signal quality (validation):")
    display(comp_df.round(4))
'''


CELL_STRATEGY = '''
# Strategy evaluation: when composite_score >= threshold, copy the trade
# Use Markowitz composite if available, fall back to IC-weighted, then equal

best_composite = 'composite_shrinkage_markowitz'
if best_composite not in c_val.columns or c_val[best_composite].notna().sum() < 10:
    best_composite = 'composite_ic_weighted'
if best_composite not in c_val.columns or c_val[best_composite].notna().sum() < 10:
    best_composite = 'composite_equal'
print(f"Using: {best_composite}")


def evaluate_strategy(df, score_col, threshold):
    fired = df[df[score_col] >= threshold].copy()
    if fired.empty:
        return {
            'threshold': threshold, 'trades': 0,
            'copyable_pnl': 0.0, 'copyable_roi': 0.0,
            'total_pnl': 0.0, 'notional': 0.0,
            'copyable_notional': 0.0, 'firing_rate': 0.0,
        }
    cnot = fired['copyable_notional'].sum()
    return {
        'threshold': threshold,
        'trades': len(fired),
        'copyable_pnl': float(fired['copyable_pnl'].sum()),
        'copyable_roi': float(fired['copyable_pnl'].sum() / cnot) if cnot > 0 else 0.0,
        'total_pnl': float(fired['pnl'].sum()),
        'notional': float(fired['notional'].sum()),
        'copyable_notional': float(cnot),
        'firing_rate': len(fired) / len(df),
    }


# Grid search on validation
thresholds = np.arange(0.0, 1.05, 0.05)
val_results = [evaluate_strategy(c_val, best_composite, t) for t in thresholds]
val_df = pd.DataFrame(val_results)
val_df['pnl_per_trade'] = val_df['copyable_pnl'] / val_df['trades'].clip(lower=1)

print("\\nGrid search (validation): top 10 by copyable_pnl")
display(val_df.sort_values('copyable_pnl', ascending=False).head(10).round(2))

# Pick best threshold (max copyable_pnl, min 20 trades)
candidates = val_df[val_df['trades'] >= 20]
if not candidates.empty:
    best_row = candidates.sort_values('copyable_pnl', ascending=False).iloc[0]
else:
    best_row = val_df.sort_values('copyable_pnl', ascending=False).iloc[0]
best_threshold = best_row['threshold']
print(f"\\nBest threshold: {best_threshold:.2f}  "
      f"(copyable_pnl=${best_row['copyable_pnl']:,.0f}, "
      f"{best_row['trades']} trades)")
'''


CELL_TEST_EVAL = '''
# Test set evaluation
test_result = evaluate_strategy(c_test, best_composite, best_threshold)
all_result = evaluate_strategy(c_test, best_composite, -np.inf)

print("Test set evaluation:")
print(f"  Threshold: {best_threshold:.2f}")
print(f"  Trades fired: {test_result['trades']:,} / {all_result['trades']:,} "
      f"({test_result['firing_rate']:.1%})")
print(f"  Copyable PnL: ${test_result['copyable_pnl']:,.0f}")
print(f"  Copyable ROI: {test_result['copyable_roi']:.4f}")
print(f"  Total PnL: ${test_result['total_pnl']:,.0f}")
print(f"  PnL per trade: ${test_result['copyable_pnl'] / max(test_result['trades'], 1):.2f}")
print()
print("vs. copying all candidate trades:")
print(f"  Copyable PnL (all): ${all_result['copyable_pnl']:,.0f}")
print(f"  Copyable ROI (all): {all_result['copyable_roi']:.4f}")

# Summary across splits
print("\\n\\n=== Strategy Summary ===")
for label, df_i, th in [('Train', c_train, best_threshold),
                          ('Val', c_val, best_threshold),
                          ('Test', c_test, best_threshold)]:
    r = evaluate_strategy(df_i, best_composite, th)
    ra = evaluate_strategy(df_i, best_composite, -np.inf)
    print(f"  {label:6s}: threshold={th:.2f}  "
          f"trades={r['trades']:>5,}/{len(df_i):>6,}  "
          f"cpnl=${r['copyable_pnl']:>8,.0f}  "
          f"croi={r['copyable_roi']:.4f}  "
          f"(all: cpnl=${ra['copyable_pnl']:>8,.0f}  croi={ra['copyable_roi']:.4f})")
'''


CELL_LEAVE_OUT = '''
# Leave-one-out signal contribution (validation)
active_cols = [c for c in signal_cols if c in c_val.columns and c_val[c].notna().sum() > 10]

if len(active_cols) >= 2:
    print("Signal contribution (leave-one-out on validation):")
    full_ic = compute_event_ic(c_val[best_composite], c_val['copyable_roi'])

    loo_results = []
    for leave_out in active_cols:
        remaining = [c for c in active_cols if c != leave_out]
        if not remaining:
            continue
        w = compute_optimal_weights(c_val, remaining, 'copyable_roi', shrinkage=0.5)
        c_val[f'composite_loo_{leave_out}'] = apply_composite_score(c_val, remaining, w)
        ic_loo = compute_event_ic(c_val[f'composite_loo_{leave_out}'], c_val['copyable_roi'])
        loo_results.append({'left_out': leave_out, 'IC': ic_loo, 'IC_drop': full_ic - ic_loo})

    loo_df = pd.DataFrame(loo_results).sort_values('IC_drop', ascending=False)
    print(f"  Full composite IC: {full_ic:.4f}")
    display(loo_df.round(4))
'''


CELL_SAVE = '''
# Persist results
import json
from datetime import datetime, timezone
from pathlib import Path

# Signal ICs on TRAIN only (no test leakage)
signal_ics = {}
for col in active_cols:
    signal_ics[col] = {
        "IC": compute_event_ic(c_train[col], c_train['copyable_roi']),
        "IR": compute_event_ir(c_train[col], c_train['copyable_roi'], c_train['dt'], freq='D'),
        "hit_rate": hit_rate(c_train[col], c_train['copyable_roi']),
    }

output = {
    "stage": 1,
    "type": "experimental_signal_framework",
    "metadata": {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "TEST_MODE": TEST_MODE,
        "n_copy_wallets": len(copy_wallets),
        "n_quality_wallets": len(quality_wallets),
        "n_candidate_trades": len(candidate_trades),
        "best_threshold": float(best_threshold),
        "best_composite": best_composite,
        "signal_windows_min": {
            "bad_leader": BAD_LEADER_WINDOW,
            "quality_wallet": QUALITY_WALLET_WINDOW,
            "vwap": VWAP_WINDOW,
        },
    },
    "signals": {
        col: vals for col, vals in signal_ics.items()
    },
    "weights": {
        "equal": {k: float(v) for k, v in w_equal.items()},
        "ic_weighted": {k: float(v) for k, v in w_ic.items()},
        "shrinkage_markowitz": {k: float(v) for k, v in w_shrink.items()},
    },
    "val_performance": val_df.round(4).to_dict(orient="records"),
    "test_performance": {
        "threshold": best_threshold,
        "copyable_pnl": test_result["copyable_pnl"],
        "copyable_roi": test_result["copyable_roi"],
        "total_pnl": test_result["total_pnl"],
        "trades": test_result["trades"],
        "firing_rate": test_result["firing_rate"],
        "all_trades_copyable_pnl": all_result["copyable_pnl"],
        "all_trades_copyable_roi": all_result["copyable_roi"],
    },
}

out_path = Path("stage1_experimental_result.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"Saved -> {out_path.resolve()}")
'''


CELL_TODO = '''
# TODO: Next Iterations

### Short-term
- [ ] **Full VWAP** - per-trade rolling VWAP (numba-accelerated like _twopass_impl.py)
- [ ] **Aggregated quality wallet volume** - sum all quality wallet volume in window, not just nearest
- [ ] **Top buyer position signal** - track cumulative positions of top N buyers per market
- [ ] **Disagreement signal** - divergence between top buyers (some buying YES, others NO)
- [ ] **Walk-forward cross-validation** - replace single val split

### Medium-term
- [ ] **Deflated Sharpe Ratio** (Bailey et al. 2014) - correct for multiple-signal testing
- [ ] **Non-linear combination** - shallow gradient-boosted ensemble over raw signals
- [ ] **Rolling IC estimation** - re-estimate weights on expanding monthly windows
- [ ] **Calibration layers** - port price_bucket and consensus scores from signal/scorer.py
- [ ] **Execution tape integration** - feed into backtest/execution_tape.py with slippage & latency

### Long-term
- [ ] **Meta-signal from wallet groups** - use polymarket_analysis.copy_groups as signal input
- [ ] **Regime detection** - adjust signal weights per market regime
- [ ] **Full portfolio backtest** - multi-wallet, multi-market with Kelly sizing & risk limits
'''

# === Assemble new cells ===

new_cells = [

    md("## Signal Quality Framework\n\n"
       "We evaluate each signal using **Information Coefficient (IC)** and\n"
       "**Information Ratio (IR)**, following Grinold & Kahn (1999),\n"
       "*Active Portfolio Management* (McGraw-Hill).\n\n"
       "| Metric | Definition | Interpretation |\n"
       "|--------|------------|----------------|\n"
       "| **IC** | Spearman rank correlation between signal value and forward copyable ROI | Does a higher signal predict better PnL? |\n"
       "| **IR** | Mean(IC) / Std(IC) across daily chunks | How consistent is the predictive power? |\n"
       "| **Hit Rate** | % of events where signal sign matches PnL sign | Directional accuracy |\n"
       "| **Bootstrap CI** | 2.5th-97.5th percentile of IC over 10k resamples | Is IC sign reliably non-zero? |\n\n"
       "Signal overlap is measured with **coincidence rate** (do they fire together?)\n"
       "and **IC correlation** (are their predictions redundant?)."),

    code(CELL_SIGNAL_QUALITY_FUNCS),

    md("## Parameters & Test Mode"),

    code(CELL_PARAMS),

    md("## Define Copy Universe (candidate trades)\n\n"
       "Trades we *could* copy. BUY trades by wallets that pass a quality filter."),

    code(CELL_COPY_UNIVERSE),

    md("## Signal 1: Bad Leader Proximity\n\n"
       "**Hypothesis:** Trades on the same (condition_id, outcome) recently after a\n"
       '"bad leader" (profitable but uncopyable wallet) tend to underperform.\n\n'
       "Already computed via `merge_asof` above. Column: `bad_leader_wallet`."),

    code(CELL_SIG_BAD_LEADER),

    md("## Signal 2: Quality Wallet Proximity\n\n"
       "**Hypothesis:** When high-quality wallets buy a token, copying within a\n"
       "window yields positive PnL. Quality wallets have high buy_roi, diversification,\n"
       "and stable returns."),

    code(CELL_SIG_QUALITY_WALLET),

    md("## Refresh candidate trades\n\n"
       "Re-slice candidate trades from the fully-annotated `df_full` (now includes quality_wallet signal + window stats)."),

    code('''# Re-slice candidate trades after signal columns added to df_full
candidate_mask = df_full['wallet'].isin(copy_wallets) & (df_full['side'] == 'BUY')
candidate_trades = df_full[candidate_mask].copy()
print(f"Candidate BUY trades (refreshed): {len(candidate_trades):,}")

if TEST_MODE and len(candidate_trades) > MAX_CANDIDATE_TRADES:
    candidate_trades = candidate_trades.sample(MAX_CANDIDATE_TRADES, random_state=42)
    print(f"  (test mode: sampled {len(candidate_trades)})")

c_train = candidate_trades[candidate_trades['dt'] < train_cutoff].copy()
c_val = candidate_trades[
    (candidate_trades['dt'] >= train_cutoff) & (candidate_trades['dt'] < val_cutoff)
].copy()
c_test = candidate_trades[candidate_trades['dt'] >= val_cutoff].copy()
print(f"  Train: {len(c_train):,}  Val: {len(c_val):,}  Test: {len(c_test):,}")
'''),

    md("## Assign signal columns\n\n"
       "Derive signal columns from `df_full` columns on refreshed candidate sets."),

    code('''# Assign signal columns on refreshed candidate sets
# (df_full already has bad_leader_wallet, qw_wallet, and window stats columns)
for df_c in [c_train, c_val, c_test]:
    # Signal 1: bad leader proximity
    df_c['sig_bad_leader'] = df_c['bad_leader_wallet'].notna().astype(float)
    # Signal 2: quality wallet proximity
    df_c['sig_qw_any'] = df_c['qw_wallet'].notna().astype(float)
    df_c['sig_qw_volume'] = df_c['qw_usdc'].fillna(0.0)
    # Signal 3: quality wallet window stats
    df_c['sig_qw_consensus'] = df_c['qw_unique_wallets'].fillna(0.0)
    df_c['sig_qw_freq'] = df_c['qw_trade_count'].fillna(0.0)
    df_c['sig_qw_reputation'] = df_c['qw_roi_sum'].fillna(0.0)

# Print IC for each signal
for sig in ['sig_bad_leader', 'sig_qw_any', 'sig_qw_volume',
            'sig_qw_consensus', 'sig_qw_freq', 'sig_qw_reputation']:
    print(f"{sig}:")
    for label, df_c in [("Train", c_train), ("Val", c_val), ("Test", c_test)]:
        rate = df_c[sig].mean() if df_c[sig].dtype.kind in 'bif' else df_c[sig].notna().mean()
        ic_v = compute_event_ic(df_c[sig], df_c['copyable_roi'])
        print(f"  {label}: firing_rate={rate:.4f}  IC={ic_v:.4f}")
    print()
for sig in ['sig_vwap_csrank', 'sig_vwap_signed']:
    if sig in c_train.columns:
        print(f"{sig}:")
        for label, df_c in [("Train", c_train), ("Val", c_val), ("Test", c_test)]:
            ic_cr = compute_event_ic(df_c[sig], df_c['copyable_roi'])
            ic_pnl = compute_event_ic(df_c[sig], df_c['pnl'])
            print(f"  {label}: IC(copyable_roi)={ic_cr:.4f}  IC(pnl)={ic_pnl:.4f}")
        print()
'''),

    md("## Signal 3: VWAP Deviation (simplified)\n\n"
       "**Hypothesis:** Buying below the trailing VWAP of quality buyers is a better entry.\n"
       "VWAP deviation = (price / vwap_15m) - 1.\n\n"
       "For TEST_MODE: bucketed 5-min VWAP approximation."),

    code(CELL_SIG_VWAP),

    md("## Signal Quality Report\n\n"
       "Compute IC, IR, hit rate, and bootstrap confidence intervals on validation."),

    code(CELL_QUALITY_REPORT),

    md("## Signal Overlap Analysis\n\n"
       "How redundant are the signals? Do they fire on the same events?"),

    code(CELL_OVERLAP),

    md("## Signal Combination\n\n"
       "Combine signals into a composite score:\n"
       "1. **Equal weight**: w_i = 1/n\n"
       "2. **IC weight**: w_i = IC_i / sum|IC_j|\n"
       "3. **Shrinkage Markowitz**: (1-lambda)*inv(Sigma)*IC + lambda/n"),

    code(CELL_COMBINATION),

    md("## Strategy Evaluation\n\n"
       "When composite_score >= threshold, copy the BUY trade. Grid-search threshold on validation."),

    code(CELL_STRATEGY),
    code(CELL_TEST_EVAL),
    code(CELL_LEAVE_OUT),

    md("## Save Results"),

    code(CELL_SAVE),

    md("## TODO: Next Iterations"),

    md(CELL_TODO),
]

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

out_path = "/Users/vobornij/projects/polymarket/notebooks/wallet_selection/stage1_experimental.ipynb"
with open(out_path, "w") as f:
    json.dump(new_nb, f, indent=1, ensure_ascii=False)

print(f"Written: {len(new_nb['cells'])} cells total")
print(f"  {len(old_cells)} old cells kept + {len(new_cells)} new cells appended")
