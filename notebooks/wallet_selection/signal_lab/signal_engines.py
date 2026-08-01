"""
Signal engines for stage1 wallet-selection.

Two reusable engines built once over ``df_full``:

- :class:`SetProximityEngine` — "did wallets of a set trade this token recently?"
  (proximity signals; the generic form of leader/quality-wallet proximity).
- :class:`PositionSignalEngine` — "what is the aggregate open position of a
  set on this token right now, and how far is the current price from their
  average entry?"  Position / value-at-cost / underwater / entry-premium
  families, computed with an exact two-pass cumsum + merge_asof over post-trade
  checkpoints (average-cost accounting, execution-order-aware).

Also provides per-wallet archetype metrics (hold times, round trips) and
archetype set definitions (gamblers, whales, retail, scalpers, ...).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised indirectly in test envs
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

from .signal_lib import bootstrap_ic, compute_event_ic


# ---------------------------------------------------------------------------
# SetProximityEngine
# ---------------------------------------------------------------------------


class SetProximityEngine:
    """Nearest-event proximity of a wallet set to candidate trades.

    Membership is done via category codes (``np.isin`` on object arrays is far
    too slow).  Events are cached per (set-name, side).
    """

    def __init__(self, df_full: pd.DataFrame):
        cat = df_full['wallet'].astype('category')
        self._codes = cat.cat.codes.to_numpy()
        self._categories = cat.cat.categories.to_numpy()
        self._is_buy = (df_full['side'].to_numpy() == 'BUY')
        self._is_sell = (df_full['side'].to_numpy() == 'SELL')
        self._compact = df_full[['dt', 'condition_id', 'outcome']]
        self._cache: dict[tuple, pd.DataFrame] = {}

    def build_union_events(self, wallets, side, key=None):
        """All trades by `wallets` on `side` as a compact event frame (hit=1.0).

        Events are cached by (key, side); pass key=set-name to reuse across phases.
        """
        cache_key = key if key is not None else id(wallets)
        k = (cache_key, side)
        if k in self._cache:
            return self._cache[k]
        wallet_arr = np.asarray(list(wallets), dtype=object)
        present = np.isin(wallet_arr, self._categories)
        codes_sub = np.searchsorted(self._categories, wallet_arr[present])
        side_mask = self._is_buy if side == 'BUY' else self._is_sell
        idx = np.flatnonzero(np.isin(self._codes, codes_sub) & side_mask)
        ev = self._compact.iloc[idx].copy()
        ev['hit'] = 1.0
        ev = ev.sort_values('dt')
        self._cache[k] = ev
        return ev

    def merge_set_signal(self, cand, events, opposite=False, col_name=None,
                         tolerance_min=5):
        """Nearest set event on same (condition_id, outcome) within the last
        `tolerance_min` minutes (exact matches excluded). NaN where none fired.
        `events` must be dt-sorted (as returned by build_union_events).
        """
        if opposite:
            events = events.assign(outcome=events['outcome'].map({'No': 'Yes', 'Yes': 'No'}))
        left = cand.sort_values('dt')
        left_idx = left.index
        merged = pd.merge_asof(
            left,
            events[['dt', 'condition_id', 'outcome', 'hit']],
            on='dt',
            by=['condition_id', 'outcome'],
            direction='backward',
            tolerance=pd.Timedelta(minutes=tolerance_min),
            allow_exact_matches=False,
        )
        merged.index = left_idx
        merged = merged.sort_index()
        if col_name is not None:
            merged = merged.rename(columns={'hit': col_name})
        return merged

    def evaluate_set(self, df, col, roi='copyable_roi', min_fires=300):
        """Fired-event count and IC on the given frame (NaN if too few fires)."""
        fires = int(df[col].notna().sum())
        sig = df[col].fillna(0.0)
        ic = compute_event_ic(sig, df[roi]) if fires >= min_fires else np.nan
        return fires, ic


# ---------------------------------------------------------------------------
# PositionSignalEngine
# ---------------------------------------------------------------------------

_SWAP = {'Yes': 'No', 'No': 'Yes'}


@njit(nogil=True)
def _vac_pass(wallet_code, key_code, is_buy, qty, price, position):
    """Running average-cost value-at-cost per (wallet, key) checkpoint."""
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


@njit(nogil=True)
def _fresh_pass(wallet_code, key_code, is_buy, qty, price, position, dt_ns):
    """Value-at-cost plus quantity-weighted average entry time per checkpoint.

    ``entry`` = average entry timestamp (ns) of the currently-open position,
    tracked like cost accounting (buys add qty*dt, sells scale proportionally).
    Returns (vac, avg_entry_ns); avg_entry_ns = 0 where the position is flat.
    """
    n = len(wallet_code)
    vac = np.zeros(n, dtype=np.float64)
    entry = np.zeros(n, dtype=np.float64)
    w_prev = -1
    k_prev = -1
    pos_old = 0.0
    cost = 0.0
    etime = 0.0
    for i in range(n):
        w = wallet_code[i]
        k = key_code[i]
        if w != w_prev or k != k_prev:
            w_prev = w
            k_prev = k
            pos_old = 0.0
            cost = 0.0
            etime = 0.0
        pos_new = position[i]
        if is_buy[i]:
            cost += qty[i] * price[i]
            etime += qty[i] * dt_ns[i]
        elif pos_old > 1e-12:
            s = pos_new / pos_old
            cost *= s
            etime *= s
        else:
            cost = 0.0
            etime = 0.0
        pos_old = pos_new
        vac[i] = cost if pos_old > 0 else 0.0
        entry[i] = etime / pos_old if pos_old > 0 else 0.0
    return vac, entry


def _fresh_recency(dt_ns, avg_entry_ns, tau_ns):
    """Exponential recency weight exp(-age/tau) of each open position."""
    age = np.clip(dt_ns - avg_entry_ns, 0, None)
    return np.exp(-age / tau_ns)


class PositionSignalEngine:
    """Exact aggregate position / value-at-cost of a wallet set per token.

    Signal = aggregate state of the set's open positions on the candidate's
    token at time t (own / opposite outcome), via two-pass cumsum over
    post-trade checkpoints (validated against a per-wallet brute force):

        A(t) = cumsum of checkpoint values at nearest checkpoint <= t
        B(t) = cumsum of values at each wallet's NEXT checkpoint <= t
        signal = A(t) - B(t)   (0 where the set holds nothing at that time)

    Value-at-cost uses average-cost accounting; checkpoints are ordered by
    (wallet, condition, outcome, dt, -position) so same-timestamp trades are
    consumed in true execution order (post-trade position as ground truth).
    """

    def __init__(self, df_full: pd.DataFrame):
        cat = df_full['wallet'].astype('category')
        self._wcodes = cat.cat.codes.to_numpy()
        self._categories = cat.cat.categories.to_numpy()
        self._base = df_full
        self._build_rec()

    # -- checkpoint construction ------------------------------------------

    def _build_rec(self):
        """Precompute vac'd checkpoints + execution-order info ONCE.

        ``rec`` = one row per trade (post-trade checkpoint), pre-sorted by
        (wallet, key, dt, -position) with the correct per-row ``vac`` and an
        ``is_last`` flag (True on each (wallet, key) final checkpoint).  A set's
        A/B tables are then a masked groupby-cumsum of position/vac per key.
        """
        df = self._base
        kA, catsA = pd.factorize(df['condition_id'])
        kB = (df['outcome'].to_numpy() == 'Yes').astype(np.int64)
        kcode = 2 * kA + kB
        self._key_map = np.empty((2 * len(catsA), 2), dtype=object)
        self._key_map[0::2, 0] = catsA
        self._key_map[0::2, 1] = 'No'
        self._key_map[1::2, 0] = catsA
        self._key_map[1::2, 1] = 'Yes'
        w = self._wcodes
        side = df['side'].to_numpy()
        qty = df['quantity'].to_numpy()
        price = df['price'].to_numpy()
        pos = df['position'].to_numpy()
        dt = df['dt'].to_numpy()
        order = np.lexsort((-pos, dt, kcode, w))
        w = w[order]
        kcode = kcode[order]
        dt = dt[order]
        pos = pos[order]
        qty = qty[order]
        price = price[order]
        side = side[order]
        dt_ns = pd.DatetimeIndex(dt).asi8
        vac, avg_entry_ns = _fresh_pass(w, kcode, side == 'BUY', qty, price, pos,
                                        dt_ns)
        is_last = np.r_[(w[1:] != w[:-1]) | (kcode[1:] != kcode[:-1]), True]

        self._rec = pd.DataFrame({
            '_w': w, '_k': kcode, 'dt': dt,
            'position': pos, 'vac': vac, 'avg_entry_ns': avg_entry_ns,
            'is_last': is_last,
        })
        self._rec['b_dt'] = self._rec.groupby(['_w', '_k'], sort=False)['dt'].shift(-1)

    def build_checkpoints(self, wallets, conditions=None) -> pd.DataFrame:
        """All trades (BUY+SELL) by `wallets` with post-trade position checkpoints.

        `conditions` optionally restricts to a set of condition_ids (e.g. the
        candidate universe), which cuts the checkpoint build cost substantially.
        """
        wallet_arr = np.asarray(list(wallets), dtype=object)
        present = np.isin(wallet_arr, self._categories)
        codes_sub = np.searchsorted(self._categories, wallet_arr[present])
        idx = np.flatnonzero(np.isin(self._wcodes, codes_sub))
        if conditions is not None:
            cond_arr = np.asarray(list(conditions), dtype=object)
            idx = idx[np.isin(self._base['condition_id'].iloc[idx].to_numpy(), cond_arr)]
        return self._base.iloc[idx][
            ['dt', 'wallet', 'condition_id', 'outcome',
             'position', 'price', 'quantity', 'side']
        ].copy()

    def compute_vac(self, ck: pd.DataFrame) -> pd.DataFrame:
        """Attach per-checkpoint value-at-cost (execution-order-aware)."""
        wc, _wu = pd.factorize(ck['wallet'])
        kc, _ku = pd.factorize(ck['condition_id'] + '|' + ck['outcome'])
        ck = ck.assign(_wcode=wc, _kcode=kc, _negpos=-ck['position'].to_numpy())
        ck = ck.sort_values(['_wcode', '_kcode', 'dt', '_negpos'], kind='stable')
        ck['vac'] = _vac_pass(ck['_wcode'].to_numpy(), ck['_kcode'].to_numpy(),
                              (ck['side'] == 'BUY').to_numpy(),
                              ck['quantity'].to_numpy(), ck['price'].to_numpy(),
                              ck['position'].to_numpy())
        return ck.drop(columns=['_wcode', '_kcode', '_negpos'])

    def cumsum_tables(self, ck: pd.DataFrame, by_cols):
        """(A, B) asof tables keyed by by_cols; cum columns for position AND vac."""
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

    def aggregate_value(self, cand, A, B, by_cols) -> pd.DataFrame:
        """Exact aggregate of the set on cand's key at cand.dt.

        Returns one column per value column in A (``cum_*`` -> ``*``), equal to
        A's nearest-checkpoint cumsum minus B's (position / vac / fresh-*).
        """
        left = cand.sort_values('dt')[['dt'] + by_cols]
        idx = left.index
        a = pd.merge_asof(left, A, on='dt', by=by_cols, direction='backward',
                          allow_exact_matches=False)
        b = pd.merge_asof(left, B, on='dt', by=by_cols, direction='backward',
                          allow_exact_matches=False)
        val_cols = [c for c in A.columns if c not in ('dt', *by_cols)]
        out = {c[4:] if c.startswith('cum_') else c:
               (a[c].fillna(0.0) - b[c].fillna(0.0)).to_numpy()
               for c in val_cols}
        return pd.DataFrame(out, index=left.index).sort_index()

    # -- signal attachment -------------------------------------------------

    def attach_position_signals(self, df_c, set_name, A, B, by_cols=None):
        """Attach position / value-at-cost / entry-premium signal columns.

        Columns (``{own,opp,total}`` = candidate outcome / opposite / sum):

        - ``sig_pos_{var}_{set}``   aggregate quantity held
        - ``sig_val_{var}_{set}``   aggregate value-at-cost (USDC)
        - ``sig_avgc_own_{set}``    val/pos/price - 1  (entry premium, own)
        - ``sig_avgc_opp_{set}``    val/pos/(1-price) - 1 (entry premium, opp)
        - ``sig_uwl_own_{set}``     val_own - pos_own * price (USDC underwater)
        - ``sig_uwl_opp_{set}``     val_opp - pos_opp * (1-price)

        Every family is evaluated on the candidate's OWN outcome and on the
        OPPOSITE outcome (``_opp``) — both directions are always attached and
        tested downstream.

        If A/B carry ``cum_fpos``/``cum_fvac`` (fresh_tau_ns build), also
        attaches the recent-entry family ``sig_fpos_*`` / ``sig_fval_*`` /
        ``sig_favgc_*`` / ``sig_fuwl_*`` (recency-weighted, so a recently
        entered underwater position counts more than a long-held one).
        """
        by_cols = by_cols or ['condition_id', 'outcome']
        own = self.aggregate_value(df_c, A, B, by_cols)
        opp = self.aggregate_value(
            df_c.assign(outcome=df_c['outcome'].map(_SWAP)), A, B, by_cols)
        p_own, p_opp = own['pos'].to_numpy(), opp['pos'].to_numpy()
        v_own, v_opp = own['vac'].to_numpy(), opp['vac'].to_numpy()
        p_cand = df_c['price'].to_numpy()
        df_c[f'sig_pos_own_{set_name}'] = p_own
        df_c[f'sig_pos_opp_{set_name}'] = p_opp
        df_c[f'sig_pos_total_{set_name}'] = p_own + p_opp
        df_c[f'sig_val_own_{set_name}'] = v_own
        df_c[f'sig_val_opp_{set_name}'] = v_opp
        df_c[f'sig_val_total_{set_name}'] = v_own + v_opp
        with np.errstate(divide='ignore', invalid='ignore'):
            df_c[f'sig_avgc_own_{set_name}'] = np.where(
                p_own > 0, v_own / p_own / p_cand - 1.0, 0.0)
            df_c[f'sig_avgc_opp_{set_name}'] = np.where(
                p_opp > 0, v_opp / p_opp / (1.0 - p_cand) - 1.0, 0.0)
        df_c[f'sig_uwl_own_{set_name}'] = v_own - p_own * p_cand
        df_c[f'sig_uwl_opp_{set_name}'] = v_opp - p_opp * (1.0 - p_cand)

        if 'fpos' in own.columns:
            fp_own, fp_opp = own['fpos'].to_numpy(), opp['fpos'].to_numpy()
            fv_own, fv_opp = own['fvac'].to_numpy(), opp['fvac'].to_numpy()
            df_c[f'sig_fpos_own_{set_name}'] = fp_own
            df_c[f'sig_fpos_opp_{set_name}'] = fp_opp
            df_c[f'sig_fval_own_{set_name}'] = fv_own
            df_c[f'sig_fval_opp_{set_name}'] = fv_opp
            with np.errstate(divide='ignore', invalid='ignore'):
                df_c[f'sig_favgc_own_{set_name}'] = np.where(
                    fp_own > 0, fv_own / fp_own / p_cand - 1.0, 0.0)
                df_c[f'sig_favgc_opp_{set_name}'] = np.where(
                    fp_opp > 0, fv_opp / fp_opp / (1.0 - p_cand) - 1.0, 0.0)
            df_c[f'sig_fuwl_own_{set_name}'] = fv_own - fp_own * p_cand
            df_c[f'sig_fuwl_opp_{set_name}'] = fv_opp - fp_opp * (1.0 - p_cand)

    def build_set(self, wallets, conditions=None, fresh_tau_ns=None):
        """Fast per-set aggregation: mask precomputed rec, cumsum per key.

        Returns (A, B) asof tables keyed by (condition_id, outcome) with
        cum_pos / cum_vac columns (compatible with :meth:`aggregate_value`).
        B excludes each (wallet, key) final checkpoint, so A - B = aggregate
        open position at the last checkpoint <= t.

        If ``fresh_tau_ns`` is given, also adds cum_fpos / cum_fvac = position
        and value-at-cost weighted by exp(-age/tau) (recent-entry emphasis).
        """
        codes_sub = np.searchsorted(self._categories, np.sort(np.asarray(list(wallets), dtype=object)))
        mask = np.isin(self._rec['_w'].to_numpy(), codes_sub)
        r = self._rec[mask]
        if conditions is not None:
            cond_arr = np.asarray(list(conditions), dtype=object)
            keep = np.isin(self._key_map[:, 0], cond_arr)
            r = r[np.isin(r['_k'].to_numpy(), np.flatnonzero(keep))]
        r = r.sort_values(['_k', 'dt'], kind='stable')
        k = r['_k'].to_numpy()
        A = pd.DataFrame({
            '_k': k,
            'dt': r['dt'].to_numpy(),
            'cum_pos': r.groupby(k, sort=False)['position'].cumsum().to_numpy(),
            'cum_vac': r.groupby(k, sort=False)['vac'].cumsum().to_numpy(),
        })
        rb = r[~r['is_last'].to_numpy()].sort_values(['_k', 'b_dt'], kind='stable')
        k_b = rb['_k'].to_numpy()
        B = pd.DataFrame({
            '_k': k_b,
            'dt': rb['b_dt'].to_numpy(),
            'cum_pos': rb.groupby(k_b, sort=False)['position'].cumsum().to_numpy(),
            'cum_vac': rb.groupby(k_b, sort=False)['vac'].cumsum().to_numpy(),
        })
        if fresh_tau_ns is not None:
            rec = _fresh_recency(pd.DatetimeIndex(r['dt']).asi8,
                                 r['avg_entry_ns'].to_numpy(), fresh_tau_ns)
            r = r.assign(_fp=r['position'].to_numpy() * rec,
                         _fv=r['vac'].to_numpy() * rec)
            A['cum_fpos'] = r.groupby(k, sort=False)['_fp'].cumsum().to_numpy()
            A['cum_fvac'] = r.groupby(k, sort=False)['_fv'].cumsum().to_numpy()
            rec_b = _fresh_recency(pd.DatetimeIndex(rb['dt']).asi8,
                                   rb['avg_entry_ns'].to_numpy(), fresh_tau_ns)
            rb = rb.assign(_fp=rb['position'].to_numpy() * rec_b,
                           _fv=rb['vac'].to_numpy() * rec_b)
            B['cum_fpos'] = rb.groupby(k_b, sort=False)['_fp'].cumsum().to_numpy()
            B['cum_fvac'] = rb.groupby(k_b, sort=False)['_fv'].cumsum().to_numpy()
        for T in (A, B):
            km = self._key_map[T['_k'].to_numpy()]
            T['condition_id'] = km[:, 0]
            T['outcome'] = km[:, 1]
            T.drop(columns='_k', inplace=True)
        A = A.sort_values('dt', kind='stable')
        B = B.sort_values('dt', kind='stable')
        return A, B


# ---------------------------------------------------------------------------
# Per-wallet archetype metrics (train-time, cheap)
# ---------------------------------------------------------------------------


def compute_hold_time_metrics(df_train: pd.DataFrame) -> pd.DataFrame:
    """Per-wallet median buy->sell hold and sell->buy flip times (minutes).

    Round-trip = a BUY immediately followed by a SELL of the same wallet on the
    same (condition_id, outcome). Returns one row per wallet with
    ``n_round_trips``, ``median_hold_min``, ``median_flip_min``,
    ``p25_hold_min``, ``round_trip_rate`` (round trips / buys).
    """
    cols = ['wallet', 'condition_id', 'outcome', 'dt', 'side']
    df = df_train[cols].copy()
    df = df.sort_values(['wallet', 'condition_id', 'outcome', 'dt'], kind='mergesort')
    key = ['wallet', 'condition_id', 'outcome']

    df['next_dt'] = df.groupby(key, sort=False)['dt'].shift(-1)
    df['next_side'] = df.groupby(key, sort=False)['side'].shift(-1)

    buy = df[df['side'] == 'BUY']
    hold = buy[buy['next_side'] == 'SELL'].copy()
    hold['hold_min'] = (hold['next_dt'] - hold['dt']).dt.total_seconds() / 60.0

    sell = df[df['side'] == 'SELL']
    flip = sell[sell['next_side'] == 'BUY'].copy()
    flip['flip_min'] = (flip['next_dt'] - flip['dt']).dt.total_seconds() / 60.0

    agg_buy = df.groupby('wallet', sort=False)['side'].agg(
        n_buys=lambda s: (s == 'BUY').sum())
    agg = agg_buy.join(
        hold.groupby('wallet', sort=False)['hold_min'].agg(
            median_hold_min='median', p25_hold_min=lambda s: s.quantile(0.25))
    ).join(
        flip.groupby('wallet', sort=False)['flip_min'].agg(
            median_flip_min='median')
    ).join(
        hold.groupby('wallet', sort=False)['hold_min'].agg(n_round_trips='size')
    ).reset_index()
    agg['median_hold_min'] = agg['median_hold_min'].fillna(np.inf)
    agg['median_flip_min'] = agg['median_flip_min'].fillna(np.inf)
    agg['p25_hold_min'] = agg['p25_hold_min'].fillna(np.inf)
    agg['n_round_trips'] = agg['n_round_trips'].fillna(0).astype(int)
    agg['round_trip_rate'] = agg['n_round_trips'] / agg['n_buys'].clip(lower=1)
    return agg


# ---------------------------------------------------------------------------
# Archetype set definitions (from wallet_vol + optional hold-time metrics)
# ---------------------------------------------------------------------------


def quantile_thresholds(series: pd.Series, qs=(0.25, 0.75, 0.8)) -> dict:
    s = series.dropna()
    return {f'p{int(q * 100)}': float(s.quantile(q)) for q in qs}


def archetype_sets(wallet_vol: pd.DataFrame, hold=None,
                   min_trade_count: int = 100) -> dict[str, pd.DataFrame]:
    """Archetype wallet sets from train-period wallet_vol (and optional hold
    time metrics). Returns dict name -> DataFrame of selected wallets.

    Definitions are data-driven (quantiles over the active population with
    ``trade_count >= min_trade_count``).
    """
    w = wallet_vol.copy()
    active = w[w['trade_count'] >= min_trade_count].copy()
    if active.empty:
        return {}
    active['avg_trade_usdc'] = active['total_notional'] / active['trade_count'].clip(lower=1)

    t_total = quantile_thresholds(active['total_notional'], (0.6, 0.75, 0.8))
    t_avg = quantile_thresholds(active['avg_trade_usdc'], (0.6, 0.75, 0.8))
    t_vol = quantile_thresholds(active['pnl_volatility'], (0.6, 0.75, 0.8))
    t_topmkt = quantile_thresholds(active['top_market_pnl_pct'], (0.6, 0.75, 0.8))
    t_posbuck = quantile_thresholds(active['positive_bucket_share'], (0.2, 0.4, 0.5))

    masks: dict[str, pd.Series] = {
        # Big total capital AND big average trade size.
        'whale': (
            (active['total_notional'] >= t_total['p75'])
            & (active['avg_trade_usdc'] >= t_avg['p75'])
        ),
        # Small average trade size (retail-sized), still active.
        'retail': (
            (active['avg_trade_usdc'] <= active['avg_trade_usdc'].quantile(0.25))
            & (active['total_notional'] <= active['total_notional'].quantile(0.6))
        ),
        # Lottery-ticket gambler: volatile, concentrated in one market, mostly
        # losing buckets but a single big winner carries total PnL.
        'gambler': (
            (active['pnl_volatility'] >= t_vol['p75'])
            & (active['top_market_pnl_pct'] >= t_topmkt['p75'])
            & (active['positive_bucket_share'] <= t_posbuck['p50'])
            & (active['num_markets'] <= active['num_markets'].quantile(0.6))
        ),
        # Net-negative on sells yet profitable overall (the original overseller).
        'overseller': (
            (active['sell_pnl'] < 0) & (active['total_pnl'] > 0)
        ),
        'overseller_deep': (
            (active['sell_pnl'] < 0) & (active['total_pnl'] > 0)
            & (active['sell_roi'] < -0.1)
        ),
        'overseller_thin': (
            (active['sell_pnl'] < 0) & (active['total_pnl'] > 0)
            & (active['buy_pnl'] < 50)
        ),
        # Consistent winners (low drawdown, diversified, positive buy edge).
        'consistent': (
            (active['buy_roi'] >= 0.05)
            & (active['max_drawdown_to_pnl'].fillna(1.0) <= 0.3)
            & (active['num_markets'] >= 20)
            & (active['copyable_roi'].fillna(0.0) >= 0.05)
        ),
        # Deeply underwater on the way in / out.
        'max_dd': (
            active['max_drawdown_to_pnl'].fillna(1.0) >= 0.6
        ),
        # Heavy both-side traders (large buy AND sell notional).
        'both_sides': (
            (active['buy_notional'] >= active['buy_notional'].quantile(0.6))
            & (active['sell_notional'] >= active['sell_notional'].quantile(0.6))
        ),
    }

    if hold is not None:
        active_h = active.copy()
        if 'median_hold_min' not in active_h.columns:
            hold_by_wallet = hold.set_index('wallet')
            for c in ['median_hold_min', 'median_flip_min', 'p25_hold_min',
                      'n_round_trips', 'round_trip_rate']:
                if c in hold_by_wallet.columns:
                    active_h[c] = active_h['wallet'].map(hold_by_wallet[c])
        active_h['median_hold_min'] = active_h['median_hold_min'].replace(np.inf, np.nan)
        active_h['median_flip_min'] = active_h['median_flip_min'].replace(np.inf, np.nan)
        h50 = active_h['median_hold_min'].quantile(0.5)
        masks['scalper'] = (
            (active_h['n_round_trips'] >= 20)
            & (active_h['median_hold_min'] <= active_h['median_hold_min'].quantile(0.25))
            & (active_h['buy_roi'] > 0) & (active_h['sell_roi'] > 0)
        )
        masks['flipper'] = (
            (active_h['median_flip_min'] <= active_h['median_flip_min'].quantile(0.25))
            & (active_h['n_round_trips'] >= 20)
        )

    out = {}
    for name, m in masks.items():
        sel = active[m].copy()
        if len(sel) >= 5:
            out[name] = sel
    return out


ALL_POSITION_KINDS = [
    ('pos', 'own'), ('pos', 'opp'), ('pos', 'total'),
    ('val', 'own'), ('val', 'opp'), ('val', 'total'),
    ('avgc', 'own'), ('avgc', 'opp'),
    ('uwl', 'own'), ('uwl', 'opp'),
]


def position_report(engine, c_train, c_val, c_test, set_wallets, set_name,
                    roi_col='copyable_roi', kinds=None, presence_min=0.005,
                    alpha=0.05, n_boot=500, seed=42, min_ic=None,
                    conditions=None):
    """Attach all position signals for one set and return per-signal IC rows.

    ICs are computed against ``roi_col`` (pass a price-residualized ROI to
    strip the favorite/price effect).  Selection: sign-consistent IC on train
    AND val, bootstrap CI of the pooled train+val IC excluding 0 (alpha), an
    optional |IC| floor ``min_ic``, and presence >= ``presence_min``.
    ``kinds`` defaults to the non-redundant pos/val x own/opp variants.
    """
    if kinds is None:
        kinds = [('pos', 'own'), ('pos', 'opp'), ('val', 'own'), ('val', 'opp')]
    A, B = engine.build_set(set_wallets, conditions)
    for df_c in (c_train, c_val, c_test):
        engine.attach_position_signals(df_c, set_name, A, B)

    rows, selected = [], []
    for kind, var in kinds:
        col = f'sig_{kind}_{var}_{set_name}'
        pres = float((c_train[col] > 0).mean())
        ics = {lbl: compute_event_ic(df_c[col].fillna(0.0), df_c[roi_col])
               for lbl, df_c in [('train', c_train), ('val', c_val), ('test', c_test)]}
        pooled = pd.concat([c_train[[col, roi_col]], c_val[[col, roi_col]]],
                           ignore_index=True)
        m, lo, hi = bootstrap_ic(pooled[col].fillna(0.0), pooled[roi_col],
                                 n_iter=n_boot, alpha=alpha, seed=seed)
        significant = np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)
        rows.append({'signal': col, 'kind': f'{kind}_{var}',
                     'presence_train': pres,
                     'IC_train': ics['train'], 'IC_val': ics['val'],
                     'IC_test': ics['test'],
                     'boot_mean_ic': m, 'boot_ci_lo': lo, 'boot_ci_hi': hi,
                     'significant': significant})
        ic_t, ic_v = ics['train'], ics['val']
        consistent = (np.isfinite(ic_t) and np.isfinite(ic_v)
                      and np.sign(ic_t) * np.sign(ic_v) > 0)
        if (consistent and significant and pres >= presence_min
                and (min_ic is None
                     or (abs(ic_t) >= min_ic and abs(ic_v) >= min_ic))):
            selected.append(col)
    rep = pd.DataFrame(rows)
    rep['|IC_train|'] = rep['IC_train'].abs()
    rep = rep.sort_values('|IC_train|', ascending=False).reset_index(drop=True)
    return rep, selected
