from pathlib import Path

import pandas as pd
import numpy as np
import datetime

# --- Fenwick Tree ---
class BIT:
    def __init__(self, n):
        self.n = n
        self.tree = np.zeros(n+1)

    def update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def query(self, i):
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def range_sum(self, l, r):
        return self.query(r) - self.query(l-1)
    

def _factor_code(f: float) -> str:
    """3-digit code for a scale factor, e.g. 0.98 -> '098' (percent, zero-padded)."""
    return f"{round(f * 100):03d}"


# ── copy variants ─────────────────────────────────────────────────────────────
# Enrichment computes, per variant (lookback window, fill-price factor), the
# total quantity of future trades at a better/scaled price.  Each variant emits
# ``avail_copy_qty_{window_min}m_{factor_code}``; factor-1.0 variants also emit
# ``avail_copy_total_vol_*`` and ``avail_copy_count_*``.  Per-fill caps are
# ``copyable_qty_{window_min}m_{factor_code}``.
COPY_VARIANTS: list[tuple[int, float]] = [
    (5 * 60, 1.0),    # 5 min  @ price × 1.00  (primary variant)
    (5 * 60, 0.95),   # 5 min  @ price × 0.95
    (20 * 60, 1.0),   # 20 min @ price × 1.00
]


def variant_suffix(window_seconds: int, factor: float) -> str:
    """Suffix encoding a copy variant, e.g. ``5m_100`` (5 min at factor 1.0)."""
    return f"{window_seconds // 60}m_{_factor_code(factor)}"


def avail_copy_qty_col(window_seconds: int, factor: float) -> str:
    """Column name for a variant's future-better-price quantity."""
    return f"avail_copy_qty_{variant_suffix(window_seconds, factor)}"


def avail_copy_total_vol_col(window_seconds: int, factor: float) -> str:
    """Column name for a variant's future-better-price volume."""
    return f"avail_copy_total_vol_{variant_suffix(window_seconds, factor)}"


def avail_copy_count_col(window_seconds: int, factor: float) -> str:
    """Column name for a variant's future-better-price trade count."""
    return f"avail_copy_count_{variant_suffix(window_seconds, factor)}"


def copyable_qty_col(window_seconds: int, factor: float) -> str:
    """Column name for a variant's per-fill copyable quantity."""
    return f"copyable_qty_{variant_suffix(window_seconds, factor)}"


def compute_future_better_price_qty(
    df: pd.DataFrame,
    window: datetime.timedelta,
    factors: tuple[float, ...] = (1.0,),
) -> pd.DataFrame:
    """
    For each trade, compute the total quantity of trades in the next `window` that
    have a better price.  Better price = lower for buys, higher for sells.

    ``factors`` lets the "better price" threshold be scaled relative to the fill
    price, modelling a copy limit order placed away from the market:

    - For a BUY fill at price ``p`` the adjusted (better) limit is ``p * f``.
    - For a SELL fill the adjusted limit is ``p / f``  (a higher, better price).
    - For the opposite (complement) token the threshold is compared in T-terms as
      ``1 - adjusted`` (i.e. against the stored "BUY T" trade tape).
    - Adjusted limits are rounded **down to 3 decimals**.

    Columns are suffixed with the variant's window+factor (see
    :func:`variant_suffix`), e.g. a 5-minute window at factor 1.0 emits
    ``avail_copy_qty_5m_100`` / ``avail_copy_total_vol_5m_100`` /
    ``avail_copy_count_5m_100``.  Factor ``1.0`` reproduces the original
    unscaled metric exactly (no rounding) and emits qty/vol/count; every
    additional factor ``f`` emits ``avail_copy_qty_{suffix}`` (qty only).

    Only trades with a strictly greater timestamp than the current fill are
    counted (same-timestamp trades are excluded), regardless of factor.
    """
    window_seconds = int(window.total_seconds())
    extra_factors = [f for f in factors if abs(f - 1.0) > 1e-12]

    qty_col = avail_copy_qty_col(window_seconds, 1.0)
    vol_col = avail_copy_total_vol_col(window_seconds, 1.0)
    count_col = avail_copy_count_col(window_seconds, 1.0)

    df = df.sort_values(["ts", "tx_hash"]).reset_index(drop=True)
    if df.empty:
        df[qty_col] = []
        df[vol_col] = []
        df[count_col] = []
        for f in extra_factors:
            df[avail_copy_qty_col(window_seconds, f)] = []
        return df

    # tree will store only "BUY T" trades. Full df contains both sides
    T = df["token_id"].iloc[0]

    df['T_price'] = np.nan

    mask1 = (df['token_id'] == T)
    mask2 = (df['token_id'] != T)

    df.loc[mask1, 'T_price'] = df.loc[mask1, 'price']
    df.loc[mask2, 'T_price'] = 1 - df.loc[mask2, 'price']


    ts = df["ts"].values
    t_price = df["T_price"].values
    price = df["price"].values
    qty = df["quantity"].values
    side = df["side"].values
    is_token = (df["token_id"] == T).to_numpy()
    tx_hash = df["tx_hash"].values

    n = len(df)

    # --- price compression ---
    unique_prices = np.sort(np.unique(t_price))
    price_to_idx = {p: i+1 for i, p in enumerate(unique_prices)}  # 1-based

    pidx = np.array([price_to_idx[p] for p in t_price])

    # "buy of T" = acquiring T: either BUY token T, or SELL token C (= buy T).
    # These are the rows that want a *lower* T-price; the complement wants higher.
    is_buy = (side == "BUY")
    is_buy_of_T = np.where(is_token, is_buy, ~is_buy)

    # --- per-factor thresholds (in compressed-price index space) ---
    # For buy-of-T rows we count trades with T_price < T_thresh (strictly lower);
    # for sell-of-T we count T_price > T_thresh (strictly higher).
    # factor 1.0 is handled with the exact pidx-based range sums below (no
    # rounding) to preserve backward compatibility; extra factors use searchsorted
    # against the floored-to-3dp adjusted threshold.
    bit_n = len(unique_prices)
    variant_hi = {}  # factor -> 1-based upper index for buy-of-T (range_sum(1, hi))
    variant_lo = {}  # factor -> 1-based lower index for sell-of-T (range_sum(lo+1, n))
    for f in extra_factors:
        adjusted = np.where(is_buy, price * f, price / f)
        adjusted = np.floor(adjusted * 1000.0) / 1000.0
        t_thresh = np.where(is_token, adjusted, 1.0 - adjusted)
        # number of unique prices strictly less than threshold
        less = np.searchsorted(unique_prices, t_thresh, side="left")
        # number of unique prices less-or-equal than threshold
        leq = np.searchsorted(unique_prices, t_thresh, side="right")
        suffix = variant_suffix(window_seconds, f)
        variant_hi[suffix] = less      # range_sum(1, less)
        variant_lo[suffix] = leq       # range_sum(leq + 1, bit_n)

    bit_qty = BIT(bit_n)
    bit_vol = BIT(bit_n)
    bit_count = BIT(bit_n)

    # --- pointers ---
    result_qty = np.zeros(n, dtype=np.float32)
    result_vol = np.zeros(n, dtype=np.float32)
    result_count = np.zeros(n, dtype=np.float32)
    result_qty_var = {suffix: np.zeros(n, dtype=np.float32) for suffix in variant_hi}

    add_ptr = 0
    remove_ptr = 0

    for i in range(n):
        end = ts[i] + window

        # add trades into window
        while add_ptr < n and ts[add_ptr] <= end:
            if (is_token[add_ptr] and side[add_ptr] == "BUY") or (not is_token[add_ptr] and side[add_ptr] == "SELL"):
                bit_qty.update(pidx[add_ptr], qty[add_ptr])
                bit_vol.update(pidx[add_ptr], qty[add_ptr] * t_price[add_ptr])
                bit_count.update(pidx[add_ptr], 1)
            add_ptr += 1

        #remove trades with the same timestamp as the current trade, since they are not "future" trades
        while remove_ptr < n and ts[remove_ptr] == ts[i]:
            if (is_token[remove_ptr] and side[remove_ptr] == "BUY") or (not is_token[remove_ptr] and side[remove_ptr] == "SELL"):
                bit_qty.update(pidx[remove_ptr], -qty[remove_ptr])
                bit_vol.update(pidx[remove_ptr], -qty[remove_ptr] * t_price[remove_ptr])
                bit_count.update(pidx[remove_ptr], -1)
            remove_ptr += 1

        # price in T (factor 1.0 — exact, unscaled)
        pi = pidx[i]
        buy_T = bool(is_buy_of_T[i])

        if buy_T:
            # better = lower price
            result_qty[i] = bit_qty.range_sum(1, pi-1)
            result_vol[i] = bit_vol.range_sum(1, pi-1)
            result_count[i] = bit_count.range_sum(1, pi-1)
            for suffix, arr in result_qty_var.items():
                arr[i] = bit_qty.range_sum(1, int(variant_hi[suffix][i]))
        else:
            # better = higher price
            result_qty[i] = bit_qty.range_sum(pi+1, bit_qty.n)
            result_vol[i] = bit_vol.range_sum(pi+1, bit_vol.n)
            result_count[i] = bit_count.range_sum(pi+1, bit_count.n)
            for suffix, arr in result_qty_var.items():
                arr[i] = bit_qty.range_sum(int(variant_lo[suffix][i]) + 1, bit_qty.n)

    df[qty_col] = result_qty
    df[vol_col] = result_vol
    df[count_col] = result_count
    for suffix, arr in result_qty_var.items():
        df[f"avail_copy_qty_{suffix}"] = arr

    # remove helper column
    del df['T_price']

    return df


def enrich_shard(
    f,
    enriched_dir: Path,
    token_df: pd.DataFrame,
    variants: list[tuple[int, float]] | None = None,
) -> None:
    """Enrich one raw shard with copyable-quantity metrics for ``variants``.

    ``variants`` is a list of ``(window_seconds, factor)`` tuples; defaults to
    :data:`COPY_VARIANTS`.  Each variant emits suffixed ``avail_copy_qty_*``
    columns plus a per-fill ``copyable_qty_*`` cap.
    """
    variants = COPY_VARIANTS if variants is None else variants
    if (enriched_dir / f"enriched_{f.name}").exists():
        print(f"Enriched file for {f.name} already exists, skipping...")
        return
    enriched_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(f)
    print(f"{len(raw)} trades in {f.name}")
    primary_qty_col = avail_copy_qty_col(variants[0][0], variants[0][1])
    if primary_qty_col in raw.columns:
        return
    raw = raw.merge(token_df[["token_id"]], on="token_id", how="inner")
    print(f"{len(raw)} trades after merging with token_df for {f.name}")

    _KEEP_COLS = [
        "tx_hash", "log_index", "block_timestamp", "condition_id", "token_id",
        "outcome", "token_winner", "wallet", "side", "price", "quantity",
        "usdc_amount", "position",
    ]
    raw = raw[_KEEP_COLS]

    raw['ts'] = pd.to_datetime(raw['block_timestamp'], utc=True, unit='s')

    # Group variants by lookback window so one pass covers all factors per window.
    windows: dict[int, list[float]] = {}
    for window_seconds, factor in variants:
        windows.setdefault(window_seconds, []).append(factor)

    parts = []
    for _, g in raw.groupby('condition_id', sort=False):
        out = g
        for window_seconds, factors in windows.items():
            out = compute_future_better_price_qty(
                out,
                window=pd.Timedelta(seconds=window_seconds),
                factors=tuple(factors),
            )
        parts.append(out)
    enriched = pd.concat(parts, ignore_index=True)

    for window_seconds, factor in variants:
        col = avail_copy_qty_col(window_seconds, factor)
        enriched[copyable_qty_col(window_seconds, factor)] = enriched['quantity'].clip(
            lower=0, upper=enriched[col],
        )
    enriched.to_parquet(enriched_dir / f"enriched_{f.name}", index=False)

    print(f"Enriched {f.name} with copyable_qty variants")