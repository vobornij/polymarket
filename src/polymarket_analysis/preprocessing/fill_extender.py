from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import datetime

from numba import njit


# --- Fenwick Tree primitives (JIT-compiled; indices are 1-based) ---
@njit(cache=True)
def _bit_update(tree, i, delta):
    n = tree.shape[0]
    while i < n:
        tree[i] += delta
        i += i & (-i)


@njit(cache=True)
def _bit_prefix(tree, i):
    s = 0.0
    while i > 0:
        s += tree[i]
        i -= i & (-i)
    return s


# --- Backward flow kernel (running accumulators, no trees needed) ---
@njit(cache=True)
def _backward_flow_kernel(
    ts, usdc, is_buy_of_T, is_token, price, window_ns,
):
    """Backward-looking flow features per row (sliding window, O(n)).

    For each row ``i`` the window is ``[ts[i] - window_ns, ts[i])`` — strictly
    before the row (same-timestamp trades are excluded).

    ``is_buy_of_T`` marks rows acquiring the reference token T (BUY T or SELL C).
    Flow toward T = Σ(usdc of is_buy_of_T rows) − Σ(usdc of the rest).

    Also tracks the running last price for each token (NOT windowed — the most
    recent own-token trade ever before the row) for ``price_chg_last_own``.

    Returns
    -------
    res_flowT : float64[n] — net USD flow toward reference token T in the window.
    res_usdc_all : float64[n] — total USD traded in the window.
    res_pchg_own : float64[n] — row price minus last own-token trade price
        (NaN when no prior own-token trade exists).
    """
    n = ts.shape[0]
    res_flowT = np.zeros(n, dtype=np.float64)
    res_usdc_all = np.zeros(n, dtype=np.float64)
    res_pchg_own = np.full(n, np.nan, dtype=np.float64)

    # Windowed running accumulators (signed by is_buy_of_T)
    acc_pos = 0.0  # Σ usdc of is_buy_of_T rows (positive T-flow)
    acc_neg = 0.0  # Σ usdc of ~is_buy_of_T rows (negative T-flow)

    # Running (non-windowed) last prices per token
    last_T_price = np.nan
    last_C_price = np.nan

    add_ptr = 0   # next row to add to the window
    evict_ptr = 0  # next row to evict from the window

    for i in range(n):
        # 1. Add all rows with ts < ts[i] (strictly before row i)
        while add_ptr < n and ts[add_ptr] < ts[i]:
            if is_buy_of_T[add_ptr]:
                acc_pos += usdc[add_ptr]
            else:
                acc_neg += usdc[add_ptr]
            # Update running last prices (not windowed)
            if is_token[add_ptr]:
                last_T_price = price[add_ptr]
            else:
                last_C_price = price[add_ptr]
            add_ptr += 1

        # 2. price_chg_last_own (running, not windowed)
        if is_token[i]:
            if not np.isnan(last_T_price):
                res_pchg_own[i] = price[i] - last_T_price
        else:
            if not np.isnan(last_C_price):
                res_pchg_own[i] = price[i] - last_C_price

        # 3. Evict rows with ts < ts[i] - window_ns
        cutoff = ts[i] - window_ns
        while evict_ptr < add_ptr and ts[evict_ptr] < cutoff:
            if is_buy_of_T[evict_ptr]:
                acc_pos -= usdc[evict_ptr]
            else:
                acc_neg -= usdc[evict_ptr]
            evict_ptr += 1

        # 4. Flow features from windowed accumulators
        res_flowT[i] = acc_pos - acc_neg
        res_usdc_all[i] = acc_pos + acc_neg

    return res_flowT, res_usdc_all, res_pchg_own


@njit(cache=True)
def _window_better_price_kernel(
    ts, t_price, qty, pidx, is_buy_of_T, bit_n, window_ns, variant_hi, variant_lo,
):
    """Sliding-window "future better-price quantity" kernel (Fenwick trees).

    Mirrors the original pure-Python implementation exactly:
    - rows are pre-sorted by (ts, tx_hash);
    - ``is_buy_of_T`` marks rows acquiring token T (stored in the trees) and rows
      wanting a *lower* T-price (range-summed from the bottom);
    - same-timestamp rows are excluded (removed from the window before the query);
    - ``variant_hi``/``variant_lo`` are per-extra-factor 1-based upper/lower
      compressed-price indices for the scaled-threshold variants.
    """
    n = ts.shape[0]
    n_extra = variant_hi.shape[0]
    tree_qty = np.zeros(bit_n + 1)
    tree_vol = np.zeros(bit_n + 1)
    tree_cnt = np.zeros(bit_n + 1)
    res_qty = np.zeros(n, dtype=np.float32)
    res_vol = np.zeros(n, dtype=np.float32)
    res_cnt = np.zeros(n, dtype=np.float32)
    res_var = np.zeros((n_extra, n), dtype=np.float32)

    add_ptr = 0
    remove_ptr = 0

    for i in range(n):
        end = ts[i] + window_ns

        # add trades into window
        while add_ptr < n and ts[add_ptr] <= end:
            if is_buy_of_T[add_ptr]:
                _bit_update(tree_qty, pidx[add_ptr], qty[add_ptr])
                _bit_update(tree_vol, pidx[add_ptr], qty[add_ptr] * t_price[add_ptr])
                _bit_update(tree_cnt, pidx[add_ptr], 1.0)
            add_ptr += 1

        # remove trades with the same timestamp as the current trade, since they
        # are not "future" trades
        while remove_ptr < n and ts[remove_ptr] == ts[i]:
            if is_buy_of_T[remove_ptr]:
                _bit_update(tree_qty, pidx[remove_ptr], -qty[remove_ptr])
                _bit_update(tree_vol, pidx[remove_ptr], -qty[remove_ptr] * t_price[remove_ptr])
                _bit_update(tree_cnt, pidx[remove_ptr], -1.0)
            remove_ptr += 1

        pi = pidx[i]

        if is_buy_of_T[i]:
            # better = lower price
            res_qty[i] = _bit_prefix(tree_qty, pi - 1)
            res_vol[i] = _bit_prefix(tree_vol, pi - 1)
            res_cnt[i] = _bit_prefix(tree_cnt, pi - 1)
            for v in range(n_extra):
                res_var[v, i] = _bit_prefix(tree_qty, variant_hi[v, i])
        else:
            # better = higher price
            res_qty[i] = _bit_prefix(tree_qty, bit_n) - _bit_prefix(tree_qty, pi)
            res_vol[i] = _bit_prefix(tree_vol, bit_n) - _bit_prefix(tree_vol, pi)
            res_cnt[i] = _bit_prefix(tree_cnt, bit_n) - _bit_prefix(tree_cnt, pi)
            for v in range(n_extra):
                res_var[v, i] = _bit_prefix(tree_qty, bit_n) - _bit_prefix(tree_qty, variant_lo[v, i])

    return res_qty, res_vol, res_cnt, res_var


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
    # (5 * 60, 0.95),   # 5 min  @ price × 0.95
    # (20 * 60, 1.0),   # 20 min @ price × 1.00
]

# ── backward flow window (seconds) ────────────────────────────────────────────
FLOW_WINDOW_S = 60  # prior-60s net trade flow against the bought token


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
    return _enrich_group(df, {window_seconds: list(factors)})


def _enrich_group(df: pd.DataFrame, windows: dict[int, list[float]]) -> pd.DataFrame:
    """Enrich one condition group for several lookback windows in one prep pass.

    Shared across windows: the (ts, tx_hash) sort, array extraction, price
    compression and the per-factor scaled thresholds (these do not depend on
    the window).  The JIT kernel is then run once per window.
    """
    df = df.sort_values(["ts", "tx_hash"]).reset_index(drop=True)
    if df.empty:
        for window_seconds, factors in windows.items():
            df[avail_copy_qty_col(window_seconds, 1.0)] = np.zeros(0, dtype=np.float32)
            df[avail_copy_total_vol_col(window_seconds, 1.0)] = np.zeros(0, dtype=np.float32)
            df[avail_copy_count_col(window_seconds, 1.0)] = np.zeros(0, dtype=np.float32)
            for f in factors:
                if abs(f - 1.0) > 1e-12:
                    df[avail_copy_qty_col(window_seconds, f)] = np.zeros(0, dtype=np.float32)
        df["flow_own_usdc_60"] = np.zeros(0, dtype=np.float64)
        df["usdc_all_60"] = np.zeros(0, dtype=np.float64)
        df["price_chg_last_own"] = np.zeros(0, dtype=np.float64)
        return df

    # tree will store only "BUY T" trades. Full df contains both sides
    T = df["token_id"].iloc[0]

    ts = df["ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
    price = df["price"].to_numpy(dtype="float64")
    t_price = np.where((df["token_id"] == T).to_numpy(), price, 1.0 - price)
    qty = df["quantity"].to_numpy(dtype="float64")
    is_buy = (df["side"] == "BUY").to_numpy()
    is_token = (df["token_id"] == T).to_numpy()
    # "buy of T" = acquiring T: either BUY token T, or SELL token C (= buy T).
    # These are the rows stored in the trees; they also want a *lower* T-price.
    is_buy_of_T = np.where(is_token, is_buy, ~is_buy)

    n = len(df)

    # --- price compression ---
    unique_prices = np.sort(np.unique(t_price))
    bit_n = len(unique_prices)
    # 1-based compressed price index (exact match always exists in unique_prices)
    pidx = np.searchsorted(unique_prices, t_price, side="left").astype(np.int64) + 1

    # --- per-extra-factor thresholds (in compressed-price index space) ---
    # For buy-of-T rows we count trades with T_price < T_thresh (strictly lower);
    # for sell-of-T we count T_price > T_thresh (strictly higher).
    # factor 1.0 is handled with the exact pidx-based range sums (no rounding)
    # to preserve backward compatibility; extra factors use searchsorted against
    # the floored-to-3dp adjusted threshold.  Thresholds do not depend on the
    # window, so they are computed once per distinct factor.
    hi_by_factor: dict[float, np.ndarray] = {}
    lo_by_factor: dict[float, np.ndarray] = {}
    for factors in windows.values():
        for f in factors:
            if abs(f - 1.0) <= 1e-12 or f in hi_by_factor:
                continue
            adjusted = np.where(is_buy, price * f, price / f)
            adjusted = np.floor(adjusted * 1000.0) / 1000.0
            t_thresh = np.where(is_token, adjusted, 1.0 - adjusted)
            # number of unique prices strictly less than / less-or-equal than threshold
            hi_by_factor[f] = np.searchsorted(unique_prices, t_thresh, side="left").astype(np.int64)
            lo_by_factor[f] = np.searchsorted(unique_prices, t_thresh, side="right").astype(np.int64)

    for window_seconds, factors in windows.items():
        extra_factors = [f for f in factors if abs(f - 1.0) > 1e-12]
        if extra_factors:
            variant_hi = np.stack([hi_by_factor[f] for f in extra_factors])
            variant_lo = np.stack([lo_by_factor[f] for f in extra_factors])
        else:
            variant_hi = np.zeros((0, n), dtype=np.int64)
            variant_lo = np.zeros((0, n), dtype=np.int64)

        result_qty, result_vol, result_count, result_qty_var = _window_better_price_kernel(
            ts, t_price, qty, pidx, is_buy_of_T, bit_n,
            window_seconds * 1_000_000_000, variant_hi, variant_lo,
        )

        df[avail_copy_qty_col(window_seconds, 1.0)] = result_qty
        df[avail_copy_total_vol_col(window_seconds, 1.0)] = result_vol
        df[avail_copy_count_col(window_seconds, 1.0)] = result_count
        for k, f in enumerate(extra_factors):
            df[avail_copy_qty_col(window_seconds, f)] = result_qty_var[k]

    # --- backward flow features (prior-60s window, token-relative) ---
    usdc = df["usdc_amount"].to_numpy(dtype=np.float64)
    res_flowT, res_usdc_all, res_pchg = _backward_flow_kernel(
        ts, usdc, is_buy_of_T, is_token, price,
        FLOW_WINDOW_S * 1_000_000_000,
    )
    df["flow_own_usdc_60"] = np.where(is_token, res_flowT, -res_flowT)
    df["usdc_all_60"] = res_usdc_all
    df["price_chg_last_own"] = res_pchg

    return df


def enrich_shard(
    f,
    enriched_dir: Path,
    token_lookup_path: Path,
    variants: list[tuple[int, float]],
    sample_rows: int | None = None,
) -> None:
    """Enrich one raw shard with copyable-quantity metrics for ``variants``.

    ``variants`` is a required list of ``(window_seconds, factor)`` tuples; each
    variant emits suffixed ``avail_copy_qty_*`` columns plus a per-fill
    ``copyable_qty_*`` cap.  The output additionally carries
    ``last_condition_trade_ts`` — the condition's max trade timestamp, derived
    from the shard itself (shards are partitioned by the first hex char of
    ``condition_id``, so a condition's trades live in exactly one shard and the
    per-shard max is the global max).

    ``sample_rows`` limits how many raw rows are read from the head of the shard
    (pipeline smoke-testing); ``None`` processes the whole shard.

    Memory layout: the shard is read in batches, each batch inner-merged with
    the token lookup's ``token_id`` column (dropping rows of markets missing
    from the lookup) and dictionary-encoded, and the enriched output is written
    incrementally per condition chunk.  Peak memory is roughly one filtered
    shard (categorical) instead of three full object-dtype copies.
    """
    out_path = enriched_dir / f"enriched_{f.name}"
    if out_path.exists():
        print(f"Enriched file for {f.name} already exists, skipping...")
        return
    enriched_dir.mkdir(parents=True, exist_ok=True)

    _KEEP_COLS = [
        "tx_hash", "log_index", "block_timestamp", "condition_id", "token_id",
        "outcome", "token_winner", "wallet", "side", "price", "quantity",
        "usdc_amount", "position",
    ]
    _CAT_COLS = ["condition_id", "token_id", "wallet", "side", "outcome"]

    parquet_file = pq.ParquetFile(f)
    if sample_rows is None:
        print(f"{parquet_file.metadata.num_rows} trades in {f.name}")
    else:
        print(f"sampling first {sample_rows:,} of {parquet_file.metadata.num_rows:,} trades in {f.name}")

    token_ids = pd.read_parquet(token_lookup_path, columns=["token_id"])

    raw_parts = []
    rows_read = 0
    for batch in parquet_file.iter_batches(batch_size=1_000_000, columns=_KEEP_COLS):
        bdf = batch.to_pandas()
        if sample_rows is not None:
            remaining = sample_rows - rows_read
            if remaining <= 0:
                break
            if len(bdf) > remaining:
                bdf = bdf.iloc[:remaining]
        rows_read += len(bdf)
        merged = bdf.merge(token_ids, on="token_id", how="inner")
        del bdf
        for col in _CAT_COLS:
            merged[col] = merged[col].astype("category")
        raw_parts.append(merged)
        if sample_rows is not None and rows_read >= sample_rows:
            break

    raw = pd.concat(raw_parts, ignore_index=True)
    del raw_parts
    print(f"{len(raw):,} trades after merging with token_df for {f.name}")
    if raw.empty:
        return

    raw['ts'] = pd.to_datetime(raw['block_timestamp'], utc=True, unit='s')

    # Group variants by lookback window so one pass covers all factors per window.
    windows: dict[int, list[float]] = {}
    for window_seconds, factor in variants:
        windows.setdefault(window_seconds, []).append(factor)

    writer = None
    buffer: list[pd.DataFrame] = []
    buffered_rows = 0
    FLUSH_ROWS = 500_000
    try:
        for _, g in raw.groupby('condition_id', sort=False, observed=True):
            out = _enrich_group(g, windows)
            for window_seconds, factor in variants:
                col = avail_copy_qty_col(window_seconds, factor)
                out[copyable_qty_col(window_seconds, factor)] = out['quantity'].clip(
                    lower=0, upper=out[col],
                )
            out["last_condition_trade_ts"] = pd.to_datetime(
                out["block_timestamp"].max(), unit="s", utc=True
            )
            buffer.append(out)
            buffered_rows += len(out)
            if buffered_rows >= FLUSH_ROWS:
                chunk = pd.concat(buffer, ignore_index=True)
                buffer, buffered_rows = [], 0
                writer = _write_enriched_chunk(writer, out_path, chunk)
        if buffer:
            chunk = pd.concat(buffer, ignore_index=True)
            writer = _write_enriched_chunk(writer, out_path, chunk)
    finally:
        if writer is not None:
            writer.close()

    print(f"Enriched {f.name} with copyable_qty variants ({len(raw):,} rows)")


def _write_enriched_chunk(writer, out_path: Path, chunk: pd.DataFrame):
    """Append one buffered chunk of enriched rows to the output parquet.

    Category columns are expanded back to plain strings first so every chunk
    produces an identical arrow schema (required by ``ParquetWriter``).
    """
    for col in chunk.columns:
        if isinstance(chunk[col].dtype, pd.CategoricalDtype):
            chunk[col] = chunk[col].astype(str)
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table)
    return writer