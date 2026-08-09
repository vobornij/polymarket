from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pandas as pd

from polymarket_analysis.preprocessing.fill_extender import (
    COPY_VARIANTS,
    avail_copy_count_col,
    avail_copy_qty_col,
    avail_copy_total_vol_col,
    compute_future_better_price_qty,
    variant_suffix,
)

RAW_SHARD = Path(__file__).resolve().parent.parent / "data" / "trades_polygon" / "1.parquet"

TEST_CIDS = [
    "0x18d19ee1593c5f758c9b652fae58b0ea98e8b28f3cc6c6ce42816dd032f9c8a7",
    "0x1e37c15513238520e046a4e3b11df38b2b406d3d94f9e18ff79b9a0bafe4754c",
    "0x102d7cb5492bf62003d39c7f884226b0399803af3cd7329815d900e7849f88ed",
    "0x1b73b2b6a11c2ab04e42aee41d8b8e2ddcbecf50a0344aeef3235e3d18f7aee6",
]


def reference_compute_future_better_price_qty(
    df: pd.DataFrame, window: pd.Timedelta, factors: tuple[float, ...] = (1.0,)
) -> pd.DataFrame:
    """O(n²) reference for compute_future_better_price_qty."""
    df = df.sort_values(["ts", "tx_hash"]).reset_index(drop=True)
    window_seconds = int(window.total_seconds())
    extra_factors = [f for f in factors if abs(f - 1.0) > 1e-12]
    if df.empty:
        df[
            [
                avail_copy_qty_col(window_seconds, 1.0),
                avail_copy_total_vol_col(window_seconds, 1.0),
                avail_copy_count_col(window_seconds, 1.0),
            ]
        ] = np.nan
        for f in extra_factors:
            df[avail_copy_qty_col(window_seconds, f)] = np.nan
        return df

    T = df["token_id"].iloc[0]
    is_token = (df["token_id"] == T).to_numpy()
    ts = df["ts"].values
    price = df["price"].values
    qty = df["quantity"].values
    side = df["side"].values
    n = len(df)

    t_price = np.where(is_token, price, 1 - price)

    result_qty = np.zeros(n, dtype=np.float64)
    result_vol = np.zeros(n, dtype=np.float64)
    result_count = np.zeros(n, dtype=np.float64)
    result_qty_var = {
        variant_suffix(window_seconds, f): np.zeros(n, dtype=np.float64)
        for f in extra_factors
    }

    for i in range(n):
        is_sell = side[i] == "SELL"
        is_T = is_token[i]
        need_higher = is_sell == is_T

        end = ts[i] + window

        j = i + 1
        while j < n and ts[j] <= ts[i]:
            j += 1

        while j < n and ts[j] <= end:
            is_bit = (is_token[j] and side[j] == "BUY") or (
                not is_token[j] and side[j] == "SELL"
            )
            if is_bit:
                if (
                    (need_higher and t_price[j] > t_price[i])
                    or (not need_higher and t_price[j] < t_price[i])
                ):
                    result_qty[i] += qty[j]
                    result_vol[i] += qty[j] * t_price[j]
                    result_count[i] += 1
            j += 1

        # scaled-factor variants
        for f in extra_factors:
            suffix = variant_suffix(window_seconds, f)
            adjusted = price[i] * f if side[i] == "BUY" else price[i] / f
            adjusted = np.floor(adjusted * 1000.0) / 1000.0
            thresh = adjusted if is_T else 1.0 - adjusted

            j = i + 1
            while j < n and ts[j] <= ts[i]:
                j += 1
            while j < n and ts[j] <= end:
                is_bit = (is_token[j] and side[j] == "BUY") or (
                    not is_token[j] and side[j] == "SELL"
                )
                if is_bit:
                    if (
                        (need_higher and t_price[j] > thresh)
                        or (not need_higher and t_price[j] < thresh)
                    ):
                        result_qty_var[suffix][i] += qty[j]
                j += 1

    df[avail_copy_qty_col(window_seconds, 1.0)] = result_qty
    df[avail_copy_total_vol_col(window_seconds, 1.0)] = result_vol
    df[avail_copy_count_col(window_seconds, 1.0)] = result_count
    for suffix, arr in result_qty_var.items():
        df[avail_copy_qty_col(window_seconds, _factor_from_suffix(suffix))] = arr
    return df


def _factor_from_suffix(suffix: str) -> float:
    """Recover the factor from a variant suffix like ``5m_095`` -> 0.95."""
    return float(int(suffix.split("_")[1])) / 100.0


def _load_cid(cid: str) -> pd.DataFrame:
    pf = pq.ParquetFile(RAW_SHARD)
    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        df = rg.to_pandas()
        mask = df["condition_id"].astype(str).str.startswith(cid[:30])
        if mask.any():
            result = df[mask].copy()
            result["ts"] = pd.to_datetime(
                result["block_timestamp"], utc=True, unit="s"
            )
            return result
    raise ValueError(f"CID {cid} not found in {RAW_SHARD}")


class TestReferenceImplementation:

    def test_agrees_with_main_impl(self):
        # Group variants by lookback window, mirroring enrich_shard.
        windows: dict[int, list[float]] = {}
        for window_seconds, factor in COPY_VARIANTS:
            windows.setdefault(window_seconds, []).append(factor)

        for cid in TEST_CIDS:
            raw = _load_cid(cid)
            main = raw
            ref = raw
            for window_seconds, factors in windows.items():
                window = pd.Timedelta(seconds=window_seconds)
                main = compute_future_better_price_qty(
                    main, window=window, factors=tuple(factors),
                )
                ref = reference_compute_future_better_price_qty(
                    ref, window=window, factors=tuple(factors),
                )

            print(f"\n=== {cid[:30]}... ({len(raw):5d} trades) ===")
            for window_seconds, factor in COPY_VARIANTS:
                suffix = variant_suffix(window_seconds, factor)
                qty_col = avail_copy_qty_col(window_seconds, factor)
                pq_ = main[qty_col].values
                rq_ = ref[qty_col].values
                qty_ok = np.max(np.abs(pq_ - rq_)) < 0.01
                assert qty_ok, f"{qty_col}: max|diff|={np.max(np.abs(pq_ - rq_)):.6f}"

                p_cq = np.minimum(main["quantity"].values, pq_)
                r_cq = np.minimum(ref["quantity"].values, rq_)
                assert np.max(np.abs(p_cq - r_cq)) < 0.01

                if abs(factor - 1.0) < 1e-12:
                    pv_ = main[avail_copy_total_vol_col(window_seconds, factor)].values
                    pc_ = main[avail_copy_count_col(window_seconds, factor)].values
                    rv_ = ref[avail_copy_total_vol_col(window_seconds, factor)].values
                    rc_ = ref[avail_copy_count_col(window_seconds, factor)].values
                    assert np.max(np.abs(pv_ - rv_)) < 0.01
                    assert np.max(np.abs(pc_ - rc_)) < 0.01

                print(
                    f"  {suffix:>8s}: qty max|diff|={np.max(np.abs(pq_ - rq_)):.6f}  "
                    f"cq max|diff|={np.max(np.abs(p_cq - r_cq)):.6f}"
                )

            # --- stats from both impls (primary variant: 5 min, factor 1.0) ---
            T = raw["token_id"].iloc[0]
            is_token = (raw["token_id"] == T).to_numpy()
            t_price = np.where(is_token, raw["price"].values, 1 - raw["price"].values)
            qty = raw["quantity"].values

            col = avail_copy_qty_col(5 * 60, 1.0)
            pq_ = main[col].values
            rq_ = ref[col].values
            p_cq = np.minimum(qty, pq_)
            r_cq = np.minimum(qty, rq_)
            pv_ = main[avail_copy_total_vol_col(5 * 60, 1.0)].values
            rv_ = ref[avail_copy_total_vol_col(5 * 60, 1.0)].values

            print(f"  {'metric':30s} {'fill_extender':>14s} {'reference':>14s} {'diff':>10s}")
            for label, mv, rv in [
                ("trades with avail_copy_qty > 0", (pq_ > 0.001).sum(), (rq_ > 0.001).sum()),
                ("total qty", qty.sum(), qty.sum()),
                ("total avail_copy_qty", pq_.sum(), rq_.sum()),
                ("total copyable_qty", p_cq.sum(), r_cq.sum()),
                ("  from T-trades", p_cq[is_token].sum(), r_cq[is_token].sum()),
                ("  from C-trades", p_cq[~is_token].sum(), r_cq[~is_token].sum()),
                ("total avail_copy_vol", pv_.sum(), rv_.sum()),
                ("total copyable_vol", (p_cq * t_price).sum(), (r_cq * t_price).sum()),
            ]:
                print(f"  {label:30s} {mv:>14.2f} {rv:>14.2f} {abs(mv-rv):>10.6f}")

            # For weather contract: show 07:37:31 fills detail
            if '18d19' in cid:
                afde_mask = raw["wallet"].astype(str).str.contains("afde46", na=False)
                ts_target = pd.Timestamp("2026-07-09 07:37:31", tz="UTC")
                at_ts = (raw["ts"] == ts_target) & afde_mask
                if at_ts.any():
                    print(f"\n  Fills at {ts_target} for wallet 0xafde46... ({at_ts.sum()} fills):")
                    print(f"  {'ts':>22} {'tx_hash':>30} {'side':6} {'tok':4} {'price':>7} {'T_price':>7} {'qty':>8} {'avail_cq':>10} {'copyable':>10}")
                    print(f"  {'-'*22} {'-'*30} {'-'*6} {'-'*4} {'-'*7} {'-'*7} {'-'*8} {'-'*10} {'-'*10}")
                    T = raw["token_id"].iloc[0]
                    for idx in np.where(at_ts)[0]:
                        r = main.iloc[idx]
                        tok = 'T' if r['token_id'] == T else 'C'
                        tp = r['price'] if tok == 'T' else 1 - r['price']
                        cq = min(r['quantity'], r[col])
                        print(f"  {str(r['ts']):>22} {str(r['tx_hash'])[:28]:>28}  {r['side']:6} {tok:4} {r['price']:>7.4f} {tp:>7.4f} {r['quantity']:>8.1f} {r[col]:>10.2f} {cq:>10.2f}")
                    print(f"  Sum copyable_qty = {p_cq[at_ts.values].sum():.2f}")
