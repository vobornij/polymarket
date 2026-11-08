"""Compute prior-60s contrarian flow features for BUY fills from raw trades.

For each BUY fill of a tracked wallet, scans ALL raw trades in the same
condition during [t-60s, t) and computes:

- flowT_usdc_60: net USD flow in T direction
    = usdc_buyT - usdc_sellT + usdc_sellC - usdc_buyC
  (buying T or selling C = positive; selling T or buying C = negative)
- usdc_all_60: total USDC traded in the window
- price_chg_last_tok: fill price - last token-T trade price before the fill

Usage: standalone script to pre-compute and cache features.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


def compute_flow_features(
    fills: pd.DataFrame,
    raw_trades_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute prior-60s flow features for BUY fills.

    Parameters
    ----------
    fills : DataFrame with columns [fill_id, condition_id, token_id, dt, price]
        Each row is a BUY fill of a tracked wallet.
    raw_trades_dir : Path to raw trades directory (sharded parquet).

    Returns
    -------
    DataFrame with [fill_id, flowT_usdc_60, usdc_all_60, price_chg_last_tok].
    """
    fills = fills.copy()
    fills["dt"] = pd.to_datetime(fills["dt"], utc=True)
    fills["shard"] = fills["condition_id"].astype(str).str[2]

    cols = [
        "condition_id", "block_timestamp", "token_id",
        "side", "price", "quantity", "usdc_amount",
    ]

    all_feats = []
    t00 = time.time()

    for shard in sorted(fills["shard"].unique()):
        sf = fills[fills["shard"] == shard]
        conds = list(sf["condition_id"].astype(str).unique())

        t0 = time.time()
        ev = (
            ds.dataset(raw_trades_dir / f"{shard}.parquet", format="parquet")
            .to_table(columns=cols, filter=ds.field("condition_id").isin(conds))
            .to_pandas()
        )
        ev = ev.sort_values(["condition_id", "block_timestamp"]).reset_index(drop=True)
        groups = {c: g for c, g in ev.groupby("condition_id", observed=True)}

        rows = []
        for cond, fl in sf.groupby("condition_id", observed=True):
            g = groups.get(cond)
            if g is None:
                for fid in fl["fill_id"].values:
                    rows.append({"fill_id": fid})
                continue

            ts = g["block_timestamp"].to_numpy(np.int64)
            tok = g["token_id"].to_numpy()
            side = g["side"].to_numpy()
            price = g["price"].to_numpy(float)
            qty = g["quantity"].to_numpy(float)
            usdc = g["usdc_amount"].to_numpy(float)

            # per-token subarrays
            sub = {}
            for t_ in np.unique(tok):
                m = tok == t_
                sub[str(t_)] = (
                    ts[m], side[m], price[m], qty[m], usdc[m],
                )

            for _, r in fl.iterrows():
                t = int(r["dt"].timestamp())
                T = str(r["token_id"])
                out = {"fill_id": r["fill_id"]}

                # token T subarray for last-price lookup
                if T in sub:
                    tsT, _, priceT, _, _ = sub[T]
                    hiT = np.searchsorted(tsT, t, side="left")
                    if hiT > 0:
                        out["price_chg_last_tok"] = float(r["price"] - priceT[hiT - 1])
                    else:
                        out["price_chg_last_tok"] = np.nan
                else:
                    out["price_chg_last_tok"] = np.nan

                # 60s window on all trades
                lo = np.searchsorted(ts, t - 60, side="left")
                hi = np.searchsorted(ts, t, side="left")
                if hi > lo:
                    sl = slice(lo, hi)
                    tt = tok[sl]
                    ss = side[sl]
                    uu = usdc[sl]
                    mT = tt == T
                    buyT = mT & (ss == "BUY")
                    sellT = mT & (ss == "SELL")
                    buyC = (~mT) & (ss == "BUY")
                    sellC = (~mT) & (ss == "SELL")
                    out["usdc_all_60"] = float(uu.sum())
                    out["flowT_usdc_60"] = float(
                        uu[buyT].sum()
                        - uu[sellT].sum()
                        + uu[sellC].sum()
                        - uu[buyC].sum()
                    )
                else:
                    out["usdc_all_60"] = 0.0
                    out["flowT_usdc_60"] = 0.0

                rows.append(out)

        ff = pd.DataFrame(rows)
        all_feats.append(ff)
        if verbose:
            print(
                f"  shard {shard}: fills={len(sf):,} events={len(ev):,} "
                f"elapsed={time.time() - t00:.1f}s",
                flush=True,
            )

    feat = pd.concat(all_feats, ignore_index=True)
    if verbose:
        print(f"  total features: {len(feat):,} in {time.time() - t00:.1f}s")
    return feat


if __name__ == "__main__":
    BASE = Path("/Users/vobornij/projects/polymarket")
    PROC = BASE / "data/polygon_trades_processed_sports"
    RAW = BASE / "data/trades_polygon_sports"
    WORK = BASE / "data/trade_signals_workspace_v2_sports"
    CACHE = WORK / "flow_features.parquet"

    sel = pd.read_parquet(WORK / "selected_wallets_v2.parquet")
    wallets = set(sel["wallet"].astype(str))
    print(f"selected wallets: {len(wallets)}")

    # Load processed BUY fills for these wallets
    d = ds.dataset(PROC, format="parquet")
    t = d.to_table(
        columns=["wallet", "condition_id", "token_id", "dt", "avg_price", "side"],
        filter=ds.field("wallet").isin(list(wallets)),
    )
    df = t.to_pandas()
    buy = df[df["side"] == "BUY"].copy()
    buy = buy.rename(columns={"avg_price": "price"})
    buy["dt"] = pd.to_datetime(buy["dt"], utc=True)
    buy = buy.sort_values(["wallet", "dt"]).reset_index(drop=True)
    buy["fill_id"] = np.arange(len(buy), dtype=np.int64)
    print(f"BUY fills for selected wallets: {len(buy):,}")
    print(f"unique conditions: {buy['condition_id'].nunique():,}")

    feat = compute_flow_features(buy, RAW)
    feat.to_parquet(CACHE, index=False)
    print(f"saved {CACHE} ({len(feat):,} rows)")

    # Also save the fill_id mapping for notebook merge
    buy[["fill_id", "wallet", "condition_id", "token_id", "dt"]].to_parquet(
        WORK / "flow_fill_ids.parquet", index=False
    )
    print(f"saved flow_fill_ids.parquet ({len(buy):,} rows)")
