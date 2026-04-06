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
    

def compute_future_better_price_qty(df: pd.DataFrame, window: datetime.timedelta) -> pd.DataFrame:
    """
    For each trade, compute the total quantity of trades in the next `window` that have a better price.
    Better price = lower for buys, higher for sells.
    """
    df = df.sort_values("ts").reset_index(drop=True)
    if(df.empty):
        df["avail_copy_qty"] = []
        df["avail_copy_total_vol"] = []
        df["avail_copy_count"] = []
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
    qty = df["quantity"].values
    side = df["side"].values
    is_token = (df["token_id"] == T).to_numpy()
    tx_hash = df["tx_hash"].values

    n = len(df)

    # --- price compression ---
    unique_prices = np.sort(np.unique(t_price))
    price_to_idx = {p: i+1 for i, p in enumerate(unique_prices)}  # 1-based

    pidx = np.array([price_to_idx[p] for p in t_price])


    bit_qty = BIT(len(unique_prices))
    bit_vol = BIT(len(unique_prices))
    bit_count = BIT(len(unique_prices))

    # --- pointers ---
    result_qty = np.zeros(n, dtype=np.float32)
    result_vol = np.zeros(n, dtype=np.float32)
    result_count = np.zeros(n, dtype=np.float32)

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

        #remove trades with the same hash
        while remove_ptr < n and tx_hash[remove_ptr] == tx_hash[i]:
            if (is_token[remove_ptr] and side[remove_ptr] == "BUY") or (not is_token[remove_ptr] and side[remove_ptr] == "SELL"):
                bit_qty.update(pidx[remove_ptr], -qty[remove_ptr])
                bit_vol.update(pidx[remove_ptr], -qty[remove_ptr] * t_price[remove_ptr])
                bit_count.update(pidx[remove_ptr], -1)
            remove_ptr += 1
    
        # price in T
        pi = pidx[i]

        if side[i] == "BUY" and is_token[i]:
            # better = lower or equal price
            result_qty[i] = bit_qty.range_sum(1, pi-1)
            result_vol[i] = bit_vol.range_sum(1, pi-1)
            result_count[i] = bit_count.range_sum(1, pi-1)
        elif side[i] == "SELL" and is_token[i]:
            # sell → better = higher or equal price
            result_qty[i] = bit_qty.range_sum(pi+1, bit_qty.n)
            result_vol[i] = bit_vol.range_sum(pi+1, bit_vol.n)
            result_count[i] = bit_count.range_sum(pi+1, bit_count.n)
        elif side[i] == "BUY" and not is_token[i]:
            # equivalent to SELL T with (1-price) → better = higher or equal price
            result_qty[i] = bit_qty.range_sum(pi+1, bit_qty.n)
            result_vol[i] = bit_vol.range_sum(pi+1, bit_vol.n)
            result_count[i] = bit_count.range_sum(pi+1, bit_count.n)
        else: # SELL and not is_token[i]
            result_qty[i] = bit_qty.range_sum(1, pi-1)
            result_vol[i] = bit_vol.range_sum(1, pi-1)
            result_count[i] = bit_count.range_sum(1, pi-1)

    df["avail_copy_qty"] = result_qty
    df["avail_copy_total_vol"] = result_vol
    df["avail_copy_count"] = result_count

    # remove helper column
    del df['T_price']

    return df


def enrich_shard(f, enriched_dir: Path, seconds: int, token_df: pd.DataFrame) -> None:
    if (enriched_dir / f"enriched_{f.name}").exists():
        print(f"Enriched file for {f.name} already exists, skipping...")
        return
    enriched_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(f)
    print(f"{len(raw)} trades in {f.name}")
    if("avail_copy_qty" in raw.columns): return
    raw = raw.merge(token_df[["token_id"]], on="token_id", how="inner")
    print(f"{len(raw)} trades after merging with token_df for {f.name}")


    raw['ts'] = pd.to_datetime(raw['block_timestamp'], utc=True, unit='s')

    enriched = raw.groupby('condition_id').apply(
        lambda df: compute_future_better_price_qty(df, window=pd.Timedelta(seconds=seconds)),
        include_groups=False
    ).reset_index()

    enriched['copyable_qty'] = enriched['quantity'].clip(lower=0, upper=enriched['avail_copy_qty'])
    enriched.to_parquet(enriched_dir / f"enriched_{f.name}", index=False)

    print(f"Enriched {f.name} with copyable_qty")