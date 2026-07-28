"""Two-pass scoring: numba + multiprocess with batched groups + global int mapping."""
import sys, time
sys.path.insert(0, "/Users/vobornij/projects/polymarket/src")
import numpy as np, pandas as pd
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from numba import njit


@njit(cache=True, nogil=True)
def _pass1_leader_scores(fi_s, li_s, fw_ids, cp_sorted, tw_ns):
    n_f = len(fi_s)
    n_l = len(li_s)
    scores = np.empty(n_l, dtype=np.float64)
    distinct = np.empty(n_l, dtype=np.int64)
    f_start = 0
    f_end = 0
    for li_idx in range(n_l):
        lt_time = li_s[li_idx]
        lt_lo = lt_time - tw_ns
        while f_start < n_f and fi_s[f_start] < lt_lo:
            f_start += 1
        while f_end < n_f and fi_s[f_end] < lt_time:
            f_end += 1
        total = 0.0
        for k in range(f_start, f_end):
            total += cp_sorted[k]
        scores[li_idx] = total
        nd = 0
        for k in range(f_start, f_end):
            w = fw_ids[k]
            found = False
            for m in range(f_start, k):
                if fw_ids[m] == w:
                    found = True
                    break
            if not found:
                nd += 1
        distinct[li_idx] = nd
    return scores, distinct


@njit(cache=True, nogil=True)
def _pass2_follower_scores(fi_s, li_s, lw_ids, lt_scores, tw_ns):
    n_f = len(fi_s)
    n_l = len(li_s)
    lt_prefix = np.empty(n_l + 1, dtype=np.float64)
    lt_prefix[0] = 0.0
    for i in range(n_l):
        lt_prefix[i + 1] = lt_prefix[i] + lt_scores[i]
    scores = np.empty(n_f, dtype=np.float64)
    distinct = np.empty(n_f, dtype=np.int64)
    l_start = 0
    l_end = 0
    for fi_idx in range(n_f):
        ft_time = fi_s[fi_idx]
        ft_lo = ft_time - tw_ns
        while l_start < n_l and li_s[l_start] < ft_lo:
            l_start += 1
        while l_end < n_l and li_s[l_end] < ft_time:
            l_end += 1
        scores[fi_idx] = lt_prefix[l_end] - lt_prefix[l_start]
        nd = 0
        for k in range(l_start, l_end):
            w = lw_ids[k]
            found = False
            for m in range(l_start, k):
                if lw_ids[m] == w:
                    found = True
                    break
            if not found:
                nd += 1
        distinct[fi_idx] = nd
    return scores, distinct


def _process_batch(batch):
    """Process a batch of (fi_s, li_s, fw_ids, cp_sorted, lw_str, fw_str, tw_ns) tuples."""
    all_leader_out = []
    all_follower_out = []

    for fi_s, li_s, fw_ids, cp_sorted, lw_ids, lw_str, fw_str, tw_ns in batch:
        n_f, n_l = len(fi_s), len(li_s)
        if n_f == 0 or n_l == 0:
            continue

        lt_scores, lt_nd = _pass1_leader_scores(fi_s, li_s, fw_ids, cp_sorted, tw_ns)
        ft_scores, ft_nd = _pass2_follower_scores(fi_s, li_s, lw_ids, lt_scores, tw_ns)

        for i in range(n_l):
            all_leader_out.append((lw_str[i], lt_scores[i], lt_nd[i]))
        for i in range(n_f):
            all_follower_out.append((fw_str[i], ft_scores[i], ft_nd[i]))

    return all_leader_out, all_follower_out


def merge_results(all_results):
    leader_agg = defaultdict(lambda: [0.0, 0, 0.0])
    follower_agg = defaultdict(lambda: [0.0, 0, 0.0])
    for leader_out, follower_out in all_results:
        for lw, score, nd in leader_out:
            leader_agg[lw][0] += score
            leader_agg[lw][1] += 1
            leader_agg[lw][2] += nd
        for fw_, score, nd in follower_out:
            follower_agg[fw_][0] += score
            follower_agg[fw_][1] += 1
            follower_agg[fw_][2] += nd
    return leader_agg, follower_agg


if __name__ == "__main__":
    fb = pd.read_pickle("/tmp/_fb.pkl")
    lt = pd.read_pickle("/tmp/_lt.pkl")

    tw_ns = int(pd.Timedelta(minutes=30).total_seconds() * 1e9)
    fb["dt_ns"] = fb["dt"].astype(np.int64)
    lt["dt_ns"] = lt["dt"].astype(np.int64)

    # Global wallet→int mapping
    all_wallets = np.concatenate([fb["wallet"].values, lt["wallet"].values])
    _, wallet_inv = np.unique(all_wallets, return_inverse=True)
    wallet_ids_all = wallet_inv.astype(np.int32)
    n_fb = len(fb)
    fb_wallet_ids = wallet_ids_all[:n_fb]
    lt_wallet_ids = wallet_ids_all[n_fb:]

    fb["wallet_id"] = fb_wallet_ids
    lt["wallet_id"] = lt_wallet_ids

    fb_groups = {}
    for (cid, out), g in fb.groupby(["condition_id", "outcome"], sort=False):
        fb_groups[(cid, out)] = (
            g["wallet_id"].values.astype(np.int32),
            g["wallet"].values,
            g["dt_ns"].values,
            g["copyable_pnl"].values,
        )

    lt_groups = {}
    for (cid, out), g in lt.groupby(["condition_id", "outcome"], sort=False):
        lt_groups[(cid, out)] = (
            g["wallet_id"].values.astype(np.int32),
            g["wallet"].values,
            g["dt_ns"].values,
        )

    common = list(set(fb_groups) & set(lt_groups))
    print(f"{len(common)} groups, {cpu_count()} cores")

    # Build batches: accumulate small groups, keep big ones separate
    BATCH_TRADE_LIMIT = 50000  # max trades per batch
    batches = []
    current_batch = []
    current_size = 0

    # Sort by total size descending — big groups go first as standalone
    group_sizes = []
    for k in common:
        nf = len(fb_groups[k][0])
        nl = len(lt_groups[k][0])
        group_sizes.append((k, nf + nl))
    group_sizes.sort(key=lambda x: -x[1])

    for k, total in group_sizes:
        fw_id, fw_str, fi_dt, fg_cp = fb_groups[k]
        lw_id, lw_str, li_dt = lt_groups[k]

        order_f = np.argsort(fi_dt)
        order_l = np.argsort(li_dt)

        item = (
            fi_dt[order_f], li_dt[order_l],
            fw_id[order_f], fg_cp[order_f],
            lw_id[order_l], lw_str[order_l], fw_str[order_f],
            tw_ns,
        )

        if total > BATCH_TRADE_LIMIT:
            # Large group gets its own batch
            batches.append([item])
        else:
            current_batch.append(item)
            current_size += total
            if current_size >= BATCH_TRADE_LIMIT:
                batches.append(current_batch)
                current_batch = []
                current_size = 0

    if current_batch:
        batches.append(current_batch)

    print(f"{len(batches)} batches from {len(common)} groups")

    # Warmup
    print("Compiling numba...")
    t0 = time.perf_counter()
    _ = _process_batch(batches[0][:1])
    print(f"Warmup: {time.perf_counter()-t0:.1f}s")

    # Parallel
    t0 = time.perf_counter()
    with Pool() as pool:
        all_results = pool.map(_process_batch, batches)
    t_par = time.perf_counter() - t0
    print(f"Parallel: {t_par:.1f}s")

    leader_agg, follower_agg = merge_results(all_results)
    print(f"Leaders: {len(leader_agg)}, Followers: {len(follower_agg)}")

    print("\nLeader samples:")
    for lw in sorted(leader_agg, key=lambda x: -leader_agg[x][0])[:5]:
        s, n, d = leader_agg[lw]
        print(f"  {lw[:12]}: score={s:.0f}, trades={n}, avg_distinct={d/n:.1f}")

    print("\nFollower samples:")
    for fw_ in sorted(follower_agg, key=lambda x: -follower_agg[x][0])[:5]:
        s, n, d = follower_agg[fw_]
        print(f"  {fw_[:12]}: score={s:.0f}, trades={n}, avg_distinct={d/n:.1f}")
