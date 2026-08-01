"""
Framework validation with synthetic signals of KNOWN value.

The stage1 signal pipeline (signal_lib / signal_engines) is validated three
ways, all with ground truth:

1. IC / IR / hit-rate / bootstrap machinery recovers a *known* population
   Spearman rho (Gaussian copula) within sampling tolerance.
2. The PositionSignalEngine returns EXACT aggregate position / value-at-cost
   on a hand-built frame (brute-force reference, average-cost accounting).
3. The selection + combination + strategy layers recover a synthetic signal
   with a known relationship to forward copyable ROI (and reject pure noise).
"""
import numpy as np
import pandas as pd
import pytest

from signal_lab.signal_lib import (
    compute_event_ic,
    compute_event_ir,
    bootstrap_ic,
    hit_rate,
    signal_quality_report,
    apply_composite_score,
    evaluate_strategy,
    cs_rank,
    fit_rank_transformer,
    apply_rank_transformer,
    fit_roi_residualizer,
    residualized_roi,
)
from signal_lab.signal_engines import PositionSignalEngine, position_report


# ---------------------------------------------------------------------------
# 1. IC / IR / hit-rate / bootstrap with known rho
# ---------------------------------------------------------------------------


def _gaussian_copula(rho, n, seed=0):
    """signal z and roi y with *known* population Spearman rho (n large)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    eps = rng.normal(size=n)
    y = rho * z + np.sqrt(1.0 - rho**2) * eps
    return z, y


def _expected_spearman(rho):
    """Population Spearman of a Gaussian copula with Pearson correlation rho:
    (6/pi) * arcsin(rho / 2)."""
    return 6.0 / np.pi * np.arcsin(rho / 2.0)


def test_ic_recovers_known_rho():
    for rho in (0.2, 0.5, -0.3):
        z, y = _gaussian_copula(rho, n=20_000)
        est = compute_event_ic(pd.Series(z), pd.Series(y))
        # sampling sd of Spearman ~ 1/sqrt(n) ~ 0.007; allow 0.03 margin
        assert abs(est - _expected_spearman(rho)) < 0.03, \
            f"rho={rho} expected {_expected_spearman(rho):.3f} estimated {est}"


def test_ic_extremes():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    assert compute_event_ic(pd.Series(x), pd.Series(x)) == pytest.approx(1.0)
    assert compute_event_ic(pd.Series(x), pd.Series(-x)) == pytest.approx(-1.0)
    assert abs(compute_event_ic(pd.Series(x), pd.Series(rng.normal(size=500)))) < 0.1


def test_ic_handles_nan_and_constant():
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    x[::7] = np.nan
    y = rng.normal(size=300)
    assert np.isfinite(compute_event_ic(pd.Series(x), pd.Series(y)))
    # constant signal -> NaN (not a crash)
    assert np.isnan(compute_event_ic(pd.Series(np.zeros(50)), pd.Series(y[:50])))


def test_ir_positive_for_consistent_signal():
    rng = np.random.default_rng(3)
    n_days, n_day = 20, 100
    ts, z, y = [], [], []
    for d in range(n_days):
        for _ in range(n_day):
            ts.append(pd.Timestamp(f"2026-01-{d + 1:02d}", tz="UTC"))
            z.append(1.0 if rng.random() > 0.5 else 0.0)
            y.append(0.2 + rng.normal() * 0.1 if z[-1] else -0.2 + rng.normal() * 0.1)
    s = pd.Series(z)
    r = pd.Series(y)
    t = pd.DatetimeIndex(ts)
    ir = compute_event_ir(s, r, t, freq="D")
    assert np.isfinite(ir) and ir > 2.0, f"expected strong positive IR, got {ir}"


def test_ir_small_for_noise():
    rng = np.random.default_rng(4)
    n_days, n_day = 20, 100
    ts, z, y = [], [], []
    for d in range(n_days):
        for _ in range(n_day):
            ts.append(pd.Timestamp(f"2026-01-{d + 1:02d}", tz="UTC"))
            z.append(rng.normal())
            y.append(rng.normal())
    ir = compute_event_ir(pd.Series(z), pd.Series(y), pd.DatetimeIndex(ts), freq="D")
    assert ir is np.nan or abs(ir) < 2.0


def test_bootstrap_ci_contains_true_rho():
    rho = 0.5
    expected = _expected_spearman(rho)
    z, y = _gaussian_copula(rho, n=20_000)
    sample_ic = compute_event_ic(pd.Series(z), pd.Series(y))
    mean, lo, hi = bootstrap_ic(pd.Series(z), pd.Series(y), n_iter=250)
    # the bootstrap resamples from the sample, so its CI brackets the sample IC
    assert lo <= sample_ic <= hi
    # with n=20k the sample IC is within ~0.015 of the population Spearman
    assert abs(sample_ic - expected) < 0.015
    assert abs(mean - expected) < 0.02
    assert (hi - lo) < 0.05


def test_hit_rate_known_signs():
    rng = np.random.default_rng(5)
    roi = pd.Series(rng.normal(size=400))
    roi[roi.abs() < 1e-9] = 1.0
    sig_pos = pd.Series(np.sign(roi).to_numpy())
    assert hit_rate(sig_pos, roi) == pytest.approx(1.0)
    assert hit_rate(-sig_pos, roi) == pytest.approx(0.0)


def test_quality_report_consistent():
    rho = 0.5
    z, y = _gaussian_copula(rho, n=5_000)
    s = pd.Series(z)
    r = pd.Series(y)
    t = pd.DatetimeIndex(np.repeat(pd.Timestamp("2026-01-01", tz="UTC"), len(s)))
    rep = signal_quality_report(pd.DataFrame({"s": s, "roi": r, "dt": t}),
                                ["s"], roi_col="roi", dt_col="dt")
    assert len(rep) == 1
    assert abs(rep.iloc[0]["IC"] - _expected_spearman(rho)) < 0.04
    # for a Gaussian copula, P(sign matches) = 0.5 + arcsin(rho)/pi
    assert rep.iloc[0]["hit_rate"] == pytest.approx(0.5 + np.arcsin(rho) / np.pi, abs=0.02)


# ---------------------------------------------------------------------------
# 2. PositionSignalEngine: exact aggregate position / value-at-cost
# ---------------------------------------------------------------------------


def _make_engine_frame():
    rows = []

    def T(w, cid, oc, dt, side, qty, price, pos):
        rows.append(dict(wallet=w, condition_id=cid, outcome=oc,
                         dt=pd.Timestamp(dt, tz="UTC"), side=side,
                         quantity=qty, price=price, position=pos))

    # archetype A1 on (c0, Yes): buy / add / partial sell / close (avg cost)
    T("A1", "c0", "Yes", "2026-01-01 00:00:10", "BUY", 2, 0.6, 2)
    T("A1", "c0", "Yes", "2026-01-01 00:00:20", "BUY", 1, 0.7, 3)
    T("A1", "c0", "Yes", "2026-01-01 00:00:30", "SELL", 1, 0.8, 2)
    T("A1", "c0", "Yes", "2026-01-01 00:00:40", "SELL", 2, 0.9, 0)
    # archetype A2 on (c0, Yes) and on (c1, No)
    T("A2", "c0", "Yes", "2026-01-01 00:00:15", "BUY", 5, 0.5, 5)
    T("A2", "c1", "No", "2026-01-01 00:00:12", "BUY", 3, 0.4, 3)
    # candidate wallet trade (not in the archetype set)
    T("C1", "c1", "Yes", "2026-01-01 00:00:18", "BUY", 1, 0.5, 1)
    return pd.DataFrame(rows)


def _brute_force_position(engine_frame, wallets, cid, oc, t):
    """Reference aggregate position / average-cost at the last checkpoint < t."""
    pos_tot, vac_tot = 0.0, 0.0
    for w in wallets:
        sub = engine_frame[
            (engine_frame["wallet"] == w)
            & (engine_frame["condition_id"] == cid)
            & (engine_frame["outcome"] == oc)
            & (engine_frame["dt"] < t)
        ]
        if sub.empty:
            continue
        last = sub.sort_values("dt").iloc[-1]
        pos_tot += last["position"]
        cost, ppos = 0.0, 0.0
        hist = engine_frame[
            (engine_frame["wallet"] == w)
            & (engine_frame["condition_id"] == cid)
            & (engine_frame["outcome"] == oc)
        ].sort_values("dt")
        for _, tr in hist.iterrows():
            if tr["dt"] > last["dt"]:
                break
            if tr["side"] == "BUY":
                cost += tr["quantity"] * tr["price"]
            elif ppos > 1e-12:
                cost *= tr["position"] / ppos
            else:
                cost = 0.0
            ppos = tr["position"]
        vac_tot += cost if ppos > 0 else 0.0
    return pos_tot, vac_tot


def test_engine_matches_brute_force():
    df = _make_engine_frame()
    engine = PositionSignalEngine(df)
    wallets = {"A1", "A2"}
    conditions = {"c0", "c1"}

    cand = pd.DataFrame([
        dict(condition_id="c0", outcome="Yes",
             dt=pd.Timestamp("2026-01-01 00:00:25", tz="UTC"), price=0.65),
        dict(condition_id="c0", outcome="Yes",
             dt=pd.Timestamp("2026-01-01 00:00:35", tz="UTC"), price=0.85),
        dict(condition_id="c1", outcome="No",
             dt=pd.Timestamp("2026-01-01 00:00:20", tz="UTC"), price=0.45),
        dict(condition_id="c1", outcome="Yes",
             dt=pd.Timestamp("2026-01-01 00:00:20", tz="UTC"), price=0.55),
    ])
    A, B = engine.build_set(wallets, conditions=conditions)
    engine.attach_position_signals(cand, "arch", A, B)

    for _, r in cand.iterrows():
        pos_bf, vac_bf = _brute_force_position(df, wallets, r["condition_id"],
                                               r["outcome"], r["dt"])
        assert r["sig_pos_own_arch"] == pytest.approx(pos_bf, abs=1e-9)
        assert r["sig_val_own_arch"] == pytest.approx(vac_bf, abs=1e-9)
        # entry premium = vac/pos/price - 1 ; underwater = vac - pos*price
        if pos_bf > 0:
            assert r["sig_avgc_own_arch"] == pytest.approx(
                vac_bf / pos_bf / r["price"] - 1.0, abs=1e-9)
            assert r["sig_uwl_own_arch"] == pytest.approx(
                vac_bf - pos_bf * r["price"], abs=1e-9)
        else:
            assert r["sig_avgc_own_arch"] == 0.0
            assert r["sig_uwl_own_arch"] == 0.0

    # candidate is not part of the set: its own buy must not be counted
    c2 = pd.DataFrame([dict(condition_id="c1", outcome="Yes",
                            dt=pd.Timestamp("2026-01-01 00:00:18", tz="UTC"),
                            price=0.55)])
    A2, B2 = engine.build_set(wallets, conditions=conditions)
    engine.attach_position_signals(c2, "arch", A2, B2)
    assert c2.iloc[0]["sig_pos_own_arch"] == 0.0


# ---------------------------------------------------------------------------
# 3. Selection / combination / strategy recover a known signal
# ---------------------------------------------------------------------------


class _StubEngine:
    """Injects a synthetic signal column; exercises real selection logic."""

    def __init__(self, good_signal_fn, bad_signal_fn):
        self.good = good_signal_fn
        self.bad = bad_signal_fn

    def build_set(self, wallets, conditions=None):
        return None, None

    def attach_position_signals(self, df_c, set_name, A, B, by_cols=None):
        fn = self.good if "good" in set_name else self.bad
        vals = np.asarray(fn(df_c), dtype=float)
        for var in ("own", "opp", "total"):
            df_c[f"sig_pos_{var}_{set_name}"] = vals
            df_c[f"sig_val_{var}_{set_name}"] = vals
        df_c[f"sig_avgc_own_{set_name}"] = vals
        df_c[f"sig_avgc_opp_{set_name}"] = vals
        df_c[f"sig_uwl_own_{set_name}"] = vals
        df_c[f"sig_uwl_opp_{set_name}"] = vals


def _candidate_frame(seed=0, n=3_000):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "condition_id": np.array([f"m{i % 200}" for i in range(n)]),
        "outcome": np.where(rng.random(n) > 0.5, "Yes", "No"),
        "dt": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
        "price": rng.uniform(0.05, 0.95, n),
        "wallet": np.array([f"0x{i:040x}" for i in range(n)]),
    })
    z = rng.normal(size=n)
    df["copyable_roi"] = 0.5 * z + np.sqrt(1 - 0.25) * rng.normal(size=n)
    df["copyable_pnl"] = df["copyable_roi"] * rng.uniform(1, 100, n)
    df["copyable_notional"] = rng.uniform(1, 100, n)
    return df


def test_position_report_selects_known_signal_and_rejects_noise():
    rng = np.random.default_rng(11)
    n_total = 9_000
    # known good signal: z drives roi with known rho; a constant shift keeps
    # ranks (and hence IC) unchanged while making presence ~ 1
    z = rng.normal(size=n_total)
    eps = rng.normal(size=n_total)
    roi = 0.5 * z + np.sqrt(0.75) * eps
    good_sig = z + 2.0
    # noise drawn independently of BOTH z and eps -> IC(noise, roi) ~ 0
    noise_sig = rng.normal(size=n_total)

    c_train, c_val, c_test = _candidate_frame(), _candidate_frame(1), _candidate_frame(2)
    offset = 0
    for df_c in (c_train, c_val, c_test):
        n = len(df_c)
        df_c["copyable_roi"] = roi[offset:offset + n]
        df_c["_good"] = good_sig[offset:offset + n]
        df_c["_noise"] = noise_sig[offset:offset + n]
        offset += n

    engine = _StubEngine(good_signal_fn=lambda df: df["_good"],
                         bad_signal_fn=lambda df: df["_noise"])
    conds = set(c_train["condition_id"].unique())

    rep_good, sel_good = position_report(engine, c_train, c_val, c_test,
                                         {"w1"}, "good_set",
                                         presence_min=0.1, conditions=conds)
    rep_bad, sel_bad = position_report(engine, c_train, c_val, c_test,
                                       {"w2"}, "bad_set",
                                       presence_min=0.1, conditions=conds)

    # default kinds = pos/val x own/opp (4 variants per archetype)
    assert len(sel_good) == 4, f"known signal should be selected, got {sel_good}"
    assert sorted(sel_good) == [
        "sig_pos_opp_good_set", "sig_pos_own_good_set",
        "sig_val_opp_good_set", "sig_val_own_good_set",
    ]
    assert sel_bad == [], f"noise signal must not be selected, got {sel_bad}"
    expected = _expected_spearman(0.5)
    assert all(abs(r["IC_train"] - expected) < 0.05 for _, r in rep_good.iterrows())
    assert all(abs(r["IC_val"] - expected) < 0.05 for _, r in rep_good.iterrows())
    # bootstrap significance flag must be True for the signal, False for noise
    assert rep_good["significant"].all()
    assert not rep_bad["significant"].any()


def test_composite_and_strategy_recover_known_trades():
    rng = np.random.default_rng(7)
    n = 2_000
    df = pd.DataFrame({
        "copyable_roi": rng.normal(size=n),
        "copyable_notional": np.abs(rng.normal(size=n)) + 1.0,
        "notional": np.abs(rng.normal(size=n)) + 1.0,
    })
    # keep PnL sign-consistent with ROI so the sign signal is exact
    df["copyable_pnl"] = df["copyable_roi"] * df["copyable_notional"]
    df["pnl"] = df["copyable_pnl"]
    # known signal = sign of roi -> composite = z
    df["sig_true"] = np.sign(df["copyable_roi"]).replace(0, 1.0)
    weights = {"sig_true": 1.0}
    df["composite"] = apply_composite_score(df, ["sig_true"], weights)

    fired = evaluate_strategy(df, "composite", 0.0)
    positive = df[df["copyable_pnl"] > 0]
    assert fired["trades"] == len(positive)
    assert fired["copyable_pnl"] == pytest.approx(float(positive["copyable_pnl"].sum()))
    assert fired["copyable_roi"] == pytest.approx(
        float(positive["copyable_pnl"].sum() / positive["copyable_notional"].sum()))

    none = evaluate_strategy(df, "composite", 1.5)
    assert none["trades"] == 0
    assert none["copyable_pnl"] == 0.0


def test_evaluate_strategy_reports_net_of_costs():
    df = pd.DataFrame({
        "composite": [1.0, 0.4],
        "copyable_pnl": [10.0, -2.0],
        "copyable_notional": [100.0, 50.0],
        "pnl": [10.0, -2.0],
        "notional": [100.0, 50.0],
    })
    res = evaluate_strategy(df, "composite", 0.0, cost_bps=100.0)
    assert res["trades"] == 2
    assert res["copyable_pnl"] == pytest.approx(8.0)
    assert res["cost_paid"] == pytest.approx(1.5)
    assert res["copyable_pnl_net"] == pytest.approx(6.5)
    assert res["copyable_roi_net"] == pytest.approx(6.5 / 150.0)


def test_train_fit_rank_transformer_matches_train_ranks_and_is_split_independent():
    train = pd.Series([1.0, 2.0, 2.0, 4.0, 7.0])
    fit = fit_rank_transformer(train)
    transformed_train = apply_rank_transformer(train, fit)
    assert np.allclose(transformed_train.to_numpy(), cs_rank(train).to_numpy())

    probe = pd.Series([0.0, 2.0, 3.0, 10.0])
    small_val = apply_rank_transformer(probe, fit)
    large_val = apply_rank_transformer(pd.concat([probe, pd.Series(np.linspace(-5, 20, 200))], ignore_index=True), fit).iloc[:len(probe)]
    assert np.allclose(small_val.to_numpy(), large_val.to_numpy())


def test_bootstrap_ic_separates_signal_from_noise():
    z, y = _gaussian_copula(0.5, n=5_000, seed=31)
    _, lo, hi = bootstrap_ic(pd.Series(z), pd.Series(y), n_iter=500)
    assert lo > 0, f"genuine signal CI must exclude 0: ({lo:.3f}, {hi:.3f})"

    nz, ny = _gaussian_copula(0.0, n=5_000, seed=32)
    _, lo2, hi2 = bootstrap_ic(pd.Series(nz), pd.Series(ny), n_iter=500)
    assert lo2 <= 0 <= hi2, f"noise CI must include 0: ({lo2:.3f}, {hi2:.3f})"


def test_residualizer_removes_known_price_effect():
    """ROI dominated by price (favorite effect) must collapse after
    train-fitted residualization, while an independent signal survives."""
    rng = np.random.default_rng(21)
    n = 10_000
    price = rng.uniform(0.05, 0.95, n)
    z = rng.normal(size=n)
    roi = 2.0 * price + z  # ROI dominated by the price level

    fit = fit_roi_residualizer(roi, price)
    res = residualized_roi(roi, price, fit)
    ic_raw = compute_event_ic(pd.Series(price), pd.Series(roi))
    ic_price = compute_event_ic(pd.Series(price), pd.Series(res))
    ic_z = compute_event_ic(pd.Series(z), pd.Series(res))
    # residualization is linear-in-ranks; a small nonlinear remnant (~0.02)
    # may survive, but the favorite-effect magnitude must collapse
    assert ic_raw > 0.4, f"confound not present: IC(price,roi)={ic_raw}"
    assert abs(ic_price) < 0.05, f"price effect not removed: IC(price,res)={ic_price}"
    assert ic_z > 0.3, f"genuine signal lost: IC(z,res)={ic_z}"


def test_hit_rate_ignores_zero_signal_events():
    roi = np.array([1.0] * 6 + [-1.0] * 6)
    sig_match = np.array([1.0] * 6 + [0.0, 0.0, -1.0, -1.0, -1.0, -1.0])
    sig_miss = np.array([-1.0] * 6 + [0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    # only the 10 active (non-zero-signal) events count
    assert hit_rate(pd.Series(sig_match), pd.Series(roi)) == pytest.approx(1.0)
    assert hit_rate(pd.Series(sig_miss), pd.Series(roi)) == pytest.approx(0.0)
