import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FLOW_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "flow"
sys.path.insert(0, str(FLOW_DIR))

from wallet_filter_explorer import (  # noqa: E402
    VARIANT_FILTERS,
    apply_selection_filters,
    build_payload,
    load_selection,
    open_wallet_filter_explorer,
)

WALLETS = ["0xa", "0xb", "0xc", "0xd"]

_STATS = {
    "num_buckets": [10, 60, 80, 90],
    "trade_count": [100, 400, 500, 600],
    "total_notional": [1000, 6000, 7000, 8000],
    "total_pnl": [100, 5000, 6000, 7000],
    "copyable_pnl": [10, 1200, 1300, 1400],
    "num_markets": [5, 35, 40, 45],
    "market_pnl_hhi": [0.5, 0.1, 0.15, 0.12],
    "estimated_buy_sharpe": [1, 4, 5, 6],
    "estimated_copyable_buy_sharpe": [1, 4, 5, 6],
    "max_drawdown_to_pnl": [0.5, 0.1, 0.2, 0.15],
    "max_copyable_drawdown_to_copyable_pnl": [0.5, 0.2, 0.25, 0.1],
    "buy_pnl": [50, 1500, 2000, 2500],
    "buy_copyable_pnl": [10, 600, 700, 800],
    "buy_roi": [0.01, 0.06, 0.08, 0.09],
    "buy_copyable_roi": [0.01, 0.08, 0.09, 0.1],
    "copyable_pnl_similarity": [0.1, 0.5, 0.6, 0.7],
}


@pytest.fixture()
def stats():
    median_dt = pd.to_datetime(["2025-12-01"] * len(WALLETS), utc=True)
    return pd.DataFrame({"wallet": WALLETS, **_STATS, "median_dt": median_dt})


@pytest.fixture()
def fills():
    rows = []
    for wallet in WALLETS:
        for i in range(3):
            rows.append(
                {
                    "wallet": wallet,
                    "dt": pd.Timestamp("2025-11-01", tz="UTC") + pd.Timedelta(days=i),
                    "pnl": 1.0,
                    "copyable_pnl": 0.5,
                    "is_train": True,
                    "side": "BUY",
                    "price": 2.0,
                    "position": 1.0,
                    "notional": 2.0,
                    "copyable_qty_5m_100": 1.0,
                    "size_share": 0.1 + i * 0.1,
                }
            )
        for i in range(2):
            rows.append(
                {
                    "wallet": wallet,
                    "dt": pd.Timestamp("2025-12-20", tz="UTC") + pd.Timedelta(days=i),
                    "pnl": 2.0,
                    "copyable_pnl": 1.0,
                    "is_train": False,
                    "side": "BUY",
                    "price": 2.0,
                    "position": 1.0,
                    "notional": 2.0,
                    "copyable_qty_5m_100": 1.0,
                    "size_share": 0.4 + i * 0.1,
                }
            )
    df = pd.DataFrame(rows)
    df["last_condition_trade_ts"] = df["dt"]
    return df


@pytest.fixture()
def flow_fills():
    base = pd.Timestamp("2025-11-01", tz="UTC")
    rows = [
        # passes: in band, contrarian flow, uptick
        (0.20, -0.5, 1.0, 1.0, 10.0),
        # fails share
        (0.20, 0.5, 1.0, 1.0, 20.0),
        # outside price band but contrarian flow
        (0.80, -0.5, 1.0, 1.0, 30.0),
        # passes nonstrict (share == thr) but fails strict uptick
        (0.30, -0.1, 1.0, -1.0, 40.0),
    ]
    df = pd.DataFrame(
        {
            "wallet": "0xa",
            "dt": [base + pd.Timedelta(days=i) for i in range(len(rows))],
            "side": "BUY",
            "price": [r[0] for r in rows],
            "flow_own_usdc_60": [r[1] for r in rows],
            "usdc_all_60": [r[2] for r in rows],
            "price_chg_last_own": [r[3] for r in rows],
            "copyable_pnl": [r[4] for r in rows],
            "pnl": [r[4] for r in rows],
            "notional": 1.0,
            "position": 1.0,
            "copyable_qty_5m_100": 1.0,
            "is_train": True,
        }
    )
    df["last_condition_trade_ts"] = df["dt"]
    return df


def post_json(url, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=10)


def trace_names(fig):
    return [t.get("name", "") for t in fig["data"]]


def decode(values):
    if isinstance(values, dict) and "bdata" in values:
        return np.frombuffer(
            base64.b64decode(values["bdata"]), dtype=values.get("dtype", "f8")
        ).tolist()
    return values


def test_payload_structure(stats, fills):
    payload = build_payload(
        stats,
        variant="sports",
        eligible_wallets={"0xa", "0xb"},
        selected_wallets={"0xa"},
        fills=fills,
    )

    assert payload["variant"] == "sports"
    assert payload["pnlFilters"] == [
        {"key": "price", "label": "price band", "default": False},
        {"key": "size_share", "label": "size share", "default": False},
        {"key": "copyable", "label": "copyable qty > 0", "default": False},
    ]
    assert payload["filterDefaults"] == {
        "price_lo": 2.0,
        "price_hi": 2.0,
        "size_share_max": 0.2,
    }
    assert payload["flags"] == ["eligible", "strategy_selected", "test_active"]
    assert payload["presetsDropped"] == []
    assert payload["presets"] == VARIANT_FILTERS["sports"]
    assert payload["defaultAxes"] == {
        "x": "buy_roi",
        "y": "estimated_copyable_buy_sharpe",
    }

    keys = {c["key"] for c in payload["columns"]}
    assert {"buy_roi", "buy_copyable_roi", "median_dt_days_ago"} <= keys
    assert len(payload["wallets"]) == len(WALLETS)

    rec = {w["wallet"]: w for w in payload["wallets"]}
    assert rec["0xa"]["eligible"] is True
    assert rec["0xa"]["strategy_selected"] is True
    assert rec["0xa"]["test_active"] is True
    assert rec["0xb"]["strategy_selected"] is False
    assert rec["0xa"]["median_dt_days_ago"] > 0


def test_payload_normalizes_infinities(stats):
    stats.loc[0, "buy_roi"] = np.inf
    payload = build_payload(stats, variant="sports")
    rec = {w["wallet"]: w for w in payload["wallets"]}
    assert rec["0xa"]["buy_roi"] is None


def test_presets_skip_all_nan_columns(stats):
    stats["num_markets"] = np.nan
    payload = build_payload(stats, variant="sports")
    assert "num_markets" not in payload["presets"]
    assert "num_markets" in payload["presetsDropped"]


def test_payload_rejects_unknown_variant(stats):
    with pytest.raises(ValueError):
        build_payload(stats, variant="not-a-variant")


def test_build_payload_requires_wallet_column(stats):
    with pytest.raises(ValueError):
        build_payload(stats.drop(columns="wallet"))


def test_server_endpoints(stats, fills, tmp_path):
    handle = open_wallet_filter_explorer(
        stats, fills=fills, out_dir=tmp_path, open_browser=False
    )
    try:
        with urllib.request.urlopen(handle.url) as resp:
            page = resp.read().decode()
        assert "Wallet filter explorer" in page
        assert "__PLOTLYJS__" not in page and "__VARIANT__" not in page

        with urllib.request.urlopen(handle.url + "data") as resp:
            data = json.loads(resp.read())
        assert len(data["wallets"]) == len(WALLETS)

        with post_json(
            handle.url + "pnl",
            {"wallets": ["0xa"], "filters": {}, "exposure": False},
        ) as resp:
            panel = json.loads(resp.read())
        assert panel["label"] == "0xa"
        assert panel["wallets"] == 1
        assert decode(panel["train"]["data"][0]["y"]) == pytest.approx([1.0, 2.0, 3.0])
        assert decode(panel["test"]["data"][0]["y"]) == pytest.approx([2.0, 4.0])
        names = trace_names(panel["train"])
        assert "picked (wallet pnl, raw)" in names
        assert "picked (total)" not in names

        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(handle.url + "curves/0xa")
        assert exc.value.code == 404
    finally:
        handle.stop()


def test_pnl_aggregate(stats, fills, tmp_path):
    handle = open_wallet_filter_explorer(
        stats, fills=fills, out_dir=tmp_path, open_browser=False
    )
    try:
        with post_json(
            handle.url + "pnl",
            {"wallets": ["0xa", "0xb"], "filters": {}, "exposure": True},
        ) as resp:
            panel = json.loads(resp.read())
        assert panel["label"] == "picked (2)"
        train = panel["train"]
        raw = {t["name"]: decode(t["y"]) for t in train["data"]}
        assert raw["picked (wallet pnl, raw)"] == pytest.approx([2.0, 4.0, 6.0])
        assert raw["picked (copyable pnl, raw)"] == pytest.approx([1.0, 2.0, 3.0])
        names = trace_names(train)
        assert "picked (total)" not in names
        exposures = [t for t in train["data"] if t.get("yaxis") == "y2"]
        assert exposures and all("exposure" in t["name"] for t in exposures)
    finally:
        handle.stop()


def test_pnl_checkable_fill_filters(stats, fills, tmp_path):
    handle = open_wallet_filter_explorer(
        stats, fills=fills, out_dir=tmp_path, open_browser=False
    )
    base = {"wallets": ["0xa"], "exposure": False}
    try:
        with post_json(handle.url + "pnl", dict(base, filters={})) as resp:
            panel = json.loads(resp.read())
        raw = {t["name"]: decode(t["y"]) for t in panel["train"]["data"]}
        assert raw["picked (wallet pnl, raw)"] == pytest.approx([1.0, 2.0, 3.0])
        assert raw["picked (copyable pnl, raw)"] == pytest.approx([0.5, 1.0, 1.5])
        assert panel["filters"] == []

        with post_json(
            handle.url + "pnl",
            dict(base, filters={"size_share": {"max": 0.2}}),
        ) as resp:
            panel = json.loads(resp.read())
        raw = {t["name"]: decode(t["y"]) for t in panel["train"]["data"]}
        assert raw["picked (wallet pnl (filtered), raw)"] == pytest.approx([1.0, 2.0])
        assert raw["picked (copyable pnl (filtered), raw)"] == pytest.approx([0.5, 1.0])
        assert panel["filters"] == ["size_share"]
    finally:
        handle.stop()


def test_pnl_checkable_flow_filter(stats, flow_fills, tmp_path):
    handle = open_wallet_filter_explorer(
        stats, fills=flow_fills, out_dir=tmp_path, open_browser=False
    )
    base = {"wallets": ["0xa"], "exposure": False}
    try:
        with post_json(
            handle.url + "pnl",
            dict(base, filters={"flow": {"share_thr": -0.1}}),
        ) as resp:
            panel = json.loads(resp.read())
        raw = {t["name"]: decode(t["y"]) for t in panel["train"]["data"]}
        assert raw["picked (copyable pnl (filtered), raw)"] == pytest.approx(
            [10.0, 40.0, 80.0]
        )

        with post_json(
            handle.url + "pnl",
            dict(base, filters={"flow": {"share_thr": -0.1, "strict": True}}),
        ) as resp:
            panel = json.loads(resp.read())
        raw = {t["name"]: decode(t["y"]) for t in panel["train"]["data"]}
        assert raw["picked (copyable pnl (filtered), raw)"] == pytest.approx(
            [10.0, 40.0]
        )
    finally:
        handle.stop()


def test_payload_flow_controls(stats, flow_fills):
    payload = build_payload(
        stats,
        fills=flow_fills,
        flow_price_lo=0.25,
        flow_price_hi=0.45,
        flow_share_thr=-0.05,
        flow_strict=True,
    )
    assert payload["flowAvailable"] is True
    assert payload["flowDefaults"] == {
        "price_lo": 0.25,
        "price_hi": 0.45,
        "share_thr": -0.05,
        "strict": True,
    }
    assert "flowCarveFracs" not in payload
    assert payload["pnlFilters"] == [
        {"key": "price", "label": "price band", "default": False},
        {"key": "flow", "label": "flow share", "default": False},
        {"key": "copyable", "label": "copyable qty > 0", "default": False},
    ]
    assert payload["filterDefaults"] == {
        "price_lo": 0.2,
        "price_hi": 0.8,
        "flow_share_max": -0.05,
        "flow_strict": True,
    }

    plain = build_payload(
        stats,
        fills=flow_fills.drop(
            columns=["flow_own_usdc_60", "usdc_all_60", "price_chg_last_own"]
        ),
    )
    assert plain["flowAvailable"] is False
    assert [f["key"] for f in plain["pnlFilters"]] == ["price", "copyable"]
    assert plain["filterDefaults"] == {"price_lo": 0.2, "price_hi": 0.8}


def test_pnl_rejects_bad_requests(stats, fills, tmp_path):
    handle = open_wallet_filter_explorer(
        stats, fills=fills, out_dir=tmp_path, open_browser=False
    )
    try:
        for payload, match in [
            ({"wallets": [], "filters": {}}, "wallets"),
            ({"wallets": "0xa", "filters": {}}, "wallets"),
            ({"wallets": ["0xa"], "filters": "x"}, "filters must be an object"),
            ({"wallets": ["0xa"], "filters": {"price": "x"}},
             "price filter must be an object"),
            ({"wallets": ["0xa"], "filters": {"price": {"lo": 2, "hi": 1}}},
             "price lo"),
            ({"wallets": ["0xa"], "filters": {"size_share": {"max": "x"}}},
             "size_share"),
            ({"wallets": ["0xa"], "filters": {"flow": {"share_thr": "x"}}},
             "flow threshold"),
        ]:
            with pytest.raises(urllib.error.HTTPError) as exc:
                post_json(handle.url + "pnl", payload)
            assert exc.value.code == 400
            assert match in exc.value.read().decode()
    finally:
        handle.stop()


def test_pnl_disabled_without_fills(stats, tmp_path):
    handle = open_wallet_filter_explorer(
        stats, out_dir=tmp_path, open_browser=False
    )
    try:
        with urllib.request.urlopen(handle.url + "data") as resp:
            data = json.loads(resp.read())
        assert data["pnlFilters"] == []
        assert data["filterDefaults"] == {}
        with pytest.raises(urllib.error.HTTPError) as exc:
            post_json(handle.url + "pnl", {"wallets": ["0xa"], "filters": {}})
        assert exc.value.code == 400
    finally:
        handle.stop()


def test_selection_roundtrip(stats, fills, tmp_path):
    handle = open_wallet_filter_explorer(
        stats, fills=fills, out_dir=tmp_path, open_browser=False
    )
    selection = {
        "variant": "sports",
        "filters": {
            "buy_roi": {"min": 0.05, "max": None, "nanOk": False},
        },
        "membership": {
            "eligible": True,
            "strategy_selected": False,
            "test_active": False,
        },
        "picked": ["0xb"],
        "passing": ["0xb", "0xc"],
        "counts": {"pool": 4, "passing": 2, "picked": 1},
        "flow": {
            "price_lo": 0.2,
            "price_hi": 0.5,
            "share_thr": -0.1,
            "strict": False,
        },
    }
    try:
        req = urllib.request.Request(
            handle.url + "selection",
            data=json.dumps(selection).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert json.loads(resp.read())["ok"] is True

        assert handle.get_selection() == selection
        assert load_selection(handle.selection_path) == selection
        assert handle.selection_path.name == "wallet_filter_selection_sports.json"

        df = apply_selection_filters(selection, stats, fills=fills)
        assert list(df["wallet"]) == ["0xb", "0xc"]
        assert list(df["picked"]) == [True, False]

        mismatched = dict(selection, passing=["0xa"])
        with pytest.raises(ValueError, match="do not match|does not match"):
            apply_selection_filters(mismatched, stats, fills=fills)
    finally:
        handle.stop()


def test_load_selection_missing(tmp_path):
    assert load_selection(tmp_path / "nope.json") is None
