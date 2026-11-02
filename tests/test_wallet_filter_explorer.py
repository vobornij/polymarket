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
def frames():
    median_dt = pd.to_datetime(["2025-12-01"] * len(WALLETS), utc=True)
    full = pd.DataFrame({"wallet": WALLETS, **_STATS, "median_dt": median_dt})
    carved = full.copy()
    carved["total_pnl"] = carved["total_pnl"] * 0.9
    carved["buy_roi"] = carved["buy_roi"] * 1.05
    return full, carved


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
                    "pnl_limited": 0.8,
                    "copyable_pnl": 0.5,
                    "copyable_pnl_limited": 0.4,
                    "is_train": True,
                    "side": "BUY",
                    "price": 2.0,
                    "position": 1.0,
                    "notional": 2.0,
                    "deployed_capital_frac": 1.0,
                    "copyable_qty_5m_100": 1.0,
                }
            )
        for i in range(2):
            rows.append(
                {
                    "wallet": wallet,
                    "dt": pd.Timestamp("2025-12-20", tz="UTC") + pd.Timedelta(days=i),
                    "pnl": 2.0,
                    "pnl_limited": 1.6,
                    "copyable_pnl": 1.0,
                    "copyable_pnl_limited": 0.8,
                    "is_train": False,
                    "side": "BUY",
                    "price": 2.0,
                    "position": 1.0,
                    "notional": 2.0,
                    "deployed_capital_frac": 1.0,
                    "copyable_qty_5m_100": 1.0,
                }
            )
    df = pd.DataFrame(rows)
    df["last_condition_trade_ts"] = df["dt"]
    df["is_carved"] = True
    for wallet in WALLETS:
        rows_of = df.index[df["wallet"] == wallet]
        df.loc[rows_of[2], "is_carved"] = False
        df.loc[rows_of[-1], "is_carved"] = False
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


def test_payload_structure(frames, fills):
    full, carved = frames
    payload = build_payload(
        full,
        carved,
        variant="sports",
        eligible_wallets={"0xa", "0xb"},
        selected_wallets={"0xa"},
        fills=fills,
    )

    assert payload["variant"] == "sports"
    assert payload["pnlVariants"] == [
        {"key": "pnl__full", "label": "wallet pnl (full)", "default": True},
        {"key": "pnl__carved", "label": "wallet pnl (carved)", "default": True},
        {"key": "pnl_limited__full", "label": "wallet pnl (capped) (full)", "default": False},
        {"key": "pnl_limited__carved", "label": "wallet pnl (capped) (carved)", "default": False},
        {"key": "copyable_pnl__full", "label": "copyable pnl (full)", "default": True},
        {"key": "copyable_pnl__carved", "label": "copyable pnl (carved)", "default": True},
        {"key": "copyable_pnl_limited__full", "label": "copyable pnl (capped) (full)", "default": False},
        {"key": "copyable_pnl_limited__carved", "label": "copyable pnl (capped) (carved)", "default": False},
    ]
    assert payload["flags"] == ["eligible", "strategy_selected", "test_active"]
    assert payload["presetsDropped"] == []
    assert payload["presets"] == VARIANT_FILTERS["sports"]
    assert payload["defaultAxes"] == {
        "x": "buy_roi__full",
        "y": "estimated_copyable_buy_sharpe__carved",
    }

    keys = {c["key"] for c in payload["columns"]}
    assert {"buy_roi__full", "buy_roi__carved", "median_dt_days_ago__full"} <= keys
    assert len(payload["wallets"]) == len(WALLETS)

    rec = {w["wallet"]: w for w in payload["wallets"]}
    assert rec["0xa"]["eligible"] is True
    assert rec["0xa"]["strategy_selected"] is True
    assert rec["0xa"]["test_active"] is True
    assert rec["0xb"]["strategy_selected"] is False
    assert rec["0xa"]["median_dt_days_ago__full"] > 0


def test_payload_normalizes_infinities(frames):
    full, carved = frames
    full.loc[0, "buy_roi"] = np.inf
    payload = build_payload(full, carved, variant="sports")
    rec = {w["wallet"]: w for w in payload["wallets"]}
    assert rec["0xa"]["buy_roi__full"] is None


def test_presets_skip_all_nan_columns(frames):
    full, carved = frames
    full["num_markets"] = np.nan
    payload = build_payload(full, carved, variant="sports")
    assert "num_markets__full" not in payload["presets"]
    assert "num_markets__full" in payload["presetsDropped"]


def test_payload_rejects_unknown_variant(frames):
    full, carved = frames
    with pytest.raises(ValueError):
        build_payload(full, carved, variant="not-a-variant")


def test_build_payload_requires_wallet_column(frames):
    full, carved = frames
    with pytest.raises(ValueError):
        build_payload(full.drop(columns="wallet"), carved)


def test_server_endpoints(frames, fills, tmp_path):
    full, carved = frames
    handle = open_wallet_filter_explorer(
        full, carved, fills=fills, out_dir=tmp_path, open_browser=False
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
            {"wallets": ["0xa"], "variants": ["pnl__full"], "exposure": False,
             "carve": False},
        ) as resp:
            panel = json.loads(resp.read())
        assert panel["label"] == "0xa"
        assert panel["wallets"] == 1
        assert decode(panel["train"]["data"][0]["y"]) == pytest.approx([1.0, 2.0, 3.0])
        assert decode(panel["test"]["data"][0]["y"]) == pytest.approx([2.0, 4.0])
        names = trace_names(panel["train"])
        assert "picked (wallet pnl (full), raw)" in names
        assert "picked (total)" not in names

        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(handle.url + "curves/0xa")
        assert exc.value.code == 404
    finally:
        handle.stop()


def test_pnl_aggregate_and_variants(frames, fills, tmp_path):
    full, carved = frames
    handle = open_wallet_filter_explorer(
        full, carved, fills=fills, exposure_fracs=(0.5, 1.0),
        out_dir=tmp_path, open_browser=False
    )
    try:
        with post_json(
            handle.url + "pnl",
            {"wallets": ["0xa", "0xb"],
             "variants": ["pnl__full", "pnl__carved",
                          "copyable_pnl__full", "copyable_pnl__carved"],
             "exposure": True, "carve": True},
        ) as resp:
            panel = json.loads(resp.read())
        assert panel["label"] == "picked (2)"
        train = panel["train"]
        raw = {t["name"]: decode(t["y"]) for t in train["data"]}
        assert raw["picked (wallet pnl (full), raw)"] == pytest.approx([2.0, 4.0, 6.0])
        assert raw["picked (wallet pnl (carved), raw)"] == pytest.approx([2.0, 4.0, 4.0])
        assert raw["picked (copyable pnl (full), raw)"] == pytest.approx([1.0, 2.0, 3.0])
        assert raw["picked (copyable pnl (carved), raw)"] == pytest.approx([1.0, 2.0, 2.0])
        names = trace_names(train)
        assert "picked (total)" not in names
        assert "picked (wallet pnl (full), frac 0.5)" in names
        assert "picked (copyable pnl (carved), frac 0.5)" in names
        exposures = [t for t in train["data"] if t.get("yaxis") == "y2"]
        assert exposures and all("exposure" in t["name"] for t in exposures)
    finally:
        handle.stop()


def test_pnl_rejects_bad_requests(frames, fills, tmp_path):
    full, carved = frames
    handle = open_wallet_filter_explorer(
        full, carved, fills=fills, out_dir=tmp_path, open_browser=False
    )
    try:
        for payload, match in [
            ({"wallets": [], "variants": ["pnl__full"]}, "wallets"),
            ({"wallets": ["0xa"], "variants": []}, "variants"),
            ({"wallets": ["0xa"], "variants": ["nope"]}, "unknown"),
            ({"wallets": ["0xa"], "variants": ["pnl"]}, "unknown"),
            ({"wallets": "0xa", "variants": ["pnl__full"]}, "wallets"),
        ]:
            with pytest.raises(urllib.error.HTTPError) as exc:
                post_json(handle.url + "pnl", payload)
            assert exc.value.code == 400
            assert match in exc.value.read().decode()
    finally:
        handle.stop()


def test_pnl_full_only_without_is_carved(frames, fills, tmp_path):
    full, carved = frames
    handle = open_wallet_filter_explorer(
        full, carved, fills=fills.drop(columns="is_carved"),
        out_dir=tmp_path, open_browser=False
    )
    try:
        with urllib.request.urlopen(handle.url + "data") as resp:
            data = json.loads(resp.read())
        assert [v["key"] for v in data["pnlVariants"]] == [
            "pnl__full", "pnl_limited__full",
            "copyable_pnl__full", "copyable_pnl_limited__full",
        ]
        with post_json(
            handle.url + "pnl",
            {"wallets": ["0xa"], "variants": ["pnl__full"]},
        ) as resp:
            panel = json.loads(resp.read())
        assert decode(panel["train"]["data"][0]["y"]) == pytest.approx([1.0, 2.0, 3.0])
        with pytest.raises(urllib.error.HTTPError) as exc:
            post_json(
                handle.url + "pnl",
                {"wallets": ["0xa"], "variants": ["pnl__carved"]},
            )
        assert exc.value.code == 400
    finally:
        handle.stop()


def test_pnl_disabled_without_fills(frames, tmp_path):
    full, carved = frames
    handle = open_wallet_filter_explorer(
        full, carved, out_dir=tmp_path, open_browser=False
    )
    try:
        with urllib.request.urlopen(handle.url + "data") as resp:
            data = json.loads(resp.read())
        assert data["pnlVariants"] == []
        with pytest.raises(urllib.error.HTTPError) as exc:
            post_json(handle.url + "pnl", {"wallets": ["0xa"], "variants": ["pnl"]})
        assert exc.value.code == 400
    finally:
        handle.stop()


def test_selection_roundtrip(frames, fills, tmp_path):
    full, carved = frames
    handle = open_wallet_filter_explorer(
        full, carved, fills=fills, out_dir=tmp_path, open_browser=False
    )
    selection = {
        "variant": "sports",
        "filters": {
            "buy_roi__full": {"min": 0.05, "max": None, "nanOk": False},
        },
        "membership": {
            "eligible": True,
            "strategy_selected": False,
            "test_active": False,
        },
        "picked": ["0xb"],
        "passing": ["0xb", "0xc"],
        "counts": {"pool": 4, "passing": 2, "picked": 1},
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

        df = apply_selection_filters(selection, full, carved, fills=fills)
        assert list(df["wallet"]) == ["0xb", "0xc"]
        assert list(df["picked"]) == [True, False]

        mismatched = dict(selection, passing=["0xa"])
        with pytest.raises(ValueError, match="do not match|does not match"):
            apply_selection_filters(mismatched, full, carved, fills=fills)
    finally:
        handle.stop()


def test_load_selection_missing(tmp_path):
    assert load_selection(tmp_path / "nope.json") is None
