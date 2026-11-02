"""Interactive wallet-filter explorer for ``stage1_wallet_strategy_selection``.

Serves a localhost web page for fine-grained wallet filtering on the two
stage-1 stat frames:

* ``full_stats``   — the uncarved BUY-train wallet stats (``wallet_vol_train_full``)
* ``carved_stats`` — the exposure-carved BUY-train wallet stats (``wallet_vol_train``)

Features
--------
* Range sliders for every numeric stat column of BOTH frames, pre-set to the
  strategy-selection variant thresholds (mirrored from
  ``notebooks/wallet_selection/signal_lab/filters.py``; strict ``>`` / ``<``
  comparisons are approximated as inclusive bounds).
* Membership checkboxes (strategy-eligible / strategy-selected / test-active).
* Scatter plot with selectable X / Y (any column, log toggles) and color modes;
  click a wallet to pick it and show its detail, lasso to pick groups.
* PnL panels (train and test in two separate graphs) overlaying the **full**
  (pre-carve) and **carved** (``is_carved``) streams: wallet PnL, capped
  (``pnl_limited`` / ``copyable_pnl_limited``), copyable PnL, exposure lines on
  a secondary axis, and extra bankroll-discipline carved fracs (train-fit
  ``wallet_capture_thresholds``). Shows either the clicked wallet or the
  aggregate of all picked wallets; variants/exposures/carve are toggles.
* Sortable table of the wallets passing the current filters, with
  "pick passing" (add every passing wallet) and shift+click range selection.
* "Save selection" POSTs the current filter config + picked wallets back to
  Python, which writes ``wallet_filter_selection_{variant}.json`` into
  ``out_dir`` — reload it in the notebook with :func:`load_selection` and
  re-derive the passing set with :func:`apply_selection_filters`.

Notebook usage (dedicated cell in ``notebooks/flow/stage1_wallet_strategy_selection.ipynb``)::

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))  # notebooks/flow

    from wallet_filter_explorer import open_wallet_filter_explorer

    explorer = open_wallet_filter_explorer(
        wallet_vol_train_full,
        wallet_vol_train,
        variant=variant,
        eligible_wallets=eligible_wallets,
        selected_wallets=copyable_wallets,
        fills=df_full_buy_full,  # pre-carve BUY stream with `is_carved`
        split_date=TRAIN_END,
        exposure_fracs=tuple(
            f for f in EXPOSURE_FRACS if f != EXPOSURE_FILTER_FRAC
        ),
        out_dir=WORKSPACE_DIR,
    )

No third-party dependencies beyond pandas/numpy/plotly (plotly.min.js is
inlined into the served page, so the page works fully offline).
"""

from __future__ import annotations

import gzip
import json
import threading
import urllib.parse
import webbrowser
from collections import OrderedDict
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.offline as _plotly_offline

from polymarket_analysis.visualization.wallet_plots import plot_wallet_selection_pnl

__all__ = [
    "ExplorerHandle",
    "VARIANT_FILTERS",
    "apply_selection_filters",
    "build_payload",
    "load_selection",
    "open_wallet_filter_explorer",
]

# ---------------------------------------------------------------------------
# Variant threshold presets (slider initial values)
#
# Mirrors the inlined rule constants in notebooks/wallet_selection/
# signal_lab/filters.py (_weather/_politics/_sports_strategy_masks).
# Strict comparisons (> / <) are approximated as inclusive bounds.
# Key suffix: ``__full`` = uncarved stats, ``__carved`` = exposure-carved
# stats — matching which frame each mask rule evaluates on.
# ---------------------------------------------------------------------------

VARIANT_FILTERS: dict[str, dict[str, dict[str, float | None]]] = {
    "sports": {
        # base rules — full (uncarved) stats
        "buy_roi__full": {"min": 0.04},
        "buy_pnl__full": {"min": 1000},
        "total_pnl__full": {"min": 4000},
        "num_markets__full": {"min": 30},
        "num_buckets__full": {"min": 50},
        "market_pnl_hhi__full": {"max": 0.2},
        "estimated_buy_sharpe__full": {"min": 3},
        "max_drawdown_to_pnl__full": {"max": 0.2},
        # carved rules — exposure-carved stats
        "buy_roi__carved": {"min": 0.05},
        "total_notional__carved": {"min": 5000},
        "buy_copyable_pnl__carved": {"min": 500},
        "buy_copyable_roi__carved": {"min": 0.07},
        "max_copyable_drawdown_to_copyable_pnl__carved": {"max": 0.3},
    },
    "politics": {
        # base + copyable rules — full stats
        "total_pnl__full": {"min": 2000},
        "buy_roi__full": {"min": 0.07},
        "copyable_pnl__full": {"min": 1000},
        "trade_count__full": {"min": 300},
        "num_buckets__full": {"min": 50},
        "estimated_buy_sharpe__full": {"min": 3},
        "market_pnl_hhi__full": {"max": 0.2},
        "max_drawdown_to_pnl__full": {"max": 0.3},
        "median_dt_days_ago__full": {"min": 90},
        "total_notional__full": {"min": 5000},
        "copyable_pnl_similarity__full": {"min": 0.10},
        "buy_copyable_pnl__full": {"min": 1000},
        "buy_copyable_roi__full": {"min": 0.05},
        "estimated_copyable_buy_sharpe__full": {"min": 3},
    },
    "weather": {
        # base + copyable rules — full stats
        "buy_roi__full": {"min": 0.04},
        "buy_pnl__full": {"min": 1000},
        "num_markets__full": {"min": 20},
        "num_buckets__full": {"min": 50},
        "max_drawdown_to_pnl__full": {"max": 0.3},
        "total_notional__full": {"min": 5000},
        "estimated_buy_sharpe__full": {"min": 3},
        "buy_copyable_pnl__full": {"min": 500},
        "buy_copyable_roi__full": {"min": 0.04},
        "estimated_copyable_buy_sharpe__full": {"min": 4},
        "max_copyable_drawdown_to_copyable_pnl__full": {"max": 0.2},
        "copyable_pnl_similarity__full": {"min": 0.40},
    },
}

# ---------------------------------------------------------------------------
# Stat-frame merging and payload building
# ---------------------------------------------------------------------------


def _frame_columns(stats: pd.DataFrame) -> list[str]:
    """Numeric stat columns of one frame (wallet excluded, median_dt special)."""
    cols: list[str] = []
    for c in stats.columns:
        if c in ("wallet", "median_dt"):
            continue
        if pd.api.types.is_numeric_dtype(stats[c]):
            cols.append(c)
    return cols


def _merged_wallet_frame(
    full_stats: pd.DataFrame,
    carved_stats: pd.DataFrame,
    *,
    eligible_wallets: set[str] | None,
    selected_wallets: set[str] | None,
    test_active_wallets: set[str] | None,
    ref_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Outer-merge both stat frames per wallet into ``<col>__full/__carved`` keys.

    ``median_dt`` is converted to ``median_dt_days_ago__<frame>`` (days before
    ``ref_ts``). Membership flag columns are added only for the provided sets.
    """
    merged: pd.DataFrame | None = None
    for frame, stats in (("full", full_stats), ("carved", carved_stats)):
        if "wallet" not in stats.columns:
            raise ValueError(f"{frame} stats frame is missing the 'wallet' column")
        if stats["wallet"].duplicated().any():
            raise ValueError(f"{frame} stats frame has duplicate wallets")

        cols = ["wallet", *_frame_columns(stats)]
        sub = stats.loc[:, cols].copy()
        if "median_dt" in stats.columns:
            median_dt = pd.to_datetime(stats["median_dt"], utc=True, errors="coerce")
            sub["median_dt_days_ago"] = (ref_ts - median_dt).dt.total_seconds() / 86400.0
        sub = sub.rename(
            columns={c: f"{c}__{frame}" for c in sub.columns if c != "wallet"}
        )
        merged = sub if merged is None else merged.merge(sub, on="wallet", how="outer")

    if eligible_wallets is not None:
        merged["eligible"] = merged["wallet"].isin(eligible_wallets)
    if selected_wallets is not None:
        merged["strategy_selected"] = merged["wallet"].isin(selected_wallets)
    if test_active_wallets is not None:
        merged["test_active"] = merged["wallet"].isin(test_active_wallets)

    # to_json emits invalid JSON for infinities; normalize them to NaN first.
    num_cols = [c for c in merged.columns if c != "wallet"]
    merged[num_cols] = merged[num_cols].replace([np.inf, -np.inf], np.nan)
    return merged


def _column_meta(merged: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-column metadata for the page (range, NaN count, integral flag)."""
    columns: list[dict[str, Any]] = []
    flag_cols = {"eligible", "strategy_selected", "test_active"}
    for c in merged.columns:
        if c == "wallet" or c in flag_cols:
            continue
        if c.endswith("__full"):
            frame = "full"
        elif c.endswith("__carved"):
            frame = "carved"
        else:
            continue
        s = pd.to_numeric(merged[c], errors="coerce")
        finite = s.dropna()
        integral = bool(
            len(finite)
            and np.equal(np.mod(finite.to_numpy(dtype=float), 1), 0).all()
        )
        columns.append(
            {
                "key": c,
                "name": c[: -len(f"__{frame}")],
                "frame": frame,
                "min": float(finite.min()) if len(finite) else None,
                "max": float(finite.max()) if len(finite) else None,
                "nanCount": int(s.isna().sum()),
                "integral": integral,
            }
        )
    return columns


def build_payload(
    full_stats: pd.DataFrame,
    carved_stats: pd.DataFrame,
    *,
    variant: str = "sports",
    eligible_wallets: set[str] | None = None,
    selected_wallets: set[str] | None = None,
    fills: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build the JSON payload served to the page at ``GET /data``.

    ``fills`` (the carved stream with ``is_train``) enables the test-active
    flag and the train/test PnL panels.
    """
    if variant not in VARIANT_FILTERS:
        raise ValueError(
            f"Unknown variant {variant!r}; expected one of {sorted(VARIANT_FILTERS)}"
        )

    test_active_wallets: set[str] | None = None
    if fills is not None:
        missing = [c for c in ("wallet", "dt", "is_train") if c not in fills.columns]
        if missing:
            raise ValueError(f"fills is missing required columns {missing}")
        is_train = fills["is_train"].astype(bool)
        test_active_wallets = set(fills.loc[~is_train, "wallet"].unique())
        ref_ts = pd.to_datetime(fills["dt"], utc=True).max()
    else:
        ref_ts = pd.Timestamp.now(tz="UTC")

    merged = _merged_wallet_frame(
        full_stats,
        carved_stats,
        eligible_wallets=eligible_wallets,
        selected_wallets=selected_wallets,
        test_active_wallets=test_active_wallets,
        ref_ts=ref_ts,
    )

    columns = _column_meta(merged)
    if not columns:
        raise ValueError("No numeric stat columns found in the input frames")

    # All-NaN columns cannot be constrained with sliders; keep them out of the
    # presets so a preset never filters on a column the page cannot show.
    keys = {c["key"] for c in columns if c["min"] is not None}
    presets = {k: v for k, v in VARIANT_FILTERS[variant].items() if k in keys}
    presets_dropped = [k for k in VARIANT_FILTERS[variant] if k not in keys]

    flags = [
        f
        for f, present in (
            ("eligible", eligible_wallets is not None),
            ("strategy_selected", selected_wallets is not None),
            ("test_active", test_active_wallets is not None),
        )
        if present
    ]

    default_axes = {
        "x": "buy_roi__full" if "buy_roi__full" in keys else columns[0]["key"],
        "y": (
            "estimated_copyable_buy_sharpe__carved"
            if "estimated_copyable_buy_sharpe__carved" in keys
            else columns[min(1, len(columns) - 1)]["key"]
        ),
    }

    variants = _available_pnl_variants(fills) if fills is not None else []
    return {
        "variant": variant,
        "flags": flags,
        "columns": columns,
        "presets": presets,
        "presetsDropped": presets_dropped,
        "pnlVariants": [
            {
                "key": c,
                "label": _variant_label(c),
                "default": _split_variant(c)[0] in ("pnl", "copyable_pnl"),
            }
            for c in variants
        ],
        "defaultAxes": default_axes,
        # to_json maps NaN -> null; booleans stay booleans.
        "wallets": json.loads(merged.to_json(orient="records")),
    }


# ---------------------------------------------------------------------------
# PnL / exposure panels (reuse the notebook's cohort plotting function)
# ---------------------------------------------------------------------------

_PNL_BASE_COLUMNS = (
    "wallet",
    "dt",
    "side",
    "last_condition_trade_ts",
    "position",
    "price",
    "notional",
    "deployed_capital_frac",
    "pnl",
    "is_train",
)
_PNL_BASE_VARIANTS = ("pnl", "pnl_limited", "copyable_pnl", "copyable_pnl_limited")
_PNL_STREAMS = ("full", "carved")
_PNL_VARIANT_LABELS = {
    "pnl": "wallet pnl",
    "pnl_limited": "wallet pnl (capped)",
    "copyable_pnl": "copyable pnl",
    "copyable_pnl_limited": "copyable pnl (capped)",
}


def _split_variant(key: str) -> tuple[str, str | None]:
    """``copyable_pnl__full`` → ``("copyable_pnl", "full")``."""
    base, sep, stream = key.rpartition("__")
    if sep and stream in _PNL_STREAMS:
        return base, stream
    return key, None


def _variant_label(key: str) -> str:
    base, stream = _split_variant(key)
    label = _PNL_VARIANT_LABELS.get(base, base)
    return f"{label} ({stream})" if stream else label


def _copyable_qty_col(fills: pd.DataFrame, pnl_col: str) -> str | None:
    """qty column whose price × qty is the copy exposure of ``pnl_col``."""
    if pnl_col == "copyable_pnl" or pnl_col == "copyable_pnl_limited":
        return "copyable_qty_5m_100"
    if pnl_col.startswith("copyable_pnl_"):
        return f"copyable_qty_{pnl_col[len('copyable_pnl_'):]}"
    return None


def _variant_available(fills: pd.DataFrame, pnl_col: str) -> bool:
    if pnl_col == "pnl":
        return "notional" in fills.columns
    if pnl_col == "pnl_limited":
        return {"notional", "deployed_capital_frac"} <= set(fills.columns)
    qty_col = _copyable_qty_col(fills, pnl_col)
    if qty_col is None or qty_col not in fills.columns or "price" not in fills.columns:
        return False
    if pnl_col == "copyable_pnl_limited":
        return "deployed_capital_frac" in fills.columns
    return True


def _available_pnl_variants(fills: pd.DataFrame) -> list[str]:
    """Base PnL variants × streams, where exposure can be derived.

    ``fills`` is the full (pre-carve) BUY stream; when it carries an
    ``is_carved`` flag every variant is offered for both the full and the
    exposure-carved stream. Extra ``copyable_pnl_*`` columns are not offered.
    """
    bases = [
        c
        for c in _PNL_BASE_VARIANTS
        if c in fills.columns and _variant_available(fills, c)
    ]
    streams = _PNL_STREAMS if "is_carved" in fills.columns else _PNL_STREAMS[:1]
    return [f"{base}__{stream}" for base in bases for stream in streams]


def _prepare_pnl_fills(fills: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    """Fill frame with ``{variant}_exposure`` columns for each full/carved key."""
    required = ("wallet", "dt", "side", "last_condition_trade_ts", "position", "price")
    missing = [c for c in required if c not in fills.columns]
    if missing:
        raise ValueError(f"fills is missing required columns {missing}")

    bases: list[str] = []
    streams: set[str] = set()
    for key in variants:
        base, stream = _split_variant(key)
        if stream is None:
            raise ValueError(f"PnL variant {key!r} needs a __full/__carved suffix")
        if base not in _PNL_BASE_VARIANTS:
            raise ValueError(f"unsupported PnL variant {key!r}")
        if base not in bases:
            bases.append(base)
        streams.add(stream)
    if "carved" in streams and "is_carved" not in fills.columns:
        raise ValueError("carved PnL variants require an 'is_carved' column")

    need = list(_PNL_BASE_COLUMNS)
    if "carved" in streams:
        need.append("is_carved")
    for base in bases:
        if base not in need:
            need.append(base)
        qty_col = _copyable_qty_col(fills, base)
        if qty_col and qty_col not in need and qty_col in fills.columns:
            need.append(qty_col)
    df = fills.loc[:, [c for c in need if c in fills.columns]].copy()

    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    df["last_condition_trade_ts"] = pd.to_datetime(
        df["last_condition_trade_ts"], utc=True
    )
    df["trade_pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    price = pd.to_numeric(df["price"], errors="coerce")
    notional = pd.to_numeric(df["notional"], errors="coerce") if "notional" in df else None
    frac = (
        pd.to_numeric(df["deployed_capital_frac"], errors="coerce")
        if "deployed_capital_frac" in df
        else None
    )
    carved = (
        df["is_carved"].fillna(False).astype(bool)
        if "is_carved" in df
        else None
    )

    for base in bases:
        values = pd.to_numeric(df[base], errors="coerce")
        if base == "pnl":
            exposure = notional
        elif base == "pnl_limited":
            exposure = notional * frac
        else:
            qty = pd.to_numeric(df[_copyable_qty_col(fills, base)], errors="coerce")
            exposure = price * qty.where(qty > 0)
            if base == "copyable_pnl_limited":
                exposure = exposure * frac
        for stream in _PNL_STREAMS:
            if stream not in streams:
                continue
            key = f"{base}__{stream}"
            if stream == "carved":
                df[key] = values.where(carved, 0.0)
                df[f"{key}_exposure"] = exposure.where(carved, 0.0)
            else:
                df[key] = values
                df[f"{key}_exposure"] = exposure
    return df


def _resolve_split_date(fills: pd.DataFrame, split_date: Any) -> pd.Timestamp:
    if split_date is not None:
        ts = pd.Timestamp(split_date)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts
    if "is_train" in fills.columns:
        train_ts = pd.to_datetime(
            fills.loc[fills["is_train"].astype(bool), "last_condition_trade_ts"],
            utc=True,
            errors="coerce",
        )
        if train_ts.notna().any():
            return train_ts.max()
    return pd.Timestamp.now(tz="UTC")


class _PnlPanel:
    """Builds (and caches) the train/test PnL figures for a wallet set."""

    def __init__(
        self,
        fills: pd.DataFrame,
        *,
        split_date: pd.Timestamp,
        exposure_fracs: Iterable[float],
        max_cache: int = 16,
    ) -> None:
        self.fills = fills
        self.split_date = split_date
        self.exposure_fracs = tuple(float(f) for f in exposure_fracs)
        self.max_cache = max_cache
        self._cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.fit_col = (
            "copyable_pnl__full"
            if "copyable_pnl__full" in fills.columns
            else None
        )

    def figures(
        self,
        wallets: list[str],
        pnl_cols: list[str],
        *,
        exposure: bool,
        carve: bool,
    ) -> dict[str, Any]:
        key = (tuple(sorted(wallets)), tuple(pnl_cols), bool(exposure), bool(carve))
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        cohort = pd.DataFrame({"wallet": list(wallets)})
        sub = self.fills[self.fills["wallet"].isin(wallets)]
        fracs = self.exposure_fracs if carve else ()
        label = wallets[0] if len(wallets) == 1 else f"picked ({len(wallets)})"
        labels = {key: _variant_label(key) for key in pnl_cols}
        fit_col = self.fit_col or pnl_cols[0]
        out: dict[str, Any] = {"label": label, "wallets": len(wallets)}
        for period in ("train", "test"):
            fig = plot_wallet_selection_pnl(
                sub,
                {"picked": cohort},
                split_date=self.split_date,
                period=period,
                title=f"{period.title()} — {label}",
                pnl_cols=list(pnl_cols),
                plot_exposure=exposure,
                exposure_fracs=fracs,
                carve_fit_col=fit_col,
                plot_total=False,
                pnl_col_labels=labels,
            )
            out[period] = json.loads(fig.to_json())

        with self._lock:
            self._cache[key] = out
            while len(self._cache) > self.max_cache:
                self._cache.popitem(last=False)
        return out


# ---------------------------------------------------------------------------
# The web page (vanilla JS + inlined plotly.min.js) — appended below
# ---------------------------------------------------------------------------

_PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Wallet filter explorer — __VARIANT__</title>
<style>
:root {
  --bg:#f6f7f9; --panel:#ffffff; --border:#e3e6ea; --text:#1d2129; --muted:#697180;
  --accent:#2563eb; --full:#0d9488; --carved:#7c3aed; --picked:#dc2626;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
body { font: 13px/1.45 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color: var(--text); background: var(--bg); }
header { position: sticky; top: 0; z-index: 20; display: flex; align-items: center; gap: 14px;
         flex-wrap: wrap; padding: 8px 14px; background: #fff; border-bottom: 1px solid var(--border); }
header h1 { font-size: 14px; margin: 0; font-weight: 650; }
header h1 .variant { color: var(--muted); font-weight: 400; }
.counters { display: flex; gap: 14px; color: var(--muted); align-items: baseline; flex-wrap: wrap; }
.counters b { color: var(--text); font-weight: 650; }
.counters .sums { color: var(--muted); font-size: 12px; }
.actions { margin-left: auto; display: flex; gap: 6px; flex-wrap: wrap; }
button { font: inherit; padding: 4px 10px; border: 1px solid var(--border); border-radius: 6px;
         background: #fff; cursor: pointer; color: var(--text); }
button:hover { background: #f2f4f7; }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
button.primary:hover { background: #1d4fd7; }
#layout { display: grid; grid-template-columns: 305px minmax(0, 1fr); gap: 10px;
          padding: 10px; align-items: start; }
aside { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
        padding: 10px; position: sticky; top: 58px; max-height: calc(100vh - 68px);
        overflow-y: auto; }
aside h3 { margin: 14px 0 6px; font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
           color: var(--muted); }
aside h3:first-child { margin-top: 2px; }
aside .search { width: 100%; padding: 5px 8px; border: 1px solid var(--border); border-radius: 6px;
                font: inherit; margin-bottom: 4px; }
.memRow label { display: flex; gap: 6px; align-items: center; cursor: pointer; margin: 3px 0; }
.presetNote { color: var(--muted); font-size: 11px; margin: 2px 0 6px; }
details.cols { border: 1px solid var(--border); border-radius: 6px; margin-top: 8px; }
details.cols summary { cursor: pointer; padding: 6px 8px; font-weight: 600; color: var(--muted); }
details.cols[open] summary { border-bottom: 1px solid var(--border); }
details.cols .colsBody { padding: 6px 8px 8px; }
.frow { padding: 5px 6px; border-radius: 6px; border-left: 3px solid transparent; margin-bottom: 3px; }
.frow.active { background: #eef3fe; border-left-color: var(--accent); }
.frow.hiddenBySearch { display: none; }
.frowTop { display: flex; align-items: baseline; gap: 6px; }
.flabel { font-weight: 600; font-size: 12px; word-break: break-all; }
.ftag { font-size: 10px; font-weight: 650; padding: 0 4px; border-radius: 4px; color: #fff; }
.ftag.full { background: var(--full); }
.ftag.carved { background: var(--carved); }
.nanCount { color: var(--muted); font-weight: 400; font-size: 10px; }
.dual { position: relative; height: 16px; background: #e9edf2; border-radius: 8px; margin: 5px 0 3px; }
.dual .fill { position: absolute; top: 3px; height: 10px; background: #b9cdf3; border-radius: 5px;
              pointer-events: none; }
.dual input[type=range] { position: absolute; inset: 0; width: 100%; height: 16px; margin: 0;
                          background: none; pointer-events: none; -webkit-appearance: none; appearance: none; }
.dual input[type=range]::-webkit-slider-runnable-track { background: transparent; height: 16px; }
.dual input[type=range]::-moz-range-track { background: transparent; height: 16px; }
.dual input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none;
    pointer-events: auto; width: 13px; height: 16px; border-radius: 4px; background: var(--accent);
    border: 1px solid #fff; cursor: ew-resize; }
.dual input[type=range]::-moz-range-thumb { pointer-events: auto; width: 13px; height: 16px;
    border-radius: 4px; background: var(--accent); border: 1px solid #fff; cursor: ew-resize; }
.fnums { display: flex; gap: 4px; align-items: center; }
.fnums input[type=number] { width: 0; flex: 1 1 0; min-width: 40px; padding: 2px 4px;
    border: 1px solid var(--border); border-radius: 4px; font: inherit; }
.fnums .nanok { display: flex; gap: 3px; align-items: center; font-size: 11px; color: var(--muted);
    cursor: pointer; white-space: nowrap; }
.frow.dead { color: var(--muted); font-size: 12px; padding: 3px 6px; }
main { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
.scatterBar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 6px; }
.scatterBar label { color: var(--muted); display: flex; gap: 4px; align-items: center; }
.scatterBar select { max-width: 240px; }
.scatterBar .hint { color: var(--muted); font-size: 11px; margin-left: auto; }
#scatter { height: 470px; }
.detailHead { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
.detailHead .placeholder { color: var(--muted); }
.detailHead code { background: #f2f4f7; padding: 3px 8px; border-radius: 6px; font-size: 12px;
                   word-break: break-all; }
.detailHead .chip { font-size: 11px; padding: 2px 8px; border-radius: 10px; background: #eef1f5;
                    color: var(--muted); }
.detailHead .chip.on { background: #dcfce7; color: #166534; }
.detailHead .chip.on.elig { background: #fef3c7; color: #92400e; }
.detailHead .chip.on.sel { background: #fee2e2; color: #991b1b; }
.detailHead .chip.pick { background: #fee2e2; color: #991b1b; }
.detailHead .stats { color: var(--muted); font-size: 12px; }
.detailHead button { padding: 2px 8px; }
.detailHead .aggLabel { font-weight: 650; }
.detailControls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
                  margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.seg button { border: 0; border-radius: 0; padding: 3px 9px; background: #fff; }
.seg button.active { background: var(--accent); color: #fff; }
.seg button + button { border-left: 1px solid var(--border); }
.ctl { display: flex; gap: 4px; align-items: center; color: var(--muted); cursor: pointer;
       font-size: 12px; }
.pnlStatus { color: var(--muted); font-size: 11px; margin-left: auto; }
.variantBoxes { display: flex; gap: 4px 12px; flex-wrap: wrap; width: 100%; }
.variantBoxes label { display: flex; gap: 3px; align-items: center; font-size: 11px;
                      color: var(--muted); cursor: pointer; }
.detailPlots { display: grid; grid-template-columns: 1fr; gap: 10px; }
.noData { color: var(--muted); font-size: 12px; padding: 8px 0; }
@media (max-width: 1100px) { #layout { grid-template-columns: 260px minmax(0,1fr); } }
.tableBar { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; margin-bottom: 6px; }
.tableBar .note { color: var(--muted); font-size: 11px; }
.tableBar input { margin-left: auto; padding: 4px 8px; border: 1px solid var(--border);
                  border-radius: 6px; font: inherit; }
#tableWrap { max-height: 430px; overflow: auto; border: 1px solid var(--border); border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
thead th { position: sticky; top: 0; background: #f6f7f9; padding: 5px 8px; text-align: right;
           cursor: pointer; border-bottom: 1px solid var(--border); white-space: nowrap; z-index: 2; }
thead th.sorted::after { content: " \25BE"; }
thead th.sorted.asc::after { content: " \25B4"; }
tbody td { padding: 4px 8px; text-align: right; border-bottom: 1px solid #f0f2f5;
           font-variant-numeric: tabular-nums; white-space: nowrap; }
tbody td:first-child, thead th:first-child { text-align: left; }
tbody td.waddr { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 11px;
                 cursor: pointer; }
tbody td.waddr:hover { color: var(--accent); }
tbody tr:hover { background: #f8fafd; }
td.flag { color: #c3c9d2; }
td.flag.on { color: var(--text); font-weight: 650; }
.emptyMsg { color: var(--muted); padding: 10px; }
</style>
<script>__PLOTLYJS__</script>
</head>
<body>
<header>
  <h1>Wallet filter explorer <span class="variant">— __VARIANT__</span></h1>
  <div class="counters">
    <span>pool <b id="cPool">–</b></span>
    <span>passing <b id="cPass">–</b></span>
    <span>picked <b id="cPick">–</b></span>
    <span class="sums" id="cSums"></span>
  </div>
  <div class="actions">
    <button id="btnPreset" title="reset sliders to the variant thresholds">Variant thresholds</button>
    <button id="btnClear" title="remove all column filters">Clear filters</button>
    <button id="lassoMode" title="what lasso/box selection does">lasso: add</button>
    <button id="btnPickPassing" title="add every wallet passing the current filters to the picked set">pick passing</button>
    <button id="btnClearPick">clear picked</button>
    <button id="btnSave" class="primary">Save selection</button>
  </div>
</header>
<div id="layout">
  <aside id="filters">
    <input class="search" id="colSearch" placeholder="search columns...">
    <div id="memBox"></div>
    <div id="presetBox"></div>
    <div id="colBoxes"></div>
  </aside>
  <main>
    <div class="card">
      <div class="scatterBar">
        <label>X <select id="xSel"></select></label>
        <label><input type="checkbox" id="logX"> log</label>
        <label>Y <select id="ySel"></select></label>
        <label><input type="checkbox" id="logY"> log</label>
        <label>color <select id="colorSel"></select></label>
        <span class="hint">click: pick wallet + show detail · lasso: pick group ·
          null / non-positive (log) values drop out of the plot</span>
      </div>
      <div id="scatter"></div>
    </div>
    <div class="card">
      <div class="detailHead" id="detailHead">
        <span class="placeholder">click a wallet in the plot or table to inspect it</span>
      </div>
      <div class="detailControls">
        <div class="seg">
          <button id="viewWallet" class="active">this wallet</button>
          <button id="viewAgg">picked aggregate</button>
        </div>
        <label class="ctl"><input type="checkbox" id="toggleExposure" checked> exposures</label>
        <label class="ctl"><input type="checkbox" id="toggleCarve" checked> carved fracs</label>
        <span class="pnlStatus" id="pnlStatus"></span>
        <div class="variantBoxes" id="variantBoxes"></div>
      </div>
      <div class="detailPlots">
        <div id="plotTrain"></div>
        <div id="plotTest"></div>
      </div>
    </div>
    <div class="card">
      <div class="tableBar">
        <b id="tableCount">–</b> wallets passing filters <span class="note" id="tableNote"></span>
        <span class="note">· shift+click a row to pick a range</span>
        <input id="tableSearch" placeholder="find wallet 0x...">
      </div>
      <div id="tableWrap"><table id="walletTable"></table></div>
    </div>
  </main>
</div>
<script>
(async function () {
  'use strict';
  const $ = (s, el) => (el || document).querySelector(s);
  const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));

  let DATA;
  try {
    DATA = await (await fetch('/data')).json();
  } catch (e) {
    document.body.innerHTML = '<div style="padding:40px;font:15px sans-serif">' +
      'Failed to load /data: ' + String(e) + '</div>';
    return;
  }

  const COLS = {};
  DATA.columns.forEach(c => { COLS[c.key] = c; });
  const WALLETS = DATA.wallets;
  const WIDX = new Map(WALLETS.map((w, i) => [w.wallet, i]));
  const FLAGS = DATA.flags || [];
  const PNL_VARIANTS = DATA.pnlVariants || [];

  const state = {
    filters: {},
    membership: {eligible: false, strategy_selected: false, test_active: false},
    picked: new Set(),
    lasso: 'add',
    x: DATA.defaultAxes.x in COLS ? DATA.defaultAxes.x : DATA.columns[0].key,
    y: DATA.defaultAxes.y in COLS ? DATA.defaultAxes.y : DATA.columns[Math.min(1, DATA.columns.length - 1)].key,
    logX: false,
    logY: false,
    colorBy: '__status__',
    sortKey: null,
    sortAsc: false,
    tableSearch: '',
    detail: null,
    pickAnchor: null,
    shownWallets: [],
    detailMode: 'wallet',
    pnlVariants: new Set(
      PNL_VARIANTS.filter(v => v.default !== false).map(v => v.key)),
    showExposure: true,
    showCarve: true,
  };
  let passMask = [];

  // ---------------- formatting ----------------
  function fmtVal(v) {
    if (v === null || v === undefined) return '—';
    if (typeof v === 'boolean') return v ? '✓' : '·';
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1e6) return v.toExponential(2);
    if (a >= 1000) return Math.round(v).toLocaleString('en-US');
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    return v.toPrecision(3);
  }
  function colLabel(key) {
    const c = COLS[key];
    return c ? c.name + ' · ' + c.frame : key;
  }
  const HTML_ESCAPES = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};
  function esc(s) {
    return String(s).replace(/[&<>"']/g, ch => HTML_ESCAPES[ch]);
  }

  // ---------------- filtering ----------------
  function isActive(f) {
    return !!f && (f.min !== null || f.max !== null || f.nanOk);
  }
  function computePassing() {
    const active = Object.entries(state.filters).filter(entry => isActive(entry[1]));
    const m = state.membership;
    return WALLETS.map(w => {
      for (let k = 0; k < active.length; k++) {
        const f = active[k][1];
        const v = w[active[k][0]];
        if (v === null || v === undefined) {
          if (!f.nanOk) return false;
        } else {
          if (f.min !== null && v < f.min) return false;
          if (f.max !== null && v > f.max) return false;
        }
      }
      if (m.eligible && !w.eligible) return false;
      if (m.strategy_selected && !w.strategy_selected) return false;
      if (m.test_active && !w.test_active) return false;
      return true;
    });
  }

  // ---------------- filter rows ----------------
  const rowByKey = new Map();

  function makeFilterRow(col) {
    const row = document.createElement('div');
    row.className = 'frow';
    row.dataset.key = col.key;
    const nanNote = col.nanCount ? ' <span class="nanCount">(' + col.nanCount + ' NaN)</span>' : '';
    if (col.min === null || col.max === null) {
      row.classList.add('dead');
      row.innerHTML = '<div class="frowTop"><span class="flabel">' + esc(col.name) +
        '</span><span class="ftag ' + col.frame + '">' + col.frame + '</span>' +
        nanNote + ' <span class="nanCount">— all NaN, no slider</span></div>';
      return row;
    }
    const step = col.integral ? 1 : ((col.max - col.min) / 500 || 0.01);
    row.innerHTML =
      '<div class="frowTop"><span class="flabel" title="' + esc(col.key) + '">' + esc(col.name) +
        '</span><span class="ftag ' + col.frame + '">' + col.frame + '</span>' + nanNote + '</div>' +
      '<div class="dual"><div class="fill"></div>' +
        '<input type="range" class="rmin" min="' + col.min + '" max="' + col.max + '" step="' + step + '">' +
        '<input type="range" class="rmax" min="' + col.min + '" max="' + col.max + '" step="' + step + '">' +
      '</div>' +
      '<div class="fnums">' +
        '<input class="nmin" type="number" step="any">' +
        '<span>–</span>' +
        '<input class="nmax" type="number" step="any">' +
        '<label class="nanok" title="NaN values pass this filter"><input type="checkbox">NaN</label>' +
      '</div>';

    const rmin = $('.rmin', row), rmax = $('.rmax', row);
    const nmin = $('.nmin', row), nmax = $('.nmax', row);
    const nanok = $('.nanok input', row);
    const fill = $('.fill', row);

    function ensure() {
      if (!state.filters[col.key]) state.filters[col.key] = {min: null, max: null, nanOk: false};
      return state.filters[col.key];
    }
    function drawFill() {
      const span = (col.max - col.min) || 1;
      const lo = (Number(rmin.value) - col.min) / span;
      const hi = (Number(rmax.value) - col.min) / span;
      fill.style.left = (lo * 100) + '%';
      fill.style.width = Math.max(0, (hi - lo) * 100) + '%';
    }
    function syncNumbers() {
      const f = state.filters[col.key];
      nmin.value = f && f.min !== null ? f.min : '';
      nmax.value = f && f.max !== null ? f.max : '';
      nmin.placeholder = fmtVal(col.min);
      nmax.placeholder = fmtVal(col.max);
    }
    function markActive() {
      row.classList.toggle('active', isActive(state.filters[col.key]));
    }
    function fromSliders() {
      const f = ensure();
      f.min = Number(rmin.value) <= col.min ? null : Number(rmin.value);
      f.max = Number(rmax.value) >= col.max ? null : Number(rmax.value);
      f.nanOk = nanok.checked;
      syncNumbers();
      drawFill();
      markActive();
      refresh();
    }
    row.syncFromState = function () {
      const f = state.filters[col.key];
      rmin.value = f && f.min !== null ? Math.min(Math.max(f.min, col.min), col.max) : col.min;
      rmax.value = f && f.max !== null ? Math.min(Math.max(f.max, col.min), col.max) : col.max;
      nanok.checked = !!(f && f.nanOk);
      syncNumbers();
      drawFill();
      markActive();
    };

    rmin.addEventListener('input', () => {
      if (Number(rmin.value) > Number(rmax.value)) rmax.value = rmin.value;
      fromSliders();
    });
    rmax.addEventListener('input', () => {
      if (Number(rmax.value) < Number(rmin.value)) rmin.value = rmax.value;
      fromSliders();
    });
    nmin.addEventListener('change', () => {
      const f = ensure();
      const t = nmin.value.trim();
      f.min = t === '' ? null : Number(t);
      if (!Number.isFinite(f.min)) f.min = null;
      if (f.min !== null) rmin.value = Math.min(Math.max(f.min, col.min), col.max);
      nmin.value = f.min === null ? '' : f.min;
      drawFill();
      markActive();
      refresh();
    });
    nmax.addEventListener('change', () => {
      const f = ensure();
      const t = nmax.value.trim();
      f.max = t === '' ? null : Number(t);
      if (!Number.isFinite(f.max)) f.max = null;
      if (f.max !== null) rmax.value = Math.min(Math.max(f.max, col.min), col.max);
      nmax.value = f.max === null ? '' : f.max;
      drawFill();
      markActive();
      refresh();
    });
    nanok.addEventListener('change', () => { ensure().nanOk = nanok.checked; markActive(); refresh(); });

    row.syncFromState();
    rowByKey.set(col.key, row);
    return row;
  }

  function syncAllRows() {
    rowByKey.forEach(row => row.syncFromState());
  }

  // ---------------- sidebar ----------------
  function buildSidebar() {
    const memLabels = {
      eligible: 'only strategy-eligible',
      strategy_selected: 'only strategy-selected',
      test_active: 'only test-active',
    };
    if (FLAGS.length) {
      const box = $('#memBox');
      box.innerHTML = '<h3>Membership</h3><div class="memRow"></div>';
      const row = $('.memRow', box);
      FLAGS.forEach(flag => {
        const label = document.createElement('label');
        label.innerHTML = '<input type="checkbox"> ' + memLabels[flag];
        label.querySelector('input').addEventListener('change', ev => {
          state.membership[flag] = ev.target.checked;
          refresh();
        });
        row.appendChild(label);
      });
    }

    const presetKeys = Object.keys(DATA.presets || {});
    const presetBox = $('#presetBox');
    if (presetKeys.length) {
      presetBox.innerHTML = '<h3>Variant thresholds — ' + esc(DATA.variant) + '</h3>';
      if (DATA.presetsDropped && DATA.presetsDropped.length) {
        const note = document.createElement('div');
        note.className = 'presetNote';
        note.textContent = 'unavailable in this dataset: ' + DATA.presetsDropped.join(', ');
        presetBox.appendChild(note);
      }
      presetKeys.forEach(k => { if (COLS[k]) presetBox.appendChild(makeFilterRow(COLS[k])); });
    }

    const presetSet = new Set(presetKeys);
    const colBoxes = $('#colBoxes');
    [['full', 'Full stats'], ['carved', 'Carved stats']].forEach(pair => {
      const frame = pair[0], title = pair[1];
      const cols = DATA.columns.filter(c => c.frame === frame && !presetSet.has(c.key));
      if (!cols.length) return;
      const details = document.createElement('details');
      details.className = 'cols';
      details.innerHTML = '<summary>' + title + ' — all columns (' + cols.length + ')</summary><div class="colsBody"></div>';
      const body = $('.colsBody', details);
      cols.forEach(c => body.appendChild(makeFilterRow(c)));
      colBoxes.appendChild(details);
    });

    $('#colSearch').addEventListener('input', ev => {
      const q = ev.target.value.trim().toLowerCase();
      $$('.frow').forEach(row => {
        const hit = !q || row.textContent.toLowerCase().includes(q);
        row.classList.toggle('hiddenBySearch', !hit);
        if (!q) return;
        const parent = row.closest('details');
        if (parent) parent.open = true;
      });
    });
  }

  // ---------------- axis / color selects ----------------
  function fillOptionSelect(sel, specials) {
    sel.innerHTML = '';
    (specials || []).forEach(pair => {
      const opt = document.createElement('option');
      opt.value = pair[0]; opt.textContent = pair[1];
      sel.appendChild(opt);
    });
    [['full', 'Full stats'], ['carved', 'Carved stats']].forEach(pair => {
      const group = document.createElement('optgroup');
      group.label = pair[1];
      DATA.columns.filter(c => c.frame === pair[0]).forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.key; opt.textContent = c.name;
        group.appendChild(opt);
      });
      sel.appendChild(group);
    });
  }

  // ---------------- scatter ----------------
  const gd = $('#scatter');

  function markerSpec() {
    const mode = state.colorBy;
    const colors = [], sizes = [], opacities = [], lineW = [], lineC = [];
    for (let i = 0; i < WALLETS.length; i++) {
      const w = WALLETS[i], pass = passMask[i], picked = state.picked.has(w.wallet);
      sizes.push(pass ? (picked ? 10 : 7) : 4);
      opacities.push(pass ? 1 : 0.32);
      lineW.push(picked ? 2 : 0);
      lineC.push('#dc2626');
      if (mode === '__status__') {
        colors.push(pass ? (w.strategy_selected ? '#dc2626' : (w.eligible ? '#f59e0b' : '#2563eb')) : '#c4c9d2');
      } else if (mode === '__picked__') {
        colors.push(picked ? '#dc2626' : (pass ? '#2563eb' : '#c4c9d2'));
      } else {
        const v = w[mode];
        if (!pass || v === null || v === undefined) colors.push('#c4c9d2');
        else colors.push(v);
      }
    }
    const spec = {color: colors, size: sizes, opacity: opacities, line: {color: lineC, width: lineW}};
    if (mode !== '__status__' && mode !== '__picked__') {
      spec.colorscale = 'Viridis';
      spec.showscale = true;
      spec.colorbar = {thickness: 10, len: 0.75, title: {text: colLabel(mode), side: 'right'}};
    }
    return spec;
  }

  function renderScatter() {
    const xs = WALLETS.map(w => w[state.x]);
    const ys = WALLETS.map(w => w[state.y]);
    const texts = WALLETS.map(w => {
      const badges = [];
      if (w.eligible) badges.push('eligible');
      if (w.strategy_selected) badges.push('selected');
      if (state.picked.has(w.wallet)) badges.push('picked');
      return '<b>' + w.wallet + '</b>' + (badges.length ? ' · ' + badges.join(' · ') : '') +
        '<br>' + esc(colLabel(state.x)) + ': ' + fmtVal(w[state.x]) +
        '<br>' + esc(colLabel(state.y)) + ': ' + fmtVal(w[state.y]);
    });
    Plotly.react(gd, [{
      x: xs, y: ys, type: 'scatter', mode: 'markers',
      text: texts, hovertemplate: '%{text}<extra></extra>',
      marker: markerSpec(),
    }], {
      margin: {l: 58, r: 16, t: 10, b: 46},
      xaxis: {title: {text: colLabel(state.x)}, type: state.logX ? 'log' : 'linear', zeroline: false},
      yaxis: {title: {text: colLabel(state.y)}, type: state.logY ? 'log' : 'linear', zeroline: false},
      dragmode: 'lasso',
      hovermode: 'closest',
    }, {displaylogo: false, responsive: true,
        modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian']});
    attachScatterEvents();
  }

  function attachScatterEvents() {
    if (typeof gd.removeAllListeners === 'function') {
      gd.removeAllListeners('plotly_click');
      gd.removeAllListeners('plotly_selected');
    }
    gd.on('plotly_click', ev => {
      const p = ev && ev.points && ev.points[0];
      if (!p || p.pointIndex === undefined || p.pointIndex === null) return;
      const wallet = WALLETS[p.pointIndex].wallet;
      togglePick(wallet);
      selectWallet(wallet);
    });
    gd.on('plotly_selected', ev => {
      if (!ev || !ev.points || !ev.points.length) return;
      ev.points.forEach(p => {
        const i = p.pointIndex !== undefined ? p.pointIndex : p.i;
        if (i === undefined || i === null) return;
        if (state.lasso === 'add') state.picked.add(WALLETS[i].wallet);
        else state.picked.delete(WALLETS[i].wallet);
      });
      refresh();
    });
  }

  // ---------------- picked set ----------------
  function togglePick(wallet) {
    if (state.picked.has(wallet)) state.picked.delete(wallet);
    else state.picked.add(wallet);
    refresh();
  }

  function pickRange(target) {
    const order = state.shownWallets;
    const from = state.pickAnchor ? order.indexOf(state.pickAnchor) : -1;
    const to = order.indexOf(target);
    if (from < 0 || to < 0) {
      state.pickAnchor = target;
      togglePick(target);
      return;
    }
    const [lo, hi] = from <= to ? [from, to] : [to, from];
    const select = !state.picked.has(target);
    for (let i = lo; i <= hi; i++) {
      if (select) state.picked.add(order[i]);
      else state.picked.delete(order[i]);
    }
    state.pickAnchor = target;
    refresh();
  }

  function pickPassing() {
    WALLETS.forEach((w, i) => { if (passMask[i]) state.picked.add(w.wallet); });
    refresh();
  }

  function updateCounters() {
    const nPass = passMask.reduce((a, b) => a + (b ? 1 : 0), 0);
    $('#cPool').textContent = WALLETS.length;
    $('#cPass').textContent = nPass;
    $('#cPick').textContent = state.picked.size;
    $('#btnPickPassing').textContent = 'pick passing (' + nPass + ')';
    const cpKey = 'copyable_pnl__carved';
    if (COLS[cpKey]) {
      let sumPass = 0, sumPick = 0;
      WALLETS.forEach((w, i) => {
        const v = w[cpKey];
        if (v === null || v === undefined) return;
        if (passMask[i]) sumPass += v;
        if (state.picked.has(w.wallet)) sumPick += v;
      });
      $('#cSums').textContent = 'Σ copyable_pnl (carved): passing ' + fmtVal(sumPass) +
        ' · picked ' + fmtVal(sumPick);
    }
  }

  // ---------------- table ----------------
  const TABLE_COLS = [
    ['wallet', 'wallet'],
    ['eligible', 'E'], ['strategy_selected', 'S'],
    ['buy_roi__full', 'buy_roi F'], ['buy_roi__carved', 'buy_roi C'],
    ['total_pnl__full', 'total_pnl F'], ['buy_pnl__full', 'buy_pnl F'],
    ['num_markets__full', 'markets F'], ['num_buckets__full', 'buckets F'],
    ['estimated_buy_sharpe__full', 'buy_sharpe F'],
    ['estimated_copyable_buy_sharpe__carved', 'cp_sharpe C'],
    ['buy_copyable_pnl__carved', 'bcpnl C'], ['buy_copyable_roi__carved', 'bcroi C'],
    ['copyable_pnl__carved', 'cpnl C'],
    ['max_drawdown_to_pnl__full', 'dd/pnl F'], ['market_pnl_hhi__full', 'hhi F'],
    ['total_notional__full', 'notional F'],
  ].filter(pair => pair[0] === 'wallet' || FLAGS.includes(pair[0]) || COLS[pair[0]]);

  function cmp(a, b) {
    if (a === b) return 0;
    if (a === null || a === undefined) return 1;
    if (b === null || b === undefined) return -1;
    if (typeof a === 'boolean') return (a ? 1 : 0) - (b ? 1 : 0);
    if (typeof a === 'number' && typeof b === 'number') return a - b;
    return String(a).localeCompare(String(b));
  }

  function cellValue(w, key) {
    const v = w[key];
    if (key === 'eligible' || key === 'strategy_selected' || key === 'test_active') {
      return '<td class="flag' + (v ? ' on' : '') + '">' + (v ? '✓' : '·') + '</td>';
    }
    return '<td>' + fmtVal(v) + '</td>';
  }

  function renderTable() {
    const table = $('#walletTable');
    let idx = [];
    WALLETS.forEach((w, i) => { if (passMask[i]) idx.push(i); });
    const total = idx.length;
    if (state.sortKey !== null) {
      idx.sort((a, b) => cmp(WALLETS[a][state.sortKey], WALLETS[b][state.sortKey]) *
        (state.sortAsc ? 1 : -1));
    }
    const q = state.tableSearch.trim().toLowerCase();
    if (q) idx = idx.filter(i => WALLETS[i].wallet.toLowerCase().includes(q));
    const LIMIT = 500;
    const shown = idx.slice(0, LIMIT);
    state.shownWallets = shown.map(i => WALLETS[i].wallet);

    const head = '<thead><tr><th>picked</th>' + TABLE_COLS.map(pair =>
      '<th data-k="' + pair[0] + '" class="' +
      (state.sortKey === pair[0] ? 'sorted' + (state.sortAsc ? ' asc' : '') : '') + '">' +
      esc(pair[1]) + '</th>').join('') + '</tr></thead>';

    let body = '';
    shown.forEach(i => {
      const w = WALLETS[i];
      body += '<tr data-w="' + w.wallet + '">' +
        '<td><input type="checkbox" class="rowpick"' +
        (state.picked.has(w.wallet) ? ' checked' : '') + '></td>' +
        TABLE_COLS.map(pair => pair[0] === 'wallet'
          ? '<td class="waddr" title="show detail">' + w.wallet + '</td>'
          : cellValue(w, pair[0])).join('') +
        '</tr>';
    });
    table.innerHTML = head + '<tbody>' + (body ||
      '<tr><td class="emptyMsg" colspan="' + (TABLE_COLS.length + 1) +
      '">no wallets pass the current filters</td></tr>') + '</tbody>';

    $('#tableCount').textContent = total;
    $('#tableNote').textContent = total > LIMIT
      ? '(showing first ' + LIMIT + ' — narrow the filters or search)' : '';

    $$('th[data-k]', table).forEach(th => th.addEventListener('click', () => {
      const k = th.dataset.k;
      if (state.sortKey === k) {
        if (state.sortAsc) { state.sortKey = null; state.sortAsc = false; }
        else state.sortAsc = true;
      } else {
        state.sortKey = k;
        state.sortAsc = false;
      }
      renderTable();
    }));

    $('tbody', table).addEventListener('click', ev => {
      const tr = ev.target.closest('tr');
      if (!tr || !tr.dataset.w) return;
      const wallet = tr.dataset.w;
      if (ev.shiftKey) { pickRange(wallet); return; }
      const onPick = ev.target.classList.contains('rowpick');
      const onAddr = ev.target.classList.contains('waddr');
      if (!onPick && !onAddr) return;
      if (onPick) { togglePick(wallet); state.pickAnchor = wallet; return; }
      if (onAddr) selectWallet(wallet);
    });
  }

  // ---------------- wallet detail / PnL panels ----------------
  function detailStatsHtml(w) {
    const keys = ['buy_roi__full', 'buy_roi__carved', 'total_pnl__full',
                  'estimated_copyable_buy_sharpe__carved', 'buy_copyable_pnl__carved',
                  'copyable_pnl__carved'];
    return keys.filter(k => COLS[k]).map(k =>
      '<span class="stats">' + esc(colLabel(k)) + ': <b>' + fmtVal(w[k]) +
      '</b></span>').join('');
  }

  function updateDetailHead() {
    const head = $('#detailHead');
    if (state.detailMode === 'agg') {
      const n = state.picked.size;
      head.innerHTML = '<span class="aggLabel">aggregate of <b>' + n +
        '</b> picked wallet' + (n === 1 ? '' : 's') + '</span>' +
        '<span class="stats">cumulative PnL of the union of their fills</span>';
      return;
    }
    const wallet = state.detail;
    const w = wallet ? WALLETS[WIDX.get(wallet)] : null;
    if (!w) {
      head.innerHTML = '<span class="placeholder">click a wallet in the plot or table to inspect it</span>';
      return;
    }
    const picked = state.picked.has(wallet);
    const chips = [];
    if (FLAGS.includes('eligible')) {
      chips.push('<span class="chip' + (w.eligible ? ' on elig' : '') + '">eligible</span>');
    }
    if (FLAGS.includes('strategy_selected')) {
      chips.push('<span class="chip' + (w.strategy_selected ? ' on sel' : '') +
        '">strategy-selected</span>');
    }
    if (FLAGS.includes('test_active')) {
      chips.push('<span class="chip' + (w.test_active ? ' on' : '') + '">test-active</span>');
    }
    chips.push('<span class="chip' + (picked ? ' pick' : '') + '">' +
      (picked ? 'picked' : 'not picked') + '</span>');
    head.innerHTML = '<code>' + wallet + '</code>' + chips.join('') +
      '<button id="btnTogglePick">' + (picked ? 'unpick' : 'pick') + '</button>' +
      detailStatsHtml(w);
    $('#btnTogglePick').addEventListener('click', () => {
      togglePick(wallet);
      updateDetailHead();
    });
  }

  function pnlWallets() {
    if (state.detailMode === 'agg') return Array.from(state.picked).sort();
    return state.detail ? [state.detail] : [];
  }

  let pnlReqId = 0;
  let pnlTimer = null;

  function schedulePnl(delay) {
    clearTimeout(pnlTimer);
    pnlTimer = setTimeout(renderPnl, delay === undefined ? 180 : delay);
  }

  function showPnlMessage(html) {
    ['plotTrain', 'plotTest'].forEach(id => {
      const div = $('#' + id);
      Plotly.purge(div);
      div.innerHTML = '<div class="noData">' + html + '</div>';
    });
  }

  function showFigure(divId, fig) {
    const div = $('#' + divId);
    const layout = Object.assign({}, fig.layout, {
      height: 370,
      margin: {l: 62, r: 14, t: 46, b: 36},
      title: Object.assign({}, fig.layout.title, {font: {size: 12}}),
      legend: {
        x: 0.01, y: 0.99, xanchor: 'left', yanchor: 'top',
        bgcolor: 'rgba(255,255,255,0.82)', bordercolor: '#e3e6ea', borderwidth: 1,
        font: {size: 9},
      },
    });
    Plotly.react(div, fig.data, layout, {displaylogo: false, responsive: true});
  }

  async function renderPnl() {
    clearTimeout(pnlTimer);
    if (!PNL_VARIANTS.length) {
      showPnlMessage('no fills passed to the explorer (PnL panels disabled)');
      $('#pnlStatus').textContent = '';
      return;
    }
    const wallets = pnlWallets();
    const variants = PNL_VARIANTS.map(v => v.key)
      .filter(k => state.pnlVariants.has(k));
    if (!wallets.length) {
      showPnlMessage(state.detailMode === 'agg'
        ? 'pick wallets (click / lasso) to show their aggregate PnL'
        : 'click a wallet in the plot or table to inspect it');
      $('#pnlStatus').textContent = '';
      return;
    }
    if (!variants.length) {
      showPnlMessage('select at least one PnL variant above');
      $('#pnlStatus').textContent = '';
      return;
    }
    const reqId = ++pnlReqId;
    $('#pnlStatus').textContent = 'loading…';
    try {
      const r = await fetch('/pnl', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          wallets,
          variants,
          exposure: state.showExposure,
          carve: state.showCarve,
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      if (reqId !== pnlReqId) return;
      showFigure('plotTrain', data.train);
      showFigure('plotTest', data.test);
      $('#pnlStatus').textContent = data.label + ' · ' + wallets.length +
        ' wallet' + (wallets.length === 1 ? '' : 's');
    } catch (e) {
      if (reqId !== pnlReqId) return;
      showPnlMessage('failed to build PnL panel: ' + esc(String(e)));
      $('#pnlStatus').textContent = '';
    }
  }

  function selectWallet(wallet) {
    state.detail = wallet;
    updateDetailHead();
    if (state.detailMode === 'wallet') renderPnl();
    else schedulePnl();
  }

  function buildDetailControls() {
    if (!PNL_VARIANTS.length) {
      ['viewWallet', 'viewAgg', 'toggleExposure', 'toggleCarve'].forEach(id => {
        $('#' + id).disabled = true;
      });
      return;
    }
    const boxes = $('#variantBoxes');
    PNL_VARIANTS.forEach(v => {
      const label = document.createElement('label');
      label.innerHTML = '<input type="checkbox"' +
        (state.pnlVariants.has(v.key) ? ' checked' : '') + '> ' + esc(v.label);
      label.querySelector('input').addEventListener('change', ev => {
        if (ev.target.checked) state.pnlVariants.add(v.key);
        else state.pnlVariants.delete(v.key);
        renderPnl();
      });
      boxes.appendChild(label);
    });

    function activateView() {
      $('#viewWallet').classList.toggle('active', state.detailMode === 'wallet');
      $('#viewAgg').classList.toggle('active', state.detailMode === 'agg');
      updateDetailHead();
      renderPnl();
    }
    $('#viewWallet').addEventListener('click', () => {
      if (state.detailMode === 'wallet') return;
      state.detailMode = 'wallet';
      if (!state.detail) state.detail = Array.from(state.picked)[0] || null;
      activateView();
    });
    $('#viewAgg').addEventListener('click', () => {
      if (state.detailMode === 'agg') return;
      state.detailMode = 'agg';
      activateView();
    });
    $('#toggleExposure').addEventListener('change', ev => {
      state.showExposure = ev.target.checked;
      renderPnl();
    });
    $('#toggleCarve').addEventListener('change', ev => {
      state.showCarve = ev.target.checked;
      renderPnl();
    });
    updateDetailHead();
  }

  // ---------------- save ----------------
  async function saveSelection() {
    const filters = {};
    Object.entries(state.filters).forEach(entry => {
      const k = entry[0], f = entry[1];
      if (!isActive(f)) return;
      filters[k] = {min: f.min, max: f.max, nanOk: !!f.nanOk};
    });
    const passing = [];
    WALLETS.forEach((w, i) => { if (passMask[i]) passing.push(w.wallet); });
    const payload = {
      variant: DATA.variant,
      filters,
      membership: {
        eligible: !!state.membership.eligible,
        strategy_selected: !!state.membership.strategy_selected,
        test_active: !!state.membership.test_active,
      },
      picked: Array.from(state.picked).sort(),
      passing: passing.sort(),
      counts: {pool: WALLETS.length, passing: passing.length, picked: state.picked.size},
    };
    const btn = $('#btnSave');
    const label = btn.textContent;
    try {
      const r = await fetch('/selection', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error(await r.text());
      btn.textContent = 'Saved ✓';
    } catch (e) {
      btn.textContent = 'Save failed';
      console.error(e);
    }
    setTimeout(() => { btn.textContent = label; }, 2500);
  }

  // ---------------- refresh ----------------
  function refresh() {
    passMask = computePassing();
    updateCounters();
    if (gd.data && gd.data.length) Plotly.restyle(gd, {marker: markerSpec()}, [0]);
    renderTable();
    updateDetailHead();
    if (state.detailMode === 'agg') schedulePnl();
  }

  // ---------------- boot ----------------
  function applyPresets() {
    state.filters = {};
    Object.entries(DATA.presets || {}).forEach(entry => {
      const k = entry[0], b = entry[1];
      state.filters[k] = {
        min: b.min !== undefined ? b.min : null,
        max: b.max !== undefined ? b.max : null,
        nanOk: false,
      };
    });
  }

  applyPresets();
  buildSidebar();

  fillOptionSelect($('#xSel'));
  fillOptionSelect($('#ySel'));
  $('#xSel').value = state.x;
  $('#ySel').value = state.y;
  fillOptionSelect($('#colorSel'), [
    ['__status__', 'group (eligible / selected)'],
    ['__picked__', 'picked'],
  ]);
  $('#colorSel').value = state.colorBy;

  $('#xSel').addEventListener('change', ev => { state.x = ev.target.value; renderScatter(); });
  $('#ySel').addEventListener('change', ev => { state.y = ev.target.value; renderScatter(); });
  $('#logX').addEventListener('change', ev => { state.logX = ev.target.checked; renderScatter(); });
  $('#logY').addEventListener('change', ev => { state.logY = ev.target.checked; renderScatter(); });
  $('#colorSel').addEventListener('change', ev => {
    state.colorBy = ev.target.value;
    renderScatter();
  });

  $('#btnPreset').addEventListener('click', () => {
    applyPresets(); syncAllRows(); refresh();
  });
  $('#btnClear').addEventListener('click', () => {
    state.filters = {}; syncAllRows(); refresh();
  });
  $('#btnClearPick').addEventListener('click', () => {
    state.picked.clear(); refresh();
  });
  $('#btnPickPassing').addEventListener('click', pickPassing);
  $('#lassoMode').addEventListener('click', ev => {
    state.lasso = state.lasso === 'add' ? 'remove' : 'add';
    ev.target.textContent = 'lasso: ' + state.lasso;
  });
  $('#btnSave').addEventListener('click', saveSelection);
  $('#tableSearch').addEventListener('input', ev => {
    state.tableSearch = ev.target.value;
    renderTable();
  });

  buildDetailControls();
  renderScatter();
  refresh();
  renderPnl();
})();
</script>
</body>
</html>
"""
# ---------------------------------------------------------------------------
# Localhost server
# ---------------------------------------------------------------------------


class _ExplorerState:
    """Immutable-ish payload plus the latest saved selection, shared by handlers."""

    def __init__(
        self,
        *,
        page_html: str,
        payload_json: bytes,
        payload_json_gz: bytes,
        panel: _PnlPanel | None,
        pnl_variants: list[str],
        selection_path: Path,
        variant: str,
    ) -> None:
        self.page_html = page_html
        self.page_gz = gzip.compress(page_html.encode("utf-8"))
        self.payload_json = payload_json
        self.payload_json_gz = payload_json_gz
        self.panel = panel
        self.pnl_variants = list(pnl_variants)
        self.selection_path = selection_path
        self.variant = variant
        self.selection: dict[str, Any] | None = None
        self.lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "WalletFilterExplorer/1.0"

    @property
    def state(self) -> _ExplorerState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        body_gz: bytes | None = None,
    ) -> None:
        accept = self.headers.get("Accept-Encoding", "") or ""
        use_gzip = body_gz is not None and "gzip" in accept
        payload = body_gz if use_gzip else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _text(self, status: int, message: str) -> None:
        self._send(status, message.encode("utf-8"), "text/plain; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        state = self.state
        if path in ("/", "/index.html"):
            body = state.page_html.encode("utf-8")
            self._send(
                200,
                body,
                "text/html; charset=utf-8",
                body_gz=state.page_gz,
            )
            return
        if path == "/data":
            self._send(
                200,
                state.payload_json,
                "application/json",
                body_gz=state.payload_json_gz,
            )
            return
        if path == "/selection":
            with state.lock:
                selection = state.selection
            if selection is None:
                self._text(404, "nothing saved yet")
            else:
                body = json.dumps(selection).encode("utf-8")
                self._send(200, body, "application/json")
            return
        self._text(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        if path == "/pnl":
            self._post_pnl()
            return
        if path != "/selection":
            self._text(404, "not found")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._text(400, f"invalid JSON: {exc}")
            return
        if not isinstance(data, dict):
            self._text(400, "expected a JSON object")
            return
        state = self.state
        data.setdefault("variant", state.variant)
        state.selection_path.parent.mkdir(parents=True, exist_ok=True)
        state.selection_path.write_text(json.dumps(data, indent=2))
        with state.lock:
            state.selection = data
        body = json.dumps({"ok": True, "path": str(state.selection_path)}).encode("utf-8")
        self._send(200, body, "application/json")

    def _post_pnl(self) -> None:
        state = self.state
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._text(400, f"invalid JSON: {exc}")
            return
        if not isinstance(data, dict):
            self._text(400, "expected a JSON object")
            return
        wallets = data.get("wallets")
        variants = data.get("variants")
        if not isinstance(wallets, list) or not all(isinstance(w, str) for w in wallets):
            self._text(400, "wallets must be a list of wallet addresses")
            return
        if not wallets:
            self._text(400, "wallets must not be empty")
            return
        if not isinstance(variants, list) or not all(isinstance(v, str) for v in variants):
            self._text(400, "variants must be a list of PnL column names")
            return
        if not variants:
            self._text(400, "variants must not be empty")
            return
        unknown = [v for v in variants if v not in state.pnl_variants]
        if unknown:
            self._text(400, f"unknown PnL variants {unknown}")
            return
        if state.panel is None:
            self._text(400, "PnL panels are disabled (no fills passed)")
            return
        wallets = list(dict.fromkeys(wallets))
        try:
            result = state.panel.figures(
                wallets,
                variants,
                exposure=bool(data.get("exposure", True)),
                carve=bool(data.get("carve", False)),
            )
        except Exception as exc:  # noqa: BLE001
            self._text(500, f"failed to build PnL panel: {exc}")
            return
        body = json.dumps(result).encode("utf-8")
        self._send(200, body, "application/json", body_gz=gzip.compress(body))


class _ExplorerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: _ExplorerState) -> None:
        self.state = state
        super().__init__(address, _Handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass


class ExplorerHandle:
    """Handle to a running explorer server (usable as a context manager)."""

    def __init__(
        self,
        server: _ExplorerServer,
        thread: threading.Thread,
        *,
        out_dir: Path,
        variant: str,
    ) -> None:
        self._server = server
        self._thread = thread
        self._out_dir = out_dir
        self._variant = variant

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/"

    @property
    def variant(self) -> str:
        return self._variant

    @property
    def selection_path(self) -> Path:
        return self._out_dir / f"wallet_filter_selection_{self._variant}.json"

    def get_selection(self) -> dict[str, Any] | None:
        """Latest selection saved from the page (``None`` if not saved yet)."""
        return load_selection(self.selection_path)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> "ExplorerHandle":
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"ExplorerHandle(url={self.url!r}, variant={self._variant!r})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def open_wallet_filter_explorer(
    full_stats: pd.DataFrame,
    carved_stats: pd.DataFrame,
    *,
    variant: str = "sports",
    eligible_wallets: Iterable[str] | None = None,
    selected_wallets: Iterable[str] | None = None,
    fills: pd.DataFrame | None = None,
    split_date: Any | None = None,
    exposure_fracs: Iterable[float] = (),
    out_dir: str | Path,
    port: int = 0,
    open_browser: bool = True,
) -> ExplorerHandle:
    """Serve the interactive wallet-filter page and return a handle.

    Parameters
    ----------
    full_stats, carved_stats:
        ``wallet_vol_train_full`` / ``wallet_vol_train`` (each needs a ``wallet``
        column; ``median_dt`` is optional).
    variant:
        Key of :data:`VARIANT_FILTERS` used to pre-set the sliders.
    eligible_wallets, selected_wallets:
        Defaults of the stage-1 masks; enable the membership checkboxes.
    fills:
        Fill-level **pre-carve** BUY stream with ``wallet, dt, side, price,
        position, notional, last_condition_trade_ts, is_train``, the PnL
        variants (``pnl``, ``pnl_limited``, ``copyable_pnl``,
        ``copyable_pnl_limited``), their qty columns, and optionally
        ``is_carved`` (when present the panels offer both the full and the
        carved stream). Enables the train/test PnL panels and the test-active
        flag.
    split_date:
        Train/test boundary (defaults to the last ``last_condition_trade_ts``
        among the ``is_train`` fills).
    exposure_fracs:
        Deployed-capital carve fractions passed to
        :func:`~polymarket_analysis.visualization.wallet_plots.plot_wallet_selection_pnl`;
        pass the notebook's ``EXPOSURE_FRACS`` minus the frac that was already
        applied as ``is_carved`` to avoid duplicates.
    out_dir:
        Directory where ``wallet_filter_selection_{variant}.json`` is written on
        "Save selection".
    port:
        Bind port (``0`` picks a free port). ``open_browser`` opens the page.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    eligible = None if eligible_wallets is None else set(map(str, eligible_wallets))
    selected = None if selected_wallets is None else set(map(str, selected_wallets))

    payload = build_payload(
        full_stats,
        carved_stats,
        variant=variant,
        eligible_wallets=eligible,
        selected_wallets=selected,
        fills=fills,
    )
    payload_json = json.dumps(payload).encode("utf-8")

    panel: _PnlPanel | None = None
    pnl_variants: list[str] = []
    if fills is not None:
        pnl_variants = _available_pnl_variants(fills)
        if pnl_variants:
            panel = _PnlPanel(
                _prepare_pnl_fills(fills, pnl_variants),
                split_date=_resolve_split_date(fills, split_date),
                exposure_fracs=exposure_fracs,
            )

    page_html = (
        _PAGE_HTML.replace("__PLOTLYJS__", _plotly_offline.get_plotlyjs())
        .replace("__VARIANT__", variant)
    )
    state = _ExplorerState(
        page_html=page_html,
        payload_json=payload_json,
        payload_json_gz=gzip.compress(payload_json),
        panel=panel,
        pnl_variants=pnl_variants,
        selection_path=out_path / f"wallet_filter_selection_{variant}.json",
        variant=variant,
    )
    server = _ExplorerServer(("127.0.0.1", int(port)), state)
    thread = threading.Thread(
        target=server.serve_forever,
        name="wallet-filter-explorer",
        daemon=True,
    )
    thread.start()

    handle = ExplorerHandle(server, thread, out_dir=out_path, variant=variant)
    if open_browser:
        webbrowser.open(handle.url)
    print(f"Wallet filter explorer ({variant}) running at {handle.url}")
    print("  save from the page, then re-run the load cell; stop with explorer.stop()")
    return handle


def load_selection(path: str | Path) -> dict[str, Any] | None:
    """Read ``wallet_filter_selection_*.json`` (``None`` if not saved yet)."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def apply_selection_filters(
    selection: dict[str, Any],
    full_stats: pd.DataFrame,
    carved_stats: pd.DataFrame,
    *,
    fills: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the merged stats rows of the wallets saved by the page.

    The saved ``passing`` wallet set is re-checked against the recorded numeric
    filters (raising on a mismatch, e.g. mismatched stat frames); the returned
    frame keeps the page order and gains a boolean ``picked`` column. Pass the
    same ``fills`` that were used to build the explorer so
    ``median_dt_days_ago__*`` bounds stay consistent.
    """
    passing = list(selection.get("passing") or [])
    picked = set(selection.get("picked") or [])
    filters = selection.get("filters") or {}

    ref_ts = (
        pd.to_datetime(fills["dt"], utc=True).max()
        if fills is not None
        else pd.Timestamp.now(tz="UTC")
    )
    merged = _merged_wallet_frame(
        full_stats,
        carved_stats,
        eligible_wallets=None,
        selected_wallets=None,
        test_active_wallets=None,
        ref_ts=ref_ts,
    )

    missing = sorted(set(passing) - set(merged["wallet"]))
    if missing:
        raise ValueError(
            f"{len(missing)} saved wallets are missing from the stats frames: {missing[:5]}"
        )

    df = merged[merged["wallet"].isin(passing)].copy()
    for key, bound in filters.items():
        if key not in df.columns:
            raise KeyError(f"saved filter {key!r} is not a column of the merged stats")
        s = pd.to_numeric(df[key], errors="coerce")
        bad = pd.Series(False, index=df.index)
        lo, hi = bound.get("min"), bound.get("max")
        if lo is not None:
            bad |= s < float(lo)
        if hi is not None:
            bad |= s > float(hi)
        if not bound.get("nanOk"):
            bad |= s.isna()
        if bad.any():
            raise ValueError(
                f"saved selection does not match the stats: {int(bad.sum())} passing "
                f"wallet(s) violate {key!r}"
            )

    order = {w: i for i, w in enumerate(passing)}
    df["picked"] = df["wallet"].isin(picked)
    df["_order"] = df["wallet"].map(order)
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
