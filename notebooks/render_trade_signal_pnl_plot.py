from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


WORKSPACE_DIR = Path("/Users/vobornij/projects/polymarket/data/trade_signals_workspace")
OPEN_BUYS_PATH = WORKSPACE_DIR / "open_buys_test.parquet"
EXEC_TAPE_PATH = WORKSPACE_DIR / "execution_tape_test.parquet"
OUTPUT_HTML = Path("/Users/vobornij/projects/polymarket/notebooks/trade_signal_pnl_timeseries.html")
OUTPUT_DISCOVERY_CSV = Path("/Users/vobornij/projects/polymarket/notebooks/trade_signal_discovery_summary.csv")


def normalize_execution_tape(execution_tape: pd.DataFrame) -> pd.DataFrame:
    tape = execution_tape.copy()

    if "exec_price_raw" not in tape.columns:
        if "price" in tape.columns:
            tape["exec_price_raw"] = tape["price"].astype(float)
        else:
            raise ValueError("Execution tape missing both exec_price_raw and price columns")

    if "exec_source" not in tape.columns:
        tape["exec_source"] = "same_token"

    if "available_qty" not in tape.columns:
        if "quantity" in tape.columns:
            tape["available_qty"] = tape["quantity"].astype(float)
        elif "available_usdc" in tape.columns:
            tape["available_qty"] = tape["available_usdc"].astype(float) / tape["exec_price_raw"].clip(lower=1e-9)
        else:
            raise ValueError(
                "Execution tape missing available_qty; rebuild with build_trade_signal_workspace.py"
            )

    required = ["market_key", "tape_dt", "exec_price_raw", "exec_source", "available_qty"]
    missing = [c for c in required if c not in tape.columns]
    if missing:
        raise ValueError(f"Execution tape missing required columns: {missing}")

    return tape


def build_tape_groups(market_tape):
    return {k: g.reset_index(drop=True) for k, g in market_tape.groupby("market_key", sort=False)}


def attach_forward_fills(
    trades,
    tape_groups,
    latency_seconds=0,
    fill_horizon_seconds=600,
    slippage_bps=50.0,
    min_fill_ratio=1.0,
    ladder_price_bounds=None,
):
    trades = trades.copy().sort_values(["market_key", "dt"]).reset_index(drop=True)
    trades["fill_search_dt"] = trades["dt"] + pd.to_timedelta(latency_seconds, unit="s")
    trades["fill_deadline_dt"] = trades["fill_search_dt"] + pd.to_timedelta(fill_horizon_seconds, unit="s")

    slippage = slippage_bps / 10_000.0
    filled_parts = []

    for market_key, group in trades.groupby("market_key", sort=False):
        tape = tape_groups.get(market_key)
        if tape is None or tape.empty:
            continue

        tape = tape.sort_values("tape_dt").reset_index(drop=True).copy()
        if ladder_price_bounds is not None:
            low, high, inclusive = ladder_price_bounds
            tape = tape[tape["exec_price_raw"].between(low, high, inclusive=inclusive)].reset_index(drop=True)
            if tape.empty:
                continue
        tape_times = tape["tape_dt"].to_numpy()
        tape_prices = tape["exec_price_raw"].to_numpy(dtype=float)
        tape_qty_remaining = tape["available_qty"].to_numpy(dtype=float).copy()
        tape_source = tape["exec_source"].to_numpy()

        for row in group.itertuples(index=False):
            start_idx = tape_times.searchsorted(row.fill_search_dt, side="right")
            if start_idx >= len(tape):
                continue

            remaining_usdc = float(row.stake_usdc)
            filled_usdc = 0.0
            filled_qty = 0.0
            same_qty = 0.0
            opp_qty = 0.0
            fill_dt = None

            j = start_idx
            while j < len(tape) and tape_times[j] <= row.fill_deadline_dt and remaining_usdc > 1e-9:
                avail_qty = tape_qty_remaining[j]
                if avail_qty <= 1e-12:
                    j += 1
                    continue

                exec_price = float(np.clip(tape_prices[j] * (1.0 + slippage), 0.001, 0.999))
                max_usdc_here = avail_qty * exec_price
                take_usdc = min(remaining_usdc, max_usdc_here)
                take_qty = take_usdc / exec_price

                tape_qty_remaining[j] -= take_qty
                remaining_usdc -= take_usdc
                filled_usdc += take_usdc
                filled_qty += take_qty
                fill_dt = tape_times[j]

                if tape_source[j] == "same_token":
                    same_qty += take_qty
                else:
                    opp_qty += take_qty
                j += 1

            fill_ratio = filled_usdc / float(row.stake_usdc) if row.stake_usdc > 0 else 0.0
            if filled_usdc <= 0 or fill_ratio + 1e-12 < min_fill_ratio:
                continue

            out = pd.DataFrame([row._asdict()])
            out["filled_usdc"] = filled_usdc
            out["filled_qty"] = filled_qty
            out["exec_price"] = filled_usdc / filled_qty
            out["exec_price_raw"] = out["exec_price"]
            out["fill_ratio"] = fill_ratio
            out["tape_dt"] = fill_dt
            out["same_fill_share"] = same_qty / filled_qty if filled_qty > 0 else np.nan
            out["opposite_fill_share"] = opp_qty / filled_qty if filled_qty > 0 else np.nan
            out["exec_source"] = (
                "same_token" if opp_qty <= 1e-12 else ("opposite_token" if same_qty <= 1e-12 else "mixed")
            )
            filled_parts.append(out)

    if not filled_parts:
        return trades.iloc[0:0].copy()
    return pd.concat(filled_parts, ignore_index=True).sort_values("dt").reset_index(drop=True)


def backtest_strategy(
    signals,
    mask,
    tape_groups,
    stake_usdc=100.0,
    latency_seconds=0,
    fill_horizon_seconds=600,
    slippage_bps=50.0,
    fee_bps=10.0,
    max_signals_per_day=20,
    dedupe_by_market=True,
    starting_capital=10_000.0,
    min_fill_ratio=1.0,
    ladder_price_bounds=None,
):
    trades = signals[mask].copy().sort_values("dt")
    if dedupe_by_market:
        trades = trades.drop_duplicates("market_key", keep="first")

    trades["trade_date"] = trades["dt"].dt.floor("D")
    if max_signals_per_day is not None:
        trades["daily_rank"] = trades.groupby("trade_date").cumcount() + 1
        trades = trades[trades["daily_rank"] <= max_signals_per_day].copy()

    trades["stake_usdc"] = float(stake_usdc)
    trades = attach_forward_fills(
        trades,
        tape_groups=tape_groups,
        latency_seconds=latency_seconds,
        fill_horizon_seconds=fill_horizon_seconds,
        slippage_bps=slippage_bps,
        min_fill_ratio=min_fill_ratio,
        ladder_price_bounds=ladder_price_bounds,
    )

    fee = fee_bps / 10_000.0
    trades["gross_roi"] = np.where(trades["token_winner"], 1.0 / trades["exec_price"] - 1.0, -1.0)
    trades["net_roi"] = trades["gross_roi"] - fee
    trades["net_pnl_usdc"] = trades["filled_usdc"] * trades["net_roi"]

    daily = (
        trades.groupby("trade_date")
        .agg(trades=("market_key", "size"), net_pnl_usdc=("net_pnl_usdc", "sum"))
        .reset_index()
        .sort_values("trade_date")
    )
    if len(daily):
        daily["cum_net_pnl_usdc"] = daily["net_pnl_usdc"].cumsum()
        daily["equity_usdc"] = starting_capital + daily["cum_net_pnl_usdc"]
    else:
        daily = pd.DataFrame(columns=["trade_date", "trades", "net_pnl_usdc", "cum_net_pnl_usdc", "equity_usdc"])

    return trades, daily


def add_daily_traces(fig, daily_map, row, col):
    for name, daily in daily_map.items():
        if daily.empty:
            continue
        daily_plot = with_zero_anchor(daily)
        fig.add_trace(
            go.Scatter(
                x=daily_plot["trade_date"],
                y=daily_plot["cum_net_pnl_usdc"],
                mode="lines",
                name=name,
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}<br>Cumulative PnL: %{y:.2f} USDC<extra></extra>",
            ),
            row=row,
            col=col,
        )


def with_zero_anchor(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily

    first_date = daily["trade_date"].min()
    anchor = pd.DataFrame(
        {
            "trade_date": [first_date - pd.Timedelta(days=1)],
            "net_pnl_usdc": [0.0],
            "cum_net_pnl_usdc": [0.0],
        }
    )
    return pd.concat([anchor, daily], ignore_index=True).sort_values("trade_date").reset_index(drop=True)


def build_strategy_sum_daily(daily_map):
    parts = []
    for name, daily in daily_map.items():
        if daily.empty:
            continue
        part = daily[["trade_date", "net_pnl_usdc"]].copy()
        part["strategy"] = name
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=["trade_date", "net_pnl_usdc", "cum_net_pnl_usdc"])

    combined = pd.concat(parts, ignore_index=True)
    summed = (
        combined.groupby("trade_date", as_index=False)
        .agg(net_pnl_usdc=("net_pnl_usdc", "sum"))
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    summed["cum_net_pnl_usdc"] = summed["net_pnl_usdc"].cumsum()
    return summed


def summarize_run(name, trades, daily, params):
    if trades.empty:
        return {
            "name": name,
            **params,
            "filled_trades": 0,
            "win_rate": np.nan,
            "net_roi_on_stake": np.nan,
            "net_pnl_usdc": 0.0,
        }

    net_pnl = float((trades["filled_usdc"] * trades["net_roi"]).sum())
    total_stake = float(trades["filled_usdc"].sum())
    return {
        "name": name,
        **params,
        "filled_trades": int(len(trades)),
        "win_rate": float(trades["token_winner"].mean()),
        "net_roi_on_stake": net_pnl / total_stake if total_stake > 0 else np.nan,
        "net_pnl_usdc": net_pnl,
    }


def build_discovery_specs():
    price_specs = [
        ("0.00-0.10", 0.00, 0.10, "left"),
        ("0.10-0.25", 0.10, 0.25, "left"),
        ("0.25-0.40", 0.25, 0.40, "left"),
        ("0.40-0.60", 0.40, 0.60, "left"),
        ("0.60-0.75", 0.60, 0.75, "left"),
        ("0.75-0.90", 0.75, 0.90, "left"),
        ("0.90-1.00", 0.90, 1.00, "both"),
    ]
    same_specs = [
        ("same==0", lambda df: df["prior_same"] == 0),
        ("same==1", lambda df: df["prior_same"] == 1),
        ("same<=1", lambda df: df["prior_same"] <= 1),
    ]
    opp_specs = [
        ("opp==0", lambda df: df["prior_opp"] == 0),
        ("opp<=1", lambda df: df["prior_opp"] <= 1),
    ]

    specs = []
    for price_label, low, high, inclusive in price_specs:
        for same_label, same_fn in same_specs:
            for opp_label, opp_fn in opp_specs:
                name = f"{price_label} | {same_label} | {opp_label}"
                specs.append(
                    {
                        "name": name,
                        "price_label": price_label,
                        "price_low": low,
                        "price_high": high,
                        "price_inclusive": inclusive,
                        "same_label": same_label,
                        "same_fn": same_fn,
                        "opp_label": opp_label,
                        "opp_fn": opp_fn,
                    }
                )
    return specs


def main():
    pio.renderers.default = "browser"

    open_buys = pd.read_parquet(OPEN_BUYS_PATH)
    open_buys["dt"] = pd.to_datetime(open_buys["dt"], utc=True)

    execution_tape = pd.read_parquet(EXEC_TAPE_PATH)
    execution_tape["tape_dt"] = pd.to_datetime(execution_tape["tape_dt"], utc=True)
    execution_tape = normalize_execution_tape(execution_tape)
    tape_groups = build_tape_groups(execution_tape)

    discovery_specs = build_discovery_specs()
    discovery_dailies = {}
    discovery_rows = []
    for spec in discovery_specs:
        mask = (
            open_buys["price"].between(
                spec["price_low"],
                spec["price_high"],
                inclusive=cast(Literal["both", "neither", "left", "right"], spec["price_inclusive"]),
            )
            & spec["same_fn"](open_buys)
            & spec["opp_fn"](open_buys)
        )
        trades, daily = backtest_strategy(
            open_buys,
            mask,
            tape_groups=tape_groups,
            ladder_price_bounds=(
                spec["price_low"],
                spec["price_high"],
                cast(Literal["both", "neither", "left", "right"], spec["price_inclusive"]),
            ),
        )
        discovery_dailies[spec["name"]] = daily
        discovery_rows.append(
            summarize_run(
                spec["name"],
                trades,
                daily,
                {
                    "price_label": spec["price_label"],
                    "same_label": spec["same_label"],
                    "opp_label": spec["opp_label"],
                },
            )
        )

    discovery_summary = pd.DataFrame(discovery_rows)
    discovery_summary = discovery_summary.sort_values(["net_roi_on_stake", "filled_trades"], ascending=[False, False])
    discovery_summary.to_csv(OUTPUT_DISCOVERY_CSV, index=False)

    plot_names = set(
        discovery_summary[discovery_summary["filled_trades"] >= 40]["name"].head(12).tolist()
    )
    if not plot_names:
        plot_names = set(discovery_summary.head(12)["name"].tolist())

    discovery_plot_dailies = {k: v for k, v in discovery_dailies.items() if k in plot_names}
    bucket_baseline = (
        discovery_summary[(discovery_summary["same_label"] == "same==0") & (discovery_summary["opp_label"] == "opp==0")]
        .sort_values("price_label")
        ["name"]
        .tolist()
    )
    bucket_plot_dailies = {name: discovery_dailies[name] for name in bucket_baseline}
    strategy_sum_daily = build_strategy_sum_daily(discovery_plot_dailies)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.10,
        subplot_titles=(
            "Discovery: Top Configurations (Systematic Grid)",
            "Bucket Baseline (same==0, opp==0)",
            "Sum of Discovery Top Configurations",
        ),
    )

    add_daily_traces(fig, discovery_plot_dailies, row=1, col=1)
    add_daily_traces(fig, bucket_plot_dailies, row=2, col=1)
    if not strategy_sum_daily.empty:
        sum_plot = with_zero_anchor(strategy_sum_daily)
        fig.add_trace(
            go.Scatter(
                x=sum_plot["trade_date"],
                y=sum_plot["cum_net_pnl_usdc"],
                mode="lines",
                name="all_strategies_sum",
                line=dict(width=3),
                hovertemplate="%{x|%Y-%m-%d}<br>All Strategies Sum<br>Cumulative PnL: %{y:.2f} USDC<extra></extra>",
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        title="Trade Signal PnL Time Series",
        template="plotly_white",
        height=1300,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=40, t=100, b=60),
    )
    fig.update_yaxes(title_text="Cumulative PnL (USDC)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative PnL (USDC)", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative PnL (USDC)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    fig.write_html(OUTPUT_HTML, include_plotlyjs=True, auto_open=False)
    fig.show(renderer="browser")
    print(OUTPUT_HTML)
    print(OUTPUT_DISCOVERY_CSV)


if __name__ == "__main__":
    main()
