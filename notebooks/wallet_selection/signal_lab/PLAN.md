# Polymarket Strategy Plan — Weather Forecast Fair Value (main) + Cross-Market + Bounded On-Chain

## Thesis

Weather markets resolve against a documented observation source (Wunderground /
Open-Meteo / METAR station). A calibrated forecast model can produce a *fair*
probability for "highest temperature in city C on day D ≥ T" at any lead time.
Where that fair probability diverges from the market price we have an edge that
is **independent of wallet selection** and **physically grounded** — the
strongest candidate in the project. Cross-market fair price is the cheapest
secondary track; on-chain signal deepening is bounded and runs only as a sanity
check on the prior conclusion that Finance/Politics are distorted under simple
selection.

End goal: **research findings only** (no live execution infrastructure). All
deliverables are reproducible backtest artefacts + written conclusions.

## Step conventions (apply to every step below)

- New work lives under `signal_lab/` in subfolders:
  - `signal_lab/weather_fv/` — Track W (primary)
  - `signal_lab/cross_market/` — Track X (secondary)
  - `signal_lab/onchain/` — Track O (bounded)
- Each step ships a CLI script with `--sample N` to run on a small subset first.
- Each step is checkpointed / resumable.
- Each step writes a `<step>_summary.json` plus a parquet / csv artefact.
- Each step is verifiable by a sibling `verify_<step>.py` that asserts schema
  invariants and — critically — **no lookahead**: any feature joined to a
  trade must have timestamp `< trade.dt`.
- Each step has explicit pass criteria; failing them writes `STOP` to the
  summary and the track halts at its gate.
- Chronological train/val/test splits, evaluated with the existing
  `signal_lab` machinery (`evaluate_signal_panel`, `bootstrap_ic`, etc.)
  wherever possible.

## Track W — Weather forecast fair value (primary)

### W0. Enrich weather market metadata

Goal: turn raw question text into structured rows `(condition_id, city, date,
metric, threshold_lo, threshold_hi, unit, resolution_source, event_slug,
neg_risk)`.

Key finding from pre-flight: the local `markets.parquet` already contains
the necessary structure; only the **resolution source** requires an API call.
This was confirmed by spot-checking the CLOB endpoint
`https://clob.polymarket.com/markets/{condition_id}` and the Gamma
`/events` endpoint. Therefore the step is split into a fast local pass and a
small API pass.

- **W0.1** Local parser (no API)
  - Inputs: `data/markets_processed/markets.parquet` (filter `primary_tag=='Weather'`).
  - Procedure: regex parse city/date/metric/threshold/unit from `question`;
    58 unique cities detected; city clean-up strips trailing `"be"` (a question
    template artefact); date from `end_date_iso`.
  - Outputs: `weather_fv/w0_markets_parsed.parquet`,
    `weather_fv/w0_summary.json` (parse rate, city list, threshold-bucket
    distribution).
  - Verify: parse rate ≥ 99%, manual spot check on 30 questions, city
    distribution sane.
  - Pass: ≥ 99% parse rate; city-set matches expected 50–60 cities.

- **W0.2** API fetch for resolution source (small, one per city × ~3 days)
  - Inputs: 1–3 sample condition_ids per (city, metric).
  - Procedure: CLOB `GET /markets/{condition_id}`; extract station name and
    URL from `description`; store as a (city → station) mapping.
  - Outputs: `weather_fv/w0_resolution_sources.json`.
  - Verify: hand-check 10 (city, station) pairs against the description text.
  - Pass: ≥ 90% of cities get a station assignment.

- **W0.3** Combine
  - Inputs: W0.1 + W0.2.
  - Outputs: `weather_fv/markets_enriched.parquet`.
  - Verify: row count = W0.1 row count; all condition_ids present.

### W1. Market self-calibration baseline (no API, no new data)

Goal: quantify *where* the market is miscalibrated using only existing trade
data + resolved outcomes. Cheapest sanity check on whether Track W has room.

- Inputs: `data/polygon_trades_processed/*.parquet` (BUY trades on Weather
  conditions; merge `token_winner` from `data/markets_processed/markets.parquet`).
- Procedure: for each trade, compute `lead_h = end_date_iso - dt`. Bin by
  `(lead_h bucket, price bin)`. Compute Brier score, reliability, and base
  rate gap. Output as a long-format CSV.
- Outputs: `weather_fv/w1_calibration.csv`, `w1_summary.json`.
- Verify: row counts per bucket; Brier on globally aggregated price is the
  global market Brier; spot-check 3 bucket rows by hand.
- Pass / early-stop gate: market Brier near 0.20 in all lead-time buckets
  (already well-calibrated) → Track W has limited room; document and stop.

### W2. Historical forecast archive fetch (heaviest step; sample first)

Goal: build per-(city, date) historical forecasts at multiple issue leads, plus
the observed value matching the resolution source.

- Inputs: W0.3's city list (ordered by traded volume from W1's bucket stats)
  and resolution source (W0.2). Open-Meteo historical forecast API
  (`previous_runs` endpoint) and the relevant observation endpoint (depends on
  the resolution source). All free / public.
- Procedure:
  - Order cities by traded volume; pick top-K for the first run.
  - For each (city, date), fetch archived forecasts for the issue-lead grid
    `{D-3, D-2, D-1, D-0 12:00Z}` (UTC, before market close).
  - Match the observation source: for "Wunderground KLGA"-style markets, use
    Open-Meteo's historical observations at the matching station.
  - Idempotent, batched, rate-limited (1 req/s default), checkpointed.
- Outputs: `weather_fv/forecasts/{city}.parquet`,
  `weather_fv/w2_summary.json`.
- Verify: forecast vs observed MAE per city is sane (1–2°C); coverage ≥ 90% of
  traded (city, date) volume; 3 (city, date) rows manually cross-checked.
- Pass: top-K cities produce MAE within sane bounds.

### W3. Probability model + calibration comparison

Goal: turn the forecast archive into a per-market probability and compare its
Brier score with the market's Brier score by lead time and price bin.

- Inputs: W2 forecasts, W0.3 enriched markets, W1 calibration buckets.
- Procedure: build a per-(city, month, lead_h) forecast-error model
  (Gaussian with historical σ, or empirical if ensemble available). Compute
  `P(max_temp ≥ threshold | forecast, lead)`. Compare with market prices on
  the same `(lead_h, price bin)` buckets as W1.
- Outputs: `weather_fv/w3_model_probs.parquet`, `w3_calibration_report.json`,
  `w3_brier_comparison.csv` (model vs market by bucket).
- Verify: leakage check — σ fit on train years only; reliability diagram on val.
- Pass gate: model Brier < market Brier in the buckets W1 flagged; same sign
  on test (test opened only once).

### W4. Trade simulation

Goal: translate the model-vs-market gap into a PnL estimate with the same
capital-constrained sizing as `signal_lab/sizing.py`.

- Inputs: W3 model probabilities, W1 trade data with prices.
- Procedure: rule = `BUY` when `model_prob − price > edge_thr` (thr picked on
  val); fills at observed trade prices; $10k capital cap, capital locked to
  resolution; chronological splits. Confounder checks: within-price-bin PnL,
  per-city PnL.
- Outputs: `weather_fv/w4_pnl_report.json`, `w4_pnl_timeseries.csv`.
- Verify: train/val/test PnL directionally consistent; per-city PnL not
  dominated by one city.
- Pass: val net PnL > 0 and test same sign; survives within-price-bin check.

### Final report (after W4)

Single document: `weather_fv/REPORT.md` summarising Tracks W, comparing with
the copy-composite edge, and recommending whether to keep investing.

## Track X — Cross-market fair price (secondary)

- **X0** Reuse W0's fetcher to get `condition_id → event_id` for multi-outcome
  events (Weather/Politics/Finance/Awards/Movies/Music).
- **X1** Intra-event consistency: from trades, reconstruct Σ(outcome prices)
  per event over time; flag events where Σ deviates from 1 by > ε for > Δ
  minutes; quantify frequency × magnitude × available notional.
- **X2** YES/NO complementarity within binary conditions.
- **X3** Cross-city/cross-day weather spillover as a `DeclarativeStrategy`
  (uses W0 city metadata). Reuse the standard `run_strategy` machinery.

Each X step ships the same artefact shape: `<step>_summary.json` + a CSV of
flagged events / signals. Pass gate: capture estimate net of fees is material
(> $1k of expected net capture across the historical period). If not → log and
close the track.

## Track O — On-chain, bounded (parallel, independent)

- **O1** Diagnose Finance/Politics distortion: re-run the existing pipeline
  per tag, decompose (wallet-set size, trade counts, market concentration,
  one-off vs recurring, price-near-resolution share, ROI confounder ICs).
  Output `onchain/o1_diagnosis.json`. Verify: reproduces the known distortion
  first.
- **O2** One-shot re-test: corrected selection per O1 findings + existing
  composite strategy via `run_strategy` (per tag). Pass = AGENTS.md criteria.
  Report and close regardless of outcome.

## Sequencing

```
W0.1 ──► W0.2 ──► W0.3 (independent of W0.1 except inputs)
W1 (no dependencies)
O1 (no dependencies)
─── parallel: W1, W0.2, O1 ───
W0.3 ──► W2 (city list) ──► W3 ──► W4
X0 (uses W0.3) ──► X1, X2, X3
O1 ──► O2
```

## Immediate actions (next 3)

1. **W0.1** — local parser (no API, fast).
2. **W1** — market calibration baseline (no API, fast).
3. **O1** — finance/politics distortion diagnosis (uses existing pipeline).

All three are independent and can be run in parallel.
