# Progress: Polymarket Strategy Plan (2026-08-06)

## TL;DR

Three tracks were kicked off in this session (Tracks W and O, plus the
PLAN.md plan document). Two of the cheap "what to do first" steps are
**done and verified**:

- **W0 (weather metadata enrichment)** — 96,287 markets parsed (99.4%
  parse rate), 55 cities mapped to Wunderground / NOAA / CWA resolution
  sources, 98.4% of markets enriched with a source.
- **W1 (market self-calibration baseline)** — 11.4M weather BUY trades
  analysed. Global market Brier = 0.0906 (vs 0.25 always-0.5 baseline).
  Mid-price bins (0.3-0.7) under-priced by +2-3% on average; this is
  where the forecast model should be able to win.
- **O1 (Finance/Politics distortion diagnosis)** — three tags compared.
  Finance selected-wallet ROI (0.29) is *worse* than rejected (0.40);
  Politics rejected wallets include 1.5-ROI insiders; simple selection
  is broken for both, with a clear fix path (lead-time-aware selection).

The next agent should continue with W2 (forecast archive fetch) and
O2 (corrected Finance/Politics re-test). The Track X (cross-market
consistency) is a fast secondary track that can be parallelised.

---

## Plan / step cards

- `signal_lab/PLAN.md` — full plan with step cards, dependencies,
  pass criteria, and exit gates.

## What this session ran (and where the artefacts are)

| Step | Script | Output | Verifier | Status |
|---|---|---|---|---|
| W0.1 | `weather_fv/w0_1_parse_markets.py --full` | `w0_markets_parsed.parquet`, `w0_unparsed.csv`, `w0_summary.json` | `verify_w0_1.py` | OK — 96,287 markets, 99.4% parse rate, 55 cities |
| W0.2 | `weather_fv/w0_2_resolution_sources.py` | `w0_resolution_sources.json/csv`, `w0_unfetched_cities.csv` | `verify_w0_2.py` | OK — 55/55 cities mapped, 41 with ICAO, 13 with station name, 1 with CWA id |
| W0.3 | `weather_fv/w0_3_combine.py` | `markets_enriched.parquet`, `w0_enriched_summary.json` | `verify_w0_3.py` | OK — 96,287 markets, 94,714 (98.4%) with resolution_source |
| W1   | `weather_fv/w1_calibration.py --full` | `w1_calibration.csv`, `w1_summary.json` | `verify_w1.py` | OK — 11.4M trades, global Brier 0.0906, +2-3% under-pricing in 0.3-0.7 bins |
| O1   | `onchain/o1_diagnose.py --all-tags` | `o1_diagnosis.json`, `o1_summary.json` | `verify_o1.py` | OK — 3 tags diagnosed, 9 metrics per tag |

Run all verifiers in this order:

```bash
.venv/bin/python -m signal_lab.weather_fv.verify_w0_1
.venv/bin/python -m signal_lab.weather_fv.verify_w0_2
.venv/bin/python -m signal_lab.weather_fv.verify_w0_3
.venv/bin/python -m signal_lab.weather_fv.verify_w1
.venv/bin/python -m signal_lab.onchain.verify_o1
```

---

## Key findings (so other agents don't re-derive them)

### Track W — Weather (main)

- **96,287 weather markets** (out of 96,837 in our snapshot, 99.4% parse
  rate on the `Will the {max,min} temperature in {City} be {…} on {date}?`
  template). 55 unique cities. The remaining 0.6% are non-temperature
  markets (hurricane, earthquake, eclipse, wildfire) — out of scope.
- **Resolution source**: 52 cities → Wunderground (default, URL embeds
  ICAO); 2 cities → NOAA (Moscow UUWW, Istanbul LTFM); 1 city → CWA
  (Taipei station 46692). The Wunderground URL pattern is the most
  reliable: it always embeds the 4-letter ICAO code we need to look up
  the observation source.
- **Market calibration is good but biased at mid-prices**:
  - Global Brier = 0.0906 (well below the 0.25 always-0.5 baseline).
  - Reliability gap (outcome_rate − mean_price) by price bin:
    - [0, 0.05): +0.5%
    - [0.05, 0.1): +1.4%
    - [0.1, 0.2): +1.9%
    - [0.2, 0.3): +1.8%
    - [0.3, 0.5): **+2.7%** ← biggest gap
    - [0.5, 0.7): **+2.8%** ← biggest gap
    - [0.7, 0.9): +1.4%
    - [0.9, 0.95): +0.2%
    - [0.95, 1.0): −0.2%
  - Reliability gap by lead-time bucket (all prices): +0.7% to +3.7%
    in the 0-6h bucket (market under-prices right before close);
    smaller (< 1.5%) further out.
  - **Implication**: a forecast model that better estimates P(max ≥ T)
    in the 0.3-0.7 price range, especially in the last 6h, has the
    most room to add value.
- **Data quirk discovered**: `markets.parquet.end_date_iso` is the
  *event date* (00:00 UTC of the day in the question), NOT the on-chain
  resolution timestamp. The market typically closes 12-24h after the
  event date; use `last_condition_trade_ts` in the trade data as the
  resolution proxy when computing lead-time. See `w1_calibration.py`
  for the working implementation.

### Track O — On-chain (bounded)

- **Weather**: simple selection works (selected ROI 0.37 > rejected 0.27).
- **Finance**: simple selection is *inverted* (selected ROI 0.29 <
  rejected 0.40). Only 32 wallets pass selection. Lead-h→outcome IC is
  0.20, suggesting "wait and see" wallets are filtering out.
- **Politics**: rejected wallets have ROI = 1.5 (insiders); selection
  filters them out, missing the actual alpha source. Market
  concentration is 49x (vs 9x in Weather/Finance).
- **Recommended O2 re-test** (one shot, see `onchain/O2_PLAN.md`):
  1. For Finance: drop the `min_buckets` and `min_trade_count`
     constraints, lower `min_buy_roi` and `min_copyable_roi`, and run
     the existing `evaluate_composite.py` per residualization target
     (price + lead_h + market concentration).
  2. For Politics: include the high-ROI one-off wallets but add a
     one-off-market penalty and an outlier-wallet trim; gate on
     directional consistency across train/val/test.

---

## Open questions / things to watch

1. **Trade data structure**: `polygon_trades_processed/0.parquet` and
   the other 15 shards are **sharded by hash, not date** — all shards
   span 2025-01 → 2026-08. So a sample from one shard is fine, but
   not representative. For W1 we explicitly iterated all shards.
2. **API quirks**:
   - CLOB requires a User-Agent header (default Python UA gets 403).
   - Gamma's `tag_slug=weather` returns only 7% of our weather
     markets — *do not rely on it for W0*. The CLOB endpoint
     `GET /markets/{condition_id}` is reliable and 1-call-per-city is
     the right approach.
3. **Trade price**: use `avg_price` (volume-weighted) not `price`
   (the column doesn't exist in the trade data).
4. **Sample size for verifiers**: the W1 calibration uses 11.4M
   weather BUYs (13.4s on a 16-shard scan). For any future W2/W3 run
   on the full data, plan for similar scale.

---

## Next steps (ordered)

1. **W2 — Historical forecast archive (highest priority)**
   Goal: build a per-(city, date, lead_time) forecast table.
   Plan: fetch Open-Meteo historical forecasts for top-K cities
   (ordered by trade count from W1) at multiple issue leads; match
   observations from the resolution source (Wunderground URL pattern
   → ICAO → open-meteo archive `daily` endpoint).
   Estimated: 30-60 minutes of fetch time, batched.
   See `signal_lab/PLAN.md` for the step card.

2. **O2 — Re-test Finance/Politics (medium priority)**
   See `onchain/O2_PLAN.md` (created in this session) for the exact
   plan: corrected selection per O1 findings, evaluate with existing
   `run_strategy` machinery, gate on AGENTS.md criteria.

3. **W3 — Probability model + calibration comparison**
   Build a per-(city, month, lead) Gaussian error model from W2
   forecasts. Compare model Brier vs market Brier (W1 buckets).
   Pass gate: model Brier < market Brier in mid-price / short-lead
   buckets.

4. **W4 — Trade simulation**
   Capital-constrained sizing like `sizing.py`; rule = buy when
   `model_prob − price > edge_thr`. Thr picked on val.

5. **Track X — Cross-market consistency** (cheap, parallel)
   X0 reuse W0 fetcher for events; X1 sum-of-outcomes scan; X2
   YES/NO complementarity; X3 cross-city/cross-day spillover as a
   `DeclarativeStrategy`.

---

## File map

```
signal_lab/
├── PLAN.md                                # full step-by-step plan
├── PROGRESS.md                            # this file
├── weather_fv/
│   ├── w0_1_parse_markets.py              # market structure parser
│   ├── w0_2_resolution_sources.py        # CLOB API fetcher
│   ├── w0_3_combine.py                    # merge
│   ├── w1_calibration.py                  # market self-calibration
│   ├── verify_w0_1.py
│   ├── verify_w0_2.py
│   ├── verify_w0_3.py
│   ├── verify_w1.py
│   ├── w0_markets_parsed.parquet          # 96k markets structured
│   ├── w0_resolution_sources.json         # 55 cities
│   ├── markets_enriched.parquet           # combined
│   ├── w0_unparsed.csv                    # 550 unparseable (out of scope)
│   ├── w1_calibration.csv                 # 54 lead×price buckets
│   └── w{0,1}_*.json                      # summary stats
└── onchain/
    ├── o1_diagnose.py
    ├── o1_diagnosis.json
    ├── o1_summary.json
    ├── verify_o1.py
    └── O2_PLAN.md                         # O2 step plan (next agent)
```
