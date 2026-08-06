# W2 — Historical forecast archive (next agent)

**Status**: Plan only. The W0 + W1 inputs are ready; W2 is the
heaviest fetch step.

## Goal

For each (city, date) traded on Polymarket (96,287 markets over 18.5
months, 55 cities), produce a row in `weather_fv/forecasts/{city}.parquet`:

```
city, date, issue_time, fcst_max, fcst_min, ensemble (or NaN), observed_max
```

The forecast issue times are a fixed grid: `{D-3 12:00Z, D-2 12:00Z,
D-1 12:00Z, D-0 12:00Z}` (UTC). The observation source is matched to
the resolution source from W0.2 (Wunderground by default; NOAA for
Moscow/Istanbul; CWA for Taipei).

## Procedure

1. **Order cities by traded volume** (descending). Read this from
   `w1_calibration.csv` (sum of `n` by lead × price, by city) or
   re-compute from trade data. Take top-K first (start with K=10 for
   the first run).

2. **Fetch forecasts from Open-Meteo**:
   - Endpoint: `https://archive-api.open-meteo.com/v1/archive`
     (observations, not forecasts).
   - Actually for *forecasts issued in the past*: use
     `https://previous-runs-api.open-meteo.com/v1/previous-runs` with
     parameters:
       - `latitude`, `longitude` (from the city's airport coordinates)
       - `start_date`, `end_date` (the event date range for that city)
       - `daily=temperature_2m_max,temperature_2m_min`
       - `hourly=` not needed for daily max/min
       - The `previous-runs` API returns the forecast issued at the
         requested lead time, *for each day in the range*, as a daily
         series. Each call gives the forecast for a single issue time
         across many dates. So a 2-week range × 1 issue time = 1 call.
   - Coordinates: hard-code the 55 (city, lat, lon) pairs in
     `weather_fv/w2_city_coords.json`. Use known airport coordinates
     (e.g., from openflights.org or Wikipedia).

3. **Fetch observations** for the same (city, date) from the matching
   source:
   - For Wunderground: use Open-Meteo's `archive-api` endpoint with
     the same coordinates. Open-Meteo's archive uses ERA5 reanalysis
     (not raw Wunderground observations) but the MAE between them
     for daily max/min is small (~1°C) for most stations.
   - For NOAA markets (Moscow, Istanbul): the URL pattern
     `https://www.weather.gov/wrh/timeseries?site=UUWW` returns daily
     observations; the `wrh/API/` endpoint gives JSON. (Or use
     Open-Meteo as a fallback.)
   - For Taipei CWA: use Open-Meteo as a fallback (reanalysis is
     good for daily max/min in Taiwan).

4. **Save per city** as `weather_fv/forecasts/{city}.parquet` and
   write `w2_summary.json` with per-city stats (n days, MAE by issue
   lead).

5. **Idempotency / rate limits**: Open-Meteo free tier ≈ 10k calls/day.
   A 90-day window × 1 call per issue time = 4 calls per city-window.
   Top-10 cities × 18 months = ~30 city-windows × 4 = 120 calls. Fine.

## Verification

- `verify_w2.py` checks:
  - per-city file has columns `date, issue_time, fcst_max, observed`
  - `observed` is non-null for ≥ 80% of rows
  - per-city MAE is ≤ 5°C for ≥ 80% of cities (sanity bound)
  - file covers ≥ 80% of (city, date) pairs traded on Polymarket
- Pass gate: coverage ≥ 80% of (city, date) volume; if not, prioritise
  untraded cities and re-run.

## Files

- `weather_fv/w2_fetch_forecasts.py` — main script
- `weather_fv/w2_city_coords.json` — input: city → (lat, lon)
- `weather_fv/w2_fetch_progress.json` — checkpoint / resume state
- `weather_fv/forecasts/{city}.parquet` — per-city output
- `weather_fv/w2_summary.json` — top-level summary
- `weather_fv/verify_w2.py`

## Estimated runtime

- City coordinates: manual lookup, ~5 minutes.
- First run (top-10 cities): ~5–10 minutes of fetch.
- Full run (55 cities): ~30–60 minutes total. Within the 600s
  budget per run if we batch by city; run in the background if
  needed.

## Pre-flight checklist (for the next agent)

- [ ] Confirm Open-Meteo's `previous-runs` API access (free, no key).
- [ ] Add city coordinates (see `w2_city_coords_template.json`).
- [ ] Decide on the first batch: which K=10 cities to prioritise.
- [ ] Add a `Weather market → city date coverage` column to W0.1's
  output (already in `markets_enriched.parquet`).
