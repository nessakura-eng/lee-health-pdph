"""
SW Florida PD Prediction Heatmap — Flask API (Multi-County)
===========================================================

Serves live Parkinson's Disease burden predictions for five Southwest
Florida counties: Lee, Charlotte, Collier, Hendry, and Glades.

Training strategy
  - Lee County models (ARIMA + Kriging + Ridge) are trained on startup.
  - The ARIMA model uses Florida statewide AAMR (CDC WONDER) and is shared
    across all counties — it is trained once and reused.
  - Kriging and Ridge models for Charlotte, Collier, Hendry, and Glades are
    trained lazily on first request via GET /api/train?county=<code>.

Endpoints (all accept optional ?county=<code>, default "lee")
  GET  /api/health
  GET  /api/counties                  — list all supported counties
  GET  /api/train?county=<code>       — lazy-train the requested county
  GET  /api/models/summary
  GET  /api/forecast?year=2035
  GET  /api/forecast/series
  GET  /api/kriging?year=2035&n=28
  GET  /api/ridge/scores
  GET  /api/zips?year=2035
  GET  /api/county?year=2035
"""

import argparse
import json
import os
import logging
import sys
import time
from functools import lru_cache

import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdc_wonder_data import YEARS, AAMR_FL, COUNTY_CONFIGS
from arima_model import ARIMA110
from kriging_model import OrdinaryKriging
from ridge_model import RidgeRegressionPD

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("pdph_api")

# ── FLASK APP ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    return response

# ── MODEL STORE ───────────────────────────────────────────────────────────────
# Flat dict with keys like "arima", "lee_kriging", "lee_ridge", "lee_krig_base"
# ARIMA is shared (FL statewide); Kriging + Ridge are per-county.
MODELS = {}
TRAIN_TIME = {}
TRAINED_COUNTIES = set()   # counties whose Kriging + Ridge have been fitted

VALID_COUNTIES = set(COUNTY_CONFIGS.keys())

# Minimum ZIP points required for Kriging to produce a meaningful variogram.
# Counties below this threshold skip Kriging and fall back to heat/bubble layers.
KRIGING_MIN_ZIPS = 3


def _county_key(county, model):
    """Return MODELS dict key for a county-specific model."""
    return f"{county}_{model}"


# ── TRAINING ──────────────────────────────────────────────────────────────────
def _train_arima():
    """Fit the shared ARIMA(1,1,0) model once on FL statewide AAMR data."""
    if "arima" in MODELS:
        return
    log.info("[ARIMA] Fitting ARIMA(1,1,0) to CDC WONDER FL AAMR 1999-2022 ...")
    t0 = time.time()
    arima = ARIMA110()
    arima.fit(YEARS, AAMR_FL)
    TRAIN_TIME["arima"] = round(time.time() - t0, 3)
    MODELS["arima"] = arima
    s = arima.summary()
    log.info(f"       phi={s['phi']:.4f}  sigma2={s['sigma2']:.4f}  "
             f"AIC={s['aic']:.2f}  [{TRAIN_TIME['arima']}s]")


def train_county(county_code):
    """
    Train Kriging + Ridge for the requested county.
    ARIMA is trained first if not already available (shared across counties).
    Safe to call multiple times — already-trained counties are skipped.
    """
    if county_code not in VALID_COUNTIES:
        raise ValueError(f"Unknown county: {county_code}")

    if county_code in TRAINED_COUNTIES:
        log.info(f"[{county_code}] Already trained — skipping.")
        return

    cfg      = COUNTY_CONFIGS[county_code]
    zip_data = cfg["zip_data"]

    log.info("=" * 60)
    log.info(f"Training models for {cfg['name']} ({county_code.upper()})")
    log.info("=" * 60)

    # 1. Shared ARIMA (train once)
    _train_arima()

    arima_base_factor = _get_arima_factor(2025)

    # 2. Ordinary Kriging (skip for counties with too few ZIPs)
    n_zips = len(zip_data)
    if n_zips >= KRIGING_MIN_ZIPS:
        log.info(f"[2/2] Fitting Ordinary Kriging ({n_zips} ZIP control points) ...")
        t0 = time.time()

        coords = np.array([[z[2], z[3]] for z in zip_data])

        # Marras et al. 2018, npj Parkinson's Disease 4:21
        marras_rate_65p = (
            0.32 * 1000 + 0.28 * 1800 + 0.20 * 3000 +
            0.12 * 4100 + 0.08 * 5200
        ) / 100000.0
        marras_rate_u65 = (0.15 * 550 + 0.05 * 250) / 100000.0

        krig_values = []
        for z in zip_data:
            _, _, _, _, pop, pct65, coastal = z
            pd_rate = (
                pct65 * marras_rate_65p + (1 - pct65) * marras_rate_u65
            ) * arima_base_factor
            if coastal:
                pd_rate *= 1.12
            krig_values.append(pd_rate)

        krig_values = np.array(krig_values)

        kriging = OrdinaryKriging()
        kriging.fit(coords, krig_values)
        elapsed = round(time.time() - t0, 3)
        TRAIN_TIME[_county_key(county_code, "kriging")] = elapsed
        MODELS[_county_key(county_code, "kriging")]    = kriging
        MODELS[_county_key(county_code, "krig_base")]  = krig_values
        s = kriging.summary()
        log.info(f"       nugget={s['nugget']:.5f}  sill={s['sill']:.5f}  "
                 f"range={s['range_param']:.4f}  [{elapsed}s]")
    else:
        log.info(f"[Kriging] Skipped — only {n_zips} ZIP(s) (need ≥{KRIGING_MIN_ZIPS})")
        MODELS[_county_key(county_code, "kriging")]   = None
        MODELS[_county_key(county_code, "krig_base")] = None

    # 3. Ridge Regression
    log.info(f"[3/3] Fitting Ridge Regression (LOO-CV) for {cfg['name']} ...")
    t0 = time.time()
    ridge = RidgeRegressionPD()
    ridge.fit(zip_data, arima_factor=arima_base_factor)
    elapsed = round(time.time() - t0, 3)
    TRAIN_TIME[_county_key(county_code, "ridge")] = elapsed
    MODELS[_county_key(county_code, "ridge")] = ridge
    s = ridge.summary()
    log.info(f"       lambda={s['lambda_selected']}  R2={s['r2_train']:.3f}  "
             f"RMSE={s['rmse_train']:.4f}  [{elapsed}s]")

    TRAINED_COUNTIES.add(county_code)
    log.info(f"[{county_code}] Training complete.")


def _ensure_county_trained(county_code):
    """Lazily train a county if not yet done."""
    if county_code not in TRAINED_COUNTIES:
        train_county(county_code)


# ── DATA HELPERS ──────────────────────────────────────────────────────────────
def _interp_bebr(year, county="lee"):
    """Interpolate BEBR medium-series population for any year (county-specific)."""
    bebr = COUNTY_CONFIGS[county]["bebr_pop"]
    keys = sorted(bebr.keys())
    if year <= keys[0]:
        return bebr[keys[0]]
    if year >= keys[-1]:
        return bebr[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= year <= keys[i + 1]:
            t = (year - keys[i]) / (keys[i + 1] - keys[i])
            return int(bebr[keys[i]] + t * (bebr[keys[i + 1]] - bebr[keys[i]]))


def _get_arima_factor(year):
    """ARIMA-forecasted AAMR scaling factor for a given year (shared model)."""
    arima = MODELS.get("arima")
    if arima is None:
        return 1.0
    base_aamr = AAMR_FL[-1]
    if year <= int(YEARS[-1]):
        idx = min(np.searchsorted(YEARS, year), len(YEARS) - 1)
        return float(arima.fitted_values[idx] / base_aamr)
    steps = year - int(YEARS[-1])
    forecasts, _, _ = arima.forecast(steps)
    return float(forecasts[-1] / base_aamr)


def _compute_zip_cases(zip_row, year, county="lee", ridge_score=None):
    """Estimate PD cases for one ZIP in a given year."""
    _, _, _, _, pop, pct65, coastal = zip_row
    af       = _get_arima_factor(year)
    bebr_base = COUNTY_CONFIGS[county]["bebr_pop"][2023]
    pop_scale = _interp_bebr(year, county) / bebr_base
    pop_yr    = pop * pop_scale

    # Marras et al. 2018, npj Parkinson's Disease 4:21
    marras_65p = (0.32 * 1000 + 0.28 * 1800 + 0.20 * 3000 + 0.12 * 4100 + 0.08 * 5200)
    marras_u65 = (0.15 * 550 + 0.05 * 250)

    cases = (
        pop_yr * pct65 * marras_65p / 100000 +
        pop_yr * (1 - pct65) * marras_u65 / 100000
    ) * af

    if coastal:
        cases *= 1.12

    if ridge_score is not None:
        ridge_adj = 1.0 + 0.15 * (ridge_score - 0.5)
        cases *= ridge_adj

    return max(0, round(cases))


def _compute_county_total(year, county="lee"):
    """Sum PD cases across all ZIPs for a given county and year."""
    ridge_key = _county_key(county, "ridge")
    ridge     = MODELS.get(ridge_key)
    if ridge is None:
        return 0
    zip_data = COUNTY_CONFIGS[county]["zip_data"]
    af = _get_arima_factor(year)
    _, scores = ridge.predict(zip_data, arima_factor=af)
    return sum(
        _compute_zip_cases(z, year, county=county, ridge_score=float(scores[i]))
        for i, z in enumerate(zip_data)
    )


def _validate_county(county):
    """Return county string if valid, else None."""
    return county if county in VALID_COUNTIES else None


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "models_ready": "lee" in TRAINED_COUNTIES,
        "trained_counties": sorted(TRAINED_COUNTIES),
        "available_counties": sorted(VALID_COUNTIES),
        "train_times_sec": TRAIN_TIME,
    })


@app.route("/api/counties")
def counties():
    """Return metadata for all supported counties."""
    result = {}
    for code, cfg in COUNTY_CONFIGS.items():
        result[code] = {
            "name":           cfg["name"],
            "fips":           cfg["fips"],
            "zip_count":      len(cfg["zip_data"]),
            "kriging_avail":  len(cfg["zip_data"]) >= KRIGING_MIN_ZIPS,
            "trained":        code in TRAINED_COUNTIES,
            "map_center":     cfg["map_center"],
            "map_zoom":       cfg["map_zoom"],
        }
    return jsonify(result)


@app.route("/api/train")
def train_endpoint():
    """
    Lazily train ML models for the requested county.
    Synchronous — returns when training is complete (~1-4 seconds).
    Safe to call multiple times (no-op if already trained).
    """
    county = request.args.get("county", "lee")
    if _validate_county(county) is None:
        return jsonify({"error": f"Unknown county: {county}"}), 400

    try:
        _ensure_county_trained(county)
    except Exception as e:
        log.error(f"Training failed for {county}: {e}")
        return jsonify({"error": str(e)}), 500

    cfg = COUNTY_CONFIGS[county]
    return jsonify({
        "county":         county,
        "name":           cfg["name"],
        "trained":        True,
        "kriging_avail":  MODELS.get(_county_key(county, "kriging")) is not None,
        "train_times_sec": {
            k: v for k, v in TRAIN_TIME.items()
            if k.startswith(county) or k == "arima"
        },
    })


@app.route("/api/models/summary")
def models_summary():
    county = request.args.get("county", "lee")
    if _validate_county(county) is None:
        return jsonify({"error": f"Unknown county: {county}"}), 400
    if county not in TRAINED_COUNTIES:
        return jsonify({"error": f"{county} models not trained — call /api/train?county={county}"}), 503

    kriging = MODELS.get(_county_key(county, "kriging"))
    return jsonify({
        "county":  county,
        "arima":   MODELS["arima"].summary(),
        "kriging": kriging.summary() if kriging else None,
        "ridge":   MODELS[_county_key(county, "ridge")].summary(),
        "training_data": {
            "source":            "CDC WONDER G20 AAMR 1999-2022, Florida statewide",
            "n_observations":    int(len(YEARS)),
            "zip_control_points": len(COUNTY_CONFIGS[county]["zip_data"]),
        },
    })


@app.route("/api/forecast")
def forecast_year():
    try:
        year = int(request.args.get("year", 2035))
    except ValueError:
        return jsonify({"error": "invalid year"}), 400

    county = request.args.get("county", "lee")
    if _validate_county(county) is None:
        return jsonify({"error": f"Unknown county: {county}"}), 400

    _ensure_county_trained(county)

    year = max(2023, min(2070, year))
    arima = MODELS.get("arima")
    if arima is None:
        return jsonify({"error": "ARIMA model not ready"}), 503

    af       = _get_arima_factor(year)
    base_aamr = float(AAMR_FL[-1])

    if year > int(YEARS[-1]):
        steps = year - int(YEARS[-1])
        fc, lo, hi = arima.forecast(steps, confidence=0.80)
        aamr_fc, aamr_lo, aamr_hi = float(fc[-1]), float(lo[-1]), float(hi[-1])
    else:
        idx = min(np.searchsorted(YEARS, year), len(YEARS) - 1)
        aamr_fc = aamr_lo = aamr_hi = float(arima.fitted_values[idx])

    pop          = _interp_bebr(year, county)
    county_cases = int(_compute_county_total(year, county))

    return jsonify({
        "year":          year,
        "county":        county,
        "aamr_forecast": round(aamr_fc, 3),
        "aamr_lower80":  round(aamr_lo, 3),
        "aamr_upper80":  round(aamr_hi, 3),
        "arima_factor":  round(af, 5),
        "bebr_pop":      pop,
        "county_cases":  county_cases,
        "model":         "ARIMA(1,1,0) MLE via scipy.optimize L-BFGS-B",
    })


@app.route("/api/forecast/series")
def forecast_series():
    arima = MODELS.get("arima")
    if arima is None:
        _train_arima()
        arima = MODELS.get("arima")
    if arima is None:
        return jsonify({"error": "ARIMA model not ready"}), 503

    hist_years  = [int(y) for y in YEARS]
    hist_aamr   = [round(float(v), 3) for v in AAMR_FL]
    hist_fitted = [round(float(v), 3) for v in arima.fitted_values]

    steps   = 2050 - int(YEARS[-1])
    fc, lo, hi = arima.forecast(steps, confidence=0.80)
    fc_years   = list(range(int(YEARS[-1]) + 1, 2051))

    return jsonify({
        "historical_years": hist_years,
        "historical_aamr":  hist_aamr,
        "fitted_values":    hist_fitted,
        "forecast_years":   fc_years,
        "forecast_aamr":    [round(float(v), 3) for v in fc],
        "ci_lower_80":      [round(float(v), 3) for v in lo],
        "ci_upper_80":      [round(float(v), 3) for v in hi],
        "arima_summary":    arima.summary(),
    })


@app.route("/api/kriging")
def kriging_grid():
    try:
        year = int(request.args.get("year", 2025))
        n    = int(request.args.get("n", 28))
    except ValueError:
        return jsonify({"error": "invalid params"}), 400

    county = request.args.get("county", "lee")
    if _validate_county(county) is None:
        return jsonify({"error": f"Unknown county: {county}"}), 400

    _ensure_county_trained(county)

    kriging = MODELS.get(_county_key(county, "kriging"))
    if kriging is None:
        return jsonify({
            "error": "kriging_unavailable",
            "reason": f"{COUNTY_CONFIGS[county]['name']} has too few ZIP control points for Kriging.",
        }), 422

    year = max(2023, min(2070, year))
    n    = max(10, min(50, n))

    af_year = _get_arima_factor(year)
    af_base = _get_arima_factor(2025)
    scale   = af_year / af_base if af_base > 0 else 1.0

    base_values  = MODELS[_county_key(county, "krig_base")] * scale
    kriging_year = OrdinaryKriging()
    kriging_year.nugget      = kriging.nugget
    kriging_year.sill        = kriging.sill
    kriging_year.range_param = kriging.range_param
    kriging_year.coords_train  = kriging.coords_train.copy()
    kriging_year.values_train  = base_values.copy()
    kriging_year.gamma_matrix  = kriging.gamma_matrix.copy()
    kriging_year.K_inv         = kriging.K_inv.copy()

    bounds = COUNTY_CONFIGS[county]["map_bounds"]
    pred_grid, var_grid, meta = kriging_year.predict_grid(
        lat_min=bounds["lat_min"], lat_max=bounds["lat_max"],
        lng_min=bounds["lng_min"], lng_max=bounds["lng_max"],
        n_grid=n
    )

    return jsonify({
        "year":         year,
        "county":       county,
        "n_grid":       n,
        "grid":         pred_grid,
        "variance":     var_grid,
        "lat_min":      bounds["lat_min"],
        "lat_max":      bounds["lat_max"],
        "lng_min":      bounds["lng_min"],
        "lng_max":      bounds["lng_max"],
        "lats":         meta["lats"],
        "lngs":         meta["lngs"],
        "raw_min":      meta["raw_min"],
        "raw_max":      meta["raw_max"],
        "arima_factor": round(af_year, 5),
        "variogram":    kriging.summary(),
        "model":        "Ordinary Kriging, exponential variogram, WLS fit",
    })


@app.route("/api/ridge/scores")
def ridge_scores():
    try:
        year = int(request.args.get("year", 2025))
    except ValueError:
        return jsonify({"error": "invalid year"}), 400

    county = request.args.get("county", "lee")
    if _validate_county(county) is None:
        return jsonify({"error": f"Unknown county: {county}"}), 400

    _ensure_county_trained(county)

    ridge = MODELS.get(_county_key(county, "ridge"))
    if ridge is None:
        return jsonify({"error": "Ridge model not ready"}), 503

    zip_data = COUNTY_CONFIGS[county]["zip_data"]
    af       = _get_arima_factor(year)
    y_hat, scores = ridge.predict(zip_data, arima_factor=af)

    result = [
        {
            "zip":        z[0],
            "name":       z[1],
            "risk_score": round(float(scores[i]), 4),
            "pd_per1000": round(float(y_hat[i]), 4),
        }
        for i, z in enumerate(zip_data)
    ]
    return jsonify({"year": year, "county": county, "scores": result, "model": ridge.summary()})


@app.route("/api/zips")
def zip_predictions():
    try:
        year = int(request.args.get("year", 2025))
    except ValueError:
        return jsonify({"error": "invalid year"}), 400

    county = request.args.get("county", "lee")
    if _validate_county(county) is None:
        return jsonify({"error": f"Unknown county: {county}"}), 400

    _ensure_county_trained(county)

    year     = max(2023, min(2070, year))
    zip_data = COUNTY_CONFIGS[county]["zip_data"]
    af       = _get_arima_factor(year)

    ridge = MODELS[_county_key(county, "ridge")]
    _, scores = ridge.predict(zip_data, arima_factor=af)

    bebr_base = COUNTY_CONFIGS[county]["bebr_pop"][2023]
    result = []
    for i, z in enumerate(zip_data):
        zip_code, name, lat, lng, pop, pct65, coastal = z
        cases    = _compute_zip_cases(z, year, county=county, ridge_score=float(scores[i]))
        pop_yr   = int(pop * _interp_bebr(year, county) / bebr_base)
        pop65_yr = int(pop_yr * pct65)
        result.append({
            "zip":        zip_code,
            "name":       name,
            "lat":        lat,
            "lng":        lng,
            "pop":        pop_yr,
            "pop65":      pop65_yr,
            "pct65":      round(pct65, 3),
            "coastal":    bool(coastal),
            "cases":      cases,
            "risk_score": round(float(scores[i]), 4),
        })

    total_cases = sum(r["cases"] for r in result)
    arima_s     = MODELS["arima"].summary()
    kriging_avail = MODELS.get(_county_key(county, "kriging")) is not None

    return jsonify({
        "year":           year,
        "county":         county,
        "county_name":    COUNTY_CONFIGS[county]["name"],
        "fips":           COUNTY_CONFIGS[county]["fips"],
        "bebr_pop":       _interp_bebr(year, county),
        "total_cases":    total_cases,
        "arima_factor":   round(af, 5),
        "arima_phi":      arima_s["phi"],
        "kriging_avail":  kriging_avail,
        "zips":           result,
        "models_used":    ["ARIMA(1,1,0)", "Ridge Regression", "BEBR interpolation"],
    })


@app.route("/api/county")
def county_stats():
    try:
        year = int(request.args.get("year", 2025))
    except ValueError:
        return jsonify({"error": "invalid year"}), 400

    county = request.args.get("county", "lee")
    if _validate_county(county) is None:
        return jsonify({"error": f"Unknown county: {county}"}), 400

    _ensure_county_trained(county)

    year  = max(2023, min(2070, year))
    af    = _get_arima_factor(year)
    pop   = _interp_bebr(year, county)
    total = _compute_county_total(year, county)
    base  = _compute_county_total(2025, county)

    arima = MODELS["arima"]
    if year > int(YEARS[-1]):
        steps = year - int(YEARS[-1])
        fc, lo, hi = arima.forecast(steps)
        aamr_fc, aamr_lo, aamr_hi = float(fc[-1]), float(lo[-1]), float(hi[-1])
    else:
        idx = min(np.searchsorted(YEARS, year), len(YEARS) - 1)
        aamr_fc = aamr_lo = aamr_hi = float(arima.fitted_values[idx])

    return jsonify({
        "year":          year,
        "county":        county,
        "county_name":   COUNTY_CONFIGS[county]["name"],
        "bebr_pop":      pop,
        "total_cases":   int(total),
        "cases_2025":    int(base),
        "growth_pct":    round((total / base - 1) * 100, 1) if base > 0 else 0,
        "aamr_forecast": round(aamr_fc, 3),
        "aamr_lower80":  round(aamr_lo, 3),
        "aamr_upper80":  round(aamr_hi, 3),
        "arima_factor":  round(af, 5),
    })


# ── STARTUP TRAINING (runs under both Gunicorn and python app.py) ─────────────
train_county("lee")

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    log.info(f"Starting PDPH API on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
