"""
Lee County PD Prediction Heatmap — Flask API
============================================

This server trains three real ML models on startup using CDC WONDER data,
then serves live predictions via REST endpoints.

Models trained:
  1. ARIMA(1,1,0) — fitted to FL G20 AAMR 1999-2022 via MLE (scipy.optimize)
  2. Ordinary Kriging — fitted to ZIP centroids via WLS variogram + Kriging system
  3. Ridge Regression — fitted with LOO-CV lambda selection (sklearn + numpy linalg)

Usage:
  python app.py           # starts on http://localhost:5000
  python app.py --port 8080

Endpoints:
  GET  /api/health                     — server + model status
  GET  /api/models/summary             — training metrics for all 3 models
  GET  /api/forecast?year=2035         — ARIMA forecast for a given year
  GET  /api/forecast/series            — full forecast series 2025-2050 + CI
  GET  /api/kriging?year=2035&n=28     — Kriging grid for a given year
  GET  /api/ridge/scores               — Ridge risk scores for all ZIPs
  GET  /api/zips?year=2035             — all ZIP predictions for a given year
  GET  /api/county?year=2035           — county-level aggregate stats
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

# Import our real ML models and data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdc_wonder_data import YEARS, AAMR_FL, BEBR_POP, ZIP_DATA
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

# CORS — in production this would be restricted to Lee Health domains
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    return response

# ── GLOBAL MODEL STATE ────────────────────────────────────────────────────────
MODELS = {}
TRAIN_TIME = {}

# ── TRAINING ──────────────────────────────────────────────────────────────────
def train_all_models():
    """
    Train all three ML models on startup.
    Called once when the server starts.
    """
    log.info("=" * 60)
    log.info("PDPH API — Training ML Models on Startup")
    log.info("=" * 60)

    # ── 1. ARIMA(1,1,0) ───────────────────────────────────────────────────────
    log.info("[1/3] Fitting ARIMA(1,1,0) to CDC WONDER AAMR 1999–2022 ...")
    t0 = time.time()
    arima = ARIMA110()
    arima.fit(YEARS, AAMR_FL)
    TRAIN_TIME["arima"] = round(time.time() - t0, 3)
    MODELS["arima"] = arima
    s = arima.summary()
    log.info(f"      phi={s['phi']:.4f}  sigma2={s['sigma2']:.4f}  "
             f"AIC={s['aic']:.2f}  BIC={s['bic']:.2f}  [{TRAIN_TIME['arima']}s]")

    # ── 2. Ordinary Kriging ────────────────────────────────────────────────────
    log.info("[2/3] Fitting Ordinary Kriging to ZIP centroids ...")
    t0 = time.time()
    
    # Compute base year (2025) PD burden values for each ZIP control point
    # using ARIMA baseline factor
    arima_base_factor = _get_arima_factor(2025)
    
    coords = np.array([[z[2], z[3]] for z in ZIP_DATA])  # [lat, lng]
    
    # PD prevalence RATE per person at each ZIP (Zhang 2010 applied to ACS age data)
    # Using per-person rates so Kriging values are on a consistent spatial scale
    zhang_rate_65p = (
        0.32 * 5540 + 0.28 * 10800 + 0.20 * 16400 +
        0.12 * 24400 + 0.08 * 29490
    ) / 100000.0  # per person
    zhang_rate_u65 = (0.15 * 2000 + 0.05 * 1000) / 100000.0

    krig_values = []
    for z in ZIP_DATA:
        _, _, _, _, pop, pct65, coastal = z
        pd_rate = (
            pct65 * zhang_rate_65p + (1 - pct65) * zhang_rate_u65
        ) * arima_base_factor
        if coastal:
            pd_rate *= 1.12
        krig_values.append(pd_rate)
    
    krig_values = np.array(krig_values)
    
    kriging = OrdinaryKriging()
    kriging.fit(coords, krig_values)
    TRAIN_TIME["kriging"] = round(time.time() - t0, 3)
    MODELS["kriging"] = kriging
    MODELS["krig_base_values"] = krig_values
    s = kriging.summary()
    log.info(f"      nugget={s['nugget']:.4f}  sill={s['sill']:.4f}  "
             f"range={s['range_param']:.4f}  [{TRAIN_TIME['kriging']}s]")

    # ── 3. Ridge Regression ────────────────────────────────────────────────────
    log.info("[3/3] Fitting Ridge Regression with LOO-CV lambda selection ...")
    t0 = time.time()
    ridge = RidgeRegressionPD()
    ridge.fit(ZIP_DATA, arima_factor=arima_base_factor)
    TRAIN_TIME["ridge"] = round(time.time() - t0, 3)
    MODELS["ridge"] = ridge
    s = ridge.summary()
    log.info(f"      lambda={s['lambda_selected']}  R2={s['r2_train']:.3f}  "
             f"RMSE={s['rmse_train']:.4f}  [{TRAIN_TIME['ridge']}s]")

    log.info("=" * 60)
    log.info("All models trained successfully. API ready.")
    log.info("=" * 60)


# ── DATA HELPERS ──────────────────────────────────────────────────────────────
def _interp_bebr(year):
    """Interpolate BEBR population for any year."""
    keys = sorted(BEBR_POP.keys())
    if year <= keys[0]:
        return BEBR_POP[keys[0]]
    if year >= keys[-1]:
        return BEBR_POP[keys[-1]]
    for i in range(len(keys)-1):
        if keys[i] <= year <= keys[i+1]:
            t = (year - keys[i]) / (keys[i+1] - keys[i])
            return int(BEBR_POP[keys[i]] + t * (BEBR_POP[keys[i+1]] - BEBR_POP[keys[i]]))


def _get_arima_factor(year):
    """
    Get the ARIMA-forecasted AAMR scaling factor for a given year.
    Uses the trained ARIMA model to forecast if year > 2022.
    """
    arima = MODELS.get("arima")
    if arima is None:
        return 1.0

    base_aamr = AAMR_FL[-1]  # 2022 baseline

    if year <= int(YEARS[-1]):
        # Within training range — use fitted value
        idx = np.searchsorted(YEARS, year)
        idx = min(idx, len(YEARS) - 1)
        return float(arima.fitted_values[idx] / base_aamr)
    else:
        # Forecast
        steps = year - int(YEARS[-1])
        forecasts, _, _ = arima.forecast(steps)
        return float(forecasts[-1] / base_aamr)


def _compute_zip_cases(zip_row, year, ridge_score=None):
    """
    Compute estimated PD cases for a ZIP in a given year.
    
    Uses:
      - ARIMA factor (live from trained model)
      - BEBR population growth
      - Zhang 2010 age-specific rates
      - Ridge score adjustment (if available)
    """
    _, _, _, _, pop, pct65, coastal = zip_row
    
    af = _get_arima_factor(year)
    pop_scale = _interp_bebr(year) / BEBR_POP[2023]
    pop_yr = pop * pop_scale
    
    # Zhang 2010 age-specific rates applied to BEBR age structure
    zhang_65p  = (0.32*5540 + 0.28*10800 + 0.20*16400 + 0.12*24400 + 0.08*29490)
    zhang_u65  = (0.15*2000 + 0.05*1000)
    
    cases = (
        pop_yr * pct65 * zhang_65p / 100000 +
        pop_yr * (1 - pct65) * zhang_u65 / 100000
    ) * af
    
    if coastal:
        cases *= 1.12  # Willis 2022
    
    # Ridge adjustment: slight weight toward model-predicted score
    if ridge_score is not None:
        # Ridge adds ±15% adjustment based on residual risk factors
        ridge_adj = 1.0 + 0.15 * (ridge_score - 0.5)
        cases *= ridge_adj
    
    return max(0, round(cases))


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "models_ready": len(MODELS) >= 3,
        "models_trained": list(MODELS.keys()),
        "train_times_sec": TRAIN_TIME,
    })


@app.route("/api/debug")
def debug():
    """Expose training error if models failed to load."""
    return jsonify({
        "models_ready": len(MODELS) >= 3,
        "models_trained": list(MODELS.keys()),
        "train_error": TRAIN_ERROR,
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "files": os.listdir(os.getcwd()),
    })


@app.route("/api/models/summary")
def models_summary():
    if not MODELS:
        return jsonify({"error": "models not trained"}), 503
    return jsonify({
        "arima":   MODELS["arima"].summary(),
        "kriging": MODELS["kriging"].summary(),
        "ridge":   MODELS["ridge"].summary(),
        "training_data": {
            "source": "CDC WONDER G20 AAMR 1999-2022, Florida statewide",
            "n_observations": int(len(YEARS)),
            "zip_control_points": len(ZIP_DATA),
        }
    })


@app.route("/api/forecast")
def forecast_year():
    """Live ARIMA forecast for a single year."""
    try:
        year = int(request.args.get("year", 2035))
    except ValueError:
        return jsonify({"error": "invalid year"}), 400

    year = max(2023, min(2070, year))
    arima = MODELS.get("arima")
    if arima is None:
        return jsonify({"error": "ARIMA model not ready"}), 503

    af = _get_arima_factor(year)
    base_aamr = float(AAMR_FL[-1])

    if year > int(YEARS[-1]):
        steps = year - int(YEARS[-1])
        fc, lo, hi = arima.forecast(steps, confidence=0.80)
        aamr_fc = float(fc[-1])
        aamr_lo = float(lo[-1])
        aamr_hi = float(hi[-1])
    else:
        idx = np.searchsorted(YEARS, year)
        idx = min(idx, len(YEARS)-1)
        aamr_fc = float(arima.fitted_values[idx])
        aamr_lo = aamr_fc
        aamr_hi = aamr_fc

    pop = _interp_bebr(year)
    county_cases = int(_compute_county_total(year))

    return jsonify({
        "year":          year,
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
    """Full ARIMA forecast series 2023-2050 with 80% CI."""
    arima = MODELS.get("arima")
    if arima is None:
        return jsonify({"error": "ARIMA model not ready"}), 503

    # Historical fitted values
    hist_years  = [int(y) for y in YEARS]
    hist_aamr   = [round(float(v), 3) for v in AAMR_FL]
    hist_fitted = [round(float(v), 3) for v in arima.fitted_values]

    # Forecast 2023-2050
    steps = 2050 - int(YEARS[-1])
    fc, lo, hi = arima.forecast(steps, confidence=0.80)
    fc_years = list(range(int(YEARS[-1]) + 1, 2051))

    return jsonify({
        "historical_years":  hist_years,
        "historical_aamr":   hist_aamr,
        "fitted_values":     hist_fitted,
        "forecast_years":    fc_years,
        "forecast_aamr":     [round(float(v), 3) for v in fc],
        "ci_lower_80":       [round(float(v), 3) for v in lo],
        "ci_upper_80":       [round(float(v), 3) for v in hi],
        "arima_summary":     arima.summary(),
    })


@app.route("/api/kriging")
def kriging_grid():
    """
    Live Kriging prediction on a spatial grid for a given year.
    The control point values are scaled by ARIMA factor for the year.
    """
    try:
        year  = int(request.args.get("year", 2025))
        n     = int(request.args.get("n", 28))
    except ValueError:
        return jsonify({"error": "invalid params"}), 400

    year = max(2023, min(2070, year))
    n = max(10, min(50, n))

    kriging = MODELS.get("kriging")
    if kriging is None:
        return jsonify({"error": "Kriging model not ready"}), 503

    # Scale control point values by ARIMA factor ratio (year vs base 2025)
    af_year = _get_arima_factor(year)
    af_base = _get_arima_factor(2025)
    scale   = af_year / af_base if af_base > 0 else 1.0

    base_values = MODELS["krig_base_values"] * scale

    # Refit with scaled values (lightweight — only updates RHS, not variogram)
    # Use the already-fitted variogram but recompute the Kriging weights
    kriging_year = OrdinaryKriging()
    kriging_year.nugget      = kriging.nugget
    kriging_year.sill        = kriging.sill
    kriging_year.range_param = kriging.range_param
    kriging_year.coords_train  = kriging.coords_train.copy()
    kriging_year.values_train  = base_values.copy()
    kriging_year.gamma_matrix  = kriging.gamma_matrix.copy()
    kriging_year.K_inv         = kriging.K_inv.copy()

    # Predict on grid
    pred_grid, var_grid, meta = kriging_year.predict_grid(
        lat_min=26.26, lat_max=26.84,
        lng_min=-82.40, lng_max=-81.50,
        n_grid=n
    )

    return jsonify({
        "year":       year,
        "n_grid":     n,
        "grid":       pred_grid,
        "variance":   var_grid,
        "lat_min":    26.26,
        "lat_max":    26.84,
        "lng_min":    -82.40,
        "lng_max":    -81.50,
        "lats":       meta["lats"],
        "lngs":       meta["lngs"],
        "raw_min":    meta["raw_min"],
        "raw_max":    meta["raw_max"],
        "arima_factor": round(af_year, 5),
        "variogram":  kriging.summary(),
        "model":      "Ordinary Kriging, exponential variogram, WLS fit",
    })


@app.route("/api/ridge/scores")
def ridge_scores():
    """Ridge regression risk scores for all ZIPs."""
    try:
        year = int(request.args.get("year", 2025))
    except ValueError:
        return jsonify({"error": "invalid year"}), 400

    ridge = MODELS.get("ridge")
    if ridge is None:
        return jsonify({"error": "Ridge model not ready"}), 503

    af = _get_arima_factor(year)
    y_hat, scores = ridge.predict(ZIP_DATA, arima_factor=af)

    result = []
    for i, z in enumerate(ZIP_DATA):
        result.append({
            "zip":        z[0],
            "name":       z[1],
            "risk_score": round(float(scores[i]), 4),
            "pd_per1000": round(float(y_hat[i]), 4),
        })

    return jsonify({
        "year":   year,
        "scores": result,
        "model":  ridge.summary(),
    })


@app.route("/api/zips")
def zip_predictions():
    """Full ZIP-level predictions for a given year."""
    try:
        year = int(request.args.get("year", 2025))
    except ValueError:
        return jsonify({"error": "invalid year"}), 400

    year = max(2023, min(2070, year))

    ridge = MODELS.get("ridge")
    af = _get_arima_factor(year)

    # Get Ridge scores
    _, scores = ridge.predict(ZIP_DATA, arima_factor=af)

    result = []
    for i, z in enumerate(ZIP_DATA):
        zip_code, name, lat, lng, pop, pct65, coastal = z
        cases    = _compute_zip_cases(z, year, ridge_score=float(scores[i]))
        pop_yr   = int(pop * _interp_bebr(year) / BEBR_POP[2023])
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

    # County aggregate
    total_cases = sum(r["cases"] for r in result)
    arima_s = MODELS["arima"].summary()

    return jsonify({
        "year":          year,
        "bebr_pop":      _interp_bebr(year),
        "total_cases":   total_cases,
        "arima_factor":  round(af, 5),
        "arima_phi":     arima_s["phi"],
        "zips":          result,
        "models_used":   ["ARIMA(1,1,0)", "Ridge Regression", "BEBR interpolation"],
    })


@app.route("/api/county")
def county_stats():
    """County-level aggregate statistics."""
    try:
        year = int(request.args.get("year", 2025))
    except ValueError:
        return jsonify({"error": "invalid year"}), 400

    year = max(2023, min(2070, year))
    af   = _get_arima_factor(year)
    pop  = _interp_bebr(year)
    total = _compute_county_total(year)
    base  = _compute_county_total(2025)

    # ARIMA forecast
    arima = MODELS["arima"]
    if year > int(YEARS[-1]):
        steps = year - int(YEARS[-1])
        fc, lo, hi = arima.forecast(steps)
        aamr_fc = float(fc[-1])
        aamr_lo = float(lo[-1])
        aamr_hi = float(hi[-1])
    else:
        idx = np.searchsorted(YEARS, year)
        idx = min(idx, len(YEARS)-1)
        aamr_fc = float(arima.fitted_values[idx])
        aamr_lo = aamr_fc
        aamr_hi = aamr_fc

    return jsonify({
        "year":         year,
        "bebr_pop":     pop,
        "total_cases":  int(total),
        "cases_2025":   int(base),
        "growth_pct":   round((total/base - 1)*100, 1) if base > 0 else 0,
        "aamr_forecast": round(aamr_fc, 3),
        "aamr_lower80":  round(aamr_lo, 3),
        "aamr_upper80":  round(aamr_hi, 3),
        "arima_factor":  round(af, 5),
    })


def _compute_county_total(year):
    """Sum PD cases across all ZIPs for a given year."""
    ridge = MODELS.get("ridge")
    if ridge is None:
        return 0
    af = _get_arima_factor(year)
    _, scores = ridge.predict(ZIP_DATA, arima_factor=af)
    return sum(
        _compute_zip_cases(z, year, ridge_score=float(scores[i]))
        for i, z in enumerate(ZIP_DATA)
    )


# ── TRAIN ON STARTUP (runs whether invoked via gunicorn or python directly) ───
TRAIN_ERROR = None
try:
    train_all_models()
except Exception as _e:
    import traceback
    TRAIN_ERROR = traceback.format_exc()
    log.error("=" * 60)
    log.error("TRAINING FAILED — full traceback:")
    log.error(TRAIN_ERROR)
    log.error("=" * 60)

# ── ENTRY POINT (local dev only) ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    log.info(f"Starting PDPH API on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
