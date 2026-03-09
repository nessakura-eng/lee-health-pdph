# Parkinson's Disease Prediction Heatmap (PDPH)
### Lee County, Florida — Lee Health Parkinson Program

A real ML-powered public health forecasting tool. Three genuine statistical models train on startup and serve live predictions via a REST API, visualized on an interactive map dashboard.

---

## Architecture

```
GitHub Repo
├── app.py                  ← Flask API (deploy to Render)
├── arima_model.py          ← ARIMA(1,1,0) via scipy MLE
├── kriging_model.py        ← Ordinary Kriging via numpy linalg
├── ridge_model.py          ← Ridge Regression with LOO-CV
├── cdc_wonder_data.py      ← CDC WONDER data + ZIP demographics
├── requirements.txt        ← Python dependencies
├── render.yaml             ← Render deployment config
├── vercel.json             ← Vercel static site config
└── frontend/
    └── index.html          ← Dashboard (deploy to Vercel)
```

**Backend:** Flask on [Render.com](https://render.com) (free tier)  
**Frontend:** Static HTML on [Vercel](https://vercel.com) (free tier)

---

## ML Models

| Model | Method | Library |
|-------|--------|---------|
| ARIMA(1,1,0) | Maximum Likelihood Estimation | `scipy.optimize` |
| Ordinary Kriging | Weighted least squares variogram + Kriging system | `numpy`, `scipy` |
| Ridge Regression | LOO-CV lambda selection via hat matrix | `scikit-learn`, `numpy` |

**Training data:** CDC WONDER G20 (Parkinson's) AAMR, Florida 1999–2022  
**Demographics:** ACS 2019–2023, BEBR/UF 2024 projections, Zhang et al. 2010

---

## Local Development

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/pdph.git
cd pdph

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the API
python app.py
# → Training models...
# → API ready at http://localhost:5000

# 5. Open the dashboard
open frontend/index.html        # Mac
# Or just double-click frontend/index.html in your file explorer
```

---

## Deployment

### Step 1 — Deploy API to Render (free)

1. Go to [render.com](https://render.com) and sign up with GitHub
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Deploy**
5. Copy your service URL: `https://pdph-api.onrender.com`

### Step 2 — Update the API URL

In `frontend/index.html`, find this line and update it:
```javascript
: "https://pdph-api.onrender.com/api";  // ← UPDATE THIS after Render deploy
```
Replace `pdph-api` with your actual Render service name.

### Step 3 — Deploy Dashboard to Vercel (free)

1. Go to [vercel.com](https://vercel.com) and sign up with GitHub
2. Click **Add New → Project**
3. Import your GitHub repo
4. Vercel auto-detects `vercel.json` — click **Deploy**
5. Your dashboard is live at `https://pdph.vercel.app`

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Server and model status |
| `GET /api/models/summary` | Live training metrics (AIC, phi, R², lambda) |
| `GET /api/forecast?year=2035` | ARIMA forecast for a single year |
| `GET /api/forecast/series` | Full 1999–2050 series with 80% CI |
| `GET /api/kriging?year=2035&n=28` | Live 28×28 Kriging grid |
| `GET /api/ridge/scores` | Ridge risk scores for all ZIPs |
| `GET /api/zips?year=2035` | All ZIP predictions |
| `GET /api/county?year=2035` | County aggregate statistics |

---

## Data Sources

- [CDC WONDER](https://wonder.cdc.gov/) Multiple Cause of Death, ICD-10 G20, 1999–2022
- [BEBR/UF](https://bebr.ufl.edu/) Bulletins 198 & 199 (2024) — Lee County projections
- [ACS 2019–2023](https://www.census.gov/programs-surveys/acs/) — 65+ population by ZIP
- Zhang et al. 2010 — Medicare PD prevalence rates (Neuroepidemiology 34:117-123)
- Willis et al. 2022 — Coastal risk adjustment (npj Parkinson's Disease 8:65)

---

> ⚠️ For public health planning and research purposes only. Not for clinical diagnosis.
