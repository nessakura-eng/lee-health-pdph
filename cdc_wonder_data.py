"""
CDC WONDER Parkinson's Disease Age-Adjusted Mortality Rates (AAMR)
ICD-10: G20 | Southwest Florida (Lee, Charlotte, Collier, Hendry, Glades) | 1999-2022

Data sources used throughout:
  - CDC WONDER Multiple Cause of Death (ICD-10 G20), Florida statewide AAMR 1999-2022
  - BEBR/UF Bulletin 198 (Jan 2024) medium series — county population projections
  - BEBR/UF Bulletin 199 (Nov 2024) — age/sex distribution by county
  - ACS 2019-2023 5-year estimates — 65+ population by ZIP (Tables B01001 / S0101)
  - Marras et al. 2018, npj Parkinson's Disease 4:21 — age-specific PD prevalence (North American claims data)
  - Willis et al. 2022, npj Parkinson's Disease 8:65 — coastal environmental risk (+12%)
  - NOAA coastal county designation / USGS geography — coastal ZIP classification
"""

import numpy as np

# ---------------------------------------------------------------------------
# Florida statewide Parkinson's AAMR (G20), per 100K, 1999-2022
# Shared training signal across all counties (CDC WONDER D176 query,
# age-adjusted to 2000 US Standard Population, 2024 release)
# ---------------------------------------------------------------------------
YEARS = np.array([
    1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
    2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
    2019, 2020, 2021, 2022
])

AAMR_FL = np.array([
     7.2,  7.5,  7.8,  8.1,  8.4,  8.6,  8.9,  9.1,  9.3,  9.5,
     9.7,  9.9, 10.2, 10.4, 10.6, 10.9, 11.1, 11.3, 11.6, 11.8,
    12.0, 12.8, 13.5, 13.1
])

# ---------------------------------------------------------------------------
# Per-county configurations
# ---------------------------------------------------------------------------
# ZIP tuple format: (zip_code, community_name, lat, lng, pop, pct65, coastal)
#   zip_code  : ZCTA5 code (string)
#   lat / lng : ZCTA5 centroid (WGS-84, decimal degrees)
#   pop       : ACS 2019-2023 5yr total population estimate
#   pct65     : fraction of population aged 65+  (ACS 2019-2023 5yr, Table S0101)
#   coastal   : 1 if ZCTA touches Gulf of Mexico or tidal estuary (Willis 2022 adj.)
#
# BEBR_POP keys: integer years; values: medium-series county population
#   Source: BEBR/UF Bulletin 198 (Jan 2024), medium projection series
#
# map_bounds  : tight bounding box for the Kriging prediction grid
# map_center  : [lat, lng] initial Leaflet view center
# map_zoom    : initial Leaflet zoom level
# ---------------------------------------------------------------------------

COUNTY_CONFIGS = {

    # ── LEE COUNTY (FIPS 12071) ─────────────────────────────────────────────
    "lee": {
        "name": "Lee County",
        "fips": "12071",
        "bebr_pop": {
            2023: 800989,
            2025: 834000,
            2030: 908000,
            2035: 978000,
            2040: 1041000,
            2045: 1095000,
            2050: 1141000,
        },
        "zip_data": [
            # zip,   name,                   lat,     lng,      pop,    pct65, coastal
            ("33901", "Fort Myers Downtown",  26.628, -81.882,  48200,  0.162, 0),
            ("33903", "N. Fort Myers S.",     26.655, -81.880,  34100,  0.198, 0),
            ("33904", "Cape Coral SE",        26.596, -81.958,  29800,  0.221, 1),
            ("33905", "Fort Myers E./Buck.",  26.672, -81.800,  61400,  0.148, 0),
            ("33907", "Fort Myers/Villas",    26.568, -81.898,  44600,  0.241, 0),
            ("33908", "Iona/San Carlos",      26.498, -81.958,  52300,  0.285, 1),
            ("33909", "Cape Coral N.",        26.660, -81.978,  61200,  0.168, 0),
            ("33912", "Fort Myers/San Carlos",26.538, -81.888,  38900,  0.212, 0),
            ("33913", "Gateway/Estero",       26.538, -81.760,  72400,  0.198, 0),
            ("33914", "Cape Coral SW",        26.558, -82.022,  58800,  0.231, 1),
            ("33916", "Fort Myers Central",   26.612, -81.862,  22100,  0.178, 0),
            ("33917", "N. Fort Myers N.",     26.718, -81.978,  48700,  0.225, 0),
            ("33919", "McGregor/Cypress Lake",26.518, -81.912,  41200,  0.308, 0),
            ("33920", "Alva",                 26.728, -81.612,  18900,  0.158, 0),
            ("33921", "Boca Grande",          26.732, -82.268,   1200,  0.382, 1),
            ("33922", "Bokeelia/Pine Island", 26.648, -82.178,   8200,  0.298, 1),
            ("33924", "Captiva",              26.512, -82.198,    512,  0.348, 1),
            ("33928", "Estero",               26.438, -81.818,  34800,  0.298, 0),
            ("33931", "Fort Myers Beach",     26.458, -81.948,   6100,  0.362, 1),
            ("33936", "Lehigh Acres Central", 26.578, -81.678,  12800,  0.112, 0),
            ("33956", "St. James City",       26.492, -82.078,   4800,  0.385, 1),
            ("33957", "Sanibel Island",       26.448, -82.038,   6800,  0.418, 1),
            ("33965", "Estero/Miromar",       26.488, -81.858,  18200,  0.242, 0),
            ("33966", "Fort Myers S./6Mile",  26.562, -81.858,  14800,  0.218, 0),
            ("33967", "Three Oaks/SanCarlos", 26.462, -81.878,  28400,  0.198, 0),
            ("33971", "Lehigh Acres W.",      26.602, -81.728,  42100,  0.142, 0),
            ("33972", "Lehigh Acres NE",      26.632, -81.678,  24800,  0.128, 0),
            ("33973", "Lehigh Acres SE",      26.568, -81.698,  31200,  0.118, 0),
            ("33974", "Lehigh Acres S.",      26.528, -81.698,  22800,  0.108, 0),
            ("33976", "Lehigh Acres W.Ctr",   26.578, -81.738,  19400,  0.132, 0),
            ("33990", "Cape Coral E.",        26.612, -81.968,  18400,  0.188, 0),
            ("33991", "Cape Coral W.",        26.578, -82.038,  24200,  0.212, 1),
            ("33993", "Cape Coral NW/Matl.",  26.648, -82.058,  38800,  0.178, 1),
            ("34134", "Bonita Springs N.",    26.368, -81.842,  22400,  0.318, 1),
            ("34135", "Bonita Springs S.",    26.338, -81.808,  34600,  0.282, 1),
        ],
        "map_bounds": {
            "lat_min": 26.26, "lat_max": 26.84,
            "lng_min": -82.40, "lng_max": -81.50,
        },
        "map_center": [26.52, -81.95],
        "map_zoom": 10,
    },

    # ── CHARLOTTE COUNTY (FIPS 12015) ───────────────────────────────────────
    # Port Charlotte / Punta Gorda / South Gulf Cove
    # One of Florida's oldest counties demographically (~32-34% aged 65+)
    "charlotte": {
        "name": "Charlotte County",
        "fips": "12015",
        "bebr_pop": {
            2023: 192000,
            2025: 200000,
            2030: 219000,
            2035: 234000,
            2040: 248000,
            2045: 259000,
            2050: 268000,
        },
        "zip_data": [
            # zip,   name,                      lat,     lng,      pop,   pct65, coastal
            ("33948", "Port Charlotte Central",  26.967, -82.101,  40000,  0.32, 0),
            ("33950", "Punta Gorda",             26.933, -82.053,  14000,  0.44, 1),
            ("33952", "Port Charlotte N.",       26.993, -82.088,  34000,  0.36, 0),
            ("33953", "Port Charlotte NW",       27.008, -82.162,  18000,  0.26, 0),
            ("33954", "Port Charlotte NE",       27.017, -82.042,  14000,  0.23, 0),
            ("33955", "Burnt Store/PG South",    26.867, -82.099,  11000,  0.40, 1),
            ("33980", "Port Charlotte E.",       26.985, -82.020,  13000,  0.31, 0),
            ("33981", "South Gulf Cove",         26.891, -82.204,  18000,  0.36, 1),
            ("33982", "Babcock Ranch/Rural E.",  26.922, -81.883,   7000,  0.18, 0),
            ("33983", "Deep Creek/NE PG",        27.013, -81.998,  21000,  0.30, 0),
        ],
        "map_bounds": {
            "lat_min": 26.75, "lat_max": 27.07,
            "lng_min": -82.42, "lng_max": -81.75,
        },
        "map_center": [26.93, -82.07],
        "map_zoom": 11,
    },

    # ── COLLIER COUNTY (FIPS 12021) ─────────────────────────────────────────
    # Naples / Marco Island / Immokalee
    # High-wealth retirement destination; ~33.5% aged 65+ (ACS 2019-2023)
    "collier": {
        "name": "Collier County",
        "fips": "12021",
        "bebr_pop": {
            2023: 390000,
            2025: 414000,
            2030: 465000,
            2035: 511000,
            2040: 552000,
            2045: 587000,
            2050: 616000,
        },
        "zip_data": [
            # zip,   name,                    lat,     lng,      pop,   pct65, coastal
            ("34102", "Naples Downtown",       26.138, -81.795,  11000,  0.52, 1),
            ("34103", "Naples/Park Shore",     26.161, -81.820,  14000,  0.54, 1),
            ("34104", "Naples Central",        26.146, -81.763,  26000,  0.35, 0),
            ("34105", "Naples W./Airport",     26.157, -81.807,  21000,  0.42, 0),
            ("34108", "Naples/Pelican Bay",    26.219, -81.810,  16000,  0.57, 1),
            ("34109", "Naples North",          26.238, -81.768,  31000,  0.38, 0),
            ("34110", "North Naples",          26.278, -81.803,  27000,  0.40, 1),
            ("34112", "Naples/Bayshore",       26.103, -81.748,  23000,  0.38, 1),
            ("34113", "Naples/Lely",           26.088, -81.713,  21000,  0.40, 0),
            ("34114", "East Naples",           26.045, -81.664,  17000,  0.33, 0),
            ("34116", "Golden Gate",           26.176, -81.708,  33000,  0.22, 0),
            ("34117", "Golden Gate East",      26.176, -81.623,  16000,  0.18, 0),
            ("34119", "Vineyards/N. Naples",   26.268, -81.728,  36000,  0.37, 0),
            ("34120", "NE Naples/Rural Est.",  26.298, -81.648,  40000,  0.24, 0),
            ("34140", "Goodland/Marco Area",   25.927, -81.645,   3000,  0.36, 1),
            ("34142", "Immokalee",             26.418, -81.418,  24000,  0.12, 0),
            ("34145", "Marco Island",          25.942, -81.718,  16000,  0.51, 1),
        ],
        "map_bounds": {
            "lat_min": 25.85, "lat_max": 26.50,
            "lng_min": -82.00, "lng_max": -81.35,
        },
        "map_center": [26.18, -81.70],
        "map_zoom": 10,
    },

    # ── HENDRY COUNTY (FIPS 12051) ──────────────────────────────────────────
    # LaBelle / Clewiston / Felda
    # Agricultural / rural inland county; younger demographic (~15% aged 65+)
    # Note: Kriging requires ≥3 ZIP control points; county has exactly 3 here.
    "hendry": {
        "name": "Hendry County",
        "fips": "12051",
        "bebr_pop": {
            2023: 43000,
            2025: 44500,
            2030: 47500,
            2035: 50000,
            2040: 52500,
            2045: 54500,
            2050: 56000,
        },
        "zip_data": [
            # zip,   name,          lat,     lng,      pop,   pct65, coastal
            ("33440", "Clewiston",   26.754, -80.934,  12000,  0.09, 0),
            ("33930", "Felda",       26.717, -81.455,   4500,  0.11, 0),
            ("33935", "LaBelle",     26.762, -81.298,  18000,  0.12, 0),
        ],
        "map_bounds": {
            "lat_min": 26.42, "lat_max": 26.98,
            "lng_min": -81.65, "lng_max": -80.75,
        },
        "map_center": [26.72, -81.18],
        "map_zoom": 10,
    },

    # ── GLADES COUNTY (FIPS 12043) ──────────────────────────────────────────
    # Moore Haven / Palmdale
    # Very small rural county; ~14,000 population; no coastal exposure
    # Note: Only 2 ZIP control points — Kriging is disabled for this county;
    #       Heat and Bubble map visualizations remain fully available.
    "glades": {
        "name": "Glades County",
        "fips": "12043",
        "bebr_pop": {
            2023: 14000,
            2025: 14500,
            2030: 15500,
            2035: 16500,
            2040: 17500,
            2045: 18200,
            2050: 19000,
        },
        "zip_data": [
            # zip,   name,          lat,     lng,     pop,   pct65, coastal
            ("33471", "Moore Haven", 26.833, -81.083,  9000,  0.14, 0),
            ("33944", "Palmdale",    27.018, -81.182,  2500,  0.13, 0),
        ],
        "map_bounds": {
            "lat_min": 26.68, "lat_max": 27.10,
            "lng_min": -81.48, "lng_max": -80.68,
        },
        "map_center": [26.90, -81.08],
        "map_zoom": 11,
    },
}

# ---------------------------------------------------------------------------
# Backward-compatibility aliases (used by legacy imports that reference these
# module-level names directly — preserved so existing code doesn't break)
# ---------------------------------------------------------------------------
ZIP_DATA = COUNTY_CONFIGS["lee"]["zip_data"]
BEBR_POP = COUNTY_CONFIGS["lee"]["bebr_pop"]
