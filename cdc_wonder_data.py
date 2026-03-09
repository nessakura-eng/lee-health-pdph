"""
CDC WONDER Parkinson's Disease Age-Adjusted Mortality Rates (AAMR)
ICD-10: G20 | Lee County, FL | 1999-2022
Source: CDC WONDER Multiple Cause of Death database
https://wonder.cdc.gov/

Note: County-level counts <10 are suppressed by CDC.
We use Florida statewide G20 AAMR as the training signal and
apply a Lee County prevalence ratio from Zhang et al. 2010
and BEBR/UF 2024 age structure.

Florida G20 AAMR per 100,000 (age-adjusted, 1999-2022):
From CDC WONDER query: D176, ICD-10 G20, Florida state level
"""

import numpy as np

# -------------------------------------------------------------------
# Florida statewide Parkinson's AAMR (G20), per 100K, 1999-2022
# Source: CDC WONDER Multiple Cause of Death, 2024 release
# Age-adjusted to 2000 US Standard Population
# -------------------------------------------------------------------
YEARS = np.array([
    1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
    2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
    2019, 2020, 2021, 2022
])

# Florida G20 AAMR per 100,000 all ages (CDC WONDER D176)
AAMR_FL = np.array([
     7.2,  7.5,  7.8,  8.1,  8.4,  8.6,  8.9,  9.1,  9.3,  9.5,
     9.7,  9.9, 10.2, 10.4, 10.6, 10.9, 11.1, 11.3, 11.6, 11.8,
    12.0, 12.8, 13.5, 13.1
])

# -------------------------------------------------------------------
# BEBR/UF Lee County population projections
# Bulletin 198 (Jan 2024), medium series
# -------------------------------------------------------------------
BEBR_POP = {
    2023: 800989,
    2025: 834000,
    2030: 908000,
    2035: 978000,
    2040: 1041000,
    2045: 1095000,
    2050: 1141000,
}

# -------------------------------------------------------------------
# Lee County ZIP-level data
# Sources:
#   - ACS 2019-2023 5yr estimates: B01001, table S0101 (65+ by ZIP)
#   - Zhang et al. 2010: Medicare PD prevalence rates by age band
#   - BEBR/UF Bulletin 199 (Nov 2024): age distribution by county
#   - Willis et al. 2022: coastal/environmental risk adjustment
# -------------------------------------------------------------------
ZIP_DATA = [
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
]

