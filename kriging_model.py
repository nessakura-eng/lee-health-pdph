"""
Ordinary Kriging spatial interpolation — implemented from scratch.

Kriging is a Gaussian process regression method for spatial data.
It finds the Best Linear Unbiased Predictor (BLUP) at unobserved
locations by fitting a variogram model to observed spatial data.

For this application:
  - Control points: ZIP code centroids with PD burden values
  - Variogram model: Exponential  gamma(h) = nugget + sill*(1 - exp(-h/range))
  - Parameters estimated by weighted least squares fit to empirical variogram
  - Kriging system solved via numpy linear algebra (Ax = b)

References:
  Cressie, N. (1993). Statistics for Spatial Data. Wiley.
  Goovaerts, P. (1997). Geostatistics for Natural Resources Evaluation. Oxford.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist


class OrdinaryKriging:
    """
    Ordinary Kriging with exponential variogram model.
    
    Variogram: gamma(h) = nugget + sill * (1 - exp(-h / a))
    where h = lag distance, a = practical range parameter
    
    The Kriging estimator:
      Z*(x0) = sum_i lambda_i * Z(x_i)
    with weights lambda_i satisfying:
      [gamma_ij + 1] [lambda]   [gamma_i0]
      [1           ] [mu   ] = [1       ]
    
    where mu is the Lagrange multiplier for the unbiasedness constraint.
    """
    
    def __init__(self):
        self.nugget = None
        self.sill = None
        self.range_param = None
        self.coords_train = None
        self.values_train = None
        self.variogram_params = None
        self.gamma_matrix = None
        self.n_lags = 10
        
    def _exponential_variogram(self, h, nugget, sill, range_param):
        """
        Exponential variogram model.
        gamma(h) = nugget + sill * (1 - exp(-h / range_param))
        """
        return nugget + sill * (1.0 - np.exp(-np.abs(h) / range_param))
    
    def _compute_empirical_variogram(self, coords, values):
        """
        Compute empirical variogram via method-of-moments.
        gamma(h) ≈ 1/(2*N(h)) * sum_{pairs in bin h} (z_i - z_j)^2
        """
        n = len(values)
        distances = []
        sq_diffs = []
        
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((coords[i,0]-coords[j,0])**2 + (coords[i,1]-coords[j,1])**2)
                sq_diff = (values[i] - values[j])**2
                distances.append(d)
                sq_diffs.append(sq_diff)
        
        distances = np.array(distances)
        sq_diffs  = np.array(sq_diffs)
        
        # Bin into n_lags lag classes
        max_dist = np.percentile(distances, 60)  # use 60% of max range
        bins = np.linspace(0, max_dist, self.n_lags + 1)
        
        lag_centers = []
        gamma_empirical = []
        n_pairs = []
        
        for k in range(self.n_lags):
            mask = (distances >= bins[k]) & (distances < bins[k+1])
            if mask.sum() >= 2:
                lag_centers.append((bins[k] + bins[k+1]) / 2)
                gamma_empirical.append(sq_diffs[mask].mean() / 2)
                n_pairs.append(mask.sum())
        
        return np.array(lag_centers), np.array(gamma_empirical), np.array(n_pairs)
    
    def fit(self, coords, values):
        """
        Fit Kriging model to spatial observations.
        
        coords: array of shape (n, 2), [lat, lng] pairs
        values: array of shape (n,), observed values
        """
        self.coords_train = np.array(coords, dtype=float)
        self.values_train = np.array(values, dtype=float)
        n = len(values)
        
        # Step 1: Fit variogram model
        lags, gamma_emp, weights = self._compute_empirical_variogram(
            self.coords_train, self.values_train
        )
        
        # Weighted least squares fit (weight by number of pairs)
        try:
            total_var = self.values_train.var()
            p0 = [total_var * 0.05,   # nugget: small fraction of total variance
                  total_var * 0.95,   # sill: most of the variance
                  np.median(lags)]    # range: median lag distance
            
            popt, pcov = curve_fit(
                self._exponential_variogram,
                lags, gamma_emp,
                p0=p0,
                sigma=1.0/np.sqrt(weights + 1),  # WLS weights
                bounds=([0, 0, 1e-6], [total_var*2, total_var*5, 10.0]),
                maxfev=5000
            )
            self.nugget, self.sill, self.range_param = popt
        except Exception:
            # Fallback: use sample statistics
            self.nugget = self.values_train.var() * 0.05
            self.sill   = self.values_train.var() * 0.95
            self.range_param = 0.18
        
        self.variogram_params = {
            "nugget": float(self.nugget),
            "sill":   float(self.sill),
            "range":  float(self.range_param),
            "model":  "exponential",
            "n_lags": self.n_lags,
            "lag_distances": lags.tolist(),
            "gamma_empirical": gamma_emp.tolist(),
        }
        
        # Step 2: Build the (n+1) × (n+1) Kriging matrix
        # C[i,j] = gamma(d(xi, xj)) for i,j in 0..n-1
        # Last row/col: Lagrange constraint (unbiasedness)
        dist_matrix = cdist(self.coords_train, self.coords_train)
        C = self._exponential_variogram(dist_matrix, self.nugget, self.sill, self.range_param)
        
        # Build full system matrix with Lagrange row/col
        K = np.zeros((n + 1, n + 1))
        K[:n, :n] = C
        K[n, :n] = 1.0
        K[:n, n] = 1.0
        K[n, n]  = 0.0
        
        self.gamma_matrix = K
        self.K_inv = np.linalg.solve(K, np.eye(n + 1))
        
        return self

    def predict(self, coords_pred):
        """
        Predict at new locations using Kriging weights.
        
        Returns: predictions, kriging variances
        """
        coords_pred = np.array(coords_pred, dtype=float)
        if coords_pred.ndim == 1:
            coords_pred = coords_pred.reshape(1, -1)
        
        n = len(self.values_train)
        preds = []
        variances = []
        
        for x0 in coords_pred:
            # Variogram values between prediction point and data points
            dists = np.sqrt(np.sum((self.coords_train - x0)**2, axis=1))
            gamma_vec = self._exponential_variogram(
                dists, self.nugget, self.sill, self.range_param
            )
            
            # Build RHS: [gamma(x0, x1), ..., gamma(x0, xn), 1]
            b = np.zeros(n + 1)
            b[:n] = gamma_vec
            b[n]  = 1.0
            
            # Solve for weights: lambda = K_inv @ b
            weights = self.K_inv @ b
            
            # Kriging estimate: Z*(x0) = sum_i lambda_i * Z(x_i)
            z_pred = float(np.dot(weights[:n], self.values_train))
            
            # Kriging variance: sigma^2_K = b^T * weights
            krig_var = float(np.dot(b, weights))
            krig_var = max(0.0, krig_var)
            
            preds.append(z_pred)
            variances.append(krig_var)
        
        return np.array(preds), np.array(variances)

    def predict_grid(self, lat_min, lat_max, lng_min, lng_max, n_grid):
        """
        Predict on a regular grid. Returns (n_grid x n_grid) arrays.
        """
        lats = np.linspace(lat_min, lat_max, n_grid)
        lngs = np.linspace(lng_min, lng_max, n_grid)
        
        grid_coords = []
        for lat in lats:
            for lng in lngs:
                grid_coords.append([lat, lng])
        
        preds, variances = self.predict(np.array(grid_coords))
        
        # Normalize predictions to [0, 1] for the heatmap
        pmin, pmax = preds.min(), preds.max()
        if pmax > pmin:
            preds_norm = (preds - pmin) / (pmax - pmin)
        else:
            preds_norm = np.zeros_like(preds)
        
        # Reshape to grid
        pred_grid = preds_norm.reshape(n_grid, n_grid)
        var_grid  = variances.reshape(n_grid, n_grid)
        
        return pred_grid.tolist(), var_grid.tolist(), {
            "lats": lats.tolist(), "lngs": lngs.tolist(),
            "raw_min": float(pmin), "raw_max": float(pmax)
        }

    def summary(self):
        return {
            "model": "Ordinary Kriging",
            "variogram": "exponential",
            "nugget":      round(float(self.nugget), 6),
            "sill":        round(float(self.sill), 6),
            "range_param": round(float(self.range_param), 6),
            "n_control_points": int(len(self.values_train)),
        }
