"""
Ridge Regression for ZIP-level Parkinson's Disease risk scoring.

Model: y = X @ beta + epsilon, with L2 penalty lambda * ||beta||^2

Features (X):
  - pct65:    proportion of population aged 65+ (ACS 2019-2023)
  - log_pop:  natural log of ZIP population (BEBR/UF 2024)
  - coastal:  binary flag (Willis et al. 2022 environmental risk)
  - pct65_sq: squared pct65 term (nonlinear age effect)

Target (y):
  - PD cases per 1000 population (derived from Zhang 2010 prevalence rates
    applied to ACS age distribution)

Ridge solution: beta = (X'X + lambda*I)^{-1} X'y

Cross-validation:
  - Leave-one-out (LOO-CV) to select optimal lambda
  - Reports R2, RMSE, MAE on LOO predictions

References:
  Ridge: Hoerl & Kennard (1970). Technometrics.
  Zhang 2010: Neuroepidemiology, 34(2): 117-123.
  Willis 2022: npj Parkinson's Disease, 8:65.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class RidgeRegressionPD:
    """
    Ridge regression with LOO-CV lambda selection.
    Fits zip-level PD risk as a function of demographic features.
    """
    
    def __init__(self, lambdas=None):
        """
        lambdas: list of regularization strengths to search over (LOO-CV)
        """
        if lambdas is None:
            self.lambdas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
        else:
            self.lambdas = lambdas
        
        self.best_lambda = None
        self.beta = None          # coefficients (scaled space)
        self.beta_orig = None     # coefficients (original feature space)
        self.intercept = None
        self.scaler = StandardScaler()
        self.feature_names = ["pct65", "log_pop", "coastal", "pct65_sq"]
        self.cv_scores = {}
        self.r2 = None
        self.rmse = None
        self.mae = None
        self.n_train = None
        
    def _build_features(self, zip_data):
        """
        Build design matrix from ZIP data records.
        zip_data: list of tuples (zip, name, lat, lng, pop, pct65, coastal)
        """
        X = []
        for row in zip_data:
            _, _, _, _, pop, pct65, coastal = row
            X.append([
                pct65,
                np.log(pop + 1),
                float(coastal),
                pct65 ** 2,
            ])
        return np.array(X)
    
    def _compute_target(self, zip_data, arima_factor=1.0):
        """
        Compute PD cases per 1000 population for each ZIP.
        
        Based on Zhang et al. 2010 Medicare age-specific prevalence:
          50-54:  500/100K,  55-59: 1000/100K,  60-64: 2000/100K
          65-69: 5540/100K,  70-74:10800/100K,  75-79:16400/100K
          80-84:24400/100K,  85+:  29490/100K
        
        Applied to BEBR/UF age distribution for Lee County.
        """
        # Zhang 2010 prevalence rates per 100,000 by age band
        zhang_rates = {
            "50_54": 500, "55_59": 1000, "60_64": 2000,
            "65_69": 5540, "70_74": 10800, "75_79": 16400,
            "80_84": 24400, "85p": 29490
        }
        
        # Age distribution within 65+ population (BEBR/UF Bulletin 199, 2024)
        age_dist_65plus = {
            "65_69": 0.32, "70_74": 0.28, "75_79": 0.20,
            "80_84": 0.12, "85p": 0.08
        }
        
        # Weighted avg rate for 65+ (per 100K)
        rate_65plus = sum(
            age_dist_65plus[k] * zhang_rates[k]
            for k in age_dist_65plus
        )
        
        # Rate for under-65 (smaller contribution)
        rate_under65 = 0.15 * zhang_rates["60_64"] + 0.05 * zhang_rates["55_59"]
        
        y = []
        for row in zip_data:
            _, _, _, _, pop, pct65, coastal = row
            pop65 = pop * pct65
            pop_under65 = pop * (1 - pct65)
            
            cases = (
                pop65 * rate_65plus / 100000 +
                pop_under65 * rate_under65 / 100000
            ) * arima_factor
            
            # Per 1000 population (normalized target)
            y.append(cases / pop * 1000 if pop > 0 else 0)
        
        return np.array(y)
    
    def _loo_cv_lambda(self, X_scaled, y):
        """
        Leave-One-Out cross-validation to select best lambda.
        For Ridge, LOO predictions can be computed efficiently via
        the hat matrix: H = X(X'X + lI)^{-1}X'
        LOO residual: e_i^LOO = e_i / (1 - H_ii)
        """
        n = len(y)
        best_lambda = self.lambdas[0]
        best_cv_mse = np.inf
        
        for lam in self.lambdas:
            I = np.eye(X_scaled.shape[1])
            A = X_scaled.T @ X_scaled + lam * I
            try:
                A_inv = np.linalg.solve(A, np.eye(A.shape[0]))
            except np.linalg.LinAlgError:
                continue
            
            H = X_scaled @ A_inv @ X_scaled.T
            beta_lam = A_inv @ X_scaled.T @ y
            y_hat = X_scaled @ beta_lam
            residuals = y - y_hat
            h_diag = np.diag(H)
            
            # LOO residuals (Sherman-Morrison-Woodbury shortcut)
            # Clip h_ii to avoid division by zero
            h_clipped = np.clip(h_diag, 0, 0.999)
            loo_residuals = residuals / (1.0 - h_clipped)
            cv_mse = np.mean(loo_residuals ** 2)
            
            self.cv_scores[lam] = float(cv_mse)
            
            if cv_mse < best_cv_mse:
                best_cv_mse = cv_mse
                best_lambda = lam
        
        return best_lambda
    
    def fit(self, zip_data, arima_factor=1.0):
        """
        Fit Ridge regression to ZIP-level data.
        zip_data: list of (zip, name, lat, lng, pop, pct65, coastal)
        arima_factor: ARIMA scaling factor for the base year
        """
        X = self._build_features(zip_data)
        y = self._compute_target(zip_data, arima_factor)
        
        self.n_train = len(y)
        
        # Add intercept column
        ones = np.ones((len(X), 1))
        X_aug = np.hstack([ones, X])
        
        # Scale features (not intercept)
        X_scaled_feats = self.scaler.fit_transform(X)
        X_scaled = np.hstack([ones, X_scaled_feats])
        
        # LOO-CV lambda selection
        self.best_lambda = self._loo_cv_lambda(X_scaled, y)
        
        # Fit final model with best lambda
        # Only penalize feature coefficients, not intercept
        I = np.eye(X_scaled.shape[1])
        I[0, 0] = 0.0  # don't penalize intercept
        
        A = X_scaled.T @ X_scaled + self.best_lambda * I
        self.beta = np.linalg.solve(A, X_scaled.T @ y)
        
        # Store intercept and feature coefficients
        self.intercept = float(self.beta[0])
        self.beta_feats = self.beta[1:]
        
        # Coefficients in original feature space (for interpretability)
        scale = self.scaler.scale_
        mean  = self.scaler.mean_
        self.beta_orig = self.beta_feats / scale
        self.intercept_orig = self.intercept - np.dot(self.beta_orig, mean)
        
        # Training metrics
        y_hat = X_scaled @ self.beta
        residuals = y - y_hat
        self.r2   = float(r2_score(y, y_hat))
        self.rmse = float(np.sqrt(mean_squared_error(y, y_hat)))
        self.mae  = float(mean_absolute_error(y, y_hat))
        self.y_train = y
        self.y_hat_train = y_hat
        
        return self
    
    def predict(self, zip_data, arima_factor=1.0):
        """
        Predict PD burden for given ZIP records.
        Returns raw predicted value (PD per 1000 pop) and 0-1 normalized risk score.
        """
        X = self._build_features(zip_data)
        ones = np.ones((len(X), 1))
        X_scaled = np.hstack([ones, self.scaler.transform(X)])
        
        y_hat = X_scaled @ self.beta
        
        # Risk score: normalize to [0, 1] based on training range
        y_min = self.y_hat_train.min()
        y_max = self.y_hat_train.max()
        if y_max > y_min:
            scores = (y_hat - y_min) / (y_max - y_min)
        else:
            scores = np.zeros_like(y_hat)
        scores = np.clip(scores, 0, 1)
        
        return y_hat, scores
    
    def summary(self):
        return {
            "model": "Ridge Regression",
            "lambda_selected": float(self.best_lambda),
            "lambda_cv_scores": {str(k): round(v, 6) for k, v in self.cv_scores.items()},
            "features": self.feature_names,
            "coefficients_scaled": {
                self.feature_names[i]: round(float(self.beta_feats[i]), 6)
                for i in range(len(self.feature_names))
            },
            "coefficients_original": {
                self.feature_names[i]: round(float(self.beta_orig[i]), 6)
                for i in range(len(self.feature_names))
            },
            "intercept": round(float(self.intercept_orig), 6),
            "r2_train":  round(self.r2,   4),
            "rmse_train": round(self.rmse, 4),
            "mae_train":  round(self.mae,  4),
            "n_train":    self.n_train,
            "selection_method": "LOO-CV",
        }
