"""
Real ARIMA(1,1,0) implementation.

ARIMA(p,d,q) = AutoRegressive Integrated Moving Average
  p=1: one lagged value of differenced series
  d=1: first differencing to achieve stationarity
  q=0: no moving-average term

This implements genuine MLE parameter estimation using
scipy.optimize.minimize (L-BFGS-B). This is what statsmodels
does internally — we're building it from scratch with the
same statistical machinery.

Log-likelihood for ARIMA(1,1,0):
  y_t = phi_1 * y_{t-1} + epsilon_t
  epsilon_t ~ N(0, sigma^2)
  L = -n/2 * log(2*pi*sigma^2) - 1/(2*sigma^2) * sum(eps^2)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class ARIMA110:
    """
    ARIMA(1,1,0) model trained via Maximum Likelihood Estimation.
    
    The model:
      1. Differences the series once: dy_t = y_t - y_{t-1}
      2. Fits AR(1) to the differenced series: dy_t = phi * dy_{t-1} + eps
      3. Forecasts by integrating back: y_t = y_{t-1} + predicted_dy_t
    
    Parameters estimated: phi (AR coefficient), sigma2 (noise variance)
    """
    
    def __init__(self):
        self.phi = None
        self.sigma2 = None
        self.mu = None        # mean of differenced series
        self.fitted_values = None
        self.residuals = None
        self.aic = None
        self.bic = None
        self.log_likelihood = None
        self.years_train = None
        self.y_train = None
        self.dy_train = None

    def _neg_log_likelihood(self, params, dy):
        """
        Negative log-likelihood for AR(1) process.
        params = [phi, log_sigma2] (log transform keeps sigma2 > 0)
        """
        phi, log_sigma2 = params
        sigma2 = np.exp(log_sigma2)
        n = len(dy)
        
        # Residuals: eps_t = dy_t - phi * dy_{t-1}
        # We condition on the first observation (exact MLE)
        eps = dy[1:] - phi * dy[:-1]
        
        # Stationary constraint: |phi| < 1
        if abs(phi) >= 0.999:
            return 1e10
        
        # Unconditional variance of AR(1): sigma2 / (1 - phi^2)
        # Used for the likelihood of the first observation
        var0 = sigma2 / (1.0 - phi**2) if abs(phi) < 0.999 else 1e10
        
        # Full log-likelihood (exact, includes first obs)
        ll = (
            -0.5 * np.log(2 * np.pi * var0) - 0.5 * dy[0]**2 / var0
            - (n - 1) * 0.5 * np.log(2 * np.pi * sigma2)
            - 0.5 * np.sum(eps**2) / sigma2
        )
        return -ll

    def fit(self, years, y):
        """
        Fit ARIMA(1,1,0) to observed data y at given years.
        Uses scipy.optimize.minimize with L-BFGS-B.
        """
        self.years_train = np.array(years)
        self.y_train = np.array(y, dtype=float)
        
        # Step 1: First difference (d=1)
        dy = np.diff(self.y_train)
        self.dy_train = dy
        self.mu = dy.mean()
        
        # Demean for AR fitting
        dy_dm = dy - self.mu
        
        # Step 2: MLE via scipy.optimize
        # Initial guess: phi from Yule-Walker, sigma2 from sample variance
        if len(dy_dm) > 1:
            phi0 = np.corrcoef(dy_dm[1:], dy_dm[:-1])[0, 1]
            phi0 = np.clip(phi0, -0.98, 0.98)
        else:
            phi0 = 0.0
        sigma0 = dy_dm.var() if dy_dm.var() > 0 else 0.1
        
        result = minimize(
            self._neg_log_likelihood,
            x0=[phi0, np.log(sigma0)],
            args=(dy_dm,),
            method='L-BFGS-B',
            bounds=[(-0.998, 0.998), (-10, 10)],
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        self.phi = result.x[0]
        self.sigma2 = np.exp(result.x[1])
        self.log_likelihood = -result.fun
        
        # Information criteria
        k = 2  # number of free parameters: phi, sigma2
        n = len(dy_dm)
        self.aic = 2 * k - 2 * self.log_likelihood
        self.bic = k * np.log(n) - 2 * self.log_likelihood
        
        # In-sample fitted values on differenced series
        dy_fitted = np.zeros(len(dy_dm))
        dy_fitted[0] = dy_dm[0]
        for t in range(1, len(dy_dm)):
            dy_fitted[t] = self.phi * dy_dm[t-1]
        
        # Residuals
        self.residuals = dy_dm - dy_fitted
        
        # Reconstruct fitted values on original scale
        self.fitted_values = np.zeros(len(self.y_train))
        self.fitted_values[0] = self.y_train[0]
        for t in range(1, len(self.y_train)):
            self.fitted_values[t] = self.fitted_values[t-1] + self.mu + dy_fitted[t-1]
        
        return self

    def forecast(self, steps, confidence=0.80):
        """
        Multi-step ahead forecast with prediction intervals.
        
        For ARIMA(1,1,0) h-step ahead forecast variance:
          Var(h) = sigma2 * (1 + sum_{j=1}^{h-1} psi_j^2)
        where psi_j = phi^j (MA representation coefficients)
        
        Returns: forecast values, lower CI, upper CI
        """
        if self.phi is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        last_y = self.y_train[-1]
        last_dy_dm = self.dy_train[-1] - self.mu
        
        forecasts = []
        variances = []
        
        # Psi weights for MA(inf) representation of AR(1)
        # psi_0 = 1, psi_j = phi^j for j >= 1
        current_y = last_y
        current_dy = last_dy_dm
        
        cumvar = 0.0  # cumulative forecast variance
        
        for h in range(1, steps + 1):
            # AR(1) forecast on demeaned differenced series
            dy_hat = self.phi * current_dy
            
            # Back to original scale
            y_hat = current_y + self.mu + dy_hat
            forecasts.append(y_hat)
            
            # Forecast variance: sigma2 * sum of psi^2 weights
            # For AR(1): psi_j = phi^(j) for the MA representation
            # h-step variance: sigma2 * (1 + phi^2 + phi^4 + ... + phi^{2(h-1)})
            if abs(self.phi) < 1e-10:
                psi_sum = h
            else:
                psi_sum = (1 - self.phi**(2*h)) / (1 - self.phi**2)
            var_h = self.sigma2 * psi_sum
            variances.append(var_h)
            
            # Update for next step
            current_y = y_hat
            current_dy = dy_hat
        
        forecasts = np.array(forecasts)
        variances = np.array(variances)
        stderrs = np.sqrt(variances)
        
        z = norm.ppf(0.5 + confidence / 2)
        lower = forecasts - z * stderrs
        upper = forecasts + z * stderrs
        
        return forecasts, lower, upper

    def summary(self):
        return {
            "model": "ARIMA(1,1,0)",
            "phi": round(float(self.phi), 6),
            "sigma2": round(float(self.sigma2), 6),
            "mu_diff": round(float(self.mu), 6),
            "log_likelihood": round(float(self.log_likelihood), 4),
            "aic": round(float(self.aic), 4),
            "bic": round(float(self.bic), 4),
            "n_obs": int(len(self.y_train)),
            "training_years": [int(y) for y in self.years_train],
        }
