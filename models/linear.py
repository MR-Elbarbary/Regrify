import numpy as np
from scipy.stats import t
from statsmodels.stats.diagnostic import het_breuschpagan
class LinearRegression():
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0
        self.n_parameters_ = None

    def _addIntercept(self, X):
        intercept_column = np.ones((X.shape[0], 1))
        return np.hstack((intercept_column, X))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.fit_intercept:
            X = self._addIntercept(X)
        self.n_parameters_ = X.shape[1]
        XTX = X.T @ X
        XTy = X.T @ y
        yTy = y.T @ y
        self.XTX_inv = np.linalg.pinv(XTX)
        self.weights_ = self.XTX_inv @ XTy
        self.SSE_ = (yTy) - self.weights_.T @ XTy
        self.n_samples_ = n = X.shape[0]
        p = self.n_parameters_
        self.df = n-p
        self.residual_variance_ = self.SSE_ / (n - p)
        yMean = np.mean(y)
        self.SST_ = yTy - n * (yMean ** 2)
        self.SSR_ = self.SST_ - self.SSE_

        if self.fit_intercept:
            self.intercept_ = self.weights_[0]
            self.coef_ = self.weights_[1:]
        else:
            self.coef_ = self.weights_
        y_hat = X @ self.weights_
        residuals = y - y_hat
        bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, X)
        self.heteroscedasticity_pval_ = bp_pval
        self.is_Constant_var = bp_pval > 0.05


    def predict(self, X, alpha=0.05, returnCI=True):
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet. Call `fit()` before `predict()`.")
        if not self.is_Constant_var:
            print(f"Warning: Model variance may not be constant (p = {self.heteroscedasticity_pval_:.4f})")
        X = np.asarray(X)

        if self.fit_intercept:
            X = self._addIntercept(X)
        yHat = X @ self.weights_
        if not returnCI:
            return yHat
        tCrit = t.ppf(1 - alpha/2, self.df)
        predVar = np.diag(X @ self.XTX_inv @ X.T)
        sePred = np.sqrt(self.residual_variance_ * (1 + predVar))
        CIUpper = yHat + tCrit * sePred
        CILower = yHat - tCrit * sePred
        return yHat, CILower, CIUpper