import numpy as np
from scipy.stats import t
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
        self.SSE_ = (y.T @ y) - self.weights_.T @ y
        self.n_samples_ = n = X.shape[0]
        p = self.n_parameters_
        self.df = n-p
        self.residual_variance_ = self.SSE_ / (n - p)
        self.SST_ = yTy - n * (XTy[0])**2
        self.SSR_ = self.SST_ - self.SSE_

        if self.fit_intercept:
            self.intercept_ = self.weights_[0]
            self.coef_ = self.weights_[1:]
        else:
            self.coef_ = self.weights_

    def predict(self, X, alpha=0.05, returnCI=True):
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet. Call `fit()` before `predict()`.")
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