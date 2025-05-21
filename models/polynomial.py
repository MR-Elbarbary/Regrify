import numpy as np
from models.ridge import RidgeRegression
from itertools import combinations_with_replacement

class PolynomialRegression:
    def __init__(self, degree=2, fit_intercept=True, baseModel = RidgeRegression, alpha =0):
        self.degree = degree
        self.fit_intercept = fit_intercept
        self.model = baseModel(alpha=alpha)
        self.powers_ = None

    def _polynomial_features(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape

        combs = list(combinations_with_replacement(range(n_features), self.degree))
        self.powers_ = combs

        X_poly = np.ones((n_samples, len(combs)))
        for i, comb in enumerate(combs):
            X_poly[:, i] = np.prod(X[:, comb], axis=1)
        return X_poly

    def fit(self, X, y):
        X_poly = self._polynomial_features(X)
        self.model.fit(X_poly, y)

    def predict(self, X, alpha=0.05, returnCI = True):
        X_poly = self._polynomial_features(X)
        return self.model.predict(X_poly, alpha=alpha, returnCI=returnCI)

    @property
    def n_samples_(self):
        return getattr(self.model, 'n_samples_', None)

    @property
    def SSE_(self):
        return getattr(self.model, 'SSE_', None)

    @property
    def SST_(self):
        return getattr(self.model, 'SST_', None)

    @property
    def n_parameters_(self):
        return getattr(self.model, 'n_parameters_', None)