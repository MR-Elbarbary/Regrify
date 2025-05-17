import numpy as np
import itertools

class CustomGridSearch:
    def __init__(self, model_class, param_grid, scoring):
        """
        model_class: class of the model (e.g., PolynomialRegression, RidgeRegression)
        param_grid: dict of hyperparameters to try
        scoring: a function that takes (model, X, y) and returns a score
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.scoring = scoring
        self.results_ = []
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_model_ = None

    def fit(self, X, y):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            model = self.model_class(**params)
            model.fit(X, y)
            score = self.scoring(model)
            self.results_.append((params, score))
            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = params
                self.best_model_ = model

    def predict(self, X):
        if self.best_model_ is None:
            raise Exception("Call `fit` first.")
        return self.best_model_.predict(X)
