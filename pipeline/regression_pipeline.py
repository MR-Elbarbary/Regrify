import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from models.linear import LinearRegression
from models.polynomial import PolynomialRegression
from models.ridge import RidgeRegression
from tuning import gridSearch
from metrics.evaluation import adjusted_r2_score
import scipy.stats as sts

class RegressionPipeline:
    def __init__(self, degree_range=(1, 5), alphas=[0.1, 1.0, 10.0]):
        self.degrees = list(range(degree_range[0], degree_range[1] + 1))
        self.alphas = alphas
        self.preprocessor = None
        self.model = None
        self.results_ = {}

    def preprocess(self, X):
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
        
        X_processed = self.preprocessor.fit_transform(X)
        return X_processed

    def check_assumptions(self, X, y):
        linear = self._check_linearity(X, y)
        independent = self._check_independence(X)
        return linear, independent

    def _check_linearity(self, X, y):
        # Dummy placeholder
        return True

    def _check_independence(self, X):
        # Dummy placeholder
        return True

    def select_model(self, linear, independent, X_processed, y):
        if linear and independent:
            model = LinearRegression()
            model.fit(X_processed, y)
            self.model = model
        elif not linear and independent:
            params = {'alpha': self.degrees}
            gs = gridSearch(PolynomialRegression(), params, adjusted_r2_score)
            gs.fit(X_processed, y)
            self.model = gs
            
        elif not independent and linear:
            # Do grid search on alpha
            params = {'alpha': self.alphas}
            gs = gridSearch(RidgeRegression(), params, adjusted_r2_score)
            gs.fit(X_processed, y)
            self.model = gs
        else:
            #Do grid search on degree + alpha
            params = {'alpha': self.alphas,
                      'degree': self.degrees}
            gs = gridSearch(PolynomialRegression(baseModel=RidgeRegression), params, adjusted_r2_score)
            gs.fit(X_processed, y)
            self.model = gs

    def fit(self, X, y):
        X_processed = self.preprocess(X)
        linear, independent = self.check_assumptions(X_processed, y)
        self.select_model(linear, independent, X_processed, y)

    def predict(self, X):
        if not self.model:
            raise Exception("Call `fit` first.")
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)