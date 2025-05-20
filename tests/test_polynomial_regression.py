import unittest
import numpy as np
from models.polynomial import PolynomialRegression
from models.linear import LinearRegression

class TestPolynomialRegression(unittest.TestCase):
    def test_fit_predict_quadratic(self):
        X = np.array([[1], [2], [3], [4]])
        y = X.flatten() ** 2
        model = PolynomialRegression(degree=2)
        model.fit(X, y)
        preds = model.predict(X, returnCI=False)
        np.testing.assert_allclose(preds, y, rtol=1e-1)

    def test_model_coefficients(self):
        X = np.array([[1], [2], [3]])
        y = X.flatten() ** 2
        model = PolynomialRegression(degree=2)
        model.fit(X, y)
        self.assertEqual(len(model.model.coef_), 1)

if __name__ == '__main__':
    unittest.main()
