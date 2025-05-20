import unittest
import numpy as np
from models.linear import LinearRegression

class TestLinearRegression(unittest.TestCase):
    def test_fit_and_predict(self):
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])  # y = 2x
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X, returnCI=False)
        np.testing.assert_allclose(preds, y, rtol=1e-1)

    def test_coefficients(self):
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        model = LinearRegression()
        model.fit(X, y)
        self.assertAlmostEqual(model.coef_[0], 2.0, places=2)

if __name__ == '__main__':
    unittest.main()
