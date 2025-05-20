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

    def test_sse_sst(self):
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X, returnCI=False)

        # Calculate expected SST and SSE manually
        y_mean = np.mean(y)
        expected_sst = np.sum((y - y_mean) ** 2)
        expected_sse = np.sum((y - preds) ** 2)
        
        self.assertAlmostEqual(model.SSE_, expected_sse, places=5)
        self.assertAlmostEqual(model.SST_, expected_sst, places=5)

if __name__ == '__main__':
    unittest.main()
