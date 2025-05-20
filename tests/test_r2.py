import unittest
import numpy as np
from models.linear import LinearRegression
from metrics.evaluation import adjusted_r2_score

class TestAdjustedR2(unittest.TestCase):

    def test_adjusted_r2_score(self):
        # Generate perfect linear data: y = 2x + 3
        np.random.seed(0)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = 2 * X.ravel() + 3

        model = LinearRegression()
        model.fit(X, y)

        # Check that R² and Adjusted R² are both ~1
        r2_adj = adjusted_r2_score(model)
        self.assertAlmostEqual(r2_adj, 1.0, places=5)

    def test_adjusted_r2_penalizes_extra_features(self):
        # Generate linear data and add noise features
        np.random.seed(0)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        noise = np.random.randn(100, 5)
        X_noise = np.hstack((X, noise))  # Add 5 noise features
        y = 2 * X.ravel() + 3

        model = LinearRegression()
        model.fit(X_noise, y)

        r2_adj = adjusted_r2_score(model)

        # Should be less than 1.0 due to noise features
        self.assertLess(r2_adj, 1.0)

if __name__ == '__main__':
    unittest.main()