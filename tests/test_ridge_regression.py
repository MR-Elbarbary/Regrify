import unittest
import numpy as np
from models.ridge import RidgeRegression

class TestRidgeRegression(unittest.TestCase):
    def test_fit_predict(self):
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])
        model = RidgeRegression(alpha=0.1)
        model.fit(X, y)
        preds = model.predict(X, returnCI=False)
        np.testing.assert_allclose(preds, y, rtol=1e-1)

    def test_regularization_effect(self):
        X = np.random.rand(100, 5)
        y = np.dot(X, np.array([1, 2, 3, 4, 5])) + 0.5
        model1 = RidgeRegression(alpha=0.0)
        model2 = RidgeRegression(alpha=100.0)
        model1.fit(X, y)
        model2.fit(X, y)
        self.assertTrue(np.linalg.norm(model1.weights_) > np.linalg.norm(model2.weights_))

if __name__ == '__main__':
    unittest.main()
