import numpy as np
from tuning.gridSearch import CustomGridSearch
from models.ridge import RidgeRegression
from metrics.evaluation import adjusted_r2_score

def test_grid_search_selects_best_model():
    X = np.random.randn(100, 2)
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1

    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
    gs = CustomGridSearch(RidgeRegression, param_grid, adjusted_r2_score)
    gs.fit(X, y)

    assert hasattr(gs.best_model_, "predict")
    y_pred = gs.predict(X)
    assert len(y_pred) == len(y)
