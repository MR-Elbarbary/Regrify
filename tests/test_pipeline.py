import numpy as np
import pandas as pd
from pipeline.regression_pipeline import RegressionPipeline

def test_pipeline_fit_predict():
    df = pd.DataFrame({
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'cat': ['a']*50 + ['b']*50
    })
    y = 5 * df['x1'] + 3 * df['x2'] + np.random.randn(100)

    pipe = RegressionPipeline()
    pipe.fit(df, y)
    preds = pipe.predict(df)

    assert len(preds) == len(y)
    assert hasattr(pipe.model, "predict")
