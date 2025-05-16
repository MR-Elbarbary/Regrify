import numpy as np
def mean_squared_error(model):
    if model.residual_variance_ is None:
        raise ValueError("Model must be fitted before calculating R².")
    return model.residual_variance_
    

def r2_score(model):
    if model.SSE_ is None or model.SST_ is None:
        raise ValueError("Model must be fitted before calculating R².")
    return 1 - model.SSE_ / model.SST_

def adjusted_r2_score(model):
    if model.SSE_ is None or model.SST_ is None:
        raise ValueError("Model must be fitted before calculating adjusted R².")
    n = model.n_samples_
    p = model.n_parameters_
    r2 = r2_score(model)
    return 1 - ((1 - r2) * (n - 1)) / (n - p)
