import numpy as np
from scipy.stats import f
from itertools import combinations
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

def partial_f_test(X_full, X_restricted, y, model_class, alpha=0.05):
    n = X_full.shape[0]
    K_full = X_full.shape[1]
    K_restricted = X_restricted.shape[1]
    
    model_full = model_class()
    model_full.fit(X_full, y)
    SSR_full = model_full.SSR_

    model_restricted = model_class()
    model_restricted.fit(X_restricted, y)
    SSR_restricted = model_restricted.SSR_
    
    numerator = (SSR_restricted - SSR_full) / (K_full - K_restricted)
    denominator = SSR_full / (n - K_full)
    F = numerator / denominator

    p_value = 1 - f.cdf(F, K_full - K_restricted, n - K_full)
    
    return F ,p_value < alpha

def run_partial_f_tests_by_kept_subset(X, y, model_class, alpha=0.05):
    n_features = X.shape[1]
    full_indices = list(range(n_features))
    results = []

    for k in range(1, n_features):  # only subsets smaller than full set
        for kept in combinations(full_indices, k):
            X_restricted = X[:, kept]
            X_full = X[:, full_indices]

            F, significant = partial_f_test(X_full, X_restricted, y, model_class, alpha)

            results.append({
                'kept_features': kept,
                'F': F,
                'significant': significant
            })
    return results