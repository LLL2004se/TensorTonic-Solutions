import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.corrcoef(X, rowvar=False)