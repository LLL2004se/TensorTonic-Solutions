
import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    x = np.asarray(x, dtype=float)

    variance = np.var(x, ddof=1)
    standard_deviation = np.std(x, ddof=1)

    return {
        "variance": float(variance),
        "standard_deviation": float(standard_deviation)
    }
