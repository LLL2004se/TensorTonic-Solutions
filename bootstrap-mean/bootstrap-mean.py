
import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        raise ValueError("Input list cannot be empty")

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")

    rng = np.random.default_rng(seed)

    # Generate bootstrap samples with replacement
    samples = rng.choice(x, size=(n_bootstrap, len(x)), replace=True)

    # Calculate mean of each bootstrap sample
    bootstrap_means = np.mean(samples, axis=1)

    # Calculate confidence interval percentiles
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return {
        "bootstrap_mean": float(np.mean(bootstrap_means)),
        "lower": float(lower),
        "upper": float(upper)
    }
