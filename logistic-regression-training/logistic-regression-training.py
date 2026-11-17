import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))


def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    N, D = X.shape

    # Initialize weights and bias
    w = np.zeros(D)
    b = 0.0

    # Gradient Descent
    for _ in range(steps):

        # 1. Calculate predictions
        z = np.dot(X, w) + b
        p = _sigmoid(z)

        # 2. Calculate errors
        error = p - y

        # 3. Calculate gradients
        dw = np.dot(X.T, error) / N
        db = np.sum(error) / N

        # 4. Update weights and bias
        w = w - lr * dw
        b = b - lr * db

    return w, float(b)