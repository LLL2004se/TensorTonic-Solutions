import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:

    x = np.asarray(x, dtype=float)

    if rng is not None:
        random_values = rng.random(x.shape)
    else:
        random_values = np.random.random(x.shape)

    mask = (random_values >= p).astype(float) / (1 - p)

    output = x * mask

    return output, mask