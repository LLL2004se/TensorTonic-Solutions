import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """

    T = np.array(T, dtype=float)
    points = np.array(points, dtype=float)

    # Handle a single point
    if points.ndim == 1:
        homogeneous_point = np.append(points, 1)
        result = T @ homogeneous_point
        return result[:3] / result[3]

    # Handle multiple points
    homogeneous_points = np.column_stack((points, np.ones(len(points))))

    result = (T @ homogeneous_points.T).T

    return result[:, :3] / result[:, 3:4]