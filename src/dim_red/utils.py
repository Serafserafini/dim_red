"""
Utility functions for data preprocessing and matrix operations.
"""

import numpy as np


def standardize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Standardizes features by centering mean to 0 and scaling variance to 1.

    Args:
        X: Input array of shape (n_samples, n_features).
        eps: Small constant to prevent division by zero for constant features.

    Returns:
        Standardized numpy array of the same shape.

    Raises:
        ValueError: If X is not a 2D numpy array.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < eps] = 1.0

    return (X - mean) / std
