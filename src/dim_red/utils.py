"""
Utility functions for feature standardization.
"""

from typing import Tuple

import numpy as np


def fit_standardization(
    X: np.ndarray, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the per-feature mean/std used for z-score standardization.

    Kept separate from ``apply_standardization`` so the same statistics fit
    on one array (e.g. a model's training set) can be reapplied to a
    different array (e.g. new structures encoded with an already-trained
    model) -- see ``dim_red.pipeline.inference``.

    Args:
        X: Input array of shape (n_samples, n_features).
        eps: Small constant below which a feature's std is treated as 0
            (constant feature), mapped to 1.0 to avoid dividing by zero.

    Returns:
        ``(mean, std)``, each shape ``(n_features,)``.

    Raises:
        ValueError: If X is not a 2D numpy array.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < eps] = 1.0
    return mean, std


def apply_standardization(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Applies previously-fit mean/std (see ``fit_standardization``) to ``X``.

    Args:
        X: Input array of shape (n_samples, n_features).
        mean: Per-feature mean, shape ``(n_features,)``.
        std: Per-feature std, shape ``(n_features,)``.

    Returns:
        Standardized numpy array, same shape as ``X``.

    Raises:
        ValueError: If X is not a 2D numpy array.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
    return (X - mean) / std
