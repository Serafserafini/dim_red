"""
Principal Component Analysis (PCA) module.
"""

from typing import Optional
import numpy as np


class PCA:
    """Principal Component Analysis (PCA) for dimensionality reduction.

    Attributes:
        n_components: Number of components to keep.
        components_: Principal axes in feature space, shape (n_components, n_features).
        explained_variance_ratio_: Percentage of variance explained by each component.
        mean_: Per-feature empirical mean estimated from the training set.
    """

    def __init__(self, n_components: int):
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCA":
        """Fit the PCA model with X.

        Args:
            X: Training data of shape (n_samples, n_features).

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")

        n_samples, n_features = X.shape
        if self.n_components > min(n_samples, n_features):
            raise ValueError(
                f"n_components={self.n_components} must be <= min(n_samples, n_features)={min(n_samples, n_features)}"
            )

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # SVD: X_centered = U * S * Vt
        _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        self.components_ = Vt[: self.n_components]

        # Calculate explained variance ratio
        total_variance = np.sum(S**2) / (n_samples - 1) if n_samples > 1 else 1.0
        component_variance = (S[: self.n_components] ** 2) / (n_samples - 1) if n_samples > 1 else np.ones(self.n_components)
        self.explained_variance_ratio_ = component_variance / total_variance if total_variance > 0 else np.zeros(self.n_components)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction to X.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            X_new: Transformed array of shape (n_samples, n_components).
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA instance is not fitted yet. Call 'fit' before 'transform'.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")

        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model with X and apply dimensionality reduction on X.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            X_new: Transformed array of shape (n_samples, n_components).
        """
        return self.fit(X).transform(X)
