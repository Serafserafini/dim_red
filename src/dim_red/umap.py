"""
Uniform Manifold Approximation and Projection (UMAP) module.
"""

from typing import Optional
import numpy as np
from umap import UMAP as SklearnUMAP


class UMAP:
    """Uniform Manifold Approximation and Projection (UMAP) for dimensionality reduction.
    """

    def __init__(self, n_components: int):
        """Initializes the UMAP estimator.

        Args:
            n_components: Number of output dimensions.

        Raises:
            ValueError: If n_components is not a positive integer.
        """
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        self.n_components = n_components
        self._umap: Optional[SklearnUMAP] = None
        self.embedding_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "UMAP":
        """Fits the UMAP model with X.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            The fitted UMAP instance.
        """
        X_arr = np.asarray(X)
        self._umap = SklearnUMAP(n_components=self.n_components)
        self._umap.fit(X_arr)
        self.embedding_ = self._umap.embedding_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction to X.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            Transformed data of shape (n_samples, n_components).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self._umap is None:
            raise RuntimeError("UMAP is not fitted yet")
        return self._umap.transform(np.asarray(X))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model with X and apply the dimensionality reduction on X.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)
