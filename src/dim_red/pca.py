"""
Principal Component Analysis (PCA) module.
"""

from typing import Optional
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA



class PCA:
    """Principal Component Analysis (PCA) for dimensionality reduction.
    """

    def __init__(self, n_components: int):
        """Initializes the PCA estimator.

        Args:
            n_components: Number of principal components to keep.

        Raises:
            ValueError: If n_components is not a positive integer.
        """
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        self.n_components = n_components
        self._pca: Optional[SklearnPCA] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCA":
        """Fits the PCA model with X.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            The fitted PCA instance.

        Raises:
            ValueError: If n_components is greater than the number of features in X.
        """
        X_arr = np.asarray(X)
        n_samples, n_features = X_arr.shape
        if self.n_components > n_features:
            raise ValueError(f"n_components must be <= number of features ({n_features})")

        self._pca = SklearnPCA(n_components=self.n_components)
        self._pca.fit(X_arr)
        self.components_ = self._pca.components_
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
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
        if self._pca is None:
            raise RuntimeError("PCA is not fitted yet")
        return self._pca.transform(np.asarray(X))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model with X and apply the dimensionality reduction on X.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)