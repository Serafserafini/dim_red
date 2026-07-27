"""
Utilities to build and split datasets for VAE training.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class VAEDatabase:
    """Container for tabular feature matrices used for VAE training.

    Attributes:
        data: 2D array with shape ``(n_samples, n_features)`` containing
            normalized or raw input vectors consumed by the VAE.
    """

    data: np.ndarray

    @classmethod
    def from_array(cls, X: np.ndarray) -> "VAEDatabase":
        """Build a :class:`VAEDatabase` instance from an array-like object.

        Args:
            X: Array-like input expected to be 2D, shaped as
                ``(n_samples, n_features)``.

        Returns:
            A new :class:`VAEDatabase` with ``float32`` data.

        Raises:
            ValueError: If ``X`` is not a 2D array.
        """

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D array instead.")
        return cls(data=X_arr)

    def train_val_split(
        self,
        val_ratio: float = 0.2,
        seed: int = 42
    ) -> Tuple["VAEDatabase", "VAEDatabase"]:
        """Split dataset into train/validation partitions.

        The split is reproducible thanks to a local RNG initialized with
        ``seed``. At least one sample is guaranteed in both partitions.

        Args:
            val_ratio: Fraction of samples assigned to validation.
                Must be strictly in ``(0, 1)``.
            seed: Random seed for reproducibility.

        Returns:
            A tuple ``(train_db, val_db)`` where each element is a
            :class:`VAEDatabase`.

        Raises:
            ValueError: If ``val_ratio`` is not strictly between 0 and 1.
        """

        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be in the open interval (0, 1)")

        n_samples = self.data.shape[0]
        n_val = max(1, int(round(n_samples * val_ratio)))
        n_val = min(n_val, n_samples - 1)

        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        return (
            VAEDatabase(self.data[train_idx]),
            VAEDatabase(self.data[val_idx]),
        )
