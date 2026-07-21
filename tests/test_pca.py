"""
Unit tests for Principal Component Analysis (PCA).
"""

import numpy as np
import pytest
from dim_red.pca import PCA


def test_pca_fit_transform_shape(sample_data):
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(sample_data)

    assert X_transformed.shape == (50, 2)
    assert pca.components_.shape == (2, 5)
    assert len(pca.explained_variance_ratio_) == 2


def test_pca_explained_variance_sum(sample_data):
    pca = PCA(n_components=5)
    pca.fit(sample_data)

    # Sum of all explained variance ratios for 5 components should equal 1.0
    assert pytest.approx(np.sum(pca.explained_variance_ratio_), 1e-5) == 1.0


def test_pca_unfitted_transform_raises(sample_data):
    pca = PCA(n_components=2)
    with pytest.raises(RuntimeError, match="not fitted yet"):
        pca.transform(sample_data)


def test_pca_invalid_n_components(sample_data):
    with pytest.raises(ValueError, match="positive integer"):
        PCA(n_components=0)

    with pytest.raises(ValueError, match="must be <="):
        pca = PCA(n_components=100)
        pca.fit(sample_data)
