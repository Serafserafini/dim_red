"""
Unit tests for Uniform Manifold Approximation and Projection (UMAP).
"""

import pytest

pytest.importorskip("umap")

from dim_red.umap import UMAP


def test_umap_fit_transform_shape(sample_data):
    umap = UMAP(n_components=2)
    X_transformed = umap.fit_transform(sample_data)

    assert X_transformed.shape == (50, 2)
    assert umap.embedding_.shape == (50, 2)


def test_umap_unfitted_transform_raises(sample_data):
    umap = UMAP(n_components=2)
    with pytest.raises(RuntimeError, match="not fitted yet"):
        umap.transform(sample_data)


def test_umap_invalid_n_components():
    with pytest.raises(ValueError, match="positive integer"):
        UMAP(n_components=0)
