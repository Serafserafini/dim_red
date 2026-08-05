"""
Unit tests for Uniform Manifold Approximation and Projection (UMAP).
"""

import numpy as np
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


def test_umap_accepts_optional_hyperparams(sample_data):
    umap = UMAP(
        n_components=2, n_neighbors=5, min_dist=0.05, metric="cosine", random_state=0
    )
    X_transformed = umap.fit_transform(sample_data)
    assert X_transformed.shape == (50, 2)


def test_umap_optional_hyperparams_default_to_none():
    umap = UMAP(n_components=2)
    assert umap.n_neighbors is None
    assert umap.min_dist is None
    assert umap.metric is None
    assert umap.random_state is None


def test_umap_random_state_gives_reproducible_embedding(sample_data):
    umap_a = UMAP(n_components=2, random_state=0)
    umap_b = UMAP(n_components=2, random_state=0)
    np.testing.assert_array_equal(
        umap_a.fit_transform(sample_data), umap_b.fit_transform(sample_data)
    )
