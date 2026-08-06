"""
Unit tests for utility functions.
"""

import numpy as np
import pytest

from dim_red.utils import apply_standardization, fit_standardization, standardize


def test_standardize_shape_and_values(sample_data):
    standardized = standardize(sample_data)
    assert standardized.shape == sample_data.shape

    # Mean should be close to 0, std close to 1
    np.testing.assert_allclose(np.mean(standardized, axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(np.std(standardized, axis=0), 1, atol=1e-7)


def test_standardize_constant_feature():
    data = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    standardized = standardize(data)

    assert standardized.shape == data.shape
    # Constant column (index 1) should remain 0 without causing NaN/Inf
    assert not np.isnan(standardized).any()
    assert not np.isinf(standardized).any()


def test_standardize_invalid_dimensions():
    with pytest.raises(ValueError, match="Expected 2D array"):
        standardize(np.array([1.0, 2.0, 3.0]))


def test_fit_then_apply_standardization_matches_standardize(sample_data):
    mean, std = fit_standardization(sample_data)
    applied = apply_standardization(sample_data, mean, std)
    np.testing.assert_allclose(applied, standardize(sample_data))


def test_apply_standardization_reuses_fit_stats_on_different_data():
    train = np.array([[0.0, 10.0], [2.0, 10.0], [4.0, 10.0]])
    mean, std = fit_standardization(train)

    new = np.array([[2.0, 10.0]])  # exactly the training mean
    applied = apply_standardization(new, mean, std)

    np.testing.assert_allclose(applied, [[0.0, 0.0]], atol=1e-7)


def test_fit_standardization_invalid_dimensions():
    with pytest.raises(ValueError, match="Expected 2D array"):
        fit_standardization(np.array([1.0, 2.0, 3.0]))


def test_apply_standardization_invalid_dimensions():
    mean, std = fit_standardization(np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="Expected 2D array"):
        apply_standardization(np.array([1.0, 2.0]), mean, std)
