"""
Unit tests for utility functions.
"""

import numpy as np
import pytest
from dim_red.utils import standardize


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
