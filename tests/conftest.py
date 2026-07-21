"""
Pytest fixtures for dim_red test suite.
"""

import numpy as np
import pytest


@pytest.fixture
def sample_data():
    """Generates a synthetic 2D numpy dataset (50 samples, 5 features)."""
    np.random.seed(42)
    # Generate 5D data with linear combinations and noise
    x1 = np.random.randn(50)
    x2 = 2.0 * x1 + np.random.randn(50) * 0.1
    x3 = -1.5 * x1 + np.random.randn(50) * 0.1
    x4 = np.random.randn(50)
    x5 = np.random.randn(50) * 0.5
    return np.column_stack([x1, x2, x3, x4, x5])
