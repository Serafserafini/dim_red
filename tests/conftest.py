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


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run tests marked slow"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: end-to-end tests that really train (need --runslow)"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
