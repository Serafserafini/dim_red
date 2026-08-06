"""
Pytest fixtures for dim_red test suite.
"""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _fast_optimizer_for_training_tests(monkeypatch):
    """Swap VeLO (the pretrained learned optimizer `vae.training`/
    `autoencoder.training`/`supcon.training` all use as their optimizer
    backend, see `prefab.optax_lopt` in each) for a plain, fast `optax.adam`
    in every test.

    VeLO is a large pretrained hypernetwork -- tracing/compiling it dominates
    the runtime of every real training call in the test suite (~15-18s per
    call observed, vs ~3s with this swap), even though these tests only
    exercise training-loop mechanics (history shape, early stopping,
    finiteness) and don't care which optimizer produced the numbers. All
    three training modules call `prefab.optax_lopt(...)` off the same
    `learned_optimization.research.general_lopt.prefab` module object, so
    patching that one attribute covers all of them.

    A no-op when jax/learned_optimization aren't installed -- the test
    modules that need them already guard with `pytest.importorskip("jax")`.
    """
    try:
        import optax
        from learned_optimization.research.general_lopt import prefab
    except ImportError:
        return

    def _fake_optax_lopt(
        num_steps, weight_decay=0.0, max_training_steps=150_000, base_lopt_fn=None
    ):
        del num_steps, weight_decay, max_training_steps, base_lopt_fn
        return optax.with_extra_args_support(optax.adam(1e-2))

    monkeypatch.setattr(prefab, "optax_lopt", _fake_optax_lopt)


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
