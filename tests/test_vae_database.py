"""
Unit tests for VAE dataset/database utilities.
"""

import numpy as np
import pytest
from dim_red.vae.database import VAEDatabase


def test_vae_database_from_array_shape(sample_data):
    db = VAEDatabase.from_array(sample_data)
    assert db.data.shape == (50, 5)
    assert db.data.dtype == np.float32


def test_vae_database_split_sizes(sample_data):
    db = VAEDatabase.from_array(sample_data)
    train_db, val_db = db.train_val_split(val_ratio=0.2, seed=0)
    assert train_db.data.shape[0] + val_db.data.shape[0] == 50
    assert train_db.data.shape[1] == 5
    assert val_db.data.shape[1] == 5


def test_vae_database_invalid_input():
    with pytest.raises(ValueError, match="Expected 2D array"):
        VAEDatabase.from_array(np.array([1.0, 2.0, 3.0]))

