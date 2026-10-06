"""
Unit tests for the SupCon encoder-only model.
"""

import pytest

pytest.importorskip("jax")

import jax.numpy as jnp
import numpy as np

from dim_red.supcon.model import SupConEncoder


def test_supcon_encoder_encode_shape():
    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    x = jnp.ones((10, 5), dtype=jnp.float32)
    z = model.encode(x)
    assert z.shape == (10, 3)


def test_supcon_encoder_encode_is_deterministic():
    """No decoder, no sampling: encoding the same input twice must give
    identical z, same guarantee as dim_red.autoencoder.Autoencoder.encode.
    """
    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    x = jnp.ones((10, 5), dtype=jnp.float32)
    z1 = model.encode(x)
    z2 = model.encode(x)
    np.testing.assert_array_equal(np.asarray(z1), np.asarray(z2))


def test_supcon_encoder_has_no_decoder_or_heads():
    model = SupConEncoder(input_dim=5, encoder_hidden_dim=[8], latent_dim=3, seed=0)
    assert not hasattr(model, "decode")
    assert not hasattr(model, "classify_family")
    assert not hasattr(model, "classify_spacegroup")
    assert "family_head" not in model.params
    assert "spacegroup_head" not in model.params
