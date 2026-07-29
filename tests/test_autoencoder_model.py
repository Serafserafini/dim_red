"""
Unit tests for Autoencoder model and encoder/decoder split utilities.
"""

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
import numpy as np

from dim_red.autoencoder.codec import split_encoder_decoder
from dim_red.autoencoder.model import Autoencoder, apply_family_mask


def test_autoencoder_forward_shapes():
    model = Autoencoder(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
    )
    x = jnp.ones((10, 5), dtype=jnp.float32)
    x_recon, z = model.forward(x)

    assert x_recon.shape == (10, 5)
    assert z.shape == (10, 2)


def test_autoencoder_encode_is_deterministic():
    """Unlike the VAE, encoding the same input twice must give identical z
    (no sampling), since there's no reparameterization trick here.
    """
    model = Autoencoder(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
    )
    x = jnp.ones((10, 5), dtype=jnp.float32)
    z1 = model.encode(x)
    z2 = model.encode(x)
    np.testing.assert_array_equal(np.asarray(z1), np.asarray(z2))


def test_split_encoder_decoder_shapes():
    model = Autoencoder(
        input_dim=5,
        encoder_hidden_dim=[8, 4],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
    )
    encoder, decoder = split_encoder_decoder(model)
    x = jnp.ones((4, 5), dtype=jnp.float32)
    z = encoder(x)
    x_recon = decoder(z)

    assert z.shape == (4, 2)
    assert x_recon.shape == (4, 5)


def test_autoencoder_without_aux_heads_has_no_head_params():
    """mode="none" (no n_family_classes/n_spacegroup_classes) must remain
    the plain autoencoder, byte-for-byte -- no head params allocated at all.
    """
    model = Autoencoder(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
    )

    assert "family_head" not in model.params
    assert "spacegroup_head" not in model.params
    assert model.n_family_classes is None
    assert model.n_spacegroup_classes is None


def test_autoencoder_family_only_head():
    model = Autoencoder(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
        n_family_classes=3,
        head_hidden_dim=4,
    )
    z = jnp.ones((6, 2), dtype=jnp.float32)

    family_logits = model.classify_family(z)
    assert family_logits.shape == (6, 3)

    with pytest.raises(ValueError, match="Spacegroup head is not configured"):
        model.classify_spacegroup(z)


def test_autoencoder_family_and_spacegroup_heads():
    model = Autoencoder(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
        n_family_classes=3,
        n_spacegroup_classes=6,
        head_hidden_dim=4,
    )
    z = jnp.ones((6, 2), dtype=jnp.float32)

    assert model.classify_family(z).shape == (6, 3)
    assert model.classify_spacegroup(z).shape == (6, 6)


def test_autoencoder_spacegroup_head_requires_family_head():
    with pytest.raises(ValueError, match="n_family_classes must be set"):
        Autoencoder(
            input_dim=5,
            encoder_hidden_dim=[8],
            decoder_hidden_dim=None,
            latent_dim=2,
            seed=0,
            n_spacegroup_classes=6,
        )


def test_apply_family_mask_hard_zeroes_implausible_classes():
    # 2 families x 4 spacegroups: family 0 -> {0,1}, family 1 -> {2,3}.
    mask = jnp.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=jnp.float32)
    spacegroup_logits = jnp.zeros((2, 4))  # uniform raw logits
    family_onehot = jax.nn.one_hot(jnp.array([0, 1]), 2)

    masked = apply_family_mask(spacegroup_logits, family_onehot, mask)
    probs = jax.nn.softmax(masked, axis=-1)

    np.testing.assert_allclose(np.asarray(probs[0, 2:]), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(probs[1, :2]), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(probs.sum(axis=1)), [1.0, 1.0], atol=1e-6)
