"""
Unit tests for VAE model and encoder/decoder split utilities.
"""

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
import numpy as np

from dim_red.vae.codec import split_encoder_decoder
from dim_red.vae.model import VAE, apply_family_mask


def test_vae_forward_shapes():
    model = VAE(
        input_dim=5,
        encoder_hidden_dim=[8],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
    )
    x = jnp.ones((10, 5), dtype=jnp.float32)
    key = jax.random.PRNGKey(1)
    x_recon, mu, logvar = model.forward(x, key)

    assert x_recon.shape == (10, 5)
    assert mu.shape == (10, 2)
    assert logvar.shape == (10, 2)


def test_split_encoder_decoder_shapes():
    model = VAE(
        input_dim=5,
        encoder_hidden_dim=[8, 4],
        decoder_hidden_dim=None,
        latent_dim=2,
        seed=0,
    )
    encoder, decoder = split_encoder_decoder(model)
    x = jnp.ones((4, 5), dtype=jnp.float32)
    mu, logvar = encoder(x)
    z = model.reparameterize(jax.random.PRNGKey(2), mu, logvar)
    x_recon = decoder(z)

    assert mu.shape == (4, 2)
    assert logvar.shape == (4, 2)
    assert x_recon.shape == (4, 5)


def test_vae_without_aux_heads_has_no_head_params():
    """mode="none" (no n_family_classes/n_spacegroup_classes) must remain
    the plain VAE, byte-for-byte -- no head params allocated at all.
    """
    model = VAE(
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


def test_vae_family_only_head():
    model = VAE(
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


def test_vae_family_and_spacegroup_heads():
    model = VAE(
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


def test_vae_spacegroup_head_requires_family_head():
    with pytest.raises(ValueError, match="n_family_classes must be set"):
        VAE(
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


def test_apply_family_mask_soft_blends_plausibility():
    mask = jnp.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=jnp.float32)
    spacegroup_logits = jnp.zeros((1, 4))
    # Soft family weighting: 50/50 between the two families.
    family_weights = jnp.array([[0.5, 0.5]])

    masked = apply_family_mask(spacegroup_logits, family_weights, mask)
    probs = np.asarray(jax.nn.softmax(masked, axis=-1))[0]

    # All four spacegroups become plausible (mask blended to all 0.5), so
    # softmax should stay uniform rather than zeroing out any half.
    np.testing.assert_allclose(probs, [0.25, 0.25, 0.25, 0.25], atol=1e-5)
