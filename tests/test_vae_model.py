"""
Unit tests for VAE model and encoder/decoder split utilities.
"""

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from dim_red.vae.codec import split_encoder_decoder
from dim_red.vae.model import VAE


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
