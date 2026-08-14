"""
Unit tests for the CGCNN Flax model (dim_red.cgcnn.model).
"""

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
import numpy as np
from ase.build import bulk

from dim_red.cgcnn.graph import atoms_list_to_graph_arrays
from dim_red.cgcnn.model import CGCNNEncoder, ConvLayer, apply_family_mask


def _tiny_graph_batch(max_species=2, radius=5.0, max_num_nbr=8, max_atoms=None):
    atoms = [bulk("Cu", cubic=True), bulk("Cu", cubic=True) * (2, 1, 1)]
    return atoms_list_to_graph_arrays(
        atoms,
        radius=radius,
        max_num_nbr=max_num_nbr,
        max_species=max_species,
        max_atoms=max_atoms,
    )


def _tiny_encoder(n_gaussian, max_species=2, **kwargs):
    defaults = dict(
        atom_fea_len=8,
        n_conv=2,
        h_fea_len=16,
        n_h=1,
        latent_dim=4,
        n_gaussian=n_gaussian,
        max_species=max_species,
        seed=0,
    )
    defaults.update(kwargs)
    return CGCNNEncoder(**defaults)


# --- ConvLayer ---------------------------------------------------------------


def test_conv_layer_output_shape_matches_input():
    batch, max_atoms, max_num_nbr, atom_fea_len, n_gaussian = 3, 5, 4, 8, 6
    layer = ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=n_gaussian)
    rng = np.random.default_rng(0)
    atom_fea = jnp.asarray(
        rng.normal(size=(batch, max_atoms, atom_fea_len)), dtype=jnp.float32
    )
    nbr_fea = jnp.asarray(
        rng.normal(size=(batch, max_atoms, max_num_nbr, n_gaussian)), dtype=jnp.float32
    )
    nbr_idx = jnp.asarray(
        rng.integers(0, max_atoms, size=(batch, max_atoms, max_num_nbr)),
        dtype=jnp.int32,
    )
    nbr_mask = jnp.ones((batch, max_atoms, max_num_nbr), dtype=jnp.float32)
    atom_mask = jnp.ones((batch, max_atoms), dtype=jnp.float32)

    variables = layer.init(
        jax.random.PRNGKey(0), atom_fea, nbr_fea, nbr_idx, nbr_mask, atom_mask
    )
    out = layer.apply(variables, atom_fea, nbr_fea, nbr_idx, nbr_mask, atom_mask)
    assert out.shape == atom_fea.shape


# --- CGCNNEncoder --------------------------------------------------------------


def test_cgcnn_encoder_output_shape():
    arrays = _tiny_graph_batch()
    n_gaussian = arrays[2].shape[-1]
    model = _tiny_encoder(n_gaussian)
    z = model.encode(arrays)
    assert z.shape == (2, 4)


def test_cgcnn_encoder_deterministic():
    arrays = _tiny_graph_batch()
    n_gaussian = arrays[2].shape[-1]
    model = _tiny_encoder(n_gaussian)
    z1 = np.asarray(model.encode(arrays))
    z2 = np.asarray(model.encode(arrays))
    np.testing.assert_array_equal(z1, z2)


def test_cgcnn_encoder_invariant_to_extra_padding():
    """The key regression test for masked-pooling correctness: encoding the
    same structures with a larger, artificially-padded max_atoms must give
    numerically identical output to the tight (unpadded) encoding."""
    tight = _tiny_graph_batch(max_atoms=None)
    real_max_atoms = tight[0].shape[1]
    padded = _tiny_graph_batch(max_atoms=real_max_atoms + 5)
    n_gaussian = tight[2].shape[-1]

    model = _tiny_encoder(n_gaussian)
    z_tight = np.asarray(model.encode(tight))
    z_padded = np.asarray(model.encode(padded))
    np.testing.assert_allclose(z_tight, z_padded, atol=1e-5)


def test_cgcnn_encoder_with_params_matches_stored_params():
    arrays = _tiny_graph_batch()
    n_gaussian = arrays[2].shape[-1]
    model = _tiny_encoder(n_gaussian)
    z_direct = model.encode(arrays)
    z_explicit = model.encode_with_params(model.params, arrays)
    np.testing.assert_array_equal(np.asarray(z_direct), np.asarray(z_explicit))


def test_cgcnn_encoder_family_and_spacegroup_heads_shapes():
    arrays = _tiny_graph_batch()
    n_gaussian = arrays[2].shape[-1]
    model = _tiny_encoder(n_gaussian, n_family_classes=3, n_spacegroup_classes=6)
    z = model.encode(arrays)
    family_logits = model.classify_family(z)
    spacegroup_logits = model.classify_spacegroup(z)
    assert family_logits.shape == (2, 3)
    assert spacegroup_logits.shape == (2, 6)


def test_cgcnn_encoder_classify_family_without_head_raises():
    arrays = _tiny_graph_batch()
    n_gaussian = arrays[2].shape[-1]
    model = _tiny_encoder(n_gaussian)
    z = model.encode(arrays)
    with pytest.raises(ValueError, match="Family head is not configured"):
        model.classify_family(z)


def test_apply_family_mask_suppresses_implausible_spacegroups():
    family_onehot = jnp.eye(2)
    mask = jnp.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    logits = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    masked = apply_family_mask(logits, family_onehot, mask)
    assert masked[0, 2] < masked[0, 0] - 5
    assert masked[1, 0] < masked[1, 1] - 5


# --- Constructor validation ----------------------------------------------------


def test_cgcnn_encoder_requires_family_when_spacegroup_set():
    with pytest.raises(ValueError, match="n_family_classes must be set"):
        CGCNNEncoder(
            atom_fea_len=4,
            n_conv=1,
            h_fea_len=4,
            n_h=1,
            latent_dim=2,
            n_gaussian=5,
            max_species=2,
            n_spacegroup_classes=3,
            seed=0,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(atom_fea_len=0),
        dict(n_conv=0),
        dict(h_fea_len=0),
        dict(n_h=0),
        dict(latent_dim=0),
        dict(n_gaussian=0),
        dict(max_species=0),
    ],
)
def test_cgcnn_encoder_rejects_non_positive_dims(kwargs):
    defaults = dict(
        atom_fea_len=4,
        n_conv=1,
        h_fea_len=4,
        n_h=1,
        latent_dim=2,
        n_gaussian=5,
        max_species=2,
        seed=0,
    )
    defaults.update(kwargs)
    with pytest.raises(ValueError, match="positive integers"):
        CGCNNEncoder(**defaults)
