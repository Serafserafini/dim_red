"""
Unit tests for the SupCon body/tail modules (ProjectionTail,
ClassificationTail, VisualizationTail).
"""

import pytest

pytest.importorskip("jax")

import jax.numpy as jnp
import numpy as np

from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.tails import (
    ClassificationTail,
    ProjectionTail,
    VisualizationTail,
    apply_family_mask,
)

# --- ProjectionTail ---------------------------------------------------------


def test_projection_tail_project_shape():
    tail = ProjectionTail(input_dim=3, hidden_dim=[8], projection_dim=5, seed=0)
    r = jnp.ones((10, 3), dtype=jnp.float32)
    z = tail.project(r)
    assert z.shape == (10, 5)


def test_projection_tail_project_is_deterministic():
    tail = ProjectionTail(input_dim=3, hidden_dim=[8], projection_dim=5, seed=0)
    r = jnp.ones((10, 3), dtype=jnp.float32)
    z1 = tail.project(r)
    z2 = tail.project(r)
    np.testing.assert_array_equal(np.asarray(z1), np.asarray(z2))


def test_projection_tail_project_with_params_matches_stored_params():
    tail = ProjectionTail(input_dim=3, hidden_dim=[8], projection_dim=5, seed=0)
    r = jnp.ones((4, 3), dtype=jnp.float32)
    z_direct = tail.project(r)
    z_explicit = tail.project_with_params(tail.params, r)
    np.testing.assert_array_equal(np.asarray(z_direct), np.asarray(z_explicit))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(input_dim=0, hidden_dim=[8], projection_dim=5),
        dict(input_dim=3, hidden_dim=[0], projection_dim=5),
        dict(input_dim=3, hidden_dim=[8], projection_dim=0),
    ],
)
def test_projection_tail_rejects_non_positive_dims(kwargs):
    with pytest.raises(ValueError, match="positive integers"):
        ProjectionTail(seed=0, **kwargs)


def test_projection_tail_params_independent_of_body_params():
    """Tails are never nested inside the body's module/param tree -- a
    projection tail's params are a wholly separate pytree from the body's
    (not a sub-tree reached by traversing the body's own params), which is
    what lets ``dim_red.supcon.training.train_supcon`` compose them as two
    independently-keyed branches of one dict (``{"body": ..., "tail": ...}``)
    without any collision, and what lets phase 2 simply omit the body's
    pytree from that dict entirely to "freeze" it.
    """
    model = SupConEncoder(input_dim=3, encoder_hidden_dim=[8], latent_dim=4, seed=0)
    tail = ProjectionTail(input_dim=4, hidden_dim=[8], projection_dim=5, seed=1)
    assert model.params is not tail.params
    combined = {"body": model.params, "tail": tail.params}
    assert combined["body"] is model.params
    assert combined["tail"] is tail.params


# --- VisualizationTail -------------------------------------------------------


@pytest.mark.parametrize("output_dim", [2, 3])
def test_visualization_tail_project_shape(output_dim):
    tail = VisualizationTail(input_dim=4, hidden_dim=[8], output_dim=output_dim, seed=0)
    r = jnp.ones((6, 4), dtype=jnp.float32)
    z = tail.project(r)
    assert z.shape == (6, output_dim)


def test_visualization_tail_rejects_invalid_output_dim():
    with pytest.raises(ValueError, match="output_dim must be 2 or 3"):
        VisualizationTail(input_dim=4, hidden_dim=[8], output_dim=4, seed=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(input_dim=0, hidden_dim=[8], output_dim=2),
        dict(input_dim=4, hidden_dim=[0], output_dim=2),
    ],
)
def test_visualization_tail_rejects_non_positive_dims(kwargs):
    with pytest.raises(ValueError, match="positive integers"):
        VisualizationTail(seed=0, **kwargs)


# --- ClassificationTail ------------------------------------------------------


def test_classification_tail_family_only():
    tail = ClassificationTail(input_dim=4, hidden_dim=8, n_family_classes=3, seed=0)
    r = jnp.ones((5, 4), dtype=jnp.float32)
    logits = tail.classify_family(r)
    assert logits.shape == (5, 3)
    with pytest.raises(ValueError, match="Spacegroup head is not configured"):
        tail.classify_spacegroup(r)


def test_classification_tail_family_and_spacegroup():
    tail = ClassificationTail(
        input_dim=4, hidden_dim=8, n_family_classes=3, n_spacegroup_classes=6, seed=0
    )
    r = jnp.ones((5, 4), dtype=jnp.float32)
    family_logits = tail.classify_family(r)
    spacegroup_logits = tail.classify_spacegroup(r)
    assert family_logits.shape == (5, 3)
    assert spacegroup_logits.shape == (5, 6)


def test_classification_tail_requires_family_when_spacegroup_set():
    with pytest.raises(ValueError, match="n_family_classes must be set"):
        ClassificationTail(input_dim=4, hidden_dim=8, n_spacegroup_classes=6, seed=0)


def test_classification_tail_requires_at_least_one_head():
    with pytest.raises(ValueError, match="At least one of"):
        ClassificationTail(input_dim=4, hidden_dim=8, seed=0)


def test_classification_tail_rejects_non_positive_dims():
    with pytest.raises(ValueError, match="positive integers"):
        ClassificationTail(input_dim=0, hidden_dim=8, n_family_classes=3, seed=0)


def test_classification_tail_with_params_matches_stored_params():
    tail = ClassificationTail(
        input_dim=4, hidden_dim=8, n_family_classes=3, n_spacegroup_classes=6, seed=0
    )
    r = jnp.ones((5, 4), dtype=jnp.float32)
    direct = tail.classify_family(r)
    explicit = tail.classify_family_with_params(tail.params, r)
    np.testing.assert_array_equal(np.asarray(direct), np.asarray(explicit))


def test_classification_tail_params_independent_of_body_params():
    """Same independence property as ``ProjectionTail`` (see
    ``test_projection_tail_params_independent_of_body_params``) -- needed so
    phase 2 (``dim_red.pipeline.tail_training``) can train this tail alone
    without the frozen body's params ever entering the optimizer's pytree.
    """
    model = SupConEncoder(input_dim=3, encoder_hidden_dim=[8], latent_dim=4, seed=0)
    tail = ClassificationTail(input_dim=4, hidden_dim=8, n_family_classes=3, seed=1)
    assert model.params is not tail.params


# --- apply_family_mask (verbatim duplicate of dim_red.vae.model's) ---------


def test_apply_family_mask_matches_vae_implementation():
    from dim_red.vae.model import apply_family_mask as vae_apply_family_mask

    spacegroup_logits = jnp.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    family_weights = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    mask = jnp.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

    supcon_result = apply_family_mask(spacegroup_logits, family_weights, mask)
    vae_result = vae_apply_family_mask(spacegroup_logits, family_weights, mask)
    np.testing.assert_allclose(np.asarray(supcon_result), np.asarray(vae_result))
