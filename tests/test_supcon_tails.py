"""
Unit tests for the SupCon body/tail modules (ProjectionTail,
ClassificationTail, VisualizationTail).
"""

import pytest

pytest.importorskip("jax")

import jax.numpy as jnp
import numpy as np
from flax import serialization

from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.tails import ClassificationTail, ProjectionTail, VisualizationTail

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
    what lets ``dim_red.supcon.training.training_first_phase`` compose them as two
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


def test_classification_tail_classify_shape():
    tail = ClassificationTail(input_dim=4, hidden_dim=8, n_classes=3, seed=0)
    r = jnp.ones((5, 4), dtype=jnp.float32)
    assert tail.classify(r).shape == (5, 3)


def test_classification_tail_rejects_non_positive_dims():
    with pytest.raises(ValueError, match="positive integers"):
        ClassificationTail(input_dim=0, hidden_dim=8, n_classes=3, seed=0)
    with pytest.raises(ValueError, match="positive integers"):
        ClassificationTail(input_dim=4, hidden_dim=8, n_classes=0, seed=0)


def test_classification_tail_hidden_dim_accepts_a_sequence_for_a_deeper_mlp():
    tail = ClassificationTail(input_dim=4, hidden_dim=[16, 8], n_classes=3, seed=0)
    r = jnp.ones((5, 4), dtype=jnp.float32)
    assert tail.classify(r).shape == (5, 3)
    # Two hidden layers (16, 8) plus the output layer: three Dense kernels.
    dense_layers = [k for k in tail.params if k.startswith("Dense_")]
    assert len(dense_layers) == 3  # Dense_0 (16), Dense_1 (8), Dense_2 (n_classes)


def test_classification_tail_rejects_empty_hidden_dim_sequence():
    with pytest.raises(ValueError, match="positive integers"):
        ClassificationTail(input_dim=4, hidden_dim=[], n_classes=3, seed=0)


def test_classification_tail_rejects_non_positive_entry_in_hidden_dim_sequence():
    with pytest.raises(ValueError, match="positive integers"):
        ClassificationTail(input_dim=4, hidden_dim=[16, 0], n_classes=3, seed=0)


def test_classification_tail_with_params_matches_stored_params():
    tail = ClassificationTail(input_dim=4, hidden_dim=8, n_classes=3, seed=0)
    r = jnp.ones((5, 4), dtype=jnp.float32)
    direct = tail.classify(r)
    explicit = tail.classify_with_params(tail.params, r)
    np.testing.assert_array_equal(np.asarray(direct), np.asarray(explicit))


def test_classification_tail_params_independent_of_body_params():
    """Same independence property as ``ProjectionTail`` (see
    ``test_projection_tail_params_independent_of_body_params``) -- needed so
    phase 2 (``dim_red.pipeline.tail_training``) can train this tail alone
    without the frozen body's params ever entering the optimizer's pytree.
    """
    model = SupConEncoder(input_dim=3, encoder_hidden_dim=[8], latent_dim=4, seed=0)
    tail = ClassificationTail(input_dim=4, hidden_dim=8, n_classes=3, seed=1)
    assert model.params is not tail.params


def test_classification_tail_load_params_bytes_roundtrip():
    source = ClassificationTail(input_dim=4, hidden_dim=[8], n_classes=3, seed=0)
    target = ClassificationTail(input_dim=4, hidden_dim=[8], n_classes=3, seed=99)
    r = jnp.ones((5, 4), dtype=jnp.float32)
    assert not np.allclose(
        np.asarray(source.classify(r)), np.asarray(target.classify(r))
    )

    target.load_params_bytes(serialization.to_bytes(source.params))
    np.testing.assert_allclose(
        np.asarray(source.classify(r)), np.asarray(target.classify(r)), atol=1e-6
    )


def test_classification_tail_load_params_bytes_accepts_legacy_family_head_layout():
    """Tails saved before the classification tail had a single head stored
    their parameters under a ``"family_head"`` key -- those checkpoints must
    keep loading."""
    source = ClassificationTail(input_dim=4, hidden_dim=[8], n_classes=3, seed=0)
    legacy_bytes = serialization.to_bytes({"family_head": source.params})
    target = ClassificationTail(input_dim=4, hidden_dim=[8], n_classes=3, seed=99)

    target.load_params_bytes(legacy_bytes)

    r = jnp.ones((5, 4), dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(source.classify(r)), np.asarray(target.classify(r)), atol=1e-6
    )
