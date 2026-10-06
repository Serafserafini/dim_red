"""SingleStack must reproduce the OLD pipeline (run_single + train_tail:
classification, visualization, hierarchical_supcon) on the same inputs.

The body (same inputs, same shapes) is compared exactly. The heads encode the
data in separate train/val calls where the old code encoded all rows at once,
so XLA may reassociate floats differently by batch shape: heads are compared
with rtol=1e-5, atol=1e-6."""

import csv

import numpy as np
import pytest
import yaml

pytest.importorskip("jax")

from flax import serialization
from flax.traverse_util import flatten_dict

from dim_red.pipeline._common import _build_vocab_ids, _split_indices_grouped
from dim_red.supcon.stack import SingleStack, StackConfig
from dim_red.supcon.training import TrainConfig
from tests.golden import golden_spec as g


def _golden_params(name):
    return flatten_dict(
        serialization.msgpack_restore((g.GOLDEN_DIR / name).read_bytes())
    )


def _assert_params(actual, golden, exact):
    flat = flatten_dict(serialization.to_state_dict(actual))
    assert set(flat) == set(golden)
    for key, expected in golden.items():
        if exact:
            np.testing.assert_array_equal(np.asarray(flat[key]), expected)
        else:
            np.testing.assert_allclose(
                np.asarray(flat[key]), expected, rtol=1e-5, atol=1e-6
            )


def _golden_column(name, column):
    with open(g.GOLDEN_DIR / name) as f:
        return [float(row[column]) for row in csv.DictReader(f)]


def _train(epochs, tau, distance):
    return TrainConfig(
        epochs=epochs,
        batch_size=g.BATCH,
        learning_rate=1e-3,
        tau=tau,
        distance=distance,
        seed=g.SEED,
    )


def _family_config():
    return StackConfig(
        encoder_hidden_dim=g.ENCODER_HIDDEN,
        latent_dim=g.LATENT,
        projection_dim=g.PROJ_DIM,
        body_train=_train(g.BODY_EPOCHS, g.TAU, g.DISTANCE),
        classifier_train=_train(g.HEAD_EPOCHS, g.VIZ_TAU, "euclidean"),
        viz_train=_train(g.HEAD_EPOCHS, g.VIZ_TAU, g.VIZ_DISTANCE),
        classifier_hidden_dim=g.CLF_HIDDEN,
        viz_hidden_dim=g.VIZ_HIDDEN,
        viz_dim=2,
        seed=g.SEED,
    )


def _expert_config():
    return StackConfig(
        encoder_hidden_dim=g.ENCODER_HIDDEN,
        latent_dim=g.LATENT,
        projection_dim=g.PROJ_DIM,
        body_train=_train(g.HEAD_EPOCHS, g.TAU, g.DISTANCE),
        classifier_train=_train(g.HEAD_EPOCHS, g.TAU, g.DISTANCE),
        viz_train=_train(g.HEAD_EPOCHS, g.VIZ_TAU, g.VIZ_DISTANCE),
        classifier_hidden_dim=g.EXPERT_CLF_HIDDEN,
        viz_hidden_dim=g.EXPERT_VIZ_HIDDEN,
        viz_dim=2,
        seed=g.SEED,
    )


def _inputs():
    inputs = np.load(g.GOLDEN_DIR / "inputs.npz")
    split = np.load(g.GOLDEN_DIR / "family_embeddings.npz", allow_pickle=True)["split"]
    return inputs["X"], [str(v) for v in inputs["labels"]], inputs["spacegroups"], split


def _old_run_single_split(n_rows):
    """Train/val indices in the order run_single used for the body (a seeded
    permutation, not sorted): row order changes the batches."""
    inputs = np.load(g.GOLDEN_DIR / "inputs.npz")
    material_ids = [str(v) for v in inputs["material_ids"]]
    assert len(material_ids) == n_rows
    return _split_indices_grouped(material_ids, g.VAL_RATIO, g.SEED)


def test_family_stack_matches_old_pipeline():
    X, labels, _, split = _inputs()
    classes, y = _build_vocab_ids(labels)
    tr = np.flatnonzero(split == "train")
    va = np.flatnonzero(split == "val")
    body_tr, body_va = _old_run_single_split(len(X))
    assert set(body_tr) == set(tr) and set(body_va) == set(va)

    stack = SingleStack(X.shape[1], _family_config())
    body_history = stack.fit_body(X[body_tr], X[body_va], y[body_tr], y[body_va])

    _assert_params(
        stack.encoder.params, _golden_params("family_body_params.msgpack"), True
    )
    _assert_params(
        stack.projection.params,
        _golden_params("family_projection_params.msgpack"),
        True,
    )
    np.testing.assert_allclose(
        body_history["train_loss"],
        _golden_column("family_body_loss_history.csv", "train_loss"),
        rtol=1e-6,
    )

    stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=len(classes))
    _assert_params(
        stack.classifier.params,
        _golden_params("family_classifier_params.msgpack"),
        False,
    )
    _assert_params(
        stack.visualizer.params, _golden_params("family_viz_params.msgpack"), False
    )


@pytest.mark.parametrize("family", g.FAMILIES)
def test_expert_stack_matches_old_hierarchical_supcon(family):
    X, labels, spacegroups, split = _inputs()
    idx = np.flatnonzero(np.asarray(labels) == family)
    classes, local = _build_vocab_ids([int(s) for s in spacegroups[idx]])
    y = np.full(len(X), -1, dtype=np.int64)
    y[idx] = local
    tr = idx[split[idx] == "train"]
    va = idx[split[idx] == "val"]

    with open(g.GOLDEN_DIR / f"expert_{family}_local_classes.yaml") as f:
        assert classes == yaml.safe_load(f)["local_spacegroup_classes"]

    stack = SingleStack(X.shape[1], _expert_config())
    stack.fit_body(X[tr], X[va], y[tr], y[va])
    _assert_params(
        stack.encoder.params,
        _golden_params(f"expert_{family}_body_params.msgpack"),
        True,
    )
    _assert_params(
        stack.projection.params,
        _golden_params(f"expert_{family}_projection_params.msgpack"),
        True,
    )

    stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=len(classes))
    _assert_params(
        stack.classifier.params,
        _golden_params(f"expert_{family}_classifier_params.msgpack"),
        False,
    )
    _assert_params(
        stack.visualizer.params,
        _golden_params(f"expert_{family}_viz_params.msgpack"),
        False,
    )
