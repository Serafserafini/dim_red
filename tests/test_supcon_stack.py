import numpy as np
import pytest
import yaml

pytest.importorskip("jax")

import jax

from dim_red.dataset import FeatureDatabase
from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.stack import (
    Batching,
    SingleStack,
    StackConfig,
    stack_config_from_dict,
    stack_config_to_dict,
)
from dim_red.supcon.tails import ProjectionTail
from dim_red.supcon.training import TrainConfig, training_first_phase


def _toy(n=40, n_features=6, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % n_classes
    centers = rng.normal(size=(n_classes, n_features)) * 3.0
    X = (centers[y] + rng.normal(size=(n, n_features)) * 0.3).astype(np.float32)
    idx = rng.permutation(n)
    tr, va = idx[: int(n * 0.75)], idx[int(n * 0.75) :]
    return X, y, tr, va


def _config(**over):
    def train(**kw):
        return TrainConfig(
            epochs=kw.pop("epochs", 3), batch_size=8, learning_rate=1e-3, seed=0, **kw
        )

    base = dict(
        encoder_hidden_dim=[8],
        latent_dim=4,
        projection_dim=5,
        body_train=train(tau=0.05, distance="cosine"),
        classifier_train=train(),
        viz_train=train(tau=0.1, distance="euclidean"),
        classifier_hidden_dim=6,
        viz_hidden_dim=[5],
        viz_dim=2,
        seed=0,
    )
    base.update(over)
    return StackConfig(**base)


def _params_equal(a, b):
    return all(
        np.array_equal(x, y)
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


def test_fit_body_equals_direct_training_first_phase_call():
    X, y, tr, va = _toy()
    cfg = _config()
    stack = SingleStack(X.shape[1], cfg)
    history = stack.fit_body(X[tr], X[va], y[tr], y[va])

    encoder = SupConEncoder(X.shape[1], cfg.encoder_hidden_dim, cfg.latent_dim, seed=0)
    projection = ProjectionTail(
        cfg.latent_dim, [cfg.latent_dim], cfg.projection_dim, seed=0
    )
    expected = training_first_phase(
        encoder,
        projection,
        FeatureDatabase.from_array(X[tr]),
        FeatureDatabase.from_array(X[va]),
        cfg.body_train,
        train_family_ids=y[tr],
        val_family_ids=y[va],
        lambda_family=1.0,
        lambda_spacegroup=0.0,
    )
    assert _params_equal(stack.encoder.params, encoder.params)
    assert _params_equal(stack.projection.params, projection.params)
    assert history["train_loss"] == expected["train_loss"]


def test_fit_heads_leaves_encoder_and_projection_untouched():
    X, y, tr, va = _toy()
    stack = SingleStack(X.shape[1], _config())
    stack.fit_body(X[tr], X[va], y[tr], y[va])
    before_enc = [np.array(p) for p in jax.tree_util.tree_leaves(stack.encoder.params)]
    before_proj = [
        np.array(p) for p in jax.tree_util.tree_leaves(stack.projection.params)
    ]
    histories = stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=3)
    after_enc = jax.tree_util.tree_leaves(stack.encoder.params)
    after_proj = jax.tree_util.tree_leaves(stack.projection.params)
    assert all(np.array_equal(a, b) for a, b in zip(before_enc, after_enc))
    assert all(np.array_equal(a, b) for a, b in zip(before_proj, after_proj))
    assert set(histories) == {"classifier", "visualization"}
    assert len(histories["classifier"]["train_loss"]) == 3
    assert stack.visualize(X).shape == (len(X), 2)
    probs = stack.predict_proba(X)
    assert probs.shape == (len(X), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_fit_heads_accepts_val_only_class():
    X, y, tr, va = _toy(n=40, n_classes=3)
    y = y.copy()
    y[va[0]] = 3  # class 3 appears only in validation
    stack = SingleStack(X.shape[1], _config())
    stack.fit_body(X[tr], X[va], y[tr], y[va])
    stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=4)
    assert stack.predict_proba(X).shape[1] == 4


def test_early_stopping_shortens_histories():
    X, y, tr, va = _toy()
    es = dict(
        early_stopping=True, early_stopping_patience=1, early_stopping_min_delta=1e9
    )
    cfg = _config(
        body_train=TrainConfig(
            epochs=8, batch_size=8, tau=0.05, distance="cosine", seed=0, **es
        ),
        classifier_train=TrainConfig(epochs=8, batch_size=8, seed=0, **es),
        viz_train=TrainConfig(epochs=8, batch_size=8, tau=0.1, seed=0, **es),
    )
    stack = SingleStack(X.shape[1], cfg)
    body = stack.fit_body(X[tr], X[va], y[tr], y[va])
    heads = stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=3)
    assert len(body["train_loss"]) < 8
    assert len(heads["classifier"]["train_loss"]) < 8
    assert len(heads["visualization"]["train_loss"]) < 8


def test_balanced_batching_requires_sub_labels():
    X, y, tr, va = _toy()
    cfg = _config(body_batching=Batching(strategy="balanced", K=4))
    stack = SingleStack(X.shape[1], cfg)
    with pytest.raises(ValueError, match="sub_train"):
        stack.fit_body(X[tr], X[va], y[tr], y[va])


def test_heads_can_be_trained_with_a_different_config():
    X, y, tr, va = _toy()
    stack = SingleStack(X.shape[1], _config())
    stack.fit_body(X[tr], X[va], y[tr], y[va])
    other = _config(classifier_hidden_dim=[7, 3], viz_hidden_dim=[4], viz_dim=3)
    stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=3, config=other)
    assert stack.visualize(X).shape == (len(X), 3)


def test_predict_before_heads_raises():
    X, y, tr, va = _toy()
    stack = SingleStack(X.shape[1], _config())
    with pytest.raises(RuntimeError, match="heads"):
        stack.predict_proba(X)


def test_config_dict_roundtrip():
    cfg = _config(
        body_batching=Batching(strategy="balanced", P=2, K=4, S=None),
        projection_hidden_dim=[3],
    )
    assert stack_config_from_dict(stack_config_to_dict(cfg)) == cfg


def test_save_and_load_roundtrip(tmp_path):
    X, y, tr, va = _toy()
    stack = SingleStack(X.shape[1], _config())
    stack.fit_body(X[tr], X[va], y[tr], y[va])
    stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=3)
    stack.save_body(tmp_path / "body")
    stack.save_heads(tmp_path / "heads")

    loaded = SingleStack.load_body(tmp_path / "body")
    loaded.load_heads(tmp_path / "heads")
    np.testing.assert_array_equal(loaded.encode(X), stack.encode(X))
    np.testing.assert_array_equal(loaded.predict_proba(X), stack.predict_proba(X))
    np.testing.assert_array_equal(loaded.visualize(X), stack.visualize(X))
    assert loaded.n_classes == 3


def _gpu_available():
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


def test_device_override_loads_a_gpu_trained_stack_on_cpu(tmp_path):
    X, y, tr, va = _toy()
    stack = SingleStack(X.shape[1], _config())
    stack.fit_body(X[tr], X[va], y[tr], y[va])
    stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=3)
    stack.save_body(tmp_path / "body")
    stack.save_heads(tmp_path / "heads")
    expected = stack.visualize(X)

    # Pretend it was trained on a GPU: only the recorded device changes.
    for name in ("body/stack.yaml", "heads/heads.yaml"):
        path = tmp_path / name
        meta = yaml.safe_load(path.read_text())
        for train_key in ("body_train", "classifier_train", "viz_train"):
            meta["config"][train_key]["device"] = "gpu"
        path.write_text(yaml.safe_dump(meta, sort_keys=False))

    if not _gpu_available():
        with pytest.raises(Exception):
            SingleStack.load_body(tmp_path / "body")

    loaded = SingleStack.load_body(tmp_path / "body", device="cpu")
    loaded.load_heads(tmp_path / "heads", device="cpu")
    np.testing.assert_allclose(loaded.visualize(X), expected, rtol=1e-5, atol=1e-6)
