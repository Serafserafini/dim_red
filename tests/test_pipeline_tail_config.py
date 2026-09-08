"""
Unit tests for phase-2 tail-training config parsing
(``dim_red.pipeline.config.TailTrainConfig`` and its nested blocks).
"""

import pytest
import yaml

from dim_red.pipeline.config import (
    ClassificationTailConfig,
    HierarchicalTailConfig,
    TailTrainConfig,
    TailTrainSettings,
    VisualizationTailConfig,
    load_tail_train_config,
    tail_train_config_to_dict,
)


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def _classification_dict(**overrides):
    d = {
        "run_dir": "runs/example",
        "tail_kind": "classification",
        "classification": {"mode": "family_and_spacegroup"},
    }
    d.update(overrides)
    return d


def _visualization_dict(**overrides):
    d = {
        "run_dir": "runs/example",
        "tail_kind": "visualization",
        "visualization": {"viz_dim": 2, "mode": "family_and_spacegroup"},
    }
    d.update(overrides)
    return d


def _hierarchical_dict(**overrides):
    d = {
        "run_dir": "runs/example",
        "tail_kind": "hierarchical",
        "hierarchical": {"head_hidden_dim": 16, "min_samples_per_expert": 10},
    }
    d.update(overrides)
    return d


# --- TailTrainConfig validation ----------------------------------------------


def test_tail_train_config_rejects_invalid_tail_kind():
    with pytest.raises(ValueError, match="tail_kind must be one of"):
        TailTrainConfig(run_dir="runs/x", tail_kind="bogus")


def test_tail_train_config_classification_requires_block():
    with pytest.raises(ValueError, match="requires a 'classification' config block"):
        TailTrainConfig(run_dir="runs/x", tail_kind="classification")


def test_tail_train_config_visualization_requires_block():
    with pytest.raises(ValueError, match="requires a 'visualization' config block"):
        TailTrainConfig(run_dir="runs/x", tail_kind="visualization")


def test_tail_train_config_hierarchical_requires_block():
    with pytest.raises(ValueError, match="requires a 'hierarchical' config block"):
        TailTrainConfig(run_dir="runs/x", tail_kind="hierarchical")


def test_tail_train_config_defaults_output_subdir_to_none():
    config = TailTrainConfig(
        run_dir="runs/x",
        tail_kind="classification",
        classification=ClassificationTailConfig(),
    )
    assert config.output_subdir is None


# --- ClassificationTailConfig -------------------------------------------------


def test_classification_tail_config_rejects_spacegroup_only():
    with pytest.raises(ValueError, match="classification.mode must be one of"):
        ClassificationTailConfig(mode="spacegroup_only")


@pytest.mark.parametrize("mode", ["family_only", "family_and_spacegroup"])
def test_classification_tail_config_accepts_valid_modes(mode):
    assert ClassificationTailConfig(mode=mode).mode == mode


def test_classification_tail_config_defaults():
    config = ClassificationTailConfig()
    assert config.mode == "family_and_spacegroup"
    assert config.lambda_family == 1.0
    assert config.lambda_spacegroup == 1.0
    assert config.head_hidden_dim == 16


# --- VisualizationTailConfig --------------------------------------------------


def test_visualization_tail_config_rejects_invalid_viz_dim():
    with pytest.raises(ValueError, match="visualization.viz_dim must be 2 or 3"):
        VisualizationTailConfig(viz_dim=5)


@pytest.mark.parametrize("viz_dim", [2, 3])
def test_visualization_tail_config_accepts_valid_viz_dims(viz_dim):
    assert VisualizationTailConfig(viz_dim=viz_dim).viz_dim == viz_dim


@pytest.mark.parametrize(
    "mode", ["family_only", "spacegroup_only", "family_and_spacegroup"]
)
def test_visualization_tail_config_accepts_all_supcon_modes(mode):
    assert VisualizationTailConfig(mode=mode).mode == mode


def test_visualization_tail_config_rejects_invalid_mode():
    with pytest.raises(ValueError, match="visualization.mode must be one of"):
        VisualizationTailConfig(mode="bogus")


def test_visualization_tail_config_rejects_invalid_distance():
    with pytest.raises(ValueError, match="visualization.distance must be one of"):
        VisualizationTailConfig(distance="bogus")


def test_visualization_tail_config_defaults():
    config = VisualizationTailConfig()
    assert config.viz_dim == 2
    assert config.tau == 0.1
    assert config.distance == "euclidean"
    assert config.lambda_norm == 0.0
    assert config.hidden_dim is None
    assert config.batching.strategy == "random"


# --- HierarchicalTailConfig ----------------------------------------------------


def test_hierarchical_tail_config_defaults():
    config = HierarchicalTailConfig()
    assert config.head_hidden_dim == 16
    assert config.min_samples_per_expert == 10


def test_hierarchical_tail_config_rejects_non_positive_head_hidden_dim():
    with pytest.raises(ValueError, match="head_hidden_dim must be a positive integer"):
        HierarchicalTailConfig(head_hidden_dim=0)


def test_hierarchical_tail_config_rejects_non_positive_min_samples_per_expert():
    with pytest.raises(
        ValueError, match="min_samples_per_expert must be a positive integer"
    ):
        HierarchicalTailConfig(min_samples_per_expert=0)


# --- YAML parsing / round-trip ------------------------------------------------


def test_load_tail_train_config_classification(tmp_path):
    d = _classification_dict()
    d["classification"]["lambda_family"] = 2.0
    d["seed"] = 7
    path = _write_yaml(tmp_path / "tail.yaml", d)
    config = load_tail_train_config(path)

    assert config.run_dir == "runs/example"
    assert config.tail_kind == "classification"
    assert config.classification.lambda_family == 2.0
    assert config.visualization is None
    assert config.seed == 7


def test_load_tail_train_config_visualization_with_balanced_batching(tmp_path):
    d = _visualization_dict()
    d["visualization"]["viz_dim"] = 3
    d["visualization"]["batching"] = {
        "strategy": "balanced",
        "balanced_params": {"K": 4, "P": 2},
    }
    path = _write_yaml(tmp_path / "tail.yaml", d)
    config = load_tail_train_config(path)

    assert config.tail_kind == "visualization"
    assert config.visualization.viz_dim == 3
    assert config.visualization.batching.strategy == "balanced"
    assert config.visualization.batching.balanced_params.K == 4
    assert config.visualization.batching.balanced_params.P == 2


def test_load_tail_train_config_early_stopping(tmp_path):
    d = _classification_dict()
    d["train"] = {"epochs": 5, "early_stopping": {"enabled": True, "patience": 3}}
    path = _write_yaml(tmp_path / "tail.yaml", d)
    config = load_tail_train_config(path)

    assert config.train.epochs == 5
    assert config.train.early_stopping.enabled is True
    assert config.train.early_stopping.patience == 3


def test_tail_train_config_to_dict_roundtrips_classification(tmp_path):
    d = _classification_dict()
    d["output_subdir"] = "my-classifier"
    path = _write_yaml(tmp_path / "tail.yaml", d)
    config = load_tail_train_config(path)

    saved_path = _write_yaml(tmp_path / "saved.yaml", tail_train_config_to_dict(config))
    reloaded = load_tail_train_config(saved_path)

    assert reloaded == config
    assert reloaded.output_subdir == "my-classifier"


def test_tail_train_config_to_dict_roundtrips_visualization(tmp_path):
    d = _visualization_dict()
    d["visualization"]["viz_dim"] = 3
    d["visualization"]["hidden_dim"] = [16, 8]
    path = _write_yaml(tmp_path / "tail.yaml", d)
    config = load_tail_train_config(path)

    saved_path = _write_yaml(tmp_path / "saved.yaml", tail_train_config_to_dict(config))
    reloaded = load_tail_train_config(saved_path)

    assert reloaded == config
    assert reloaded.visualization.viz_dim == 3
    assert reloaded.visualization.hidden_dim == [16, 8]


def test_load_tail_train_config_hierarchical(tmp_path):
    d = _hierarchical_dict()
    d["hierarchical"]["min_samples_per_expert"] = 5
    path = _write_yaml(tmp_path / "tail.yaml", d)
    config = load_tail_train_config(path)

    assert config.tail_kind == "hierarchical"
    assert config.hierarchical.head_hidden_dim == 16
    assert config.hierarchical.min_samples_per_expert == 5
    assert config.classification is None
    assert config.visualization is None


def test_tail_train_config_to_dict_roundtrips_hierarchical(tmp_path):
    d = _hierarchical_dict()
    d["output_subdir"] = "my-hierarchical"
    path = _write_yaml(tmp_path / "tail.yaml", d)
    config = load_tail_train_config(path)

    saved_path = _write_yaml(tmp_path / "saved.yaml", tail_train_config_to_dict(config))
    reloaded = load_tail_train_config(saved_path)

    assert reloaded == config
    assert reloaded.output_subdir == "my-hierarchical"


# --- TailTrainSettings.optimizer ---------------------------------------------


def test_tail_train_settings_defaults_optimizer_to_adam():
    assert TailTrainSettings().optimizer == "adam"


def test_tail_train_settings_rejects_invalid_optimizer():
    with pytest.raises(ValueError, match="train.optimizer must be one of"):
        TailTrainSettings(optimizer="bogus")


@pytest.mark.parametrize("optimizer", ["adam", "velo"])
def test_tail_train_settings_accepts_valid_optimizers(optimizer):
    assert TailTrainSettings(optimizer=optimizer).optimizer == optimizer


def test_load_tail_train_config_parses_train_optimizer(tmp_path):
    d = _classification_dict()
    d["train"] = {"optimizer": "velo"}
    path = _write_yaml(tmp_path / "tail.yaml", d)
    config = load_tail_train_config(path)
    assert config.train.optimizer == "velo"


def test_tail_train_config_to_dict_roundtrips_optimizer(tmp_path):
    d = _classification_dict()
    d["train"] = {"optimizer": "velo"}
    path = _write_yaml(tmp_path / "tail.yaml", d)
    config = load_tail_train_config(path)

    saved_path = _write_yaml(tmp_path / "saved.yaml", tail_train_config_to_dict(config))
    reloaded = load_tail_train_config(saved_path)

    assert reloaded == config
    assert reloaded.train.optimizer == "velo"
