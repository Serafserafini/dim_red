"""
Unit tests for pipeline config parsing (YAML -> dataclasses) of the cgcnn
``RunConfig`` and the generic sweep-config loader.
"""

import pytest
import yaml

from dim_red.pipeline.config import (
    AugmentationConfig,
    AuxHeadsConfig,
    BalancedBatchingParams,
    BatchingConfig,
    EarlyStoppingConfig,
    EncoderConfig,
    FetchConfig,
    GraphConfig,
    MaceConfig,
    PyxtalConfig,
    RunConfig,
    SoapConfig,
    TrainSettings,
    flatten_config_dict,
    load_run_config,
    load_sweep_config,
    run_config_from_dict,
    run_config_to_dict,
)


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def _single_run_dict():
    return {
        "seed": 7,
        "output_dir": "runs",
        "fetch": {"crystal_systems": ["cubic", "hexagonal"], "limit_per_system": 5},
        "soap": {"r_cut": 4.0, "n_max": 2, "l_max": 2},
        "encoder": {"encoder_hidden_dim": [16, 8], "latent_dim": 2},
        "train": {"epochs": 3, "batch_size": 4, "val_ratio": 0.25},
        # model defaults to cgcnn, which requires an active aux-heads mode.
        "aux_heads": {"mode": "family_only"},
    }


def test_load_run_config_roundtrip(tmp_path):
    config_path = _write_yaml(tmp_path / "run.yaml", _single_run_dict())
    config = load_run_config(config_path)

    assert config.seed == 7
    assert config.fetch.crystal_systems == ["cubic", "hexagonal"]
    assert config.fetch.limit_per_system == 5
    assert config.soap.r_cut == 4.0
    assert config.soap.n_max == 2
    assert config.encoder.encoder_hidden_dim == [16, 8]
    assert config.encoder.latent_dim == 2
    assert config.train.epochs == 3
    assert config.train.val_ratio == 0.25

    # Defaults not present in the YAML should fall back to the dataclass defaults.
    assert config.soap.sigma == 0.5
    assert config.train.device == "cpu"

    # AuxHeadsConfig's own default is mode="none" (invalid for cgcnn, hence
    # the explicit block in _single_run_dict).
    assert AuxHeadsConfig() == AuxHeadsConfig(mode="none")
    assert config.aux_heads == AuxHeadsConfig(mode="family_only")

    # model_kind defaults to "cgcnn" when the config doesn't mention it at
    # all.
    assert config.model_kind == "cgcnn"


def test_run_config_still_reads_the_legacy_vae_block_name(tmp_path):
    """``encoder:`` was called ``vae:`` before the VAE/autoencoder model
    kinds were removed -- older configs and saved config.yaml files must keep
    loading, with the removed decoder-only keys ignored."""
    d = _single_run_dict()
    d["vae"] = {
        "encoder_hidden_dim": [16, 8],
        "latent_dim": 2,
        "decoder_hidden_dim": [8, 16],
        "mirror": False,
    }
    del d["encoder"]
    d["train"]["beta"] = 0.5
    d["train"]["optimizer"] = "velo"
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))
    assert config.encoder.encoder_hidden_dim == [16, 8]
    assert config.encoder.latent_dim == 2


def test_run_config_parses_model_kind(tmp_path):
    d = _single_run_dict()
    d["model"] = "cgcnn"
    d["aux_heads"] = {"mode": "family_only"}
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))
    assert config.model_kind == "cgcnn"


@pytest.mark.parametrize("removed", ["vae", "autoencoder", "mace"])
def test_run_config_rejects_removed_model_kinds(tmp_path, removed):
    d = _single_run_dict()
    d["model"] = removed
    with pytest.raises(ValueError, match="model must be 'cgcnn'"):
        load_run_config(_write_yaml(tmp_path / "run.yaml", d))


def test_run_config_rejects_invalid_model_kind():
    with pytest.raises(ValueError, match="model_kind must be one of"):
        RunConfig(
            soap=SoapConfig(),
            encoder=EncoderConfig(encoder_hidden_dim=[8], latent_dim=2),
            train=TrainSettings(),
            fetch=FetchConfig(crystal_systems=["cubic"]),
            model_kind="bogus",
        )


def test_run_config_to_dict_roundtrips_model_kind(tmp_path):
    d = _single_run_dict()
    d["model"] = "cgcnn"
    d["aux_heads"] = {"mode": "family_only"}
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))
    saved_path = _write_yaml(tmp_path / "saved.yaml", run_config_to_dict(config))
    reloaded = load_run_config(saved_path)
    assert reloaded == config
    assert reloaded.model_kind == "cgcnn"


def test_run_config_defaults_to_fetch_data_source(tmp_path):
    d = _single_run_dict()
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))
    assert config.data_source == "fetch"
    assert config.pyxtal is None
    assert config.fetch is not None


def test_run_config_rejects_invalid_data_source():
    with pytest.raises(ValueError, match="data_source must be one of"):
        RunConfig(
            soap=SoapConfig(),
            encoder=EncoderConfig(encoder_hidden_dim=[8], latent_dim=2),
            train=TrainSettings(),
            fetch=FetchConfig(crystal_systems=["cubic"]),
            data_source="bogus",
        )


def test_run_config_fetch_data_source_requires_fetch_block():
    with pytest.raises(ValueError, match="requires a 'fetch' config block"):
        RunConfig(
            soap=SoapConfig(),
            encoder=EncoderConfig(encoder_hidden_dim=[8], latent_dim=2),
            train=TrainSettings(),
            data_source="fetch",
        )


def test_run_config_pyxtal_data_source_requires_pyxtal_block():
    with pytest.raises(ValueError, match="requires a 'pyxtal' config block"):
        RunConfig(
            soap=SoapConfig(),
            encoder=EncoderConfig(encoder_hidden_dim=[8], latent_dim=2),
            train=TrainSettings(),
            data_source="pyxtal",
        )


def test_run_config_from_dict_parses_pyxtal_data_source(tmp_path):
    d = _single_run_dict()
    del d["fetch"]  # not required for data_source="pyxtal"
    d["data_source"] = "pyxtal"
    d["pyxtal"] = {
        "families": ["cubic"],
        "structures_per_family": 20,
        "distribution": "random",
        "n_species": 2,
    }
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))

    assert config.data_source == "pyxtal"
    assert config.fetch is None
    assert config.pyxtal == PyxtalConfig(
        families=["cubic"],
        structures_per_family=20,
        distribution="random",
        n_species=2,
    )


def test_run_config_from_dict_still_requires_crystal_systems_for_fetch(tmp_path):
    d = _single_run_dict()
    del d["fetch"]["crystal_systems"]
    with pytest.raises(KeyError):
        load_run_config(_write_yaml(tmp_path / "run.yaml", d))


def test_run_config_from_dict_requires_fetch_block_for_fetch_source(tmp_path):
    d = _single_run_dict()
    del d["fetch"]
    with pytest.raises(KeyError):
        load_run_config(_write_yaml(tmp_path / "run.yaml", d))


def test_run_config_to_dict_roundtrips_pyxtal_data_source(tmp_path):
    d = _single_run_dict()
    del d["fetch"]
    d["data_source"] = "pyxtal"
    d["pyxtal"] = {"spacegroups": [225, 1], "structures_per_spacegroup": 3}
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))

    saved_path = _write_yaml(tmp_path / "saved.yaml", run_config_to_dict(config))
    reloaded = load_run_config(saved_path)

    assert reloaded == config
    assert reloaded.data_source == "pyxtal"
    assert reloaded.fetch is None
    assert reloaded.pyxtal.spacegroups == [225, 1]
    assert reloaded.pyxtal.structures_per_spacegroup == 3


def test_pyxtal_config_seed_defaults_to_none(tmp_path):
    d = _single_run_dict()
    del d["fetch"]
    d["data_source"] = "pyxtal"
    d["pyxtal"] = {"spacegroups": [225], "structures_per_spacegroup": 1}
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))
    assert config.pyxtal.seed is None


def test_pyxtal_config_seed_roundtrips(tmp_path):
    d = _single_run_dict()
    del d["fetch"]
    d["data_source"] = "pyxtal"
    d["pyxtal"] = {"spacegroups": [225], "structures_per_spacegroup": 1, "seed": 123}
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))
    assert config.pyxtal.seed == 123

    saved_path = _write_yaml(tmp_path / "saved.yaml", run_config_to_dict(config))
    reloaded = load_run_config(saved_path)
    assert reloaded.pyxtal.seed == 123


def test_run_config_to_dict_omits_pyxtal_block_for_fetch_source(tmp_path):
    config_path = _write_yaml(tmp_path / "run.yaml", _single_run_dict())
    config = load_run_config(config_path)
    saved = run_config_to_dict(config)
    assert "pyxtal" not in saved
    assert "fetch" in saved


def test_run_config_to_dict_reloads_identically(tmp_path):
    config_path = _write_yaml(tmp_path / "run.yaml", _single_run_dict())
    config = load_run_config(config_path)

    saved_path = _write_yaml(tmp_path / "saved.yaml", run_config_to_dict(config))
    reloaded = load_run_config(saved_path)

    assert reloaded == config


def test_flatten_config_dict_produces_dotted_paths():
    flat = flatten_config_dict(
        {
            "seed": 7,
            "fetch": {"crystal_systems": ["cubic"], "limit_per_system": 5},
            "encoder": {"encoder_hidden_dim": [16, 8], "latent_dim": 2},
        }
    )
    assert flat == {
        "seed": 7,
        "fetch.crystal_systems": ["cubic"],
        "fetch.limit_per_system": 5,
        "encoder.encoder_hidden_dim": [16, 8],
        "encoder.latent_dim": 2,
    }


def test_aux_heads_config_rejects_invalid_mode():
    with pytest.raises(ValueError, match="aux_heads.mode must be one of"):
        AuxHeadsConfig(mode="bogus")


# --- augmentation ------------------------------------------------------------


def test_run_config_defaults_augmentation_to_none(tmp_path):
    config_path = _write_yaml(tmp_path / "run.yaml", _single_run_dict())
    config = load_run_config(config_path)
    assert config.augmentation is None


def test_run_config_parses_augmentation_block(tmp_path):
    d = _single_run_dict()
    d["augmentation"] = {
        "n_augmented": 3,
        "keep_original": False,
        "jitter_probability": 1.0,
        "jitter_std": 0.1,
        "vacancy_probability": 0.5,
        "vacancy_atom_probability": 0.2,
        "max_vacancies": 2,
        "supercell_radius": 5.0,
        "seed": 123,
    }
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))

    assert config.augmentation == AugmentationConfig(
        n_augmented=3,
        keep_original=False,
        jitter_probability=1.0,
        jitter_std=0.1,
        vacancy_probability=0.5,
        vacancy_atom_probability=0.2,
        max_vacancies=2,
        supercell_radius=5.0,
        seed=123,
    )


def test_augmentation_config_seed_defaults_to_none(tmp_path):
    d = _single_run_dict()
    d["augmentation"] = {"n_augmented": 1}
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))
    assert config.augmentation.seed is None


def test_augmentation_config_supercell_radius_defaults_to_none(tmp_path):
    d = _single_run_dict()
    d["augmentation"] = {"n_augmented": 1}
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))
    assert config.augmentation.supercell_radius is None


def test_run_config_to_dict_roundtrips_augmentation_block(tmp_path):
    d = _single_run_dict()
    d["augmentation"] = {
        "n_augmented": 2,
        "vacancy_probability": 0.3,
        "supercell_radius": 6.0,
        "seed": 7,
    }
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))

    saved_path = _write_yaml(tmp_path / "saved.yaml", run_config_to_dict(config))
    reloaded = load_run_config(saved_path)

    assert reloaded == config
    assert reloaded.augmentation.n_augmented == 2
    assert reloaded.augmentation.vacancy_probability == 0.3
    assert reloaded.augmentation.supercell_radius == 6.0
    assert reloaded.augmentation.seed == 7


def test_run_config_to_dict_omits_augmentation_block_when_none(tmp_path):
    config_path = _write_yaml(tmp_path / "run.yaml", _single_run_dict())
    config = load_run_config(config_path)
    saved = run_config_to_dict(config)
    assert "augmentation" not in saved


def test_augmentation_config_validates_probabilities():
    with pytest.raises(ValueError, match="jitter_probability"):
        AugmentationConfig(jitter_probability=1.5)
    with pytest.raises(ValueError, match="vacancy_probability"):
        AugmentationConfig(vacancy_probability=-0.1)
    with pytest.raises(ValueError, match="jitter_std"):
        AugmentationConfig(jitter_std=-1.0)
    with pytest.raises(ValueError, match="max_vacancies"):
        AugmentationConfig(max_vacancies=-1)
    with pytest.raises(ValueError, match="n_augmented"):
        AugmentationConfig(n_augmented=-1)
    with pytest.raises(ValueError, match="supercell_radius"):
        AugmentationConfig(supercell_radius=0.0)


def test_run_config_parses_aux_heads_block(tmp_path):
    d = _single_run_dict()
    d["aux_heads"] = {"mode": "family_only", "lambda_family": 2.0, "head_hidden_dim": 8}
    config_path = _write_yaml(tmp_path / "run.yaml", d)
    config = load_run_config(config_path)

    assert config.aux_heads.mode == "family_only"
    assert config.aux_heads.lambda_family == 2.0
    assert config.aux_heads.head_hidden_dim == 8
    # Untouched field keeps its default.
    assert config.aux_heads.lambda_spacegroup == 1.0


def test_run_config_to_dict_roundtrips_aux_heads(tmp_path):
    d = _single_run_dict()
    d["aux_heads"] = {
        "mode": "family_and_spacegroup",
        "lambda_family": 0.5,
        "lambda_spacegroup": 2.0,
    }
    config_path = _write_yaml(tmp_path / "run.yaml", d)
    config = load_run_config(config_path)

    saved_path = _write_yaml(tmp_path / "saved.yaml", run_config_to_dict(config))
    reloaded = load_run_config(saved_path)

    assert reloaded == config
    assert reloaded.aux_heads.mode == "family_and_spacegroup"


def test_batching_config_rejects_invalid_strategy():
    with pytest.raises(ValueError, match="batching.strategy must be one of"):
        BatchingConfig(strategy="bogus")


def test_batching_config_balanced_requires_positive_k():
    with pytest.raises(
        ValueError, match="balanced_params.K must be a positive integer"
    ):
        BatchingConfig(strategy="balanced")
    with pytest.raises(
        ValueError, match="balanced_params.K must be a positive integer"
    ):
        BatchingConfig(strategy="balanced", balanced_params=BalancedBatchingParams(K=0))


def test_batching_config_balanced_with_valid_k_is_accepted():
    config = BatchingConfig(
        strategy="balanced", balanced_params=BalancedBatchingParams(P=4, K=16, S=3)
    )
    assert config.balanced_params.P == 4
    assert config.balanced_params.K == 16
    assert config.balanced_params.S == 3


def test_run_config_defaults_early_stopping_disabled(tmp_path):
    config_path = _write_yaml(tmp_path / "run.yaml", _single_run_dict())
    config = load_run_config(config_path)
    assert config.train.early_stopping == EarlyStoppingConfig(enabled=False)


def test_early_stopping_config_rejects_non_positive_patience():
    with pytest.raises(ValueError, match="early_stopping.patience must be a positive"):
        EarlyStoppingConfig(patience=0)
    with pytest.raises(ValueError, match="early_stopping.patience must be a positive"):
        EarlyStoppingConfig(patience=-1)


def test_early_stopping_config_rejects_negative_min_delta():
    with pytest.raises(ValueError, match="early_stopping.min_delta must be >= 0"):
        EarlyStoppingConfig(min_delta=-0.1)


def test_run_config_parses_train_early_stopping_block(tmp_path):
    d = _single_run_dict()
    d["train"]["early_stopping"] = {
        "enabled": True,
        "patience": 5,
        "min_delta": 0.01,
        "restore_best_weights": False,
    }
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))

    assert config.train.early_stopping.enabled is True
    assert config.train.early_stopping.patience == 5
    assert config.train.early_stopping.min_delta == 0.01
    assert config.train.early_stopping.restore_best_weights is False
    # Rest of the "train" block still parses normally alongside the nested
    # early_stopping sub-block.
    assert config.train.epochs == 3


def test_run_config_to_dict_roundtrips_early_stopping(tmp_path):
    d = _single_run_dict()
    d["train"]["early_stopping"] = {"enabled": True, "patience": 7}
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))

    saved_path = _write_yaml(tmp_path / "saved.yaml", run_config_to_dict(config))
    reloaded = load_run_config(saved_path)

    assert reloaded == config
    assert reloaded.train.early_stopping.enabled is True
    assert reloaded.train.early_stopping.patience == 7
    # Untouched field keeps its default.
    assert reloaded.train.early_stopping.min_delta == 0.0


def test_existing_config_without_early_stopping_block_still_loads(tmp_path):
    """Backward compatibility: a config predating this feature (no
    "early_stopping:" key under "train:" at all) must still load fine.
    """
    d = _single_run_dict()
    assert "early_stopping" not in d["train"]
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))
    assert config.train.early_stopping.enabled is False


def test_removed_train_keys_are_ignored_without_failing(tmp_path):
    """``train.optimizer`` (VeLO was removed, Adam is the only optimizer) and
    ``train.beta`` (VAE-only) may still sit in older configs / saved
    config.yaml files: they are ignored, not an error."""
    d = _single_run_dict()
    d["train"]["optimizer"] = "velo"
    d["train"]["beta"] = 0.5
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))
    assert not hasattr(config.train, "optimizer")
    assert not hasattr(config.train, "beta")


# --- GraphConfig / model_kind == "cgcnn" ------------------------------------


def _cgcnn_run_dict():
    d = _single_run_dict()
    d["model"] = "cgcnn"
    d["aux_heads"] = {"mode": "family_and_spacegroup"}
    return d


def test_run_config_defaults_graph_block(tmp_path):
    config_path = _write_yaml(tmp_path / "run.yaml", _single_run_dict())
    config = load_run_config(config_path)
    assert config.graph == GraphConfig()


def test_graph_config_n_gaussian_property():
    g = GraphConfig(radius=8.0, dmin=0.0, step=0.2)
    assert g.n_gaussian == 41
    assert g.resolved_dmax == 8.0


def test_graph_config_explicit_dmax_overrides_radius():
    g = GraphConfig(radius=8.0, dmax=5.0, dmin=0.0, step=0.2)
    assert g.resolved_dmax == 5.0
    assert g.n_gaussian == 26


def test_graph_config_graph_kwargs_excludes_architecture_fields():
    g = GraphConfig(atom_fea_len=32, n_conv=5, h_fea_len=64, n_h=2, max_species=4)
    kwargs = g.graph_kwargs()
    assert set(kwargs) == {
        "radius",
        "max_num_nbr",
        "dmin",
        "dmax",
        "step",
        "max_species",
    }
    assert kwargs["max_species"] == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(radius=0),
        dict(max_num_nbr=0),
        dict(step=0),
        dict(dmax=0.0, dmin=0.0),
        dict(max_species=0),
        dict(atom_fea_len=0),
        dict(n_conv=0),
        dict(h_fea_len=0),
        dict(n_h=0),
    ],
)
def test_graph_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        GraphConfig(**kwargs)


def test_run_config_parses_model_kind_cgcnn(tmp_path):
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", _cgcnn_run_dict()))
    assert config.model_kind == "cgcnn"


def test_run_config_parses_graph_block(tmp_path):
    d = _cgcnn_run_dict()
    d["graph"] = {"radius": 6.0, "max_num_nbr": 10, "max_species": 4, "n_conv": 2}
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))

    assert config.graph.radius == 6.0
    assert config.graph.max_num_nbr == 10
    assert config.graph.max_species == 4
    assert config.graph.n_conv == 2
    # Untouched field keeps its default.
    assert config.graph.atom_fea_len == 64


def test_run_config_cgcnn_requires_aux_heads_mode_not_none(tmp_path):
    d = _cgcnn_run_dict()
    d["aux_heads"] = {"mode": "none"}
    with pytest.raises(ValueError, match="requires aux_heads.mode != 'none'"):
        load_run_config(_write_yaml(tmp_path / "run.yaml", d))


def test_run_config_cgcnn_default_aux_heads_mode_none_raises(tmp_path):
    # No aux_heads block at all -- AuxHeadsConfig's own default
    # (mode="none") should still trip the cgcnn-specific check.
    d = _single_run_dict()
    del d["aux_heads"]
    d["model"] = "cgcnn"
    with pytest.raises(ValueError, match="requires aux_heads.mode != 'none'"):
        load_run_config(_write_yaml(tmp_path / "run.yaml", d))


def test_run_config_to_dict_roundtrips_graph_block(tmp_path):
    d = _cgcnn_run_dict()
    d["graph"] = {"radius": 5.0, "max_species": 6}
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))

    saved_path = _write_yaml(tmp_path / "saved.yaml", run_config_to_dict(config))
    reloaded = load_run_config(saved_path)

    assert reloaded == config
    assert reloaded.graph.radius == 5.0
    assert reloaded.graph.max_species == 6


def test_run_config_to_dict_always_includes_graph_block(tmp_path):
    # Unlike fetch/pyxtal/augmentation/tails, "graph" is always serialized
    # (same treatment as "soap"), even for non-cgcnn model kinds.
    config_path = _write_yaml(tmp_path / "run.yaml", _single_run_dict())
    config = load_run_config(config_path)
    saved = run_config_to_dict(config)
    assert "graph" in saved


# --- MaceConfig ----------------------------------------------------------------
#
# No longer a RunConfig field; still the data.mace block of a supcon_mace
# FullStack (dim_red.pipeline.full_stack_config).


def test_mace_config_mace_kwargs():
    m = MaceConfig(checkpoint_path="/ckpt", r_max=6.0, pooling="sum")
    assert m.mace_kwargs() == {
        "checkpoint_path": "/ckpt",
        "r_max": 6.0,
        "pooling": "sum",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(r_max=0),
        dict(r_max=-1.0),
        dict(pooling="max"),
    ],
)
def test_mace_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        MaceConfig(**kwargs)


# --- removed features: explicit errors / tolerated leftovers ------------------


def test_run_config_rejects_non_cgcnn_models_with_a_pointer():
    with pytest.raises(ValueError, match="FullStack"):
        run_config_from_dict(
            {
                "model": "supcon",
                "data_source": "pyxtal",
                "pyxtal": {"structures_per_spacegroup": 1},
                "encoder": {"encoder_hidden_dim": [4], "latent_dim": 2},
            }
        )


@pytest.mark.parametrize("model", ["supcon", "supcon_mace"])
def test_run_config_rejects_supcon_models_from_yaml(tmp_path, model):
    d = _single_run_dict()
    d["model"] = model
    with pytest.raises(ValueError, match="FullStack"):
        load_run_config(_write_yaml(tmp_path / "run.yaml", d))


def test_run_config_rejects_top_level_tails_block(tmp_path):
    d = _single_run_dict()
    d["tails"] = {"visualization": {"viz_dim": 2}}
    with pytest.raises(ValueError, match="dimred-train-tail"):
        load_run_config(_write_yaml(tmp_path / "run.yaml", d))


def test_run_config_ignores_removed_supcon_batching_and_mace_blocks(tmp_path):
    """Every config.yaml run_single saved before supcon/batching/mace were
    removed carries all three blocks (cgcnn runs included), so they must
    still load -- the blocks are ignored, not an error."""
    d = _single_run_dict()
    d["supcon"] = {"mode": "family_only", "tau": 0.05, "projection_dim": 128}
    d["batching"] = {"strategy": "balanced", "balanced_params": {"K": 4}}
    d["mace"] = {"checkpoint_path": None, "r_max": 5.0, "pooling": "mean"}
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))

    assert config == load_run_config(
        _write_yaml(tmp_path / "plain.yaml", _single_run_dict())
    )
    saved = run_config_to_dict(config)
    assert "supcon" not in saved
    assert "batching" not in saved
    assert "mace" not in saved
    assert "tails" not in saved


def test_load_sweep_config_reads_base_and_grid(tmp_path):
    sweep_dict = {
        "base": {**_single_run_dict(), "output_dir": "sweeps"},
        "grid": {"encoder.encoder_hidden_dim": [[16], [16, 8]], "seed": [0, 1]},
    }
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))

    assert sweep.output_dir == "sweeps"
    assert sweep.base["fetch"]["crystal_systems"] == ["cubic", "hexagonal"]
    assert sweep.grid == {"encoder.encoder_hidden_dim": [[16], [16, 8]], "seed": [0, 1]}


@pytest.mark.parametrize("model_kind", ["supcon", "supcon_mace"])
def test_run_config_dataclass_refuses_supcon_models_with_a_pointer(model_kind):
    with pytest.raises(ValueError, match="model_kind must be one of.*FullStack"):
        RunConfig(
            soap=SoapConfig(),
            encoder=EncoderConfig(encoder_hidden_dim=[8], latent_dim=2),
            train=TrainSettings(),
            fetch=FetchConfig(crystal_systems=["cubic"]),
            aux_heads=AuxHeadsConfig(mode="family_only"),
            model_kind=model_kind,
        )


@pytest.mark.parametrize("content", ["", "- a\n- b\n"])
def test_load_yaml_rejects_non_mapping(tmp_path, content):
    from dim_red.pipeline.config import load_yaml

    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(path)
