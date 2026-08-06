"""
Unit tests for pipeline config parsing (YAML -> dataclasses) and generic
sweep-grid expansion (any dotted-path config field can be a grid axis).
"""

import pytest
import yaml

from dim_red.pipeline.config import (
    AugmentationConfig,
    AuxHeadsConfig,
    BalancedBatchingParams,
    BatchingConfig,
    EarlyStoppingConfig,
    FetchConfig,
    PyxtalConfig,
    RunConfig,
    SoapConfig,
    SupConConfig,
    TrainSettings,
    VAEArchConfig,
    expand_sweep,
    flatten_config_dict,
    load_run_config,
    load_sweep_config,
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
        "vae": {"encoder_hidden_dim": [16, 8], "latent_dim": 2},
        "train": {"epochs": 3, "batch_size": 4, "beta": 0.5, "val_ratio": 0.25},
    }


def test_load_run_config_roundtrip(tmp_path):
    config_path = _write_yaml(tmp_path / "run.yaml", _single_run_dict())
    config = load_run_config(config_path)

    assert config.seed == 7
    assert config.fetch.crystal_systems == ["cubic", "hexagonal"]
    assert config.fetch.limit_per_system == 5
    assert config.soap.r_cut == 4.0
    assert config.soap.n_max == 2
    assert config.vae.encoder_hidden_dim == [16, 8]
    assert config.vae.latent_dim == 2
    assert config.vae.mirror is True
    assert config.train.epochs == 3
    assert config.train.val_ratio == 0.25

    # Defaults not present in the YAML should fall back to the dataclass defaults.
    assert config.soap.sigma == 0.5
    assert config.train.device == "cpu"

    # aux_heads defaults to mode="none" (plain VAE, current behavior) when
    # the config doesn't mention it at all.
    assert config.aux_heads == AuxHeadsConfig(mode="none")

    # model_kind defaults to "vae" (current behavior, unchanged) when the
    # config doesn't mention it at all.
    assert config.model_kind == "vae"


def test_run_config_parses_model_kind(tmp_path):
    d = _single_run_dict()
    d["model"] = "autoencoder"
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))
    assert config.model_kind == "autoencoder"


def test_run_config_rejects_invalid_model_kind():
    with pytest.raises(ValueError, match="model_kind must be one of"):
        RunConfig(
            soap=SoapConfig(),
            vae=VAEArchConfig(encoder_hidden_dim=[8], latent_dim=2),
            train=TrainSettings(),
            fetch=FetchConfig(crystal_systems=["cubic"]),
            model_kind="bogus",
        )


def test_run_config_to_dict_roundtrips_model_kind(tmp_path):
    d = _single_run_dict()
    d["model"] = "autoencoder"
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))
    saved_path = _write_yaml(tmp_path / "saved.yaml", run_config_to_dict(config))
    reloaded = load_run_config(saved_path)
    assert reloaded == config
    assert reloaded.model_kind == "autoencoder"


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
            vae=VAEArchConfig(encoder_hidden_dim=[8], latent_dim=2),
            train=TrainSettings(),
            fetch=FetchConfig(crystal_systems=["cubic"]),
            data_source="bogus",
        )


def test_run_config_fetch_data_source_requires_fetch_block():
    with pytest.raises(ValueError, match="requires a 'fetch' config block"):
        RunConfig(
            soap=SoapConfig(),
            vae=VAEArchConfig(encoder_hidden_dim=[8], latent_dim=2),
            train=TrainSettings(),
            data_source="fetch",
        )


def test_run_config_pyxtal_data_source_requires_pyxtal_block():
    with pytest.raises(ValueError, match="requires a 'pyxtal' config block"):
        RunConfig(
            soap=SoapConfig(),
            vae=VAEArchConfig(encoder_hidden_dim=[8], latent_dim=2),
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


def test_expand_sweep_can_vary_data_source(tmp_path):
    """Sweeping "data_source" (fetch vs pyxtal) works via the same generic
    grid mechanism as model_kind -- no special-casing needed in expand_sweep.
    """
    base = _sweep_base()
    base["pyxtal"] = {"structures_per_spacegroup": 2, "spacegroups": [225]}
    sweep_dict = {"base": base, "grid": {"data_source": ["fetch", "pyxtal"]}}
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))
    runs = expand_sweep(sweep)
    assert {r.data_source for r in runs} == {"fetch", "pyxtal"}


def test_expand_sweep_can_vary_model_kind(tmp_path):
    """The generic grid mechanism already supports this for free: sweeping
    "model" produces both a VAE and an Autoencoder RunConfig, with no
    special-casing needed in expand_sweep.
    """
    sweep_dict = {
        "base": _single_run_dict(),
        "grid": {"model": ["vae", "autoencoder"]},
    }
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))
    runs = expand_sweep(sweep)
    assert {r.model_kind for r in runs} == {"vae", "autoencoder"}


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
            "vae": {"encoder_hidden_dim": [16, 8], "latent_dim": 2},
        }
    )
    assert flat == {
        "seed": 7,
        "fetch.crystal_systems": ["cubic"],
        "fetch.limit_per_system": 5,
        "vae.encoder_hidden_dim": [16, 8],
        "vae.latent_dim": 2,
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


def test_run_config_defaults_supcon_block(tmp_path):
    config_path = _write_yaml(tmp_path / "run.yaml", _single_run_dict())
    config = load_run_config(config_path)
    assert config.supcon == SupConConfig(mode="family_and_spacegroup")


def test_supcon_config_rejects_invalid_mode():
    with pytest.raises(ValueError, match="supcon.mode must be one of"):
        SupConConfig(mode="bogus")


@pytest.mark.parametrize(
    "mode", ["family_only", "spacegroup_only", "family_and_spacegroup"]
)
def test_supcon_config_accepts_all_valid_modes(mode):
    assert SupConConfig(mode=mode).mode == mode


def test_supcon_config_defaults_distance_to_euclidean():
    assert SupConConfig().distance == "euclidean"


def test_supcon_config_rejects_invalid_distance():
    with pytest.raises(ValueError, match="supcon.distance must be one of"):
        SupConConfig(distance="bogus")


@pytest.mark.parametrize("distance", ["euclidean", "cosine"])
def test_supcon_config_accepts_all_valid_distances(distance):
    assert SupConConfig(distance=distance).distance == distance


def test_run_config_parses_model_kind_supcon(tmp_path):
    d = _single_run_dict()
    d["model"] = "supcon"
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))
    assert config.model_kind == "supcon"


def test_run_config_parses_supcon_block(tmp_path):
    d = _single_run_dict()
    d["model"] = "supcon"
    d["supcon"] = {"mode": "spacegroup_only", "lambda_spacegroup": 2.0, "tau": 0.05}
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))

    assert config.supcon.mode == "spacegroup_only"
    assert config.supcon.lambda_spacegroup == 2.0
    assert config.supcon.tau == 0.05
    # Untouched field keeps its default.
    assert config.supcon.lambda_family == 1.0


def test_supcon_config_lambda_norm_defaults_to_zero(tmp_path):
    config_path = _write_yaml(tmp_path / "run.yaml", _single_run_dict())
    config = load_run_config(config_path)
    assert config.supcon.lambda_norm == 0.0


def test_run_config_parses_supcon_lambda_norm(tmp_path):
    d = _single_run_dict()
    d["model"] = "supcon"
    d["supcon"] = {"lambda_norm": 0.25}
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))
    assert config.supcon.lambda_norm == 0.25


def test_run_config_to_dict_roundtrips_supcon_lambda_norm(tmp_path):
    d = _single_run_dict()
    d["model"] = "supcon"
    d["supcon"] = {"lambda_norm": 0.4}
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))

    saved_path = _write_yaml(tmp_path / "saved.yaml", run_config_to_dict(config))
    reloaded = load_run_config(saved_path)

    assert reloaded == config
    assert reloaded.supcon.lambda_norm == 0.4


def test_expand_sweep_can_vary_supcon_lambda_norm(tmp_path):
    base = _single_run_dict()
    base["model"] = "supcon"
    sweep_dict = {
        "base": base,
        "grid": {"supcon.lambda_norm": [0.0, 0.1, 0.5]},
    }
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))
    runs = expand_sweep(sweep)
    assert {r.supcon.lambda_norm for r in runs} == {0.0, 0.1, 0.5}


def test_run_config_to_dict_roundtrips_supcon_block(tmp_path):
    d = _single_run_dict()
    d["model"] = "supcon"
    d["supcon"] = {"mode": "family_only", "lambda_family": 0.7, "tau": 0.2}
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))

    saved_path = _write_yaml(tmp_path / "saved.yaml", run_config_to_dict(config))
    reloaded = load_run_config(saved_path)

    assert reloaded == config
    assert reloaded.model_kind == "supcon"
    assert reloaded.supcon.mode == "family_only"
    assert reloaded.supcon.tau == 0.2


def test_expand_sweep_can_vary_supcon_mode(tmp_path):
    base = _single_run_dict()
    base["model"] = "supcon"
    sweep_dict = {
        "base": base,
        "grid": {
            "supcon.mode": ["family_only", "spacegroup_only", "family_and_spacegroup"]
        },
    }
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))
    runs = expand_sweep(sweep)

    assert {r.supcon.mode for r in runs} == {
        "family_only",
        "spacegroup_only",
        "family_and_spacegroup",
    }
    assert all(r.model_kind == "supcon" for r in runs)


def test_sweep_config_invalid_supcon_mode_raises_on_expand(tmp_path):
    base = _single_run_dict()
    base["model"] = "supcon"
    sweep_dict = {"base": base, "grid": {"supcon.mode": ["bogus"]}}
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))
    with pytest.raises(ValueError, match="supcon.mode must be one of"):
        expand_sweep(sweep)


def test_run_config_defaults_batching_random(tmp_path):
    config_path = _write_yaml(tmp_path / "run.yaml", _single_run_dict())
    config = load_run_config(config_path)
    assert config.batching == BatchingConfig(strategy="random")


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


def test_run_config_parses_batching_block(tmp_path):
    d = _single_run_dict()
    d["model"] = "supcon"
    d["batching"] = {
        "strategy": "balanced",
        "balanced_params": {"P": 2, "K": 8, "S": None},
    }
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))

    assert config.batching.strategy == "balanced"
    assert config.batching.balanced_params.P == 2
    assert config.batching.balanced_params.K == 8
    assert config.batching.balanced_params.S is None


def test_run_config_to_dict_roundtrips_batching_block(tmp_path):
    d = _single_run_dict()
    d["model"] = "supcon"
    d["batching"] = {
        "strategy": "balanced",
        "balanced_params": {"P": None, "K": 12, "S": 4},
    }
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))

    saved_path = _write_yaml(tmp_path / "saved.yaml", run_config_to_dict(config))
    reloaded = load_run_config(saved_path)

    assert reloaded == config
    assert reloaded.batching.strategy == "balanced"
    assert reloaded.batching.balanced_params.K == 12
    assert reloaded.batching.balanced_params.S == 4


def test_existing_config_without_batching_block_still_loads(tmp_path):
    """Backward compatibility: a config predating the batching feature (no
    "batching:" key at all) must still load fine, defaulting to "random".
    """
    d = _single_run_dict()
    assert "batching" not in d
    config = load_run_config(_write_yaml(tmp_path / "run.yaml", d))
    assert config.batching.strategy == "random"


def test_expand_sweep_can_vary_batching_strategy(tmp_path):
    base = _single_run_dict()
    base["model"] = "supcon"
    base["batching"] = {"balanced_params": {"K": 8}}
    sweep_dict = {
        "base": base,
        "grid": {"batching.strategy": ["random", "balanced"]},
    }
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))
    runs = expand_sweep(sweep)

    assert {r.batching.strategy for r in runs} == {"random", "balanced"}


def test_sweep_config_invalid_batching_strategy_raises_on_expand(tmp_path):
    sweep_dict = {
        "base": _single_run_dict(),
        "grid": {"batching.strategy": ["bogus"]},
    }
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))
    with pytest.raises(ValueError, match="batching.strategy must be one of"):
        expand_sweep(sweep)


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


def test_expand_sweep_can_vary_early_stopping_patience(tmp_path):
    sweep_dict = {
        "base": _single_run_dict(),
        "grid": {"train.early_stopping.patience": [3, 5, 10]},
    }
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))
    runs = expand_sweep(sweep)
    assert {r.train.early_stopping.patience for r in runs} == {3, 5, 10}


def test_sweep_config_invalid_early_stopping_patience_raises_on_expand(tmp_path):
    sweep_dict = {
        "base": _single_run_dict(),
        "grid": {"train.early_stopping.patience": [0]},
    }
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))
    with pytest.raises(ValueError, match="early_stopping.patience must be a positive"):
        expand_sweep(sweep)


def _sweep_base():
    return {
        "seed": 1,
        "output_dir": "runs",
        "fetch": {"crystal_systems": ["cubic"], "limit_per_system": 5},
        "soap": {"r_cut": 4.0, "n_max": 2, "l_max": 2},
        "vae": {"encoder_hidden_dim": [16], "latent_dim": 3},
        "train": {"epochs": 1, "batch_size": 4},
    }


def test_load_sweep_config_and_expand(tmp_path):
    sweep_dict = {
        "base": _sweep_base(),
        "grid": {
            "fetch.crystal_systems": [["cubic"], ["cubic", "hexagonal"]],
            "vae.encoder_hidden_dim": [[16], [16, 8]],
        },
    }
    config_path = _write_yaml(tmp_path / "sweep.yaml", sweep_dict)
    sweep = load_sweep_config(config_path)

    assert sweep.output_dir == "runs"
    assert sweep.api_key is None

    runs = expand_sweep(sweep)

    # 2 crystal-system sets x 2 hidden-layer configs = 4 runs.
    assert len(runs) == 4
    assert all(isinstance(r, RunConfig) for r in runs)

    combos = {
        (tuple(r.fetch.crystal_systems), tuple(r.vae.encoder_hidden_dim)) for r in runs
    }
    assert combos == {
        (("cubic",), (16,)),
        (("cubic",), (16, 8)),
        (("cubic", "hexagonal"), (16,)),
        (("cubic", "hexagonal"), (16, 8)),
    }
    # Non-swept settings are shared across every expanded run.
    assert all(r.vae.latent_dim == 3 for r in runs)
    assert all(r.train.epochs == 1 for r in runs)

    # No aux_heads block -> mode="none" for every expanded run, matching the
    # plain-VAE default.
    assert all(r.aux_heads == AuxHeadsConfig(mode="none") for r in runs)


def test_sweep_api_key_reads_from_fetch_block(tmp_path):
    base = _sweep_base()
    base["fetch"]["api_key"] = "secret"
    sweep_dict = {"base": base, "grid": {}}
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))
    assert sweep.api_key == "secret"


def test_expand_sweep_with_empty_grid_returns_single_base_run(tmp_path):
    sweep_dict = {"base": _sweep_base()}
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))
    runs = expand_sweep(sweep)

    assert len(runs) == 1
    assert runs[0].fetch.crystal_systems == ["cubic"]
    assert runs[0].vae.encoder_hidden_dim == [16]


def test_expand_sweep_can_vary_any_dotted_path(tmp_path):
    """Grid axes are not limited to hidden dims/crystal systems -- any
    RunConfig field reachable by a dotted path can be swept, e.g. training
    hyperparameters or the run's seed.
    """
    sweep_dict = {
        "base": _sweep_base(),
        "grid": {
            "train.learning_rate": [0.001, 0.0005],
            "train.beta": [1.0, 2.0],
            "seed": [0, 1, 2],
        },
    }
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))
    runs = expand_sweep(sweep)

    # 2 learning rates x 2 betas x 3 seeds = 12 runs.
    assert len(runs) == 12
    combos = {(r.train.learning_rate, r.train.beta, r.seed) for r in runs}
    assert len(combos) == 12
    assert {r.seed for r in runs} == {0, 1, 2}
    assert {r.train.learning_rate for r in runs} == {0.001, 0.0005}
    assert {r.train.beta for r in runs} == {1.0, 2.0}
    # Everything not in the grid stays shared across every expanded run.
    assert all(r.fetch.crystal_systems == ["cubic"] for r in runs)


def test_sweep_config_invalid_aux_heads_mode_raises_on_expand(tmp_path):
    sweep_dict = {
        "base": _sweep_base(),
        "grid": {"aux_heads.mode": ["bogus"]},
    }
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))
    with pytest.raises(ValueError, match="aux_heads.mode must be one of"):
        expand_sweep(sweep)


def test_expand_sweep_family_only_expands_lambda_family_axis(tmp_path):
    base = _sweep_base()
    base["aux_heads"] = {"mode": "family_only"}
    base["fetch"]["crystal_systems"] = ["cubic"]
    sweep_dict = {
        "base": base,
        "grid": {
            "fetch.crystal_systems": [["cubic"], ["cubic", "hexagonal"]],
            "aux_heads.lambda_family": [0.5, 1.0, 2.0],
        },
    }
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))
    runs = expand_sweep(sweep)

    # 2 crystal-system sets x 3 lambda_family values.
    assert len(runs) == 6
    assert all(r.aux_heads.mode == "family_only" for r in runs)
    lambdas = sorted({r.aux_heads.lambda_family for r in runs})
    assert lambdas == [0.5, 1.0, 2.0]
    # lambda_spacegroup is irrelevant in family_only mode -> stays default.
    assert all(r.aux_heads.lambda_spacegroup == 1.0 for r in runs)


def test_expand_sweep_family_and_spacegroup_expands_both_lambda_axes(tmp_path):
    base = _sweep_base()
    base["aux_heads"] = {"mode": "family_and_spacegroup"}
    sweep_dict = {
        "base": base,
        "grid": {
            "aux_heads.lambda_family": [0.5, 1.0],
            "aux_heads.lambda_spacegroup": [0.1, 0.2],
        },
    }
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", sweep_dict))
    runs = expand_sweep(sweep)

    # 2 lf x 2 lsg = 4.
    assert len(runs) == 4
    combos = {(r.aux_heads.lambda_family, r.aux_heads.lambda_spacegroup) for r in runs}
    assert combos == {(0.5, 0.1), (0.5, 0.2), (1.0, 0.1), (1.0, 0.2)}
