"""
Unit tests for pipeline config parsing (YAML -> dataclasses) and sweep-grid
expansion.
"""

import pytest
import yaml

from dim_red.pipeline.config import (
    AuxHeadsConfig,
    RunConfig,
    SoapConfig,
    SweepConfig,
    TrainSettings,
    expand_sweep,
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
        "data": {"crystal_systems": ["cubic", "hexagonal"], "limit_per_system": 5},
        "soap": {"r_cut": 4.0, "n_max": 2, "l_max": 2},
        "vae": {"encoder_hidden_dim": [16, 8], "latent_dim": 2},
        "train": {"epochs": 3, "batch_size": 4, "beta": 0.5, "val_ratio": 0.25},
    }


def test_load_run_config_roundtrip(tmp_path):
    config_path = _write_yaml(tmp_path / "run.yaml", _single_run_dict())
    config = load_run_config(config_path)

    assert config.seed == 7
    assert config.crystal_systems == ["cubic", "hexagonal"]
    assert config.limit_per_system == 5
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


def test_run_config_to_dict_reloads_identically(tmp_path):
    config_path = _write_yaml(tmp_path / "run.yaml", _single_run_dict())
    config = load_run_config(config_path)

    saved_path = _write_yaml(tmp_path / "saved.yaml", run_config_to_dict(config))
    reloaded = load_run_config(saved_path)

    assert reloaded == config


def test_load_sweep_config_and_expand(tmp_path):
    sweep_dict = {
        "seed": 1,
        "output_dir": "runs",
        "data": {
            "crystal_system_sets": [["cubic"], ["cubic", "hexagonal"]],
            "limit_per_system": 5,
        },
        "soap": {"r_cut": 4.0, "n_max": 2, "l_max": 2},
        "vae": {"hidden_layer_configs": [[16], [16, 8]], "latent_dim": 3},
        "train": {"epochs": 1, "batch_size": 4},
    }
    config_path = _write_yaml(tmp_path / "sweep.yaml", sweep_dict)
    sweep = load_sweep_config(config_path)

    assert sweep.crystal_system_sets == [["cubic"], ["cubic", "hexagonal"]]
    assert sweep.hidden_layer_configs == [[16], [16, 8]]
    assert sweep.latent_dim == 3

    runs = expand_sweep(sweep)

    # 2 crystal-system sets x 2 hidden-layer configs = 4 runs.
    assert len(runs) == 4
    assert all(isinstance(r, RunConfig) for r in runs)

    combos = {(tuple(r.crystal_systems), tuple(r.vae.encoder_hidden_dim)) for r in runs}
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


def test_aux_heads_config_rejects_invalid_mode():
    with pytest.raises(ValueError, match="aux_heads.mode must be one of"):
        AuxHeadsConfig(mode="bogus")


def test_sweep_config_rejects_invalid_aux_heads_mode():
    with pytest.raises(ValueError, match="aux_heads_mode must be one of"):
        SweepConfig(
            crystal_system_sets=[["cubic"]],
            hidden_layer_configs=[[16]],
            limit_per_system=5,
            soap=SoapConfig(),
            latent_dim=2,
            train=TrainSettings(),
            aux_heads_mode="bogus",
        )


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


def _sweep_dict_with_aux(aux_heads_dict):
    return {
        "data": {
            "crystal_system_sets": [["cubic"], ["cubic", "hexagonal"]],
            "limit_per_system": 5,
        },
        "soap": {"r_cut": 4.0, "n_max": 2, "l_max": 2},
        "vae": {"hidden_layer_configs": [[16]], "latent_dim": 2},
        "aux_heads": aux_heads_dict,
        "train": {"epochs": 1, "batch_size": 4},
    }


def test_expand_sweep_family_only_expands_lambda_family_axis(tmp_path):
    d = _sweep_dict_with_aux(
        {"mode": "family_only", "lambda_family_values": [0.5, 1.0, 2.0]}
    )
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", d))
    runs = expand_sweep(sweep)

    # 2 crystal-system sets x 1 hidden-layer config x 3 lambda_family values.
    assert len(runs) == 6
    assert all(r.aux_heads.mode == "family_only" for r in runs)
    lambdas = sorted({r.aux_heads.lambda_family for r in runs})
    assert lambdas == [0.5, 1.0, 2.0]
    # lambda_spacegroup is irrelevant in family_only mode -> stays default.
    assert all(r.aux_heads.lambda_spacegroup == 1.0 for r in runs)


def test_expand_sweep_family_and_spacegroup_expands_both_lambda_axes(tmp_path):
    d = _sweep_dict_with_aux(
        {
            "mode": "family_and_spacegroup",
            "lambda_family_values": [0.5, 1.0],
            "lambda_spacegroup_values": [0.1, 0.2],
        }
    )
    sweep = load_sweep_config(_write_yaml(tmp_path / "sweep.yaml", d))
    runs = expand_sweep(sweep)

    # 2 crystal-system sets x 1 hidden-layer config x 2 lf x 2 lsg = 8.
    assert len(runs) == 8
    combos = {(r.aux_heads.lambda_family, r.aux_heads.lambda_spacegroup) for r in runs}
    assert combos == {(0.5, 0.1), (0.5, 0.2), (1.0, 0.1), (1.0, 0.2)}
