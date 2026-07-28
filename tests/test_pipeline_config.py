"""
Unit tests for pipeline config parsing (YAML -> dataclasses) and sweep-grid
expansion.
"""

import yaml

from dim_red.pipeline.config import (
    RunConfig,
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
