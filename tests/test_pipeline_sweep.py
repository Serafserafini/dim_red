"""
Unit tests for sweep-level run-directory naming: each ``run_sweep`` call gets
its own ``<date>-<n>`` directory, and the individual run folders inside it
are named from swept hyperparameters only (no timestamp).
"""

from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

pytest.importorskip("jax")

from dim_red.pipeline.config import SweepConfig
from dim_red.pipeline.sweep import run_sweep


def _fake_atoms(symbol: str, material_id: str) -> Atoms:
    atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
    atoms.info["material_id"] = material_id
    return atoms


def _make_sweep_config(tmp_path) -> SweepConfig:
    base = {
        "seed": 0,
        "output_dir": str(tmp_path / "runs"),
        "fetch": {"crystal_systems": ["cubic"], "limit_per_system": 8},
        "soap": {"r_cut": 3.0, "n_max": 2, "l_max": 2},
        "vae": {"encoder_hidden_dim": [4], "latent_dim": 2},
        "train": {"epochs": 1, "batch_size": 4, "val_ratio": 0.25},
    }
    return SweepConfig(base=base, grid={"vae.encoder_hidden_dim": [[4], [8]]})


def _patch_dataset(n_samples: int = 8, n_features: int = 5):
    fake_atoms = [_fake_atoms("Cu", f"mp-{i}") for i in range(n_samples)]
    rng = np.random.default_rng(0)
    fake_soap = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    return patch(
        "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
        return_value=fake_atoms,
    ), patch("dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap)


def test_run_sweep_run_dirs_have_no_timestamp(tmp_path):
    config = _make_sweep_config(tmp_path)
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        run_dirs = run_sweep(config)

    assert len(run_dirs) == 2
    names = {d.name for d in run_dirs}
    assert names == {"hd-4_cs-cubic", "hd-8_cs-cubic"}
    for d in run_dirs:
        assert d.parent.name.startswith("2")  # date prefix e.g. "20260729-1"
        assert "-" in d.parent.name


def test_run_sweep_invocations_get_incrementing_dated_dirs(tmp_path):
    config = _make_sweep_config(tmp_path)
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        first_run_dirs = run_sweep(config)
        second_run_dirs = run_sweep(config)

    first_sweep_dir = first_run_dirs[0].parent
    second_sweep_dir = second_run_dirs[0].parent
    assert first_sweep_dir != second_sweep_dir

    first_n = int(first_sweep_dir.name.split("-")[-1])
    second_n = int(second_sweep_dir.name.split("-")[-1])
    assert second_n == first_n + 1

    # Same hyperparam-only names reused across sweep invocations, just under
    # different parent (sweep) directories.
    assert {d.name for d in first_run_dirs} == {d.name for d in second_run_dirs}


def test_run_sweep_reuses_dataset_cache_across_invocations(tmp_path):
    config = _make_sweep_config(tmp_path)
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch as mock_fetch, soap_patch:
        run_sweep(config)
        run_sweep(config)

    # Both sweep invocations use the same crystal-system set, so the shared
    # dataset cache (independent of the per-invocation dated directory)
    # should mean fetch runs only once overall.
    assert mock_fetch.call_count == 1


def test_run_sweep_can_vary_a_training_hyperparam_not_in_any_named_axis(tmp_path):
    """The grid isn't limited to hidden dims/crystal systems -- any dotted
    config path works, e.g. sweeping the learning rate alone.
    """
    base = {
        "seed": 0,
        "output_dir": str(tmp_path / "runs"),
        "fetch": {"crystal_systems": ["cubic"], "limit_per_system": 8},
        "soap": {"r_cut": 3.0, "n_max": 2, "l_max": 2},
        "vae": {"encoder_hidden_dim": [4], "latent_dim": 2},
        "train": {"epochs": 1, "batch_size": 4, "val_ratio": 0.25},
    }
    config = SweepConfig(base=base, grid={"train.learning_rate": [0.01, 0.001, 0.0001]})
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        run_dirs = run_sweep(config)

    assert len(run_dirs) == 3
    for run_dir in run_dirs:
        assert (run_dir / "config.yaml").exists()


def test_run_sweep_writes_readme_entry(tmp_path):
    config = _make_sweep_config(tmp_path)
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        run_dirs = run_sweep(config)

    readme_path = run_dirs[0].parent.parent / "README.md"
    assert readme_path.exists()
    content = readme_path.read_text()
    sweep_name = run_dirs[0].parent.name
    assert f"## {sweep_name}" in content
    assert "Sweep axes: vae.encoder_hidden_dim" in content
    # Non-default settings: values shared by every run in the sweep show up
    # as a single value; the swept axis itself shows up as "varies: ...".
    assert "fetch.limit_per_system: 8" in content
    assert "data_source: fetch" not in content  # data_source left at default
    assert "vae.encoder_hidden_dim: varies: [4] | [8]" in content


def test_run_sweep_appends_to_existing_readme(tmp_path):
    config = _make_sweep_config(tmp_path)
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        first_run_dirs = run_sweep(config)
        second_run_dirs = run_sweep(config)

    readme_path = first_run_dirs[0].parent.parent / "README.md"
    content = readme_path.read_text()
    assert content.count("# Sweeps") == 1
    assert f"## {first_run_dirs[0].parent.name}" in content
    assert f"## {second_run_dirs[0].parent.name}" in content


def test_run_sweep_can_vary_supcon_tau_and_mode(tmp_path):
    base = {
        "seed": 0,
        "output_dir": str(tmp_path / "runs"),
        "model": "supcon",
        "fetch": {"crystal_systems": ["cubic"], "limit_per_system": 8},
        "soap": {"r_cut": 3.0, "n_max": 2, "l_max": 2},
        "vae": {"encoder_hidden_dim": [4], "latent_dim": 2},
        "train": {"epochs": 1, "batch_size": 4, "val_ratio": 0.25},
    }
    config = SweepConfig(
        base=base,
        grid={
            "supcon.tau": [0.05, 0.1],
            "supcon.mode": ["family_only", "family_and_spacegroup"],
        },
    )
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        run_dirs = run_sweep(config)

    # 2 tau x 2 modes = 4 runs, all under model-supcon_..., no name collisions
    # (tau/mode are baked into the name itself, not just deduplication
    # suffixes -- see make_run_name's "supcon-<mode>_tau<tau>" tag).
    assert len(run_dirs) == 4
    names = {d.name for d in run_dirs}
    assert len(names) == 4
    assert all(name.startswith("model-supcon") for name in names)
    assert any("tau0.05" in name for name in names)
    assert any("tau0.1_" in name for name in names)
    assert any("supcon-family_only" in name for name in names)
    assert any("supcon-family_and_spacegroup" in name for name in names)
    for run_dir in run_dirs:
        assert (run_dir / "loss_history.csv").exists()
        assert (run_dir / "embeddings.npz").exists()


def test_run_sweep_can_vary_model_kind(tmp_path):
    """Sweeping "model" produces both a VAE and an Autoencoder run in the
    same sweep directory, distinctly named (no collision), each trained
    through its own path end-to-end.
    """
    base = {
        "seed": 0,
        "output_dir": str(tmp_path / "runs"),
        "fetch": {"crystal_systems": ["cubic"], "limit_per_system": 8},
        "soap": {"r_cut": 3.0, "n_max": 2, "l_max": 2},
        "vae": {"encoder_hidden_dim": [4], "latent_dim": 2},
        "train": {"epochs": 1, "batch_size": 4, "val_ratio": 0.25},
    }
    config = SweepConfig(base=base, grid={"model": ["vae", "autoencoder"]})
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        run_dirs = run_sweep(config)

    assert len(run_dirs) == 2
    names = {d.name for d in run_dirs}
    assert names == {"hd-4_cs-cubic", "model-autoencoder_hd-4_cs-cubic"}
    for run_dir in run_dirs:
        assert (run_dir / "loss_history.csv").exists()
