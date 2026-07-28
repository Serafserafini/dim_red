"""
Integration test for a single pipeline run: mocks the network fetch and SOAP
computation (no Materials Project access needed) but exercises the real VAE
training loop end-to-end on tiny synthetic data, and checks that every
expected artifact is written to the run directory.
"""

from unittest.mock import patch

import numpy as np
import pytest
import yaml
from ase import Atoms

pytest.importorskip("jax")

from dim_red.pipeline.config import RunConfig, SoapConfig, TrainSettings, VAEArchConfig
from dim_red.pipeline.single_run import run_single


def _fake_atoms(symbol: str, material_id: str) -> Atoms:
    atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
    atoms.info["material_id"] = material_id
    return atoms


def _make_config(tmp_path, name=None) -> RunConfig:
    return RunConfig(
        crystal_systems=["cubic"],
        limit_per_system=8,
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        name=name,
    )


def _patch_dataset(n_samples: int = 8, n_features: int = 5):
    fake_atoms = [_fake_atoms("Cu", f"mp-{i}") for i in range(n_samples)]
    rng = np.random.default_rng(0)
    fake_soap = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    return patch(
        "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
        return_value=fake_atoms,
    ), patch("dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap)


def test_run_single_writes_expected_artifacts(tmp_path):
    config = _make_config(tmp_path, name="test-run")
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        run_dir = run_single(config)

    assert run_dir.name == "test-run"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "loss_history.csv").exists()
    assert (run_dir / "embeddings.npz").exists()
    assert (run_dir / "embeddings_plot.png").exists()
    assert (run_dir / "model_params.msgpack").exists()

    with open(run_dir / "config.yaml") as f:
        saved = yaml.safe_load(f)
    assert saved["data"]["crystal_systems"] == ["cubic"]
    assert saved["vae"]["encoder_hidden_dim"] == [4]

    embeddings = np.load(run_dir / "embeddings.npz")
    n_total = 8  # full dataset (train + val), not just the held-out split
    assert embeddings["embeddings"].shape == (n_total, 2)  # latent_dim=2
    assert embeddings["labels"].shape[0] == n_total
    assert embeddings["material_ids"].shape[0] == n_total
    assert set(embeddings["split"].tolist()) == {"train", "val"}
    assert (embeddings["split"] == "val").sum() == 2  # val_ratio=0.25 of 8 samples

    with open(run_dir / "loss_history.csv") as f:
        header = f.readline().strip().split(",")
    assert header == [
        "epoch",
        "train_loss",
        "train_recon",
        "train_kl",
        "val_loss",
        "val_recon",
        "val_kl",
    ]


def test_run_single_auto_name_encodes_swept_params(tmp_path):
    config = _make_config(tmp_path, name=None)
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        run_dir = run_single(config)

    assert "hd-4" in run_dir.name
    assert "cs-cubic" in run_dir.name


def test_run_single_reuses_dataset_cache_across_runs(tmp_path):
    config_a = _make_config(tmp_path, name="run-a")
    config_b = _make_config(tmp_path, name="run-b")
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch as mock_fetch, soap_patch:
        run_single(config_a)
        run_single(config_b)

    # Same crystal_systems/soap/limit -> both runs should share one cache
    # entry, so fetch only happens once across the two runs.
    assert mock_fetch.call_count == 1
