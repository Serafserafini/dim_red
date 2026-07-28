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

from dim_red.pipeline.config import (
    AuxHeadsConfig,
    RunConfig,
    SoapConfig,
    TrainSettings,
    VAEArchConfig,
)
from dim_red.pipeline.single_run import run_single


def _fake_atoms(symbol: str, material_id: str, spacegroup: int = None) -> Atoms:
    atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
    atoms.info["material_id"] = material_id
    if spacegroup is not None:
        atoms.info["spacegroup"] = spacegroup
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


def test_run_single_family_and_spacegroup_aux_heads(tmp_path):
    config = RunConfig(
        crystal_systems=["cubic", "hexagonal"],
        limit_per_system=10,
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[8], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.2),
        aux_heads=AuxHeadsConfig(
            mode="family_and_spacegroup",
            lambda_family=1.0,
            lambda_spacegroup=0.5,
            head_hidden_dim=4,
        ),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        name=None,
    )

    spacegroup_offsets = {"cubic": 195, "hexagonal": 168}

    def fake_fetch(crystal_system, api_key=None, limit=10):
        offset = spacegroup_offsets[crystal_system]
        return [
            _fake_atoms("Cu", f"mp-{crystal_system}-{i}", offset + (i % 2))
            for i in range(limit)
        ]

    rng = np.random.default_rng(0)

    def fake_compute_soap(atoms, **kwargs):
        n = len(atoms) if isinstance(atoms, list) else 1
        return rng.normal(size=(n, 5)).astype(np.float32)

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=fake_fetch,
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap", side_effect=fake_compute_soap
        ),
    ):
        run_dir = run_single(config)

    assert "aux-family_and_spacegroup" in run_dir.name

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
        "train_family_ce",
        "val_family_ce",
        "train_spacegroup_ce",
        "val_spacegroup_ce",
    ]

    embeddings = np.load(run_dir / "embeddings.npz")
    n_total = 20  # 2 crystal systems x limit_per_system=10
    assert set(embeddings["family_classes"].tolist()) == {"Cubic", "Hexagonal"}
    assert set(embeddings["spacegroup_classes"].tolist()) == {168, 169, 195, 196}
    assert embeddings["family_probs"].shape == (n_total, 2)
    assert embeddings["spacegroup_probs"].shape == (n_total, 4)
    np.testing.assert_allclose(
        embeddings["family_probs"].sum(axis=1), np.ones(n_total), atol=1e-5
    )
    np.testing.assert_allclose(
        embeddings["spacegroup_probs"].sum(axis=1), np.ones(n_total), atol=1e-5
    )


def test_run_single_family_only_aux_heads_no_spacegroup_artifacts(tmp_path):
    config = RunConfig(
        crystal_systems=["cubic", "hexagonal"],
        limit_per_system=8,
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        aux_heads=AuxHeadsConfig(
            mode="family_only", lambda_family=1.0, head_hidden_dim=4
        ),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        name="family-only-run",
    )

    def fake_fetch(crystal_system, api_key=None, limit=10):
        return [_fake_atoms("Cu", f"mp-{crystal_system}-{i}") for i in range(limit)]

    rng = np.random.default_rng(0)

    def fake_compute_soap(atoms, **kwargs):
        n = len(atoms) if isinstance(atoms, list) else 1
        return rng.normal(size=(n, 5)).astype(np.float32)

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=fake_fetch,
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap", side_effect=fake_compute_soap
        ),
    ):
        run_dir = run_single(config)

    with open(run_dir / "loss_history.csv") as f:
        header = f.readline().strip().split(",")
    assert "train_family_ce" in header
    assert "train_spacegroup_ce" not in header

    embeddings = np.load(run_dir / "embeddings.npz")
    assert "family_probs" in embeddings.files
    assert "spacegroup_probs" not in embeddings.files
