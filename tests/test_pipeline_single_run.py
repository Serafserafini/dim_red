"""
Integration test for a single pipeline run: mocks the network fetch and SOAP
computation (no Materials Project access needed) but exercises the real VAE
training loop end-to-end on tiny synthetic data, and checks that every
expected artifact is written to the run directory.
"""

import csv
import dataclasses
import logging
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import read as read_atoms

pytest.importorskip("jax")

from flax import serialization

from dim_red.pipeline.config import (
    AugmentationConfig,
    AutoTailsConfig,
    AuxHeadsConfig,
    BalancedBatchingParams,
    BatchingConfig,
    ClassificationTailConfig,
    EarlyStoppingConfig,
    FetchConfig,
    GraphConfig,
    MaceConfig,
    PyxtalConfig,
    RunConfig,
    SoapConfig,
    SupConConfig,
    TailTrainSettings,
    TrainSettings,
    VAEArchConfig,
    VisualizationTailConfig,
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
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=8),
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
    assert (run_dir / "dataset.extxyz").exists()
    assert (run_dir / "embeddings.npz").exists()
    assert (run_dir / "embeddings_plot.png").exists()
    assert (run_dir / "model_params.msgpack").exists()

    with open(run_dir / "config.yaml") as f:
        saved = yaml.safe_load(f)
    assert saved["fetch"]["crystal_systems"] == ["cubic"]
    assert saved["vae"]["encoder_hidden_dim"] == [4]

    embeddings = np.load(run_dir / "embeddings.npz")
    n_total = 8  # full dataset (train + val), not just the held-out split
    assert embeddings["embeddings"].shape == (n_total, 2)  # latent_dim=2
    assert embeddings["labels"].shape[0] == n_total
    assert embeddings["material_ids"].shape[0] == n_total
    assert set(embeddings["split"].tolist()) == {"train", "val"}
    assert (embeddings["split"] == "val").sum() == 2  # val_ratio=0.25 of 8 samples

    # Standardization stats the SOAP features were derived from -- saved so
    # dim_red.pipeline.inference.load_trained_run never needs to recompute
    # SOAP on this run's own training set.
    n_features = embeddings["features"].shape[1]
    assert embeddings["feature_mean"].shape == (n_features,)
    assert embeddings["feature_std"].shape == (n_features,)

    # dataset.extxyz holds the exact structures used, same order/count as
    # embeddings.npz's arrays.
    dataset_atoms = read_atoms(run_dir / "dataset.extxyz", index=":")
    assert len(dataset_atoms) == n_total
    assert [a.info["material_id"] for a in dataset_atoms] == embeddings[
        "material_ids"
    ].tolist()

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


def test_run_single_optimizer_velo_still_works_end_to_end(tmp_path):
    """ "velo" is opt-in now (default is "adam") -- confirm it still works
    end-to-end through the full pipeline, not just the lower-level
    train_vae/train_supcon/... functions.
    """
    config = _make_config(tmp_path, name="test-run-velo")
    config = dataclasses.replace(
        config, train=dataclasses.replace(config.train, optimizer="velo")
    )
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        run_dir = run_single(config)

    assert (run_dir / "model_params.msgpack").exists()
    with open(run_dir / "run.log") as f:
        run_log = f.read()
    assert "optimizer=velo" in run_log


def test_run_single_auto_name_encodes_swept_params(tmp_path):
    config = _make_config(tmp_path, name=None)
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        run_dir = run_single(config)

    assert "hd-4" in run_dir.name
    assert "cs-cubic" in run_dir.name


def test_run_single_autoencoder_writes_expected_artifacts(tmp_path):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=8),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="autoencoder",
    )
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        run_dir = run_single(config)

    assert "model-autoencoder" in run_dir.name
    assert (run_dir / "embeddings.npz").exists()
    assert (run_dir / "model_params.msgpack").exists()

    embeddings = np.load(run_dir / "embeddings.npz")
    assert embeddings["embeddings"].shape == (8, 2)

    # No KL columns at all for a plain Autoencoder (unlike the VAE's header).
    with open(run_dir / "loss_history.csv") as f:
        header = f.readline().strip().split(",")
    assert header == [
        "epoch",
        "train_loss",
        "train_recon",
        "val_loss",
        "val_recon",
    ]


def test_run_single_autoencoder_and_vae_dont_collide_in_same_sweep_dir(tmp_path):
    base_kwargs = dict(
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=8),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        seed=0,
        output_dir=str(tmp_path / "runs"),
    )
    vae_config = RunConfig(**base_kwargs, model_kind="vae")
    ae_config = RunConfig(**base_kwargs, model_kind="autoencoder")
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        vae_run_dir = run_single(vae_config)
        ae_run_dir = run_single(ae_config)

    assert vae_run_dir != ae_run_dir
    assert "model-autoencoder" in ae_run_dir.name
    assert "model-" not in vae_run_dir.name  # default "vae" stays untagged


def test_run_single_pyxtal_data_source_writes_expected_artifacts(tmp_path):
    pytest.importorskip("pyxtal")

    config = RunConfig(
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        data_source="pyxtal",
        pyxtal=PyxtalConfig(spacegroups=[225], structures_per_spacegroup=8),
    )

    n_samples, n_features = 8, 5
    fake_atoms = []
    for i in range(n_samples):
        atoms = Atoms("Cu", positions=[[0.0, 0.0, 0.0]])
        atoms.info["material_id"] = f"pyxtal-225-{i}"
        atoms.info["spacegroup"] = 225
        atoms.info["family"] = "Cubic"
        fake_atoms.append(atoms)
    rng = np.random.default_rng(0)
    fake_soap = rng.normal(size=(n_samples, n_features)).astype(np.float32)

    with (
        patch("dim_red.generate.generate_structures", return_value=fake_atoms),
        patch("dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap),
    ):
        run_dir = run_single(config)

    assert "pyxtal-225" in run_dir.name
    assert (run_dir / "embeddings.npz").exists()
    assert (run_dir / "dataset.extxyz").exists()

    with open(run_dir / "config.yaml") as f:
        saved = yaml.safe_load(f)
    assert saved["data_source"] == "pyxtal"
    assert saved["pyxtal"]["spacegroups"] == [225]

    embeddings = np.load(run_dir / "embeddings.npz")
    assert embeddings["embeddings"].shape == (n_samples, 2)
    assert set(embeddings["labels"].tolist()) == {"Cubic"}

    dataset_atoms = read_atoms(run_dir / "dataset.extxyz", index=":")
    assert len(dataset_atoms) == n_samples
    assert {a.info["family"] for a in dataset_atoms} == {"Cubic"}
    assert set(embeddings["spacegroups"].tolist()) == {225}


def test_run_single_augmentation_expands_dataset(tmp_path):
    """RunConfig.augmentation (thermal-noise jitter / vacancies) runs before
    SOAP, end-to-end: the fetched structures get expanded into
    original+augmented copies, and every downstream artifact (dataset.extxyz,
    embeddings.npz) reflects the larger, post-augmentation count.
    """
    n_fetched = 4
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=n_fetched),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        augmentation=AugmentationConfig(
            n_augmented=1, jitter_probability=1.0, jitter_std=0.05, seed=0
        ),
        seed=0,
        output_dir=str(tmp_path / "runs"),
    )
    fake_atoms = [_fake_atoms("Cu", f"mp-{i}") for i in range(n_fetched)]

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            return_value=fake_atoms,
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap",
            side_effect=_fake_compute_soap(),
        ),
    ):
        run_dir = run_single(config)

    n_total = n_fetched * 2  # 1 original + 1 augmented copy per fetched structure
    embeddings = np.load(run_dir / "embeddings.npz")
    assert embeddings["embeddings"].shape == (n_total, 2)
    dataset_atoms = read_atoms(run_dir / "dataset.extxyz", index=":")
    assert len(dataset_atoms) == n_total
    assert sum(a.info["augmented"] for a in dataset_atoms) == n_fetched


def test_run_single_augmentation_supercell_radius_expands_single_atom_structures(
    tmp_path,
):
    """A 1-atom periodic cell (like pyxtal can generate) has almost nothing
    for jitter/vacancy augmentation to work with -- supercell_radius should
    expand every fetched structure first, end-to-end through run_single, so
    downstream artifacts reflect structures with more than 1 atom each.
    """
    n_fetched = 2
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=n_fetched),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        augmentation=AugmentationConfig(
            n_augmented=1,
            jitter_probability=1.0,
            jitter_std=0.05,
            supercell_radius=5.0,
            seed=0,
        ),
        seed=0,
        output_dir=str(tmp_path / "runs"),
    )

    def _fake_periodic_atoms(symbol, material_id):
        atoms = Atoms(
            symbol, positions=[[0.0, 0.0, 0.0]], cell=(2.0, 2.0, 2.0), pbc=True
        )
        atoms.info["material_id"] = material_id
        return atoms

    fake_atoms = [_fake_periodic_atoms("Cu", f"mp-{i}") for i in range(n_fetched)]

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            return_value=fake_atoms,
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap",
            side_effect=_fake_compute_soap(),
        ),
    ):
        run_dir = run_single(config)

    dataset_atoms = read_atoms(run_dir / "dataset.extxyz", index=":")
    assert len(dataset_atoms) == n_fetched * 2  # 1 original + 1 augmented each
    # perpendicular width 2.0, radius 5.0 -> ceil(10/2)=5 repeats/axis -> 125 atoms.
    assert all(len(a) == 125 for a in dataset_atoms)


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
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=10),
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


def test_run_single_autoencoder_family_only_aux_heads(tmp_path):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=8),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        aux_heads=AuxHeadsConfig(
            mode="family_only", lambda_family=1.0, head_hidden_dim=4
        ),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="autoencoder",
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
    assert "train_kl" not in header

    embeddings = np.load(run_dir / "embeddings.npz")
    n_total = 16
    assert "family_probs" in embeddings.files
    np.testing.assert_allclose(
        embeddings["family_probs"].sum(axis=1), np.ones(n_total), atol=1e-5
    )


def test_run_single_family_only_aux_heads_no_spacegroup_artifacts(tmp_path):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=8),
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


def _fake_fetch_with_spacegroups(spacegroup_offsets):
    def fake_fetch(crystal_system, api_key=None, limit=10):
        offset = spacegroup_offsets[crystal_system]
        return [
            _fake_atoms("Cu", f"mp-{crystal_system}-{i}", offset + (i % 2))
            for i in range(limit)
        ]

    return fake_fetch


def _fake_compute_soap(seed=0, n_features=5):
    rng = np.random.default_rng(seed)

    def fake_compute_soap(atoms, **kwargs):
        n = len(atoms) if isinstance(atoms, list) else 1
        return rng.normal(size=(n, n_features)).astype(np.float32)

    return fake_compute_soap


def test_run_single_supcon_family_and_spacegroup_writes_expected_artifacts(tmp_path):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=10),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[8], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.2),
        supcon=SupConConfig(
            mode="family_and_spacegroup",
            lambda_family=1.0,
            lambda_spacegroup=0.5,
            tau=0.1,
            projection_dim=6,
        ),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="supcon",
    )
    spacegroup_offsets = {"cubic": 195, "hexagonal": 168}

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=_fake_fetch_with_spacegroups(spacegroup_offsets),
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap",
            side_effect=_fake_compute_soap(),
        ),
    ):
        run_dir = run_single(config)

    assert "model-supcon" in run_dir.name
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "loss_history.csv").exists()
    assert (run_dir / "embeddings.npz").exists()
    assert (run_dir / "model_params.msgpack").exists()

    assert (run_dir / "projection_params.msgpack").exists()
    with open(run_dir / "projection_params.msgpack", "rb") as f:
        projection_params = serialization.msgpack_restore(f.read())
    last_layer = sorted(projection_params, key=lambda k: int(k.split("_")[-1]))[-1]
    assert projection_params[last_layer]["bias"].shape == (6,)

    with open(run_dir / "loss_history.csv") as f:
        header = f.readline().strip().split(",")
    assert header == [
        "epoch",
        "train_loss",
        "val_loss",
        "train_norm_penalty",
        "val_norm_penalty",
        "train_family_supcon",
        "val_family_supcon",
        "train_spacegroup_supcon",
        "val_spacegroup_supcon",
    ]

    embeddings = np.load(run_dir / "embeddings.npz")
    n_total = 20  # 2 crystal systems x limit_per_system=10
    assert embeddings["embeddings"].shape == (n_total, 2)
    # No classifier heads at all for SupCon -- none of these fields exist.
    assert "family_probs" not in embeddings.files
    assert "spacegroup_probs" not in embeddings.files
    assert "family_classes" not in embeddings.files
    assert "spacegroup_classes" not in embeddings.files


def test_run_single_supcon_lambda_norm_threaded_through_config(tmp_path):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=10),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[8], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.2),
        supcon=SupConConfig(mode="family_only", tau=0.1, lambda_norm=0.3),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="supcon",
    )

    def fake_fetch(crystal_system, api_key=None, limit=10):
        return [_fake_atoms("Cu", f"mp-{crystal_system}-{i}") for i in range(limit)]

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=fake_fetch,
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap",
            side_effect=_fake_compute_soap(),
        ),
    ):
        run_dir = run_single(config)

    with open(run_dir / "loss_history.csv") as f:
        header = f.readline().strip().split(",")
    assert "train_norm_penalty" in header
    assert "val_norm_penalty" in header

    with open(run_dir / "run.log") as f:
        run_log = f.read()
    assert "lambda_norm=0.300" in run_log


def test_run_single_supcon_default_lambda_norm_still_writes_norm_penalty_columns(
    tmp_path,
):
    """lambda_norm defaults to 0.0 (disabled) but train_norm_penalty/
    val_norm_penalty must still always be present in loss_history.csv --
    it's a diagnostic, tracked regardless of whether it's weighted in.
    """
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=8),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        supcon=SupConConfig(mode="family_only"),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="supcon",
    )
    assert config.supcon.lambda_norm == 0.0

    def fake_fetch(crystal_system, api_key=None, limit=10):
        return [_fake_atoms("Cu", f"mp-{crystal_system}-{i}") for i in range(limit)]

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=fake_fetch,
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap",
            side_effect=_fake_compute_soap(),
        ),
    ):
        run_dir = run_single(config)

    with open(run_dir / "loss_history.csv") as f:
        header = f.readline().strip().split(",")
    assert "train_norm_penalty" in header
    assert "val_norm_penalty" in header


def test_run_single_supcon_family_only_mode_history_and_name(tmp_path):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=8),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        supcon=SupConConfig(mode="family_only"),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="supcon",
    )

    def fake_fetch(crystal_system, api_key=None, limit=10):
        return [_fake_atoms("Cu", f"mp-{crystal_system}-{i}") for i in range(limit)]

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=fake_fetch,
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap",
            side_effect=_fake_compute_soap(),
        ),
    ):
        run_dir = run_single(config)

    with open(run_dir / "loss_history.csv") as f:
        header = f.readline().strip().split(",")
    assert "train_family_supcon" in header
    assert "train_spacegroup_supcon" not in header

    embeddings = np.load(run_dir / "embeddings.npz")
    assert "family_probs" not in embeddings.files


def test_run_single_supcon_and_vae_dont_collide_in_same_sweep_dir(tmp_path):
    base_kwargs = dict(
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=8),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        seed=0,
        output_dir=str(tmp_path / "runs"),
    )
    vae_config = RunConfig(**base_kwargs, model_kind="vae")
    supcon_config = RunConfig(
        **base_kwargs, model_kind="supcon", supcon=SupConConfig(mode="family_only")
    )
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        vae_run_dir = run_single(vae_config)
        supcon_run_dir = run_single(supcon_config)

    assert vae_run_dir != supcon_run_dir
    assert "model-supcon" in supcon_run_dir.name
    assert (supcon_run_dir / "embeddings.npz").exists()


def test_run_single_supcon_balanced_batching_writes_expected_artifacts(
    tmp_path, caplog
):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=20),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[8], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.2),
        supcon=SupConConfig(mode="family_and_spacegroup", tau=0.1),
        batching=BatchingConfig(
            strategy="balanced",
            balanced_params=BalancedBatchingParams(P=None, K=6, S=2),
        ),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="supcon",
    )
    spacegroup_offsets = {"cubic": 195, "hexagonal": 168}

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=_fake_fetch_with_spacegroups(spacegroup_offsets),
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap",
            side_effect=_fake_compute_soap(),
        ),
        caplog.at_level(logging.WARNING, logger="dim_red.pipeline"),
    ):
        run_dir = run_single(config)

    assert "model-supcon" in run_dir.name
    assert (run_dir / "loss_history.csv").exists()
    assert (run_dir / "embeddings.npz").exists()

    # The effective-batch-size warning (train.batch_size ignored for
    # balanced batching) must show up in run.log, not just stderr.
    with open(run_dir / "run.log") as f:
        run_log = f.read()
    assert "is ignored for training batches" in run_log

    embeddings = np.load(run_dir / "embeddings.npz")
    n_total = 40  # 2 crystal systems x limit_per_system=20
    assert embeddings["embeddings"].shape == (n_total, 2)


def test_run_single_supcon_balanced_batching_spacegroup_only_mode(tmp_path):
    """batching.strategy='balanced' still groups by family even when
    supcon.mode='spacegroup_only' means the family SupCon loss term itself
    is inactive -- the two are decoupled (see single_run.py's
    need_family_ids/balanced_batching logic).
    """
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=20),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[8], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.2),
        supcon=SupConConfig(mode="spacegroup_only", tau=0.1),
        batching=BatchingConfig(
            strategy="balanced", balanced_params=BalancedBatchingParams(K=6)
        ),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="supcon",
    )
    spacegroup_offsets = {"cubic": 195, "hexagonal": 168}

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=_fake_fetch_with_spacegroups(spacegroup_offsets),
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap",
            side_effect=_fake_compute_soap(),
        ),
    ):
        run_dir = run_single(config)

    with open(run_dir / "loss_history.csv") as f:
        header = f.readline().strip().split(",")
    assert "train_spacegroup_supcon" in header
    assert "train_family_supcon" not in header


def test_run_single_supcon_random_batching_is_default_and_backward_compatible(
    tmp_path,
):
    """A RunConfig built without ever touching `batching` (as every
    pre-existing supcon test in this file already does) must behave exactly
    as before -- no warning, plain shuffle.
    """
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=8),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        supcon=SupConConfig(mode="family_only"),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="supcon",
    )
    assert config.batching.strategy == "random"

    def fake_fetch(crystal_system, api_key=None, limit=10):
        return [_fake_atoms("Cu", f"mp-{crystal_system}-{i}") for i in range(limit)]

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=fake_fetch,
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap",
            side_effect=_fake_compute_soap(),
        ),
    ):
        run_dir = run_single(config)

    with open(run_dir / "run.log") as f:
        run_log = f.read()
    assert "is ignored for training batches" not in run_log
    assert (run_dir / "embeddings.npz").exists()


def test_run_single_vae_early_stopping_stops_before_configured_epochs(tmp_path):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=8),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(
            epochs=40,
            batch_size=4,
            val_ratio=0.25,
            early_stopping=EarlyStoppingConfig(enabled=True, patience=2),
        ),
        seed=0,
        output_dir=str(tmp_path / "runs"),
    )
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        run_dir = run_single(config)

    with open(run_dir / "loss_history.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) < 40

    with open(run_dir / "run.log") as f:
        run_log = f.read()
    assert "Early stopping: training stopped after" in run_log

    # Downstream artifacts (encode over the whole dataset, plot, params
    # save) must all still work fine with a shorter-than-configured history.
    assert (run_dir / "embeddings.npz").exists()
    assert (run_dir / "model_params.msgpack").exists()
    embeddings = np.load(run_dir / "embeddings.npz")
    assert embeddings["embeddings"].shape == (8, 2)


def test_run_single_autoencoder_early_stopping_stops_before_configured_epochs(
    tmp_path,
):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=8),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(
            epochs=40,
            batch_size=4,
            val_ratio=0.25,
            early_stopping=EarlyStoppingConfig(enabled=True, patience=2),
        ),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="autoencoder",
    )
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        run_dir = run_single(config)

    with open(run_dir / "loss_history.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) < 40
    with open(run_dir / "run.log") as f:
        assert "Early stopping: training stopped after" in f.read()


def test_run_single_supcon_early_stopping_stops_before_configured_epochs(tmp_path):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=10),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[8], latent_dim=2),
        train=TrainSettings(
            epochs=40,
            batch_size=4,
            val_ratio=0.2,
            # Tuned against VeLO's convergence timing (faked fast by
            # tests/conftest.py's fixture) -- this test is about early
            # stopping's own bookkeeping, not optimizer choice.
            optimizer="velo",
            early_stopping=EarlyStoppingConfig(enabled=True, patience=2),
        ),
        supcon=SupConConfig(mode="family_only", tau=0.1),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="supcon",
    )

    def fake_fetch(crystal_system, api_key=None, limit=10):
        return [_fake_atoms("Cu", f"mp-{crystal_system}-{i}") for i in range(limit)]

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=fake_fetch,
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap",
            side_effect=_fake_compute_soap(),
        ),
    ):
        run_dir = run_single(config)

    with open(run_dir / "loss_history.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) < 40
    with open(run_dir / "run.log") as f:
        assert "Early stopping: training stopped after" in f.read()


def test_run_single_early_stopping_disabled_by_default(tmp_path):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=8),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(epochs=3, batch_size=4, val_ratio=0.25),
        seed=0,
        output_dir=str(tmp_path / "runs"),
    )
    assert config.train.early_stopping.enabled is False
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        run_dir = run_single(config)

    with open(run_dir / "loss_history.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    with open(run_dir / "run.log") as f:
        assert "Early stopping" not in f.read()


def _make_supcon_config_with_tails(tmp_path, tails):
    return RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=10),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[8], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.2),
        supcon=SupConConfig(mode="family_and_spacegroup", tau=0.1, projection_dim=6),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="supcon",
        tails=tails,
    )


def test_run_single_supcon_auto_trains_both_tails(tmp_path):
    tails = AutoTailsConfig(
        classification=ClassificationTailConfig(mode="family_only"),
        visualization=VisualizationTailConfig(viz_dim=2),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )
    config = _make_supcon_config_with_tails(tmp_path, tails)
    spacegroup_offsets = {"cubic": 195, "hexagonal": 168}

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=_fake_fetch_with_spacegroups(spacegroup_offsets),
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap",
            side_effect=_fake_compute_soap(),
        ),
    ):
        run_dir = run_single(config)

    classification_dir = run_dir / "tails" / "classification"
    visualization_dir = run_dir / "tails" / "visualization"
    assert (classification_dir / "tail_predictions.npz").exists()
    assert (classification_dir / "tail_params.msgpack").exists()
    assert (classification_dir / "run.log").exists()
    assert (visualization_dir / "tail_embeddings.npz").exists()
    assert (visualization_dir / "tail_params.msgpack").exists()
    assert (visualization_dir / "run.log").exists()

    with open(run_dir / "run.log") as f:
        run_log = f.read()
    assert "Auto-training classification tail on this run" in run_log
    assert "Auto-trained classification tail saved to" in run_log
    assert "Auto-training visualization tail on this run" in run_log
    assert "Auto-trained visualization tail saved to" in run_log


def test_run_single_supcon_auto_trains_classification_tail_only(tmp_path):
    tails = AutoTailsConfig(
        classification=ClassificationTailConfig(mode="family_only"),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )
    config = _make_supcon_config_with_tails(tmp_path, tails)
    spacegroup_offsets = {"cubic": 195, "hexagonal": 168}

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=_fake_fetch_with_spacegroups(spacegroup_offsets),
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap",
            side_effect=_fake_compute_soap(),
        ),
    ):
        run_dir = run_single(config)

    assert (run_dir / "tails" / "classification").exists()
    assert not (run_dir / "tails" / "visualization").exists()


def test_run_single_tails_ignored_for_non_supcon_model_kind(tmp_path):
    tails = AutoTailsConfig(
        classification=ClassificationTailConfig(mode="family_only"),
        visualization=VisualizationTailConfig(viz_dim=2),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=8),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        tails=tails,
    )
    fetch_patch, soap_patch = _patch_dataset()

    with fetch_patch, soap_patch:
        run_dir = run_single(config)

    assert not (run_dir / "tails").exists()


# --- model_kind == "cgcnn" ---------------------------------------------------
#
# No SOAP mocking needed (dim_red.cgcnn.graph is cheap, pure NumPy/ASE) --
# only fetch_structures_by_crystal_system is mocked, with real (if tiny)
# periodic structures so graph construction is meaningful.


def _fake_cgcnn_atoms(symbol: str, material_id: str, spacegroup: int) -> Atoms:
    atoms = Atoms(
        symbol * 2,
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    atoms.info["material_id"] = material_id
    atoms.info["spacegroup"] = spacegroup
    return atoms


def _fake_fetch_cgcnn_with_spacegroups(spacegroup_offsets):
    def fake_fetch(crystal_system, api_key=None, limit=10):
        offset = spacegroup_offsets[crystal_system]
        return [
            _fake_cgcnn_atoms("Cu", f"mp-{crystal_system}-{i}", offset + (i % 2))
            for i in range(limit)
        ]

    return fake_fetch


def _make_cgcnn_config(tmp_path, aux_mode="family_only", tails=None) -> RunConfig:
    return RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=10),
        soap=SoapConfig(),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=4),
        train=TrainSettings(epochs=2, batch_size=4, val_ratio=0.2),
        graph=GraphConfig(
            radius=3.0,
            max_num_nbr=4,
            max_species=2,
            atom_fea_len=8,
            n_conv=1,
            h_fea_len=8,
        ),
        aux_heads=AuxHeadsConfig(mode=aux_mode, head_hidden_dim=4),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="cgcnn",
        tails=tails,
    )


def test_run_single_cgcnn_writes_expected_artifacts(tmp_path):
    config = _make_cgcnn_config(tmp_path)
    spacegroup_offsets = {"cubic": 195, "hexagonal": 168}

    with patch(
        "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
        side_effect=_fake_fetch_cgcnn_with_spacegroups(spacegroup_offsets),
    ):
        run_dir = run_single(config)

    assert "model-cgcnn" in run_dir.name
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "loss_history.csv").exists()
    assert (run_dir / "dataset.extxyz").exists()
    assert (run_dir / "embeddings.npz").exists()
    assert (run_dir / "embeddings_plot.png").exists()
    assert (run_dir / "model_params.msgpack").exists()

    embeddings = np.load(run_dir / "embeddings.npz")
    n_total = 20  # 2 crystal systems x limit_per_system=10
    assert embeddings["embeddings"].shape == (n_total, 4)  # latent_dim=4
    assert "family_probs" in embeddings.files
    assert "family_classes" in embeddings.files
    # No natural flat feature vector/standardization stats exist for graph
    # features -- these are SOAP-only fields, omitted entirely for cgcnn.
    assert "features" not in embeddings.files
    assert "feature_mean" not in embeddings.files
    assert "feature_std" not in embeddings.files


def test_run_single_cgcnn_family_and_spacegroup_writes_expected_artifacts(tmp_path):
    config = _make_cgcnn_config(tmp_path, aux_mode="family_and_spacegroup")
    spacegroup_offsets = {"cubic": 195, "hexagonal": 168}

    with patch(
        "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
        side_effect=_fake_fetch_cgcnn_with_spacegroups(spacegroup_offsets),
    ):
        run_dir = run_single(config)

    with open(run_dir / "loss_history.csv") as f:
        header = f.readline().strip().split(",")
    assert header == [
        "epoch",
        "train_loss",
        "train_family_ce",
        "val_loss",
        "val_family_ce",
        "train_spacegroup_ce",
        "val_spacegroup_ce",
    ]

    embeddings = np.load(run_dir / "embeddings.npz")
    n_total = 20
    assert set(embeddings["family_classes"].tolist()) == {"Cubic", "Hexagonal"}
    assert set(embeddings["spacegroup_classes"].tolist()) == {168, 169, 195, 196}
    assert embeddings["family_probs"].shape == (n_total, 2)
    assert embeddings["spacegroup_probs"].shape == (n_total, 4)
    np.testing.assert_allclose(
        embeddings["family_probs"].sum(axis=1), np.ones(n_total), atol=1e-5
    )


def test_run_single_cgcnn_auto_trains_visualization_tail(tmp_path):
    tails = AutoTailsConfig(
        visualization=VisualizationTailConfig(viz_dim=2, mode="family_only"),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )
    config = _make_cgcnn_config(tmp_path, tails=tails)
    spacegroup_offsets = {"cubic": 195, "hexagonal": 168}

    with patch(
        "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
        side_effect=_fake_fetch_cgcnn_with_spacegroups(spacegroup_offsets),
    ):
        run_dir = run_single(config)

    visualization_dir = run_dir / "tails" / "visualization"
    assert (visualization_dir / "tail_embeddings.npz").exists()
    assert (visualization_dir / "tail_params.msgpack").exists()


# --- model_kind == "mace" -----------------------------------------------------
#
# dim_red.mace.model.MaceEncoder requires mace_jax (not installed in every
# test environment), so it's mocked out entirely here -- the same pattern
# tests/test_pipeline_dataset_cache.py uses. Reuses _fake_cgcnn_atoms/
# _fake_fetch_cgcnn_with_spacegroups above (generic to any model_kind).


class _FakeMaceEncoder:
    """Deterministic stand-in for dim_red.mace.model.MaceEncoder: encode()
    returns one fixed-width row per structure, ignoring atoms_list content.
    ``params`` is a plain dict of numpy arrays -- flax.serialization can
    (de)serialize any pytree of arrays, not just an nn.Module's output, so
    this round-trips through run_single's model_params.msgpack save exactly
    like a real MaceEncoder's frozen checkpoint params would.
    """

    _DIM = 3

    def __init__(self, **kwargs):
        self.params = {"dummy": np.zeros(1, dtype=np.float32)}

    def encode(self, atoms_list):
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(atoms_list), self._DIM)).astype(np.float32)


def _make_mace_config(tmp_path, tails=None) -> RunConfig:
    return RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=10),
        soap=SoapConfig(),
        vae=VAEArchConfig(encoder_hidden_dim=[1], latent_dim=1),
        train=TrainSettings(device="cpu"),
        mace=MaceConfig(checkpoint_path="/fake/ckpt", r_max=5.0),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="mace",
        tails=tails,
    )


def test_run_single_mace_writes_expected_artifacts(tmp_path):
    config = _make_mace_config(tmp_path)
    spacegroup_offsets = {"cubic": 195, "hexagonal": 168}

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=_fake_fetch_cgcnn_with_spacegroups(spacegroup_offsets),
        ),
        patch("dim_red.mace.model.MaceEncoder", _FakeMaceEncoder),
    ):
        run_dir = run_single(config)

    assert "model-mace" in run_dir.name
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "dataset.extxyz").exists()
    assert (run_dir / "embeddings.npz").exists()
    assert (run_dir / "embeddings_plot.png").exists()
    assert (run_dir / "model_params.msgpack").exists()
    # No training loop at all for a frozen body -- no loss to report.
    assert not (run_dir / "loss_history.csv").exists()

    embeddings = np.load(run_dir / "embeddings.npz")
    n_total = 20  # 2 crystal systems x limit_per_system=10
    assert embeddings["embeddings"].shape == (n_total, 3)  # _FakeMaceEncoder._DIM
    # SOAP-shaped payload (feature_mean/feature_std always present, unlike
    # cgcnn's graph-shaped payload which omits them entirely).
    assert "feature_mean" in embeddings.files
    assert "feature_std" in embeddings.files
    assert "features" in embeddings.files
    # No classifier heads on a frozen body -- never saved for this model_kind.
    assert "family_probs" not in embeddings.files
    assert "spacegroup_probs" not in embeddings.files


def test_run_single_mace_auto_trains_classification_tail(tmp_path):
    tails = AutoTailsConfig(
        classification=ClassificationTailConfig(mode="family_only"),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )
    config = _make_mace_config(tmp_path, tails=tails)
    spacegroup_offsets = {"cubic": 195, "hexagonal": 168}

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=_fake_fetch_cgcnn_with_spacegroups(spacegroup_offsets),
        ),
        patch("dim_red.mace.model.MaceEncoder", _FakeMaceEncoder),
    ):
        run_dir = run_single(config)

    classification_dir = run_dir / "tails" / "classification"
    assert (classification_dir / "tail_predictions.npz").exists()
    assert (classification_dir / "tail_params.msgpack").exists()


def test_run_single_tails_none_leaves_behavior_unchanged(tmp_path):
    config = _make_supcon_config_with_tails(tmp_path, tails=None)
    spacegroup_offsets = {"cubic": 195, "hexagonal": 168}

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=_fake_fetch_with_spacegroups(spacegroup_offsets),
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap",
            side_effect=_fake_compute_soap(),
        ),
    ):
        run_dir = run_single(config)

    assert not (run_dir / "tails").exists()
