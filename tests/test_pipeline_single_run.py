"""
Integration test for a single pipeline run: mocks the network fetch /
structure generation (no Materials Project access needed) but exercises the
real CGCNN training loop end-to-end on tiny synthetic data, and checks that
every expected artifact is written to the run directory.

``run_single`` trains only ``model_kind == "cgcnn"``; supcon/supcon_mace runs
go through ``dim_red.pipeline.full_stack.FullStack``.
"""

import csv
import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import read as read_atoms

pytest.importorskip("jax")

from dim_red.pipeline.config import (
    AugmentationConfig,
    AuxHeadsConfig,
    EarlyStoppingConfig,
    EncoderConfig,
    FetchConfig,
    GraphConfig,
    PyxtalConfig,
    RunConfig,
    SoapConfig,
    TrainSettings,
)
from dim_red.pipeline.single_run import run_single

# No SOAP mocking needed (dim_red.cgcnn.graph is cheap, pure NumPy/ASE) --
# only fetch_structures_by_crystal_system (or pyxtal's generate_structures) is
# mocked, with real (if tiny) periodic structures so graph construction is
# meaningful.

_SPACEGROUP_OFFSETS = {"cubic": 195, "hexagonal": 168}


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


def _fake_fetch_cgcnn_with_spacegroups(spacegroup_offsets=_SPACEGROUP_OFFSETS):
    def fake_fetch(crystal_system, api_key=None, limit=10):
        offset = spacegroup_offsets[crystal_system]
        return [
            _fake_cgcnn_atoms("Cu", f"mp-{crystal_system}-{i}", offset + (i % 2))
            for i in range(limit)
        ]

    return fake_fetch


def _patch_fetch(**kwargs):
    if not kwargs:
        kwargs = {"side_effect": _fake_fetch_cgcnn_with_spacegroups()}
    return patch(
        "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system", **kwargs
    )


def _make_cgcnn_config(tmp_path, aux_mode="family_only") -> RunConfig:
    return RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=10),
        soap=SoapConfig(),
        encoder=EncoderConfig(encoder_hidden_dim=[4], latent_dim=4),
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
    )


@pytest.mark.parametrize("model_kind", ["supcon", "supcon_mace"])
def test_run_single_refuses_supcon_with_a_pointer_to_fullstack(tmp_path, model_kind):
    # RunConfig itself can no longer hold a non-cgcnn model_kind (see
    # tests/test_pipeline_config.py), so the run_single guard -- defense in
    # depth -- is exercised with a minimal stand-in config.
    config = SimpleNamespace(
        model_kind=model_kind, output_dir=str(tmp_path / "runs"), name=None
    )
    with pytest.raises(ValueError, match="FullStack"):
        run_single(config)
    # Refused before anything is written.
    assert not (tmp_path / "runs").exists()


def test_run_single_cgcnn_writes_expected_artifacts(tmp_path):
    config = _make_cgcnn_config(tmp_path)

    with _patch_fetch():
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

    with _patch_fetch():
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


def test_run_single_auto_name_encodes_swept_params(tmp_path):
    config = _make_cgcnn_config(tmp_path)

    with _patch_fetch():
        run_dir = run_single(config)

    assert run_dir.name == "model-cgcnn_hd-4_cs-cubic-hexagonal_aux-family_only_lf1"


def test_run_single_explicit_name_is_used(tmp_path):
    config = dataclasses.replace(_make_cgcnn_config(tmp_path), name="test-run")

    with _patch_fetch():
        run_dir = run_single(config)

    assert run_dir.name == "test-run"
    with open(run_dir / "config.yaml") as f:
        saved = yaml.safe_load(f)
    assert saved["model"] == "cgcnn"
    assert saved["fetch"]["crystal_systems"] == ["cubic", "hexagonal"]


def test_run_single_pyxtal_data_source_writes_expected_artifacts(tmp_path):
    pytest.importorskip("pyxtal")

    config = dataclasses.replace(
        _make_cgcnn_config(tmp_path),
        fetch=None,
        data_source="pyxtal",
        pyxtal=PyxtalConfig(spacegroups=[225], structures_per_spacegroup=8),
    )

    n_samples = 8
    fake_atoms = []
    for i in range(n_samples):
        atoms = _fake_cgcnn_atoms("Cu", f"pyxtal-225-{i}", 225)
        atoms.info["family"] = "Cubic"
        fake_atoms.append(atoms)

    with patch("dim_red.generate.generate_structures", return_value=fake_atoms):
        run_dir = run_single(config)

    assert "pyxtal-225" in run_dir.name
    assert (run_dir / "embeddings.npz").exists()
    assert (run_dir / "dataset.extxyz").exists()

    with open(run_dir / "config.yaml") as f:
        saved = yaml.safe_load(f)
    assert saved["data_source"] == "pyxtal"
    assert saved["pyxtal"]["spacegroups"] == [225]

    embeddings = np.load(run_dir / "embeddings.npz")
    assert embeddings["embeddings"].shape == (n_samples, 4)
    assert set(embeddings["labels"].tolist()) == {"Cubic"}

    dataset_atoms = read_atoms(run_dir / "dataset.extxyz", index=":")
    assert len(dataset_atoms) == n_samples
    assert {a.info["family"] for a in dataset_atoms} == {"Cubic"}
    assert set(embeddings["spacegroups"].tolist()) == {225}


def test_run_single_augmentation_expands_dataset(tmp_path):
    """RunConfig.augmentation (thermal-noise jitter / vacancies) runs before
    graph construction, end-to-end: the fetched structures get expanded into
    original+augmented copies, and every downstream artifact (dataset.extxyz,
    embeddings.npz) reflects the larger, post-augmentation count.
    """
    n_per_system = 4
    config = dataclasses.replace(
        _make_cgcnn_config(tmp_path),
        fetch=FetchConfig(
            crystal_systems=["cubic", "hexagonal"], limit_per_system=n_per_system
        ),
        augmentation=AugmentationConfig(
            n_augmented=1, jitter_probability=1.0, jitter_std=0.05, seed=0
        ),
    )

    with _patch_fetch():
        run_dir = run_single(config)

    n_fetched = 2 * n_per_system
    n_total = n_fetched * 2  # 1 original + 1 augmented copy per fetched structure
    embeddings = np.load(run_dir / "embeddings.npz")
    assert embeddings["embeddings"].shape == (n_total, 4)
    dataset_atoms = read_atoms(run_dir / "dataset.extxyz", index=":")
    assert len(dataset_atoms) == n_total
    assert sum(a.info["augmented"] for a in dataset_atoms) == n_fetched


def test_run_single_augmentation_split_never_separates_sibling_copies(tmp_path):
    """Every augmented copy of a structure shares its original's
    ``material_id`` (dim_red.augmentation.augment_structures) -- the train/val
    split must therefore keep an original and all of its augmented copies on
    the same side, never splitting siblings across train and val (which
    would leak a near-duplicate of a training row into validation).
    """
    config = dataclasses.replace(
        _make_cgcnn_config(tmp_path),
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=4),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        augmentation=AugmentationConfig(
            n_augmented=3, jitter_probability=1.0, jitter_std=0.05, seed=0
        ),
    )

    with _patch_fetch():
        run_dir = run_single(config)

    embeddings = np.load(run_dir / "embeddings.npz")
    material_ids = embeddings["material_ids"]
    split = embeddings["split"]
    for material_id in set(material_ids.tolist()):
        splits_for_group = set(split[material_ids == material_id].tolist())
        assert len(splits_for_group) == 1, (
            f"material_id={material_id!r} has copies split across "
            f"{splits_for_group} -- train/val split leaked sibling copies"
        )
    # Sanity check both sides are still non-empty (val_ratio=0.25 of 8 groups
    # -> 2 groups in val, each contributing 1 original + 3 augmented copies).
    assert set(split.tolist()) == {"train", "val"}
    assert (split == "val").sum() == 2 * 4


def test_run_single_augmentation_supercell_radius_expands_single_atom_structures(
    tmp_path,
):
    """A 1-atom periodic cell (like pyxtal can generate) has almost nothing
    for jitter/vacancy augmentation to work with -- supercell_radius should
    expand every fetched structure first, end-to-end through run_single, so
    downstream artifacts reflect structures with more than 1 atom each.
    """
    n_fetched = 2
    config = dataclasses.replace(
        _make_cgcnn_config(tmp_path),
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=n_fetched),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        augmentation=AugmentationConfig(
            n_augmented=1,
            jitter_probability=1.0,
            jitter_std=0.05,
            supercell_radius=5.0,
            seed=0,
        ),
    )

    def _fake_periodic_atoms(symbol, material_id):
        atoms = Atoms(
            symbol, positions=[[0.0, 0.0, 0.0]], cell=(2.0, 2.0, 2.0), pbc=True
        )
        atoms.info["material_id"] = material_id
        return atoms

    fake_atoms = [_fake_periodic_atoms("Cu", f"mp-{i}") for i in range(n_fetched)]

    with _patch_fetch(return_value=fake_atoms):
        run_dir = run_single(config)

    dataset_atoms = read_atoms(run_dir / "dataset.extxyz", index=":")
    assert len(dataset_atoms) == n_fetched * 2  # 1 original + 1 augmented each
    # perpendicular width 2.0, radius 5.0 -> ceil(10/2)=5 repeats/axis -> 125 atoms.
    assert all(len(a) == 125 for a in dataset_atoms)


def test_run_single_reuses_dataset_cache_across_runs(tmp_path):
    base = _make_cgcnn_config(tmp_path)
    config_a = dataclasses.replace(base, name="run-a")
    config_b = dataclasses.replace(base, name="run-b")

    with _patch_fetch() as mock_fetch:
        run_single(config_a)
        n_calls_first_run = mock_fetch.call_count
        run_single(config_b)

    # Same fetch/graph settings -> both runs share one cache entry, so the
    # second run never fetches.
    assert n_calls_first_run > 0
    assert mock_fetch.call_count == n_calls_first_run


def test_run_single_early_stopping_stops_before_configured_epochs(tmp_path):
    config = dataclasses.replace(
        _make_cgcnn_config(tmp_path),
        train=TrainSettings(
            epochs=40,
            batch_size=4,
            val_ratio=0.2,
            # An improvement of this size never happens, so training stops
            # after patience + 1 epochs -- the test is about early stopping's
            # own bookkeeping, not how fast the optimizer converges.
            early_stopping=EarlyStoppingConfig(
                enabled=True, patience=2, min_delta=100.0
            ),
        ),
    )

    with _patch_fetch():
        run_dir = run_single(config)

    with open(run_dir / "loss_history.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) < 40
    with open(run_dir / "run.log") as f:
        assert "Early stopping: training stopped after" in f.read()


def test_run_single_early_stopping_disabled_by_default(tmp_path):
    config = dataclasses.replace(
        _make_cgcnn_config(tmp_path),
        train=TrainSettings(epochs=3, batch_size=4, val_ratio=0.2),
    )
    assert config.train.early_stopping.enabled is False

    with _patch_fetch():
        run_dir = run_single(config)

    with open(run_dir / "loss_history.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    with open(run_dir / "run.log") as f:
        assert "Early stopping" not in f.read()
