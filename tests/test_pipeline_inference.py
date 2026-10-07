"""
Integration tests for dim_red.pipeline.inference: applying an already-
trained cgcnn run's model to new structures. Builds a real run directory via
run_single (mocking the network fetch, same pattern as
test_pipeline_single_run.py) and then exercises
load_trained_run/encode_structures/apply_model_to_structures against it.

No SOAP mocking needed anywhere here (cgcnn never calls compute_soap) --
only fetch_structures_by_crystal_system is mocked, with real (if tiny)
periodic structures.
"""

from unittest.mock import patch

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write as write_atoms

pytest.importorskip("jax")

from dim_red.pipeline.config import (
    AuxHeadsConfig,
    EncoderConfig,
    FetchConfig,
    GraphConfig,
    RunConfig,
    SoapConfig,
    TrainSettings,
    run_config_to_dict,
)
from dim_red.pipeline.inference import (
    LoadedRun,
    RunEmbeddings,
    apply_model_to_structures,
    encode_structures,
    load_run_embeddings,
    load_trained_run,
)
from dim_red.pipeline.single_run import run_single

N_TRAIN_CGCNN = 8


def _fake_cgcnn_atoms(symbol: str, material_id: str) -> Atoms:
    atoms = Atoms(
        symbol * 2,
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    atoms.info["material_id"] = material_id
    atoms.info["spacegroup"] = 195
    return atoms


def _train_a_cgcnn_run(tmp_path, name="cgcnn-run", latent_dim=4):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=N_TRAIN_CGCNN),
        soap=SoapConfig(),
        encoder=EncoderConfig(encoder_hidden_dim=[4], latent_dim=latent_dim),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        graph=GraphConfig(
            radius=3.0,
            max_num_nbr=4,
            max_species=1,
            atom_fea_len=8,
            n_conv=1,
            h_fea_len=8,
        ),
        aux_heads=AuxHeadsConfig(mode="family_only", head_hidden_dim=4),
        model_kind="cgcnn",
        seed=0,
        output_dir=str(tmp_path / "runs"),
        name=name,
    )
    fake_atoms = [_fake_cgcnn_atoms("Cu", f"mp-{i}") for i in range(N_TRAIN_CGCNN)]
    with patch(
        "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
        return_value=fake_atoms,
    ):
        run_dir = run_single(config)
    return run_dir


def test_load_trained_run_cgcnn(tmp_path):
    run_dir = _train_a_cgcnn_run(tmp_path)
    loaded = load_trained_run(run_dir)

    assert isinstance(loaded, LoadedRun)
    assert loaded.config.model_kind == "cgcnn"
    assert loaded.embeddings["embeddings"].shape == (N_TRAIN_CGCNN, 4)


def test_load_trained_run_missing_files_raises(tmp_path):
    empty_dir = tmp_path / "not-a-run"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_trained_run(empty_dir)


def test_load_trained_run_refuses_a_non_cgcnn_run_with_a_pointer_to_fullstack(
    tmp_path,
):
    run_dir = tmp_path / "supcon-run"
    run_dir.mkdir()
    # A config.yaml as run_single wrote it for a supcon run: same shape as a
    # cgcnn one, with "model: supcon".
    saved = run_config_to_dict(
        RunConfig(
            fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=4),
            soap=SoapConfig(),
            encoder=EncoderConfig(encoder_hidden_dim=[4], latent_dim=2),
            train=TrainSettings(epochs=1, batch_size=4),
            aux_heads=AuxHeadsConfig(mode="family_only"),
        )
    )
    saved["model"] = "supcon"
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(saved, f, sort_keys=False)
    for name in ("dataset.extxyz", "model_params.msgpack"):
        (run_dir / name).write_bytes(b"")
    np.savez(run_dir / "embeddings.npz", embeddings=np.zeros((1, 2)))

    with pytest.raises(ValueError, match="FullStack"):
        load_trained_run(run_dir)


def test_load_run_embeddings_reads_config_and_embeddings_only(tmp_path):
    """load_run_embeddings must work even without dataset.extxyz/
    model_params.msgpack, since dim_red.pipeline.tail_training.train_tail
    relies on exactly that.
    """
    run_dir = _train_a_cgcnn_run(tmp_path)
    (run_dir / "dataset.extxyz").unlink()
    (run_dir / "model_params.msgpack").unlink()

    loaded = load_run_embeddings(run_dir)

    assert isinstance(loaded, RunEmbeddings)
    assert loaded.config.model_kind == "cgcnn"
    assert loaded.embeddings["embeddings"].shape == (N_TRAIN_CGCNN, 4)


def test_load_run_embeddings_missing_files_raises(tmp_path):
    empty_dir = tmp_path / "not-a-run"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_run_embeddings(empty_dir)


def test_encode_structures_cgcnn_new_structure_larger_than_training_max_atoms(tmp_path):
    """The key regression test for max_atoms being a per-call array-shape
    convenience, not a trained-parameter constraint: a brand-new structure
    with more atoms than any training structure ever had must encode
    without error."""
    run_dir = _train_a_cgcnn_run(tmp_path)
    loaded = load_trained_run(run_dir)

    big = Atoms(
        "Cu" * 20,
        positions=np.random.default_rng(0).uniform(0, 10, size=(20, 3)),
        cell=[15, 15, 15],
        pbc=True,
    )
    z = encode_structures(loaded, [big])
    assert z.shape == (1, 4)  # latent_dim=4
    assert np.all(np.isfinite(z))


def test_encode_structures_cgcnn_rejects_empty_list(tmp_path):
    run_dir = _train_a_cgcnn_run(tmp_path)
    loaded = load_trained_run(run_dir)
    with pytest.raises(ValueError, match="atoms_list is empty"):
        encode_structures(loaded, [])


def test_apply_model_to_structures_cgcnn_writes_expected_artifacts(tmp_path):
    run_dir = _train_a_cgcnn_run(tmp_path)

    new_atoms = [_fake_cgcnn_atoms("Cu", "new-1"), _fake_cgcnn_atoms("Cu", "new-2")]
    structures_path = tmp_path / "new_cgcnn_structures.extxyz"
    write_atoms(str(structures_path), new_atoms, format="extxyz")

    output_dir = apply_model_to_structures(run_dir, structures_path)

    assert output_dir == run_dir / "applied"
    npz_path = output_dir / "new_cgcnn_structures_embeddings.npz"
    plot_path = output_dir / "new_cgcnn_structures_latent_space.png"
    assert npz_path.exists()
    assert plot_path.exists()
    with np.load(npz_path) as npz:
        assert npz["embeddings"].shape == (2, 4)
        assert npz["material_ids"].tolist() == ["new-1", "new-2"]
        assert "labels" not in npz.files


def test_apply_model_to_structures_uses_label_field(tmp_path):
    run_dir = _train_a_cgcnn_run(tmp_path, latent_dim=2)

    new_atoms = [_fake_cgcnn_atoms("Cu", "new-1"), _fake_cgcnn_atoms("Cu", "new-2")]
    new_atoms[0].info["family"] = "Cubic"
    new_atoms[1].info["family"] = "Hexagonal"
    structures_path = tmp_path / "labeled.extxyz"
    write_atoms(str(structures_path), new_atoms, format="extxyz")

    custom_output_dir = tmp_path / "custom_out"
    output_dir = apply_model_to_structures(
        run_dir,
        structures_path,
        output_dir=custom_output_dir,
        label_field="family",
    )

    assert output_dir == custom_output_dir
    with np.load(output_dir / "labeled_embeddings.npz") as npz:
        assert npz["embeddings"].shape == (2, 2)
        assert npz["labels"].tolist() == ["Cubic", "Hexagonal"]


def test_apply_model_to_structures_projects_non_2d_latent_via_umap(tmp_path):
    pytest.importorskip("umap")
    from dim_red.pipeline.compare import LatentUmapParams

    run_dir = _train_a_cgcnn_run(tmp_path, latent_dim=3, name="run-3d")

    new_atoms = [_fake_cgcnn_atoms("Cu", "new-1"), _fake_cgcnn_atoms("Cu", "new-2")]
    structures_path = tmp_path / "new_structures_3d.extxyz"
    write_atoms(str(structures_path), new_atoms, format="extxyz")

    output_dir = apply_model_to_structures(
        run_dir,
        structures_path,
        umap_params=LatentUmapParams(n_neighbors=3),
    )

    with np.load(output_dir / "new_structures_3d_embeddings.npz") as npz:
        assert npz["embeddings"].shape == (2, 3)  # raw latent coords, not projected
    assert (output_dir / "new_structures_3d_latent_space.png").exists()
