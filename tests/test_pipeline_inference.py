"""
Integration tests for dim_red.pipeline.inference: applying an already-
trained run's model to new structures. Builds a real run directory via
run_single (mocking network fetch + SOAP, same pattern as
test_pipeline_single_run.py) and then exercises
load_trained_run/encode_structures/apply_model_to_structures against it.
"""

import logging
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.io import write as write_atoms

pytest.importorskip("jax")

from dim_red.pipeline.config import (
    AuxHeadsConfig,
    FetchConfig,
    GraphConfig,
    HierarchicalTailConfig,
    MaceConfig,
    RunConfig,
    SoapConfig,
    SupConConfig,
    TailTrainConfig,
    TailTrainSettings,
    TrainSettings,
    VAEArchConfig,
)
from dim_red.pipeline.inference import (
    LoadedRun,
    RunEmbeddings,
    apply_model_to_structures,
    encode_structures,
    load_run_embeddings,
    load_trained_run,
    predict_hierarchical,
)
from dim_red.pipeline.single_run import run_single
from dim_red.pipeline.tail_training import train_tail

N_TRAIN = 8
N_FEATURES = 5


def _fake_atoms(symbol: str, material_id: str) -> Atoms:
    atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
    atoms.info["material_id"] = material_id
    return atoms


def _make_config(
    tmp_path, model_kind="vae", latent_dim=2, name="base-run"
) -> RunConfig:
    return RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=N_TRAIN),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=latent_dim),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        aux_heads=AuxHeadsConfig(mode="none"),
        model_kind=model_kind,
        seed=0,
        output_dir=str(tmp_path / "runs"),
        name=name,
    )


def _train_a_run(tmp_path, **config_kwargs):
    config = _make_config(tmp_path, **config_kwargs)
    fake_atoms = [_fake_atoms("Cu", f"mp-{i}") for i in range(N_TRAIN)]
    rng = np.random.default_rng(0)
    fake_soap_train = rng.normal(size=(N_TRAIN, N_FEATURES)).astype(np.float32)

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            return_value=fake_atoms,
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap_train
        ),
    ):
        run_dir = run_single(config)
    return run_dir, fake_soap_train


def _inference_soap_side_effect(fake_soap_train, seed=7):
    def _side_effect(atoms, **kwargs):
        n = len(atoms)
        if n == N_TRAIN:
            # Recomputing SOAP on the exact training structures must
            # reproduce exactly what run_single saw, so load_trained_run's
            # standardization-mismatch check stays quiet.
            return fake_soap_train
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n, N_FEATURES)).astype(np.float32)

    return _side_effect


def test_load_trained_run_reconstructs_model_and_standardization(tmp_path):
    run_dir, fake_soap_train = _train_a_run(tmp_path)

    # The fast path (feature_mean/feature_std already saved in embeddings.npz
    # by run_single) must never call compute_soap at all -- that's the whole
    # point of this fix, so any call here is a regression.
    with patch(
        "dim_red.pipeline.inference.compute_soap",
        side_effect=AssertionError(
            "load_trained_run's fast path must not recompute SOAP"
        ),
    ):
        loaded = load_trained_run(run_dir)

    assert isinstance(loaded, LoadedRun)
    assert loaded.config.model_kind == "vae"
    assert loaded.species == ["Cu"]
    assert loaded.feature_mean.shape == (N_FEATURES,)
    assert loaded.feature_std.shape == (N_FEATURES,)
    np.testing.assert_allclose(
        loaded.feature_mean, fake_soap_train.mean(axis=0), atol=1e-5
    )
    assert loaded.embeddings["embeddings"].shape == (N_TRAIN, 2)


def _strip_standardization_stats(run_dir):
    """Rewrites run_dir/embeddings.npz without feature_mean/feature_std, to
    simulate a run written before those fields existed (forcing
    load_trained_run's slow fallback path)."""
    npz_path = run_dir / "embeddings.npz"
    with np.load(npz_path) as npz:
        payload = {
            k: v for k, v in npz.items() if k not in ("feature_mean", "feature_std")
        }
    np.savez(npz_path, **payload)


def test_load_trained_run_falls_back_for_runs_missing_standardization_stats(
    tmp_path, caplog
):
    run_dir, fake_soap_train = _train_a_run(tmp_path)
    _strip_standardization_stats(run_dir)

    with (
        patch(
            "dim_red.pipeline.inference.compute_soap",
            side_effect=_inference_soap_side_effect(fake_soap_train),
        ),
        caplog.at_level(logging.WARNING, logger="dim_red.pipeline"),
    ):
        loaded = load_trained_run(run_dir)

    assert loaded.feature_mean.shape == (N_FEATURES,)
    assert loaded.feature_std.shape == (N_FEATURES,)
    np.testing.assert_allclose(
        loaded.feature_mean, fake_soap_train.mean(axis=0), atol=1e-5
    )
    assert any("no feature_mean/feature_std" in rec.message for rec in caplog.records)


def test_load_trained_run_warns_on_standardization_mismatch(tmp_path, caplog):
    """The mismatch check only runs on the slow fallback path (the fast path
    has nothing recomputed to compare against embeddings.npz['features'])."""
    run_dir, _fake_soap_train = _train_a_run(tmp_path)
    _strip_standardization_stats(run_dir)

    def _mismatched_side_effect(atoms, **kwargs):
        rng = np.random.default_rng(999)
        return rng.normal(size=(len(atoms), N_FEATURES)).astype(np.float32)

    with (
        patch(
            "dim_red.pipeline.inference.compute_soap",
            side_effect=_mismatched_side_effect,
        ),
        caplog.at_level(logging.WARNING, logger="dim_red.pipeline"),
    ):
        load_trained_run(run_dir)

    assert any("differ from" in rec.message for rec in caplog.records)


def test_load_trained_run_missing_files_raises(tmp_path):
    empty_dir = tmp_path / "not-a-run"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_trained_run(empty_dir)


def test_load_run_embeddings_reads_config_and_embeddings_only(tmp_path):
    """load_run_embeddings must never touch compute_soap -- it doesn't
    reconstruct the model, resolve species, or need standardization stats at
    all -- and must work even without dataset.extxyz/model_params.msgpack,
    since dim_red.pipeline.tail_training.train_tail relies on exactly that.
    """
    run_dir, _fake_soap_train = _train_a_run(tmp_path)
    (run_dir / "dataset.extxyz").unlink()
    (run_dir / "model_params.msgpack").unlink()

    with patch(
        "dim_red.pipeline.inference.compute_soap",
        side_effect=AssertionError("load_run_embeddings must not touch SOAP"),
    ):
        loaded = load_run_embeddings(run_dir)

    assert isinstance(loaded, RunEmbeddings)
    assert loaded.config.model_kind == "vae"
    assert loaded.embeddings["embeddings"].shape == (N_TRAIN, 2)


def test_load_run_embeddings_missing_files_raises(tmp_path):
    empty_dir = tmp_path / "not-a-run"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_run_embeddings(empty_dir)


def test_encode_structures_returns_finite_latent_coords(tmp_path):
    run_dir, fake_soap_train = _train_a_run(tmp_path)

    with patch(
        "dim_red.pipeline.inference.compute_soap",
        side_effect=_inference_soap_side_effect(fake_soap_train),
    ):
        loaded = load_trained_run(run_dir)
        new_atoms = [_fake_atoms("Cu", "new-1"), _fake_atoms("Cu", "new-2")]
        z = encode_structures(loaded, new_atoms)

    assert z.shape == (2, 2)  # latent_dim=2
    assert np.all(np.isfinite(z))


def test_encode_structures_rejects_empty_list(tmp_path):
    run_dir, fake_soap_train = _train_a_run(tmp_path)
    with patch(
        "dim_red.pipeline.inference.compute_soap",
        side_effect=_inference_soap_side_effect(fake_soap_train),
    ):
        loaded = load_trained_run(run_dir)
    with pytest.raises(ValueError, match="atoms_list is empty"):
        encode_structures(loaded, [])


def test_apply_model_to_structures_writes_expected_artifacts(tmp_path):
    run_dir, fake_soap_train = _train_a_run(tmp_path)

    new_atoms = [
        _fake_atoms("Cu", "new-1"),
        _fake_atoms("Cu", "new-2"),
        _fake_atoms("Cu", "new-3"),
    ]
    structures_path = tmp_path / "new_structures.extxyz"
    write_atoms(str(structures_path), new_atoms, format="extxyz")

    with patch(
        "dim_red.pipeline.inference.compute_soap",
        side_effect=_inference_soap_side_effect(fake_soap_train),
    ):
        output_dir = apply_model_to_structures(run_dir, structures_path)

    assert output_dir == run_dir / "applied"
    npz_path = output_dir / "new_structures_embeddings.npz"
    plot_path = output_dir / "new_structures_latent_space.png"
    assert npz_path.exists()
    assert plot_path.exists()

    with np.load(npz_path) as npz:
        assert npz["embeddings"].shape == (3, 2)
        assert npz["material_ids"].tolist() == ["new-1", "new-2", "new-3"]
        assert "labels" not in npz.files


def test_apply_model_to_structures_uses_label_field(tmp_path):
    run_dir, fake_soap_train = _train_a_run(tmp_path)

    new_atoms = [_fake_atoms("Cu", "new-1"), _fake_atoms("Cu", "new-2")]
    new_atoms[0].info["family"] = "Cubic"
    new_atoms[1].info["family"] = "Hexagonal"
    structures_path = tmp_path / "labeled.extxyz"
    write_atoms(str(structures_path), new_atoms, format="extxyz")

    custom_output_dir = tmp_path / "custom_out"
    with patch(
        "dim_red.pipeline.inference.compute_soap",
        side_effect=_inference_soap_side_effect(fake_soap_train),
    ):
        output_dir = apply_model_to_structures(
            run_dir,
            structures_path,
            output_dir=custom_output_dir,
            label_field="family",
        )

    assert output_dir == custom_output_dir
    with np.load(output_dir / "labeled_embeddings.npz") as npz:
        assert npz["labels"].tolist() == ["Cubic", "Hexagonal"]


def test_apply_model_to_structures_projects_non_2d_latent_via_umap(tmp_path):
    pytest.importorskip("umap")
    from dim_red.pipeline.compare import LatentUmapParams

    run_dir, fake_soap_train = _train_a_run(tmp_path, latent_dim=3, name="run-3d")

    new_atoms = [_fake_atoms("Cu", "new-1"), _fake_atoms("Cu", "new-2")]
    structures_path = tmp_path / "new_structures_3d.extxyz"
    write_atoms(str(structures_path), new_atoms, format="extxyz")

    with patch(
        "dim_red.pipeline.inference.compute_soap",
        side_effect=_inference_soap_side_effect(fake_soap_train),
    ):
        output_dir = apply_model_to_structures(
            run_dir,
            structures_path,
            umap_params=LatentUmapParams(n_neighbors=3),
        )

    with np.load(output_dir / "new_structures_3d_embeddings.npz") as npz:
        assert npz["embeddings"].shape == (2, 3)  # raw latent coords, not projected
    assert (output_dir / "new_structures_3d_latent_space.png").exists()


# --- model_kind == "cgcnn" ---------------------------------------------------
#
# No SOAP mocking needed anywhere here (cgcnn never calls compute_soap) --
# only fetch_structures_by_crystal_system is mocked, with real (if tiny)
# periodic structures.

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


def _train_a_cgcnn_run(tmp_path, name="cgcnn-run"):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=N_TRAIN_CGCNN),
        soap=SoapConfig(),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=4),
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

    assert loaded.config.model_kind == "cgcnn"
    assert loaded.species == []
    assert loaded.feature_mean.shape == (0,)
    assert loaded.feature_std.shape == (0,)


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

    npz_path = output_dir / "new_cgcnn_structures_embeddings.npz"
    plot_path = output_dir / "new_cgcnn_structures_latent_space.png"
    assert npz_path.exists()
    assert plot_path.exists()
    with np.load(npz_path) as npz:
        assert npz["embeddings"].shape == (2, 4)
        assert npz["material_ids"].tolist() == ["new-1", "new-2"]


# --- model_kind == "mace" -----------------------------------------------------
#
# dim_red.mace.model.MaceEncoder requires mace_jax (not installed in every
# test environment), so it's mocked out entirely here -- same pattern as
# test_pipeline_single_run.py's _FakeMaceEncoder.

N_TRAIN_MACE = 8


class _FakeMaceEncoder:
    """Deterministic stand-in for dim_red.mace.model.MaceEncoder -- returns
    one fixed-width row per structure, keyed only by each structure's
    material_id (so distinct new structures don't collide with training
    ones), ignoring geometry entirely.
    """

    _DIM = 3

    def __init__(self, **kwargs):
        self.params = {"dummy": np.zeros(1, dtype=np.float32)}

    def encode(self, atoms_list):
        rows = []
        for a in atoms_list:
            seed = abs(hash(a.info.get("material_id", "unknown"))) % (2**31)
            rows.append(np.random.default_rng(seed).normal(size=self._DIM))
        return np.asarray(rows, dtype=np.float32)


def _fake_mace_atoms(symbol: str, material_id: str) -> Atoms:
    atoms = Atoms(
        symbol * 2,
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    atoms.info["material_id"] = material_id
    atoms.info["spacegroup"] = 195
    return atoms


def _train_a_mace_run(tmp_path, name="mace-run"):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=N_TRAIN_MACE),
        soap=SoapConfig(),
        vae=VAEArchConfig(encoder_hidden_dim=[1], latent_dim=1),
        train=TrainSettings(device="cpu"),
        mace=MaceConfig(checkpoint_path="/fake/ckpt", r_max=5.0),
        model_kind="mace",
        seed=0,
        output_dir=str(tmp_path / "runs"),
        name=name,
    )
    fake_atoms = [_fake_mace_atoms("Cu", f"mp-{i}") for i in range(N_TRAIN_MACE)]
    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            return_value=fake_atoms,
        ),
        patch("dim_red.mace.model.MaceEncoder", _FakeMaceEncoder),
    ):
        run_dir = run_single(config)
    return run_dir


def test_load_trained_run_mace(tmp_path):
    run_dir = _train_a_mace_run(tmp_path)
    with patch("dim_red.mace.model.MaceEncoder", _FakeMaceEncoder):
        loaded = load_trained_run(run_dir)

    assert loaded.config.model_kind == "mace"
    assert loaded.species == []
    # Real (non-placeholder) standardization stats, unlike cgcnn's zeros/ones
    # -- mace embeddings are standardized the same way SOAP features are.
    assert loaded.feature_mean.shape == (3,)
    assert loaded.feature_std.shape == (3,)


def test_encode_structures_mace_applies_saved_standardization(tmp_path):
    run_dir = _train_a_mace_run(tmp_path)
    with patch("dim_red.mace.model.MaceEncoder", _FakeMaceEncoder):
        loaded = load_trained_run(run_dir)
        z = encode_structures(loaded, [_fake_mace_atoms("Cu", "new-1")])

    assert z.shape == (1, 3)
    assert np.all(np.isfinite(z))


def test_encode_structures_mace_rejects_empty_list(tmp_path):
    run_dir = _train_a_mace_run(tmp_path)
    with patch("dim_red.mace.model.MaceEncoder", _FakeMaceEncoder):
        loaded = load_trained_run(run_dir)
    with pytest.raises(ValueError, match="atoms_list is empty"):
        encode_structures(loaded, [])


def test_apply_model_to_structures_mace_writes_expected_artifacts(tmp_path):
    run_dir = _train_a_mace_run(tmp_path)

    new_atoms = [_fake_mace_atoms("Cu", "new-1"), _fake_mace_atoms("Cu", "new-2")]
    structures_path = tmp_path / "new_mace_structures.extxyz"
    write_atoms(str(structures_path), new_atoms, format="extxyz")

    with patch("dim_red.mace.model.MaceEncoder", _FakeMaceEncoder):
        output_dir = apply_model_to_structures(run_dir, structures_path)

    npz_path = output_dir / "new_mace_structures_embeddings.npz"
    plot_path = output_dir / "new_mace_structures_latent_space.png"
    assert npz_path.exists()
    assert plot_path.exists()
    with np.load(npz_path) as npz:
        assert npz["embeddings"].shape == (2, 3)
        assert npz["material_ids"].tolist() == ["new-1", "new-2"]


# --- predict_hierarchical -----------------------------------------------------


def _fake_atoms_with_spacegroup(
    symbol: str, material_id: str, spacegroup: int
) -> Atoms:
    atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
    atoms.info["material_id"] = material_id
    atoms.info["spacegroup"] = spacegroup
    return atoms


def _train_a_supcon_run_with_hierarchical_tail(tmp_path, min_samples_per_expert=5):
    """A completed model_kind='supcon' run (2 families, 2 spacegroups each,
    mirroring test_pipeline_tail_training.py's _train_supcon_run) with a
    hierarchical tail already trained on top of it.
    """
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=10),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[8], latent_dim=3),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        supcon=SupConConfig(mode="family_and_spacegroup", tau=0.1, projection_dim=6),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="supcon",
    )
    spacegroup_offsets = {"cubic": 195, "hexagonal": 168}
    rng = np.random.default_rng(0)

    def fake_fetch(crystal_system, api_key=None, limit=10):
        offset = spacegroup_offsets[crystal_system]
        return [
            _fake_atoms_with_spacegroup(
                "Cu", f"mp-{crystal_system}-{i}", offset + (i % 2)
            )
            for i in range(limit)
        ]

    def fake_train_soap(atoms, **kwargs):
        n = len(atoms) if isinstance(atoms, list) else 1
        return rng.normal(size=(n, N_FEATURES)).astype(np.float32)

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=fake_fetch,
        ),
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap", side_effect=fake_train_soap
        ),
    ):
        run_dir = run_single(config)

    tail_config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="hierarchical",
        hierarchical=HierarchicalTailConfig(
            head_hidden_dim=8, min_samples_per_expert=min_samples_per_expert
        ),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )
    train_tail(tail_config)
    return run_dir


def test_predict_hierarchical_predicts_family_and_spacegroup_for_new_structures(
    tmp_path,
):
    run_dir = _train_a_supcon_run_with_hierarchical_tail(tmp_path)
    new_atoms = [Atoms("Cu", positions=[[0.0, 0.0, 0.0]]) for _ in range(3)]

    def fake_new_soap(atoms, **kwargs):
        n = len(atoms)
        return np.random.default_rng(1).normal(size=(n, N_FEATURES)).astype(np.float32)

    with patch("dim_red.pipeline.inference.compute_soap", side_effect=fake_new_soap):
        prediction = predict_hierarchical(run_dir, new_atoms)

    assert len(prediction.family_pred) == 3
    assert set(prediction.family_pred) <= {"Cubic", "Hexagonal"}
    assert prediction.family_probs.shape == (3, 2)
    np.testing.assert_allclose(
        prediction.family_probs.sum(axis=1), np.ones(3), atol=1e-4
    )
    assert prediction.spacegroup_pred.shape == (3,)
    assert all(isinstance(sg, (int, np.integer)) for sg in prediction.spacegroup_pred)
    assert len(prediction.spacegroup_probs) == 3
    for family, probs in zip(prediction.family_pred, prediction.spacegroup_probs):
        # Both families got a dedicated expert (min_samples_per_expert=5, see
        # test_train_tail_hierarchical_writes_expected_artifacts) -- every
        # row should have a real probability distribution, not a fallback.
        assert probs is not None
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-4)


def test_predict_hierarchical_uses_fallback_for_low_sample_families(tmp_path):
    run_dir = _train_a_supcon_run_with_hierarchical_tail(
        tmp_path, min_samples_per_expert=100
    )
    new_atoms = [Atoms("Cu", positions=[[0.0, 0.0, 0.0]])]

    def fake_new_soap(atoms, **kwargs):
        n = len(atoms)
        return np.random.default_rng(1).normal(size=(n, N_FEATURES)).astype(np.float32)

    with patch("dim_red.pipeline.inference.compute_soap", side_effect=fake_new_soap):
        prediction = predict_hierarchical(run_dir, new_atoms)

    # Every family fell back (min_samples_per_expert=100 exceeds both
    # families' training-row counts) -- no expert to report a distribution
    # from.
    assert prediction.spacegroup_probs == [None]


def test_predict_hierarchical_raises_for_missing_tail(tmp_path):
    run_dir, _ = _train_a_run(tmp_path, model_kind="vae")
    with pytest.raises(FileNotFoundError, match="tail_params.msgpack"):
        predict_hierarchical(run_dir, [_fake_atoms("Cu", "new-1")])
