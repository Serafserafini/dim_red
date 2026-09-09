"""
Integration tests for dim_red.pipeline.tail_training: phase 2 of the SupCon
body/tail workflow -- freeze an already-trained ``model: supcon`` run's body
and train a classification or visualization tail on its saved
representations. Builds a real run directory via run_single (mocking network
fetch + SOAP, same pattern as test_pipeline_single_run.py) and then
exercises train_tail against it.
"""

from unittest.mock import patch

import numpy as np
import pytest
import yaml
from ase import Atoms

pytest.importorskip("jax")

from dim_red.pipeline.config import (
    AuxHeadsConfig,
    BalancedBatchingParams,
    BatchingConfig,
    ClassificationTailConfig,
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
    VisualizationTailConfig,
)
from dim_red.pipeline.single_run import run_single
from dim_red.pipeline.tail_training import train_tail


def _fake_atoms(symbol: str, material_id: str, spacegroup: int) -> Atoms:
    atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
    atoms.info["material_id"] = material_id
    atoms.info["spacegroup"] = spacegroup
    return atoms


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


def _train_supcon_run(tmp_path, projection_dim=6, latent_dim=3):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=10),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[8], latent_dim=latent_dim),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        supcon=SupConConfig(
            mode="family_and_spacegroup", tau=0.1, projection_dim=projection_dim
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
    return run_dir


def _train_a_vae_run(tmp_path, model_kind="vae"):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=8),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(epochs=1, batch_size=4, val_ratio=0.25),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind=model_kind,
    )
    fake_atoms = [_fake_atoms("Cu", f"mp-{i}", 1) for i in range(8)]
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
    return run_dir


def _train_an_autoencoder_run(tmp_path):
    return _train_a_vae_run(tmp_path, model_kind="autoencoder")


def test_train_tail_classification_writes_expected_artifacts(tmp_path):
    run_dir = _train_supcon_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="classification",
        classification=ClassificationTailConfig(mode="family_and_spacegroup"),
        train=TailTrainSettings(epochs=2, batch_size=4),
    )

    tail_dir = train_tail(config)

    assert tail_dir.parent.name == "tails"
    assert tail_dir.name == "classification"
    assert (tail_dir / "tail_config.yaml").exists()
    assert (tail_dir / "tail_params.msgpack").exists()
    assert (tail_dir / "loss_history.csv").exists()
    assert (tail_dir / "tail_predictions.npz").exists()
    assert (tail_dir / "run.log").exists()

    with open(tail_dir / "run.log") as f:
        run_log = f.read()
    assert "Training tail_kind=classification" in run_log
    assert "Tail training complete" in run_log
    assert "Tail artifacts saved to" in run_log

    with open(tail_dir / "loss_history.csv") as f:
        header = f.readline().strip().split(",")
    assert header == [
        "epoch",
        "train_loss",
        "val_loss",
        "train_family_ce",
        "val_family_ce",
        "train_spacegroup_ce",
        "val_spacegroup_ce",
    ]

    predictions = np.load(tail_dir / "tail_predictions.npz")
    n_total = 20  # 2 crystal systems x limit_per_system=10
    assert predictions["family_probs"].shape == (n_total, 2)  # 2 families
    assert predictions["spacegroup_probs"].shape[0] == n_total
    np.testing.assert_allclose(
        predictions["family_probs"].sum(axis=1), np.ones(n_total), atol=1e-4
    )
    # tail_predictions.npz is self-sufficient: true labels are saved
    # alongside the predicted probabilities, no join against the parent
    # run's embeddings.npz needed to evaluate the classifier later.
    assert predictions["labels"].shape[0] == n_total
    assert predictions["spacegroups"].shape[0] == n_total

    # A confusion matrix, a per-class precision/recall/F1 bar chart, and a
    # calibration diagram, for both family and spacegroup, on both splits.
    for level in ("family", "spacegroup"):
        for split in ("train", "val"):
            for kind in ("confusion_matrix", "classification_report", "calibration"):
                assert (tail_dir / f"{kind}_{level}_{split}.png").exists()


def test_train_tail_classification_family_only_has_no_spacegroup_columns(tmp_path):
    run_dir = _train_supcon_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="classification",
        classification=ClassificationTailConfig(mode="family_only"),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )

    tail_dir = train_tail(config)

    with open(tail_dir / "loss_history.csv") as f:
        header = f.readline().strip().split(",")
    assert "train_family_ce" in header
    assert "train_spacegroup_ce" not in header

    predictions = np.load(tail_dir / "tail_predictions.npz")
    assert "family_probs" in predictions.files
    assert "spacegroup_probs" not in predictions.files
    assert "labels" in predictions.files
    assert "spacegroups" not in predictions.files

    # No spacegroup term active -> only the family evaluation plots exist.
    for split in ("train", "val"):
        for kind in ("confusion_matrix", "classification_report", "calibration"):
            assert (tail_dir / f"{kind}_family_{split}.png").exists()
            assert not (tail_dir / f"{kind}_spacegroup_{split}.png").exists()


def test_train_tail_visualization_2d_writes_expected_artifacts(tmp_path):
    run_dir = _train_supcon_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="visualization",
        visualization=VisualizationTailConfig(viz_dim=2, mode="family_and_spacegroup"),
        train=TailTrainSettings(epochs=2, batch_size=4),
    )

    tail_dir = train_tail(config)

    assert (tail_dir / "tail_embeddings.npz").exists()
    assert (tail_dir / "viz_plot_family.png").exists()
    assert (tail_dir / "viz_plot_spacegroup.png").exists()
    assert (tail_dir / "run.log").exists()
    with open(tail_dir / "run.log") as f:
        run_log = f.read()
    assert "Training tail_kind=visualization" in run_log

    embeddings = np.load(tail_dir / "tail_embeddings.npz")
    n_total = 20
    assert embeddings["embeddings"].shape == (n_total, 2)
    assert embeddings["labels"].shape[0] == n_total
    assert set(embeddings["split"].tolist()) == {"train", "val"}


def test_train_tail_visualization_3d_writes_expected_artifacts(tmp_path):
    run_dir = _train_supcon_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="visualization",
        visualization=VisualizationTailConfig(viz_dim=3, mode="family_only"),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )

    tail_dir = train_tail(config)

    embeddings = np.load(tail_dir / "tail_embeddings.npz")
    assert embeddings["embeddings"].shape == (20, 3)
    assert (tail_dir / "viz_plot_family.png").exists()
    # mode="family_only" -- no spacegroup term active, no spacegroup plot.
    assert not (tail_dir / "viz_plot_spacegroup.png").exists()


def test_train_tail_visualization_balanced_batching(tmp_path):
    run_dir = _train_supcon_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="visualization",
        visualization=VisualizationTailConfig(
            viz_dim=2,
            mode="family_and_spacegroup",
            batching=BatchingConfig(
                strategy="balanced", balanced_params=BalancedBatchingParams(K=4, S=2)
            ),
        ),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )

    tail_dir = train_tail(config)

    assert (tail_dir / "tail_embeddings.npz").exists()


def test_train_tail_hierarchical_writes_expected_artifacts(tmp_path):
    run_dir = _train_supcon_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="hierarchical",
        hierarchical=HierarchicalTailConfig(
            head_hidden_dim=8, min_samples_per_expert=5
        ),
        train=TailTrainSettings(epochs=2, batch_size=4),
    )

    tail_dir = train_tail(config)

    assert tail_dir.parent.name == "tails"
    assert tail_dir.name == "hierarchical"
    assert (tail_dir / "tail_config.yaml").exists()
    assert (tail_dir / "tail_predictions.npz").exists()
    assert (tail_dir / "family_expert_status.yaml").exists()
    assert (tail_dir / "run.log").exists()

    # Stage 1 (family).
    assert (tail_dir / "family" / "tail_params.msgpack").exists()
    assert (tail_dir / "family" / "loss_history.csv").exists()

    # Both families ("Cubic"/"Hexagonal") have enough training rows (7/8,
    # both >= min_samples_per_expert=5) and 2 distinct spacegroups each -->
    # both get a dedicated expert, not a fallback.
    with open(tail_dir / "family_expert_status.yaml") as f:
        status = yaml.safe_load(f)["families"]
    status_by_family = {entry["family"]: entry for entry in status}
    assert set(status_by_family) == {"Cubic", "Hexagonal"}
    for family, entry in status_by_family.items():
        assert entry["expert"] is True
        assert entry["n_local_spacegroup_classes"] == 2
        expert_dir = tail_dir / "experts" / family
        assert (expert_dir / "tail_params.msgpack").exists()
        assert (expert_dir / "loss_history.csv").exists()
        assert (expert_dir / "local_spacegroup_classes.yaml").exists()

    predictions = np.load(tail_dir / "tail_predictions.npz")
    n_total = 20  # 2 crystal systems x limit_per_system=10
    assert predictions["family_probs"].shape == (n_total, 2)
    assert predictions["spacegroup_probs"].shape[0] == n_total
    assert (
        predictions["spacegroup_probs_oracle"].shape
        == predictions["spacegroup_probs"].shape
    )
    np.testing.assert_allclose(
        predictions["family_probs"].sum(axis=1), np.ones(n_total), atol=1e-4
    )
    np.testing.assert_allclose(
        predictions["spacegroup_probs"].sum(axis=1), np.ones(n_total), atol=1e-4
    )
    np.testing.assert_allclose(
        predictions["spacegroup_probs_oracle"].sum(axis=1), np.ones(n_total), atol=1e-4
    )
    assert predictions["labels"].shape[0] == n_total
    assert predictions["spacegroups"].shape[0] == n_total

    # Same plot suite as "classification" (family + spacegroup, train + val).
    for level in ("family", "spacegroup"):
        for split in ("train", "val"):
            for kind in ("confusion_matrix", "classification_report", "calibration"):
                assert (tail_dir / f"{kind}_{level}_{split}.png").exists()

    with open(tail_dir / "run.log") as f:
        run_log = f.read()
    assert "Training tail_kind=hierarchical" in run_log
    assert "Tail artifacts saved to" in run_log


def test_train_tail_hierarchical_falls_back_for_low_sample_families(tmp_path):
    run_dir = _train_supcon_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="hierarchical",
        # Both families have well under 100 training rows -- forces every
        # family into the majority-value fallback, no expert trained.
        hierarchical=HierarchicalTailConfig(
            head_hidden_dim=8, min_samples_per_expert=100
        ),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )

    tail_dir = train_tail(config)

    with open(tail_dir / "family_expert_status.yaml") as f:
        status = yaml.safe_load(f)["families"]
    assert len(status) == 2
    for entry in status:
        assert entry["expert"] is False
        assert isinstance(entry["fallback_spacegroup"], int)
    assert not (tail_dir / "experts").exists()

    predictions = np.load(tail_dir / "tail_predictions.npz")
    # Every fallback prediction is a one-hot distribution over the fallback
    # spacegroup -- still a valid probability distribution.
    np.testing.assert_allclose(
        predictions["spacegroup_probs"].sum(axis=1),
        np.ones(predictions["spacegroup_probs"].shape[0]),
        atol=1e-4,
    )


@pytest.mark.parametrize("tail_kind", ["classification", "hierarchical"])
def test_train_tail_rejects_classification_and_hierarchical_for_vae_run(
    tmp_path, tail_kind
):
    # classification/hierarchical are redundant with vae/autoencoder's own
    # aux_heads classification, so they stay restricted to supcon/cgcnn/mace
    # -- unlike visualization, see test_train_tail_visualization_accepts_*
    # below.
    run_dir = _train_a_vae_run(tmp_path)
    kwargs = (
        {"classification": ClassificationTailConfig(mode="family_only")}
        if tail_kind == "classification"
        else {"hierarchical": HierarchicalTailConfig()}
    )
    config = TailTrainConfig(run_dir=str(run_dir), tail_kind=tail_kind, **kwargs)

    with pytest.raises(
        ValueError, match="requires a completed run whose model_kind is one of"
    ):
        train_tail(config)


@pytest.mark.parametrize("run_builder", [_train_a_vae_run, _train_an_autoencoder_run])
def test_train_tail_visualization_accepts_vae_and_autoencoder_runs(
    tmp_path, run_builder
):
    # Unlike classification/hierarchical, a visualization tail has no
    # built-in vae/autoencoder equivalent (aux_heads only ever produces a
    # classifier), so it's offered for every model_kind.
    run_dir = run_builder(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="visualization",
        visualization=VisualizationTailConfig(viz_dim=2, mode="family_only"),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )

    tail_dir = train_tail(config)

    assert (tail_dir / "tail_embeddings.npz").exists()
    assert (tail_dir / "viz_plot_family.png").exists()
    embeddings = np.load(tail_dir / "tail_embeddings.npz")
    assert embeddings["embeddings"].shape == (8, 2)


def test_train_tail_reruns_dedup_output_dir(tmp_path):
    run_dir = _train_supcon_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="classification",
        classification=ClassificationTailConfig(mode="family_only"),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )

    tail_dir_1 = train_tail(config)
    tail_dir_2 = train_tail(config)

    assert tail_dir_1 != tail_dir_2
    assert tail_dir_1.exists() and tail_dir_2.exists()

    # Each run gets its own run.log, and the shared "dim_red.pipeline"
    # logger's per-call FileHandler must be detached afterwards -- otherwise
    # the second run's log lines would also leak into the first run's file.
    with open(tail_dir_1 / "run.log") as f:
        log_1 = f.read()
    with open(tail_dir_2 / "run.log") as f:
        log_2 = f.read()
    assert str(tail_dir_2) not in log_1
    assert "Tail artifacts saved to" in log_1
    assert "Tail artifacts saved to" in log_2


def test_train_tail_never_recomputes_soap_for_training_points(tmp_path):
    """train_tail must reuse embeddings.npz['embeddings'] directly -- it
    should never call dim_red.pipeline.inference.compute_soap or
    encode_structures at all (train_tail uses the lightweight
    load_run_embeddings, not load_trained_run, so it never reconstructs the
    body or touches SOAP in any way -- any call here is a regression).
    """
    run_dir = _train_supcon_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="classification",
        classification=ClassificationTailConfig(mode="family_only"),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )

    with (
        patch(
            "dim_red.pipeline.inference.compute_soap",
            side_effect=AssertionError("train_tail must not recompute SOAP"),
        ),
        patch("dim_red.pipeline.inference.encode_structures") as mock_encode,
    ):
        train_tail(config)

    mock_encode.assert_not_called()


def test_train_tail_does_not_require_dataset_or_model_params(tmp_path):
    """train_tail only needs config.yaml/embeddings.npz -- deleting
    dataset.extxyz and model_params.msgpack (the two artifacts only
    load_trained_run's full model-reconstruction path needs) must not break
    it.
    """
    run_dir = _train_supcon_run(tmp_path)
    (run_dir / "dataset.extxyz").unlink()
    (run_dir / "model_params.msgpack").unlink()
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="classification",
        classification=ClassificationTailConfig(mode="family_only"),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )

    tail_dir = train_tail(config)

    assert (tail_dir / "tail_predictions.npz").exists()


def test_train_tail_optimizer_velo_still_works_end_to_end(tmp_path):
    """ "velo" is opt-in now (default is "adam") -- confirm it still works
    end-to-end through train_tail, not just the lower-level
    train_classification_tail/train_visualization_tail functions.
    """
    run_dir = _train_supcon_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="classification",
        classification=ClassificationTailConfig(mode="family_only"),
        train=TailTrainSettings(epochs=1, batch_size=4, optimizer="velo"),
    )

    tail_dir = train_tail(config)

    assert (tail_dir / "tail_params.msgpack").exists()
    with open(tail_dir / "run.log") as f:
        run_log = f.read()
    assert "optimizer=velo" in run_log


# --- model_kind == "cgcnn" ---------------------------------------------------


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


def _train_a_cgcnn_run(tmp_path):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=10),
        soap=SoapConfig(),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=3),
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
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="cgcnn",
    )
    spacegroup_offsets = {"cubic": 195, "hexagonal": 168}
    with patch(
        "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
        side_effect=lambda crystal_system, api_key=None, limit=10: [
            _fake_cgcnn_atoms(
                "Cu",
                f"mp-{crystal_system}-{i}",
                spacegroup_offsets[crystal_system] + (i % 2),
            )
            for i in range(limit)
        ],
    ):
        run_dir = run_single(config)
    return run_dir


def test_train_tail_accepts_cgcnn_run(tmp_path):
    run_dir = _train_a_cgcnn_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="visualization",
        visualization=VisualizationTailConfig(viz_dim=2, mode="family_only"),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )

    tail_dir = train_tail(config)

    assert (tail_dir / "tail_embeddings.npz").exists()
    assert (tail_dir / "tail_params.msgpack").exists()


class _FakeMaceEncoder:
    """Deterministic stand-in for dim_red.mace.model.MaceEncoder -- see
    test_pipeline_single_run.py's identical fixture."""

    _DIM = 3

    def __init__(self, **kwargs):
        self.params = {"dummy": np.zeros(1, dtype=np.float32)}

    def encode(self, atoms_list):
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(atoms_list), self._DIM)).astype(np.float32)


def _train_a_mace_run(tmp_path):
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=["cubic", "hexagonal"], limit_per_system=10),
        soap=SoapConfig(),
        vae=VAEArchConfig(encoder_hidden_dim=[1], latent_dim=1),
        train=TrainSettings(device="cpu"),
        mace=MaceConfig(checkpoint_path="/fake/ckpt", r_max=5.0),
        seed=0,
        output_dir=str(tmp_path / "runs"),
        model_kind="mace",
    )
    spacegroup_offsets = {"cubic": 195, "hexagonal": 168}
    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            side_effect=lambda crystal_system, api_key=None, limit=10: [
                _fake_cgcnn_atoms(
                    "Cu",
                    f"mp-{crystal_system}-{i}",
                    spacegroup_offsets[crystal_system] + (i % 2),
                )
                for i in range(limit)
            ],
        ),
        patch("dim_red.mace.model.MaceEncoder", _FakeMaceEncoder),
    ):
        run_dir = run_single(config)
    return run_dir


def test_train_tail_accepts_mace_run(tmp_path):
    run_dir = _train_a_mace_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="classification",
        classification=ClassificationTailConfig(mode="family_only"),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )

    tail_dir = train_tail(config)

    assert (tail_dir / "tail_predictions.npz").exists()
    assert (tail_dir / "tail_params.msgpack").exists()
