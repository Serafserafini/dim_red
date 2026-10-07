"""
Integration tests for dim_red.pipeline.tail_training: freeze an
already-trained ``model: cgcnn`` run's body and train a classification or
visualization tail on its saved representations. Builds a real run directory
via run_single (mocking the network fetch, same pattern as
test_pipeline_single_run.py) and then exercises train_tail against it.
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
    EncoderConfig,
    FetchConfig,
    GraphConfig,
    RunConfig,
    SoapConfig,
    TailTrainConfig,
    TailTrainSettings,
    TrainSettings,
    VisualizationTailConfig,
    run_config_to_dict,
)
from dim_red.pipeline.single_run import run_single
from dim_red.pipeline.tail_training import train_tail


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
        encoder=EncoderConfig(encoder_hidden_dim=[4], latent_dim=3),
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


def test_tail_kinds_and_model_kinds_are_cgcnn_only():
    from dim_red.pipeline import tail_training

    assert tail_training._TAIL_MODEL_KINDS == {
        "classification": ("cgcnn",),
        "visualization": ("cgcnn",),
    }


def test_train_tail_classification_writes_expected_artifacts(tmp_path):
    run_dir = _train_a_cgcnn_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="classification",
        classification=ClassificationTailConfig(),
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
        "train_ce",
        "val_ce",
    ]

    predictions = np.load(tail_dir / "tail_predictions.npz")
    n_total = 20  # 2 crystal systems x limit_per_system=10
    assert predictions["family_probs"].shape == (n_total, 2)  # 2 families
    np.testing.assert_allclose(
        predictions["family_probs"].sum(axis=1), np.ones(n_total), atol=1e-4
    )
    # tail_predictions.npz is self-sufficient: true labels are saved
    # alongside the predicted probabilities, no join against the parent
    # run's embeddings.npz needed to evaluate the classifier later.
    assert predictions["labels"].shape[0] == n_total
    # A family classifier has no spacegroup output.
    assert "spacegroup_probs" not in predictions.files

    # A confusion matrix, a per-class precision/recall/F1 bar chart and a
    # calibration diagram, on both splits (family level only).
    for split in ("train", "val"):
        for kind in ("confusion_matrix", "classification_report", "calibration"):
            assert (tail_dir / f"{kind}_family_{split}.png").exists()
            assert not (tail_dir / f"{kind}_spacegroup_{split}.png").exists()


def test_train_tail_visualization_2d_writes_expected_artifacts(tmp_path):
    run_dir = _train_a_cgcnn_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="visualization",
        visualization=VisualizationTailConfig(viz_dim=2, mode="family_and_spacegroup"),
        train=TailTrainSettings(epochs=2, batch_size=4),
    )

    tail_dir = train_tail(config)

    assert (tail_dir / "tail_embeddings.npz").exists()
    assert (tail_dir / "tail_params.msgpack").exists()
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
    run_dir = _train_a_cgcnn_run(tmp_path)
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
    run_dir = _train_a_cgcnn_run(tmp_path)
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


def test_train_tail_reruns_dedup_output_dir(tmp_path):
    run_dir = _train_a_cgcnn_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="classification",
        classification=ClassificationTailConfig(),
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


def test_train_tail_never_reconstructs_the_body(tmp_path):
    """train_tail must reuse embeddings.npz['embeddings'] directly -- it
    uses the lightweight load_run_embeddings, never load_trained_run or
    encode_structures (any call here is a regression).
    """
    run_dir = _train_a_cgcnn_run(tmp_path)
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="classification",
        classification=ClassificationTailConfig(),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )

    with (
        patch(
            "dim_red.pipeline.inference.load_trained_run",
            side_effect=AssertionError("train_tail must not reconstruct the body"),
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
    run_dir = _train_a_cgcnn_run(tmp_path)
    (run_dir / "dataset.extxyz").unlink()
    (run_dir / "model_params.msgpack").unlink()
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="classification",
        classification=ClassificationTailConfig(),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )

    tail_dir = train_tail(config)

    assert (tail_dir / "tail_predictions.npz").exists()


def test_train_tail_rejects_a_non_cgcnn_run(tmp_path):
    """A hand-built run directory claiming model_kind=supcon is refused
    before any tail directory is created."""
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
    np.savez(
        run_dir / "embeddings.npz",
        embeddings=np.zeros((4, 2), dtype=np.float32),
        labels=np.array(["Cubic"] * 4),
        material_ids=np.array([f"mp-{i}" for i in range(4)]),
        spacegroups=np.full(4, 195, dtype=np.int64),
        split=np.array(["train", "train", "train", "val"]),
    )
    config = TailTrainConfig(
        run_dir=str(run_dir),
        tail_kind="classification",
        classification=ClassificationTailConfig(),
        train=TailTrainSettings(epochs=1, batch_size=4),
    )

    # Refused while parsing the run's config.yaml (supcon configs use the
    # FullStack schema), before any tail directory is created.
    with pytest.raises(ValueError, match="FullStack"):
        train_tail(config)
    assert not (run_dir / "tails").exists()
