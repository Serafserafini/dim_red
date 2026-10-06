"""Regenerate the golden references. Run ONCE, against the pre-FullStack code:

    PYTHONPATH=src:. python tests/golden/make_golden.py

Runs the OLD pipeline (``run_single`` + ``train_tail`` classification,
visualization and hierarchical_supcon) on the synthetic dataset of
``golden_spec.make_inputs`` -- dataset builder and SOAP patched so the inputs
are exactly that dataset -- and copies the resulting parameters, loss
histories and embeddings into ``tests/golden/data/``."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.io import write

from dim_red.pipeline.config import (
    ClassificationTailConfig,
    EncoderConfig,
    HierarchicalSupconTailConfig,
    PyxtalConfig,
    RunConfig,
    SoapConfig,
    SupConConfig,
    TailTrainConfig,
    TailTrainSettings,
    TrainSettings,
    VisualizationTailConfig,
)
from dim_red.pipeline.single_run import run_single
from dim_red.pipeline.tail_training import train_tail
from tests.golden import golden_spec as g


def main() -> None:
    X, labels, material_ids, spacegroups = g.make_inputs()
    work = Path(tempfile.mkdtemp(prefix="golden_"))

    structures_path = work / "structures.extxyz"
    atoms = []
    for material_id, sg in zip(material_ids, spacegroups):
        a = Atoms("Cu", positions=[[0.0, 0.0, 0.0]])
        a.info["material_id"] = material_id
        a.info["spacegroup"] = sg
        atoms.append(a)
    write(str(structures_path), atoms, format="extxyz")

    n_features = X.shape[1]
    fake_dataset = (
        X,
        labels,
        material_ids,
        spacegroups,
        structures_path,
        np.zeros(n_features, dtype=np.float32),  # mean 0, std 1: the
        np.ones(n_features, dtype=np.float32),  # standardized features are X
    )

    run_config = RunConfig(
        data_source="pyxtal",
        soap=SoapConfig(),
        pyxtal=PyxtalConfig(structures_per_spacegroup=1),
        encoder=EncoderConfig(encoder_hidden_dim=g.ENCODER_HIDDEN, latent_dim=g.LATENT),
        train=TrainSettings(
            epochs=g.BODY_EPOCHS, batch_size=g.BATCH, val_ratio=g.VAL_RATIO
        ),
        supcon=SupConConfig(
            mode="family_only",
            tau=g.TAU,
            distance=g.DISTANCE,
            projection_dim=g.PROJ_DIM,
        ),
        seed=g.SEED,
        output_dir=str(work / "runs"),
        model_kind="supcon",
    )
    head_train = TailTrainSettings(epochs=g.HEAD_EPOCHS, batch_size=g.BATCH)

    with (
        patch(
            "dim_red.pipeline.single_run.build_dataset_for_run",
            return_value=fake_dataset,
        ),
        patch("dim_red.pipeline.tail_training.compute_soap", return_value=X),
    ):
        run_dir = run_single(run_config)
        clf_dir = train_tail(
            TailTrainConfig(
                run_dir=str(run_dir),
                tail_kind="classification",
                classification=ClassificationTailConfig(head_hidden_dim=g.CLF_HIDDEN),
                train=head_train,
                seed=g.SEED,
            )
        )
        viz_dir = train_tail(
            TailTrainConfig(
                run_dir=str(run_dir),
                tail_kind="visualization",
                visualization=VisualizationTailConfig(
                    viz_dim=2,
                    mode="family_only",
                    tau=g.VIZ_TAU,
                    distance=g.VIZ_DISTANCE,
                    hidden_dim=g.VIZ_HIDDEN,
                ),
                train=head_train,
                seed=g.SEED,
            )
        )
        hier_dir = train_tail(
            TailTrainConfig(
                run_dir=str(run_dir),
                tail_kind="hierarchical_supcon",
                hierarchical_supcon=HierarchicalSupconTailConfig(
                    head_hidden_dim=g.CLF_HIDDEN,
                    min_samples_per_expert=g.MIN_SAMPLES_PER_EXPERT,
                    sg_encoder_hidden_dim=g.ENCODER_HIDDEN,
                    sg_latent_dim=g.LATENT,
                    sg_tau=g.TAU,
                    sg_distance=g.DISTANCE,
                    sg_projection_dim=g.PROJ_DIM,
                    sg_classifier_hidden_dim=g.EXPERT_CLF_HIDDEN,
                    sg_visualization_hidden_dim=g.EXPERT_VIZ_HIDDEN,
                    sg_visualization_hidden_dim_by_family={},
                    sg_visualization_tau=g.VIZ_TAU,
                    sg_visualization_distance=g.VIZ_DISTANCE,
                ),
                train=head_train,
                seed=g.SEED,
            )
        )

    embeddings = np.load(run_dir / "embeddings.npz", allow_pickle=True)
    split = embeddings["split"]
    labels_arr = np.asarray(labels)
    for family in g.FAMILIES:
        in_family = labels_arr == family
        assert (in_family & (split == "val")).any(), f"{family} has no val rows"
        assert (in_family & (split == "train")).any(), f"{family} has no train rows"

    out = g.GOLDEN_DIR
    out.mkdir(parents=True, exist_ok=True)
    copies = {
        run_dir / "model_params.msgpack": "family_body_params.msgpack",
        run_dir / "projection_params.msgpack": "family_projection_params.msgpack",
        run_dir / "loss_history.csv": "family_body_loss_history.csv",
        run_dir / "embeddings.npz": "family_embeddings.npz",
        clf_dir / "tail_params.msgpack": "family_classifier_params.msgpack",
        clf_dir / "loss_history.csv": "family_classifier_loss_history.csv",
        viz_dir / "tail_params.msgpack": "family_viz_params.msgpack",
        viz_dir / "loss_history.csv": "family_viz_loss_history.csv",
    }
    for family in g.FAMILIES:
        sg_dir = hier_dir / "sg_experts" / family
        for src, dst in (
            ("sg_body_params.msgpack", "body_params.msgpack"),
            ("sg_projection_params.msgpack", "projection_params.msgpack"),
            ("classifier_tail_params.msgpack", "classifier_params.msgpack"),
            ("visualization_tail_params.msgpack", "viz_params.msgpack"),
            ("sg_body_loss_history.csv", "body_loss_history.csv"),
            ("classifier_loss_history.csv", "classifier_loss_history.csv"),
            ("visualization_loss_history.csv", "viz_loss_history.csv"),
            ("local_spacegroup_classes.yaml", "local_classes.yaml"),
        ):
            copies[sg_dir / src] = f"expert_{family}_{dst}"
    for src, dst in copies.items():
        shutil.copy(src, out / dst)

    np.savez(
        out / "inputs.npz",
        X=X,
        labels=np.asarray(labels),
        material_ids=np.asarray(material_ids),
        spacegroups=np.asarray(spacegroups, dtype=np.int64),
    )
    print(f"Golden references written to {out}")


if __name__ == "__main__":
    main()
