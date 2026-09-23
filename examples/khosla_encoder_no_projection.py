"""
Experimental branch (experiment/khosla-dims-sweep): "sweep D" -- trains the
SupCon body with the Supervised Contrastive loss computed directly on its
own representation r, with NO separate ProjectionTail at all (see
dim_red.supcon.training.train_supcon_no_projection's own docstring).
Tests whether round 7's 128-dim throwaway projection tail is actually
pulling its weight versus just letting the encoder's own output absorb the
contrastive loss directly, across several encoder widths.

Same native-SOAP-feature reuse pattern as
examples/tail_replaces_projection_classifier.py (round 7 predates
embeddings.npz's cached "features" field, so it's recomputed via
_compute_native_soap_features), same architecture/training recipe as round
7 otherwise (encoder [128,64], tau=0.05/cosine, mode=family_only), plus a
family classification tail (light recipe, matching round 7's own) and a
family visualization tail (heavy recipe, euclidean, matching this
project's other round-15 configs) trained on top of the resulting frozen
body, exactly like round 7's own two-tail setup.

Run with (one latent_dim per invocation, for easy SLURM array
parallelism):
    python examples/khosla_encoder_no_projection.py \
        runs/tuning_supcon_family_only_round7/20260914-1/model-supcon_hd-128-64_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp1_supcon-family_only_tau0.05_lf1-3 \
        runs/experiment_khosla_no_projection_dims_8 \
        8
"""

import json
import sys
from pathlib import Path

import numpy as np
from flax import serialization

from dim_red.analysis.metrics import embedding_quality_metrics
from dim_red.analysis.plotting import plot_reduced_space
from dim_red.pipeline.inference import load_run_embeddings
from dim_red.pipeline.tail_training import _compute_native_soap_features
from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.tail_training import TailTrainConfig as SupConTailTrainConfig
from dim_red.supcon.tail_training import (
    train_classification_tail,
    train_visualization_tail,
)
from dim_red.supcon.tails import ClassificationTail, VisualizationTail
from dim_red.supcon.training import TrainConfig, train_supcon_no_projection
from dim_red.vae.database import VAEDatabase


def main(reference_run_dir: str, output_dir: str, latent_dim: int) -> None:
    ref_dir = Path(reference_run_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_run_embeddings(str(ref_dir))
    features = _compute_native_soap_features(ref_dir, loaded.config, loaded.embeddings)
    d = loaded.embeddings
    labels = d["labels"]
    split = d["split"]
    train_idx = split == "train"
    val_idx = split == "val"

    family_classes = sorted(set(labels.tolist()))
    family_to_id = {c: i for i, c in enumerate(family_classes)}
    family_ids = np.array([family_to_id[label] for label in labels.tolist()])

    print(
        f"Loaded {features.shape[0]} rows ({int(train_idx.sum())} train / "
        f"{int(val_idx.sum())} val), input_dim={features.shape[1]}, "
        f"latent_dim={latent_dim}, {len(family_classes)} family classes: "
        f"{family_classes}"
    )

    # --- Body: no projection tail, loss directly on r -----------------------
    model = SupConEncoder(
        input_dim=features.shape[1],
        encoder_hidden_dim=[128, 64],
        latent_dim=latent_dim,
        seed=42,
    )
    body_config = TrainConfig(
        epochs=200,
        batch_size=128,
        learning_rate=0.001,
        optimizer="adam",
        tau=0.05,
        distance="cosine",
        seed=42,
        device="gpu",
        early_stopping=True,
        early_stopping_patience=20,
        early_stopping_min_delta=0.0001,
        early_stopping_restore_best=True,
    )
    train_db = VAEDatabase.from_array(features[train_idx])
    val_db = VAEDatabase.from_array(features[val_idx])
    body_history = train_supcon_no_projection(
        model,
        train_db,
        val_db,
        body_config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
        lambda_family=1.0,
    )
    n_body_epochs = len(body_history["train_loss"])
    print(
        f"Body trained {n_body_epochs}/{body_config.epochs} epochs, final "
        f"train_loss={body_history['train_loss'][-1]:.4f} "
        f"val_loss={body_history['val_loss'][-1]:.4f}"
    )
    with open(out_dir / "body_params.msgpack", "wb") as f:
        f.write(serialization.to_bytes(model.params))
    with open(out_dir / "body_loss_history.csv", "w") as f:
        keys = list(body_history.keys())
        f.write("epoch," + ",".join(keys) + "\n")
        for i in range(n_body_epochs):
            f.write(f"{i}," + ",".join(str(body_history[k][i]) for k in keys) + "\n")

    r_all = np.asarray(model.encode(features))
    r_train, r_val = r_all[train_idx], r_all[val_idx]

    # --- Family classification tail (light recipe, matching round 7) --------
    clf_tail = ClassificationTail(
        input_dim=latent_dim,
        hidden_dim=16,
        n_family_classes=len(family_classes),
        seed=42,
    )
    clf_config = SupConTailTrainConfig(
        epochs=40,
        batch_size=32,
        learning_rate=0.001,
        optimizer="adam",
        seed=42,
        device="gpu",
        early_stopping=True,
        early_stopping_patience=10,
        early_stopping_min_delta=0.0001,
        early_stopping_restore_best=True,
    )
    train_classification_tail(
        clf_tail,
        r_train,
        r_val,
        clf_config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
        lambda_family=1.0,
    )
    val_pred = np.asarray(clf_tail.classify_family(r_val)).argmax(axis=1)
    val_accuracy = float((val_pred == family_ids[val_idx]).mean())
    print(f"Family classification validation accuracy: {val_accuracy:.4f}")
    with open(out_dir / "classification_tail_params.msgpack", "wb") as f:
        f.write(serialization.to_bytes(clf_tail.params))

    # --- Family visualization tail (heavy recipe, euclidean) ----------------
    viz_tail = VisualizationTail(
        input_dim=latent_dim, hidden_dim=[32, 16], output_dim=2, seed=42
    )
    viz_config = SupConTailTrainConfig(
        epochs=200,
        batch_size=128,
        learning_rate=0.001,
        optimizer="adam",
        seed=42,
        device="gpu",
        early_stopping=True,
        early_stopping_patience=20,
        early_stopping_min_delta=0.0001,
        early_stopping_restore_best=True,
    )
    train_visualization_tail(
        viz_tail,
        r_train,
        r_val,
        viz_config,
        train_family_ids=family_ids[train_idx],
        val_family_ids=family_ids[val_idx],
        lambda_family=1.0,
        lambda_norm=0.0,
    )
    z_all = np.asarray(viz_tail.project(r_all))
    viz_metrics = embedding_quality_metrics(z_all, {"family": labels})
    print(f"Family visualization quality: {viz_metrics}")
    with open(out_dir / "visualization_tail_params.msgpack", "wb") as f:
        f.write(serialization.to_bytes(viz_tail.params))
    np.savez(
        out_dir / "visualization_embeddings.npz",
        embeddings=z_all,
        labels=labels,
        material_ids=d["material_ids"],
        split=split,
    )
    plot_reduced_space(
        z_all,
        labels.tolist(),
        title=f"No-projection body, latent_dim={latent_dim}",
        save_path=str(out_dir / "viz_plot_family.png"),
    )

    with open(out_dir / "result.json", "w") as f:
        json.dump(
            {
                "reference_run_dir": str(ref_dir),
                "latent_dim": latent_dim,
                "family_classes": family_classes,
                "body_epochs_run": n_body_epochs,
                "family_val_accuracy": val_accuracy,
                "visualization_quality": viz_metrics,
            },
            f,
            indent=2,
        )
    print(f"Artifacts saved to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python khosla_encoder_no_projection.py "
            "<reference_run_dir> <output_dir> <latent_dim>"
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
