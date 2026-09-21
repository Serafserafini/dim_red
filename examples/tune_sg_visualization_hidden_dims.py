"""
Round 15/16 follow-up: tune only the visualization tail's own hidden_dim
(width and depth) per family, keeping everything else frozen -- the
SupCon SG body used is the one from "best combo"
(configs/single_run_supcon_best_combo.example.yaml's
hierarchical_supcon sibling -- encoder [256,128], latent_dim 32,
projection_dim 128), already trained and never retrained here. No new
SOAP recompute either -- reuses that run's own cached
embeddings.npz["features"].

This deliberately does NOT go through dim_red.pipeline.tail_training's
hierarchical_supcon machinery (which retrains the SG body + classifier +
visualizer together, for all 7 families at once) -- only a fresh
VisualizationTail per hidden_dim candidate, on the frozen family-specific
SG embedding, exactly like dim_red.supcon.tail_training.
train_visualization_tail already does for any other frozen-body
visualization tail.

Round 16 ran this for Orthorhombic/Tetragonal (the two most visually
confused families per the round 15 cross-run analysis in
experiments/round15_dims_notes.md). Cubic is deliberately excluded from
every round of this sweep -- confirmed repeatedly (rounds 13-15) to be
structurally insensitive to capacity/dimension changes of any kind, not
just the SG body's own.

Run with (defaults to every family except Cubic and the two already
covered by round 16; pass explicit family names to target others):
    python examples/tune_sg_visualization_hidden_dims.py
    python examples/tune_sg_visualization_hidden_dims.py Hexagonal Monoclinic
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from flax import serialization

from dim_red.analysis.metrics import embedding_quality_metrics
from dim_red.analysis.plotting import plot_reduced_space
from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.tail_training import TailTrainConfig as SupConTailTrainConfig
from dim_red.supcon.tail_training import train_visualization_tail
from dim_red.supcon.tails import VisualizationTail

RUN_DIR = Path(
    "runs/experiment_pipeline_best_combo/"
    "model-supcon_hd-256-128_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp1_supcon-family_only_tau0.05_lf1"
)
SG_DIR = RUN_DIR / "tails" / "hierarchical_supcon_best_combo" / "sg_experts"
OUT_DIR = Path("runs/experiment_viz_tune_orthorhombic_tetragonal")

DEFAULT_FAMILIES = ["Hexagonal", "Monoclinic", "Triclinic", "Trigonal"]
CANDIDATES = {
    "64_32": [64, 32],  # baseline -- already trained as part of best_combo itself
    "128_64": [128, 64],
    "64_32_16": [64, 32, 16],
    "128_64_32": [128, 64, 32],
}


def main():
    families = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_FAMILIES
    if "Cubic" in families:
        raise ValueError(
            "Cubic is deliberately excluded from this sweep -- see the module "
            "docstring (rounds 13-15 already established it doesn't respond "
            "to capacity/dimension changes)."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(RUN_DIR / "embeddings.npz", allow_pickle=True)
    features = d["features"].astype(np.float32)
    labels = d["labels"]
    split = d["split"]
    material_ids = d["material_ids"]
    spacegroups_all = d["spacegroups"]

    results = {}
    for family in families:
        family_mask = labels == family
        family_positions = np.flatnonzero(family_mask)
        train_mask = family_mask & (split == "train")
        val_mask = family_mask & (split == "val")
        pos_lookup = {row: i for i, row in enumerate(family_positions)}
        train_pos = np.array([pos_lookup[i] for i in np.flatnonzero(train_mask)])
        val_pos = np.array([pos_lookup[i] for i in np.flatnonzero(val_mask)])

        with open(SG_DIR / family / "local_spacegroup_classes.yaml") as f:
            local_classes = yaml.safe_load(f)["local_spacegroup_classes"]
        local_class_to_id = {c: i for i, c in enumerate(local_classes)}
        local_spacegroups = spacegroups_all[family_mask]
        local_sg_ids = np.array(
            [local_class_to_id[sg] for sg in local_spacegroups.tolist()]
        )

        body = SupConEncoder(
            input_dim=features.shape[1],
            encoder_hidden_dim=[256, 128],
            latent_dim=32,
            seed=42,
        )
        with open(SG_DIR / family / "sg_body_params.msgpack", "rb") as f:
            body.params = serialization.msgpack_restore(f.read())
        r_family = np.asarray(body.encode(features[family_mask]))

        print(
            f"=== {family}: {len(family_positions)} points, "
            f"{len(local_classes)} local spacegroups ==="
        )

        family_results = {}
        for tag, hidden_dim in CANDIDATES.items():
            viz_tail = VisualizationTail(
                input_dim=32, hidden_dim=hidden_dim, output_dim=2, seed=42
            )
            config = SupConTailTrainConfig(
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
            history = train_visualization_tail(
                viz_tail,
                r_family[train_pos],
                r_family[val_pos],
                config,
                train_spacegroup_ids=local_sg_ids[train_pos],
                val_spacegroup_ids=local_sg_ids[val_pos],
                lambda_family=0.0,
                lambda_spacegroup=1.0,
                lambda_norm=0.0,
            )
            z_family = np.asarray(viz_tail.project(r_family))
            metrics = embedding_quality_metrics(z_family, {"sg": local_spacegroups})
            n_epochs = len(history["train_loss"])
            print(
                f"  hidden_dim={hidden_dim} ({tag}): {n_epochs}/200 epochs, "
                f"knn_acc={metrics['sg_knn_accuracy']:.4f} "
                f"silhouette={metrics['sg_silhouette']:.4f}"
            )
            family_results[tag] = {"hidden_dim": hidden_dim, "metrics": metrics}

            fam_out = OUT_DIR / family / tag
            fam_out.mkdir(parents=True, exist_ok=True)
            with open(fam_out / "visualization_tail_params.msgpack", "wb") as f:
                f.write(serialization.to_bytes(viz_tail.params))
            np.savez(
                fam_out / "visualization_embeddings.npz",
                embeddings=z_family,
                spacegroups=local_spacegroups,
                material_ids=material_ids[family_mask],
                split=split[family_mask],
            )
            plot_reduced_space(
                z_family,
                [str(sg) for sg in local_spacegroups.tolist()],
                title=f"{family} viz hidden_dim={hidden_dim}",
                save_path=str(fam_out / "viz_plot_spacegroup.png"),
            )

        results[family] = family_results

    # Merge with any results already saved by a previous invocation (e.g.
    # round 16's Orthorhombic/Tetragonal run) instead of overwriting them.
    result_path = OUT_DIR / "result.json"
    if result_path.exists():
        with open(result_path) as f:
            existing = json.load(f)
        existing.update(results)
        results = existing
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
