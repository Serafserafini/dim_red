"""
Fine-tune only the visualization tail's own ``hidden_dim`` (width/depth) for
chosen families of an already-trained ``hierarchical_supcon`` tail, leaving
everything else frozen.

For each requested family it loads that family's already-trained SupCon SG
body (``tails/<subdir>/sg_experts/<family>/sg_body_params.msgpack``), encodes
the family's rows from the run's own ``embeddings.npz["features"]`` (no SOAP
recompute), and trains one fresh ``VisualizationTail`` per candidate
``hidden_dim`` with ``dim_red.supcon.tail_training.train_visualization_tail``
-- the same routine ``hierarchical_supcon`` uses -- instead of retraining
every family's SG body/classifier/visualizer together. Architecture settings
(encoder widths, latent dim, tau/distance of the visualizer) are read from
the tail's own ``tail_config.yaml``.

Each candidate is saved to ``<out_dir>/<family>/<tag>/`` (``tag`` = hidden dims
joined by ``_``): ``visualization_tail_params.msgpack``,
``visualization_embeddings.npz`` and a plot. ``<out_dir>/result.json`` collects
the embedding-quality metrics (merged with any previous invocation's results),
for you to read and judge -- nothing is selected automatically. To use a
winning width in future training, set it under
``hierarchical_supcon.sg_visualization_hidden_dim_by_family`` (see
``configs/tail_train_hierarchical_supcon_best_combo.example.yaml``).

Example:
    python examples/tune_sg_visualization_hidden_dims.py \\
        --run-dir experiments/runs/<run> \\
        --subdir hierarchical_supcon_best_combo \\
        --out-dir runs/viz_tune \\
        --families Orthorhombic Tetragonal \\
        --candidates 64,32 128,64 64,32,16 128,64,32
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from flax import serialization

from dim_red.analysis.metrics import embedding_quality_metrics
from dim_red.analysis.plotting import plot_reduced_space
from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.tail_training import TailTrainConfig, train_visualization_tail
from dim_red.supcon.tails import VisualizationTail


def _parse_candidate(text: str):
    return [int(x) for x in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--subdir", required=True, help="tails/<subdir> of the hierarchical_supcon tail"
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument(
        "--candidates",
        nargs="+",
        type=_parse_candidate,
        default=[[64, 32], [128, 64], [64, 32, 16], [128, 64, 32]],
        help="hidden_dim candidates, comma-separated widths (e.g. 64,32,16)",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="gpu")
    args = parser.parse_args()

    tail_dir = args.run_dir / "tails" / args.subdir
    sg_dir = tail_dir / "sg_experts"
    with open(tail_dir / "tail_config.yaml") as f:
        tail_cfg = yaml.safe_load(f)
    hs = tail_cfg["hierarchical_supcon"]
    seed = tail_cfg.get("seed", 42)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.run_dir / "embeddings.npz", allow_pickle=True)
    features = d["features"].astype(np.float32)
    labels = d["labels"]
    split = d["split"]
    material_ids = d["material_ids"]
    spacegroups_all = d["spacegroups"]

    results = {}
    for family in args.families:
        family_mask = labels == family
        family_positions = np.flatnonzero(family_mask)
        pos_lookup = {row: i for i, row in enumerate(family_positions)}
        train_pos = np.array(
            [pos_lookup[i] for i in np.flatnonzero(family_mask & (split == "train"))]
        )
        val_pos = np.array(
            [pos_lookup[i] for i in np.flatnonzero(family_mask & (split == "val"))]
        )

        with open(sg_dir / family / "local_spacegroup_classes.yaml") as f:
            local_classes = yaml.safe_load(f)["local_spacegroup_classes"]
        class_to_id = {c: i for i, c in enumerate(local_classes)}
        local_spacegroups = spacegroups_all[family_mask]
        local_sg_ids = np.array([class_to_id[sg] for sg in local_spacegroups.tolist()])

        body = SupConEncoder(
            input_dim=features.shape[1],
            encoder_hidden_dim=hs["sg_encoder_hidden_dim"],
            latent_dim=hs["sg_latent_dim"],
            seed=seed,
        )
        with open(sg_dir / family / "sg_body_params.msgpack", "rb") as f:
            body.params = serialization.msgpack_restore(f.read())
        r_family = np.asarray(body.encode(features[family_mask]))

        print(
            f"=== {family}: {len(family_positions)} points, {len(local_classes)} local spacegroups ==="
        )
        family_results = {}
        for hidden_dim in args.candidates:
            tag = "_".join(str(w) for w in hidden_dim)
            viz_tail = VisualizationTail(
                input_dim=hs["sg_latent_dim"],
                hidden_dim=hidden_dim,
                output_dim=2,
                seed=seed,
            )
            config = TailTrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=0.001,
                tau=hs.get("sg_visualization_tau", 0.1),
                distance=hs.get("sg_visualization_distance", "euclidean"),
                seed=seed,
                device=args.device,
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
                lambda_norm=hs.get("sg_visualization_lambda_norm", 0.0),
            )
            z_family = np.asarray(viz_tail.project(r_family))
            metrics = embedding_quality_metrics(z_family, {"sg": local_spacegroups})
            print(
                f"  hidden_dim={hidden_dim}: {len(history['train_loss'])}/{args.epochs} epochs, "
                f"knn_acc={metrics['sg_knn_accuracy']:.4f} silhouette={metrics['sg_silhouette']:.4f}"
            )
            family_results[tag] = {"hidden_dim": hidden_dim, "metrics": metrics}

            fam_out = args.out_dir / family / tag
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

    result_path = args.out_dir / "result.json"
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
