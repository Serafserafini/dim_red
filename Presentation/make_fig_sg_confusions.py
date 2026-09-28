"""Talk figures (slide 12): space-group confusion matrices, Cubic and Monoclinic.

Same data as experiments/plots/round17/confusion_{cubic,monoclinic}_best.png
(round-17 notes, experiments/round17_cubic_root_cause_notes.md): the
best_combo run's hierarchical_supcon tail, per-family SG experts with ORACLE
family routing (``spacegroup_probs_oracle`` in tail_predictions.npz), on ALL
points of the family (train + val) -- this reproduces the reported
accuracies 0.497 (Cubic, 2500 pts, 36 SGs) and 0.998 (Monoclinic, 2500 pts,
13 SGs). Val-only numbers are saved alongside for reference.

Rendered for a 185 x 185 px slot (1.85 in at 100 px/in): row-normalised,
white -> TU blue #006699, fixed 0-1 scale, no text in cells, no colorbar,
no title, no tick labels.

Writes Presentation/fig_sg_confusion_{cubic,monoclinic}.png and
Presentation/fig_sg_confusions_metrics.json.

Usage (from the repo root, conda env ``dmred``):
    python Presentation/make_fig_sg_confusions.py [--no-axis-labels]
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "Presentation"
RUN_DIR = Path(
    glob.glob(
        str(
            REPO / "experiments/runs/experiment_pipeline_best_combo/"
            "model-supcon_hd-256-128_*_supcon-family_only_tau0.05_lf1"
        )
    )[0]
)
PRED = RUN_DIR / "tails/hierarchical_supcon_best_combo/tail_predictions.npz"
FAMILIES = ("Cubic", "Monoclinic")
# Cell grid: light lines for the 13x13 matrix, very thin for 36x36.
GRID_LW = {"Cubic": 0.12, "Monoclinic": 0.35}
SIZE = 1.85  # inches (185 px slot)


def confusion(p, family, split=None):
    mask = p["labels"] == family
    if split is not None:
        mask &= p["split"] == split
    true = p["spacegroups"][mask]
    pred = p["spacegroup_classes"][p["spacegroup_probs_oracle"][mask].argmax(1)]
    classes = np.unique(true)
    idx = {int(c): i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), int)
    n_outside = 0
    for t, q in zip(true, pred):
        if int(q) in idx:
            cm[idx[int(t)], idx[int(q)]] += 1
        else:
            n_outside += 1
    acc = float(np.mean(true == pred))
    return cm, classes, acc, int(mask.sum()), n_outside


def draw(cm, family, axis_labels, out_stem):
    k = len(cm)
    frac = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("white_tublue", ["#FFFFFF", "#006699"])
    fig = plt.figure(figsize=(SIZE, SIZE), facecolor="white")
    pad = 0.02
    lab = 0.13 if axis_labels else 0.0  # inches reserved for the axis label
    side = SIZE - 2 * pad - lab
    ax = fig.add_axes(
        [(pad + lab) / SIZE, (pad + lab) / SIZE, side / SIZE, side / SIZE]
    )
    ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, k), minor=True)
    ax.set_yticks(np.arange(-0.5, k), minor=True)
    ax.grid(which="minor", color="#DDDDDD", linewidth=GRID_LW[family])
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")
        spine.set_linewidth(0.4)
    if axis_labels:
        ax.set_xlabel("Predicted SG", fontsize=7, labelpad=1.5, color="#333333")
        ax.set_ylabel("True SG", fontsize=7, labelpad=1.5, color="#333333")
    for ext in ("png",):
        fig.savefig(f"{out_stem}.{ext}", dpi=600, facecolor="white", bbox_inches=None)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-axis-labels", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    p = np.load(PRED)
    metrics = {}
    for fam in FAMILIES:
        cm, classes, acc, n, n_out = confusion(p, fam)
        _, _, acc_val, n_val, _ = confusion(p, fam, "val")
        draw(
            cm,
            fam,
            not args.no_axis_labels,
            args.out_dir / f"fig_sg_confusion_{fam.lower()}",
        )
        metrics[fam] = {
            "source": str(PRED.relative_to(REPO)),
            "predictions": "spacegroup_probs_oracle (true-family routing to the SG expert)",
            "split": "all (train+val)",
            "n_points": n,
            "n_spacegroups": int(len(classes)),
            "spacegroup_range": [int(classes.min()), int(classes.max())],
            "accuracy": acc,
            "accuracy_val_only": acc_val,
            "n_val": n_val,
            "n_predictions_outside_family": n_out,
            "confusion_counts": {"order": classes.tolist(), "matrix": cm.tolist()},
        }
    if args.out_dir == OUT_DIR:
        (OUT_DIR / "fig_sg_confusions_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
    print(
        json.dumps(
            {
                f: {k: v for k, v in m.items() if k != "confusion_counts"}
                for f, m in metrics.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
