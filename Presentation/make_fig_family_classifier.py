"""Talk figures for slide 11 ("family classifier: clean separation"), from the
round-15 best_combo two-step model's family_only SupCon body:

  - fig_family_map.png: its 2D visualization tail
    (tails/visualization_family_only_euclidean), coloured by crystal family
    with the same 7 base hues as fig_single_stage_sg_map (all points).
  - fig_family_confusion.png: its family classification tail
    (tails/classification, family_only), validation split, row-normalised.

Also writes fig_family_classifier_metrics.json.

Usage (from the repo root, conda env ``dmred``):
    python Presentation/make_fig_family_classifier.py [RUN_DIR]
"""

import colorsys
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_fig_single_stage_sg import FAMILIES  # noqa: E402  (shared palette)

from dim_red.analysis.metrics import embedding_quality_metrics  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "Presentation"
VIZ_SUBDIR = "visualization_family_only_euclidean"
CLF_SUBDIR = "classification"

FULL_NAMES = [
    "Triclinic",
    "Monoclinic",
    "Orthorhombic",
    "Tetragonal",
    "Trigonal",
    "Hexagonal",
    "Cubic",
]
# Flat per-family colour: the family's base hue (shared with
# make_fig_single_stage_sg.py), lightness alternating dark/light between
# neighbouring families so the three adjacent blues stay distinguishable.
FAMILY_LIGHT = [0.50, 0.33, 0.56, 0.36, 0.55, 0.42, 0.45]
FAMILY_SAT = 0.62


def family_color(hue_deg: float, light: float):
    return colorsys.hls_to_rgb(hue_deg / 360.0, light, FAMILY_SAT)


def _find_run_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    hits = sorted(
        glob.glob(
            str(
                REPO / "experiments/runs/experiment_pipeline_best_combo/"
                "model-supcon_hd-256-128_*_supcon-family_only_tau0.05_lf1"
            )
        )
    )
    if not hits:
        raise SystemExit("round-15 best_combo run dir not found")
    return Path(hits[-1])


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _draw_key(fig, rect, colors) -> None:
    """Compact key below the map: 2 rows (4 + 3) of small solid squares +
    full family names (the "SG a-b" sublabels don't fit at slide size).
    Column x-positions are uneven so "Orthorhombic" gets room."""
    kax = fig.add_axes(rect)
    col_x = [0.0, 0.235, 0.49, 0.775]
    sq_w = 0.028  # axes-fraction; ~square given the key axes' aspect
    for fi, name in enumerate(FULL_NAMES):
        row, col = divmod(fi, 4)
        yc = 0.75 - 0.5 * row
        kax.add_patch(
            plt.Rectangle((col_x[col], yc - 0.19), sq_w, 0.38, color=colors[name], lw=0)
        )
        kax.text(
            col_x[col] + sq_w + 0.015,
            yc,
            name,
            fontsize=10.5,
            va="center",
            ha="left",
            color="#222222",
        )
    kax.set_xlim(0, 1)
    kax.set_ylim(0, 1)
    kax.axis("off")


def make_map(run_dir: Path, colors: dict) -> dict:
    viz = np.load(run_dir / "tails" / VIZ_SUBDIR / "tail_embeddings.npz")
    z, fams, sgs = viz["embeddings"], viz["labels"], viz["spacegroups"].astype(int)
    point_colors = np.array([colors[f] for f in fams])
    order = np.random.default_rng(0).permutation(len(z))

    # Slide slot: 472 x 390 px on a 1280 x 720 slide (100 px/in).
    fig = plt.figure(figsize=(4.72, 3.90), facecolor="white")
    ax = fig.add_axes([0.01, 0.16, 0.98, 0.83])
    ax.scatter(
        z[order, 0],
        z[order, 1],
        c=point_colors[order],
        s=MAP_S,
        alpha=0.62,
        edgecolors="black",
        linewidths=0.15,
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Horizontal stretch so the (tall) embedding fills the ~1.4:1 map area.
    ax.set_aspect(MAP_ASPECT, adjustable="datalim")
    _draw_key(fig, [0.04, 0.015, 0.95, 0.13], colors)
    for ext in ("png",):
        fig.savefig(
            OUT_DIR / f"fig_family_map.{ext}",
            dpi=600,
            facecolor="white",
            bbox_inches=None,
        )
    plt.close(fig)

    out = {}
    split = viz["split"]
    for name, mask in (("all", np.ones(len(z), bool)), ("val", split == "val")):
        m = embedding_quality_metrics(z[mask], {"family": fams[mask]})
        out[name] = {k: float(v) for k, v in m.items()}
        out[name]["n_points"] = int(mask.sum())
    del sgs
    return out


MAP_ASPECT = 0.55  # ~1.8x horizontal stretch
MAP_S = 9


def make_confusion(run_dir: Path) -> dict:
    p = np.load(run_dir / "tails" / CLF_SUBDIR / "tail_predictions.npz")
    val = p["split"] == "val"
    pred = p["family_classes"][p["family_probs"][val].argmax(1)]
    true = p["labels"][val]
    k = len(FULL_NAMES)
    cm = np.zeros((k, k), int)
    idx = {n: i for i, n in enumerate(FULL_NAMES)}
    for t, q in zip(true, pred):
        cm[idx[t], idx[q]] += 1
    pct = 100.0 * cm / cm.sum(1, keepdims=True)

    cmap = LinearSegmentedColormap.from_list("white_tublue", ["#FFFFFF", "#006699"])
    # Slide slot: 472 x 390 px (4.72 x 3.90 in at 100 px/in); square cells,
    # axes placed explicitly (no tight crop) so the output is exactly 472:390.
    fig = plt.figure(figsize=(4.72, 3.90), facecolor="white")
    h_in = 3.90 - 0.93 - 0.05
    ax = fig.add_axes([1.40 / 4.72, 0.93 / 3.90, h_in / 4.72, h_in / 3.90])
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
    for i in range(k):
        for j in range(k):
            if cm[i, j] == 0:
                continue
            v = pct[i, j]
            label = f"{v:.0f}" if v >= 0.5 else "<1"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=9.5,
                color="white" if v > 55 else ("#222222" if v >= 0.5 else "#999999"),
            )
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(
        FULL_NAMES, rotation=45, ha="right", rotation_mode="anchor", fontsize=10
    )
    ax.set_yticklabels(FULL_NAMES, fontsize=10)
    ax.set_xlabel("Predicted family", fontsize=11.5, labelpad=2)
    ax.set_ylabel("True family", fontsize=11.5, labelpad=4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, k), minor=True)
    ax.set_yticks(np.arange(-0.5, k), minor=True)
    ax.grid(which="minor", color="#DDDDDD", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    for ext in ("png",):
        fig.savefig(
            OUT_DIR / f"fig_family_confusion.{ext}",
            dpi=600,
            facecolor="white",
            bbox_inches=None,
        )
    plt.close(fig)

    return {
        "split": "val",
        "n_val": int(val.sum()),
        "accuracy": float(np.trace(cm) / cm.sum()),
        # Train+val together -- what pipeline.compare.classification_accuracies_from_npz
        # (and the round-15 notes' 0.969) report; NOT a held-out number.
        "accuracy_all_points_train_plus_val": float(
            np.mean(p["family_classes"][p["family_probs"].argmax(1)] == p["labels"])
        ),
        "per_family_recall": {
            n: float(cm[i, i] / cm[i].sum()) for i, n in enumerate(FULL_NAMES)
        },
        "confusion_counts": {"order": FULL_NAMES, "matrix": cm.tolist()},
    }


def main() -> None:
    _style()
    run_dir = _find_run_dir()
    colors = {
        n: family_color(h, L)
        for n, (_, _, _, h), L in zip(FULL_NAMES, FAMILIES, FAMILY_LIGHT)
    }
    metrics = {
        "run_dir": str(run_dir.relative_to(REPO)),
        "viz_tail": VIZ_SUBDIR,
        "classification_tail": CLF_SUBDIR,
        "viz2d_family": make_map(run_dir, colors),
        "classification_val": make_confusion(run_dir),
    }
    out = OUT_DIR / "fig_family_classifier_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
