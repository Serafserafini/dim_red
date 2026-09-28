"""Talk figure: 2D map of the SINGLE-STAGE control model, coloured by space group.

Single-stage = round-15 best_combo SupCon body trained directly with
``supcon.mode: family_and_spacegroup`` (no family -> per-family SG experts),
plus a family_and_spacegroup 2D visualization tail on top. See
configs/presentation_single_stage_family_and_spacegroup.yaml and
configs/presentation_single_stage_viz_tail.yaml.

Colouring: one base hue per crystal family (blue -> magenta, presentation
palette), and within a family every space group gets its own shade of that
hue (lightness/saturation/small hue jitter, assigned in golden-ratio-scrambled
order so neighbouring SG numbers don't look alike) -- so "hue = family,
shade = space group", and SG mixing inside a family blob shows up as speckle.

Writes Presentation/fig_single_stage_sg_map.png and
Presentation/fig_single_stage_sg_metrics.json. (The earlier colour-by-SG-number
version is kept as fig_single_stage_sg_map_bynumber.png.)

Usage (from the repo root, conda env ``dmred``):
    python Presentation/make_fig_single_stage_sg.py [RUN_DIR]
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

from dim_red.analysis.metrics import embedding_quality_metrics  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "Presentation"
VIZ_SUBDIR = "visualization_family_and_spacegroup_euclidean"

# (short name, first SG, last SG, base hue in degrees). Hues run blue ->
# magenta: #006699 (~200 deg) / #5485AB-#72ADD5 (~207) ... #BA4682 (~331).
FAMILIES = [
    ("Triclinic", 1, 2, 186.0),
    ("Monoclinic", 3, 15, 204.0),
    ("Orthorhombic", 16, 74, 222.0),
    ("Tetragonal", 75, 142, 248.0),
    ("Trigonal", 143, 167, 276.0),
    ("Hexagonal", 168, 194, 306.0),
    ("Cubic", 195, 230, 336.0),
]
HUE_JITTER = 9.0  # +- degrees within a family
LIGHT_RANGE = (0.22, 0.68)  # HLS lightness: dark but never pale on white
SAT_RANGE = (0.38, 0.72)
GOLDEN = 0.6180339887498949


def _find_run_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    hits = sorted(
        glob.glob(str(REPO / "experiments/runs/presentation_single_stage/model-*"))
    )
    if not hits:
        raise SystemExit("No run under experiments/runs/presentation_single_stage/")
    return Path(hits[-1])


def _family_sg_colors() -> dict:
    """``{sg: rgb}`` for all 230 SGs. Deterministic: the k-th SG of a family
    (in SG-number order) takes lightness from the golden-ratio sequence
    frac(k * phi), saturation and hue jitter from two further, differently
    phased low-discrepancy sequences -- consecutive SGs land far apart in
    lightness, and shades spread evenly over the whole range."""
    colors = {}
    for fi, (_, lo, hi, hue) in enumerate(FAMILIES):
        n = hi - lo + 1
        for k in range(n):
            if n == 1:
                t = 0.5
            else:
                t = (0.5 + k * GOLDEN) % 1.0
            u = (0.3 + k * 0.7548776662) % 1.0  # plastic-ratio step
            w = (0.1 + k * 0.5698402910) % 1.0
            light = LIGHT_RANGE[0] + t * (LIGHT_RANGE[1] - LIGHT_RANGE[0])
            sat = SAT_RANGE[0] + u * (SAT_RANGE[1] - SAT_RANGE[0])
            h = (hue + (2 * w - 1) * HUE_JITTER) / 360.0
            colors[lo + k] = colorsys.hls_to_rgb(h % 1.0, light, sat)
    return colors


def _draw_family_key(fig, rect, colors) -> None:
    """Slim key: one block per family (tric. at the bottom), each drawn as
    thin vertical stripes of that family's SG shades in lightness order."""
    kax = fig.add_axes(rect)
    n_fam = len(FAMILIES)
    gap = 0.18
    for fi, (name, lo, hi, _) in enumerate(FAMILIES):
        sg_cols = [colors[s] for s in range(lo, hi + 1)]
        sg_cols.sort(key=lambda c: colorsys.rgb_to_hls(*c)[1])
        y0 = fi + gap / 2
        m = len(sg_cols)
        for j, c in enumerate(sg_cols):
            kax.add_patch(plt.Rectangle((j / m, y0), 1.0 / m, 1 - gap, color=c, lw=0))
        rng = f"{lo}\u2013{hi}"
        kax.text(
            1.12,
            y0 + (1 - gap) / 2 + 0.13,
            name,
            fontsize=14,
            va="center",
            ha="left",
            color="#222222",
        )
        kax.text(
            1.12,
            y0 + (1 - gap) / 2 - 0.22,
            f"SG {rng}",
            fontsize=11,
            va="center",
            ha="left",
            color="#666666",
        )
    kax.set_xlim(0, 1)
    kax.set_ylim(0, n_fam)
    kax.axis("off")


def main() -> None:
    run_dir = _find_run_dir()
    viz = np.load(run_dir / "tails" / VIZ_SUBDIR / "tail_embeddings.npz")
    z = viz["embeddings"]
    sgs = viz["spacegroups"].astype(int)
    fams = viz["labels"]
    split = viz["split"]

    # --- figure: all points (train + val), same as the pipeline's viz plots.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    colors = _family_sg_colors()
    point_colors = np.array([colors[s] for s in sgs])

    rng = np.random.default_rng(0)
    order = rng.permutation(len(z))  # random draw order: no SG sits on top

    fig = plt.figure(figsize=(12, 6.75), facecolor="white")
    ax = fig.add_axes([0.005, 0.01, 0.765, 0.98])
    ax.scatter(
        z[order, 0],
        z[order, 1],
        c=point_colors[order],
        s=28,
        alpha=0.62,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # ~1.67x horizontal stretch so the ~square embedding fills the
    # 16:9 canvas without distorting cluster shapes much.
    ax.set_aspect(0.6, adjustable="datalim")

    _draw_family_key(fig, [0.78, 0.18, 0.045, 0.64], colors)

    png = OUT_DIR / "fig_single_stage_sg_map.png"
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {png}")

    # --- metrics on the 2D embedding (all points and val split only).
    metrics = {"run_dir": str(run_dir.relative_to(REPO)), "viz_tail": VIZ_SUBDIR}
    for name, mask in (("all", np.ones(len(z), bool)), ("val", split == "val")):
        m = embedding_quality_metrics(
            z[mask], {"family": fams[mask], "spacegroup": sgs[mask]}
        )
        metrics[f"viz2d_{name}"] = {
            k: (None if np.isnan(v) else float(v)) for k, v in m.items()
        }
        metrics[f"viz2d_{name}"]["n_points"] = int(mask.sum())

    clf_path = run_dir / "tails" / "classification" / "tail_predictions.npz"
    if clf_path.exists():
        p = np.load(clf_path)
        body_split = np.load(run_dir / "embeddings.npz")["split"]
        acc = {}
        for split_name in ("train", "val"):
            sel = body_split == split_name
            fam_pred = p["family_classes"][p["family_probs"][sel].argmax(1)]
            sg_pred = p["spacegroup_classes"][p["spacegroup_probs"][sel].argmax(1)]
            acc[split_name] = {
                "family_accuracy": float(np.mean(fam_pred == p["labels"][sel])),
                "spacegroup_accuracy": float(
                    np.mean(sg_pred.astype(int) == p["spacegroups"][sel].astype(int))
                ),
            }
        metrics["classification_tail"] = acc

    out = OUT_DIR / "fig_single_stage_sg_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
