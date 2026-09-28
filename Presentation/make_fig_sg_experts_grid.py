"""Talk figure (slide 12): per-family space-group experts, 2x2 grid.

Same embeddings as experiments/plots/round16/grid_all_families_BEST.png
("hierarchical_supcon (best combo body) -- best visualization tail per
family"), restricted to 4 families:

  - Cubic      [64, 32]     -- the best_combo run's own hierarchical_supcon
                               SG expert viz tail (never tuned in round 16)
  - Monoclinic [64, 32, 16] -- round-16 tuning output
  - Trigonal   [64, 32, 16] -- round-16 tuning output
  - Tetragonal [128, 64]    -- round-16 tuning output

(tuning outputs from examples/tune_sg_visualization_hidden_dims.py, written
to experiments/runs/experiment_viz_tune_orthorhombic_tetragonal/<family>/<dims>/).

Points coloured with a panel-local categorical palette (panel_palette): every
SG within a panel gets a clearly distinct colour from the blue -> violet ->
magenta presentation range (NOT the slide-10 per-SG shades).

``--source mace`` instead draws the same grid from the round-18 supcon_mace
run's hierarchical_supcon per-family SG experts
(runs/model-supcon_mace_hd-256-128_*/tails/
hierarchical_supcon_supcon_mace_best_combo_scale/sg_experts/<family>/), whose
per-family viz-tail hidden dims (tail_config.yaml) are the same as above;
its metrics also carry the per-family oracle SG accuracy from that tail's
tail_predictions.npz.

Writes Presentation/fig_sg_experts_grid{,_mace}.png and
Presentation/fig_sg_experts_grid{,_mace}_metrics.json.

Usage (from the repo root, conda env ``dmred``):
    python Presentation/make_fig_sg_experts_grid.py [--source soap|mace]
"""

import argparse
import colorsys
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_fig_single_stage_sg import FAMILIES  # noqa: E402  (base hues)

from dim_red.analysis.metrics import embedding_quality_metrics  # noqa: E402

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
TUNE_DIR = REPO / "experiments/runs/experiment_viz_tune_orthorhombic_tetragonal"

# (family, hidden_dim tag, source npz) in panel order: TL, TR, BL, BR.
PANELS = [
    (
        "Monoclinic",
        "64_32_16",
        TUNE_DIR / "Monoclinic/64_32_16/visualization_embeddings.npz",
    ),
    (
        "Trigonal",
        "64_32_16",
        TUNE_DIR / "Trigonal/64_32_16/visualization_embeddings.npz",
    ),
    (
        "Tetragonal",
        "128_64",
        TUNE_DIR / "Tetragonal/128_64/visualization_embeddings.npz",
    ),
    (
        "Cubic",
        "64_32",
        RUN_DIR
        / "tails/hierarchical_supcon_best_combo/sg_experts/Cubic"
        / "visualization_embeddings.npz",
    ),
]
MACE_TAIL = Path(
    glob.glob(
        str(
            REPO / "runs/model-supcon_mace_hd-256-128_*_supcon-family_only_tau0.05_lf1"
            "/tails/hierarchical_supcon_supcon_mace_best_combo_scale"
        )
    )[0]
)
MACE_PANELS = [
    (fam, tag, MACE_TAIL / "sg_experts" / fam / "visualization_embeddings.npz")
    for fam, tag, _ in PANELS
]
# source -> (panels, output stem, tail_predictions.npz for oracle SG acc or None)
SOURCES = {
    "soap": (PANELS, "fig_sg_experts_grid", None),
    "mace": (
        MACE_PANELS,
        "fig_sg_experts_grid_mace",
        MACE_TAIL / "tail_predictions.npz",
    ),
}
GOLDEN = 0.6180339887498949
MAX_STRETCH = 1.3
# --source mace only: points whose distance r from the panel's median point
# exceeds median(r) + OUTLIER_K * MAD(r) are left out of the drawing (and so
# of the axis range); metrics still use all points. Drops only the 5 far SG-114
# Tetragonal points (same count for any K in 8..20).
OUTLIER_K = 10.0
LIGHTS = (0.30, 0.52, 0.40, 0.62, 0.35, 0.47, 0.57)


def panel_palette(sg_numbers) -> dict:
    """Panel-local categorical palette: the k-th SG (in SG-number order)
    gets hue frac(k*phi) spread over the presentation's teal -> blue ->
    violet -> magenta range (widened slightly toward cyan and magenta-red
    for panels with many SGs), lightness cycling through 7 levels (never
    pale on white) and saturation alternating. Deterministic."""
    sgs = sorted(int(s) for s in sg_numbers)
    n = len(sgs)
    h_lo, h_hi = (178.0, 352.0) if n > 40 else (185.0, 340.0)
    out = {}
    for k, sg in enumerate(sgs):
        t = (0.5 + k * GOLDEN) % 1.0
        hue = (h_lo + t * (h_hi - h_lo)) / 360.0
        light = LIGHTS[k % len(LIGHTS)]
        sat = 0.80 if k % 2 == 0 else 0.58
        out[sg] = colorsys.hls_to_rgb(hue, light, sat)
    return out


# Family base hues, shared with the slide-10 script (full family names).
BASE_HUE = {name: hue for name, _, _, hue in FAMILIES}


def oracle_sg_accuracy(pred_path, family) -> dict:
    p = np.load(pred_path)
    m = p["labels"] == family
    pred = p["spacegroup_classes"][p["spacegroup_probs_oracle"][m].argmax(1)]
    true, val = p["spacegroups"][m], p["split"][m] == "val"
    return {
        "all": float(np.mean(pred == true)),
        "val": float(np.mean(pred[val] == true[val])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(SOURCES), default="soap")
    source = ap.parse_args().source
    panels, stem, pred_path = SOURCES[source]
    hide_outliers = source == "mace"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    # Slide slot 560 x 437 px (100 px/in convention).
    FW, FH = 5.60, 4.37
    fig = plt.figure(figsize=(FW, FH), facecolor="white")
    # Layout in inches, converted to figure fractions.
    m_in, gap_x_in, gap_y_in, label_in = 0.04, 0.10, 0.06, 0.22
    w_in = (FW - 2 * m_in - gap_x_in) / 2
    h_in = (FH - 2 * m_in - gap_y_in - 2 * label_in) / 2
    w, h = w_in / FW, h_in / FH
    # Dot area scaled to panel area (reference: 2.82 x 2.52 in panels, s=9).
    dot_s = 9 * (w_in * h_in) / (2.82 * 2.52)
    metrics = {}
    stretches = {}
    hidden = {}
    for k, (family, tag, path) in enumerate(panels):
        d = np.load(path)
        z, sgs, split = d["embeddings"], d["spacegroups"].astype(int), d["split"]
        row, col = divmod(k, 2)
        x0 = (m_in + col * (w_in + gap_x_in)) / FW
        y0 = (FH - m_in - (row + 1) * (h_in + label_in) - row * gap_y_in) / FH
        ax = fig.add_axes([x0, y0, w, h])
        colors = panel_palette(np.unique(sgs))
        zp, sp = z, sgs
        if hide_outliers:
            r = np.linalg.norm(z - np.median(z, 0), axis=1)
            med = np.median(r)
            keep = r <= med + OUTLIER_K * np.median(np.abs(r - med))
            zp, sp = z[keep], sgs[keep]
            hidden[family] = int((~keep).sum())
        order = np.random.default_rng(0).permutation(len(zp))
        ax.scatter(
            zp[order, 0],
            zp[order, 1],
            c=np.array([colors[s] for s in sp])[order],
            s=dot_s,
            alpha=0.66,
            edgecolors="black",
            linewidths=0.12,
            rasterized=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Horizontal stretch (x units drawn wider than y units) only as much
        # as needed to fill the wide panel, capped at MAX_STRETCH.
        dx, dy = np.ptp(zp[:, 0]), np.ptp(zp[:, 1])
        stretch = float(np.clip((w_in / h_in) / (dx / dy), 1.0, MAX_STRETCH))
        ax.set_aspect(1.0 / stretch, adjustable="datalim")
        ax.margins(0.03)
        stretches[family] = stretch
        if hidden.get(family):
            n_h = hidden[family]
            ax.text(
                0.99,
                0.01,
                f"{n_h} outlier{'s' if n_h > 1 else ''} not shown",
                transform=ax.transAxes,
                fontsize=8,
                color="#888888",
                ha="right",
                va="bottom",
            )
        n_sg = len(np.unique(sgs))
        label_col = colorsys.hls_to_rgb(BASE_HUE[family] / 360.0, 0.36, 0.60)
        ty = y0 + h + 0.03 / FH
        t = fig.text(
            x0 + 0.02 / FW,
            ty,
            family,
            fontsize=11.5,
            fontweight="bold",
            color=label_col,
            ha="left",
            va="bottom",
        )
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        x_end = fig.transFigure.inverted().transform((bb.x1, bb.y0))[0]
        fig.text(
            x_end + 0.06 / FW,
            ty,
            f"({n_sg} space groups)",
            fontsize=9.5,
            color="#555555",
            ha="left",
            va="bottom",
        )

        fam_m = {
            "source": str(path.relative_to(REPO)),
            "viz_hidden_dim": tag,
            "n_points": int(len(z)),
            "n_spacegroups": n_sg,
        }
        for name, mask in (("all", np.ones(len(z), bool)), ("val", split == "val")):
            m = embedding_quality_metrics(z[mask], {"spacegroup": sgs[mask]})
            fam_m[name] = {
                k2: (None if np.isnan(v) else float(v)) for k2, v in m.items()
            }
        if pred_path is not None:
            fam_m["oracle_sg_accuracy"] = oracle_sg_accuracy(pred_path, family)
        metrics[family] = fam_m

    for ext in ("png",):
        fig.savefig(
            OUT_DIR / f"{stem}.{ext}",
            dpi=600,
            facecolor="white",
            bbox_inches=None,
        )
    plt.close(fig)
    print("horizontal stretch per panel:", stretches, file=sys.stderr)
    if hide_outliers:
        print("points not shown per panel:", hidden, file=sys.stderr)
    out = OUT_DIR / f"{stem}_metrics.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
