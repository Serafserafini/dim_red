"""
Evaluate EVERY frame of a set of nested-sampling (NS) replica trajectories
with the round 15 "best combo" SupCon body + its round 15
``hierarchical_supcon`` tail (+ round 16's per-family SG visualizers), and
save everything needed to later compute NS partition-function-weighted
thermal averages (e.g. P(family | T, P)) without ever touching the model or
the trajectories again.

Companion to ``examples/ns_grid_from_trajectories.ipynb`` (``mace_embedding``
branch), which only classifies ONE representative frame per (T, P) -- the one
whose enthalpy is closest to <H>(T). Here every frame gets classified, so the
thermal average can be done properly afterwards:

    P(f | T) = sum_i w_i exp(-H_i / kT) 1[f_i = f] / sum_i w_i exp(-H_i / kT)

Input layout (legacy pymatnest/jaxnest, same as
``experiments/test_strucutres/ti_trajs/``): ``ns.<i>.energies`` (one row per
NS iteration: ``iter enthalpy volume``), ``ns.<i>.traj.extxyz`` (the culled
configuration every ``traj_interval`` iterations, tagged with ``info["iter"]``
-- so each frame IS a dead point, matched exactly by ``iter``) and a shared
``ns.inp`` (``MC_cell_P`` = one pressure per replica, in GPa).

Output, per replica, ``<output-dir>/ns.<i>.eval.npz`` (see ``evaluate_replica``
for every key) plus ``<output-dir>/ns.<i>.eval_meta.yaml``. Work is
checkpointed per chunk of frames (``<output-dir>/ns.<i>.chunks/``) so a job
that gets killed resumes where it stopped; the chunk directory is removed
once the merged file is written.

Every family's SG expert (and SG visualizer) is run on EVERY frame -- not just
on the frames predicted to belong to that family -- so the SG distribution
can later be marginalized softly over the family probabilities instead of
only through the hard family prediction.

Run with (from repo root, ``dmred`` active):
    python examples/evaluate_ns_trajectories.py --replica 0 \
        --input-dir experiments/test_strucutres/ti_trajs \
        --output-dir runs/ns_traj_eval/ti_trajs
"""

import argparse
import datetime
import logging
import shutil
import subprocess
import time
from pathlib import Path

import ase.io
import jax
import numpy as np
import yaml
from flax import serialization

from dim_red.pipeline.inference import load_trained_run
from dim_red.soap import compute_soap
from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.tails import ClassificationTail, VisualizationTail
from dim_red.utils import apply_standardization

logger = logging.getLogger("evaluate_ns_trajectories")

DEFAULT_RUN_DIR = (
    "runs/experiment_pipeline_best_combo/"
    "model-supcon_hd-256-128_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp1_supcon-family_only_tau0.05_lf1"
)
DEFAULT_SG_VIZ_TUNE_DIR = "runs/experiment_viz_tune_orthorhombic_tetragonal"
HIERARCHICAL_SUBDIR = "hierarchical_supcon_best_combo"
FAMILY_VIZ_SUBDIR = "visualization_family_only_euclidean"

# Round 16's winning per-family SG-visualizer hidden_dim (see
# experiments/round16_sg_viz_tuning_notes.md). Cubic was never tuned -- it
# keeps best_combo's own baseline visualizer from sg_experts/Cubic.
SG_VIZ_WINNER_TAG = {
    "Hexagonal": "64_32_16",
    "Monoclinic": "64_32_16",
    "Orthorhombic": "64_32_16",
    "Tetragonal": "128_64",
    "Triclinic": "64_32_16",
    "Trigonal": "64_32_16",
}
CANDIDATE_HIDDEN_DIMS = {
    "64_32": [64, 32],
    "128_64": [128, 64],
    "64_32_16": [64, 32, 16],
    "128_64_32": [128, 64, 32],
}

# Legacy .extxyz info keys copied per frame when present (NaN otherwise).
FRAME_INFO_KEYS = (
    "energy_uncert",
    "forces_uncert",
    "forces_uncert_normed",
    "neighbors",
)

GPA_TO_EVA3 = 0.006241509  # same constant jaxrens uses for pressure_units: gpa


# --------------------------------------------------------------------------
# Legacy NS file readers
# --------------------------------------------------------------------------


def parse_ns_inp(path: Path) -> dict:
    """Minimal key=value parser for a legacy pymatnest/jaxnest ``ns.inp``."""
    inp = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        inp[key.strip()] = value.strip()
    return inp


def read_ns_energies(path: Path):
    """Read a legacy ``ns.<i>.energies`` file.

    Header: ``n_walkers n_cull dof flat_V_prior n_atoms``; then one row per
    dead point: ``iteration enthalpy volume``.
    """
    with open(path) as f:
        n_walkers, n_cull, _dof, _flat_v_prior, n_atoms = f.readline().split()
    data = np.loadtxt(path, skiprows=1)
    return (
        data[:, 0].astype(np.int64),
        data[:, 1],
        data[:, 2],
        int(n_walkers),
        int(n_cull),
        int(n_atoms),
    )


def log_prior_mass_weights(n_dead: int, n_live: int, n_cull: int) -> np.ndarray:
    """``log w_i`` of each dead point -- same formula as
    ``jaxrens.postprocess.thermodynamics.calc_log_weights`` (inlined so this
    script doesn't need a jaxrens checkout on the cluster):
    ``log_t = log((N - n_cull) / (N + 1 - n_cull))``,
    ``log w_i = i * log_t + log(1 - t)``.
    """
    log_t = np.log((n_live - n_cull) / (n_live + 1 - n_cull))
    return np.arange(n_dead, dtype=np.float64) * log_t + np.log1p(-np.exp(log_t))


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------


def _load_params(module_wrapper, path: Path):
    with open(path, "rb") as f:
        module_wrapper.params = serialization.from_bytes(
            module_wrapper.params, f.read()
        )
    return module_wrapper


def load_models(run_dir: Path, sg_viz_tune_dir: Path):
    """Everything needed to go from raw SOAP to every saved output."""
    loaded = load_trained_run(run_dir)
    latent_dim = loaded.config.vae.latent_dim
    hier_dir = run_dir / "tails" / HIERARCHICAL_SUBDIR
    with open(hier_dir / "tail_config.yaml") as f:
        hs = yaml.safe_load(f)["hierarchical_supcon"]
    with np.load(hier_dir / "tail_predictions.npz", allow_pickle=True) as npz:
        family_classes = [str(c) for c in npz["family_classes"].tolist()]

    family_tail = _load_params(
        ClassificationTail(
            input_dim=latent_dim,
            hidden_dim=hs["head_hidden_dim"],
            n_family_classes=len(family_classes),
        ),
        hier_dir / "family" / "tail_params.msgpack",
    )

    family_viz_dir = run_dir / "tails" / FAMILY_VIZ_SUBDIR
    with open(family_viz_dir / "tail_config.yaml") as f:
        viz_cfg = yaml.safe_load(f)["visualization"]
    family_viz = _load_params(
        VisualizationTail(
            input_dim=latent_dim,
            hidden_dim=viz_cfg["hidden_dim"],
            output_dim=viz_cfg["viz_dim"],
        ),
        family_viz_dir / "tail_params.msgpack",
    )

    input_dim = int(loaded.feature_mean.shape[0])
    experts = {}
    for family in family_classes:
        expert_dir = hier_dir / "sg_experts" / family
        if not (expert_dir / "sg_body_params.msgpack").exists():
            logger.warning("no SG expert for %s (%s) -- skipped", family, expert_dir)
            continue
        with open(expert_dir / "local_spacegroup_classes.yaml") as f:
            local_classes = [
                int(c) for c in yaml.safe_load(f)["local_spacegroup_classes"]
            ]
        body = _load_params(
            SupConEncoder(
                input_dim=input_dim,
                encoder_hidden_dim=hs["sg_encoder_hidden_dim"],
                latent_dim=hs["sg_latent_dim"],
            ),
            expert_dir / "sg_body_params.msgpack",
        )
        classifier = _load_params(
            ClassificationTail(
                input_dim=hs["sg_latent_dim"],
                hidden_dim=hs["sg_classifier_hidden_dim"],
                n_family_classes=len(local_classes),
            ),
            expert_dir / "classifier_tail_params.msgpack",
        )
        if family in SG_VIZ_WINNER_TAG:
            viz_tag = SG_VIZ_WINNER_TAG[family]
            viz_hidden = CANDIDATE_HIDDEN_DIMS[viz_tag]
            viz_params_path = (
                sg_viz_tune_dir / family / viz_tag / "visualization_tail_params.msgpack"
            )
        else:
            viz_tag = "best_combo_baseline"
            viz_hidden = hs["sg_visualization_hidden_dim"]
            viz_params_path = expert_dir / "visualization_tail_params.msgpack"
        viz = _load_params(
            VisualizationTail(
                input_dim=hs["sg_latent_dim"], hidden_dim=viz_hidden, output_dim=2
            ),
            viz_params_path,
        )
        experts[family] = dict(
            local_classes=local_classes,
            body=body,
            classifier=classifier,
            viz=viz,
            viz_tag=viz_tag,
            viz_hidden_dim=list(viz_hidden),
        )
    return loaded, family_classes, family_tail, family_viz, experts


# --------------------------------------------------------------------------
# Per-chunk evaluation
# --------------------------------------------------------------------------


def evaluate_chunk(frames, loaded, family_tail, family_viz, experts) -> dict:
    """All per-frame outputs for one chunk of trajectory frames."""
    soap_kwargs = dict(loaded.config.soap.as_kwargs())
    soap_kwargs["species"] = loaded.species
    soap_kwargs["average"] = "off"
    # Per-atom SOAP; its mean over atoms is bit-identical to dscribe's
    # average="outer" (the model's actual input), and the spread around that
    # mean is a free, classifier-independent local-order parameter
    # (small for a -- even hot -- crystal, large for a liquid).
    per_atom = np.asarray(compute_soap(frames, **soap_kwargs))  # (n, n_atoms, D)
    soap_mean = per_atom.mean(axis=1)
    dev = per_atom - soap_mean[:, None, :]
    msd = np.mean(np.sum(dev**2, axis=-1), axis=1)
    norm_mean = np.linalg.norm(soap_mean, axis=-1)
    cos_to_mean = np.einsum("nad,nd->na", per_atom, soap_mean) / (
        np.linalg.norm(per_atom, axis=-1) * norm_mean[:, None]
    )

    X_std = apply_standardization(soap_mean, loaded.feature_mean, loaded.feature_std)
    r_main = np.asarray(loaded.model.encode(X_std))
    family_logits = np.asarray(family_tail.classify_family(r_main))
    out = dict(
        soap_mean=soap_mean.astype(np.float32),
        soap_atom_msd=msd,
        soap_atom_msd_rel=msd / norm_mean**2,
        soap_atom_cos_mean=cos_to_mean.mean(axis=1),
        soap_atom_cos_min=cos_to_mean.min(axis=1),
        r_main=r_main.astype(np.float32),
        family_logits=family_logits.astype(np.float32),
        z_family=np.asarray(family_viz.project(r_main)).astype(np.float32),
    )
    for family, ex in experts.items():
        r_sg = np.asarray(ex["body"].encode(X_std))
        out[f"r_sg__{family}"] = r_sg.astype(np.float32)
        out[f"sg_logits__{family}"] = np.asarray(
            ex["classifier"].classify_family(r_sg)
        ).astype(np.float32)
        out[f"z_sg__{family}"] = np.asarray(ex["viz"].project(r_sg)).astype(np.float32)
    return out


def frame_scalars(frames, pressure_eva3: float) -> dict:
    """Per-frame metadata read straight off the trajectory frames."""
    # ASE moves extxyz's per-frame energy/forces off info/arrays onto a
    # SinglePointCalculator -- read them back from there.
    results = [a.calc.results if a.calc is not None else {} for a in frames]
    out = {
        "iter": np.array([a.info["iter"] for a in frames], dtype=np.int64),
        "cell": np.array([a.cell.array for a in frames], dtype=np.float64),
        "positions": np.array([a.positions for a in frames], dtype=np.float32),
        "numbers": np.array([a.numbers for a in frames], dtype=np.int16),
        "volume_traj": np.array([a.get_volume() for a in frames], dtype=np.float64),
        "energy": np.array(
            [r.get("energy", np.nan) for r in results], dtype=np.float64
        ),
        "forces": np.array(
            [
                r.get("forces", np.full((len(a), 3), np.nan))
                for r, a in zip(results, frames)
            ],
            dtype=np.float32,
        ),
    }
    for key in FRAME_INFO_KEYS:
        out[f"info_{key}"] = np.array(
            [float(a.info.get(key, np.nan)) for a in frames], dtype=np.float64
        )
    # NPT NS energies column is H = U + P V -- rebuild it from the frame itself
    # as a consistency check against the matched .energies row.
    out["enthalpy_traj"] = out["energy"] + pressure_eva3 * out["volume_traj"]
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- purely informational
        return "unknown"


def _iter_chunks(traj_path: Path, chunk_size: int, stride: int, max_frames):
    """Yield (chunk_id, first_traj_index, frames, traj_indices)."""
    chunk, idxs, chunk_id = [], [], 0
    for traj_idx, atoms in enumerate(ase.io.iread(traj_path, index=":")):
        if max_frames is not None and traj_idx >= max_frames:
            break
        if traj_idx % stride:
            continue
        chunk.append(atoms)
        idxs.append(traj_idx)
        if len(chunk) == chunk_size:
            yield chunk_id, chunk, np.array(idxs, dtype=np.int64)
            chunk, idxs, chunk_id = [], [], chunk_id + 1
    if chunk:
        yield chunk_id, chunk, np.array(idxs, dtype=np.int64)


def evaluate_replica(args, replica: int, models) -> Path:
    loaded, family_classes, family_tail, family_viz, experts = models
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.file_prefix}.{replica}"
    final_path = output_dir / f"{prefix}.eval.npz"
    if final_path.exists() and not args.overwrite:
        logger.info(
            "%s already exists -- skipping (pass --overwrite to redo)", final_path
        )
        return final_path

    ns_inp = parse_ns_inp(input_dir / "ns.inp")
    pressures_gpa = [float(x) for x in ns_inp["MC_cell_P"].split()]
    if ns_inp.get("pressure_conversion", "gpa_to_eva3") != "gpa_to_eva3":
        raise ValueError(
            f"unexpected pressure_conversion={ns_inp['pressure_conversion']!r}"
        )
    pressure_gpa = pressures_gpa[replica]
    pressure_eva3 = pressure_gpa * GPA_TO_EVA3

    dead_iter, dead_H, dead_V, n_walkers, n_cull, n_atoms = read_ns_energies(
        input_dir / f"{prefix}.energies"
    )
    if np.any(np.diff(dead_iter) < 0):
        raise ValueError(f"{prefix}.energies iterations are not sorted")
    dead_log_w = log_prior_mass_weights(len(dead_H), n_walkers, n_cull)
    logger.info(
        "replica %d: P=%g GPa, n_dead=%d, n_walkers=%d, n_cull=%d, n_atoms=%d",
        replica,
        pressure_gpa,
        len(dead_H),
        n_walkers,
        n_cull,
        n_atoms,
    )

    chunk_dir = output_dir / f"{prefix}.chunks"
    chunk_dir.mkdir(exist_ok=True)
    traj_path = input_dir / f"{prefix}.traj.extxyz"
    t0 = time.time()
    n_done = 0
    for chunk_id, frames, traj_idx in _iter_chunks(
        traj_path, args.chunk_size, args.stride, args.max_frames
    ):
        chunk_path = chunk_dir / f"chunk_{chunk_id:05d}.npz"
        n_done += len(frames)
        if chunk_path.exists():
            continue
        tc = time.time()
        data = frame_scalars(frames, pressure_eva3)
        data["traj_index"] = traj_idx
        data.update(evaluate_chunk(frames, loaded, family_tail, family_viz, experts))
        tmp = chunk_path.with_suffix(".tmp.npz")
        np.savez(tmp, **data)
        tmp.rename(chunk_path)
        logger.info(
            "replica %d chunk %d: %d frames in %.1fs (total %d frames, %.0fs)",
            replica,
            chunk_id,
            len(frames),
            time.time() - tc,
            n_done,
            time.time() - t0,
        )

    chunk_paths = sorted(chunk_dir.glob("chunk_*[0-9].npz"))
    parts = [dict(np.load(p)) for p in chunk_paths]
    frames_data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts

    # Exact frame -> dead-point match (each frame IS a culled configuration).
    dead_index = np.searchsorted(dead_iter, frames_data["iter"])
    dead_index = np.clip(dead_index, 0, len(dead_iter) - 1)
    matched = dead_iter[dead_index] == frames_data["iter"]
    if not matched.all():
        logger.warning(
            "replica %d: %d/%d frames have no .energies row with the same iter",
            replica,
            int((~matched).sum()),
            len(matched),
        )
    dH = np.abs(dead_H[dead_index] - frames_data["enthalpy_traj"])[matched]
    logger.info(
        "replica %d: |H(.energies) - (E + PV)(traj)| max=%.3g eV, median=%.3g eV",
        replica,
        dH.max(),
        np.median(dH),
    )

    # Hierarchical prediction: family argmax, then that family's own SG expert.
    family_pred_idx = frames_data["family_logits"].argmax(axis=1)
    sg_pred = np.zeros(len(family_pred_idx), dtype=np.int64)
    for fi, family in enumerate(family_classes):
        rows = family_pred_idx == fi
        if family not in experts or not rows.any():
            continue
        local = np.asarray(experts[family]["local_classes"])
        sg_pred[rows] = local[frames_data[f"sg_logits__{family}"][rows].argmax(axis=1)]

    np.savez(
        final_path,
        # --- per frame -----------------------------------------------------
        **frames_data,
        dead_index=dead_index,
        dead_matched=matched,
        enthalpy=dead_H[dead_index],
        volume=dead_V[dead_index],
        log_w=dead_log_w[dead_index],
        family_pred_idx=family_pred_idx,
        sg_pred=sg_pred,
        # --- every dead point of the run (not only the saved frames) -------
        all_dead_iter=dead_iter,
        all_dead_enthalpy=dead_H,
        all_dead_volume=dead_V,
        all_dead_log_w=dead_log_w,
        # --- constants ------------------------------------------------------
        family_classes=np.array(family_classes),
        pressure_gpa=pressure_gpa,
        pressure_eva3=pressure_eva3,
        n_walkers=n_walkers,
        n_cull=n_cull,
        n_atoms=n_atoms,
        replica=replica,
        **{
            f"local_sg_classes__{f}": np.asarray(ex["local_classes"], dtype=np.int64)
            for f, ex in experts.items()
        },
    )
    meta = dict(
        replica=replica,
        pressure_gpa=pressure_gpa,
        input_dir=str(input_dir.resolve()),
        run_dir=str(Path(args.run_dir).resolve()),
        hierarchical_subdir=HIERARCHICAL_SUBDIR,
        family_viz_subdir=FAMILY_VIZ_SUBDIR,
        sg_viz_tune_dir=str(Path(args.sg_viz_tune_dir).resolve()),
        sg_viz={
            f: dict(tag=ex["viz_tag"], hidden_dim=ex["viz_hidden_dim"])
            for f, ex in experts.items()
        },
        soap=dict(loaded.config.soap.as_kwargs(), species=list(loaded.species)),
        family_classes=family_classes,
        local_sg_classes={f: ex["local_classes"] for f, ex in experts.items()},
        n_frames=int(len(frames_data["iter"])),
        stride=args.stride,
        max_frames=args.max_frames,
        n_walkers=n_walkers,
        n_cull=n_cull,
        n_atoms=n_atoms,
        n_dead=int(len(dead_H)),
        frames_without_dead_match=int((~matched).sum()),
        enthalpy_match_max_abs_diff_ev=float(dH.max()),
        git_commit=_git_commit(),
        created=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    with open(output_dir / f"{prefix}.eval_meta.yaml", "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    shutil.rmtree(chunk_dir)
    logger.info(
        "replica %d: wrote %s (%d frames)", replica, final_path, meta["n_frames"]
    )
    return final_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--replica",
        type=int,
        nargs="+",
        required=True,
        help="replica index/indices i (ns.<i>.*) to evaluate",
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--sg-viz-tune-dir", default=DEFAULT_SG_VIZ_TUNE_DIR)
    parser.add_argument("--file-prefix", default="ns")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=1, help="keep every n-th frame")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="only look at the first N trajectory frames (smoke tests)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger.info("jax devices: %s", jax.devices())
    models = load_models(Path(args.run_dir), Path(args.sg_viz_tune_dir))
    for replica in args.replica:
        evaluate_replica(args, replica, models)


if __name__ == "__main__":
    main()
