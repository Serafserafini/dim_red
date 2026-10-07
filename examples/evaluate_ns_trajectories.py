"""
Evaluate EVERY frame of a set of nested-sampling (NS) replica trajectories
with a trained ``FullStack`` run (family stack + per-crystal-system expert
stacks, one heads set), and save everything needed to later compute NS
partition-function-weighted thermal averages (e.g. P(family | T, P)) without
ever touching the model or the trajectories again.

Companion to ``examples/ns_grid_from_trajectories.ipynb``, which only
classifies ONE representative frame per (T, P) -- the one whose enthalpy is
closest to <H>(T). Here every frame gets classified, so the thermal average
can be done properly afterwards:

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

Every trained expert stack (classifier and visualization head) is run on
EVERY frame -- not just on the frames predicted to belong to its crystal
system -- so the SG distribution can later be marginalized softly over the
family probabilities instead of only through the hard family prediction. The
hard prediction (``family_pred_idx``, ``sg_pred``) routes exactly like
``FullStack.predict``: argmax family, then that family's expert (``sg_pred``
is ``-1`` for a frame whose predicted family has no trained expert).
``family_logits`` / ``sg_logits__<Family>`` are log class probabilities
(softmax-equivalent to the classifier logits).

Run with (from repo root, ``dmred`` active):
    python examples/evaluate_ns_trajectories.py --replica 0 \
        --run-dir runs/<full_stack_run> \
        --input-dir experiments/test_strucutres/ti_trajs \
        --output-dir runs/ns_traj_eval/ti_trajs
"""

import argparse
import dataclasses
import datetime
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import ase.io
import jax
import numpy as np
import yaml

from dim_red.pipeline.featurize import featurize_structures, standardize
from dim_red.pipeline.full_stack import FullStack
from dim_red.pipeline.run_layout import EXPERT_NAMES, FAMILY, resolve_heads_name
from dim_red.soap import compute_soap

logger = logging.getLogger("evaluate_ns_trajectories")

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


def _log_proba(proba: np.ndarray) -> np.ndarray:
    return np.log(np.clip(proba, 1e-12, 1.0)).astype(np.float32)


def load_models(run_dir: Path, heads_name, device: str) -> dict:
    """Every trained stack of a FullStack run, loaded once: the same
    ``SingleStack`` (body + heads), class list and feature standardization
    ``FullStack.predict`` uses -- held in memory so each chunk of frames
    doesn't reload them."""
    full_stack = FullStack.open(run_dir)
    trained = full_stack.stack_names()
    if FAMILY not in trained:
        raise ValueError(f"{run_dir}: family stack is not trained")
    stacks = {}
    for name in trained:
        stack_dir = run_dir / "stacks" / name
        heads = resolve_heads_name(stack_dir, heads_name)
        with open(stack_dir / "classes.yaml") as f:
            classes = yaml.safe_load(f)["classes"]
        with np.load(stack_dir / "embeddings.npz") as npz:
            mean, std = npz["feature_mean"], npz["feature_std"]
        stacks[name] = dict(
            stack=full_stack.load_stack(name, heads, device=device),
            heads_name=heads,
            classes=[str(c) for c in classes] if name == FAMILY else classes,
            mean=mean,
            std=std,
            data=full_stack.config.stacks[name].data,
        )
    for name in EXPERT_NAMES:
        if name not in stacks:
            logger.warning("no expert stack %r in %s -- skipped", name, run_dir)
    return dict(
        run_dir=run_dir,
        model_kind=full_stack.config.model_kind,
        family_classes=stacks[FAMILY]["classes"],
        family=stacks[FAMILY],
        # Family label (e.g. "Cubic") -> its expert stack.
        experts={n.capitalize(): s for n, s in stacks.items() if n in EXPERT_NAMES},
    )


# --------------------------------------------------------------------------
# Per-chunk evaluation
# --------------------------------------------------------------------------


def evaluate_chunk(frames, models) -> dict:
    """All per-frame outputs for one chunk of trajectory frames."""
    family = models["family"]
    soap_kwargs = dict(family["data"].soap.as_kwargs())
    present = {s for a in frames for s in a.get_chemical_symbols()}
    soap_kwargs["species"] = sorted(present | set(soap_kwargs.get("species") or []))
    soap_kwargs["average"] = "off"
    # Per-atom SOAP (the family stack's settings); its mean over atoms is
    # dscribe's average="outer" (the model's actual input), and the spread
    # around that mean is a free, classifier-independent local-order parameter
    # (small for a -- even hot -- crystal, large for a liquid).
    per_atom = np.asarray(compute_soap(frames, **soap_kwargs))  # (n, n_atoms, D)
    soap_mean = per_atom.mean(axis=1)
    dev = per_atom - soap_mean[:, None, :]
    msd = np.mean(np.sum(dev**2, axis=-1), axis=1)
    norm_mean = np.linalg.norm(soap_mean, axis=-1)
    cos_to_mean = np.einsum("nad,nd->na", per_atom, soap_mean) / (
        np.linalg.norm(per_atom, axis=-1) * norm_mean[:, None]
    )

    # Raw model input per distinct featurizer config: the per-atom mean above
    # when a stack uses the family stack's own SOAP settings, otherwise
    # featurize_structures (other SOAP settings, or MACE).
    raw_cache = {}

    def raw_features(data):
        if models["model_kind"] != "supcon_mace" and data.soap == family["data"].soap:
            return soap_mean
        key = json.dumps(dataclasses.asdict(data), sort_keys=True, default=str)
        if key not in raw_cache:
            raw_cache[key] = featurize_structures(frames, models["model_kind"], data)
        return raw_cache[key]

    def run(entry):
        X = standardize(raw_features(entry["data"]), entry["mean"], entry["std"])
        stack = entry["stack"]
        return (
            stack.encode(X).astype(np.float32),
            _log_proba(stack.predict_proba(X)),
            stack.visualize(X).astype(np.float32),
        )

    r_main, family_logits, z_family = run(family)
    out = dict(
        soap_mean=soap_mean.astype(np.float32),
        soap_atom_msd=msd,
        soap_atom_msd_rel=msd / norm_mean**2,
        soap_atom_cos_mean=cos_to_mean.mean(axis=1),
        soap_atom_cos_min=cos_to_mean.min(axis=1),
        r_main=r_main,
        family_logits=family_logits,
        z_family=z_family,
    )
    for fam, entry in models["experts"].items():
        r_sg, sg_logits, z_sg = run(entry)
        out[f"r_sg__{fam}"] = r_sg
        out[f"sg_logits__{fam}"] = sg_logits
        out[f"z_sg__{fam}"] = z_sg
    return out


def routed_prediction(frames_data: dict, models) -> tuple:
    """``(family_pred_idx, sg_pred)``, routed like ``FullStack.predict``:
    argmax family, then that family's own expert; ``-1`` without one."""
    family_pred_idx = frames_data["family_logits"].argmax(axis=1)
    sg_pred = np.full(len(family_pred_idx), -1, dtype=np.int64)
    for fi, fam in enumerate(models["family_classes"]):
        rows = family_pred_idx == fi
        if fam not in models["experts"] or not rows.any():
            continue
        local = np.asarray(models["experts"][fam]["classes"], dtype=np.int64)
        sg_pred[rows] = local[frames_data[f"sg_logits__{fam}"][rows].argmax(axis=1)]
    return family_pred_idx, sg_pred


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
    family_classes, experts = models["family_classes"], models["experts"]
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
        data.update(evaluate_chunk(frames, models))
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
    family_pred_idx, sg_pred = routed_prediction(frames_data, models)

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
            f"local_sg_classes__{f}": np.asarray(ex["classes"], dtype=np.int64)
            for f, ex in experts.items()
        },
    )
    meta = dict(
        replica=replica,
        pressure_gpa=pressure_gpa,
        input_dir=str(input_dir.resolve()),
        run_dir=str(Path(args.run_dir).resolve()),
        heads_name=models["family"]["heads_name"],
        model_kind=models["model_kind"],
        soap=models["family"]["data"].soap.as_kwargs(),
        family_classes=family_classes,
        local_sg_classes={
            f: [int(c) for c in ex["classes"]] for f, ex in experts.items()
        },
        expert_heads_name={f: ex["heads_name"] for f, ex in experts.items()},
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
    parser.add_argument("--run-dir", required=True, help="FullStack run directory")
    parser.add_argument(
        "--heads-name",
        default=None,
        help="heads set to use (default: the only one each stack has)",
    )
    parser.add_argument(
        "--device", default="cpu", help="jax backend for the params (cpu/gpu)"
    )
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
    models = load_models(Path(args.run_dir), args.heads_name, args.device)
    for replica in args.replica:
        evaluate_replica(args, replica, models)


if __name__ == "__main__":
    main()
