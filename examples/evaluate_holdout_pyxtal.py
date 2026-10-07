"""
Evaluate already-trained FullStack runs on a brand-new, never-seen pyxtal test set.

A run's own ``split == "val"`` rows are drawn from the same generated
database as its training rows (augmented copies of one structure are kept
together by the grouped split, but the generation settings, seed and
distribution are shared), so the val numbers can be optimistic. This script
regenerates a fresh pyxtal database with a different seed (default 20260925),
matching each run's family stack generation settings (``data.pyxtal`` of
``stacks/family/config.yaml``: families, ``n_species``, ``species_pool``,
``candidate_num_ions``, ``factor``, ``max_count``) but with
``structures_per_family`` forced to 500, uniform, and applies only the family
stack's ``augmentation.supercell_radius`` step
(``dim_red.augmentation.make_supercell_for_radius``) -- no jitter, no
vacancies, no near-duplicate filtering. Nothing is retrained.

Per run (``FullStack.open(run_dir)``, one ``--heads-name``) it computes, on
the test set:

* family accuracy / balanced accuracy / per-family recall of the family
  stack's classifier, next to the same numbers on the family stack's own val
  rows (``stacks/family/heads/<heads>/predictions.npz``);
* when expert stacks are trained: SG accuracy end-to-end
  (``FullStack.predict``: predicted family -> that family's expert; a frame
  routed to a family without a trained expert counts as wrong) and with oracle
  routing (true family -> that family's expert, ``FullStack.predict_stack``),
  plus per-family oracle SG accuracy, next to the experts' own val accuracy;
* the 2D family silhouette of the family visualization head on the test
  points (``dim_red.analysis.metrics.embedding_quality_metrics``), next to the
  same metric on the family stack's own val rows.

Generated test sets are cached (``<cache-dir>/testset_<hash>.extxyz`` plus a
``.json`` with the generation settings, per-family counts and pyxtal
failures) and reused by every run with the same generation config.

Run with (from repo root, ``dmred`` active):
    python examples/evaluate_holdout_pyxtal.py --sweep-dir runs/my_sweep \
        --cache-dir runs/holdout_pyxtal/cache --output-dir runs/holdout_pyxtal
"""

import argparse
import csv
import dataclasses
import datetime
import hashlib
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import ase.io
import jax
import numpy as np
import yaml
from sklearn.metrics import balanced_accuracy_score, recall_score

from dim_red.analysis.metrics import embedding_quality_metrics
from dim_red.augmentation import make_supercell_for_radius
from dim_red.generate import GenerationConfig, generate_structures
from dim_red.pipeline.full_stack import FullStack
from dim_red.pipeline.run_layout import (
    EXPERT_NAMES,
    FAMILY,
    discover_full_stack_runs,
    resolve_heads_name,
)

logger = logging.getLogger("evaluate_holdout_pyxtal")

TEST_SEED = 20260925
STRUCTURES_PER_FAMILY = 500


# --------------------------------------------------------------------------
# Test-set generation / caching
# --------------------------------------------------------------------------


def _hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def test_generation_kwargs(pyxtal_config, seed: int, per_family: int) -> Dict[str, Any]:
    """``GenerationConfig`` kwargs for a run's test set: its family stack's
    ``data.pyxtal`` block (converted exactly like the stack's own dataset
    build does), but with a new seed and ``per_family`` structures per family,
    uniformly split over each family's space groups.
    """
    kwargs = dataclasses.asdict(pyxtal_config)
    kwargs.pop("seed", None)
    if kwargs.get("candidate_num_ions") is None:
        kwargs.pop("candidate_num_ions")
    kwargs.update(
        structures_per_spacegroup=None,
        structures_per_family=per_family,
        distribution="uniform",
        seed=seed,
    )
    return kwargs


class _GenerationFailureLog(logging.Handler):
    """Collects ``dim_red.generate``'s per-space-group failure warnings."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.failures: Dict[int, Dict[str, Any]] = {}

    def emit(self, record):
        if record.msg.startswith("Spacegroup %d (%s): failed"):
            sg, family, n_failed, n_requested = record.args
            self.failures[int(sg)] = dict(
                family=family, failed=int(n_failed), requested=int(n_requested)
            )


def get_or_generate_test_set(gen_kwargs: Dict[str, Any], cache_dir: Path):
    """Generated (pre-supercell) test structures + their metadata, cached."""
    key = _hash(gen_kwargs)
    path = cache_dir / f"testset_{key}.extxyz"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        logger.info("test set cache hit: %s", path)
        return key, ase.io.read(path, index=":"), json.loads(meta_path.read_text())

    logger.info("generating test set %s: %s", key, gen_kwargs)
    handler = _GenerationFailureLog()
    gen_logger = logging.getLogger("dim_red.generate")
    gen_logger.addHandler(handler)
    t0 = time.time()
    try:
        atoms_list = generate_structures(GenerationConfig(**gen_kwargs))
    finally:
        gen_logger.removeHandler(handler)
    per_family: Dict[str, int] = {}
    for a in atoms_list:
        per_family[a.info["family"]] = per_family.get(a.info["family"], 0) + 1
    meta = dict(
        key=key,
        generation=gen_kwargs,
        n_structures=len(atoms_list),
        per_family=dict(sorted(per_family.items())),
        failed_spacegroups=handler.failures,
        n_failed=int(sum(f["failed"] for f in handler.failures.values())),
        species=sorted({s for a in atoms_list for s in a.get_chemical_symbols()}),
        seconds=round(time.time() - t0, 1),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    ase.io.write(path, atoms_list, format="extxyz")
    meta_path.write_text(json.dumps(meta, indent=1))
    logger.info(
        "test set %s: %d structures, per family %s, %d failures in %d SGs",
        key,
        len(atoms_list),
        meta["per_family"],
        meta["n_failed"],
        len(handler.failures),
    )
    return key, atoms_list, meta


# --------------------------------------------------------------------------
# Saved val outputs of a stack's heads
# --------------------------------------------------------------------------


def _stack_classes(stack_dir: Path) -> list:
    with open(stack_dir / "classes.yaml") as f:
        return yaml.safe_load(f)["classes"]


def _val_predictions(heads_dir: Path):
    """``(probs, label_ids)`` of the stack's own val rows."""
    with np.load(heads_dir / "predictions.npz") as npz:
        val = npz["split"].astype(str) == "val"
        return npz["probs"][val], npz["label_ids"][val]


def _val_viz(heads_dir: Path):
    """``(viz embeddings, label_ids)`` of the stack's own val rows."""
    with np.load(heads_dir / "viz_embeddings.npz") as npz:
        val = npz["split"].astype(str) == "val"
        return npz["embeddings"][val], npz["label_ids"][val]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def family_metrics(true_family, pred_family, family_classes) -> Dict[str, Any]:
    recalls = recall_score(
        true_family, pred_family, labels=family_classes, average=None, zero_division=0
    )
    return dict(
        acc=float(np.mean(true_family == pred_family)),
        bal_acc=float(balanced_accuracy_score(true_family, pred_family)),
        recall=dict(zip(family_classes, map(float, recalls))),
    )


def sg_metrics(true_family, true_sg, oracle, e2e, families, vocab) -> Dict[str, Any]:
    """``oracle``/``e2e`` hold ``-1`` where no expert gave a prediction (counted
    as wrong in the overall accuracies); per-family oracle accuracy is only
    reported for families with a trained expert (``vocab``)."""
    per_family = {
        f: float(np.mean(oracle[true_family == f] == true_sg[true_family == f]))
        for f in families
        if f in vocab and np.any(true_family == f)
    }
    out_of_vocab = {
        f: int(np.sum(~np.isin(true_sg[true_family == f], vocab.get(f, []))))
        for f in families
    }
    return dict(
        sg_oracle=float(np.mean(oracle == true_sg)),
        sg_e2e=float(np.mean(e2e == true_sg)),
        sg_e2e_no_expert=int(np.sum(e2e < 0)),
        sg_oracle_per_family=per_family,
        sg_out_of_vocab=out_of_vocab,
    )


def silhouette(z, labels) -> Dict[str, float]:
    return {
        k: float(v)
        for k, v in embedding_quality_metrics(z, {"family": np.asarray(labels)}).items()
    }


# --------------------------------------------------------------------------
# Per-run evaluation
# --------------------------------------------------------------------------


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.cache_dir = Path(args.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.test_sets: Dict[str, Dict[str, Any]] = {}

    # -- test set for a run -------------------------------------------------

    def test_set_for(self, data_config):
        gen_kwargs = test_generation_kwargs(
            data_config.pyxtal, self.args.seed, self.args.per_family
        )
        key, atoms_raw, meta = get_or_generate_test_set(gen_kwargs, self.cache_dir)
        aug = data_config.augmentation
        radius = aug.supercell_radius if aug is not None else None
        test_id = f"{key}_sc{radius}"
        if test_id not in self.test_sets:
            atoms = (
                [make_supercell_for_radius(a, radius) for a in atoms_raw]
                if radius
                else atoms_raw
            )
            n_atoms = np.array([len(a) for a in atoms])
            self.test_sets[test_id] = dict(
                atoms=atoms,
                meta=dict(
                    meta,
                    supercell_radius=radius,
                    n_atoms_min=int(n_atoms.min()),
                    n_atoms_median=float(np.median(n_atoms)),
                    n_atoms_max=int(n_atoms.max()),
                ),
            )
        return test_id, self.test_sets[test_id]

    # -- one run --------------------------------------------------------------

    def evaluate(self, run_dir: Path) -> Dict[str, Any]:
        full_stack = FullStack.open(run_dir)
        trained = full_stack.stack_names()
        if FAMILY not in trained:
            raise ValueError(f"{run_dir}: family stack is not trained")
        family_dir = run_dir / "stacks" / FAMILY
        heads_name = resolve_heads_name(family_dir, self.args.heads_name)
        test_id, test = self.test_set_for(full_stack.config.stacks[FAMILY].data)
        atoms = test["atoms"]
        true_family = np.array([a.info["family"] for a in atoms])
        true_sg = np.array([int(a.info["spacegroup"]) for a in atoms])
        device = self.args.device

        prediction = full_stack.predict(
            atoms, heads_name=self.args.heads_name, device=device
        )
        result = dict(
            test_set=test_id,
            model_kind=full_stack.config.model_kind,
            heads_name=heads_name,
            trained_stacks=trained,
        )

        family_classes = [str(c) for c in _stack_classes(family_dir)]
        self._check_labels(run_dir, family_classes, true_family)
        pred_family = np.asarray(prediction.family)
        result["family_classes"] = family_classes
        result["test"] = family_metrics(true_family, pred_family, family_classes)
        heads_dir = family_dir / "heads" / heads_name
        val_probs, val_ids = _val_predictions(heads_dir)
        val_true = np.asarray(family_classes)[val_ids]
        val_pred = np.asarray(family_classes)[val_probs.argmax(axis=1)]
        result["val"] = family_metrics(val_true, val_pred, family_classes)

        experts = [n for n in trained if n in EXPERT_NAMES]
        if experts:
            sg = self._spacegroups(
                full_stack, experts, self.args.heads_name, atoms, true_family
            )
            e2e = np.array(
                [-1 if s is None else int(s) for s in prediction.spacegroup],
                dtype=np.int64,
            )
            oracle, vocab = sg.pop("oracle_pred"), sg.pop("vocab")
            sg.update(
                sg_metrics(true_family, true_sg, oracle, e2e, family_classes, vocab)
            )
            result["sg"] = sg

        result["test_viz"] = silhouette(prediction.viz_family, true_family)
        val_z, val_viz_ids = _val_viz(heads_dir)
        result["val_viz"] = silhouette(val_z, np.asarray(family_classes)[val_viz_ids])
        return result

    def _spacegroups(self, full_stack, experts, heads_name, atoms, true_family):
        """Oracle routing (true family -> its expert) plus each expert's own
        val accuracy. Families without a trained expert stay ``-1``."""
        oracle = np.full(len(atoms), -1, dtype=np.int64)
        vocab: Dict[str, List[int]] = {}
        val_acc: Dict[str, float] = {}
        n_val_correct = n_val = 0
        for name in experts:
            family = name.capitalize()
            stack_dir = full_stack.run_dir / "stacks" / name
            classes = [int(c) for c in _stack_classes(stack_dir)]
            vocab[family] = classes
            probs, ids = _val_predictions(
                stack_dir / "heads" / resolve_heads_name(stack_dir, heads_name)
            )
            correct = probs.argmax(axis=1) == ids
            val_acc[family] = float(correct.mean()) if len(ids) else float("nan")
            n_val_correct += int(correct.sum())
            n_val += len(ids)
            rows = np.flatnonzero(true_family == family)
            if not rows.size:
                continue
            stack_pred = full_stack.predict_stack(
                name,
                [atoms[i] for i in rows],
                heads_name=heads_name,
                device=self.args.device,
            )
            oracle[rows] = np.asarray(stack_pred.labels, dtype=np.int64)
        return dict(
            oracle_pred=oracle,
            vocab=vocab,
            val_sg_oracle=n_val_correct / n_val if n_val else None,
            val_sg_oracle_per_family=val_acc,
        )

    @staticmethod
    def _check_labels(run_dir, family_classes, true_family):
        unknown = sorted(set(true_family.tolist()) - set(family_classes))
        if unknown:
            raise ValueError(
                f"{run_dir}: test families {unknown} not in {family_classes}"
            )


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


FAMILIES = [
    "Cubic",
    "Hexagonal",
    "Monoclinic",
    "Orthorhombic",
    "Tetragonal",
    "Triclinic",
    "Trigonal",
]


def flat_row(run_dir: Path, res: Dict[str, Any]) -> Dict:
    test, val = res["test"], res["val"]
    sg = res.get("sg", {})
    row = dict(
        key=run_dir.name,
        heads_name=res["heads_name"],
        model_kind=res["model_kind"],
        run_dir=str(run_dir),
        test_set=res["test_set"],
        val_acc=val["acc"],
        test_acc=test["acc"],
        delta_acc=test["acc"] - val["acc"],
        val_bal_acc=val["bal_acc"],
        test_bal_acc=test["bal_acc"],
        val_sg_oracle=sg.get("val_sg_oracle"),
        test_sg_oracle=sg.get("sg_oracle"),
        test_sg_e2e=sg.get("sg_e2e"),
        test_sg_e2e_no_expert=sg.get("sg_e2e_no_expert"),
        val_silhouette=res["val_viz"].get("family_silhouette"),
        test_silhouette=res["test_viz"].get("family_silhouette"),
        test_viz_knn=res["test_viz"].get("family_knn_accuracy"),
    )
    for f in FAMILIES:
        row[f"val_recall_{f}"] = val["recall"].get(f)
        row[f"test_recall_{f}"] = test["recall"].get(f)
    for f in FAMILIES:
        row[f"val_sg_oracle_{f}"] = sg.get("val_sg_oracle_per_family", {}).get(f)
        row[f"test_sg_oracle_{f}"] = sg.get("sg_oracle_per_family", {}).get(f)
    row["sg_out_of_vocab"] = (
        sum(sg["sg_out_of_vocab"].values()) if "sg_out_of_vocab" in sg else None
    )
    return row


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- purely informational
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run-dir", nargs="*", default=[], help="FullStack run directories"
    )
    parser.add_argument(
        "--sweep-dir",
        nargs="*",
        default=[],
        help="directories whose immediate FullStack-run subdirectories are all evaluated",
    )
    parser.add_argument(
        "--heads-name",
        default=None,
        help="heads set to evaluate (default: the only one each stack has)",
    )
    parser.add_argument("--output-dir", default="runs/holdout_pyxtal")
    parser.add_argument("--cache-dir", default="runs/holdout_pyxtal/cache")
    parser.add_argument("--seed", type=int, default=TEST_SEED)
    parser.add_argument("--per-family", type=int, default=STRUCTURES_PER_FAMILY)
    parser.add_argument(
        "--device", default="cpu", help="jax backend for the params (cpu/gpu)"
    )
    parser.add_argument(
        "--only", nargs="*", default=None, help="substrings of run-dir names to run"
    )
    parser.add_argument(
        "--csv-copy", default=None, help="also copy the CSV to this path"
    )
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.run_dir]
    for sweep_dir in args.sweep_dir:
        run_dirs.extend(discover_full_stack_runs(sweep_dir))
    if not run_dirs:
        parser.error("give at least one --run-dir or --sweep-dir")
    run_dirs = [
        p for p in run_dirs if not args.only or any(o in p.name for o in args.only)
    ]

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("dim_red.generate").setLevel(logging.WARNING)
    logger.info("jax devices: %s", jax.devices())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator(args)
    rows, details, skipped = [], [], []
    for run_dir in run_dirs:
        t0 = time.time()
        logger.info("=== %s", run_dir)
        try:
            res = evaluator.evaluate(run_dir)
        except Exception as exc:  # noqa: BLE001 -- report and keep going
            logger.exception("%s failed", run_dir)
            skipped.append(dict(run_dir=str(run_dir), reason=repr(exc)))
            continue
        row = flat_row(run_dir, res)
        rows.append(row)
        details.append(dict(run_dir=str(run_dir), **res, row=row))
        logger.info(
            "%s: val %.4f -> test %.4f (bal %.4f) sg oracle %s e2e %s sil %s [%.0fs]",
            row["key"],
            row["val_acc"],
            row["test_acc"],
            row["test_bal_acc"],
            row["test_sg_oracle"],
            row["test_sg_e2e"],
            row["test_silhouette"],
            time.time() - t0,
        )

    csv_path = out_dir / "holdout_accuracy.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = dict(
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        git_commit=_git_commit(),
        seed=args.seed,
        structures_per_family=args.per_family,
        heads_name=args.heads_name,
        test_sets={k: v["meta"] for k, v in evaluator.test_sets.items()},
        skipped=skipped,
        models=details,
    )
    json_path = out_dir / "holdout_accuracy.json"
    json_path.write_text(json.dumps(payload, indent=1, default=str))
    if args.csv_copy:
        shutil.copy(csv_path, args.csv_copy)
    logger.info(
        "wrote %s and %s (%d runs, %d skipped)",
        csv_path,
        json_path,
        len(rows),
        len(skipped),
    )


if __name__ == "__main__":
    main()
