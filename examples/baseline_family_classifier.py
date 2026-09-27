"""
Simple, fast sklearn baselines (logistic regression + a small MLP) for
crystal-family classification -- a "floor" to measure any dim_red model
against before spending time tuning SupCon/CGCNN/MACE architectures.

Motivation: every family-classification number produced so far by
``dim_red.pipeline`` sits on top of a real trained encoder (SupCon body,
autoencoder, CGCNN, ...), so a disappointing held-out accuracy is ambiguous
-- is the *encoder* the bottleneck, or is the task itself hard given SOAP/
MACE features (see ``dim_red.pipeline.single_run``'s grouped train/val split
fix and the held-out generalization gap documented in
``examples/evaluate_holdout_pyxtal.py``)? A plain linear/shallow-MLP
classifier directly on SOAP-averaged (or, with ``--include-mace``, frozen
MACE-pooled) features answers that: if a cheap linear model already gets
most of the way to what a heavily-tuned SupCon body gets on the same held-out
test set, the bottleneck is the features/task, not encoder capacity or
hyperparameters -- see the repo-root ``CLAUDE.md``/session notes on why
tuning restarted from scratch.

Does *not* import jax/flax/dim_red.pipeline.single_run at all (deliberately
-- this script's whole point is to be a fast, dependency-light sanity check
you can run before touching any of the JAX training code), so its train/val
split is a small, self-contained reimplementation of
``dim_red.pipeline.single_run._split_indices_grouped``'s grouping-by-
``material_id`` logic (same rationale: augmented copies of a structure share
``material_id``, so splitting per-row would leak near-duplicates across
train/val -- see that function's docstring for the full explanation).

Reports accuracy on:
  * train (fit) and val (held out from the *same* pyxtal generation, grouped
    split) -- comparable to what ``dimred-benchmark`` reports for a trained
    model on the same dataset;
  * a genuinely fresh test set, generated with a different seed and no
    augmentation at all -- the same spirit as
    ``examples/evaluate_holdout_pyxtal.py``'s held-out set, the honest
    "structures the model/script has never touched" number.

Run with (from repo root, ``dmred`` active):
    python examples/baseline_family_classifier.py \\
        --families Cubic Hexagonal Monoclinic Orthorhombic Tetragonal Trigonal Triclinic \\
        --structures-per-family 300 --output baseline_results.csv

Add ``--include-mace --mace-checkpoint <path>`` to also fit the same two
classifiers on frozen MACE-pooled embeddings (requires ``mace_jax`` --
optional, see ``src/dim_red/mace/CLAUDE.md``).
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms

from dim_red.augmentation import AugmentationConfig, augment_structures
from dim_red.soap import compute_soap
from dim_red.utils import apply_standardization, fit_standardization

logger = logging.getLogger("baseline_family_classifier")

_ALL_FAMILIES = [
    "Cubic",
    "Hexagonal",
    "Monoclinic",
    "Orthorhombic",
    "Tetragonal",
    "Trigonal",
    "Triclinic",
]


def _split_indices_grouped(
    material_ids: Sequence[str], val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Standalone copy of
    ``dim_red.pipeline.single_run._split_indices_grouped`` (numpy-only, no
    jax/flax import needed for this script) -- see that function's docstring
    for why grouping by ``material_id`` matters whenever augmentation is
    involved. Degenerates to a plain per-row split when every ``material_id``
    is already unique (e.g. ``--no-augment``).
    """
    n_samples = len(material_ids)
    material_ids_arr = np.asarray(material_ids)
    unique_groups, inverse = np.unique(material_ids_arr, return_inverse=True)
    n_groups = unique_groups.shape[0]
    rng = np.random.default_rng(seed)
    if n_groups == n_samples:
        n_val = max(1, int(round(n_samples * val_ratio)))
        n_val = min(n_val, n_samples - 1)
        indices = rng.permutation(n_samples)
        return indices[n_val:], indices[:n_val]

    n_val_groups = max(1, int(round(n_groups * val_ratio)))
    n_val_groups = min(n_val_groups, n_groups - 1)
    shuffled_group_order = rng.permutation(n_groups)
    val_group_ids = shuffled_group_order[:n_val_groups]
    val_mask = np.isin(inverse, val_group_ids)
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _generate(
    families: Sequence[str],
    structures_per_family: int,
    n_species: int,
    seed: int,
) -> List[Atoms]:
    # Lazy import: pyxtal isn't a hard dim_red dependency.
    from dim_red.generate import GenerationConfig, generate_structures

    return generate_structures(
        GenerationConfig(
            families=list(families),
            structures_per_family=structures_per_family,
            n_species=n_species,
            seed=seed,
        )
    )


def _soap_features(
    atoms_list: List[Atoms], soap_kwargs: Dict, species: List[str]
) -> np.ndarray:
    """``species`` must be the SAME list for the train/val set and the test
    set (resolved once from the train/val structures, see ``build_split``) --
    dscribe's SOAP descriptor width depends on how many species it's told to
    expect, and pyxtal's per-structure placeholder-species sampling means the
    species actually *present* can easily differ between two independently
    generated structure sets, which would otherwise silently produce
    differently-shaped (and therefore incomparable/unstandardizable) feature
    matrices for train/val vs. test.
    """
    kwargs = dict(soap_kwargs)
    kwargs["species"] = species
    kwargs["average"] = "outer"
    vectors = compute_soap(atoms_list, **kwargs)
    return np.asarray(vectors).reshape(len(atoms_list), -1)


def _mace_features(
    atoms_list: List[Atoms], species: List[str], checkpoint_path: str, r_max: float
) -> np.ndarray:
    # species unused -- MACE embeds atomic numbers directly, no fixed species
    # vocabulary/width to keep consistent across train/val vs. test.
    # Lazy import: mace_jax isn't a hard dim_red dependency.
    from dim_red.mace.model import MaceEncoder

    encoder = MaceEncoder(checkpoint_path=checkpoint_path, r_max=r_max, pooling="mean")
    return np.asarray(encoder.encode(atoms_list), dtype=np.float64)


def build_split(
    args: argparse.Namespace,
    featurize,
) -> DatasetSplit:
    """Generates train+val structures (optionally augmented) and a separate,
    never-augmented test set from a different seed, featurizes both with
    ``featurize`` (a ``List[Atoms] -> np.ndarray`` callable, either SOAP or
    MACE), fits standardization on train+val only, and applies it to test.
    """
    trainval_atoms = _generate(
        args.families, args.structures_per_family, args.n_species, args.seed
    )
    if args.augment:
        aug_config = AugmentationConfig(
            n_augmented=args.n_augmented,
            jitter_probability=0.5,
            jitter_std=0.05,
            vacancy_probability=0.5 if args.vacancy else 0.0,
            vacancy_atom_probability=0.05,
            seed=args.seed,
        )
        trainval_atoms = augment_structures(trainval_atoms, aug_config)
        material_ids = [a.info["material_id"] for a in trainval_atoms]
    else:
        material_ids = [a.info["material_id"] for a in trainval_atoms]
    labels_trainval = np.array([a.info["family"] for a in trainval_atoms])

    test_atoms = _generate(
        args.families, args.test_structures_per_family, args.n_species, args.test_seed
    )
    labels_test = np.array([a.info["family"] for a in test_atoms])

    # Species resolved from train/val only, then reused unchanged for test --
    # same assumption dim_red.pipeline.inference.encode_structures makes
    # (every species in a newly-applied structure must already be in the
    # training species list, or dscribe's SOAP computation raises). At this
    # script's default scale (hundreds of structures per family, drawn from
    # a ~20-element placeholder pool with n_species=1) coverage is
    # essentially guaranteed; at very small --structures-per-family a test
    # structure can draw a species train never saw, and this raises instead
    # of silently producing mismatched feature widths.
    species = sorted({s for a in trainval_atoms for s in a.get_chemical_symbols()})
    X_trainval = featurize(trainval_atoms, species)
    X_test = featurize(test_atoms, species)

    mean, std = fit_standardization(X_trainval)
    X_trainval = apply_standardization(X_trainval, mean, std)
    X_test = apply_standardization(X_test, mean, std)

    train_idx, val_idx = _split_indices_grouped(material_ids, args.val_ratio, args.seed)
    return DatasetSplit(
        X_train=X_trainval[train_idx],
        X_val=X_trainval[val_idx],
        y_train=labels_trainval[train_idx],
        y_val=labels_trainval[val_idx],
        X_test=X_test,
        y_test=labels_test,
    )


def evaluate_classifier(name: str, clf, split: DatasetSplit) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    clf.fit(split.X_train, split.y_train)
    row = {"classifier": name}
    for subset_name, X, y in (
        ("train", split.X_train, split.y_train),
        ("val", split.X_val, split.y_val),
        ("test", split.X_test, split.y_test),
    ):
        pred = clf.predict(X)
        row[f"{subset_name}_acc"] = accuracy_score(y, pred)
        row[f"{subset_name}_balanced_acc"] = balanced_accuracy_score(y, pred)
    return row


def run_baselines(feature_name: str, split: DatasetSplit) -> List[Dict[str, float]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0),
        "mlp_64": MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=500, early_stopping=True, random_state=0
        ),
    }
    rows = []
    for name, clf in classifiers.items():
        logger.info("Fitting %s on %s features", name, feature_name)
        row = evaluate_classifier(name, clf, split)
        row["features"] = feature_name
        rows.append(row)
        logger.info(
            "%s/%s: train_acc=%.4f val_acc=%.4f test_acc=%.4f",
            feature_name,
            name,
            row["train_acc"],
            row["val_acc"],
            row["test_acc"],
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", default=_ALL_FAMILIES)
    parser.add_argument("--structures-per-family", type=int, default=300)
    parser.add_argument(
        "--test-structures-per-family",
        type=int,
        default=100,
        help="Held-out test set size per family (different seed, never augmented).",
    )
    parser.add_argument("--n-species", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-seed",
        type=int,
        default=None,
        help="Defaults to --seed + 999983 (an arbitrary large prime offset) so "
        "the test set is never accidentally the same as the train/val one.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--augment", action="store_true", help="Apply jitter (+vacancy) augmentation."
    )
    parser.add_argument("--n-augmented", type=int, default=3)
    parser.add_argument("--vacancy", action="store_true")
    parser.add_argument("--r-cut", type=float, default=5.0)
    parser.add_argument("--n-max", type=int, default=4)
    parser.add_argument("--l-max", type=int, default=3)
    parser.add_argument("--include-mace", action="store_true")
    parser.add_argument("--mace-checkpoint", type=str, default=None)
    parser.add_argument("--mace-r-max", type=float, default=6.0)
    parser.add_argument("--output", type=str, default="baseline_results.csv")
    args = parser.parse_args()
    if args.test_seed is None:
        args.test_seed = args.seed + 999983

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    soap_kwargs = dict(r_cut=args.r_cut, n_max=args.n_max, l_max=args.l_max)
    logger.info("Building SOAP baseline dataset (families=%s)", args.families)
    soap_split = build_split(
        args, lambda atoms, species: _soap_features(atoms, soap_kwargs, species)
    )
    rows = run_baselines("soap", soap_split)

    if args.include_mace:
        if not args.mace_checkpoint:
            raise SystemExit("--include-mace requires --mace-checkpoint <path>")
        logger.info("Building MACE baseline dataset")
        mace_split = build_split(
            args,
            lambda atoms, species: _mace_features(
                atoms, species, args.mace_checkpoint, args.mace_r_max
            ),
        )
        rows.extend(run_baselines("mace", mace_split))

    output_path = Path(args.output)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d row(s) to %s", len(rows), output_path)

    print("\n=== Baseline family-classification accuracy ===")
    header = f"{'features':<8} {'classifier':<20} {'train':>8} {'val':>8} {'test':>8}"
    print(header)
    for row in rows:
        print(
            f"{row['features']:<8} {row['classifier']:<20} "
            f"{row['train_acc']:>8.4f} {row['val_acc']:>8.4f} {row['test_acc']:>8.4f}"
        )


if __name__ == "__main__":
    main()
