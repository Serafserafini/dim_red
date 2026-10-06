"""
Evaluate already-trained runs on a brand-new, never-seen pyxtal test set.

Every accuracy reported so far was measured on each run's own ``split ==
"val"`` rows. That split is drawn per *row* after augmentation, so a val row
is very often a jittered/vacancy copy of a structure whose other copies are in
``train`` -- the val numbers are optimistic. This script regenerates a fresh
pyxtal database with a different seed (default 20260925), matching each run's
own generation settings (``pyxtal:`` block of its ``config.yaml``: families,
``n_species``, ``species_pool``, ``candidate_num_ions``, ``factor``,
``max_count``) but with ``structures_per_family`` forced to 500, uniform, and
applies only the run's ``augmentation.supercell_radius`` step
(``dim_red.augmentation.make_supercell_for_radius``) -- no jitter, no
vacancies, no near-duplicate filtering. Nothing is retrained.

Per model it computes, on the test set:

* family accuracy / balanced accuracy / per-family recall, from the same head
  the benchmark used (a ``classification`` tail or a ``hierarchical_supcon``
  tail's own family stage);
* for runs with ``hierarchical_supcon`` tails: SG accuracy with oracle routing
  (true family -> that family's expert) and end-to-end (predicted family),
  plus per-family oracle SG accuracy;
* for runs with a visualization tail: the 2D family silhouette of the test
  points (``dim_red.analysis.metrics.embedding_quality_metrics``), next to the
  same metric recomputed on the run's own val rows.

The previously measured val numbers are read from
``experiments/val_accuracy_all_runs.csv`` (or ``--val-csv``).

Generated test sets are cached (``<cache-dir>/testset_<hash>.extxyz`` plus a
``.json`` with the generation settings, per-family counts and pyxtal
failures) and reused by every run with the same generation config; raw
featurizations (SOAP / MACE) of each test set are cached next to them.

Run with (from repo root, ``dmred`` active):
    python examples/evaluate_holdout_pyxtal.py \
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import ase.io
import jax
import numpy as np
import yaml
from flax import serialization
from sklearn.metrics import balanced_accuracy_score, recall_score

from dim_red.analysis.metrics import embedding_quality_metrics
from dim_red.augmentation import make_supercell_for_radius
from dim_red.generate import GenerationConfig, generate_structures
from dim_red.pipeline.config import load_run_config
from dim_red.pipeline.inference import encode_structures, load_trained_run
from dim_red.soap import compute_soap
from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.tails import ClassificationTail, ProjectionTail, VisualizationTail
from dim_red.utils import apply_standardization

logger = logging.getLogger("evaluate_holdout_pyxtal")

TEST_SEED = 20260925
STRUCTURES_PER_FAMILY = 500
VAL_CSV = Path("experiments/val_accuracy_all_runs.csv")
N_CROSSCHECK = 64  # test structures re-encoded through encode_structures
MACE_GROUP_SIZE = 250

R1 = "experiments/runs/tuning_supcon_family_only_round1/20260908-1/"
R7_REF = (
    "experiments/runs/tuning_supcon_family_only_round7/20260914-1/"
    "model-supcon_hd-128-64_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp1_"
    "supcon-family_only_tau0.05_lf1-3"
)
R15 = (
    "experiments/runs/experiment_pipeline_{name}/model-supcon_hd-{enc}_pyxtal-"
    "cub-hex-mon-ort-tet-tri-tri_nsp1_supcon-family_only_tau0.05_lf1"
)
SUPCON_MACE = (
    "runs/model-supcon_mace_hd-256-128_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp1_"
    "supcon-family_only_tau0.05_lf1"
)
TAIL_REPLACES_PROJ = (
    "experiments/runs/experiment_tail_replaces_projection/model-supcon_hd-128-64_"
    "pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp1_supcon-family_only_tau0.05_lf1"
)

R19 = (
    "runs/round19/model-supcon_mace_hd-{enc}_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp1_"
    "supcon-family_only_tau0.05_lf1{suffix}"
)
R19_VARIANTS = {  # variant -> (encoder, run-dir suffix); 3500 structures/family
    "baseline": ("256-128", "-2"),
    "viz_wider": ("256-128", "-3"),
    "projection_wider": ("256-128", "-4"),
    "projection_deeper": ("256-128", "-5"),
    "viz_deeper": ("256-128", "-6"),
    "classifier_wider": ("256-128", "-7"),
    "classifier_deeper": ("256-128", "-8"),
    "body_deeper": ("256-128-64", ""),
    "body_wider": ("512-256", ""),
}

CLS = "tail:classification"


@dataclass
class ModelSpec:
    """One row of the output table.

    ``kind``: ``classification`` (a family classification tail) or
    ``hierarchical_supcon``. ``viz``: a ``tails/`` subdir holding a
    visualization tail, or ``"projection"`` for a run whose 2D
    ``projection_params.msgpack`` is its visualization.
    """

    key: str
    round: str
    run_dir: str
    kind: str
    tail: Optional[str] = None
    viz: Optional[str] = None
    val_source: str = CLS
    note: str = ""


def _r15(key, name, enc, clf="classification", hier=None, viz=None, note=""):
    run = R15.format(name=name, enc=enc)
    specs = [ModelSpec(key, "R15", run, "classification", clf, viz, CLS, note)]
    if hier:
        specs.append(
            ModelSpec(
                f"{key} +hier_supcon",
                "R15",
                run,
                "hierarchical_supcon",
                hier,
                None,
                f"tail:{hier}",
                note,
            )
        )
    return specs


MODELS: List[ModelSpec] = [
    ModelSpec(
        "R1 SupCon cos t0.1 [128,64] rnd",
        "R1",
        R1 + "model-supcon_hd-128-64_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp2_"
        "supcon-family_only_tau0.1_lf1-3",
        "classification",
        "classification",
    ),
    ModelSpec(
        "R5 SupCon cos t0.05 [128,64] bal",
        "R5",
        "experiments/runs/tuning_supcon_family_only_round5/20260914-1/"
        "model-supcon_hd-128-64_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp1_"
        "supcon-family_only_tau0.05_lf1-4",
        "classification",
        "classification",
    ),
    ModelSpec(
        "R7 SupCon ref cos t0.05 [128,64] rnd",
        "R7",
        R7_REF,
        "classification",
        "classification",
        "visualization_round8_topmodel_family_only_euclidean",
    ),
    *[
        ModelSpec(
            f"R7 ref +{sub[len('hierarchical_'):]}",
            rnd,
            R7_REF,
            kind,
            sub,
            val_source=f"tail:{sub}",
        )
        for rnd, kind, sub in [
            ("R13", "hierarchical_supcon", "hierarchical_r13_supcon_sg"),
            ("R14", "hierarchical_supcon", "hierarchical_r14_supcon_sg_latent32_BEST"),
        ]
    ],
    ModelSpec(
        "R8 SupCon cos t0.05 [128,64] rnd",
        "R8",
        "experiments/runs/tuning_supcon_family_only_round8/20260915-1/"
        "model-supcon_hd-128-64_pyxtal-cub-hex-mon-ort-tet-tri-tri_nsp1_"
        "supcon-family_only_tau0.05_lf1-3",
        "classification",
        "classification",
        note="trained on per-SG generation (15/SG); tested on per-family 500/family",
    ),
    *_r15(
        "R15 dims_8_4",
        "dims_8_4",
        "128-64",
        hier="hierarchical_supcon_dims_8_4",
        viz="visualization_family_only_euclidean",
    ),
    *_r15(
        "R15 dims_8_2",
        "dims_8_2",
        "128-64",
        hier="hierarchical_supcon_dims_8_2",
        viz="visualization_family_only_euclidean",
    ),
    *_r15(
        "R15 dims_32_16",
        "dims_32_16",
        "128-64",
        hier="hierarchical_supcon_dims_32_16",
        viz="visualization_family_only_euclidean",
    ),
    ModelSpec(
        "R15 dims_32_16 clf32",
        "R15",
        R15.format(name="dims_32_16", enc="128-64"),
        "classification",
        "classification_clf32",
        "visualization_viz64_32",
        "tail:classification_clf32",
        "viz column = visualization_viz64_32",
    ),
    ModelSpec(
        "R15 dims_32_16 +hier_supcon clf32",
        "R15",
        R15.format(name="dims_32_16", enc="128-64"),
        "hierarchical_supcon",
        "hierarchical_supcon_dims32_16_clf32",
        None,
        "tail:hierarchical_supcon_dims32_16_clf32",
    ),
    ModelSpec(
        "R15 dims_32_16 +hier_supcon viz64_32",
        "R15",
        R15.format(name="dims_32_16", enc="128-64"),
        "hierarchical_supcon",
        "hierarchical_supcon_dims32_16_viz64_32",
        None,
        "tail:hierarchical_supcon_dims32_16_viz64_32",
    ),
    *_r15(
        "R15 encoder_256_128",
        "encoder_256_128",
        "256-128",
        hier="hierarchical_supcon_encoder_256_128",
        viz="visualization_family_only_euclidean",
    ),
    *_r15(
        "R15 best_combo",
        "best_combo",
        "256-128",
        hier="hierarchical_supcon_best_combo",
        viz="visualization_family_only_euclidean",
    ),
    ModelSpec(
        "R18 supcon_mace [256,128]",
        "R18",
        SUPCON_MACE,
        "classification",
        "classification",
        "visualization",
    ),
    ModelSpec(
        "R18 supcon_mace +hier_supcon",
        "R18",
        SUPCON_MACE,
        "hierarchical_supcon",
        "hierarchical_supcon_supcon_mace_best_combo_scale",
        None,
        "tail:hierarchical_supcon_supcon_mace_best_combo_scale",
    ),
    *[
        spec
        for variant, (enc, suffix) in R19_VARIANTS.items()
        for spec in (
            ModelSpec(
                f"R19 supcon_mace {variant}",
                "R19",
                R19.format(enc=enc, suffix=suffix),
                "classification",
                "classification",
                "visualization",
            ),
            ModelSpec(
                f"R19 supcon_mace {variant} +hier_supcon",
                "R19",
                R19.format(enc=enc, suffix=suffix),
                "hierarchical_supcon",
                f"hierarchical_supcon_round19_{variant}",
                None,
                f"tail:hierarchical_supcon_round19_{variant}",
            ),
        )
    ],
    ModelSpec(
        "TailReplacesProj A-viz (proj_dim 2)",
        "TailReplacesProj",
        TAIL_REPLACES_PROJ,
        "classification",
        "classification",
        "projection",
    ),
]


# --------------------------------------------------------------------------
# Test-set generation / caching
# --------------------------------------------------------------------------


def _hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def test_generation_kwargs(run_config, seed: int, per_family: int) -> Dict[str, Any]:
    """``GenerationConfig`` kwargs for a run's test set: its own ``pyxtal:``
    block (converted exactly like ``dataset_cache.get_or_build_pyxtal_dataset``
    does), but with a new seed and ``per_family`` structures per family,
    uniformly split over each family's space groups.
    """
    kwargs = dataclasses.asdict(run_config.pyxtal)
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
# Featurization (cached per test set) -- same recipe as
# dim_red.pipeline.inference.encode_structures, computed once and shared
# across every run with the same featurizer settings.
# --------------------------------------------------------------------------


class FeatureCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.memory: Dict[str, np.ndarray] = {}

    def get(self, name: str, payload: Dict[str, Any], compute):
        key = f"{name}_{_hash(payload)}"
        if key in self.memory:
            return self.memory[key]
        path = self.cache_dir / f"features_{key}.npy"
        if path.exists():
            arr = np.load(path)
        else:
            t0 = time.time()
            arr = np.asarray(compute(), dtype=np.float32)
            np.save(path, arr)
            logger.info(
                "computed %s features %s in %.0fs", name, arr.shape, time.time() - t0
            )
        self.memory[key] = arr
        return arr


def raw_soap(atoms_list, soap_config, species, testset_id, cache: FeatureCache):
    kwargs = dict(soap_config.as_kwargs())
    kwargs["species"] = list(species)
    kwargs["average"] = "outer"

    def compute():
        return np.asarray(compute_soap(atoms_list, **kwargs)).reshape(
            len(atoms_list), -1
        )

    return cache.get("soap", dict(testset=testset_id, **kwargs), compute)


def raw_mace(atoms_list, mace_config, testset_id, cache: FeatureCache):
    def compute():
        from dim_red.mace.model import MaceEncoder

        # MaceEncoder pads every chunk of one encode() call to the largest
        # single structure's edge count; one huge supercell then blows up an
        # 8 GB GPU. Encoding size-sorted groups separately keeps each call's
        # padding local (same per-structure output, one JIT shape per group).
        encoder = MaceEncoder(**mace_config.mace_kwargs())
        order = np.argsort([len(a) for a in atoms_list], kind="stable")
        out = [None] * len(atoms_list)
        for start in range(0, len(order), MACE_GROUP_SIZE):
            group = order[start : start + MACE_GROUP_SIZE]
            pooled = encoder.encode([atoms_list[i] for i in group])
            for i, vec in zip(group, pooled):
                out[i] = vec
        return np.stack(out)

    return cache.get(
        "mace", dict(testset=testset_id, **mace_config.mace_kwargs()), compute
    )


# --------------------------------------------------------------------------
# Model loading helpers
# --------------------------------------------------------------------------


def _load_params(wrapper, path: Path):
    with open(path, "rb") as f:
        wrapper.params = serialization.from_bytes(wrapper.params, f.read())
    return wrapper


def load_run(run_dir: Path, shadow_root: Path):
    """``load_trained_run``, working around runs whose ``dataset.extxyz`` was
    removed (the full-size MACE runs): for ``supcon_mace``,
    ``load_trained_run`` never reads that file -- it only checks it exists --
    so an empty placeholder in a symlinked shadow copy of the run dir is
    enough (the run dir itself is left untouched).
    """
    if (run_dir / "dataset.extxyz").exists():
        return load_trained_run(run_dir)
    config = load_run_config(run_dir / "config.yaml")
    if config.model_kind != "supcon_mace":
        raise FileNotFoundError(f"{run_dir} has no dataset.extxyz")
    shadow = shadow_root / run_dir.name
    shadow.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "model_params.msgpack", "embeddings.npz"):
        link = shadow / name
        if not link.exists():
            link.symlink_to((run_dir / name).resolve())
    (shadow / "dataset.extxyz").touch()
    logger.info(
        "%s has no dataset.extxyz -- loading through placeholder shadow dir %s",
        run_dir,
        shadow,
    )
    return load_trained_run(shadow)


def _tail_yaml(tail_dir: Path) -> Dict[str, Any]:
    with open(tail_dir / "tail_config.yaml") as f:
        return yaml.safe_load(f)


def _npz_classes(tail_dir: Path, key: str = "family_classes") -> List[str]:
    with np.load(tail_dir / "tail_predictions.npz", allow_pickle=True) as npz:
        return [str(c) for c in npz[key].tolist()]


def _expert_status(tail_dir: Path) -> Dict[str, Dict[str, Any]]:
    with open(tail_dir / "family_expert_status.yaml") as f:
        return {e["family"]: e for e in yaml.safe_load(f)["families"]}


def _local_classes(expert_dir: Path) -> List[int]:
    with open(expert_dir / "local_spacegroup_classes.yaml") as f:
        return [int(c) for c in yaml.safe_load(f)["local_spacegroup_classes"]]


def _softmax(logits) -> np.ndarray:
    return np.asarray(jax.nn.softmax(np.asarray(logits), axis=-1))


def load_classification_tail(
    path: Path, input_dim: int, hidden_dim, n_classes: int
) -> ClassificationTail:
    """Rebuild a ``ClassificationTail`` and load its saved ``tail_params``."""
    tail = ClassificationTail(
        input_dim=input_dim, hidden_dim=hidden_dim, n_classes=n_classes
    )
    tail.load_params_bytes(Path(path).read_bytes())
    return tail


def load_viz_tail(tail_dir: Path, input_dim: int) -> VisualizationTail:
    viz_cfg = _tail_yaml(tail_dir)["visualization"]
    return _load_params(
        VisualizationTail(
            input_dim=input_dim,
            hidden_dim=viz_cfg.get("hidden_dim") or [input_dim],
            output_dim=viz_cfg["viz_dim"],
        ),
        tail_dir / "tail_params.msgpack",
    )


# --------------------------------------------------------------------------
# SG prediction (oracle routing + end-to-end) for hierarchical_supcon tails
# --------------------------------------------------------------------------


def _route_sg(families, true_family, pred_family, status, expert_fn):
    """SG predictions routed by true (oracle) and predicted (e2e) family.

    ``expert_fn(family, rows) -> local SG predictions`` for a family that has
    an expert; families without one fall back to their recorded majority SG.
    """
    n = len(true_family)
    oracle = np.zeros(n, dtype=np.int64)
    e2e = np.zeros(n, dtype=np.int64)
    for family in families:
        for routed, out in ((true_family, oracle), (pred_family, e2e)):
            rows = np.flatnonzero(routed == family)
            if not rows.size:
                continue
            st = status[family]
            if not st["expert"]:
                out[rows] = int(st["fallback_spacegroup"])
            else:
                out[rows] = expert_fn(family, rows)
    return oracle, e2e


def hierarchical_supcon_sg(tail_dir, families, x_native, true_family, pred_family):
    """``hierarchical_supcon`` tail: per-family SupCon SG body + classifier,
    on native standardized features."""
    hs = _tail_yaml(tail_dir)["hierarchical_supcon"]
    status = _expert_status(tail_dir)
    vocab: Dict[str, List[int]] = {}
    cache: Dict[str, Any] = {}

    def expert_fn(family, rows):
        if family not in cache:
            expert_dir = tail_dir / "sg_experts" / family
            vocab[family] = _local_classes(expert_dir)
            body = _load_params(
                SupConEncoder(
                    input_dim=x_native.shape[1],
                    encoder_hidden_dim=hs["sg_encoder_hidden_dim"],
                    latent_dim=hs["sg_latent_dim"],
                ),
                expert_dir / "sg_body_params.msgpack",
            )
            classifier = load_classification_tail(
                expert_dir / "classifier_tail_params.msgpack",
                input_dim=hs["sg_latent_dim"],
                hidden_dim=hs["sg_classifier_hidden_dim"],
                n_classes=len(vocab[family]),
            )
            cache[family] = (body, classifier)
        body, classifier = cache[family]
        logits = np.asarray(classifier.classify(body.encode(x_native[rows])))
        return np.asarray(vocab[family])[logits.argmax(axis=1)]

    oracle, e2e = _route_sg(families, true_family, pred_family, status, expert_fn)
    for family in families:
        if status[family]["expert"] and family not in vocab:
            vocab[family] = _local_classes(tail_dir / "sg_experts" / family)
    return oracle, e2e, vocab


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
    per_family = {
        f: float(np.mean(oracle[true_family == f] == true_sg[true_family == f]))
        for f in families
        if np.any(true_family == f)
    }
    out_of_vocab = {
        f: int(np.sum(~np.isin(true_sg[true_family == f], vocab.get(f, []))))
        for f in families
    }
    return dict(
        sg_oracle=float(np.mean(oracle == true_sg)),
        sg_e2e=float(np.mean(e2e == true_sg)),
        sg_oracle_per_family=per_family,
        sg_out_of_vocab=out_of_vocab,
    )


def silhouette(z, labels) -> Dict[str, float]:
    return {
        k: float(v)
        for k, v in embedding_quality_metrics(z, {"family": np.asarray(labels)}).items()
    }


def val_family_recall(probs, labels, split, family_classes) -> Dict[str, float]:
    val = split == "val"
    pred = np.asarray(family_classes)[probs[val].argmax(axis=1)]
    return family_metrics(labels[val], pred, family_classes)["recall"]


# --------------------------------------------------------------------------
# Per-model evaluation
# --------------------------------------------------------------------------


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.cache_dir = Path(args.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.features = FeatureCache(self.cache_dir)
        self.loaded_runs: Dict[str, Any] = {}
        self.test_sets: Dict[str, Dict[str, Any]] = {}
        self.crosschecks: Dict[str, float] = {}

    # -- test set for a run -------------------------------------------------

    def test_set_for(self, run_config):
        gen_kwargs = test_generation_kwargs(
            run_config, self.args.seed, self.args.per_family
        )
        key, atoms_raw, meta = get_or_generate_test_set(gen_kwargs, self.cache_dir)
        aug = run_config.augmentation
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

    # -- representations ------------------------------------------------------

    def load(self, run_dir: Path):
        key = str(run_dir)
        if key not in self.loaded_runs:
            self.loaded_runs[key] = load_run(run_dir, self.cache_dir / "shadow_runs")
        return self.loaded_runs[key]

    def representations(self, loaded, atoms, test_id):
        """``(r, x_native)``: body embedding and native standardized input
        features (SOAP or MACE) of the test structures."""
        kind = loaded.config.model_kind
        if kind == "supcon_mace":
            raw = raw_mace(atoms, loaded.config.mace, test_id, self.features)
        else:
            missing = sorted(
                {s for a in atoms for s in a.get_chemical_symbols()}
                - set(loaded.species)
            )
            if missing:
                raise ValueError(
                    f"test species {missing} not in {loaded.run_dir}'s species "
                    f"{loaded.species}"
                )
            raw = raw_soap(
                atoms, loaded.config.soap, loaded.species, test_id, self.features
            )
        x_std = apply_standardization(raw, loaded.feature_mean, loaded.feature_std)
        r = np.asarray(loaded.model.encode(x_std))
        self._crosscheck(loaded, atoms, r)
        return r, x_std

    def _crosscheck(self, loaded, atoms, r):
        """The cached-feature path must match ``encode_structures`` exactly."""
        key = str(loaded.run_dir)
        if key in self.crosschecks:
            return
        r_ref = encode_structures(loaded, atoms[:N_CROSSCHECK])
        # Relative to the embedding's own scale: float32 noise in SOAP
        # components with a tiny training std gets amplified by
        # standardization, so wide (e.g. 12960-dim) inputs differ by ~1e-2.
        diff = float(np.max(np.abs(r_ref - r[:N_CROSSCHECK])))
        diff /= float(np.max(np.abs(r_ref))) or 1.0
        self.crosschecks[key] = diff
        logger.info("encode_structures cross-check %s: rel max|diff|=%.2e", key, diff)
        if diff > 1e-2:
            raise RuntimeError(
                f"cached features disagree with encode_structures: {diff}"
            )

    # -- one spec -------------------------------------------------------------

    def evaluate(self, spec: ModelSpec) -> Dict[str, Any]:
        run_dir = Path(spec.run_dir)
        loaded = self.load(run_dir)
        test_id, test = self.test_set_for(loaded.config)
        atoms = test["atoms"]
        true_family = np.array([a.info["family"] for a in atoms])
        true_sg = np.array([int(a.info["spacegroup"]) for a in atoms])
        r, x_std = self.representations(loaded, atoms, test_id)
        emb = loaded.embeddings
        result = dict(test_set=test_id, model_kind=loaded.config.model_kind)

        tail_dir = run_dir / "tails" / spec.tail
        family_classes = _npz_classes(tail_dir)
        cfg = _tail_yaml(tail_dir)
        if spec.kind == "classification":
            tail = load_classification_tail(
                tail_dir / "tail_params.msgpack",
                input_dim=r.shape[1],
                hidden_dim=cfg["classification"]["head_hidden_dim"],
                n_classes=len(family_classes),
            )
        else:
            tail = load_classification_tail(
                tail_dir / "family" / "tail_params.msgpack",
                input_dim=r.shape[1],
                hidden_dim=cfg[spec.kind]["head_hidden_dim"],
                n_classes=len(family_classes),
            )
        family_probs = _softmax(tail.classify(r))
        with np.load(tail_dir / "tail_predictions.npz", allow_pickle=True) as npz:
            val_probs = npz["family_probs"]
            val_labels = npz["labels"].astype(str)
            val_split = npz["split"].astype(str)

        self._check_labels(spec, family_classes, true_family)
        pred_family = np.asarray(family_classes)[family_probs.argmax(axis=1)]
        result["family_classes"] = family_classes
        result["test"] = family_metrics(true_family, pred_family, family_classes)
        result["val_recall"] = val_family_recall(
            val_probs, val_labels, val_split, family_classes
        )

        sg = None
        if spec.kind == "hierarchical_supcon":
            oracle, e2e, vocab = hierarchical_supcon_sg(
                run_dir / "tails" / spec.tail,
                family_classes,
                x_std,
                true_family,
                pred_family,
            )
            sg = sg_metrics(true_family, true_sg, oracle, e2e, family_classes, vocab)
        if sg is not None:
            result["sg"] = sg

        if spec.viz:
            if spec.viz == "projection":
                viz = _load_params(
                    ProjectionTail(
                        input_dim=r.shape[1],
                        hidden_dim=loaded.config.supcon.projection_hidden_dim
                        or [r.shape[1]],
                        projection_dim=loaded.config.supcon.projection_dim,
                    ),
                    run_dir / "projection_params.msgpack",
                )
            else:
                viz = load_viz_tail(run_dir / "tails" / spec.viz, r.shape[1])
            result["test_viz"] = silhouette(np.asarray(viz.project(r)), true_family)
            val_rows = emb["split"].astype(str) == "val"
            result["val_viz"] = silhouette(
                np.asarray(viz.project(emb["embeddings"][val_rows])),
                emb["labels"].astype(str)[val_rows],
            )
        return result

    @staticmethod
    def _check_labels(spec, family_classes, true_family):
        unknown = sorted(set(true_family.tolist()) - set(family_classes))
        if unknown:
            raise ValueError(
                f"{spec.key}: test families {unknown} not in {family_classes}"
            )
        if list(family_classes) != sorted(family_classes):
            raise ValueError(f"{spec.key}: family_classes not sorted: {family_classes}")


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def load_val_table(path: Path = VAL_CSV) -> Dict[tuple, Dict[str, str]]:
    with open(path) as f:
        return {(row["run"], row["source"]): row for row in csv.DictReader(f)}


def _float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


FAMILIES = [
    "Cubic",
    "Hexagonal",
    "Monoclinic",
    "Orthorhombic",
    "Tetragonal",
    "Triclinic",
    "Trigonal",
]


def flat_row(spec: ModelSpec, res: Dict[str, Any], val: Dict[str, str]) -> Dict:
    test = res["test"]
    sg = res.get("sg", {})
    val_acc = _float(val.get("acc_val"))
    row = dict(
        key=spec.key,
        round=spec.round,
        model_kind=res["model_kind"],
        family_source=spec.tail or spec.kind,
        viz_tail=spec.viz or "",
        run_dir=spec.run_dir,
        test_set=res["test_set"],
        val_acc=val_acc,
        test_acc=test["acc"],
        delta_acc=None if val_acc is None else test["acc"] - val_acc,
        val_bal_acc=_float(val.get("bal_acc_val")),
        test_bal_acc=test["bal_acc"],
        val_sg_oracle=_float(val.get("sgo_val")),
        test_sg_oracle=sg.get("sg_oracle"),
        val_sg_e2e=_float(val.get("sg_e2e_val")),
        test_sg_e2e=sg.get("sg_e2e"),
        val_silhouette=res.get("val_viz", {}).get("family_silhouette"),
        test_silhouette=res.get("test_viz", {}).get("family_silhouette"),
        test_viz_knn=res.get("test_viz", {}).get("family_knn_accuracy"),
    )
    for f in FAMILIES:
        row[f"val_recall_{f}"] = res["val_recall"].get(f)
        row[f"test_recall_{f}"] = test["recall"].get(f)
    for f in FAMILIES:
        row[f"test_sg_oracle_{f}"] = sg.get("sg_oracle_per_family", {}).get(f)
    row["sg_out_of_vocab"] = (
        sum(sg["sg_out_of_vocab"].values()) if "sg_out_of_vocab" in sg else None
    )
    row["note"] = spec.note
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
    parser.add_argument("--output-dir", default="runs/holdout_pyxtal")
    parser.add_argument("--cache-dir", default="runs/holdout_pyxtal/cache")
    parser.add_argument("--seed", type=int, default=TEST_SEED)
    parser.add_argument("--per-family", type=int, default=STRUCTURES_PER_FAMILY)
    parser.add_argument(
        "--only", nargs="*", default=None, help="substrings of model keys to run"
    )
    parser.add_argument(
        "--val-csv",
        default=str(VAL_CSV),
        help="val-accuracy table (same columns as the default) to read val numbers from",
    )
    parser.add_argument(
        "--csv-copy", default=None, help="also copy the CSV to this path"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("dim_red.generate").setLevel(logging.WARNING)
    logger.info("jax devices: %s", jax.devices())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [s for s in MODELS if not args.only or any(o in s.key for o in args.only)]
    val_table = load_val_table(Path(args.val_csv))
    evaluator = Evaluator(args)
    rows, details, skipped = [], [], []
    for spec in specs:
        t0 = time.time()
        logger.info("=== %s", spec.key)
        try:
            res = evaluator.evaluate(spec)
        except Exception as exc:  # noqa: BLE001 -- report and keep going
            logger.exception("%s failed", spec.key)
            skipped.append(dict(key=spec.key, run_dir=spec.run_dir, reason=repr(exc)))
            continue
        val = val_table.get((spec.run_dir, spec.val_source), {})
        if not val:
            logger.warning("%s: no val row in %s", spec.key, args.val_csv)
        row = flat_row(spec, res, val)
        rows.append(row)
        details.append(dict(spec=dataclasses.asdict(spec), val_csv=val, **res, row=row))
        logger.info(
            "%s: val %s -> test %.4f (bal %.4f) sg oracle %s e2e %s sil %s [%.0fs]",
            spec.key,
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
        test_sets={k: v["meta"] for k, v in evaluator.test_sets.items()},
        encode_structures_crosscheck_max_abs_diff=evaluator.crosschecks,
        skipped=skipped,
        models=details,
    )
    json_path = out_dir / "holdout_accuracy.json"
    json_path.write_text(json.dumps(payload, indent=1, default=str))
    if args.csv_copy:
        shutil.copy(csv_path, args.csv_copy)
    logger.info(
        "wrote %s and %s (%d models, %d skipped)",
        csv_path,
        json_path,
        len(rows),
        len(skipped),
    )


if __name__ == "__main__":
    main()
