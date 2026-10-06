# SingleStack / FullStack Core Implementation Plan (Plan A of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `SingleStack` (encoder + projection → classification + visualization tails) and `FullStack` (1 family stack + up to 7 independent expert stacks, each with its own pyxtal dataset and config), verified against golden references taken from the current training code.

**Architecture:** `SingleStack` (`src/dim_red/supcon/stack.py`) is a thin, I/O-free wrapper over the existing primitives (`training_first_phase`, `train_classification_tail`, `train_visualization_tail`). `FullStack` (`src/dim_red/pipeline/full_stack.py`) builds one dataset per stack (existing `dataset_cache`), splits/trains/saves each stack independently into `stacks/<name>/`. Nothing existing is removed or rewired in this plan: the old pipeline keeps working untouched.

**Tech Stack:** Python, JAX/Flax/Optax (existing primitives), numpy, PyYAML, pytest. Run everything inside the `dmred` conda env (`conda activate dmred`; check `$CONDA_DEFAULT_ENV`).

**Spec:** `docs/superpowers/specs/2026-10-06-singlestack-fullstack-design.md` (this plan implements spec steps 0, 1, 2 and the `fit_body`/`fit_heads`/selection/errors part of `FullStack`). **Plan B** (written after this plan lands, because it must reread `inference.py`, `compare.py`, `benchmark.py`, `cli.py`, `sweep.py`) covers: `FullStack.predict` and feature handling for new structures, the legacy-layout reader, wiring `dimred-run/sweep/train-tail` to `FullStack`, removing the old supcon training paths, moving `_classifier_eval_plots`, migrating configs/scripts/notebooks/docs.

## Global Constraints

- Only `data_source: pyxtal` is supported by `FullStack` (spec D3). `model_kind` ∈ `supcon`, `supcon_mace`; `cgcnn` is untouched.
- Never run the full `pytest` suite on your own initiative; run only the test files named in each task (project rule). The `slow` end-to-end test only runs with `--runslow`.
- Do not `pip install` anything; everything needed is in `dmred`.
- `dim_red/pipeline/config.py` is **not modified** and stays importable without jax. The new `full_stack_config.py` imports `dim_red.supcon.stack` and therefore needs jax; only `FullStack` imports it.
- New public symbols in `dim_red.supcon` follow the lazy `__getattr__` pattern in `supcon/__init__.py`.
- The three training routines and `ProjectionTail`/`VisualizationTail` are **not modified** (spec D11).
- Every stack uses its own seed: `seed = run_seed + index`, index = position in `("family", "triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic")`; the pyxtal generation seed defaults to the stack seed (spec: independent datasets).
- Role decides the label (spec D7): `family` stack → crystal-system name from `atoms.info["family"]`; expert stacks → spacegroup number. Every stack contrasts/classifies/visualizes on that single label.
- Commits: one per task, message ends with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Pre-commit runs black/isort; if it reformats files, `git add` them again and re-run the commit.

## Review Focus

Inputs/conditions the spec implies but the happy-path tests do not exercise; each is pinned by a test in the owning task.

1. A class present only in validation (n_classes larger than the largest train label) must not crash `fit_heads` — Task 1 (`test_fit_heads_accepts_val_only_class`).
2. Early stopping enabled (histories shorter than `epochs`) must work through `fit_body` and `fit_heads` — Task 1 (`test_early_stopping_shortens_histories`).
3. `viz_dim: 3` must train and save without trying to draw a 2D plot — Task 4 (`test_fit_heads_viz_dim_3`).
4. A typo in an expert name (`cubik`) must fail fast with the valid names listed, before any dataset is built — Task 2.
5. A stack with too few rows or a single class must fail with the stack's name in the message and leave nothing half-written on disk — Task 4.
6. Re-running `fit_body` on an already trained stack, or `fit_heads` with an existing heads name, must raise instead of overwriting — Task 4.

## File Structure

| File | Responsibility |
|---|---|
| `tests/golden/__init__.py`, `golden_spec.py`, `make_golden.py`, `data/*` | Synthetic inputs, shared constants, script that runs the OLD code and stores references |
| `src/dim_red/supcon/stack.py` | `Batching`, `StackConfig`, `SingleStack`, config (de)serialization |
| `src/dim_red/supcon/__init__.py` | Lazy exports for the new symbols |
| `src/dim_red/pipeline/full_stack_config.py` | YAML → `FullStackConfig`/`StackSpec` (defaults merge, expert restriction, validation) |
| `src/dim_red/pipeline/dataset_cache.py` | New public `resolve_augmentation` (the old private function delegates to it) |
| `src/dim_red/pipeline/stack_data.py` | `StackDataset`, `build_stack_dataset` (one dataset per stack) |
| `src/dim_red/pipeline/full_stack.py` | `FullStack`: create/open, `fit_body`, `fit_heads`, layout writer |
| `tests/conftest.py` | `--runslow` option and `slow` marker |
| tests | `test_golden_reference.py`, `test_supcon_stack.py`, `test_supcon_stack_golden.py`, `test_pipeline_full_stack_config.py`, `test_pipeline_stack_data.py`, `test_pipeline_full_stack.py`, `test_pipeline_full_stack_e2e.py` |

---

### Task 0: Golden references from the current code

Must run **before** any other task, while the old code is untouched.

**Files:**
- Create: `tests/golden/__init__.py` (empty), `tests/golden/golden_spec.py`, `tests/golden/make_golden.py`
- Create (generated): `tests/golden/data/*`
- Test: `tests/test_golden_reference.py`

**Interfaces:**
- Produces: `tests.golden.golden_spec` constants (`GOLDEN_DIR`, `SEED`, `FAMILIES`, `SPACEGROUPS`, `PER_SPACEGROUP`, `VAL_RATIO`, `N_FEATURES`, `ENCODER_HIDDEN`, `LATENT`, `PROJ_DIM`, `TAU`, `DISTANCE`, `BODY_EPOCHS`, `BATCH`, `HEAD_EPOCHS`, `CLF_HIDDEN`, `VIZ_HIDDEN`, `VIZ_TAU`, `VIZ_DISTANCE`, `EXPERT_CLF_HIDDEN`, `EXPERT_VIZ_HIDDEN`, `MIN_SAMPLES_PER_EXPERT`), `make_inputs()`, `FAMILY_FILES`, `expert_files(family)`; the files in `tests/golden/data/` consumed by Task 1.

- [ ] **Step 1: Write `tests/golden/golden_spec.py`**

```python
"""Constants and synthetic inputs shared by ``make_golden.py`` (which runs the
OLD training code once) and the tests that compare ``SingleStack`` to its output."""

from pathlib import Path

import numpy as np

GOLDEN_DIR = Path(__file__).parent / "data"

SEED = 0
N_FEATURES = 12
FAMILIES = ["Alpha", "Beta", "Gamma"]
SPACEGROUPS = {"Alpha": [1, 2], "Beta": [3, 4], "Gamma": [5, 6]}
PER_SPACEGROUP = 10  # 3 families x 2 spacegroups x 10 = 60 structures
VAL_RATIO = 0.25

ENCODER_HIDDEN = [8]
LATENT = 4
PROJ_DIM = 6
TAU = 0.05
DISTANCE = "cosine"
BODY_EPOCHS = 3
BATCH = 8
HEAD_EPOCHS = 2
CLF_HIDDEN = 8
VIZ_HIDDEN = [6]
VIZ_TAU = 0.1
VIZ_DISTANCE = "euclidean"
EXPERT_CLF_HIDDEN = 4
EXPERT_VIZ_HIDDEN = [4]
MIN_SAMPLES_PER_EXPERT = 5

FAMILY_FILES = [
    "inputs.npz",
    "family_body_params.msgpack",
    "family_projection_params.msgpack",
    "family_body_loss_history.csv",
    "family_embeddings.npz",
    "family_classifier_params.msgpack",
    "family_classifier_loss_history.csv",
    "family_viz_params.msgpack",
    "family_viz_loss_history.csv",
]


def expert_files(family: str) -> list:
    return [
        f"expert_{family}_{suffix}"
        for suffix in (
            "body_params.msgpack",
            "projection_params.msgpack",
            "classifier_params.msgpack",
            "viz_params.msgpack",
            "body_loss_history.csv",
            "classifier_loss_history.csv",
            "viz_loss_history.csv",
            "local_classes.yaml",
        )
    ]


def make_inputs():
    """Deterministic synthetic dataset: returns ``(X, labels, material_ids,
    spacegroups)`` with ``X`` float32 of shape ``(60, N_FEATURES)``."""
    rng = np.random.default_rng(1234)
    X, labels, material_ids, spacegroups = [], [], [], []
    for family in FAMILIES:
        for sg in SPACEGROUPS[family]:
            center = rng.normal(size=N_FEATURES) * 2.0
            for i in range(PER_SPACEGROUP):
                X.append(center + rng.normal(size=N_FEATURES) * 0.5)
                labels.append(family)
                material_ids.append(f"g-{sg}-{i}")
                spacegroups.append(sg)
    return np.asarray(X, dtype=np.float32), labels, material_ids, spacegroups
```

- [ ] **Step 2: Write `tests/golden/make_golden.py`**

```python
"""Regenerate the golden references. Run ONCE, against the pre-FullStack code:

    PYTHONPATH=src:. python tests/golden/make_golden.py

Runs the OLD pipeline (``run_single`` + ``train_tail`` classification,
visualization and hierarchical_supcon) on the synthetic dataset of
``golden_spec.make_inputs`` -- dataset builder and SOAP patched so the inputs
are exactly that dataset -- and copies the resulting parameters, loss
histories and embeddings into ``tests/golden/data/``."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.io import write

from dim_red.pipeline.config import (
    ClassificationTailConfig,
    EncoderConfig,
    HierarchicalSupconTailConfig,
    PyxtalConfig,
    RunConfig,
    SupConConfig,
    TailTrainConfig,
    TailTrainSettings,
    TrainSettings,
    VisualizationTailConfig,
)
from dim_red.pipeline.single_run import run_single
from dim_red.pipeline.tail_training import train_tail
from tests.golden import golden_spec as g


def main() -> None:
    X, labels, material_ids, spacegroups = g.make_inputs()
    work = Path(tempfile.mkdtemp(prefix="golden_"))

    structures_path = work / "structures.extxyz"
    atoms = []
    for material_id, sg in zip(material_ids, spacegroups):
        a = Atoms("Cu", positions=[[0.0, 0.0, 0.0]])
        a.info["material_id"] = material_id
        a.info["spacegroup"] = sg
        atoms.append(a)
    write(str(structures_path), atoms, format="extxyz")

    n_features = X.shape[1]
    fake_dataset = (
        X,
        labels,
        material_ids,
        spacegroups,
        structures_path,
        np.zeros(n_features, dtype=np.float32),  # mean 0, std 1: the
        np.ones(n_features, dtype=np.float32),  # standardized features are X
    )

    run_config = RunConfig(
        data_source="pyxtal",
        pyxtal=PyxtalConfig(structures_per_spacegroup=1),
        encoder=EncoderConfig(encoder_hidden_dim=g.ENCODER_HIDDEN, latent_dim=g.LATENT),
        train=TrainSettings(
            epochs=g.BODY_EPOCHS, batch_size=g.BATCH, val_ratio=g.VAL_RATIO
        ),
        supcon=SupConConfig(
            mode="family_only",
            tau=g.TAU,
            distance=g.DISTANCE,
            projection_dim=g.PROJ_DIM,
        ),
        seed=g.SEED,
        output_dir=str(work / "runs"),
        model_kind="supcon",
    )
    head_train = TailTrainSettings(epochs=g.HEAD_EPOCHS, batch_size=g.BATCH)

    with (
        patch(
            "dim_red.pipeline.single_run.build_dataset_for_run",
            return_value=fake_dataset,
        ),
        patch("dim_red.pipeline.tail_training.compute_soap", return_value=X),
    ):
        run_dir = run_single(run_config)
        clf_dir = train_tail(
            TailTrainConfig(
                run_dir=str(run_dir),
                tail_kind="classification",
                classification=ClassificationTailConfig(head_hidden_dim=g.CLF_HIDDEN),
                train=head_train,
                seed=g.SEED,
            )
        )
        viz_dir = train_tail(
            TailTrainConfig(
                run_dir=str(run_dir),
                tail_kind="visualization",
                visualization=VisualizationTailConfig(
                    viz_dim=2,
                    mode="family_only",
                    tau=g.VIZ_TAU,
                    distance=g.VIZ_DISTANCE,
                    hidden_dim=g.VIZ_HIDDEN,
                ),
                train=head_train,
                seed=g.SEED,
            )
        )
        hier_dir = train_tail(
            TailTrainConfig(
                run_dir=str(run_dir),
                tail_kind="hierarchical_supcon",
                hierarchical_supcon=HierarchicalSupconTailConfig(
                    head_hidden_dim=g.CLF_HIDDEN,
                    min_samples_per_expert=g.MIN_SAMPLES_PER_EXPERT,
                    sg_encoder_hidden_dim=g.ENCODER_HIDDEN,
                    sg_latent_dim=g.LATENT,
                    sg_tau=g.TAU,
                    sg_distance=g.DISTANCE,
                    sg_projection_dim=g.PROJ_DIM,
                    sg_classifier_hidden_dim=g.EXPERT_CLF_HIDDEN,
                    sg_visualization_hidden_dim=g.EXPERT_VIZ_HIDDEN,
                    sg_visualization_hidden_dim_by_family={},
                    sg_visualization_tau=g.VIZ_TAU,
                    sg_visualization_distance=g.VIZ_DISTANCE,
                ),
                train=head_train,
                seed=g.SEED,
            )
        )

    embeddings = np.load(run_dir / "embeddings.npz", allow_pickle=True)
    split = embeddings["split"]
    labels_arr = np.asarray(labels)
    for family in g.FAMILIES:
        in_family = labels_arr == family
        assert (in_family & (split == "val")).any(), f"{family} has no val rows"
        assert (in_family & (split == "train")).any(), f"{family} has no train rows"

    out = g.GOLDEN_DIR
    out.mkdir(parents=True, exist_ok=True)
    copies = {
        run_dir / "model_params.msgpack": "family_body_params.msgpack",
        run_dir / "projection_params.msgpack": "family_projection_params.msgpack",
        run_dir / "loss_history.csv": "family_body_loss_history.csv",
        run_dir / "embeddings.npz": "family_embeddings.npz",
        clf_dir / "tail_params.msgpack": "family_classifier_params.msgpack",
        clf_dir / "loss_history.csv": "family_classifier_loss_history.csv",
        viz_dir / "tail_params.msgpack": "family_viz_params.msgpack",
        viz_dir / "loss_history.csv": "family_viz_loss_history.csv",
    }
    for family in g.FAMILIES:
        sg_dir = hier_dir / "sg_experts" / family
        for src, dst in (
            ("sg_body_params.msgpack", "body_params.msgpack"),
            ("sg_projection_params.msgpack", "projection_params.msgpack"),
            ("classifier_tail_params.msgpack", "classifier_params.msgpack"),
            ("visualization_tail_params.msgpack", "viz_params.msgpack"),
            ("sg_body_loss_history.csv", "body_loss_history.csv"),
            ("classifier_loss_history.csv", "classifier_loss_history.csv"),
            ("visualization_loss_history.csv", "viz_loss_history.csv"),
            ("local_spacegroup_classes.yaml", "local_classes.yaml"),
        ):
            copies[sg_dir / src] = f"expert_{family}_{dst}"
    for src, dst in copies.items():
        shutil.copy(src, out / dst)

    np.savez(
        out / "inputs.npz",
        X=X,
        labels=np.asarray(labels),
        material_ids=np.asarray(material_ids),
        spacegroups=np.asarray(spacegroups, dtype=np.int64),
    )
    print(f"Golden references written to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the failing test `tests/test_golden_reference.py`**

```python
"""Guards the golden references: they must exist and be consistent with the
synthetic inputs (regenerate with tests/golden/make_golden.py)."""

import numpy as np

from tests.golden import golden_spec as g


def test_golden_files_exist():
    for name in g.FAMILY_FILES:
        assert (g.GOLDEN_DIR / name).exists(), name
    for family in g.FAMILIES:
        for name in g.expert_files(family):
            assert (g.GOLDEN_DIR / name).exists(), name


def test_golden_inputs_match_spec_and_split_covers_every_family():
    inputs = np.load(g.GOLDEN_DIR / "inputs.npz")
    n = len(g.FAMILIES) * 2 * g.PER_SPACEGROUP
    assert inputs["X"].shape == (n, g.N_FEATURES)
    X, *_ = g.make_inputs()
    np.testing.assert_array_equal(inputs["X"], X)

    split = np.load(g.GOLDEN_DIR / "family_embeddings.npz", allow_pickle=True)["split"]
    assert split.shape == (n,)
    labels = inputs["labels"]
    for family in g.FAMILIES:
        for side in ("train", "val"):
            assert ((labels == family) & (split == side)).any()
```

- [ ] **Step 4: Run it to verify it fails**

Run: `pytest tests/test_golden_reference.py -v`
Expected: FAIL (`tests/golden/data` does not exist yet; the first test fails on the missing files).

- [ ] **Step 5: Generate the golden data with the OLD code**

Run: `PYTHONPATH=src:. python tests/golden/make_golden.py`
Expected: prints `Golden references written to .../tests/golden/data`. If the script fails on a patch target or a config field, fix the script (not the pipeline) — the target names are the ones used by `run_single` (`build_dataset_for_run` imported at module level) and `tail_training.compute_soap`.

- [ ] **Step 6: Run the test to verify it passes**

Run: `pytest tests/test_golden_reference.py -v`
Expected: 2 passed.

- [ ] **Step 7: Check the data files are not git-ignored, then commit**

```bash
git check-ignore -v tests/golden/data/inputs.npz || echo "not ignored"
git add tests/golden tests/test_golden_reference.py
git commit -m "Add golden references from the current supcon training code

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 1: `SingleStack`

**Files:**
- Create: `src/dim_red/supcon/stack.py`
- Modify: `src/dim_red/supcon/__init__.py`
- Test: `tests/test_supcon_stack.py`, `tests/test_supcon_stack_golden.py`

**Interfaces:**
- Consumes: `SupConEncoder`, `ProjectionTail`, `ClassificationTail`, `VisualizationTail`, `TrainConfig`, `training_first_phase`, `train_classification_tail`, `train_visualization_tail`, `FeatureDatabase`; Task 0 golden data.
- Produces (exact):
  - `Batching(strategy="random", P=None, K=None, S=None)` (frozen dataclass)
  - `StackConfig(encoder_hidden_dim, latent_dim, projection_dim, body_train: TrainConfig, classifier_train: TrainConfig, viz_train: TrainConfig, projection_hidden_dim=None, classifier_hidden_dim=16, viz_hidden_dim=None, viz_dim=2, lambda_norm=0.0, viz_lambda_norm=0.0, body_batching=Batching(), viz_batching=Batching(), seed=42)`
  - `stack_config_to_dict(config) -> dict`, `stack_config_from_dict(d) -> StackConfig`
  - `SingleStack(input_dim, config)` with `.encoder`, `.projection`, `.classifier`, `.visualizer`, `.config`, `.n_classes`;
    `fit_body(X_train, X_val, y_train, y_val, sub_train=None) -> history`;
    `fit_heads(X_train, X_val, y_train, y_val, n_classes, sub_train=None, config=None) -> {"classifier": history, "visualization": history}`;
    `encode(X) -> np.ndarray`, `predict_proba(X) -> np.ndarray`, `visualize(X) -> np.ndarray`;
    `save_body(dir)`, `SingleStack.load_body(dir)`, `save_heads(dir)`, `load_heads(dir)`.

- [ ] **Step 1: Write the failing unit tests `tests/test_supcon_stack.py`**

```python
import numpy as np
import pytest

pytest.importorskip("jax")

import jax

from dim_red.dataset import FeatureDatabase
from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.stack import (
    Batching,
    SingleStack,
    StackConfig,
    stack_config_from_dict,
    stack_config_to_dict,
)
from dim_red.supcon.tails import ProjectionTail
from dim_red.supcon.training import TrainConfig, training_first_phase


def _toy(n=40, n_features=6, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % n_classes
    centers = rng.normal(size=(n_classes, n_features)) * 3.0
    X = (centers[y] + rng.normal(size=(n, n_features)) * 0.3).astype(np.float32)
    idx = rng.permutation(n)
    tr, va = idx[: int(n * 0.75)], idx[int(n * 0.75) :]
    return X, y, tr, va


def _config(**over):
    def train(**kw):
        return TrainConfig(
            epochs=kw.pop("epochs", 3), batch_size=8, learning_rate=1e-3, seed=0, **kw
        )

    base = dict(
        encoder_hidden_dim=[8],
        latent_dim=4,
        projection_dim=5,
        body_train=train(tau=0.05, distance="cosine"),
        classifier_train=train(),
        viz_train=train(tau=0.1, distance="euclidean"),
        classifier_hidden_dim=6,
        viz_hidden_dim=[5],
        viz_dim=2,
        seed=0,
    )
    base.update(over)
    return StackConfig(**base)


def _params_equal(a, b):
    return all(
        np.array_equal(x, y)
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


def test_fit_body_equals_direct_training_first_phase_call():
    X, y, tr, va = _toy()
    cfg = _config()
    stack = SingleStack(X.shape[1], cfg)
    history = stack.fit_body(X[tr], X[va], y[tr], y[va])

    encoder = SupConEncoder(X.shape[1], cfg.encoder_hidden_dim, cfg.latent_dim, seed=0)
    projection = ProjectionTail(cfg.latent_dim, [cfg.latent_dim], cfg.projection_dim, seed=0)
    expected = training_first_phase(
        encoder,
        projection,
        FeatureDatabase.from_array(X[tr]),
        FeatureDatabase.from_array(X[va]),
        cfg.body_train,
        train_family_ids=y[tr],
        val_family_ids=y[va],
        lambda_family=1.0,
        lambda_spacegroup=0.0,
    )
    assert _params_equal(stack.encoder.params, encoder.params)
    assert _params_equal(stack.projection.params, projection.params)
    assert history["train_loss"] == expected["train_loss"]


def test_fit_heads_leaves_encoder_and_projection_untouched():
    X, y, tr, va = _toy()
    stack = SingleStack(X.shape[1], _config())
    stack.fit_body(X[tr], X[va], y[tr], y[va])
    before_enc = [np.array(p) for p in jax.tree_util.tree_leaves(stack.encoder.params)]
    before_proj = [np.array(p) for p in jax.tree_util.tree_leaves(stack.projection.params)]
    histories = stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=3)
    after_enc = jax.tree_util.tree_leaves(stack.encoder.params)
    after_proj = jax.tree_util.tree_leaves(stack.projection.params)
    assert all(np.array_equal(a, b) for a, b in zip(before_enc, after_enc))
    assert all(np.array_equal(a, b) for a, b in zip(before_proj, after_proj))
    assert set(histories) == {"classifier", "visualization"}
    assert len(histories["classifier"]["train_loss"]) == 3
    assert stack.visualize(X).shape == (len(X), 2)
    probs = stack.predict_proba(X)
    assert probs.shape == (len(X), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_fit_heads_accepts_val_only_class():
    X, y, tr, va = _toy(n=40, n_classes=3)
    y = y.copy()
    y[va[0]] = 3  # class 3 appears only in validation
    stack = SingleStack(X.shape[1], _config())
    stack.fit_body(X[tr], X[va], y[tr], y[va])
    stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=4)
    assert stack.predict_proba(X).shape[1] == 4


def test_early_stopping_shortens_histories():
    X, y, tr, va = _toy()
    es = dict(early_stopping=True, early_stopping_patience=1, early_stopping_min_delta=1e9)
    cfg = _config(
        body_train=TrainConfig(epochs=8, batch_size=8, tau=0.05, distance="cosine", seed=0, **es),
        classifier_train=TrainConfig(epochs=8, batch_size=8, seed=0, **es),
        viz_train=TrainConfig(epochs=8, batch_size=8, tau=0.1, seed=0, **es),
    )
    stack = SingleStack(X.shape[1], cfg)
    body = stack.fit_body(X[tr], X[va], y[tr], y[va])
    heads = stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=3)
    assert len(body["train_loss"]) < 8
    assert len(heads["classifier"]["train_loss"]) < 8
    assert len(heads["visualization"]["train_loss"]) < 8


def test_balanced_batching_requires_sub_labels():
    X, y, tr, va = _toy()
    cfg = _config(body_batching=Batching(strategy="balanced", K=4))
    stack = SingleStack(X.shape[1], cfg)
    with pytest.raises(ValueError, match="sub_train"):
        stack.fit_body(X[tr], X[va], y[tr], y[va])


def test_heads_can_be_trained_with_a_different_config():
    X, y, tr, va = _toy()
    stack = SingleStack(X.shape[1], _config())
    stack.fit_body(X[tr], X[va], y[tr], y[va])
    other = _config(classifier_hidden_dim=[7, 3], viz_hidden_dim=[4], viz_dim=3)
    stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=3, config=other)
    assert stack.visualize(X).shape == (len(X), 3)


def test_predict_before_heads_raises():
    X, y, tr, va = _toy()
    stack = SingleStack(X.shape[1], _config())
    with pytest.raises(RuntimeError, match="heads"):
        stack.predict_proba(X)


def test_config_dict_roundtrip():
    cfg = _config(
        body_batching=Batching(strategy="balanced", P=2, K=4, S=None),
        projection_hidden_dim=[3],
    )
    assert stack_config_from_dict(stack_config_to_dict(cfg)) == cfg


def test_save_and_load_roundtrip(tmp_path):
    X, y, tr, va = _toy()
    stack = SingleStack(X.shape[1], _config())
    stack.fit_body(X[tr], X[va], y[tr], y[va])
    stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=3)
    stack.save_body(tmp_path / "body")
    stack.save_heads(tmp_path / "heads")

    loaded = SingleStack.load_body(tmp_path / "body")
    loaded.load_heads(tmp_path / "heads")
    np.testing.assert_array_equal(loaded.encode(X), stack.encode(X))
    np.testing.assert_array_equal(loaded.predict_proba(X), stack.predict_proba(X))
    np.testing.assert_array_equal(loaded.visualize(X), stack.visualize(X))
    assert loaded.n_classes == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_supcon_stack.py -v`
Expected: FAIL with `ModuleNotFoundError: dim_red.supcon.stack`.

- [ ] **Step 3: Implement `src/dim_red/supcon/stack.py`**

```python
"""``SingleStack``: one encoder + projection tail (trained jointly with the
SupCon loss, phase 1 = ``fit_body``) and one classification tail + one
visualization tail (trained on the frozen encoder's output, phase 2 =
``fit_heads``). The same object serves the family level and every
per-crystal-system expert -- only the data, the label and the config differ.

Arrays and models only: no file layout, no plots, no pipeline imports. The
label is a single integer array: it is the contrastive label of the body, the
class of the classifier and the contrastive label of the visualization tail.
"""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import yaml
from flax import serialization

from dim_red.dataset import FeatureDatabase
from dim_red.supcon.model import SupConEncoder
from dim_red.supcon.tail_training import (
    train_classification_tail,
    train_visualization_tail,
)
from dim_red.supcon.tails import ClassificationTail, ProjectionTail, VisualizationTail
from dim_red.supcon.training import TrainConfig, training_first_phase

History = Dict[str, List[float]]


@dataclass(frozen=True)
class Batching:
    """How training batches are formed (same meaning as
    ``training_first_phase``'s ``batching_*`` arguments)."""

    strategy: str = "random"
    P: Optional[int] = None
    K: Optional[int] = None
    S: Optional[int] = None


@dataclass(frozen=True)
class StackConfig:
    """Everything one ``SingleStack`` needs. ``body_train``/``classifier_train``/
    ``viz_train`` are the loop settings of the three training stages; ``tau``/
    ``distance`` of ``body_train`` and ``viz_train`` are the contrastive
    temperature/similarity of the body and of the visualization tail
    (``classifier_train``'s are ignored by the cross-entropy loss)."""

    encoder_hidden_dim: Sequence[int]
    latent_dim: int
    projection_dim: int
    body_train: TrainConfig
    classifier_train: TrainConfig
    viz_train: TrainConfig
    projection_hidden_dim: Optional[Sequence[int]] = None
    classifier_hidden_dim: Union[int, Sequence[int]] = 16
    viz_hidden_dim: Optional[Sequence[int]] = None
    viz_dim: int = 2
    lambda_norm: float = 0.0
    viz_lambda_norm: float = 0.0
    body_batching: Batching = field(default_factory=Batching)
    viz_batching: Batching = field(default_factory=Batching)
    seed: int = 42


def stack_config_to_dict(config: StackConfig) -> dict:
    """JSON/YAML-friendly dict (lists instead of tuples)."""
    return json.loads(json.dumps(dataclasses.asdict(config)))


def stack_config_from_dict(d: dict) -> StackConfig:
    d = dict(d)
    for key in ("body_train", "classifier_train", "viz_train"):
        d[key] = TrainConfig(**d[key])
    for key in ("body_batching", "viz_batching"):
        d[key] = Batching(**d.get(key, {}))
    return StackConfig(**d)


def _batching_kwargs(batching: Batching, y_train, sub_train) -> dict:
    balanced = batching.strategy == "balanced"
    if balanced and sub_train is None:
        raise ValueError(
            "balanced batching needs sub_train (secondary labels, e.g. "
            "spacegroup ids, used to stratify the batches)"
        )
    return dict(
        batching_strategy=batching.strategy,
        batching_family_ids=np.asarray(y_train) if balanced else None,
        batching_spacegroup_ids=np.asarray(sub_train) if balanced else None,
        batching_P=batching.P,
        batching_K=batching.K,
        batching_S=batching.S,
    )


def _write_params(path: Path, params) -> None:
    path.write_bytes(serialization.to_bytes(params))


def _read_params(path: Path, template):
    return serialization.from_bytes(template, path.read_bytes())


class SingleStack:
    """Encoder + projection tail (``fit_body``), then classification and
    visualization tails on the frozen encoder's output (``fit_heads``)."""

    def __init__(self, input_dim: int, config: StackConfig):
        self.input_dim = int(input_dim)
        self.config = config
        self.encoder = SupConEncoder(
            input_dim=self.input_dim,
            encoder_hidden_dim=config.encoder_hidden_dim,
            latent_dim=config.latent_dim,
            seed=config.seed,
        )
        self.projection = ProjectionTail(
            input_dim=config.latent_dim,
            hidden_dim=config.projection_hidden_dim or [config.latent_dim],
            projection_dim=config.projection_dim,
            seed=config.seed,
        )
        self.classifier: Optional[ClassificationTail] = None
        self.visualizer: Optional[VisualizationTail] = None
        self.n_classes: Optional[int] = None
        self._heads_config: Optional[StackConfig] = None

    # -- phase 1 ---------------------------------------------------------
    def fit_body(self, X_train, X_val, y_train, y_val, sub_train=None) -> History:
        """Train encoder + projection tail jointly with the SupCon loss on
        ``y``. ``sub_train`` (e.g. spacegroup ids) is only needed for
        ``body_batching.strategy == "balanced"``."""
        cfg = self.config
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        return training_first_phase(
            self.encoder,
            self.projection,
            FeatureDatabase.from_array(X_train),
            FeatureDatabase.from_array(X_val),
            cfg.body_train,
            train_family_ids=y_train,
            val_family_ids=y_val,
            lambda_family=1.0,
            lambda_spacegroup=0.0,
            lambda_norm=cfg.lambda_norm,
            **_batching_kwargs(cfg.body_batching, y_train, sub_train),
        )

    # -- phase 2 ---------------------------------------------------------
    def _build_heads(self, cfg: StackConfig, n_classes: int) -> None:
        self.classifier = ClassificationTail(
            input_dim=self.config.latent_dim,
            hidden_dim=cfg.classifier_hidden_dim,
            n_classes=n_classes,
            seed=cfg.seed,
        )
        self.visualizer = VisualizationTail(
            input_dim=self.config.latent_dim,
            hidden_dim=cfg.viz_hidden_dim or [self.config.latent_dim],
            output_dim=cfg.viz_dim,
            seed=cfg.seed,
        )
        self.n_classes = int(n_classes)
        self._heads_config = cfg

    def fit_heads(
        self,
        X_train,
        X_val,
        y_train,
        y_val,
        n_classes: int,
        sub_train=None,
        config: Optional[StackConfig] = None,
    ) -> Dict[str, History]:
        """Train the classification and visualization tails on the frozen
        encoder's output (the projection tail is not used). ``config``
        overrides the head hyperparameters (``classifier_*``, ``viz_*``,
        ``seed``); encoder fields always come from the stack's own config."""
        cfg = config if config is not None else self.config
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        r_train = self.encode(X_train)
        r_val = self.encode(X_val)
        self._build_heads(cfg, n_classes)

        classifier_history = train_classification_tail(
            self.classifier,
            r_train,
            r_val,
            cfg.classifier_train,
            train_labels=y_train,
            val_labels=y_val,
        )
        visualization_history = train_visualization_tail(
            self.visualizer,
            r_train,
            r_val,
            cfg.viz_train,
            train_family_ids=y_train,
            val_family_ids=y_val,
            lambda_family=1.0,
            lambda_spacegroup=0.0,
            lambda_norm=cfg.viz_lambda_norm,
            **_batching_kwargs(cfg.viz_batching, y_train, sub_train),
        )
        return {
            "classifier": classifier_history,
            "visualization": visualization_history,
        }

    # -- inference -------------------------------------------------------
    def encode(self, X) -> np.ndarray:
        return np.asarray(self.encoder.encode(X))

    def _require_heads(self) -> None:
        if self.classifier is None or self.visualizer is None:
            raise RuntimeError("heads are not trained or loaded: call fit_heads/load_heads")

    def predict_proba(self, X) -> np.ndarray:
        self._require_heads()
        logits = np.asarray(self.classifier.classify(self.encode(X)))
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def visualize(self, X) -> np.ndarray:
        self._require_heads()
        return np.asarray(self.visualizer.project(self.encode(X)))

    # -- persistence -----------------------------------------------------
    def save_body(self, directory) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _write_params(directory / "encoder_params.msgpack", self.encoder.params)
        _write_params(directory / "projection_params.msgpack", self.projection.params)
        with open(directory / "stack.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "input_dim": self.input_dim,
                    "config": stack_config_to_dict(self.config),
                },
                f,
                sort_keys=False,
            )

    @classmethod
    def load_body(cls, directory) -> "SingleStack":
        directory = Path(directory)
        with open(directory / "stack.yaml") as f:
            meta = yaml.safe_load(f)
        stack = cls(meta["input_dim"], stack_config_from_dict(meta["config"]))
        stack.encoder.params = _read_params(
            directory / "encoder_params.msgpack", stack.encoder.params
        )
        stack.projection.params = _read_params(
            directory / "projection_params.msgpack", stack.projection.params
        )
        return stack

    def save_heads(self, directory) -> None:
        self._require_heads()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _write_params(directory / "classifier_params.msgpack", self.classifier.params)
        _write_params(directory / "viz_params.msgpack", self.visualizer.params)
        with open(directory / "heads.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "n_classes": self.n_classes,
                    "config": stack_config_to_dict(self._heads_config),
                },
                f,
                sort_keys=False,
            )

    def load_heads(self, directory) -> None:
        directory = Path(directory)
        with open(directory / "heads.yaml") as f:
            meta = yaml.safe_load(f)
        self._build_heads(stack_config_from_dict(meta["config"]), meta["n_classes"])
        self.classifier.params = _read_params(
            directory / "classifier_params.msgpack", self.classifier.params
        )
        self.visualizer.params = _read_params(
            directory / "viz_params.msgpack", self.visualizer.params
        )
```

- [ ] **Step 4: Export lazily from `src/dim_red/supcon/__init__.py`**

Add `"Batching"`, `"StackConfig"`, `"SingleStack"` to `__all__` and, before the final `raise AttributeError`, add:

```python
    if name in {"Batching", "StackConfig", "SingleStack"}:
        stack = import_module("dim_red.supcon.stack")
        return getattr(stack, name)
```

- [ ] **Step 5: Run unit tests**

Run: `pytest tests/test_supcon_stack.py -v`
Expected: all pass. If `test_fit_body_equals_direct_training_first_phase_call` fails on `history["train_loss"]`, the call arguments in `fit_body` differ from the direct call — fix `fit_body`, not the test.

- [ ] **Step 6: Write the golden comparison tests `tests/test_supcon_stack_golden.py`**

```python
"""SingleStack must reproduce the OLD pipeline (run_single + train_tail:
classification, visualization, hierarchical_supcon) on the same inputs.

The body (same inputs, same shapes) is compared exactly. The heads encode the
data in separate train/val calls where the old code encoded all rows at once,
so XLA may reassociate floats differently by batch shape: heads are compared
with rtol=1e-5, atol=1e-6."""

import csv

import numpy as np
import pytest
import yaml

pytest.importorskip("jax")

from flax import serialization
from flax.traverse_util import flatten_dict

from dim_red.pipeline._common import _build_vocab_ids
from dim_red.supcon.stack import SingleStack, StackConfig
from dim_red.supcon.training import TrainConfig
from tests.golden import golden_spec as g


def _golden_params(name):
    return flatten_dict(serialization.msgpack_restore((g.GOLDEN_DIR / name).read_bytes()))


def _assert_params(actual, golden, exact):
    flat = flatten_dict(serialization.to_state_dict(actual))
    assert set(flat) == set(golden)
    for key, expected in golden.items():
        if exact:
            np.testing.assert_array_equal(np.asarray(flat[key]), expected)
        else:
            np.testing.assert_allclose(
                np.asarray(flat[key]), expected, rtol=1e-5, atol=1e-6
            )


def _golden_column(name, column):
    with open(g.GOLDEN_DIR / name) as f:
        return [float(row[column]) for row in csv.DictReader(f)]


def _train(epochs, tau, distance):
    return TrainConfig(
        epochs=epochs,
        batch_size=g.BATCH,
        learning_rate=1e-3,
        tau=tau,
        distance=distance,
        seed=g.SEED,
    )


def _family_config():
    return StackConfig(
        encoder_hidden_dim=g.ENCODER_HIDDEN,
        latent_dim=g.LATENT,
        projection_dim=g.PROJ_DIM,
        body_train=_train(g.BODY_EPOCHS, g.TAU, g.DISTANCE),
        classifier_train=_train(g.HEAD_EPOCHS, g.VIZ_TAU, "euclidean"),
        viz_train=_train(g.HEAD_EPOCHS, g.VIZ_TAU, g.VIZ_DISTANCE),
        classifier_hidden_dim=g.CLF_HIDDEN,
        viz_hidden_dim=g.VIZ_HIDDEN,
        viz_dim=2,
        seed=g.SEED,
    )


def _expert_config():
    return StackConfig(
        encoder_hidden_dim=g.ENCODER_HIDDEN,
        latent_dim=g.LATENT,
        projection_dim=g.PROJ_DIM,
        body_train=_train(g.HEAD_EPOCHS, g.TAU, g.DISTANCE),
        classifier_train=_train(g.HEAD_EPOCHS, g.TAU, g.DISTANCE),
        viz_train=_train(g.HEAD_EPOCHS, g.VIZ_TAU, g.VIZ_DISTANCE),
        classifier_hidden_dim=g.EXPERT_CLF_HIDDEN,
        viz_hidden_dim=g.EXPERT_VIZ_HIDDEN,
        viz_dim=2,
        seed=g.SEED,
    )


def _inputs():
    inputs = np.load(g.GOLDEN_DIR / "inputs.npz")
    split = np.load(g.GOLDEN_DIR / "family_embeddings.npz", allow_pickle=True)["split"]
    return inputs["X"], [str(v) for v in inputs["labels"]], inputs["spacegroups"], split


def test_family_stack_matches_old_pipeline():
    X, labels, _, split = _inputs()
    classes, y = _build_vocab_ids(labels)
    tr = np.flatnonzero(split == "train")
    va = np.flatnonzero(split == "val")

    stack = SingleStack(X.shape[1], _family_config())
    body_history = stack.fit_body(X[tr], X[va], y[tr], y[va])

    _assert_params(stack.encoder.params, _golden_params("family_body_params.msgpack"), True)
    _assert_params(
        stack.projection.params, _golden_params("family_projection_params.msgpack"), True
    )
    np.testing.assert_allclose(
        body_history["train_loss"],
        _golden_column("family_body_loss_history.csv", "train_loss"),
        rtol=1e-6,
    )

    stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=len(classes))
    _assert_params(
        stack.classifier.params, _golden_params("family_classifier_params.msgpack"), False
    )
    _assert_params(
        stack.visualizer.params, _golden_params("family_viz_params.msgpack"), False
    )


@pytest.mark.parametrize("family", g.FAMILIES)
def test_expert_stack_matches_old_hierarchical_supcon(family):
    X, labels, spacegroups, split = _inputs()
    idx = np.flatnonzero(np.asarray(labels) == family)
    classes, local = _build_vocab_ids([int(s) for s in spacegroups[idx]])
    y = np.full(len(X), -1, dtype=np.int64)
    y[idx] = local
    tr = idx[split[idx] == "train"]
    va = idx[split[idx] == "val"]

    with open(g.GOLDEN_DIR / f"expert_{family}_local_classes.yaml") as f:
        assert classes == yaml.safe_load(f)["local_spacegroup_classes"]

    stack = SingleStack(X.shape[1], _expert_config())
    stack.fit_body(X[tr], X[va], y[tr], y[va])
    _assert_params(
        stack.encoder.params, _golden_params(f"expert_{family}_body_params.msgpack"), True
    )
    _assert_params(
        stack.projection.params,
        _golden_params(f"expert_{family}_projection_params.msgpack"),
        True,
    )

    stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=len(classes))
    _assert_params(
        stack.classifier.params,
        _golden_params(f"expert_{family}_classifier_params.msgpack"),
        False,
    )
    _assert_params(
        stack.visualizer.params,
        _golden_params(f"expert_{family}_viz_params.msgpack"),
        False,
    )
```

- [ ] **Step 7: Run the golden tests**

Run: `pytest tests/test_supcon_stack_golden.py -v`
Expected: 4 passed. If a body comparison fails, the SingleStack call differs from the old orchestration (check seeds, `projection_hidden_dim or [latent_dim]`, label slot, `lambda_*`). If only a *head* comparison fails by more than 1e-5, report it before loosening any tolerance: it means the old orchestration does something the plan missed.

- [ ] **Step 8: Commit**

```bash
git add src/dim_red/supcon/stack.py src/dim_red/supcon/__init__.py tests/test_supcon_stack.py tests/test_supcon_stack_golden.py
git commit -m "Add SingleStack: fit_body/fit_heads over the existing primitives, checked against golden references

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `FullStackConfig` (YAML → resolved specs)

**Files:**
- Create: `src/dim_red/pipeline/full_stack_config.py`
- Test: `tests/test_pipeline_full_stack_config.py`

**Interfaces:**
- Consumes: `StackConfig`, `Batching`, `stack_config_to_dict`, `stack_config_from_dict` (Task 1); from `dim_red.pipeline.config`: `AugmentationConfig`, `MaceConfig`, `PyxtalConfig`, `SoapConfig`, `TailTrainSettings`, `TrainSettings`, `_dataclass_from_dict`, `_parse_train_settings`, `load_yaml`.
- Produces (exact):
  - constants `FAMILY = "family"`, `EXPERT_NAMES` (tuple of the 7 lowercase systems), `STACK_ORDER`
  - `StackDataConfig(pyxtal, augmentation=None, soap=SoapConfig(), mace=MaceConfig())`
  - `StackSpec(name, data, model: StackConfig, val_ratio: float, seed: int, min_train_rows: int = 10)`
  - `FullStackConfig(name, seed, model_kind, output_dir, stacks: Dict[str, StackSpec])`
  - `full_stack_config_from_dict(d)`, `load_full_stack_config(path)`, `parse_stack_spec(name, block, run_seed)`
  - `stack_spec_to_dict/from_dict`, `full_stack_config_to_dict`, `full_stack_config_from_resolved_dict`

- [ ] **Step 1: Write the failing tests `tests/test_pipeline_full_stack_config.py`**

```python
import pytest
import yaml

pytest.importorskip("jax")

from dim_red.pipeline.full_stack_config import (
    FAMILY,
    full_stack_config_from_dict,
    full_stack_config_from_resolved_dict,
    full_stack_config_to_dict,
    load_full_stack_config,
)


def _block(**over):
    block = {
        "data": {"pyxtal": {"structures_per_spacegroup": 2}},
        "encoder": {"encoder_hidden_dim": [8], "latent_dim": 4},
        "train": {"epochs": 2, "batch_size": 8},
    }
    block.update(over)
    return block


def _config(**over):
    d = {
        "name": "t",
        "seed": 10,
        "family": _block(),
        "experts": {"defaults": _block(), "cubic": {}, "tetragonal": {}},
    }
    d.update(over)
    return d


def test_defaults_are_merged_under_each_expert_override():
    d = _config()
    d["experts"]["cubic"] = {"encoder": {"latent_dim": 6}}
    cfg = full_stack_config_from_dict(d)
    assert cfg.stacks["cubic"].model.latent_dim == 6
    assert list(cfg.stacks["cubic"].model.encoder_hidden_dim) == [8]  # from defaults
    assert cfg.stacks["tetragonal"].model.latent_dim == 4


def test_stack_order_is_family_then_canonical_experts():
    d = _config()
    d["experts"] = {"defaults": _block(), "cubic": {}, "triclinic": {}}
    cfg = full_stack_config_from_dict(d)
    assert list(cfg.stacks) == ["family", "triclinic", "cubic"]


def test_expert_pyxtal_is_restricted_to_its_system_and_family_is_not():
    cfg = full_stack_config_from_dict(_config())
    assert cfg.stacks["cubic"].data.pyxtal.families == ["Cubic"]
    assert cfg.stacks["tetragonal"].data.pyxtal.families == ["Tetragonal"]
    assert cfg.stacks[FAMILY].data.pyxtal.families is None


def test_expert_with_a_different_family_is_rejected():
    d = _config()
    d["experts"]["cubic"] = {"data": {"pyxtal": {"families": ["Hexagonal"]}}}
    with pytest.raises(ValueError, match="cubic"):
        full_stack_config_from_dict(d)


def test_expert_with_explicit_spacegroups_is_left_alone():
    d = _config()
    d["experts"]["cubic"] = {
        "data": {"pyxtal": {"spacegroups": [195, 200], "structures_per_spacegroup": 3}}
    }
    cfg = full_stack_config_from_dict(d)
    pyxtal = cfg.stacks["cubic"].data.pyxtal
    assert pyxtal.spacegroups == [195, 200]
    assert pyxtal.families is None


def test_unknown_expert_name_lists_valid_names():
    d = _config()
    d["experts"]["cubik"] = {}
    with pytest.raises(ValueError, match="cubik") as exc:
        full_stack_config_from_dict(d)
    assert "cubic" in str(exc.value)


def test_config_without_stacks_is_rejected():
    with pytest.raises(ValueError, match="no stacks"):
        full_stack_config_from_dict({"name": "t"})


def test_family_is_optional():
    d = _config()
    del d["family"]
    cfg = full_stack_config_from_dict(d)
    assert FAMILY not in cfg.stacks and "cubic" in cfg.stacks


def test_seeds_follow_the_canonical_stack_index_and_fill_pyxtal_seed():
    cfg = full_stack_config_from_dict(_config())
    assert cfg.stacks[FAMILY].seed == 10
    assert cfg.stacks["tetragonal"].seed == 14
    assert cfg.stacks["cubic"].seed == 17
    assert cfg.stacks["cubic"].data.pyxtal.seed == 17
    assert cfg.stacks["cubic"].model.seed == 17
    assert cfg.stacks["cubic"].model.body_train.seed == 17


def test_explicit_pyxtal_seed_is_kept():
    d = _config()
    d["family"]["data"]["pyxtal"]["seed"] = 99
    cfg = full_stack_config_from_dict(d)
    assert cfg.stacks[FAMILY].data.pyxtal.seed == 99


def test_head_training_settings_inherit_the_body_train_block():
    d = _config()
    d["family"]["train"] = {"epochs": 7, "batch_size": 16, "val_ratio": 0.3}
    d["family"]["viz"] = {"train": {"epochs": 3}}
    cfg = full_stack_config_from_dict(d)
    model = cfg.stacks[FAMILY].model
    assert model.body_train.epochs == 7
    assert model.classifier_train.epochs == 7 and model.classifier_train.batch_size == 16
    assert model.viz_train.epochs == 3
    assert cfg.stacks[FAMILY].val_ratio == 0.3


def test_contrastive_and_viz_defaults():
    model = full_stack_config_from_dict(_config()).stacks[FAMILY].model
    assert (model.body_train.tau, model.body_train.distance) == (0.05, "cosine")
    assert (model.viz_train.tau, model.viz_train.distance) == (0.1, "euclidean")
    assert model.viz_dim == 2 and model.projection_dim == 128


def test_invalid_distance_is_rejected():
    d = _config()
    d["family"]["contrastive"] = {"distance": "manhattan"}
    with pytest.raises(ValueError, match="distance"):
        full_stack_config_from_dict(d)


def test_missing_encoder_fields_are_rejected():
    d = _config()
    d["family"]["encoder"] = {"latent_dim": 4}
    with pytest.raises(ValueError, match="encoder_hidden_dim"):
        full_stack_config_from_dict(d)


def test_unknown_block_key_is_rejected():
    d = _config()
    d["family"]["encoder_hidden"] = [8]
    with pytest.raises(ValueError, match="encoder_hidden"):
        full_stack_config_from_dict(d)


def test_supcon_mace_requires_a_checkpoint():
    d = _config(model_kind="supcon_mace")
    with pytest.raises(ValueError, match="checkpoint_path"):
        full_stack_config_from_dict(d)


def test_resolved_dict_roundtrip():
    cfg = full_stack_config_from_dict(_config())
    assert full_stack_config_from_resolved_dict(full_stack_config_to_dict(cfg)) == cfg


def test_load_from_yaml_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(_config()))
    cfg = load_full_stack_config(path)
    assert set(cfg.stacks) == {"family", "cubic", "tetragonal"}
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_pipeline_full_stack_config.py -v`
Expected: FAIL with `ModuleNotFoundError: dim_red.pipeline.full_stack_config`.

- [ ] **Step 3: Implement `src/dim_red/pipeline/full_stack_config.py`**

```python
"""Config for ``FullStack``: a ``family`` stack plus up to 7 per-crystal-system
expert stacks, each with its own dataset (pyxtal) and hyperparameters. Experts
share an optional ``experts.defaults`` block that is deep-merged under each
expert's own block, so every stack is resolved into a complete ``StackSpec``
before any dataset is built.

Needs jax (it imports ``dim_red.supcon.stack``); ``dim_red.pipeline.config``
itself is untouched and stays importable without it.
"""

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from dim_red.pipeline.config import (
    AugmentationConfig,
    MaceConfig,
    PyxtalConfig,
    SoapConfig,
    TailTrainSettings,
    TrainSettings,
    _dataclass_from_dict,
    _parse_train_settings,
    load_yaml,
)
from dim_red.supcon.stack import (
    Batching,
    StackConfig,
    stack_config_from_dict,
    stack_config_to_dict,
)
from dim_red.supcon.training import TrainConfig

FAMILY = "family"
EXPERT_NAMES = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)
STACK_ORDER = (FAMILY,) + EXPERT_NAMES

_MODEL_KINDS = ("supcon", "supcon_mace")
_DISTANCES = ("euclidean", "cosine")
_TOP_KEYS = {"name", "seed", "model_kind", "output_dir", "family", "experts"}
_BLOCK_KEYS = {
    "seed",
    "min_train_rows",
    "data",
    "encoder",
    "projection",
    "contrastive",
    "train",
    "batching",
    "classifier",
    "viz",
}


@dataclass(frozen=True)
class StackDataConfig:
    pyxtal: PyxtalConfig
    augmentation: Optional[AugmentationConfig] = None
    soap: SoapConfig = field(default_factory=SoapConfig)
    mace: MaceConfig = field(default_factory=MaceConfig)


@dataclass(frozen=True)
class StackSpec:
    """One fully resolved stack: its data, its model config, its split ratio,
    its seed and the minimum number of training rows it needs."""

    name: str
    data: StackDataConfig
    model: StackConfig
    val_ratio: float
    seed: int
    min_train_rows: int = 10


@dataclass(frozen=True)
class FullStackConfig:
    name: str
    seed: int
    model_kind: str
    output_dir: str
    stacks: Dict[str, StackSpec]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _check_distance(name: str, where: str, value: str) -> str:
    if value not in _DISTANCES:
        raise ValueError(
            f"stack {name!r}: {where}.distance must be one of {_DISTANCES}, got {value!r}"
        )
    return value


def _batching(name: str, d: dict) -> Batching:
    batching = Batching(**{k: d[k] for k in ("strategy", "P", "K", "S") if k in d})
    if batching.strategy not in ("random", "balanced"):
        raise ValueError(
            f"stack {name!r}: batching.strategy must be 'random' or 'balanced', "
            f"got {batching.strategy!r}"
        )
    if batching.strategy == "balanced" and not (batching.K and batching.K > 0):
        raise ValueError(f"stack {name!r}: balanced batching needs a positive K")
    return batching


def _train_config(settings, tau: float, distance: str, seed: int) -> TrainConfig:
    return TrainConfig(
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        tau=tau,
        distance=distance,
        seed=seed,
        device=settings.device,
        early_stopping=settings.early_stopping.enabled,
        early_stopping_patience=settings.early_stopping.patience,
        early_stopping_min_delta=settings.early_stopping.min_delta,
        early_stopping_restore_best=settings.early_stopping.restore_best_weights,
    )


def _restrict_pyxtal(name: str, pyxtal: dict) -> dict:
    """An expert generates only its own crystal system. Explicit
    ``spacegroups`` are left alone (the user chose them)."""
    if name == FAMILY or pyxtal.get("spacegroups") is not None:
        return pyxtal
    system = name.capitalize()
    families = pyxtal.get("families")
    if families is not None and [f.capitalize() for f in families] != [system]:
        raise ValueError(
            f"stack {name!r}: pyxtal.families must be [{system!r}] (or omitted), "
            f"got {families!r}"
        )
    return {**pyxtal, "families": [system]}


def parse_stack_spec(name: str, block: Dict[str, Any], run_seed: int) -> StackSpec:
    unknown = set(block) - _BLOCK_KEYS
    if unknown:
        raise ValueError(
            f"stack {name!r}: unknown keys {sorted(unknown)}; valid keys: "
            f"{sorted(_BLOCK_KEYS)}"
        )
    seed = int(block.get("seed", run_seed + STACK_ORDER.index(name)))

    data_block = block.get("data")
    if not isinstance(data_block, dict) or "pyxtal" not in data_block:
        raise ValueError(
            f"stack {name!r}: 'data' must contain a 'pyxtal' block "
            "(FullStack supports data_source pyxtal only)"
        )
    pyxtal = _dataclass_from_dict(
        PyxtalConfig, _restrict_pyxtal(name, dict(data_block["pyxtal"]))
    )
    if pyxtal.seed is None:
        pyxtal = dataclasses.replace(pyxtal, seed=seed)
    augmentation = (
        _dataclass_from_dict(AugmentationConfig, data_block["augmentation"])
        if data_block.get("augmentation") is not None
        else None
    )
    data = StackDataConfig(
        pyxtal=pyxtal,
        augmentation=augmentation,
        soap=_dataclass_from_dict(SoapConfig, data_block.get("soap", {})),
        mace=_dataclass_from_dict(MaceConfig, data_block.get("mace", {})),
    )

    encoder = block.get("encoder") or {}
    missing = {"encoder_hidden_dim", "latent_dim"} - set(encoder)
    if missing:
        raise ValueError(f"stack {name!r}: encoder block needs {sorted(missing)}")
    projection = block.get("projection", {})
    contrastive = block.get("contrastive", {})
    distance = _check_distance(
        name, "contrastive", contrastive.get("distance", "cosine")
    )
    classifier = block.get("classifier", {})
    viz = block.get("viz", {})
    viz_distance = _check_distance(name, "viz", viz.get("distance", "euclidean"))
    viz_dim = int(viz.get("viz_dim", 2))
    if viz_dim not in (2, 3):
        raise ValueError(f"stack {name!r}: viz.viz_dim must be 2 or 3, got {viz_dim}")

    train_dict = block.get("train", {})
    head_default = {k: v for k, v in train_dict.items() if k != "val_ratio"}
    body_settings = _parse_train_settings(TrainSettings, train_dict)
    classifier_settings = _parse_train_settings(
        TailTrainSettings, classifier.get("train", head_default)
    )
    viz_settings = _parse_train_settings(
        TailTrainSettings, viz.get("train", head_default)
    )

    model = StackConfig(
        encoder_hidden_dim=list(encoder["encoder_hidden_dim"]),
        latent_dim=int(encoder["latent_dim"]),
        projection_dim=int(projection.get("projection_dim", 128)),
        projection_hidden_dim=projection.get("projection_hidden_dim"),
        body_train=_train_config(
            body_settings, float(contrastive.get("tau", 0.05)), distance, seed
        ),
        classifier_train=_train_config(classifier_settings, 0.1, "euclidean", seed),
        viz_train=_train_config(
            viz_settings, float(viz.get("tau", 0.1)), viz_distance, seed
        ),
        classifier_hidden_dim=classifier.get("hidden_dim", 16),
        viz_hidden_dim=viz.get("hidden_dim"),
        viz_dim=viz_dim,
        lambda_norm=float(contrastive.get("lambda_norm", 0.0)),
        viz_lambda_norm=float(viz.get("lambda_norm", 0.0)),
        body_batching=_batching(name, block.get("batching", {})),
        viz_batching=_batching(name, viz.get("batching", {})),
        seed=seed,
    )
    return StackSpec(
        name=name,
        data=data,
        model=model,
        val_ratio=body_settings.val_ratio,
        seed=seed,
        min_train_rows=int(block.get("min_train_rows", 10)),
    )


def full_stack_config_from_dict(d: Dict[str, Any]) -> FullStackConfig:
    unknown = set(d) - _TOP_KEYS
    if unknown:
        raise ValueError(
            f"unknown top-level keys {sorted(unknown)}; valid keys: {sorted(_TOP_KEYS)}"
        )
    model_kind = d.get("model_kind", "supcon")
    if model_kind not in _MODEL_KINDS:
        raise ValueError(f"model_kind must be one of {_MODEL_KINDS}, got {model_kind!r}")
    seed = int(d.get("seed", 42))

    experts = dict(d.get("experts") or {})
    defaults = experts.pop("defaults", None) or {}
    bad = set(experts) - set(EXPERT_NAMES)
    if bad:
        raise ValueError(
            f"unknown expert name(s) {sorted(bad)}; valid names: {list(EXPERT_NAMES)}"
        )

    blocks: Dict[str, dict] = {}
    if d.get("family") is not None:
        blocks[FAMILY] = dict(d["family"])
    for name in EXPERT_NAMES:
        if name in experts:
            blocks[name] = _deep_merge(defaults, experts[name] or {})
    if not blocks:
        raise ValueError(
            "config defines no stacks: set 'family' and/or at least one expert "
            "under 'experts'"
        )

    stacks = {name: parse_stack_spec(name, block, seed) for name, block in blocks.items()}
    if model_kind == "supcon_mace":
        for name, spec in stacks.items():
            if not spec.data.mace.checkpoint_path:
                raise ValueError(
                    f"stack {name!r}: model_kind supcon_mace needs "
                    "data.mace.checkpoint_path"
                )
    return FullStackConfig(
        name=str(d.get("name", "full-stack")),
        seed=seed,
        model_kind=model_kind,
        output_dir=str(d.get("output_dir", "runs")),
        stacks=stacks,
    )


def load_full_stack_config(path: Union[str, Path]) -> FullStackConfig:
    return full_stack_config_from_dict(load_yaml(path))


# -- resolved (fully expanded) dicts, used for what is written on disk -------
def stack_spec_to_dict(spec: StackSpec) -> dict:
    d = dataclasses.asdict(spec)
    d["model"] = stack_config_to_dict(spec.model)
    return json.loads(json.dumps(d))


def stack_spec_from_dict(d: dict) -> StackSpec:
    data = d["data"]
    return StackSpec(
        name=d["name"],
        data=StackDataConfig(
            pyxtal=PyxtalConfig(**data["pyxtal"]),
            augmentation=(
                AugmentationConfig(**data["augmentation"])
                if data.get("augmentation")
                else None
            ),
            soap=SoapConfig(**data["soap"]),
            mace=MaceConfig(**data["mace"]),
        ),
        model=stack_config_from_dict(d["model"]),
        val_ratio=d["val_ratio"],
        seed=d["seed"],
        min_train_rows=d["min_train_rows"],
    )


def full_stack_config_to_dict(config: FullStackConfig) -> dict:
    return {
        "name": config.name,
        "seed": config.seed,
        "model_kind": config.model_kind,
        "output_dir": config.output_dir,
        "stacks": {n: stack_spec_to_dict(s) for n, s in config.stacks.items()},
    }


def full_stack_config_from_resolved_dict(d: dict) -> FullStackConfig:
    return FullStackConfig(
        name=d["name"],
        seed=d["seed"],
        model_kind=d["model_kind"],
        output_dir=d["output_dir"],
        stacks={n: stack_spec_from_dict(s) for n, s in d["stacks"].items()},
    )
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_pipeline_full_stack_config.py -v`
Expected: all pass. If `test_resolved_dict_roundtrip` fails on equality, print both configs and look for tuple-vs-list or int-vs-float differences introduced by the JSON round trip; fix the `*_from_dict` side, not the test.

- [ ] **Step 5: Commit**

```bash
git add src/dim_red/pipeline/full_stack_config.py tests/test_pipeline_full_stack_config.py
git commit -m "Add FullStackConfig: per-stack resolved specs with expert defaults and pyxtal restriction

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: One dataset per stack

**Files:**
- Modify: `src/dim_red/pipeline/dataset_cache.py` (the `_resolve_augmentation` function, ~lines 579–606)
- Create: `src/dim_red/pipeline/stack_data.py`
- Test: `tests/test_pipeline_stack_data.py`

**Interfaces:**
- Consumes: `StackSpec` (Task 2); `get_or_build_pyxtal_dataset(pyxtal_config, seed, soap_kwargs, cache_dir, augmentation)` and `get_or_build_pyxtal_mace_dataset(pyxtal_config, seed, mace_kwargs, cache_dir, augmentation)` from `dataset_cache`.
- Produces: `resolve_augmentation(augmentation, default_seed) -> Optional[dim_red.augmentation.AugmentationConfig]`; `StackDataset(X, labels, material_ids, spacegroups, structures_path, feature_mean, feature_std)`; `build_stack_dataset(spec, model_kind, cache_dir) -> StackDataset`.

- [ ] **Step 1: Write the failing tests `tests/test_pipeline_stack_data.py`**

```python
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("jax")

from dim_red.pipeline.full_stack_config import FAMILY, full_stack_config_from_dict
from dim_red.pipeline.stack_data import StackDataset, build_stack_dataset


def _spec(name=FAMILY, model_kind="supcon", extra_data=None, seed=5):
    data = {"pyxtal": {"structures_per_spacegroup": 2}}
    data.update(extra_data or {})
    block = {
        "data": data,
        "encoder": {"encoder_hidden_dim": [4], "latent_dim": 2},
    }
    d = {"seed": seed, "model_kind": model_kind}
    if model_kind == "supcon_mace":
        data["mace"] = {"checkpoint_path": "ckpt"}
    if name == FAMILY:
        d["family"] = block
    else:
        d["experts"] = {name: block}
    return full_stack_config_from_dict(d).stacks[name]


def _fake_result():
    return (
        np.zeros((3, 4), dtype=np.float32),
        ["Cubic"] * 3,
        ["a", "b", "c"],
        [195, 195, 196],
        Path("s.extxyz"),
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
    )


def test_soap_dataset_uses_the_stack_seed_and_soap_kwargs():
    spec = _spec("cubic", seed=5)
    with patch(
        "dim_red.pipeline.stack_data.get_or_build_pyxtal_dataset",
        return_value=_fake_result(),
    ) as soap_builder:
        ds = build_stack_dataset(spec, "supcon", "cache")
    kwargs = soap_builder.call_args.kwargs
    assert kwargs["seed"] == spec.data.pyxtal.seed == 5 + 7
    assert kwargs["pyxtal_config"].families == ["Cubic"]
    assert kwargs["soap_kwargs"] == spec.data.soap.as_kwargs()
    assert kwargs["augmentation"] is None
    assert isinstance(ds, StackDataset) and ds.X.shape == (3, 4)
    assert ds.spacegroups == [195, 195, 196]


def test_mace_model_kind_routes_to_the_mace_builder():
    spec = _spec(model_kind="supcon_mace")
    with patch(
        "dim_red.pipeline.stack_data.get_or_build_pyxtal_mace_dataset",
        return_value=_fake_result(),
    ) as mace_builder:
        build_stack_dataset(spec, "supcon_mace", "cache")
    assert mace_builder.call_args.kwargs["mace_kwargs"] == spec.data.mace.mace_kwargs()


def test_augmentation_seed_defaults_to_the_stack_seed():
    spec = _spec(
        extra_data={"augmentation": {"n_augmented": 1, "jitter_std": 0.01}}, seed=3
    )
    with patch(
        "dim_red.pipeline.stack_data.get_or_build_pyxtal_dataset",
        return_value=_fake_result(),
    ) as soap_builder:
        build_stack_dataset(spec, "supcon", "cache")
    augmentation = soap_builder.call_args.kwargs["augmentation"]
    assert augmentation.n_augmented == 1 and augmentation.seed == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_pipeline_stack_data.py -v`
Expected: FAIL (`ModuleNotFoundError: dim_red.pipeline.stack_data`).

- [ ] **Step 3: Refactor `_resolve_augmentation` in `dataset_cache.py`**

Read the current function (it starts at `def _resolve_augmentation(config: "RunConfig")`), then replace the whole function with:

```python
def resolve_augmentation(
    augmentation: Optional["PipelineAugmentationConfig"], default_seed: int
) -> Optional[AugmentationConfig]:
    """Converts the pipeline's own ``AugmentationConfig`` (see
    ``dim_red.pipeline.config``, independent of ``dim_red.augmentation``) into
    the real ``dim_red.augmentation.AugmentationConfig``. ``None`` when
    augmentation is disabled; ``seed`` falls back to ``default_seed``."""
    if augmentation is None:
        return None
    seed = augmentation.seed if augmentation.seed is not None else default_seed
    return AugmentationConfig(
        n_augmented=augmentation.n_augmented,
        keep_original=augmentation.keep_original,
        jitter_probability=augmentation.jitter_probability,
        jitter_std=augmentation.jitter_std,
        vacancy_probability=augmentation.vacancy_probability,
        vacancy_atom_probability=augmentation.vacancy_atom_probability,
        max_vacancies=augmentation.max_vacancies,
        supercell_radius=augmentation.supercell_radius,
        seed=seed,
    )


def _resolve_augmentation(config: "RunConfig") -> Optional[AugmentationConfig]:
    """``resolve_augmentation`` applied to ``RunConfig.augmentation`` with
    ``RunConfig.seed`` as the fallback seed."""
    return resolve_augmentation(config.augmentation, config.seed)
```

If the original function body used different field names than the ones above, copy the fields exactly from the original — the replacement must be behavior-identical. The type hint `"PipelineAugmentationConfig"` is a string annotation for the pipeline-config dataclass; add under the existing `TYPE_CHECKING` imports (if present) `from dim_red.pipeline.config import AugmentationConfig as PipelineAugmentationConfig`, otherwise leave the string annotation as is (it is never evaluated).

- [ ] **Step 4: Run the existing dataset-cache tests (behavior must be unchanged)**

Run: `pytest tests/test_pipeline_dataset_cache.py tests/test_pipeline_dataset_cache_characterization.py -v`
Expected: all pass (same results as before the edit).

- [ ] **Step 5: Implement `src/dim_red/pipeline/stack_data.py`**

```python
"""One dataset per stack: every ``SingleStack`` of a ``FullStack`` gets its own
pyxtal-generated (and optionally augmented) structures, featurized with SOAP
or MACE through the existing dataset cache. Stacks never share structures:
their pyxtal seeds differ (see ``full_stack_config``)."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

from dim_red.pipeline.dataset_cache import (
    get_or_build_pyxtal_dataset,
    get_or_build_pyxtal_mace_dataset,
    resolve_augmentation,
)
from dim_red.pipeline.full_stack_config import StackSpec


@dataclass(frozen=True)
class StackDataset:
    """Standardized features plus the per-structure metadata of one stack."""

    X: np.ndarray
    labels: List[str]  # crystal family of each structure
    material_ids: List[str]
    spacegroups: List[int]
    structures_path: Path
    feature_mean: np.ndarray
    feature_std: np.ndarray


def build_stack_dataset(
    spec: StackSpec, model_kind: str, cache_dir: Union[str, Path]
) -> StackDataset:
    augmentation = resolve_augmentation(spec.data.augmentation, spec.seed)
    seed = spec.data.pyxtal.seed if spec.data.pyxtal.seed is not None else spec.seed
    if model_kind == "supcon_mace":
        result = get_or_build_pyxtal_mace_dataset(
            pyxtal_config=spec.data.pyxtal,
            seed=seed,
            mace_kwargs=spec.data.mace.mace_kwargs(),
            cache_dir=cache_dir,
            augmentation=augmentation,
        )
    else:
        result = get_or_build_pyxtal_dataset(
            pyxtal_config=spec.data.pyxtal,
            seed=seed,
            soap_kwargs=spec.data.soap.as_kwargs(),
            cache_dir=cache_dir,
            augmentation=augmentation,
        )
    X, labels, material_ids, spacegroups, structures_path, mean, std = result
    return StackDataset(
        X=X,
        labels=list(labels),
        material_ids=list(material_ids),
        spacegroups=[int(s) for s in spacegroups],
        structures_path=Path(structures_path),
        feature_mean=mean,
        feature_std=std,
    )
```

- [ ] **Step 6: Run the new tests**

Run: `pytest tests/test_pipeline_stack_data.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/dim_red/pipeline/dataset_cache.py src/dim_red/pipeline/stack_data.py tests/test_pipeline_stack_data.py
git commit -m "Add per-stack dataset builder; expose resolve_augmentation

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `FullStack`

**Files:**
- Create: `src/dim_red/pipeline/full_stack.py`
- Test: `tests/test_pipeline_full_stack.py`

**Interfaces:**
- Consumes: `SingleStack` (Task 1), `FullStackConfig`/`StackSpec`/`FAMILY`/resolved-dict functions (Task 2), `build_stack_dataset`/`StackDataset` (Task 3), `_build_vocab_ids`, `_make_unique_run_dir`, `_save_loss_history`, `_split_indices_grouped` (`pipeline/_common.py`), `plot_reduced_space` (`dim_red.analysis.plotting`).
- Produces (exact):
  - `FullStack.create(config, cache_dir=None) -> FullStack`; `FullStack.open(run_dir, config=None, cache_dir=None) -> FullStack`; attribute `.run_dir`, `.config`
  - `fit_body(stacks=None) -> Dict[str, History]`
  - `fit_heads(heads_name, stacks=None, config=None) -> Dict[str, Dict[str, History]]`
  - `load_stack(name, heads_name=None) -> SingleStack`
- Layout (per spec):
  `<run>/config.yaml` (resolved), `<run>/stacks/<name>/{config.yaml, dataset.extxyz, classes.yaml, embeddings.npz, body/{encoder_params.msgpack, projection_params.msgpack, stack.yaml, loss_history.csv}, heads/<heads_name>/{classifier_params.msgpack, viz_params.msgpack, heads.yaml, classifier_loss_history.csv, viz_loss_history.csv, predictions.npz, viz_embeddings.npz, viz_plot.png (2D only)}}`.
  `embeddings.npz` keys: `embeddings, features, labels, label_ids, spacegroups, spacegroup_ids, material_ids, split, feature_mean, feature_std`. `classes.yaml`: `{role: family|spacegroup, classes: [...]}`.

- [ ] **Step 1: Write the failing tests `tests/test_pipeline_full_stack.py`**

```python
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write

pytest.importorskip("jax")

from dim_red.pipeline.full_stack import FullStack
from dim_red.pipeline.full_stack_config import FAMILY, full_stack_config_from_dict
from dim_red.pipeline.stack_data import StackDataset

SYSTEM_SPACEGROUPS = {"cubic": [195, 196], "tetragonal": [75, 76], "hexagonal": [168, 169]}


def _block(**over):
    block = {
        "data": {"pyxtal": {"structures_per_spacegroup": 2}},
        "encoder": {"encoder_hidden_dim": [8], "latent_dim": 4},
        "projection": {"projection_dim": 5},
        "train": {"epochs": 2, "batch_size": 8},
        "classifier": {"hidden_dim": 6},
        "viz": {"hidden_dim": [5]},
        "min_train_rows": 5,
    }
    block.update(over)
    return block


def _config(tmp_path, experts=("cubic", "tetragonal"), **over):
    d = {
        "name": "fs",
        "seed": 3,
        "output_dir": str(tmp_path / "runs"),
        "family": _block(),
        "experts": {"defaults": _block(), **{e: {} for e in experts}},
    }
    d.update(over)
    return full_stack_config_from_dict(d)


def _fake_builder(tmp_path, calls, rows_per_sg=12):
    def build(spec, model_kind, cache_dir):
        calls.append(spec)
        rng = np.random.default_rng(spec.seed)
        if spec.name == FAMILY:
            plan = [
                (system.capitalize(), sg)
                for system, sgs in SYSTEM_SPACEGROUPS.items()
                for sg in sgs
            ]
        else:
            plan = [(spec.name.capitalize(), sg) for sg in SYSTEM_SPACEGROUPS[spec.name]]
        X, labels, material_ids, spacegroups = [], [], [], []
        for k, (family, sg) in enumerate(plan):
            center = rng.normal(size=6) * 3.0
            for i in range(rows_per_sg):
                X.append(center + rng.normal(size=6) * 0.3)
                labels.append(family)
                material_ids.append(f"{spec.name}-{sg}-{i}")
                spacegroups.append(sg)
        path = tmp_path / f"{spec.name}-{len(calls)}.extxyz"
        write(str(path), [Atoms("Cu", positions=[[0, 0, 0]])] * len(X), format="extxyz")
        return StackDataset(
            X=np.asarray(X, dtype=np.float32),
            labels=labels,
            material_ids=material_ids,
            spacegroups=spacegroups,
            structures_path=path,
            feature_mean=np.zeros(6, dtype=np.float32),
            feature_std=np.ones(6, dtype=np.float32),
        )

    return build


@pytest.fixture
def patched(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "dim_red.pipeline.full_stack.build_stack_dataset", _fake_builder(tmp_path, calls)
    )
    return calls


def test_fit_body_trains_every_configured_stack_with_its_own_dataset(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    histories = fs.fit_body()

    assert set(histories) == {"family", "cubic", "tetragonal"}
    assert [spec.name for spec in patched] == ["family", "cubic", "tetragonal"]
    assert len({spec.seed for spec in patched}) == 3  # independent seeds

    for name in histories:
        stack_dir = fs.run_dir / "stacks" / name
        for rel in (
            "config.yaml",
            "dataset.extxyz",
            "classes.yaml",
            "embeddings.npz",
            "body/encoder_params.msgpack",
            "body/projection_params.msgpack",
            "body/stack.yaml",
            "body/loss_history.csv",
        ):
            assert (stack_dir / rel).exists(), f"{name}/{rel}"
    with open(fs.run_dir / "stacks" / "cubic" / "classes.yaml") as f:
        classes = yaml.safe_load(f)
    assert classes == {"role": "spacegroup", "classes": [195, 196]}
    with open(fs.run_dir / "stacks" / "family" / "classes.yaml") as f:
        assert yaml.safe_load(f)["role"] == "family"


def test_subset_of_experts_only_creates_those_stacks(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    fs.fit_body(stacks=["cubic"])
    assert [p.name for p in sorted((fs.run_dir / "stacks").iterdir())] == ["cubic"]


def test_unknown_stack_in_selection_is_rejected(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="hexagonal"):
        fs.fit_body(stacks=["hexagonal"])
    assert not (fs.run_dir / "stacks").exists()


def test_config_without_family_trains_experts_only(tmp_path, patched):
    d = {
        "name": "fs",
        "output_dir": str(tmp_path / "runs"),
        "experts": {"defaults": _block(), "cubic": {}},
    }
    fs = FullStack.create(full_stack_config_from_dict(d), cache_dir=tmp_path / "cache")
    fs.fit_body()
    assert [p.name for p in (fs.run_dir / "stacks").iterdir()] == ["cubic"]


def test_fit_body_twice_on_the_same_stack_raises(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    fs.fit_body(stacks=["cubic"])
    with pytest.raises(FileExistsError, match="cubic"):
        fs.fit_body(stacks=["cubic"])


def test_a_stack_can_be_added_later_through_open(tmp_path, patched):
    config = _config(tmp_path, experts=("cubic", "tetragonal"))
    fs = FullStack.create(config, cache_dir=tmp_path / "cache")
    fs.fit_body(stacks=["cubic"])

    reopened = FullStack.open(fs.run_dir, cache_dir=tmp_path / "cache")
    reopened.fit_body(stacks=["tetragonal"])
    assert {p.name for p in (fs.run_dir / "stacks").iterdir()} == {"cubic", "tetragonal"}
    again = FullStack.open(fs.run_dir)
    assert {"cubic", "tetragonal"} <= set(again.config.stacks)


def test_too_few_training_rows_names_the_stack_and_writes_nothing(tmp_path, patched):
    d_block = _block(min_train_rows=10_000)
    d = {
        "output_dir": str(tmp_path / "runs"),
        "experts": {"defaults": d_block, "cubic": {}},
    }
    fs = FullStack.create(full_stack_config_from_dict(d), cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="cubic"):
        fs.fit_body()
    assert not (fs.run_dir / "stacks" / "cubic").exists()


def test_single_class_dataset_is_rejected(tmp_path, monkeypatch):
    calls = []
    builder = _fake_builder(tmp_path, calls)

    def one_class(spec, model_kind, cache_dir):
        ds = builder(spec, model_kind, cache_dir)
        return StackDataset(
            **{**ds.__dict__, "spacegroups": [195] * len(ds.spacegroups)}
        )

    monkeypatch.setattr("dim_red.pipeline.full_stack.build_stack_dataset", one_class)
    d = {"output_dir": str(tmp_path / "runs"), "experts": {"defaults": _block(), "cubic": {}}}
    fs = FullStack.create(full_stack_config_from_dict(d), cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="at least 2"):
        fs.fit_body()


def test_fit_heads_writes_heads_and_does_not_touch_the_body(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    fs.fit_body()
    body_file = fs.run_dir / "stacks" / "cubic" / "body" / "encoder_params.msgpack"
    before = body_file.read_bytes()

    result = fs.fit_heads("h1")
    assert set(result) == {"family", "cubic", "tetragonal"}
    for name in result:
        heads_dir = fs.run_dir / "stacks" / name / "heads" / "h1"
        for rel in (
            "classifier_params.msgpack",
            "viz_params.msgpack",
            "heads.yaml",
            "classifier_loss_history.csv",
            "viz_loss_history.csv",
            "predictions.npz",
            "viz_embeddings.npz",
            "viz_plot.png",
        ):
            assert (heads_dir / rel).exists(), f"{name}/{rel}"
        preds = np.load(heads_dir / "predictions.npz")
        np.testing.assert_allclose(preds["probs"].sum(axis=1), 1.0, atol=1e-5)
    assert body_file.read_bytes() == before


def test_fit_heads_twice_with_different_names_on_one_body(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    fs.fit_body(stacks=["cubic"])
    fs.fit_heads("a", stacks=["cubic"])
    other = _config(tmp_path)  # same shape of config, passed explicitly
    fs.fit_heads("b", stacks=["cubic"], config=other)
    heads = {p.name for p in (fs.run_dir / "stacks" / "cubic" / "heads").iterdir()}
    assert heads == {"a", "b"}
    with pytest.raises(FileExistsError, match="a"):
        fs.fit_heads("a", stacks=["cubic"])


def test_fit_heads_requires_a_trained_body(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    with pytest.raises(FileNotFoundError, match="cubic"):
        fs.fit_heads("h", stacks=["cubic"])


def test_fit_heads_viz_dim_3(tmp_path, patched):
    d = {
        "output_dir": str(tmp_path / "runs"),
        "experts": {
            "defaults": _block(viz={"hidden_dim": [5], "viz_dim": 3}),
            "cubic": {},
        },
    }
    fs = FullStack.create(full_stack_config_from_dict(d), cache_dir=tmp_path / "cache")
    fs.fit_body()
    fs.fit_heads("h")
    heads_dir = fs.run_dir / "stacks" / "cubic" / "heads" / "h"
    assert np.load(heads_dir / "viz_embeddings.npz")["embeddings"].shape[1] == 3
    assert not (heads_dir / "viz_plot.png").exists()


def test_load_stack_returns_a_usable_single_stack(tmp_path, patched):
    fs = FullStack.create(_config(tmp_path), cache_dir=tmp_path / "cache")
    fs.fit_body(stacks=["cubic"])
    fs.fit_heads("h", stacks=["cubic"])
    stack = fs.load_stack("cubic", heads_name="h")
    X = np.load(fs.run_dir / "stacks" / "cubic" / "embeddings.npz")["features"]
    assert stack.predict_proba(X).shape == (len(X), 2)
    assert stack.encode(X).shape == (len(X), 4)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_pipeline_full_stack.py -v`
Expected: FAIL (`ModuleNotFoundError: dim_red.pipeline.full_stack`).

- [ ] **Step 3: Implement `src/dim_red/pipeline/full_stack.py`**

```python
"""``FullStack``: runs 1 family ``SingleStack`` plus up to 7 per-crystal-system
expert ``SingleStack``s, each completely independent -- its own pyxtal
dataset, split, standardization, config and seed. ``fit_body`` and
``fit_heads`` are separate so several head configurations can be trained on
the same saved bodies. A subset of the configured stacks can be trained (and
more can be added to the same run later).

Layout (``<run_dir>``)::

    config.yaml                       resolved FullStackConfig
    stacks/<name>/config.yaml         resolved StackSpec
                  dataset.extxyz  classes.yaml  embeddings.npz
                  body/   encoder_params, projection_params, stack.yaml, loss_history.csv
                  heads/<heads_name>/   classifier/viz params, heads.yaml, histories,
                                        predictions.npz, viz_embeddings.npz, viz_plot.png
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import yaml

from dim_red.analysis.plotting import plot_reduced_space
from dim_red.pipeline._common import (
    _build_vocab_ids,
    _make_unique_run_dir,
    _save_loss_history,
    _split_indices_grouped,
)
from dim_red.pipeline.full_stack_config import (
    FAMILY,
    FullStackConfig,
    StackSpec,
    full_stack_config_from_resolved_dict,
    full_stack_config_to_dict,
    stack_spec_to_dict,
)
from dim_red.pipeline.stack_data import StackDataset, build_stack_dataset
from dim_red.supcon.stack import SingleStack

logger = logging.getLogger("dim_red.pipeline")

History = Dict[str, List[float]]


def _labels_for(name: str, dataset: StackDataset) -> list:
    """The family stack contrasts on crystal systems, an expert on spacegroups."""
    if name == FAMILY:
        return list(dataset.labels)
    return [int(s) for s in dataset.spacegroups]


class FullStack:
    def __init__(
        self,
        run_dir: Union[str, Path],
        config: FullStackConfig,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.run_dir = Path(run_dir)
        self.config = config
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(config.output_dir) / "_dataset_cache"
        )

    # -- construction ----------------------------------------------------
    @classmethod
    def create(
        cls, config: FullStackConfig, cache_dir: Optional[Union[str, Path]] = None
    ) -> "FullStack":
        run_dir = _make_unique_run_dir(Path(config.output_dir), config.name)
        full_stack = cls(run_dir, config, cache_dir)
        full_stack._write_run_config()
        return full_stack

    @classmethod
    def open(
        cls,
        run_dir: Union[str, Path],
        config: Optional[FullStackConfig] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> "FullStack":
        run_dir = Path(run_dir)
        if config is None:
            with open(run_dir / "config.yaml") as f:
                config = full_stack_config_from_resolved_dict(yaml.safe_load(f))
        full_stack = cls(run_dir, config, cache_dir)
        full_stack._write_run_config()  # records stacks added through `config`
        return full_stack

    def _write_run_config(self) -> None:
        """Writes the resolved config, keeping stacks already recorded by
        earlier calls (a run can gain stacks over time)."""
        path = self.run_dir / "config.yaml"
        resolved = full_stack_config_to_dict(self.config)
        if path.exists():
            with open(path) as f:
                previous = yaml.safe_load(f) or {}
            merged = dict(previous.get("stacks", {}))
            merged.update(resolved["stacks"])
            resolved["stacks"] = merged
        with open(path, "w") as f:
            yaml.safe_dump(resolved, f, sort_keys=False)

    # -- helpers ---------------------------------------------------------
    def _select(self, stacks: Optional[Sequence[str]], config: FullStackConfig) -> List[str]:
        if stacks is None:
            return list(config.stacks)
        unknown = [s for s in stacks if s not in config.stacks]
        if unknown:
            raise ValueError(
                f"stack(s) {unknown} are not configured; configured stacks: "
                f"{list(config.stacks)}"
            )
        return list(stacks)

    def _stack_dir(self, name: str) -> Path:
        return self.run_dir / "stacks" / name

    @staticmethod
    def _validate(name: str, spec: StackSpec, classes: list, n_train: int) -> None:
        if len(classes) < 2:
            raise ValueError(
                f"stack {name!r}: its dataset has {len(classes)} class(es); at "
                "least 2 are needed"
            )
        if n_train < spec.min_train_rows:
            raise ValueError(
                f"stack {name!r}: only {n_train} training rows "
                f"(min_train_rows={spec.min_train_rows})"
            )

    # -- phase 1 ---------------------------------------------------------
    def fit_body(self, stacks: Optional[Sequence[str]] = None) -> Dict[str, History]:
        """Train encoder + projection of each selected stack (all configured
        stacks by default). Raises ``FileExistsError`` before training anything
        if one of them already has a trained body."""
        names = self._select(stacks, self.config)
        existing = [n for n in names if self._stack_dir(n).exists()]
        if existing:
            raise FileExistsError(
                f"stack(s) {existing} already trained in {self.run_dir}; "
                "select other stacks or use a new run"
            )
        histories = {}
        for name in names:
            histories[name] = self._fit_body_one(name, self.config.stacks[name])
        return histories

    def _fit_body_one(self, name: str, spec: StackSpec) -> History:
        dataset = build_stack_dataset(spec, self.config.model_kind, self.cache_dir)
        classes, y = _build_vocab_ids(_labels_for(name, dataset))
        sg_classes, sg_ids = _build_vocab_ids([int(s) for s in dataset.spacegroups])
        train_idx, val_idx = _split_indices_grouped(
            dataset.material_ids, spec.val_ratio, spec.seed
        )
        self._validate(name, spec, classes, len(train_idx))

        X = np.asarray(dataset.X, dtype=np.float32)
        stack = SingleStack(X.shape[1], spec.model)
        history = stack.fit_body(
            X[train_idx],
            X[val_idx],
            y[train_idx],
            y[val_idx],
            sub_train=sg_ids[train_idx] if name == FAMILY else None,
        )
        logger.info(
            "Stack %r: body trained on %d rows (%d classes), %d epochs",
            name,
            len(train_idx),
            len(classes),
            len(history["train_loss"]),
        )

        stack_dir = self._stack_dir(name)
        stack_dir.mkdir(parents=True)
        stack.save_body(stack_dir / "body")
        _save_loss_history(stack_dir / "body" / "loss_history.csv", history)
        shutil.copy(dataset.structures_path, stack_dir / "dataset.extxyz")
        split = np.full(len(X), "train", dtype="<U5")
        split[val_idx] = "val"
        np.savez(
            stack_dir / "embeddings.npz",
            embeddings=stack.encode(X),
            features=X,
            labels=np.asarray([str(v) for v in _labels_for(name, dataset)]),
            label_ids=y,
            spacegroups=np.asarray(dataset.spacegroups, dtype=np.int64),
            spacegroup_ids=sg_ids,
            material_ids=np.asarray(dataset.material_ids),
            split=split,
            feature_mean=dataset.feature_mean,
            feature_std=dataset.feature_std,
        )
        with open(stack_dir / "classes.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "role": "family" if name == FAMILY else "spacegroup",
                    "classes": [str(c) if name == FAMILY else int(c) for c in classes],
                },
                f,
                sort_keys=False,
            )
        with open(stack_dir / "config.yaml", "w") as f:
            yaml.safe_dump(stack_spec_to_dict(spec), f, sort_keys=False)
        return history

    # -- phase 2 ---------------------------------------------------------
    def fit_heads(
        self,
        heads_name: str,
        stacks: Optional[Sequence[str]] = None,
        config: Optional[FullStackConfig] = None,
    ) -> Dict[str, Dict[str, History]]:
        """Train classification + visualization tails on the saved, frozen
        bodies. ``config`` (optional) supplies different head hyperparameters;
        by default the run's own are used. Several ``heads_name``s can coexist
        on the same body."""
        cfg = config if config is not None else self.config
        names = self._select(stacks, cfg)
        for name in names:
            if not (self._stack_dir(name) / "body" / "stack.yaml").exists():
                raise FileNotFoundError(
                    f"stack {name!r} has no trained body in {self.run_dir}: run "
                    "fit_body first"
                )
            if (self._stack_dir(name) / "heads" / heads_name).exists():
                raise FileExistsError(
                    f"heads {heads_name!r} already exist for stack {name!r}"
                )
        return {name: self._fit_heads_one(name, heads_name, cfg.stacks[name]) for name in names}

    def _fit_heads_one(
        self, name: str, heads_name: str, spec: StackSpec
    ) -> Dict[str, History]:
        stack_dir = self._stack_dir(name)
        emb = np.load(stack_dir / "embeddings.npz")
        with open(stack_dir / "classes.yaml") as f:
            classes = yaml.safe_load(f)["classes"]
        split = emb["split"]
        train = np.flatnonzero(split == "train")
        val = np.flatnonzero(split == "val")
        X, y, sg_ids = emb["features"], emb["label_ids"], emb["spacegroup_ids"]

        stack = SingleStack.load_body(stack_dir / "body")
        histories = stack.fit_heads(
            X[train],
            X[val],
            y[train],
            y[val],
            n_classes=len(classes),
            sub_train=sg_ids[train] if name == FAMILY else None,
            config=spec.model,
        )

        heads_dir = stack_dir / "heads" / heads_name
        stack.save_heads(heads_dir)
        _save_loss_history(heads_dir / "classifier_loss_history.csv", histories["classifier"])
        _save_loss_history(heads_dir / "viz_loss_history.csv", histories["visualization"])
        np.savez(
            heads_dir / "predictions.npz",
            material_ids=emb["material_ids"],
            split=split,
            label_ids=y,
            probs=stack.predict_proba(X),
        )
        z = stack.visualize(X)
        np.savez(
            heads_dir / "viz_embeddings.npz",
            embeddings=z,
            label_ids=y,
            material_ids=emb["material_ids"],
            split=split,
        )
        if z.shape[1] == 2:
            plot_reduced_space(
                z,
                [str(classes[i]) for i in y],
                title=f"{name} / {heads_name}",
                save_path=str(heads_dir / "viz_plot.png"),
                legend=False,
            )
        return histories

    # -- loading ---------------------------------------------------------
    def load_stack(self, name: str, heads_name: Optional[str] = None) -> SingleStack:
        stack = SingleStack.load_body(self._stack_dir(name) / "body")
        if heads_name is not None:
            stack.load_heads(self._stack_dir(name) / "heads" / heads_name)
        return stack
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_pipeline_full_stack.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/dim_red/pipeline/full_stack.py tests/test_pipeline_full_stack.py
git commit -m "Add FullStack: independent stacks with fit_body/fit_heads, subset selection and per-stack layout

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: End-to-end smoke test (slow, opt-in)

**Files:**
- Modify: `tests/conftest.py`
- Test: `tests/test_pipeline_full_stack_e2e.py`

**Interfaces:**
- Consumes: `FullStack`, `full_stack_config_from_dict`; real `pyxtal` + `dscribe` (no mocks).
- Produces: the `--runslow` pytest option and the `slow` marker.

- [ ] **Step 1: Add the opt-in mechanism to `tests/conftest.py`**

Append:

```python
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run tests marked slow"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: end-to-end tests that really train (need --runslow)"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
```

- [ ] **Step 2: Write `tests/test_pipeline_full_stack_e2e.py`**

```python
"""Real (unmocked) FullStack run on a tiny pyxtal configuration. Slow: run
with ``pytest tests/test_pipeline_full_stack_e2e.py --runslow``."""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("pyxtal")
pytest.importorskip("dscribe")

from dim_red.pipeline.full_stack import FullStack
from dim_red.pipeline.full_stack_config import full_stack_config_from_dict


def _block(spacegroups, per_sg):
    return {
        "data": {
            "pyxtal": {"spacegroups": spacegroups, "structures_per_spacegroup": per_sg},
            "soap": {"r_cut": 3.0, "n_max": 2, "l_max": 2},
        },
        "encoder": {"encoder_hidden_dim": [16], "latent_dim": 4},
        "projection": {"projection_dim": 8},
        "train": {"epochs": 30, "batch_size": 8, "learning_rate": 1e-2},
        "classifier": {"hidden_dim": 8},
        "viz": {"hidden_dim": [8]},
        "min_train_rows": 6,
    }


@pytest.mark.slow
def test_full_stack_trains_end_to_end_on_tiny_pyxtal(tmp_path):
    config = full_stack_config_from_dict(
        {
            "name": "e2e",
            "seed": 0,
            "output_dir": str(tmp_path / "runs"),
            "family": _block([16, 75, 195], 8),
            "experts": {
                "cubic": _block([195, 200], 8),
                "tetragonal": _block([75, 81], 8),
            },
        }
    )
    fs = FullStack.create(config)
    bodies = fs.fit_body()
    heads = fs.fit_heads("h")

    assert set(bodies) == {"family", "cubic", "tetragonal"}
    for name, history in bodies.items():
        assert np.isfinite(history["train_loss"]).all(), name
    for name, result in heads.items():
        clf = result["classifier"]["train_loss"]
        assert np.isfinite(clf).all() and clf[-1] < clf[0], name
        preds = np.load(fs.run_dir / "stacks" / name / "heads" / "h" / "predictions.npz")
        np.testing.assert_allclose(preds["probs"].sum(axis=1), 1.0, atol=1e-4)
```

- [ ] **Step 3: Verify it is skipped by default, then run it once explicitly**

Run: `pytest tests/test_pipeline_full_stack_e2e.py -v`
Expected: 1 skipped ("need --runslow to run").

Run: `pytest tests/test_pipeline_full_stack_e2e.py --runslow -v`
Expected: 1 passed (may take a few minutes: pyxtal generation + SOAP + training). If pyxtal cannot generate one of the space groups (a warning in the log, or `ValueError: pyxtal generated no structures`), swap that space group for another of the same system (cubic 195–230, tetragonal 75–142) rather than loosening assertions. If `clf[-1] < clf[0]` is flaky, raise `epochs` before touching the assertion.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_pipeline_full_stack_e2e.py
git commit -m "Add opt-in slow end-to-end FullStack smoke test (--runslow)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Plan self-review

- **Spec coverage:** D1 (`fit_body`/`fit_heads`) → Task 1; D2 (8 independent stacks, own dataset/split/standardization) → Tasks 3–4; D3 (pyxtal only) → Task 2 parser requires `data.pyxtal`; D4 → true-family datasets come from per-system pyxtal restriction (Task 2) and expert labels = spacegroup (Task 4); D5 (subset/add later) → Task 4 tests; D7 (no `labels:` in YAML) → Task 2 keys; D8 (`defaults`) → Task 2; D9 (no fallback, error) → Task 4 `_validate`; D10/D11 and test levels 1–3 → Tasks 0, 1, 5; layout (spec §Layout) → Task 4. **Deferred to Plan B (stated in the header):** `predict` with routing, legacy reader, CLI/sweep wiring and removals, classifier-evaluation plots, config/doc migration, `family_expert_status`-style summaries.
- **Placeholders:** none; the only "read then replace" step is the `_resolve_augmentation` refactor, which names the exact function and gives the full replacement.
- **Type consistency:** `StackConfig`, `Batching`, `SingleStack` method names/signatures, `StackSpec` fields, `FullStack` method names and layout file names are used identically in Tasks 1–5.
- **Behavior difference to be aware of (not a bug):** the family stack's old "family_and_spacegroup" visualization default is gone (spec D7: one label per stack); `fit_body` history keys use `*_family_supcon` for expert stacks too (label always goes through the family slot).
