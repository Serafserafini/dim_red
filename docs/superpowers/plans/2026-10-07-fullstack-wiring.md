# FullStack Wiring Implementation Plan (Plan B of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `FullStack` (plan A, already on this branch) the only way to train, read, compare, benchmark, sweep and apply `supcon`/`supcon_mace` runs; remove the old supcon training paths; migrate configs, scripts, notebook, slurm scripts and docs.

**Architecture:** One jax-free module `pipeline/run_layout.py` knows where files live in a run directory and exposes each stack as a `RunData` (the same duck-typed record `compare`/`benchmark` already consume), so both modules keep their plotting code and only change how they enumerate stacks. `FullStack` gains `predict`/`predict_stack` (features computed once per structure with SOAP `mu2`, standardized per stack) and `apply_to_structures`. CLI commands dispatch on the config's top-level key (`model_kind` = FullStack, `model:` = cgcnn). Old supcon code is deleted last, after the new paths are green.

**Tech Stack:** Python, JAX/Flax (existing), numpy, PyYAML, matplotlib, pytest. Run everything inside the `dmred` conda env (`conda activate dmred`, check `$CONDA_DEFAULT_ENV`).

**Spec:** `docs/superpowers/specs/2026-10-07-fullstack-wiring-design.md` (plus the 6 deviations recorded in Task 0). Plan A: `docs/superpowers/plans/2026-10-06-singlestack-fullstack-core.md`.

## Global Constraints

- Work in this worktree (`.claude/worktrees/singlestack-fullstack`, branch `feature/singlestack-fullstack`). Never `cd` to the main checkout.
- Never run the full `pytest` suite on your own initiative; run only the test files named in each task (project rule). The `slow` end-to-end test only runs with `--runslow`.
- Do not `pip install` anything; everything needed is in `dmred`.
- No backward compatibility with old runs (spec B1): there is exactly one run layout, `stacks/<name>/`. `cgcnn` keeps its flat layout and is not read by `compare`/`benchmark` (B3).
- `dim_red.pipeline.run_layout`, `compare`, `benchmark` and `cli` must stay importable **without jax** (`dimred-compare` never needs jax). Only `full_stack*.py`, `featurize.py`, `sweep.py` and `supcon/stack.py` may import jax.
- `FullStackConfig` requires `data.soap.element_agnostic: true` for `model_kind: supcon` (spec B4).
- The three training routines and `ProjectionTail`/`VisualizationTail` are not modified (plan A D11). History keys `train_family_supcon`/`val_family_supcon` stay as they are in the CSVs (spec B6); only plot labels change.
- Pre-commit runs black/isort; if it reformats files, `git add` them again and re-run the commit. Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. One commit per task. Do not re-run tests after a formatting-only pass.
- A saved/unsaved stack name is one of `("family", "triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic")`; directories named `.<x>.tmp` (partial atomic writes) are never stacks or heads.

## Review Focus

Conditions the spec implies but the happy path does not exercise; each is pinned by a test in the owning task.

1. A run interrupted mid-write leaves `stacks/.cubic.tmp` or `heads/.h.tmp`: no reader may list it as a stack/heads — Task 3 (`test_hidden_tmp_dirs_are_never_stacks_or_heads`).
2. A run trained on GPU is opened on a CPU-only machine: `load(device="cpu")` must work while the saved `train.device` says `gpu` — Task 2.
3. Two different heads exist for a stack and the caller does not name one: error listing the names, never a silent pick — Task 3 and Task 4.
4. `predict` on a structure whose family has no trained expert: spacegroup is an explicit `None`, never a guess — Task 4.
5. A new structure containing an element absent from every training dataset must featurize identically to training (`mu2`) — Task 4 (`test_featurization_ignores_other_structures_in_the_batch`).
6. `compare` over runs that do not all contain the same stacks: skip with a log line, never silently or with an exception — Task 6.
7. A config key typo / old-schema YAML must fail at load, and every shipped example YAML must load — Task 11 (`test_example_configs_load`).

## File Structure

| File | Responsibility |
|---|---|
| `src/dim_red/pipeline/run_layout.py` (new) | Stack-name constants, layout helpers, `RunData`, `open_stack`, `discover_full_stack_runs`. jax-free. |
| `src/dim_red/pipeline/featurize.py` (new) | Raw features of new structures (SOAP `mu2` or MACE) + standardization. |
| `src/dim_red/pipeline/full_stack.py` | Adds `predict`, `predict_stack`, `stack_names`, `load_stack(device=)`, `apply_to_structures`. |
| `src/dim_red/pipeline/full_stack_config.py` | `mu2` validation; constants now imported from `run_layout`. |
| `src/dim_red/supcon/stack.py` | `load_body`/`load_heads` accept `device`. |
| `src/dim_red/pipeline/compare.py`, `benchmark.py` | Per-stack suites; use `RunData` from `run_layout`. |
| `src/dim_red/pipeline/sweep.py` | Rewritten: FullStack-only grid sweep. |
| `src/dim_red/pipeline/cli.py`, `pyproject.toml` | Dispatch by config kind; new `dimred-train-heads`. |
| `src/dim_red/pipeline/{single_run,tail_training,inference,config}.py` | Old supcon paths removed (cgcnn kept). |
| `tests/fullstack_helpers.py` (new) | Tiny trained runs with a fake dataset builder; `write_fake_run` for compare/benchmark. |
| `configs/`, `examples/`, `slurm/`, `docs/`, `CLAUDE.md`s | Migration. |

---

### Task 0: Record spec deviations

Findings made while reading the code for this plan; the spec must say what will actually be built.

**Files:**
- Modify: `docs/superpowers/specs/2026-10-07-fullstack-wiring-design.md`

- [ ] **Step 1: Apply these six edits to the spec**

1. Section 4, `dimred-sweep` bullet: delete the sentence "La regola 'salva le feature solo per il primo run con lo stesso dataset' passa a livello di stack." and add: "`fit_heads` ricalcola le head dalle `features` salvate in `embeddings.npz` di ogni stack, quindi le feature si salvano sempre; la deduplicazione dello spazio su disco per dataset condiviso non si porta nel nuovo layout. Lo sweep scrive `sweep.yaml` (base + grid) nella cartella dello sweep al posto del README con le differenze dai default."
2. Section 4, "Si cancella": replace "Prima della cancellazione `_classifier_eval_plots` si sposta in un modulo condiviso." with "`_classifier_eval_plots` resta in `tail_training.py`: lo usa ancora `train_tail` per cgcnn. `FullStack` non produce matrici di confusione (fuori perimetro)."
3. Section 3, last bullet: replace with "`dimred-apply` su una run nuova scrive `<stem>_predictions.csv` e `<stem>_viz.npz` in `<run_dir>/applied` (o `--output-dir`); non disegna più il grafico nello spazio latente."
4. Section 4, `dimred-train-heads`: add "Il YAML ha lo stesso schema di `dimred-run`; di ogni stack elencato si leggono solo i blocchi `classifier`/`viz`/`seed` (gli altri devono esserci per la validazione ma sono ignorati). `--heads-name` è obbligatorio."
5. Section 2, `compare`: add "I plot di `compare` usano il solo `loss_history.csv` del corpo di ogni stack; le head servono per i grafici di accuratezza e per la viz 2D. Se per una run esiste più di un set di head serve `--heads-name`; se non ne esiste nessuno i grafici di accuratezza sono saltati."
6. Section 2, `benchmark`: add "La colonna `wall_clock_seconds` sparisce (le nuove run non scrivono `run.log`). Per gli stack esperti le metriche di qualità `family_*` non esistono (solo `spacegroup_*`)."

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-10-07-fullstack-wiring-design.md
git commit -m "Record plan B spec deviations found while reading the consumers

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 1: `mu2` is mandatory for `supcon`

**Files:**
- Modify: `src/dim_red/pipeline/full_stack_config.py` (function `full_stack_config_from_dict`)
- Modify (fixtures): `tests/test_pipeline_full_stack_config.py`, `tests/test_pipeline_full_stack.py`, `tests/test_pipeline_stack_data.py`, `tests/test_pipeline_full_stack_e2e.py`
- Test: `tests/test_pipeline_full_stack_config.py`

**Interfaces:**
- Produces: `full_stack_config_from_dict` raises `ValueError` containing `element_agnostic` when `model_kind == "supcon"` and any stack has `data.soap.element_agnostic == False`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_pipeline_full_stack_config.py`)

```python
def test_supcon_requires_element_agnostic_soap():
    d = _config()
    d["family"]["data"]["soap"] = {"element_agnostic": False}
    with pytest.raises(ValueError, match="element_agnostic"):
        full_stack_config_from_dict(d)


def test_supcon_default_soap_is_rejected_with_a_fix_hint():
    d = _config()
    del d["family"]["data"]["soap"]
    with pytest.raises(ValueError, match="stack 'family'.*element_agnostic: true"):
        full_stack_config_from_dict(d)


def test_supcon_mace_does_not_need_mu2():
    d = _config(model_kind="supcon_mace")
    for block in (d["family"], d["experts"]["defaults"]):
        block["data"].pop("soap", None)
        block["data"]["mace"] = {"checkpoint_path": "x.msgpack"}
    cfg = full_stack_config_from_dict(d)
    assert cfg.model_kind == "supcon_mace"
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_pipeline_full_stack_config.py -k "element_agnostic or mu2 or default_soap" -v`
Expected: FAIL (`DID NOT RAISE`).

- [ ] **Step 3: Implement.** In `full_stack_config_from_dict`, directly after the `stacks = {...}` dict comprehension and before the `if model_kind == "supcon_mace":` block, insert:

```python
    if model_kind == "supcon":
        for name, spec in stacks.items():
            if not spec.data.soap.element_agnostic:
                raise ValueError(
                    f"stack {name!r}: model_kind supcon needs "
                    "data.soap.element_agnostic: true (SOAP 'mu2' compression), "
                    "so a stack's features depend only on geometry, never on "
                    "which chemical species its dataset happened to contain"
                )
```

- [ ] **Step 4: Fix the fixtures.** Every test helper that builds a `supcon` config must now enable `mu2`. Find them:

Run: `grep -n '"pyxtal"' tests/test_pipeline_full_stack_config.py tests/test_pipeline_full_stack.py tests/test_pipeline_stack_data.py tests/test_pipeline_full_stack_e2e.py`

In each `"data": {"pyxtal": {...}}` dict add `"soap": {"element_agnostic": True}` next to `"pyxtal"` (for `test_supcon_mace_does_not_need_mu2` above the `soap` key is removed on purpose). Then:

Run: `pytest tests/test_pipeline_full_stack_config.py tests/test_pipeline_full_stack.py tests/test_pipeline_stack_data.py -v`
Expected: PASS. (`test_pipeline_full_stack_e2e.py` is `slow`; check it collects: `pytest tests/test_pipeline_full_stack_e2e.py -v` → skipped.)

- [ ] **Step 5: Commit**

```bash
git add src/dim_red/pipeline/full_stack_config.py tests/test_pipeline_full_stack_config.py tests/test_pipeline_full_stack.py tests/test_pipeline_stack_data.py tests/test_pipeline_full_stack_e2e.py
git commit -m "Require SOAP mu2 (element_agnostic) for supcon FullStack configs

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Device override when loading

**Files:**
- Modify: `src/dim_red/supcon/stack.py` (`load_body`, `load_heads`)
- Modify: `src/dim_red/pipeline/full_stack.py` (`load_stack`)
- Test: `tests/test_supcon_stack.py`

**Interfaces:**
- Produces: `SingleStack.load_body(directory, device: Optional[str] = None)`, `SingleStack.load_heads(directory, device: Optional[str] = None)`, `FullStack.load_stack(name, heads_name=None, device=None)`. `device=None` keeps today's behaviour (the saved `train.device`).

- [ ] **Step 1: Write the failing test** (append to `tests/test_supcon_stack.py`)

```python
def _gpu_available():
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


def test_device_override_loads_a_gpu_trained_stack_on_cpu(tmp_path):
    X, y, tr, va = _toy()
    stack = SingleStack(X.shape[1], _config())
    stack.fit_body(X[tr], X[va], y[tr], y[va])
    stack.fit_heads(X[tr], X[va], y[tr], y[va], n_classes=3)
    stack.save_body(tmp_path / "body")
    stack.save_heads(tmp_path / "heads")
    expected = stack.visualize(X)

    # Pretend it was trained on a GPU: only the recorded device changes.
    for name, key in (("body/stack.yaml", "body_train"), ("heads/heads.yaml", "viz_train")):
        path = tmp_path / name
        meta = yaml.safe_load(path.read_text())
        for train_key in ("body_train", "classifier_train", "viz_train"):
            meta["config"][train_key]["device"] = "gpu"
        path.write_text(yaml.safe_dump(meta, sort_keys=False))

    if not _gpu_available():
        with pytest.raises(Exception):
            SingleStack.load_body(tmp_path / "body")

    loaded = SingleStack.load_body(tmp_path / "body", device="cpu")
    loaded.load_heads(tmp_path / "heads", device="cpu")
    np.testing.assert_allclose(loaded.visualize(X), expected, rtol=1e-5, atol=1e-6)
```

Add `import yaml` to the file's imports if missing.

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_supcon_stack.py::test_device_override_loads_a_gpu_trained_stack_on_cpu -v`
Expected: FAIL (`TypeError: load_body() got an unexpected keyword argument 'device'`).

- [ ] **Step 3: Implement** in `src/dim_red/supcon/stack.py`:

`load_body` becomes:

```python
    @classmethod
    def load_body(cls, directory, device: Optional[str] = None) -> "SingleStack":
        """``device`` overrides the device the params were trained on (e.g.
        ``"cpu"`` to open a GPU-trained stack on a CPU-only machine)."""
        directory = Path(directory)
        with open(directory / "stack.yaml") as f:
            meta = yaml.safe_load(f)
        stack = cls(meta["input_dim"], stack_config_from_dict(meta["config"]))
        where = device or stack.config.body_train.device
        stack.encoder.params = _read_params(
            directory / "encoder_params.msgpack", stack.encoder.params, where
        )
        stack.projection.params = _read_params(
            directory / "projection_params.msgpack", stack.projection.params, where
        )
        return stack
```

`load_heads` becomes:

```python
    def load_heads(self, directory, device: Optional[str] = None) -> None:
        directory = Path(directory)
        with open(directory / "heads.yaml") as f:
            meta = yaml.safe_load(f)
        self._build_heads(stack_config_from_dict(meta["config"]), meta["n_classes"])
        heads = self._heads_config
        self.classifier.params = _read_params(
            directory / "classifier_params.msgpack",
            self.classifier.params,
            device or heads.classifier_train.device,
        )
        self.visualizer.params = _read_params(
            directory / "viz_params.msgpack",
            self.visualizer.params,
            device or heads.viz_train.device,
        )
```

In `src/dim_red/pipeline/full_stack.py` replace `load_stack`:

```python
    def load_stack(
        self,
        name: str,
        heads_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> SingleStack:
        stack = SingleStack.load_body(self._stack_dir(name) / "body", device=device)
        if heads_name is not None:
            stack.load_heads(
                self._stack_dir(name) / "heads" / heads_name, device=device
            )
        return stack
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_supcon_stack.py tests/test_supcon_stack_golden.py tests/test_pipeline_full_stack.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dim_red/supcon/stack.py src/dim_red/pipeline/full_stack.py tests/test_supcon_stack.py
git commit -m "Add a load-time device override so GPU-trained stacks open on CPU machines

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `run_layout` (jax-free reader) and test helpers

**Files:**
- Create: `src/dim_red/pipeline/run_layout.py`
- Modify: `src/dim_red/pipeline/full_stack_config.py` (import the three constants from `run_layout`)
- Create: `tests/fullstack_helpers.py`
- Test: `tests/test_pipeline_run_layout.py`

**Interfaces:**
- Produces (all in `dim_red.pipeline.run_layout`, no jax):
  - `FAMILY: str`, `EXPERT_NAMES: Tuple[str, ...]`, `STACK_ORDER: Tuple[str, ...]`
  - `is_hidden_tmp(path) -> bool`
  - `is_full_stack_run(path) -> bool`
  - `discover_full_stack_runs(sweep_dir) -> List[Path]` (raises `ValueError` if none)
  - `trained_stack_names(run_dir) -> List[str]` (canonical order; a stack is trained iff `body/stack.yaml` exists)
  - `head_names(stack_dir) -> List[str]` (dirs with `heads.yaml`)
  - `resolve_heads_name(stack_dir, heads_name=None) -> str`
  - `load_loss_history(path) -> Dict[str, np.ndarray]`
  - `@dataclass RunData(run_dir, stack, model_kind, flat_config, loss_history, embeddings, viz_embeddings=None, heads_name=None)` with property `label -> run_dir.name`
  - `open_stack(run_dir, stack, heads_name=None, allow_no_heads=False) -> RunData`
- `RunData.embeddings` is the stack's `embeddings.npz` as a dict, **plus**: family stack → `family_probs` (n, k), `family_classes` (k, str); expert stack → `spacegroup_probs`, `spacegroup_classes` (int64), and `labels` overridden to the constant `<Stack>` (e.g. `"Cubic"`) so family-colored plots work.
- `RunData.run_dir` is the **run root**; `RunData.stack` names the stack.

- [ ] **Step 1: Write the test helpers** `tests/fullstack_helpers.py`

```python
"""Helpers for the FullStack wiring tests: tiny *trained* runs built without
pyxtal/SOAP (a fake dataset builder), and a writer of fake FullStack-layout
run directories (no training) for compare/benchmark tests."""

import csv
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.io import write

from dim_red.pipeline.full_stack import FullStack
from dim_red.pipeline.full_stack_config import FAMILY, full_stack_config_from_dict
from dim_red.pipeline.stack_data import StackDataset

SYSTEM_SPACEGROUPS = {
    "cubic": [195, 196],
    "tetragonal": [75, 76],
    "hexagonal": [168, 169],
}


def block(learning_rate=1e-3, latent_dim=4, epochs=2, **over):
    b = {
        "data": {
            "pyxtal": {"structures_per_spacegroup": 2},
            "soap": {"element_agnostic": True},
        },
        "encoder": {"encoder_hidden_dim": [8], "latent_dim": latent_dim},
        "projection": {"projection_dim": 5},
        "train": {"epochs": epochs, "batch_size": 8, "learning_rate": learning_rate},
        "classifier": {"hidden_dim": 6},
        "viz": {"hidden_dim": [5]},
        "min_train_rows": 5,
    }
    b.update(over)
    return b


def config_dict(
    output_dir,
    name="fs",
    experts=("cubic", "tetragonal"),
    with_family=True,
    **block_kwargs,
):
    d = {
        "name": name,
        "seed": 3,
        "output_dir": str(output_dir),
        "experts": {"defaults": block(**block_kwargs), **{e: {} for e in experts}},
    }
    if with_family:
        d["family"] = block(**block_kwargs)
    return d


def fake_builder(tmp_path, calls, rows_per_sg=12):
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
            plan = [
                (spec.name.capitalize(), sg) for sg in SYSTEM_SPACEGROUPS[spec.name]
            ]
        X, labels, material_ids, spacegroups = [], [], [], []
        for _, (family, sg) in enumerate(plan):
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


def make_run(
    tmp_path,
    monkeypatch,
    *,
    output_dir=None,
    name="fs",
    experts=("cubic", "tetragonal"),
    with_family=True,
    heads=("default",),
    **block_kwargs,
) -> FullStack:
    """Train a tiny FullStack run (body + the named heads) on fake datasets."""
    calls = []
    monkeypatch.setattr(
        "dim_red.pipeline.full_stack.build_stack_dataset",
        fake_builder(tmp_path, calls),
    )
    cfg = full_stack_config_from_dict(
        config_dict(
            output_dir if output_dir is not None else tmp_path / "runs",
            name=name,
            experts=experts,
            with_family=with_family,
            **block_kwargs,
        )
    )
    fs = FullStack.create(cfg, cache_dir=tmp_path / "cache")
    fs.fit_body()
    for heads_name in heads:
        fs.fit_heads(heads_name)
    return fs


def _write_loss_history(path, epochs=3, val_loss_final=1.0, aux=False):
    fields = ["epoch", "train_loss", "val_loss", "train_family_supcon", "val_family_supcon"]
    if aux:
        fields += ["train_family_ce", "val_family_ce"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            val = val_loss_final + (epochs - epoch) * 0.1
            row = {
                "epoch": epoch,
                "train_loss": val + 0.05,
                "val_loss": val,
                "train_family_supcon": val * 0.9,
                "val_family_supcon": val * 0.8,
            }
            if aux:
                row.update(train_family_ce=0.2, val_family_ce=0.25)
            writer.writerow(row)


def write_fake_run(
    run_dir,
    *,
    stacks=("family",),
    hidden_dim=(4,),
    latent_dim=2,
    learning_rate=1e-3,
    val_loss_final=1.0,
    n=8,
    heads=("default",),
    families=("Cubic",),
    model_kind="supcon",
    n_features=None,
    aux=False,
) -> Path:
    """A FullStack-layout run directory with fake (untrained) contents: every
    file ``open_stack`` reads, nothing else. ``heads=()`` writes no heads."""
    run_dir = Path(run_dir)
    rng = np.random.default_rng(0)
    specs = {}
    for stack in stacks:
        spec = {
            "name": stack,
            "seed": 0,
            "val_ratio": 0.25,
            "min_train_rows": 5,
            "data": {"pyxtal": {"families": list(families)}},
            "model": {
                "encoder_hidden_dim": list(hidden_dim),
                "latent_dim": latent_dim,
                "body_train": {
                    "learning_rate": learning_rate,
                    "epochs": 3,
                    "batch_size": 4,
                },
            },
        }
        specs[stack] = spec
        stack_dir = run_dir / "stacks" / stack
        (stack_dir / "body").mkdir(parents=True)
        (stack_dir / "body" / "stack.yaml").write_text("{}\n")
        _write_loss_history(
            stack_dir / "body" / "loss_history.csv",
            val_loss_final=val_loss_final,
            aux=aux,
        )
        (stack_dir / "config.yaml").write_text(yaml.safe_dump(spec))

        if stack == FAMILY:
            classes = list(families)
            labels = np.array([classes[i % len(classes)] for i in range(n)])
            spacegroups = np.array([195 + (i % 2) for i in range(n)], dtype=np.int64)
            label_ids = np.array([i % len(classes) for i in range(n)])
            role = "family"
        else:
            classes = [195, 196]
            spacegroups = np.array([classes[i % 2] for i in range(n)], dtype=np.int64)
            labels = np.array([str(s) for s in spacegroups])
            label_ids = np.array([i % 2 for i in range(n)])
            role = "spacegroup"
        (stack_dir / "classes.yaml").write_text(
            yaml.safe_dump({"role": role, "classes": classes})
        )
        split = np.array(["train"] * (n - 2) + ["val"] * 2)
        payload = dict(
            embeddings=rng.normal(size=(n, latent_dim)).astype(np.float32),
            labels=labels,
            label_ids=label_ids,
            spacegroups=spacegroups,
            spacegroup_ids=label_ids,
            material_ids=np.array([f"{stack}-{i}" for i in range(n)]),
            split=split,
            feature_mean=np.zeros(3, dtype=np.float32),
            feature_std=np.ones(3, dtype=np.float32),
        )
        if n_features is not None:
            payload["features"] = rng.normal(size=(n, n_features)).astype(np.float32)
        np.savez(stack_dir / "embeddings.npz", **payload)

        for heads_name in heads:
            heads_dir = stack_dir / "heads" / heads_name
            heads_dir.mkdir(parents=True)
            (heads_dir / "heads.yaml").write_text("{}\n")
            probs = np.eye(len(classes), dtype=np.float32)[label_ids]
            np.savez(
                heads_dir / "predictions.npz",
                material_ids=payload["material_ids"],
                split=split,
                label_ids=label_ids,
                probs=probs,
            )
            np.savez(
                heads_dir / "viz_embeddings.npz",
                embeddings=rng.normal(size=(n, 2)).astype(np.float32),
                label_ids=label_ids,
                material_ids=payload["material_ids"],
                split=split,
            )
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "name": run_dir.name,
                "seed": 0,
                "model_kind": model_kind,
                "output_dir": str(run_dir.parent),
                "stacks": specs,
            }
        )
    )
    return run_dir
```

- [ ] **Step 2: Write the failing tests** `tests/test_pipeline_run_layout.py`

```python
import numpy as np
import pytest

from dim_red.pipeline.compare import classification_accuracies_from_npz
from dim_red.pipeline.run_layout import (
    FAMILY,
    STACK_ORDER,
    discover_full_stack_runs,
    head_names,
    is_full_stack_run,
    is_hidden_tmp,
    open_stack,
    resolve_heads_name,
    trained_stack_names,
)
from tests.fullstack_helpers import write_fake_run


def test_constants_are_the_canonical_stack_order():
    assert STACK_ORDER[0] == FAMILY
    assert STACK_ORDER[-1] == "cubic"
    assert len(STACK_ORDER) == 8


def test_hidden_tmp_dirs_are_never_stacks_or_heads(tmp_path):
    run = write_fake_run(tmp_path / "r", stacks=("family", "cubic"))
    # an interrupted atomic write leaves these behind
    (run / "stacks" / ".tetragonal.tmp" / "body").mkdir(parents=True)
    (run / "stacks" / ".tetragonal.tmp" / "body" / "stack.yaml").write_text("{}")
    heads = run / "stacks" / "family" / "heads"
    (heads / ".viz2.tmp").mkdir()
    (heads / ".viz2.tmp" / "heads.yaml").write_text("{}")
    assert is_hidden_tmp(heads / ".viz2.tmp")
    assert not is_hidden_tmp(heads / "default")
    assert trained_stack_names(run) == ["family", "cubic"]
    assert head_names(run / "stacks" / "family") == ["default"]


def test_trained_stack_names_use_canonical_order(tmp_path):
    run = write_fake_run(tmp_path / "r", stacks=("cubic", "family", "triclinic"))
    assert trained_stack_names(run) == ["family", "triclinic", "cubic"]


def test_a_stack_without_a_body_is_not_trained(tmp_path):
    run = write_fake_run(tmp_path / "r", stacks=("family", "cubic"))
    (run / "stacks" / "cubic" / "body" / "stack.yaml").unlink()
    assert trained_stack_names(run) == ["family"]


def test_discover_full_stack_runs(tmp_path):
    write_fake_run(tmp_path / "s" / "a")
    write_fake_run(tmp_path / "s" / "b")
    (tmp_path / "s" / "not_a_run").mkdir()
    (tmp_path / "s" / ".c.tmp").mkdir()
    assert [p.name for p in discover_full_stack_runs(tmp_path / "s")] == ["a", "b"]
    assert is_full_stack_run(tmp_path / "s" / "a")
    assert not is_full_stack_run(tmp_path / "s" / "not_a_run")


def test_discover_full_stack_runs_raises_when_empty(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No completed FullStack runs"):
        discover_full_stack_runs(tmp_path / "empty")


def test_resolve_heads_name_requires_a_choice_when_ambiguous(tmp_path):
    run = write_fake_run(tmp_path / "r", heads=("a", "b"))
    stack_dir = run / "stacks" / "family"
    with pytest.raises(ValueError, match=r"several heads.*\['a', 'b'\]"):
        resolve_heads_name(stack_dir)
    assert resolve_heads_name(stack_dir, "b") == "b"
    with pytest.raises(ValueError, match="no heads named 'zzz'"):
        resolve_heads_name(stack_dir, "zzz")


def test_resolve_heads_name_with_none_trained(tmp_path):
    run = write_fake_run(tmp_path / "r", heads=())
    with pytest.raises(ValueError, match="no heads trained"):
        resolve_heads_name(run / "stacks" / "family")


def test_open_family_stack_exposes_classification_payload(tmp_path):
    run = write_fake_run(tmp_path / "r", families=("Cubic", "Hexagonal"), n=8)
    data = open_stack(run, FAMILY)
    assert data.stack == FAMILY and data.run_dir == run
    assert data.model_kind == "supcon" and data.heads_name == "default"
    assert data.label == "r"
    assert data.viz_embeddings.shape == (8, 2)
    assert data.flat_config["model.latent_dim"] == 2
    assert data.loss_history["val_loss"].shape == (3,)
    accs = classification_accuracies_from_npz(data.embeddings)
    assert accs["family"] == 1.0 and "spacegroup" not in accs


def test_open_expert_stack_exposes_spacegroup_payload(tmp_path):
    run = write_fake_run(tmp_path / "r", stacks=("family", "cubic"), n=8)
    data = open_stack(run, "cubic")
    accs = classification_accuracies_from_npz(data.embeddings)
    assert accs["spacegroup"] == 1.0 and "family" not in accs
    assert set(data.embeddings["labels"]) == {"Cubic"}
    assert data.embeddings["spacegroup_classes"].dtype == np.int64


def test_open_stack_without_heads_requires_opt_in(tmp_path):
    run = write_fake_run(tmp_path / "r", heads=())
    with pytest.raises(ValueError, match="no heads trained"):
        open_stack(run, FAMILY)
    data = open_stack(run, FAMILY, allow_no_heads=True)
    assert data.heads_name is None and data.viz_embeddings is None
    assert "family_probs" not in data.embeddings


def test_open_untrained_stack_names_the_trained_ones(tmp_path):
    run = write_fake_run(tmp_path / "r")
    with pytest.raises(ValueError, match=r"'cubic' is not trained.*\['family'\]"):
        open_stack(run, "cubic")
```

- [ ] **Step 3: Run to verify they fail**

Run: `pytest tests/test_pipeline_run_layout.py -v`
Expected: FAIL (`ModuleNotFoundError: dim_red.pipeline.run_layout`).

- [ ] **Step 4: Implement** `src/dim_red/pipeline/run_layout.py`

```python
"""Reading a ``FullStack`` run directory (jax-free, so ``dimred-compare`` and
``dimred-benchmark`` never need jax).

Layout: ``<run_dir>/config.yaml`` and ``<run_dir>/stacks/<name>/...`` (see
``dim_red.pipeline.full_stack``). Directories named ``.<x>.tmp`` are
half-written atomic writes of an interrupted run and are never listed.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from dim_red.pipeline.config import flatten_config_dict

FAMILY = "family"
EXPERT_NAMES: Tuple[str, ...] = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)
STACK_ORDER: Tuple[str, ...] = (FAMILY,) + EXPERT_NAMES

_HIDDEN_TMP = re.compile(r"^\..*\.tmp$")


def is_hidden_tmp(path: Union[str, Path]) -> bool:
    return bool(_HIDDEN_TMP.match(Path(path).name))


def _visible_dirs(parent: Path) -> List[Path]:
    if not parent.is_dir():
        return []
    return sorted(p for p in parent.iterdir() if p.is_dir() and not is_hidden_tmp(p))


def is_full_stack_run(path: Union[str, Path]) -> bool:
    path = Path(path)
    return (path / "config.yaml").is_file() and (path / "stacks").is_dir()


def discover_full_stack_runs(sweep_dir: Union[str, Path]) -> List[Path]:
    """Every immediate subdirectory of ``sweep_dir`` that is a FullStack run,
    sorted by name for reproducible plot ordering."""
    sweep_dir = Path(sweep_dir)
    runs = [p for p in _visible_dirs(sweep_dir) if is_full_stack_run(p)]
    if not runs:
        raise ValueError(f"No completed FullStack runs found directly under {sweep_dir}")
    return runs


def trained_stack_names(run_dir: Union[str, Path]) -> List[str]:
    """Stacks whose body is trained, in canonical order."""
    present = {
        p.name
        for p in _visible_dirs(Path(run_dir) / "stacks")
        if (p / "body" / "stack.yaml").is_file()
    }
    return [n for n in STACK_ORDER if n in present]


def head_names(stack_dir: Union[str, Path]) -> List[str]:
    return [
        p.name
        for p in _visible_dirs(Path(stack_dir) / "heads")
        if (p / "heads.yaml").is_file()
    ]


def resolve_heads_name(
    stack_dir: Union[str, Path], heads_name: Optional[str] = None
) -> str:
    """The heads set to use: the named one, or the only one if exactly one
    exists. Never picks silently between several."""
    names = head_names(stack_dir)
    if heads_name is not None:
        if heads_name not in names:
            raise ValueError(
                f"{stack_dir}: no heads named {heads_name!r}; available: {names}"
            )
        return heads_name
    if len(names) == 1:
        return names[0]
    if not names:
        raise ValueError(
            f"{stack_dir}: no heads trained yet; run fit_heads / dimred-train-heads"
        )
    raise ValueError(
        f"{stack_dir}: several heads available {names}; choose one with "
        "heads_name / --heads-name"
    )


def load_loss_history(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {
        key: np.array([float(row[key]) for row in rows]) for key in reader.fieldnames
    }


@dataclass
class RunData:
    """One stack of one run, as ``compare``/``benchmark`` consume it.

    ``run_dir`` is the run root and ``stack`` names the stack, so the pair is
    unique inside one comparison pass (a pass covers one stack across runs).
    """

    run_dir: Path
    stack: str
    model_kind: str
    flat_config: Dict[str, Any]
    loss_history: Dict[str, np.ndarray]
    embeddings: Dict[str, np.ndarray]
    viz_embeddings: Optional[np.ndarray] = None
    heads_name: Optional[str] = None

    @property
    def label(self) -> str:
        """Run directory name -- unique within a pass, used for identification."""
        return self.run_dir.name


def open_stack(
    run_dir: Union[str, Path],
    stack: str,
    heads_name: Optional[str] = None,
    allow_no_heads: bool = False,
) -> RunData:
    run_dir = Path(run_dir)
    trained = trained_stack_names(run_dir)
    if stack not in trained:
        raise ValueError(
            f"{run_dir}: stack {stack!r} is not trained; trained stacks: {trained}"
        )
    stack_dir = run_dir / "stacks" / stack
    run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    spec = yaml.safe_load((stack_dir / "config.yaml").read_text())
    classes = yaml.safe_load((stack_dir / "classes.yaml").read_text())["classes"]
    with np.load(stack_dir / "embeddings.npz") as npz:
        embeddings = dict(npz.items())
    if stack != FAMILY:
        # an expert's own labels are spacegroups; family-colored plots get the
        # single crystal system it covers
        embeddings["labels"] = np.full(
            len(embeddings["spacegroups"]), stack.capitalize()
        )

    used: Optional[str] = None
    viz: Optional[np.ndarray] = None
    if not (heads_name is None and allow_no_heads and not head_names(stack_dir)):
        used = resolve_heads_name(stack_dir, heads_name)
    if used is not None:
        heads_dir = stack_dir / "heads" / used
        with np.load(heads_dir / "predictions.npz") as npz:
            probs = npz["probs"]
        if stack == FAMILY:
            embeddings["family_probs"] = probs
            embeddings["family_classes"] = np.asarray([str(c) for c in classes])
        else:
            embeddings["spacegroup_probs"] = probs
            embeddings["spacegroup_classes"] = np.asarray(classes, dtype=np.int64)
        with np.load(heads_dir / "viz_embeddings.npz") as npz:
            viz = npz["embeddings"]

    return RunData(
        run_dir=run_dir,
        stack=stack,
        model_kind=run_cfg["model_kind"],
        flat_config=flatten_config_dict(spec),
        loss_history=load_loss_history(stack_dir / "body" / "loss_history.csv"),
        embeddings=embeddings,
        viz_embeddings=viz,
        heads_name=used,
    )
```

In `src/dim_red/pipeline/full_stack_config.py` delete the three definitions (`FAMILY = "family"`, `EXPERT_NAMES = (...)`, `STACK_ORDER = ...`) and add to the imports:

```python
from dim_red.pipeline.run_layout import EXPERT_NAMES, FAMILY, STACK_ORDER
```

(`full_stack.py` and the tests keep importing `FAMILY`/`STACK_ORDER` from `full_stack_config`; re-export works because the names are imported there.)

- [ ] **Step 5: Run to verify they pass**

Run: `pytest tests/test_pipeline_run_layout.py tests/test_pipeline_full_stack_config.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dim_red/pipeline/run_layout.py src/dim_red/pipeline/full_stack_config.py tests/fullstack_helpers.py tests/test_pipeline_run_layout.py
git commit -m "Add jax-free run_layout reader and shared FullStack test helpers

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Featurization and `FullStack.predict`

**Files:**
- Create: `src/dim_red/pipeline/featurize.py`
- Modify: `src/dim_red/pipeline/full_stack.py`
- Test: `tests/test_pipeline_featurize.py`, `tests/test_pipeline_full_stack_predict.py`

**Interfaces:**
- Consumes: `StackDataConfig` (`full_stack_config`), `trained_stack_names`/`resolve_heads_name` (`run_layout`), `SingleStack.load_body/load_heads/predict_proba/visualize` (device arg from Task 2).
- Produces:
  - `featurize_structures(atoms_list, model_kind, data) -> np.ndarray` raw (unstandardized) `(n, d)` float32
  - `standardize(raw, mean, std) -> np.ndarray` float32
  - `@dataclass(frozen=True) StackPrediction(classes: list, proba: np.ndarray, viz: np.ndarray)`
  - `@dataclass(frozen=True) Prediction(family, family_proba, expert, spacegroup, spacegroup_proba, viz_family, viz_expert)` — lists of length n: `family: List[str]`, `family_proba: np.ndarray (n,)` max prob, `expert: List[Optional[str]]`, `spacegroup: List[Optional[int]]`, `spacegroup_proba: List[Optional[float]]`, `viz_family: np.ndarray (n, d)`, `viz_expert: List[Optional[np.ndarray]]`
  - `FullStack.stack_names() -> List[str]`, `FullStack.predict_stack(name, structures, heads_name=None, device="cpu") -> StackPrediction`, `FullStack.predict(structures, heads_name=None, device="cpu") -> Prediction`
  - `DEFAULT_HEADS_NAME = "default"` in `full_stack.py`

- [ ] **Step 1: Write the failing featurize tests** `tests/test_pipeline_featurize.py`

```python
import numpy as np
import pytest
from ase.build import bulk

pytest.importorskip("dscribe")
pytest.importorskip("jax")

from dim_red.pipeline.config import MaceConfig, PyxtalConfig, SoapConfig
from dim_red.pipeline.featurize import featurize_structures, standardize
from dim_red.pipeline.full_stack_config import StackDataConfig


def _data(**soap):
    base = dict(r_cut=4.0, n_max=3, l_max=2, sigma=0.5, element_agnostic=True)
    base.update(soap)
    return StackDataConfig(pyxtal=PyxtalConfig(), soap=SoapConfig(**base))


def _nacl():
    return bulk("NaCl", "rocksalt", a=5.6, cubic=True)


def _si():
    return bulk("Si", "diamond", a=5.43, cubic=True)


def test_featurization_ignores_other_structures_in_the_batch():
    alone = featurize_structures([_nacl()], "supcon", _data())
    in_batch = featurize_structures([_si(), _nacl()], "supcon", _data())
    np.testing.assert_allclose(alone[0], in_batch[1], atol=1e-6)


def test_elements_do_not_change_the_vector():
    a = _nacl()
    b = _nacl()
    b.set_chemical_symbols(["K" if s == "Na" else "Br" for s in b.get_chemical_symbols()])
    fa = featurize_structures([a], "supcon", _data())
    fb = featurize_structures([b], "supcon", _data())
    np.testing.assert_allclose(fa, fb, atol=1e-6)


def test_output_is_float32_matrix():
    out = featurize_structures([_nacl(), _si()], "supcon", _data())
    assert out.dtype == np.float32 and out.ndim == 2 and out.shape[0] == 2


def test_non_agnostic_soap_is_refused():
    with pytest.raises(ValueError, match="element_agnostic"):
        featurize_structures([_nacl()], "supcon", _data(element_agnostic=False))


def test_empty_input_is_refused():
    with pytest.raises(ValueError, match="empty"):
        featurize_structures([], "supcon", _data())


def test_standardize_matches_the_training_formula():
    raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mean = np.array([2.0, 3.0], dtype=np.float32)
    std = np.array([1.0, 2.0], dtype=np.float32)
    out = standardize(raw, mean, std)
    np.testing.assert_allclose(out, [[-1.0, -0.5], [1.0, 0.5]])
    assert out.dtype == np.float32
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_pipeline_featurize.py -v`
Expected: FAIL (`ModuleNotFoundError: dim_red.pipeline.featurize`).

- [ ] **Step 3: Implement** `src/dim_red/pipeline/featurize.py`

```python
"""Features of *new* structures, computed exactly like training did: SOAP with
the ``mu2`` (element-agnostic) compression -- which depends only on geometry,
so the species list never has to match any training dataset -- or MACE
embeddings, then the per-stack standardization saved with each stack."""

from typing import List

import numpy as np
from ase import Atoms

from dim_red.pipeline.full_stack_config import StackDataConfig
from dim_red.soap import compute_soap
from dim_red.utils import apply_standardization


def featurize_structures(
    atoms_list: List[Atoms], model_kind: str, data: StackDataConfig
) -> np.ndarray:
    """Raw (not yet standardized) global feature vector per structure.

    Periodicity is decided by ``compute_soap`` over the whole list (as in
    training): keep periodic and non-periodic structures in separate calls."""
    if not atoms_list:
        raise ValueError("atoms_list is empty -- nothing to featurize")
    if model_kind == "supcon_mace":
        from dim_red.mace.model import MaceEncoder

        encoder = MaceEncoder(**data.mace.mace_kwargs())
        return np.asarray(encoder.encode(atoms_list), dtype=np.float32)

    kwargs = data.soap.as_kwargs()
    if not kwargs["element_agnostic"]:
        raise ValueError(
            "featurizing new structures needs data.soap.element_agnostic: true "
            "(SOAP 'mu2'): without it the feature size depends on the training "
            "dataset's species"
        )
    present = {s for a in atoms_list for s in a.get_chemical_symbols()}
    kwargs["species"] = sorted(present | set(kwargs.get("species") or []))
    kwargs["average"] = "outer"
    vectors = compute_soap(atoms_list, **kwargs)
    return np.asarray(vectors, dtype=np.float32).reshape(len(atoms_list), -1)


def standardize(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return apply_standardization(raw, mean, std).astype(np.float32)
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_pipeline_featurize.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing predict tests** `tests/test_pipeline_full_stack_predict.py`

```python
import numpy as np
import pytest
from ase import Atoms

pytest.importorskip("jax")

from dim_red.pipeline import full_stack as fs_mod
from dim_red.pipeline.full_stack import FullStack, Prediction, StackPrediction
from dim_red.pipeline.run_layout import FAMILY
from tests.fullstack_helpers import SYSTEM_SPACEGROUPS, make_run


def _structures(n):
    return [Atoms("Cu", positions=[[0, 0, 0]]) for _ in range(n)]


@pytest.fixture
def featurize_calls(monkeypatch):
    calls = []

    def fake(atoms_list, model_kind, data):
        calls.append(len(atoms_list))
        return np.stack(
            [np.arange(6, dtype=np.float32) + i for i in range(len(atoms_list))]
        )

    monkeypatch.setattr(fs_mod, "featurize_structures", fake)
    return calls


def test_stack_names_lists_trained_stacks(tmp_path, monkeypatch):
    fs = make_run(tmp_path, monkeypatch)
    assert fs.stack_names() == ["family", "tetragonal", "cubic"]


def test_predict_routes_each_structure_to_its_systems_expert(
    tmp_path, monkeypatch, featurize_calls
):
    fs = FullStack.open(make_run(tmp_path, monkeypatch).run_dir)
    real = FullStack._predict_stack

    def stub(self, name, structures, heads_name, device, cache, rows=None):
        if name == FAMILY:  # force the routing: Cubic, Hexagonal, Tetragonal, Cubic
            proba = np.eye(3)[[0, 1, 2, 0]]
            return StackPrediction(
                classes=["Cubic", "Hexagonal", "Tetragonal"],
                proba=proba,
                viz=np.zeros((4, 2)),
            )
        return real(self, name, structures, heads_name, device, cache, rows)

    monkeypatch.setattr(FullStack, "_predict_stack", stub)
    pred = fs.predict(_structures(4))

    assert isinstance(pred, Prediction)
    assert pred.family == ["Cubic", "Hexagonal", "Tetragonal", "Cubic"]
    assert pred.expert == ["cubic", None, "tetragonal", "cubic"]
    assert pred.spacegroup[1] is None and pred.spacegroup_proba[1] is None
    assert pred.viz_expert[1] is None
    assert pred.spacegroup[0] in SYSTEM_SPACEGROUPS["cubic"]
    assert pred.spacegroup[3] in SYSTEM_SPACEGROUPS["cubic"]
    assert pred.spacegroup[2] in SYSTEM_SPACEGROUPS["tetragonal"]
    assert pred.viz_expert[0].shape == (2,)
    assert pred.viz_family.shape == (4, 2)
    assert pred.family_proba.shape == (4,)
    assert 0.0 < pred.spacegroup_proba[0] <= 1.0


def test_features_are_computed_once_per_distinct_data_config(
    tmp_path, monkeypatch, featurize_calls
):
    fs = FullStack.open(make_run(tmp_path, monkeypatch).run_dir)
    fs.predict(_structures(5))
    assert featurize_calls == [5]  # family + both experts share one featurization


def test_predict_stack_works_alone_and_without_a_family_stack(
    tmp_path, monkeypatch, featurize_calls
):
    fs = FullStack.open(make_run(tmp_path, monkeypatch, with_family=False).run_dir)
    assert fs.stack_names() == ["tetragonal", "cubic"]
    out = fs.predict_stack("cubic", _structures(3))
    assert out.classes == SYSTEM_SPACEGROUPS["cubic"]
    assert out.proba.shape == (3, 2) and out.viz.shape == (3, 2)
    np.testing.assert_allclose(out.proba.sum(axis=1), 1.0, rtol=1e-5)
    with pytest.raises(ValueError, match="family stack.*predict_stack"):
        fs.predict(_structures(3))


def test_predict_stack_rejects_an_untrained_stack(tmp_path, monkeypatch, featurize_calls):
    fs = FullStack.open(make_run(tmp_path, monkeypatch).run_dir)
    with pytest.raises(ValueError, match=r"'hexagonal'.*trained stacks"):
        fs.predict_stack("hexagonal", _structures(2))


def test_predict_requires_a_heads_choice_when_ambiguous(
    tmp_path, monkeypatch, featurize_calls
):
    fs = FullStack.open(make_run(tmp_path, monkeypatch, heads=("a", "b")).run_dir)
    with pytest.raises(ValueError, match="several heads"):
        fs.predict(_structures(2))
    assert len(fs.predict(_structures(2), heads_name="a").family) == 2


def test_predict_rejects_empty_input(tmp_path, monkeypatch, featurize_calls):
    fs = FullStack.open(make_run(tmp_path, monkeypatch).run_dir)
    with pytest.raises(ValueError, match="empty"):
        fs.predict([])
```

- [ ] **Step 6: Run to verify they fail**

Run: `pytest tests/test_pipeline_full_stack_predict.py -v`
Expected: FAIL (`ImportError: cannot import name 'Prediction'`).

- [ ] **Step 7: Implement** in `src/dim_red/pipeline/full_stack.py`.

Imports (merge with existing ones): `import dataclasses`, `import json`, `from dataclasses import dataclass`, `from typing import Any, Dict, List, Optional, Sequence, Union`, `from dim_red.pipeline.featurize import featurize_structures, standardize`, `from dim_red.pipeline.run_layout import resolve_heads_name, trained_stack_names`.

Module level (after `History = ...`):

```python
DEFAULT_HEADS_NAME = "default"


@dataclass(frozen=True)
class StackPrediction:
    """One stack's output for a batch: class names (columns of ``proba``),
    class probabilities and the 2D/3D visualization coordinates."""

    classes: list
    proba: np.ndarray
    viz: np.ndarray

    @property
    def labels(self) -> list:
        return [self.classes[i] for i in self.proba.argmax(axis=1)]


@dataclass(frozen=True)
class Prediction:
    """Family -> expert prediction for a batch of structures. ``expert``,
    ``spacegroup``, ``spacegroup_proba`` and ``viz_expert`` hold ``None`` for a
    structure whose crystal system has no trained expert: an explicit missing
    value, never a guess."""

    family: List[str]
    family_proba: np.ndarray
    expert: List[Optional[str]]
    spacegroup: List[Optional[int]]
    spacegroup_proba: List[Optional[float]]
    viz_family: np.ndarray
    viz_expert: List[Optional[np.ndarray]]
```

Methods to add to `FullStack` (replace the old `load_stack` with the Task 2 version if not already, and add):

```python
    # -- inference -------------------------------------------------------
    def stack_names(self) -> List[str]:
        """Trained stacks on disk, in canonical order."""
        return trained_stack_names(self.run_dir)

    def _stack_meta(self, name: str):
        stack_dir = self._stack_dir(name)
        with open(stack_dir / "classes.yaml") as f:
            classes = yaml.safe_load(f)["classes"]
        with np.load(stack_dir / "embeddings.npz") as npz:
            return classes, npz["feature_mean"], npz["feature_std"]

    def _raw_features(self, structures, spec: StackSpec, cache: Dict[str, Any]):
        data = spec.data.mace if self.config.model_kind == "supcon_mace" else spec.data.soap
        key = json.dumps(dataclasses.asdict(data), sort_keys=True)
        if key not in cache:
            cache[key] = featurize_structures(
                structures, self.config.model_kind, spec.data
            )
        return cache[key]

    def _predict_stack(
        self,
        name: str,
        structures: list,
        heads_name: Optional[str],
        device: str,
        cache: Dict[str, Any],
        rows: Optional[Sequence[int]] = None,
    ) -> StackPrediction:
        trained = self.stack_names()
        if name not in trained:
            raise ValueError(
                f"stack {name!r} is not trained in {self.run_dir}; trained "
                f"stacks: {trained}"
            )
        spec = self.config.stacks[name]
        raw = self._raw_features(structures, spec, cache)
        if rows is not None:
            raw = raw[list(rows)]
        classes, mean, std = self._stack_meta(name)
        X = standardize(raw, mean, std)
        heads = resolve_heads_name(self._stack_dir(name), heads_name)
        stack = self.load_stack(name, heads, device=device)
        return StackPrediction(
            classes=classes, proba=stack.predict_proba(X), viz=stack.visualize(X)
        )

    def predict_stack(
        self,
        name: str,
        structures,
        heads_name: Optional[str] = None,
        device: str = "cpu",
    ) -> StackPrediction:
        """One stack alone (no routing). ``device`` is where the params are
        placed, whatever device they were trained on."""
        structures = list(structures)
        if not structures:
            raise ValueError("structures is empty -- nothing to predict")
        return self._predict_stack(name, structures, heads_name, device, {})

    def predict(
        self,
        structures,
        heads_name: Optional[str] = None,
        device: str = "cpu",
    ) -> Prediction:
        """Family stack assigns the crystal system; each structure is then
        sent to that system's expert (if one is trained) for its spacegroup."""
        structures = list(structures)
        if not structures:
            raise ValueError("structures is empty -- nothing to predict")
        trained = self.stack_names()
        if FAMILY not in trained:
            raise ValueError(
                "predict routes through the family stack, which is not trained "
                f"in {self.run_dir}; use predict_stack(name, structures) for a "
                "single stack"
            )
        cache: Dict[str, Any] = {}
        fam = self._predict_stack(FAMILY, structures, heads_name, device, cache)
        family = [str(fam.classes[i]) for i in fam.proba.argmax(axis=1)]
        n = len(structures)
        expert: List[Optional[str]] = [None] * n
        spacegroup: List[Optional[int]] = [None] * n
        spacegroup_proba: List[Optional[float]] = [None] * n
        viz_expert: List[Optional[np.ndarray]] = [None] * n
        for system in sorted(set(family)):
            name = system.lower()
            if name not in trained or name == FAMILY:
                continue
            rows = [i for i, f in enumerate(family) if f == system]
            exp = self._predict_stack(
                name, structures, heads_name, device, cache, rows=rows
            )
            for k, i in enumerate(rows):
                top = int(exp.proba[k].argmax())
                expert[i] = name
                spacegroup[i] = int(exp.classes[top])
                spacegroup_proba[i] = float(exp.proba[k, top])
                viz_expert[i] = exp.viz[k]
        return Prediction(
            family=family,
            family_proba=fam.proba.max(axis=1),
            expert=expert,
            spacegroup=spacegroup,
            spacegroup_proba=spacegroup_proba,
            viz_family=fam.viz,
            viz_expert=viz_expert,
        )
```

- [ ] **Step 8: Run to verify they pass**

Run: `pytest tests/test_pipeline_full_stack_predict.py tests/test_pipeline_featurize.py tests/test_pipeline_full_stack.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/dim_red/pipeline/featurize.py src/dim_red/pipeline/full_stack.py tests/test_pipeline_featurize.py tests/test_pipeline_full_stack_predict.py
git commit -m "Add FullStack.predict/predict_stack with per-stack standardization and mu2 featurization

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `apply_to_structures`

**Files:**
- Modify: `src/dim_red/pipeline/full_stack.py`
- Test: `tests/test_pipeline_full_stack_predict.py`

**Interfaces:**
- Produces: `apply_to_structures(run_dir, structures_path, output_dir=None, heads_name=None, device="cpu") -> Path` (the output dir). Writes `<stem>_predictions.csv` (columns `material_id,family,family_proba,expert,spacegroup,spacegroup_proba`; empty cells for missing values) and `<stem>_viz.npz` (`material_ids`, `viz_family`, `viz_expert` with NaN rows where missing).

- [ ] **Step 1: Write the failing test** (append to `tests/test_pipeline_full_stack_predict.py`)

```python
import csv

from ase.io import write

from dim_red.pipeline.full_stack import apply_to_structures


def test_apply_writes_predictions_csv_and_viz(tmp_path, monkeypatch, featurize_calls):
    run = make_run(tmp_path, monkeypatch).run_dir
    path = tmp_path / "new.extxyz"
    atoms = _structures(3)
    atoms[0].info["material_id"] = "mine-0"
    write(str(path), atoms, format="extxyz")

    out = apply_to_structures(run, path)

    assert out == run / "applied"
    rows = list(csv.DictReader(open(out / "new_predictions.csv")))
    assert [r["material_id"] for r in rows] == ["mine-0", "new-1", "new-2"]
    assert set(rows[0]) == {
        "material_id", "family", "family_proba", "expert", "spacegroup", "spacegroup_proba",
    }
    for r in rows:
        assert r["family"] in {"Cubic", "Tetragonal", "Hexagonal"}
        if r["expert"] == "":
            assert r["spacegroup"] == "" and r["spacegroup_proba"] == ""
        else:
            assert int(r["spacegroup"]) in SYSTEM_SPACEGROUPS[r["expert"]]
    with np.load(out / "new_viz.npz") as npz:
        assert npz["viz_family"].shape == (3, 2)
        assert npz["viz_expert"].shape == (3, 2)
        assert list(npz["material_ids"]) == ["mine-0", "new-1", "new-2"]


def test_apply_rejects_an_empty_structure_file(tmp_path, monkeypatch, featurize_calls):
    run = make_run(tmp_path, monkeypatch).run_dir
    empty = tmp_path / "empty.extxyz"
    empty.write_text("")
    with pytest.raises(ValueError, match="No structures"):
        apply_to_structures(run, empty)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_pipeline_full_stack_predict.py -k apply -v`
Expected: FAIL (`ImportError: cannot import name 'apply_to_structures'`).

- [ ] **Step 3: Implement.** Add `import csv` to `full_stack.py` imports and append at module end:

```python
def apply_to_structures(
    run_dir: Union[str, Path],
    structures_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    heads_name: Optional[str] = None,
    device: str = "cpu",
) -> Path:
    """Predict every structure in an extended-XYZ file with an already-trained
    run and write ``<stem>_predictions.csv`` and ``<stem>_viz.npz`` to
    ``output_dir`` (default ``<run_dir>/applied``)."""
    from ase.io import read as read_atoms

    run_dir = Path(run_dir)
    structures = read_atoms(str(structures_path), index=":")
    if not structures:
        raise ValueError(f"No structures found in {structures_path}")
    prediction = FullStack.open(run_dir).predict(
        structures, heads_name=heads_name, device=device
    )

    stem = Path(structures_path).stem
    out = Path(output_dir) if output_dir else run_dir / "applied"
    out.mkdir(parents=True, exist_ok=True)
    material_ids = [
        a.info.get("material_id", f"{stem}-{i}") for i, a in enumerate(structures)
    ]
    with open(out / f"{stem}_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["material_id", "family", "family_proba", "expert", "spacegroup",
             "spacegroup_proba"]
        )
        for i, mid in enumerate(material_ids):
            has_expert = prediction.expert[i] is not None
            writer.writerow(
                [
                    mid,
                    prediction.family[i],
                    f"{prediction.family_proba[i]:.6f}",
                    prediction.expert[i] if has_expert else "",
                    prediction.spacegroup[i] if has_expert else "",
                    f"{prediction.spacegroup_proba[i]:.6f}" if has_expert else "",
                ]
            )
    viz_dim = prediction.viz_family.shape[1]
    viz_expert = np.full((len(structures), viz_dim), np.nan, dtype=np.float32)
    for i, z in enumerate(prediction.viz_expert):
        if z is not None:
            viz_expert[i, : len(z)] = z
    np.savez(
        out / f"{stem}_viz.npz",
        material_ids=np.asarray(material_ids),
        viz_family=prediction.viz_family,
        viz_expert=viz_expert,
    )
    logger.info("Applied %s to %d structures; saved to %s", run_dir, len(structures), out)
    return out
```

(The expert viz dimension can differ from the family's; if `len(z) > viz_dim` size the array from the first non-None expert. Implement exactly that: `dim = max([len(z) for z in prediction.viz_expert if z is not None] + [viz_dim])` and use `dim` for both the NaN array width and slicing.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_pipeline_full_stack_predict.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dim_red/pipeline/full_stack.py tests/test_pipeline_full_stack_predict.py
git commit -m "Add apply_to_structures: predictions CSV and viz coordinates for new structures

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `compare` per stack

**Files:**
- Modify: `src/dim_red/pipeline/compare.py`
- Modify (rewrite helper + adapt): `tests/test_pipeline_compare.py`
- Test: `tests/test_pipeline_compare.py`

**Interfaces:**
- Consumes: `RunData`, `open_stack`, `discover_full_stack_runs`, `trained_stack_names`, `STACK_ORDER` from `run_layout`.
- Produces: `load_runs(sweep_dir, stack=FAMILY, heads_name=None) -> List[RunData]`; `display_metric(metric, stack) -> str`; `generate_comparison_report(sweep_dir, output_dir=None, write_data_files=False, umap_params=None, heads_name=None) -> Path` writing `<output_dir>/<stack>/...` per stack. All plot functions keep their current signatures (they take `List[RunData]`).

- [ ] **Step 1: Adapt the test module to the new layout.** In `tests/test_pipeline_compare.py`:

  1. Replace the imports of `RunConfig`, `AuxHeadsConfig`, `EncoderConfig`, `FetchConfig`, `SoapConfig`, `TrainSettings`, `run_config_to_dict`, `csv`, `yaml` (unused afterwards) with `from tests.fullstack_helpers import write_fake_run`; remove `discover_runs` and `hierarchical_accuracies_from_npz` from the `dim_red.pipeline.compare` import list and add `display_metric`; add `from dim_red.pipeline.run_layout import FAMILY, discover_full_stack_runs`.
  2. Delete `_write_loss_history` and replace the body of `_write_run` (keep its signature) with a delegation:

```python
def _write_run(
    run_dir,
    hidden_dim,
    crystal_systems,
    val_loss_final,
    with_aux=False,
    n=6,
    learning_rate=1e-3,
    n_features=None,
    latent_dim=2,
):
    return write_fake_run(
        run_dir,
        hidden_dim=hidden_dim,
        families=[c.capitalize() for c in crystal_systems],
        val_loss_final=val_loss_final,
        aux=with_aux,
        n=n,
        learning_rate=learning_rate,
        n_features=n_features,
        latent_dim=latent_dim,
        heads=("default",) if with_aux else (),
    )
```

  3. Replace `discover_runs` tests by `discover_full_stack_runs` (`test_discover_runs_finds_completed_run_dirs`/`..._raises_when_empty`); delete `test_hierarchical_accuracies_*`.
  4. Config keys in the old tests changed: `fetch.crystal_systems` → `data.pyxtal.families`, `encoder.encoder_hidden_dim` → `model.encoder_hidden_dim`, `train.learning_rate` → `model.body_train.learning_rate`, `encoder.latent_dim` → `model.latent_dim`. Update the expected key strings/`safe_name`s (`final_val_loss_vs_model_body_train_learning_rate.png`).
  5. Tests that wrote `loss_history` columns `train_recon`/`train_kl` must expect the new columns (`train_loss`, `val_loss`, `train_family_supcon`, `val_family_supcon`, plus `*_family_ce` when `with_aux`); update `available_loss_metrics` expectations (known order first, then sorted extras).
  6. `generate_comparison_report` outputs now live in `<out>/family/`: update `test_generate_comparison_report_*` paths accordingly.

- [ ] **Step 2: Add the new failing tests** (append):

```python
def test_report_writes_one_suite_per_stack(tmp_path, caplog):
    import logging

    sweep = tmp_path / "20261007-1"
    write_fake_run(sweep / "a", stacks=("family", "cubic"), learning_rate=1e-3)
    write_fake_run(sweep / "b", stacks=("family", "cubic"), learning_rate=2e-3)
    write_fake_run(sweep / "c", stacks=("family",), learning_rate=3e-3)

    with caplog.at_level(logging.INFO, logger="dim_red.pipeline"):
        out = generate_comparison_report(sweep)

    assert out == sweep / "comparison"
    for stack in ("family", "cubic"):
        assert (out / stack / "loss_curves.png").exists()
        assert (out / stack / "latent_space_grid.png").exists()
        assert (
            out / stack / "final_val_loss_vs_model_body_train_learning_rate.png"
        ).exists()
    assert any("skipping run" in r.message and "'cubic'" in r.message and "c" in r.message
               for r in caplog.records)


def test_load_runs_selects_the_requested_stack(tmp_path):
    sweep = tmp_path / "s"
    write_fake_run(sweep / "a", stacks=("family", "cubic"))
    runs = load_runs(sweep, stack="cubic")
    assert [r.stack for r in runs] == ["cubic"]
    assert "spacegroup_probs" in runs[0].embeddings


def test_report_requires_a_heads_choice_when_ambiguous(tmp_path):
    sweep = tmp_path / "s"
    write_fake_run(sweep / "a", heads=("x", "y"))
    with pytest.raises(ValueError, match="several heads"):
        generate_comparison_report(sweep)
    generate_comparison_report(sweep, heads_name="x")


def test_display_metric_names_the_label_of_the_stack():
    assert display_metric("train_family_supcon", FAMILY) == "train_family_supcon"
    assert display_metric("val_family_supcon", "cubic") == "val_spacegroup_supcon"
    assert display_metric("val_loss", "cubic") == "val_loss"
```

- [ ] **Step 3: Run to verify failures**

Run: `pytest tests/test_pipeline_compare.py -v`
Expected: FAIL (imports of removed/new names).

- [ ] **Step 4: Implement in `src/dim_red/pipeline/compare.py`.**

  1. Imports: replace the `from dim_red.pipeline.config import (RunConfig, flatten_config_dict, load_run_config, run_config_to_dict)` block by `from dim_red.pipeline.run_layout import (FAMILY, STACK_ORDER, RunData, discover_full_stack_runs, open_stack, trained_stack_names)`; drop `from dataclasses import dataclass` and `import csv` only if grep shows them unused afterwards (`write_csv` still uses `csv`).
  2. Delete: the `RunData` dataclass, `discover_runs`, `_load_loss_history`, `load_run`, `hierarchical_accuracies_from_npz`.
  3. `load_runs`:

```python
def load_runs(
    sweep_dir: Union[str, Path],
    stack: str = FAMILY,
    heads_name: Optional[str] = None,
) -> List[RunData]:
    """``stack`` of every FullStack run directly under ``sweep_dir`` that has
    it trained (runs without it are skipped with a log line)."""
    runs = []
    for run_dir in discover_full_stack_runs(sweep_dir):
        if stack not in trained_stack_names(run_dir):
            logger.info("skipping run %s: stack %r is not trained", run_dir.name, stack)
            continue
        runs.append(open_stack(run_dir, stack, heads_name, allow_no_heads=True))
    return runs
```

  4. `_NON_HYPERPARAM_KEYS = {"output_dir", "name", "seed", "data.pyxtal.seed"}` (constant or per-stack seeds differ between stacks but not between runs; `seed` stays out of labels — if a sweep varies the run `seed`, the per-stack `seed` keys still vary and show up as `seed`; keep `"seed"` **out** of this set. Final value: `{"output_dir", "name"}`).
  5. `_format_hyperparam_value`: `if path in ("fetch.crystal_systems", "data.pyxtal.families"):`; add `"families": "cs"` to `_HYPERPARAM_KEY_ABBREV`.
  6. Add after `_NON_HYPERPARAM_KEYS`:

```python
def display_metric(metric: str, stack: str) -> str:
    """Plot label of a loss-history column. The CSV keeps the historical
    ``*_family_supcon`` names, but in an expert stack that term contrasts
    spacegroups."""
    if stack == FAMILY:
        return metric
    return metric.replace("family_supcon", "spacegroup_supcon")
```

  7. In `plot_loss_curves` replace `ax.set_ylabel(metric)` / `ax.set_title(metric)` with `ax.set_ylabel(display_metric(metric, runs[0].stack))` / `ax.set_title(display_metric(metric, runs[0].stack))`; in `plot_final_metric_vs_hyperparam` use `shown = display_metric(metric, runs[0].stack)` for `ax.set_ylabel(f"final {shown}")` and the title.
  8. `RunData.flat_config` is now a field: delete any `run.flat_config` call parentheses if present (none — it was a property).
  9. Replace `generate_comparison_report` by a per-stack driver; move the old body into `_write_stack_report`:

```python
def _write_stack_report(
    runs: List[RunData],
    output_dir: Path,
    write_data_files: bool,
    umap_params: Optional[LatentUmapParams],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def _csv_path(name: str) -> Optional[Path]:
        return output_dir / name if write_data_files else None

    plot_loss_curves(
        runs,
        metrics=available_loss_metrics(runs),
        save_path=output_dir / "loss_curves.png",
        csv_path=_csv_path("loss_curves.csv"),
    )
    if len(runs) > 1:
        varying = varying_hyperparams(runs)
        for metric in available_loss_metrics(runs):
            for hyperparam in varying:
                safe_name = hyperparam.replace(".", "_")
                plot_final_metric_vs_hyperparam(
                    runs,
                    hyperparam,
                    metric=metric,
                    save_path=output_dir / f"final_{metric}_vs_{safe_name}.png",
                    csv_path=_csv_path(f"final_{metric}_vs_{safe_name}.csv"),
                )
    else:
        logger.info("Only one run found; skipping final-metric-vs-hyperparameter plots")
    plot_spacegroup_family_histogram(
        runs,
        save_path=output_dir / "spacegroup_histogram.png",
        csv_path=_csv_path("spacegroup_histogram.csv"),
    )
    plot_latent_space_grid(
        runs,
        save_path=output_dir / "latent_space_grid.png",
        csv_path=_csv_path("latent_space_grid.csv"),
        umap_params=umap_params,
    )
    plot_aux_accuracy_comparison(
        runs,
        save_path=output_dir / "aux_heads_accuracy.png",
        csv_path=_csv_path("aux_heads_accuracy.csv"),
    )


def generate_comparison_report(
    sweep_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    write_data_files: bool = False,
    umap_params: Optional[LatentUmapParams] = None,
    heads_name: Optional[str] = None,
) -> Path:
    """Render the full comparison suite **once per stack** for every FullStack
    run directly under ``sweep_dir``, into ``<output_dir>/<stack>/`` (default
    ``<sweep_dir>/comparison``). A run that lacks a stack is skipped, with a
    log line, from that stack's suite only. Needs a ``heads_name`` when a
    stack has several heads sets; stacks with no heads get no accuracy plots.
    """
    sweep_dir = Path(sweep_dir)
    run_dirs = discover_full_stack_runs(sweep_dir)
    output_dir = Path(output_dir) if output_dir else sweep_dir / "comparison"
    stacks = [
        s
        for s in STACK_ORDER
        if any(s in trained_stack_names(d) for d in run_dirs)
    ]
    logger.info("Comparing %d run(s) from %s over stacks %s", len(run_dirs), sweep_dir, stacks)
    for stack in stacks:
        runs = load_runs(sweep_dir, stack=stack, heads_name=heads_name)
        _write_stack_report(runs, output_dir / stack, write_data_files, umap_params)
    logger.info("Comparison report saved to %s", output_dir)
    return output_dir
```

  Update the module docstring's first paragraph ("discovers every run directory written by ``dim_red.pipeline.sweep.run_sweep``" stays true) and add: "One suite per stack."

  10. Remove now-unused imports (`grep -n "RunConfig\|load_run_config\|run_config_to_dict\|flatten_config_dict" src/dim_red/pipeline/compare.py` must print nothing).

- [ ] **Step 5: Run to verify**

Run: `pytest tests/test_pipeline_compare.py -v`
Expected: PASS (fix the remaining adapted tests until green; delete a test only if it exercises a removed feature, naming it in the commit message).

- [ ] **Step 6: Commit**

```bash
git add src/dim_red/pipeline/compare.py tests/test_pipeline_compare.py
git commit -m "Make dimred-compare render one plot suite per stack

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `benchmark` per stack

**Files:**
- Modify: `src/dim_red/pipeline/benchmark.py`
- Modify (adapt): `tests/test_pipeline_benchmark.py`
- Test: `tests/test_pipeline_benchmark.py`

**Interfaces:**
- Produces: `collect_run_dirs(inputs) -> List[Path]` (FullStack run roots); `benchmark_row(run: RunData, key_hyperparams=...) -> Dict` with a `stack` column; `generate_benchmark_table(inputs, output_csv, key_hyperparams=_DEFAULT_KEY_HYPERPARAMS, plot=False, write_data_files=False, heads_name=None) -> Path` — one row per (run, stack); plots rendered per stack into `benchmark_plots/<stack>/`.
- `_DEFAULT_KEY_HYPERPARAMS = ("model.encoder_hidden_dim", "model.latent_dim", "model.body_train.learning_rate", "model.body_train.batch_size", "model.body_train.epochs", "seed")`

- [ ] **Step 1: Adapt the tests.** In `tests/test_pipeline_benchmark.py` delete `_write_loss_history`, `_write_run`, `_write_tail_predictions`, `_write_visualization_tail`, `_write_log` and every test that exists only for them (wall clock, `tails/…`, `hierarchical_supcon*`, old `collect_run_dirs` on flat dirs); build runs with `write_fake_run` from `tests.fullstack_helpers`; keep the pure tests (`_csv_safe`, `_rows_for_plots`-based plot tests) unchanged. Update `_DEFAULT_KEY_HYPERPARAMS` expectations to the tuple above.

- [ ] **Step 2: Add the failing tests** (append):

```python
import csv

from dim_red.pipeline.benchmark import generate_benchmark_table, collect_run_dirs
from tests.fullstack_helpers import write_fake_run


def _sweep(tmp_path):
    sweep = tmp_path / "s"
    write_fake_run(sweep / "a", stacks=("family", "cubic"), families=("Cubic", "Hexagonal"))
    write_fake_run(
        sweep / "b",
        stacks=("family", "cubic"),
        families=("Cubic", "Hexagonal"),
        learning_rate=2e-3,
    )
    return sweep


def test_table_has_one_row_per_run_and_stack(tmp_path):
    out = generate_benchmark_table([_sweep(tmp_path)], tmp_path / "t.csv")
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 4
    assert {r["stack"] for r in rows} == {"family", "cubic"}
    family = [r for r in rows if r["stack"] == "family"]
    cubic = [r for r in rows if r["stack"] == "cubic"]
    assert all(r["family"] != "" for r in family)
    assert all(r["spacegroup"] != "" for r in cubic)
    assert all(r["family"] == "" for r in cubic)  # experts have no family metrics
    assert all(r["model_kind"] == "supcon" for r in rows)
    assert "wall_clock_seconds" not in rows[0]


def test_default_hyperparams_are_the_new_dotted_paths(tmp_path):
    out = generate_benchmark_table([_sweep(tmp_path)], tmp_path / "t.csv")
    header = next(csv.reader(open(out)))
    for key in ("model.latent_dim", "model.encoder_hidden_dim", "model.body_train.learning_rate"):
        assert key in header


def test_collect_run_dirs_accepts_runs_and_sweeps(tmp_path):
    sweep = _sweep(tmp_path)
    assert collect_run_dirs([sweep]) == sorted(p.resolve() for p in sweep.iterdir())
    assert collect_run_dirs([sweep / "a", sweep]) == collect_run_dirs([sweep])


def test_plots_are_written_per_stack(tmp_path):
    generate_benchmark_table([_sweep(tmp_path)], tmp_path / "t.csv", plot=True)
    plots = tmp_path / "benchmark_plots"
    assert any((plots / "family").iterdir())
    assert any((plots / "cubic").iterdir())


def test_heads_name_is_forwarded(tmp_path):
    sweep = tmp_path / "s"
    write_fake_run(sweep / "a", heads=("x", "y"))
    with pytest.raises(ValueError, match="several heads"):
        generate_benchmark_table([sweep], tmp_path / "t.csv")
    generate_benchmark_table([sweep], tmp_path / "t.csv", heads_name="x")
```

- [ ] **Step 3: Run to verify failures**

Run: `pytest tests/test_pipeline_benchmark.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement in `src/dim_red/pipeline/benchmark.py`.**

  1. Imports: `from dim_red.pipeline.compare import (...)` drop `RunData, discover_runs, load_run`; add `from dim_red.pipeline.run_layout import FAMILY, RunData, discover_full_stack_runs, is_full_stack_run, open_stack, trained_stack_names`.
  2. Delete `_LOG_TIMESTAMP_FORMAT`, `_log_span_seconds`, `run_wall_clock_seconds` and the `wall_clock_seconds` row field.
  3. Set `_DEFAULT_KEY_HYPERPARAMS` to the tuple in the interfaces block.
  4. Replace `_2D_METRIC_KEYS` by a function:

```python
def _2d_metric_keys(names: Sequence[str]) -> tuple:
    return tuple(
        f"{name}_2d_{suffix}"
        for name in names
        for suffix in ("silhouette", "kmeans_ari", "kmeans_nmi", "knn_accuracy")
    )
```

  5. `_resolve_2d_embedding(run)`: replace the `tails/visualization` file lookup by

```python
    if run.viz_embeddings is not None and run.viz_embeddings.shape[1] == 2:
        return run.viz_embeddings
```

  (rest unchanged; update its docstring item 1 to "the stack's own trained 2D visualization head's embeddings").
  6. `collect_run_dirs`: an entry that `is_full_stack_run` is itself a run; otherwise `discover_full_stack_runs(path)`; resolve/dedupe/sort as before.
  7. `run_classification_accuracies(run)` becomes `return classification_accuracies_from_npz(run.embeddings)`.
  8. `benchmark_row`:

```python
def _label_sets(run: RunData) -> Dict[str, np.ndarray]:
    sets = {"spacegroup": run.embeddings["spacegroups"]}
    if run.stack == FAMILY:
        sets = {"family": run.embeddings["labels"], **sets}
    return sets


def benchmark_row(
    run: RunData,
    key_hyperparams: Sequence[str] = _DEFAULT_KEY_HYPERPARAMS,
) -> Dict[str, Any]:
    flat = run.flat_config
    label_sets = _label_sets(run)
    row: Dict[str, Any] = {}
    row.update(run_classification_accuracies(run))

    embedding_2d = _resolve_2d_embedding(run)
    if embedding_2d is not None:
        row.update(
            embedding_quality_metrics(
                embedding_2d, {f"{k}_2d": v for k, v in label_sets.items()}
            )
        )
    else:
        row.update({k: float("nan") for k in _2d_metric_keys(list(label_sets))})

    row["run_dir"] = str(run.run_dir)
    row["stack"] = run.stack
    row["model_kind"] = run.model_kind
    row["n_samples"] = int(run.embeddings["embeddings"].shape[0])
    for key in key_hyperparams:
        row[key] = _csv_safe(flat.get(key, ""))
    for metric, values in run.loss_history.items():
        if metric == "epoch":
            continue
        row[f"final_{metric}"] = float(values[-1]) if len(values) else float("nan")
    row.update(embedding_quality_metrics(run.embeddings["embeddings"], label_sets))
    return row
```

  Keep the long docstring, edited: "one row per (run, stack)"; the "key" column of the row `seed` comes from the **stack's** `seed` (flat key `seed`).
  9. `generate_benchmark_table(..., heads_name=None)`:

```python
    run_dirs = collect_run_dirs(inputs)
    rows = [
        benchmark_row(open_stack(d, stack, heads_name, allow_no_heads=True), key_hyperparams)
        for d in run_dirs
        for stack in trained_stack_names(d)
    ]
```

  and the plot block:

```python
    if plot:
        plots_dir = output_csv.parent / "benchmark_plots"
        for stack in dict.fromkeys(r["stack"] for r in rows):
            generate_benchmark_plots(
                [r for r in rows if r["stack"] == stack],
                plots_dir / stack,
                write_data_files=write_data_files,
            )
```

  10. Update the module docstring to say rows are per (run, stack).

- [ ] **Step 5: Run to verify**

Run: `pytest tests/test_pipeline_benchmark.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dim_red/pipeline/benchmark.py tests/test_pipeline_benchmark.py
git commit -m "Make dimred-benchmark produce one row and one plot set per stack

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: FullStack-only sweep

**Files:**
- Rewrite: `src/dim_red/pipeline/sweep.py`
- Modify: `src/dim_red/pipeline/config.py` (`SweepConfig.api_key` removal only; `expand_sweep` is removed in Task 10)
- Rewrite: `tests/test_pipeline_sweep.py`
- Test: `tests/test_pipeline_sweep.py`

**Interfaces:**
- Consumes: `SweepConfig(base, grid)`, `_set_dotted`, `full_stack_config_from_dict`, `FullStack`, `DEFAULT_HEADS_NAME`.
- Produces: `run_sweep(sweep: SweepConfig, cache_dir=None) -> List[Path]` — one FullStack run (body + heads `default`) per grid combination under a new `<output_dir>/<date>-<n>/`, run name `<base name>_<axis>-<value>_...`; writes `<sweep_dir>/sweep.yaml` (`base`, `grid`).

- [ ] **Step 1: Write the failing tests** (replace the whole file `tests/test_pipeline_sweep.py`)

```python
import pytest
import yaml

pytest.importorskip("jax")

from dim_red.pipeline.config import SweepConfig
from dim_red.pipeline.run_layout import discover_full_stack_runs, trained_stack_names
from dim_red.pipeline.sweep import run_sweep
from tests.fullstack_helpers import config_dict, fake_builder


@pytest.fixture
def patched(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dim_red.pipeline.full_stack.build_stack_dataset",
        fake_builder(tmp_path, []),
    )


def _sweep(tmp_path, grid):
    base = config_dict(tmp_path / "runs", experts=("cubic",))
    return SweepConfig(base=base, grid=grid)


def test_one_run_per_grid_combination(tmp_path, patched):
    sweep = _sweep(
        tmp_path,
        {"family.encoder.latent_dim": [3, 5], "experts.defaults.train.epochs": [2]},
    )
    run_dirs = run_sweep(sweep, cache_dir=tmp_path / "cache")
    assert len(run_dirs) == 2
    assert {p.parent for p in run_dirs} == {tmp_path / "runs" / run_dirs[0].parent.name}
    assert sorted(p.name for p in run_dirs) == [
        "fs_latent_dim-3_epochs-2",
        "fs_latent_dim-5_epochs-2",
    ]
    for d in run_dirs:
        assert trained_stack_names(d) == ["family", "cubic"]
        assert (d / "stacks" / "family" / "heads" / "default" / "heads.yaml").exists()


def test_sweep_dir_records_base_and_grid(tmp_path, patched):
    sweep = _sweep(tmp_path, {"family.encoder.latent_dim": [3]})
    (run_dir,) = run_sweep(sweep, cache_dir=tmp_path / "cache")
    saved = yaml.safe_load((run_dir.parent / "sweep.yaml").read_text())
    assert saved["grid"] == {"family.encoder.latent_dim": [3]}
    assert saved["base"]["name"] == "fs"


def test_discover_finds_the_sweeps_runs(tmp_path, patched):
    sweep = _sweep(tmp_path, {"family.encoder.latent_dim": [3, 5]})
    run_dirs = run_sweep(sweep, cache_dir=tmp_path / "cache")
    assert discover_full_stack_runs(run_dirs[0].parent) == sorted(run_dirs)


def test_second_sweep_gets_a_new_numbered_dir(tmp_path, patched):
    sweep = _sweep(tmp_path, {"family.encoder.latent_dim": [3]})
    (a,) = run_sweep(sweep, cache_dir=tmp_path / "cache")
    (b,) = run_sweep(sweep, cache_dir=tmp_path / "cache")
    assert a.parent != b.parent and b.parent.name.endswith("-2")


def test_invalid_combination_fails_before_anything_is_trained(tmp_path, patched):
    sweep = _sweep(tmp_path, {"family.encoder.latent_dimm": [3]})
    with pytest.raises(ValueError, match="unknown keys"):
        run_sweep(sweep, cache_dir=tmp_path / "cache")
    assert not (tmp_path / "runs").exists()


def test_cgcnn_style_base_is_refused_with_a_hint(tmp_path):
    sweep = SweepConfig(base={"model": "cgcnn", "output_dir": str(tmp_path)}, grid={})
    with pytest.raises(ValueError, match="dimred-run"):
        run_sweep(sweep)


def test_empty_grid_runs_the_base_once(tmp_path, patched):
    sweep = _sweep(tmp_path, {})
    run_dirs = run_sweep(sweep, cache_dir=tmp_path / "cache")
    assert [p.name for p in run_dirs] == ["fs"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_pipeline_sweep.py -v`
Expected: FAIL (old `run_sweep` expects `RunConfig`).

- [ ] **Step 3: Implement** — replace `src/dim_red/pipeline/sweep.py` entirely:

```python
"""Grid sweeps over ``FullStack`` configs: the Cartesian product of however
many dotted-path axes ``SweepConfig.grid`` declares (e.g.
``family.encoder.latent_dim``, ``experts.defaults.train.epochs``), one
``FullStack`` run (body + heads ``default``) per combination under a fresh
``<output_dir>/<date>-<n>/`` directory."""

from __future__ import annotations

import copy
import itertools
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import yaml

from dim_red.pipeline.config import SweepConfig, _set_dotted
from dim_red.pipeline.full_stack import DEFAULT_HEADS_NAME, FullStack
from dim_red.pipeline.full_stack_config import full_stack_config_from_dict

logger = logging.getLogger("dim_red.pipeline")

_SWEEP_DIR_RE = re.compile(r"^\d{8}-(\d+)$")


def _next_sweep_dir(output_dir: Path) -> Path:
    """``<output_dir>/<date>-<n>`` where ``n`` is one more than the highest
    sibling number already present (so concurrent sweeps never collide)."""
    today = datetime.now().strftime("%Y%m%d")
    last_n = 0
    if output_dir.exists():
        for p in output_dir.iterdir():
            m = _SWEEP_DIR_RE.match(p.name) if p.is_dir() else None
            if m:
                last_n = max(last_n, int(m.group(1)))
    sweep_dir = output_dir / f"{today}-{last_n + 1}"
    sweep_dir.mkdir(parents=True, exist_ok=False)
    return sweep_dir


def _slug(value) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", str(value)).strip("-")


def _run_name(base_name: str, keys: List[str], combo) -> str:
    parts = [f"{key.rsplit('.', 1)[-1]}-{_slug(value)}" for key, value in zip(keys, combo)]
    return "_".join([base_name, *parts])


def run_sweep(
    sweep: SweepConfig, cache_dir: Optional[Union[str, Path]] = None
) -> List[Path]:
    """Expand ``sweep.grid`` over ``sweep.base`` and run every combination
    sequentially. Every combination is parsed (and so validated) before the
    first one trains. Returns the run directories in grid order."""
    if "model_kind" not in sweep.base and "model" in sweep.base:
        raise ValueError(
            "dimred-sweep only supports FullStack configs (model_kind: "
            "supcon|supcon_mace); run cgcnn configs with dimred-run"
        )
    keys = list(sweep.grid)
    value_lists = [sweep.grid[k] for k in keys]
    combos = list(itertools.product(*value_lists)) if keys else [()]
    configs = []
    for combo in combos:
        d = copy.deepcopy(sweep.base)
        for key, value in zip(keys, combo):
            _set_dotted(d, key, value)
        config = full_stack_config_from_dict(d)
        if keys:
            config = _rename(config, _run_name(config.name, keys, combo))
        configs.append(config)

    output_dir = Path(sweep.output_dir)
    resolved_cache_dir = Path(cache_dir) if cache_dir else output_dir / "_dataset_cache"
    sweep_dir = _next_sweep_dir(output_dir)
    with open(sweep_dir / "sweep.yaml", "w") as f:
        yaml.safe_dump({"base": sweep.base, "grid": sweep.grid}, f, sort_keys=False)
    logger.info(
        "Starting sweep: %d axis/axes (%s) = %d run(s), saved under %s",
        len(keys), sorted(keys), len(configs), sweep_dir,
    )

    run_dirs = []
    for i, config in enumerate(configs, start=1):
        logger.info("[%d/%d] %s", i, len(configs), config.name)
        full_stack = FullStack.create(
            _with_output_dir(config, sweep_dir), cache_dir=resolved_cache_dir
        )
        full_stack.fit_body()
        full_stack.fit_heads(DEFAULT_HEADS_NAME)
        run_dirs.append(full_stack.run_dir)
    logger.info("Sweep complete: %d run(s) saved under %s", len(run_dirs), sweep_dir)
    return run_dirs


def _rename(config, name: str):
    import dataclasses

    return dataclasses.replace(config, name=name)


def _with_output_dir(config, sweep_dir: Path):
    import dataclasses

    return dataclasses.replace(config, output_dir=str(sweep_dir))
```

(Move the two `import dataclasses` to module top; shown inline only to keep the snippet short — black/isort will sort them.) In `config.py`, delete the `SweepConfig.api_key` property (it reads `base["fetch"]`, which no longer exists for FullStack).

The "invalid combination" test relies on the `_check_known_keys` message `unknown keys [...]` from plan A's parser and on the fact that all configs are parsed before `_next_sweep_dir` creates anything (`tmp_path/"runs"` must not exist).

- [ ] **Step 4: Run to verify**

Run: `pytest tests/test_pipeline_sweep.py -v`
Expected: PASS. (`tests/test_pipeline_config.py` still imports `SweepConfig.api_key`? Run `grep -n "api_key" tests/test_pipeline_config.py`; delete only that assertion.)

- [ ] **Step 5: Commit**

```bash
git add src/dim_red/pipeline/sweep.py src/dim_red/pipeline/config.py tests/test_pipeline_sweep.py tests/test_pipeline_config.py
git commit -m "Rewrite dimred-sweep as a FullStack-only grid sweep

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: CLI wiring

**Files:**
- Modify: `src/dim_red/pipeline/cli.py`, `pyproject.toml`
- Rewrite: `tests/test_pipeline_cli.py`
- Test: `tests/test_pipeline_cli.py`

**Interfaces:**
- Produces in `cli.py`: `_is_full_stack_config(d: dict) -> bool` (true iff any of `model_kind`, `family`, `experts` is a key); `_do_run`, `_do_rerun`, `_do_sweep`, `_do_apply(…, heads_name=None)`, `_do_compare(…, heads_name=None)`, `_do_benchmark(…, heads_name=None)`, new `_do_train_heads(config_path, run_dir, heads_name, stacks=None, cache_dir=None) -> Path` and `train_heads_command(argv=None)`; console script `dimred-train-heads`.

- [ ] **Step 1: Write the failing tests** (replace `tests/test_pipeline_cli.py`)

```python
import csv

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write

pytest.importorskip("jax")

from dim_red.pipeline import cli
from dim_red.pipeline import full_stack as fs_mod
from dim_red.pipeline.run_layout import head_names, trained_stack_names
from tests.fullstack_helpers import config_dict, fake_builder


@pytest.fixture
def patched(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dim_red.pipeline.full_stack.build_stack_dataset", fake_builder(tmp_path, [])
    )


def _write_config(tmp_path, name="cfg.yaml", **kwargs):
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump(config_dict(tmp_path / "runs", experts=("cubic",), **kwargs))
    )
    return str(path)


def test_config_kind_is_detected_by_key():
    assert cli._is_full_stack_config({"model_kind": "supcon"})
    assert cli._is_full_stack_config({"family": {}})
    assert cli._is_full_stack_config({"experts": {}})
    assert not cli._is_full_stack_config({"model": "cgcnn", "encoder": {}})


def test_run_trains_body_and_default_heads(tmp_path, patched):
    run_dir = cli._do_run(_write_config(tmp_path), cache_dir=str(tmp_path / "cache"))
    assert trained_stack_names(run_dir) == ["family", "cubic"]
    assert head_names(run_dir / "stacks" / "family") == ["default"]


def test_run_dispatches_cgcnn_configs_to_run_single(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "dim_red.pipeline.single_run.run_single",
        lambda config, cache_dir=None, **kw: seen.setdefault("kind", config.model_kind)
        and tmp_path,
    )
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": "cgcnn",
                "data_source": "pyxtal",
                "pyxtal": {"structures_per_spacegroup": 1},
                "encoder": {"encoder_hidden_dim": [8], "latent_dim": 4},
            }
        )
    )
    cli._do_run(str(path), None)
    assert seen["kind"] == "cgcnn"


def test_rerun_creates_a_new_run_dir_with_the_same_stacks(tmp_path, patched):
    first = cli._do_run(_write_config(tmp_path), cache_dir=str(tmp_path / "cache"))
    second = cli._do_rerun(str(first), cache_dir=str(tmp_path / "cache"))
    assert second != first and second.parent == first.parent
    assert trained_stack_names(second) == trained_stack_names(first)


def test_train_heads_adds_a_named_heads_set(tmp_path, patched):
    run = cli._do_run(_write_config(tmp_path), cache_dir=str(tmp_path / "cache"))
    heads_cfg = _write_config(tmp_path, name="heads.yaml", latent_dim=4)
    out = cli._do_train_heads(heads_cfg, str(run), heads_name="wide", stacks=["family"])
    assert head_names(run / "stacks" / "family") == ["default", "wide"]
    assert head_names(run / "stacks" / "cubic") == ["default"]
    assert out == run


def test_train_heads_refuses_an_existing_name(tmp_path, patched):
    run = cli._do_run(_write_config(tmp_path), cache_dir=str(tmp_path / "cache"))
    with pytest.raises(FileExistsError):
        cli._do_train_heads(_write_config(tmp_path, name="h.yaml"), str(run), "default")


def test_apply_on_a_fullstack_run_writes_predictions(tmp_path, patched, monkeypatch):
    run = cli._do_run(_write_config(tmp_path), cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(
        fs_mod,
        "featurize_structures",
        lambda atoms, kind, data: np.stack(
            [np.arange(6, dtype=np.float32) + i for i in range(len(atoms))]
        ),
    )
    path = tmp_path / "new.extxyz"
    write(str(path), [Atoms("Cu", positions=[[0, 0, 0]])] * 2, format="extxyz")
    out = cli._do_apply(str(path), str(run), None, None)
    assert len(list(csv.DictReader(open(out / "new_predictions.csv")))) == 2


def test_sweep_runs_every_combination(tmp_path, patched):
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        yaml.safe_dump(
            {
                "base": config_dict(tmp_path / "runs", experts=("cubic",)),
                "grid": {"family.encoder.latent_dim": [3, 5]},
            }
        )
    )
    run_dirs = cli._do_sweep(str(sweep), str(tmp_path / "cache"))
    assert len(run_dirs) == 2


def test_compare_and_benchmark_forward_heads_name(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "dim_red.pipeline.compare.generate_comparison_report",
        lambda sweep_dir, **kw: seen.update(compare=kw) or tmp_path,
    )
    monkeypatch.setattr(
        "dim_red.pipeline.benchmark.generate_benchmark_table",
        lambda inputs, out, **kw: seen.update(bench=kw) or tmp_path,
    )
    cli._do_compare("s", None, heads_name="h")
    cli._do_benchmark(["s"], str(tmp_path / "t.csv"), None, heads_name="h")
    assert seen["compare"]["heads_name"] == "h"
    assert seen["bench"]["heads_name"] == "h"


def test_train_heads_console_script_is_registered():
    import tomllib
    from pathlib import Path

    scripts = tomllib.loads(Path("pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["dimred-train-heads"] == "dim_red.pipeline.cli:train_heads_command"
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_pipeline_cli.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement in `src/dim_red/pipeline/cli.py`.**

  1. Module docstring header: replace the example list with

```
    dimred-run configs/full_stack.example.yaml
    dimred-sweep configs/full_stack_sweep.example.yaml
    dimred-rerun runs/full-stack-example
    dimred-train-heads configs/train_heads.example.yaml runs/full-stack-example --heads-name wide
    dimred-compare runs/20261007-1
    dimred-apply new_structures.extxyz runs/full-stack-example
    dimred-benchmark runs/20261007-1 --output runs/benchmark.csv
    dimred-train-tail configs/tail_train_classification.example.yaml <cgcnn run dir>
```

  2. Add helper and replace `_do_run`, `_do_rerun`, `_do_sweep`:

```python
def _is_full_stack_config(d: dict) -> bool:
    return any(key in d for key in ("model_kind", "family", "experts"))


def _do_run(config_path: str, cache_dir: Optional[str]) -> Path:
    from dim_red.pipeline.config import load_yaml

    _configure_console_logging()
    raw = load_yaml(config_path)
    cache = Path(cache_dir) if cache_dir else None
    if _is_full_stack_config(raw):
        from dim_red.pipeline.full_stack import DEFAULT_HEADS_NAME, FullStack
        from dim_red.pipeline.full_stack_config import full_stack_config_from_dict

        full_stack = FullStack.create(full_stack_config_from_dict(raw), cache_dir=cache)
        full_stack.fit_body()
        full_stack.fit_heads(DEFAULT_HEADS_NAME)
        run_dir = full_stack.run_dir
    else:
        from dim_red.pipeline.config import run_config_from_dict
        from dim_red.pipeline.single_run import run_single

        run_dir = run_single(run_config_from_dict(raw), cache_dir=cache)
    print(f"Run complete: {run_dir}")
    return run_dir


def _do_sweep(config_path: str, cache_dir: Optional[str]) -> list:
    from dim_red.pipeline.config import load_sweep_config
    from dim_red.pipeline.sweep import run_sweep

    _configure_console_logging()
    sweep_config = load_sweep_config(config_path)
    run_dirs = run_sweep(sweep_config, cache_dir=Path(cache_dir) if cache_dir else None)
    print(f"Sweep complete: {len(run_dirs)} run(s).")
    for run_dir in run_dirs:
        print(f" - {run_dir}")
    return run_dirs


def _do_rerun(run_dir: str, cache_dir: Optional[str]) -> Path:
    import yaml

    _configure_console_logging()
    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        raw = yaml.safe_load(f)
    cache = Path(cache_dir) if cache_dir else None
    if "stacks" in raw:  # a FullStack run: config.yaml is the resolved config
        from dim_red.pipeline.full_stack import DEFAULT_HEADS_NAME, FullStack
        from dim_red.pipeline.full_stack_config import (
            full_stack_config_from_resolved_dict,
        )

        # FullStack.create allocates a fresh, deduplicated directory.
        full_stack = FullStack.create(
            full_stack_config_from_resolved_dict(raw), cache_dir=cache
        )
        full_stack.fit_body()
        full_stack.fit_heads(DEFAULT_HEADS_NAME)
        new_run_dir = full_stack.run_dir
    else:
        from dim_red.pipeline.config import load_run_config
        from dim_red.pipeline.single_run import run_single

        config = dataclasses.replace(load_run_config(run_dir / "config.yaml"), name=None)
        new_run_dir = run_single(config, cache_dir=cache)
    print(f"Rerun complete: {new_run_dir}")
    return new_run_dir
```

  3. Add `_do_train_heads` and the command:

```python
def _do_train_heads(
    config_path: str,
    run_dir: str,
    heads_name: str,
    stacks: Optional[list] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    from dim_red.pipeline.config import load_yaml
    from dim_red.pipeline.full_stack import FullStack
    from dim_red.pipeline.full_stack_config import full_stack_config_from_dict

    _configure_console_logging()
    heads_config = full_stack_config_from_dict(load_yaml(config_path))
    full_stack = FullStack.open(run_dir, cache_dir=Path(cache_dir) if cache_dir else None)
    full_stack.fit_heads(heads_name, stacks=stacks, config=heads_config)
    print(f"Heads {heads_name!r} trained in {full_stack.run_dir}")
    return full_stack.run_dir


def train_heads_command(argv=None) -> None:
    """``dimred-train-heads <config> <run_dir> --heads-name NAME``: train a new
    set of classification + visualization heads on already-trained bodies."""
    parser = argparse.ArgumentParser(
        description="Train a named set of classification + visualization heads "
        "on the frozen bodies of an existing FullStack run. The config has the "
        "same schema as dimred-run's; only each listed stack's classifier/viz/"
        "seed blocks are read."
    )
    parser.add_argument("config", type=str, help="Path to a FullStack YAML config.")
    parser.add_argument("run_dir", type=str, help="Existing FullStack run directory.")
    parser.add_argument(
        "--heads-name", type=str, required=True, help="Name of the new heads set."
    )
    parser.add_argument(
        "--stacks",
        type=str,
        default=None,
        help="Comma-separated stacks to train (default: every stack in the config).",
    )
    args = parser.parse_args(argv)
    stacks = [s.strip() for s in args.stacks.split(",")] if args.stacks else None
    _do_train_heads(args.config, args.run_dir, args.heads_name, stacks)
```

  4. `_do_apply(..., heads_name=None)`: at the top, `from dim_red.pipeline.run_layout import is_full_stack_run`; if `is_full_stack_run(run_dir)`: `from dim_red.pipeline.full_stack import apply_to_structures`; `result_dir = apply_to_structures(run_dir, structures_path, output_dir=output_dir, heads_name=heads_name)`; print; return. Otherwise keep the existing cgcnn code. `apply_command`: add `--heads-name` (default None) and pass it; adjust the `run_dir` help text ("a completed run directory").
  5. `_do_compare(..., heads_name=None)` forwards `heads_name=heads_name` to `generate_comparison_report`; `compare_command` adds `--heads-name`. `_do_benchmark(..., heads_name=None)` forwards `heads_name=heads_name`; `benchmark_command` adds `--heads-name`, and the `--key-hyperparams` help text lists the new defaults (`model.encoder_hidden_dim,model.latent_dim,model.body_train.learning_rate,model.body_train.batch_size,model.body_train.epochs,seed`), `inputs` help drops "cross-model_kind" wording ("one row per run and stack").
  6. `train_tail_command` docstring/description: cgcnn-only ("freeze an already-trained cgcnn run's body and train a classification or visualization tail"); `_do_train_tail` unchanged.
  7. `pyproject.toml`: under `[project.scripts]` add `dimred-train-heads = "dim_red.pipeline.cli:train_heads_command"`.

- [ ] **Step 4: Run to verify**

Run: `pytest tests/test_pipeline_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dim_red/pipeline/cli.py pyproject.toml tests/test_pipeline_cli.py
git commit -m "Wire dimred-run/rerun/sweep/apply/compare/benchmark to FullStack; add dimred-train-heads

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Remove the old supcon paths

Everything new is green; now delete. **Do it as four commits (10a–10d), re-reading each branch before cutting it** — `run_single`/`train_tail` share branches with cgcnn. After each sub-step run only the named test files.

**Files:**
- Modify: `src/dim_red/pipeline/single_run.py`, `tail_training.py`, `inference.py`, `config.py`, `dataset_cache.py` (only if a deleted function becomes unused, see 10d)
- Modify/delete: `tests/test_pipeline_single_run.py`, `tests/test_pipeline_tail_training.py`, `tests/test_pipeline_tail_config.py`, `tests/test_pipeline_inference.py`, `tests/test_pipeline_config.py`, `tests/test_pipeline_dataset_cache.py`, `tests/test_supcon_tail_training.py`

**Interfaces:**
- After this task: `run_single` raises `ValueError` for `model_kind != "cgcnn"`; `RunConfig.model_kind` defaults to `"cgcnn"`; `TailTrainConfig.tail_kind ∈ {"classification", "visualization"}`; `tail_training._TAIL_MODEL_KINDS == {"classification": ("cgcnn",), "visualization": ("cgcnn",)}`.

#### 10a — `single_run.py`

- [ ] **Step 1: Add the guard** as the first statement of `run_single`'s body (after its docstring):

```python
    if config.model_kind != "cgcnn":
        raise ValueError(
            f"run_single only trains model_kind 'cgcnn', got {config.model_kind!r}. "
            "supcon/supcon_mace runs use dim_red.pipeline.full_stack.FullStack "
            "(dimred-run with a model_kind: config)."
        )
```

- [ ] **Step 2: Add the failing test** to `tests/test_pipeline_single_run.py`:

```python
def test_run_single_refuses_supcon_with_a_pointer_to_fullstack(tmp_path):
    config = _config(tmp_path, model_kind="supcon")  # reuse the module's config helper
    with pytest.raises(ValueError, match="FullStack"):
        run_single(config)
```

(Use whichever config-builder helper the module already has; if it has none for a `model_kind` override, build `dataclasses.replace(<existing cgcnn config>, model_kind="supcon")`.)

Run: `pytest tests/test_pipeline_single_run.py::test_run_single_refuses_supcon_with_a_pointer_to_fullstack -v` → PASS (the guard exists).

- [ ] **Step 3: Delete the dead code in `single_run.py`**, one construct at a time, keeping the cgcnn arm of each `if is_supcon: … elif is_cgcnn: …`:
  - the imports `train_tail`, `SupConEncoder`, `ProjectionTail`, `SupConTrainConfig`, `training_first_phase`;
  - `uses_mace_features`, `is_supcon`, `aux_mode = config.supcon.mode` and every `if is_supcon:` / `balanced_batching` branch (keep the cgcnn arm unconditionally);
  - the supcon part of `make_run_name` (`if config.model_kind in ("supcon", "supcon_mace"): …`);
  - the whole "Auto-train tails" block at the end (`if config.tails is not None: …`, the `can_classify_tail` logic);
  - docstring/comment references to supcon.
  Verify: `grep -n "is_supcon\|uses_mace\|supcon\|SupCon\|mace\|ProjectionTail\|train_tail\|config.tails\|config.batching" src/dim_red/pipeline/single_run.py` must print **only** lines you deliberately keep (the `("family_supcon", "family_supcon")` pairs in the history-column table if cgcnn history logging still needs a mapping — otherwise delete them too).

- [ ] **Step 4: Prune `tests/test_pipeline_single_run.py`**: delete every test whose name or body uses `model_kind="supcon"`/`"supcon_mace"`, `supcon:`/`batching:` config or `tails:`; keep the cgcnn tests. Run: `pytest tests/test_pipeline_single_run.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dim_red/pipeline/single_run.py tests/test_pipeline_single_run.py
git commit -m "Remove supcon/supcon_mace training and auto-tails from run_single (cgcnn only)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

#### 10b — `tail_training.py` and `inference.py`

- [ ] **Step 1: Failing test** in `tests/test_pipeline_tail_training.py`:

```python
def test_tail_kinds_and_model_kinds_are_cgcnn_only():
    from dim_red.pipeline import tail_training

    assert tail_training._TAIL_MODEL_KINDS == {
        "classification": ("cgcnn",),
        "visualization": ("cgcnn",),
    }
```

Run → FAIL.

- [ ] **Step 2: Implement.** In `tail_training.py`: set `_TAIL_MODEL_KINDS` to the dict above; delete `_train_hierarchical_supcon`, `_sanitize_family_dirname`, `_compute_native_soap_features`, `_compute_native_mace_features` **after** `grep -n "_sanitize_family_dirname\|_compute_native" src/dim_red/pipeline/*.py` shows no use outside the functions being deleted; remove the `hierarchical_supcon` dispatch in `train_tail`; keep `_classifier_eval_plots` (cgcnn tails use it) and update the module docstring. In `inference.py`: delete the `supcon`/`supcon_mace` branches of `_build_model`, `load_trained_run` (the standardization-recompute block, `_resolve_species`, `_raw_soap_matrix`, the `feature_mean/feature_std` plumbing if only supcon used it) and `encode_structures` (cgcnn branch only); keep `load_run_embeddings`/`RunEmbeddings` (used by `train_tail`), `plot_applied_in_latent_space`, `apply_model_to_structures`.
  Verify: `grep -n "supcon\|mace\|soap\|SupCon\|hierarchical" src/dim_red/pipeline/tail_training.py src/dim_red/pipeline/inference.py` prints only deliberate leftovers (the `dim_red.supcon.tail_training` primitive imports used by cgcnn tails stay).

- [ ] **Step 3: Prune tests**: `tests/test_pipeline_tail_training.py`, `tests/test_pipeline_inference.py`, `tests/test_supcon_tail_training.py` (only tests that import `pipeline.tail_training`/`inference` supcon behaviour; the primitives' own tests in `test_supcon_tail_training.py` stay). Run: `pytest tests/test_pipeline_tail_training.py tests/test_pipeline_inference.py tests/test_supcon_tail_training.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add src/dim_red/pipeline/tail_training.py src/dim_red/pipeline/inference.py tests/test_pipeline_tail_training.py tests/test_pipeline_inference.py tests/test_supcon_tail_training.py
git commit -m "Remove hierarchical_supcon and supcon inference paths (cgcnn tails only)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

#### 10c — `config.py`

- [ ] **Step 1: Failing tests** in `tests/test_pipeline_config.py` / `tests/test_pipeline_tail_config.py`:

```python
def test_run_config_rejects_non_cgcnn_models_with_a_pointer():
    with pytest.raises(ValueError, match="FullStack"):
        run_config_from_dict(
            {"model": "supcon", "data_source": "pyxtal",
             "pyxtal": {"structures_per_spacegroup": 1},
             "encoder": {"encoder_hidden_dim": [4], "latent_dim": 2}}
        )


def test_tail_kind_hierarchical_supcon_is_gone():
    with pytest.raises(ValueError, match="tail_kind"):
        TailTrainConfig(tail_kind="hierarchical_supcon")
```

Run → FAIL.

- [ ] **Step 2: Implement.** In `config.py`: in `run_config_from_dict` read `model_kind = str(d.get("model", "cgcnn"))` and raise `ValueError("model must be 'cgcnn' here; supcon/supcon_mace configs use the FullStack schema (model_kind: …) — see dim_red.pipeline.full_stack_config")` when it is not `"cgcnn"`; `RunConfig.model_kind` default `"cgcnn"`. Delete: `HierarchicalSupconTailConfig`, `_parse_hierarchical_supcon_tail_config`, `AutoTailsConfig` and `RunConfig.tails` (field, parsing block, the `tails` part of `run_config_to_dict`), `TailTrainConfig.hierarchical_supcon`, `"hierarchical_supcon"` from `_TAIL_KINDS` and from `tail_train_config_from_dict`/`tail_train_config_to_dict`, `expand_sweep`. Delete `SupConConfig`, `BalancedBatchingParams`, `BatchingConfig` and the `RunConfig` fields `supcon`, `batching`, `mace` **only if** `grep -rn "SupConConfig\|BalancedBatchingParams\|BatchingConfig\|\.supcon\b\|\.batching\b" src` shows no remaining user (FullStack uses its own `Batching`); **keep** `SoapConfig`, `MaceConfig`, `PyxtalConfig`, `AugmentationConfig`, `EarlyStoppingConfig`, `TrainSettings`, `TailTrainSettings`, `_dataclass_from_dict`, `_parse_train_settings`, `load_yaml`, `flatten_config_dict`, `_set_dotted`, `SweepConfig`, `load_sweep_config` (all used by `full_stack_config`/`sweep`).
  Verify: `python -c "import dim_red.pipeline.full_stack_config, dim_red.pipeline.sweep, dim_red.pipeline.cli"` succeeds.

- [ ] **Step 3: Prune tests** in `tests/test_pipeline_config.py`, `tests/test_pipeline_tail_config.py`, `tests/test_pipeline_dataset_cache.py` (supcon/batching/tails/hierarchical/expand_sweep only). Run: `pytest tests/test_pipeline_config.py tests/test_pipeline_tail_config.py tests/test_pipeline_dataset_cache.py tests/test_pipeline_full_stack_config.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add src/dim_red/pipeline/config.py tests/test_pipeline_config.py tests/test_pipeline_tail_config.py tests/test_pipeline_dataset_cache.py
git commit -m "Remove supcon, batching, auto-tail and hierarchical_supcon config from RunConfig

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

#### 10d — leftovers

- [ ] **Step 1:** `grep -rn "hierarchical_supcon\|HierarchicalSupcon\|predict_hierarchical\|_train_hierarchical" src tests --include=*.py` must print nothing; fix or delete what remains. `grep -n "def \|^class " src/dim_red/pipeline/dataset_cache.py | head -60` and `grep -rn "<name>" src` for each function: delete a `dataset_cache` function only if nothing in `src`, `tests`, `examples` calls it (the `get_or_build_pyxtal_*` functions used by `stack_data` and the cgcnn graph builders stay).
- [ ] **Step 2:** Run the touched test files: `pytest tests/test_pipeline_dataset_cache.py tests/test_pipeline_dataset_cache_characterization.py tests/test_pipeline_stack_data.py -v` → PASS.
- [ ] **Step 3: Commit** (`git commit -m "Remove dead supcon helpers left behind by the FullStack migration"`, with the Co-Authored-By trailer).

---

### Task 11: Migrate the example configs; test that every config loads

**Files:**
- Create: `configs/full_stack.example.yaml`, `configs/full_stack_best_combo.example.yaml`, `configs/full_stack_sweep.example.yaml`, `configs/train_heads.example.yaml`
- Delete: see Step 3
- Modify: `configs/tail_train_classification.example.yaml`, `configs/tail_train_visualization.example.yaml` (comments only: cgcnn)
- Test: `tests/test_example_configs.py`

**Interfaces:**
- Produces: `tests/test_example_configs.py::test_example_configs_load` — every YAML under `configs/` loads with exactly one loader: FullStack (`model_kind`/`family`/`experts`), sweep (`base` + `grid`; the base and one grid combination must parse), tail (`tail_kind`), else `run_config_from_dict`.

- [ ] **Step 1: Write the failing test** `tests/test_example_configs.py`

```python
import copy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("jax")

from dim_red.pipeline.config import _set_dotted, run_config_from_dict, tail_train_config_from_dict
from dim_red.pipeline.full_stack_config import full_stack_config_from_dict

CONFIGS = sorted(Path("configs").rglob("*.yaml"))


def _is_full_stack(d):
    return any(k in d for k in ("model_kind", "family", "experts"))


def _load(path):
    d = yaml.safe_load(path.read_text())
    if "base" in d and "grid" in d:
        base = d["base"]
        assert _is_full_stack(base), f"{path}: sweep base must be a FullStack config"
        full_stack_config_from_dict(base)
        first = copy.deepcopy(base)
        for key, values in d["grid"].items():
            _set_dotted(first, key, values[0])
        return full_stack_config_from_dict(first)
    if _is_full_stack(d):
        return full_stack_config_from_dict(d)
    if "tail_kind" in d:
        return tail_train_config_from_dict(d)
    return run_config_from_dict(d)


def test_there_are_example_configs():
    assert len(CONFIGS) >= 6


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p))
def test_example_configs_load(path):
    _load(path)


def test_full_stack_examples_cover_family_and_experts():
    cfg = full_stack_config_from_dict(
        yaml.safe_load(Path("configs/full_stack.example.yaml").read_text())
    )
    assert "family" in cfg.stacks and len(cfg.stacks) >= 3
    assert all(s.data.soap.element_agnostic for s in cfg.stacks.values())
```

Run: `pytest tests/test_example_configs.py -v` → FAIL (old configs fail, new ones missing).

- [ ] **Step 2: Create the new example configs.**

`configs/full_stack.example.yaml` (small, CPU, quick):

```yaml
# Quick try-it-out FullStack run: one family stack (crystal system) and two
# per-system experts (spacegroup), each with its OWN pyxtal dataset, split,
# standardization and seed. No MP_API_KEY needed.
# Run with: dimred-run configs/full_stack.example.yaml
name: full-stack-example
seed: 42
model_kind: supcon # or supcon_mace (then data.mace.checkpoint_path is required)
output_dir: runs

family:
  data:
    pyxtal:
      families: [cubic, hexagonal, orthorhombic]
      structures_per_family: 30
      distribution: uniform
      n_species: 2
    soap: {r_cut: 5.0, n_max: 4, l_max: 3, sigma: 0.5, element_agnostic: true} # mu2 is required for supcon
  encoder: {encoder_hidden_dim: [64, 32], latent_dim: 8}
  projection: {projection_dim: 32}
  contrastive: {tau: 0.1, distance: cosine}
  classifier: {hidden_dim: 16}
  viz: {hidden_dim: [16], viz_dim: 2, tau: 0.1, distance: euclidean}
  train: {epochs: 10, batch_size: 32, val_ratio: 0.2, device: cpu}

experts:
  defaults: # merged under each expert's own block
    data:
      pyxtal: {structures_per_spacegroup: 12, distribution: uniform, n_species: 2}
      soap: {r_cut: 5.0, n_max: 4, l_max: 3, sigma: 0.5, element_agnostic: true}
    encoder: {encoder_hidden_dim: [32], latent_dim: 8}
    projection: {projection_dim: 16}
    contrastive: {tau: 0.1, distance: cosine}
    classifier: {hidden_dim: 16}
    viz: {hidden_dim: [16], viz_dim: 2}
    train: {epochs: 10, batch_size: 32, val_ratio: 0.2, device: cpu}
  cubic: {}
  tetragonal: {encoder: {latent_dim: 6}} # per-expert override
```

`configs/full_stack_best_combo.example.yaml` — the round-7/round-15 recipe, GPU:

```yaml
# The "best combo" recipe (round 7 + round 15 findings): contrastive tau 0.05
# with cosine distance, wide trunk [256, 128] / latent 32, projection 128,
# classifier hidden 32, visualizer [64, 32], 200 epochs with early stopping.
# Run with: dimred-run configs/full_stack_best_combo.example.yaml --cache-dir runs/_shared_dataset_cache
name: full-stack-best-combo
seed: 42
model_kind: supcon
output_dir: runs/full_stack_best_combo

family:
  data:
    pyxtal:
      families: [triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic]
      structures_per_family: 500
      distribution: uniform
      n_species: 1
    augmentation:
      n_augmented: 4
      keep_original: true
      jitter_probability: 0.5
      jitter_std: 0.05
      vacancy_probability: 0.5
      vacancy_atom_probability: 1
      max_vacancies: 1
      supercell_radius: 4.0
    soap: {r_cut: 6.0, n_max: 8, l_max: 6, sigma: 0.5, element_agnostic: true}
  encoder: {encoder_hidden_dim: [256, 128], latent_dim: 32}
  projection: {projection_dim: 128}
  contrastive: {tau: 0.05, distance: cosine, lambda_norm: 0.0}
  classifier: {hidden_dim: 32}
  viz: {hidden_dim: [64, 32], viz_dim: 2, tau: 0.1, distance: euclidean}
  train: &train
    epochs: 200
    batch_size: 128
    learning_rate: 0.001
    val_ratio: 0.2
    device: gpu
    early_stopping: {enabled: true, patience: 20, min_delta: 0.0001, restore_best_weights: true}
  batching: {strategy: random}

experts:
  defaults:
    data:
      pyxtal: {structures_per_spacegroup: 60, distribution: uniform, n_species: 1}
      augmentation:
        n_augmented: 4
        keep_original: true
        jitter_probability: 0.5
        jitter_std: 0.05
        vacancy_probability: 0.5
        vacancy_atom_probability: 1
        max_vacancies: 1
        supercell_radius: 4.0
      soap: {r_cut: 6.0, n_max: 8, l_max: 6, sigma: 0.5, element_agnostic: true}
    encoder: {encoder_hidden_dim: [128, 64], latent_dim: 16}
    projection: {projection_dim: 64}
    contrastive: {tau: 0.05, distance: cosine}
    classifier: {hidden_dim: 32}
    viz: {hidden_dim: [64, 32], viz_dim: 2, tau: 0.1, distance: euclidean}
    train: *train
  triclinic: {}
  monoclinic: {}
  orthorhombic: {}
  tetragonal: {}
  trigonal: {}
  hexagonal: {}
  cubic: {}
```

`configs/full_stack_sweep.example.yaml`:

```yaml
# Grid sweep over a FullStack config: one run (body + heads "default") per
# combination. Axes are dotted paths into the base config.
# Run with: dimred-sweep configs/full_stack_sweep.example.yaml
# Then: dimred-compare runs/<date>-<n>   and   dimred-benchmark runs/<date>-<n> --output runs/<date>-<n>/benchmark.csv
base:
  name: sweep
  seed: 42
  model_kind: supcon
  output_dir: runs/full_stack_sweep
  family:
    data:
      pyxtal: {families: [cubic, hexagonal, orthorhombic], structures_per_family: 30, n_species: 2}
      soap: {r_cut: 5.0, n_max: 4, l_max: 3, sigma: 0.5, element_agnostic: true}
    encoder: {encoder_hidden_dim: [64, 32], latent_dim: 8}
    train: {epochs: 10, batch_size: 32, device: cpu}
grid:
  family.encoder.latent_dim: [4, 8]
  family.contrastive.tau: [0.05, 0.1]
```

`configs/train_heads.example.yaml`: a copy of `configs/full_stack.example.yaml` with a header comment

```yaml
# Same schema as dimred-run. For every stack it lists, ONLY the classifier,
# viz and seed blocks are read (the other blocks must be present so the file
# validates, but are ignored); the trained bodies are never touched.
# Run with: dimred-train-heads configs/train_heads.example.yaml <run_dir> --heads-name wide [--stacks family,cubic]
```

and `family.viz.hidden_dim: [64, 32]`, `family.classifier.hidden_dim: 32`, experts' `viz.hidden_dim: [64, 32]`, `name: train-heads-example`.

- [ ] **Step 3: Delete the obsolete configs.** First confirm each is supcon-only:

Run: `grep -L "model: cgcnn" configs/*.yaml configs/round19_sweep/*.yaml | sort`

`git rm` exactly the files from that list **except** the four new `full_stack*`/`train_heads*` files and the two kept tail examples (`tail_train_classification.example.yaml`, `tail_train_visualization.example.yaml`): that removes `single_run_supcon*.yaml` (4), `single_run_supcon_hierarchical.example.yaml`, `single_run_pyxtal.example.yaml`, `sweep_supcon.example.yaml`, `tuning_sweep_supcon*.yaml` (2), `benchmark_pyxtal_base.example.yaml` (if it is supcon-only), `tail_train_hierarchical_supcon_*.yaml` (4), `tail_train_visualization_best_combo.example.yaml`, `tail_train_visualization_encoder_256_128.example.yaml`, the whole `configs/round19_sweep/` directory. Also `git rm configs/sweep_cgcnn.example.yaml configs/tuning_sweep_cgcnn.example.yaml slurm/tuning_sweep_cgcnn.yaml slurm/tuning_sweep_supcon.yaml` (no `dimred-sweep` for cgcnn; the supcon one is replaced by `full_stack_sweep.example.yaml`).

In the two kept tail examples change the header comments to say "cgcnn run" (and drop the `hierarchical_supcon` mention in `tail_kind`'s comment); in `tail_train_visualization.example.yaml` likewise.

- [ ] **Step 4: Run**

Run: `pytest tests/test_example_configs.py -v`
Expected: PASS for every remaining YAML. Any failure is a real config error: fix the YAML, never the test.

- [ ] **Step 5: Commit**

```bash
git add -A configs slurm/tuning_sweep_cgcnn.yaml slurm/tuning_sweep_supcon.yaml tests/test_example_configs.py
git commit -m "Migrate example configs to the FullStack schema; test that every config loads

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Scripts, notebook, slurm

**Files:**
- Modify: `examples/evaluate_holdout_pyxtal.py`, `examples/evaluate_ns_trajectories.py`, `examples/ns_grid_from_trajectories.ipynb`
- Delete: `examples/tune_sg_visualization_hidden_dims.py`
- Unchanged (verify): `examples/baseline_family_classifier.py`, `examples/pyxtal_structure_generation.ipynb`
- Modify/delete: `slurm/*.sbatch`

These scripts are bespoke analyses of past experiments; no unit tests exist. The acceptance criteria below are mechanical. **Old → new API map:**

| Old (removed) | New |
|---|---|
| `load_trained_run(run_dir)` + `encode_structures(loaded, atoms)` + hand-built `SupConEncoder`/`ClassificationTail`/`VisualizationTail` from `tails/<subdir>` | `FullStack.open(run_dir)`; `.predict(atoms, heads_name=…)` → `Prediction` (family, family_proba, expert, spacegroup, spacegroup_proba, viz_family, viz_expert); `.predict_stack(name, atoms)` → `StackPrediction(classes, proba, viz)` |
| `compute_soap(...)` + `apply_standardization(...)` on new structures | done inside `predict` (`dim_red.pipeline.featurize`) |
| `config.soap.species` list handling | not needed (`mu2`) |
| `HIERARCHICAL_SUBDIR`, `FAMILY_VIZ_SUBDIR`, `SG_VIZ_*` | `heads_name` of the FullStack run |
| per-frame `family_logits` | `np.log(np.clip(StackPrediction.proba, 1e-12, 1))` of the family stack |
| `sg_pred` | `Prediction.spacegroup` with `-1` where `None` |

- [ ] **Step 1: `evaluate_holdout_pyxtal.py`.** Read the script. Keep: held-out structure generation (`generate_structures`, `make_supercell_for_radius`), metrics (`embedding_quality_metrics`), report/plot code and CLI flags. Replace: the run loading/encoding/expert-routing section with `FullStack.open(run_dir)` + `predict(...)`; replace `--hierarchical-subdir`/viz-subdir flags with `--heads-name`; delete outputs that only the old oracle routing could produce (`spacegroup_probs_oracle`) and say so in the commit message. Acceptance: `python examples/evaluate_holdout_pyxtal.py --help` exits 0; `grep -n "load_trained_run\|encode_structures\|load_run_config\|SupConEncoder\|ClassificationTail\|VisualizationTail\|ProjectionTail" examples/evaluate_holdout_pyxtal.py` prints nothing.
- [ ] **Step 2: `evaluate_ns_trajectories.py`.** Same treatment; keep the per-replica `ns.<i>.eval.npz` format the notebook reads (keys: `iter, enthalpy, log_w, family_logits, family_pred_idx, sg_pred, soap_atom_msd_rel, family_classes, pressure_gpa`, per the notebook's `EVAL_KEYS`), producing `family_logits`/`sg_pred` via the map above. Acceptance: `--help` exits 0 and the same `grep` as Step 1 prints nothing. If a key cannot be produced from `Prediction` (e.g. raw SOAP vectors that only the old script saved), keep computing it with `dim_red.pipeline.featurize.featurize_structures` rather than dropping it.
- [ ] **Step 3: notebook `ns_grid_from_trajectories.ipynb`.** Cells 20–21 (run dir, `HIERARCHICAL_SUBDIR`, `FAMILY_VIZ_SUBDIR`, `SG_VIZ_*`, `load_trained_run` imports) → `FullStack.open(DIMRED_RUN_DIR)` and `heads_name`; cell 24's text mentions the old tail names: update it to say "family-level visualization head". Edit with `NotebookEdit`; do not re-execute the notebook (it needs external data). Acceptance: `python -c "import json,sys; json.load(open('examples/ns_grid_from_trajectories.ipynb'))"` succeeds and `grep -c "load_trained_run\|hierarchical_supcon" examples/ns_grid_from_trajectories.ipynb` prints `0`.
- [ ] **Step 4:** `git rm examples/tune_sg_visualization_hidden_dims.py` (its job — tuning the visualizer width per expert — is now `dimred-train-heads` with per-expert `viz.hidden_dim` and `dimred-compare`/`dimred-benchmark` over the heads sets). Verify `examples/baseline_family_classifier.py --help` still exits 0.
- [ ] **Step 5: slurm.** `git rm slurm/round19_phase1.sbatch slurm/round19_phase2.sbatch slurm/supcon_mace_hierarchical_supcon_best_combo_scale.sbatch`. `tune_all_models.sbatch`: reduce the per-kind task array to a single FullStack sweep: `SWEEP_CONFIG=configs/full_stack_sweep.example.yaml` (or the user's own via `--export`), `dimred-sweep "${SWEEP_CONFIG}" --cache-dir runs/_shared_dataset_cache`, then `dimred-benchmark "${SWEEP_DIR}" --output "${SWEEP_DIR}/benchmark_table.csv"`; update its comments ("cgcnn is trained with dimred-run"). `evaluate_ns_trajectories.sbatch`: update the script's flags to match Step 2. Leave the `.novalocal` retry block and `compare.sbatch`/`ablation_family_only_cpu.sbatch`/`supcon_mace_best_combo_scale.sbatch`/`warmup_shared_dataset.sbatch` command lines (`dimred-run`, `dimred-compare` are still valid; their configs are user-owned). Acceptance: `bash -n slurm/*.sbatch` exits 0 for every kept file; `grep -n "dimred-train-tail\|hierarchical\|round19\|tuning_sweep" slurm/*.sbatch` prints nothing.
- [ ] **Step 6: Commit**

```bash
git add -A examples slurm
git commit -m "Migrate evaluation scripts, notebook and slurm jobs to FullStack

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Documentation

**Files:**
- Create: `docs/fullstack.md`
- Modify: `docs/index.md` (toctree), `docs/architecture.md`, `docs/pipeline.md`, `docs/runconfig.md`, `docs/tails.md`, `docs/model_kinds.md`, `docs/training.md`, `docs/examples.md`, `docs/aux_supcon.md`, `docs/tools.md`
- Modify: `CLAUDE.md`, `src/dim_red/pipeline/CLAUDE.md`, `src/dim_red/supcon/CLAUDE.md`, `REFACTOR_TODO.md`

- [ ] **Step 1: Write `docs/fullstack.md`** with these sections, each short: **What it is** (family stack + up to 7 independent experts, same architecture, label = crystal system / spacegroup; two sub-phases per stack); **Config** (point at `configs/full_stack.example.yaml`; `model_kind`, `family`, `experts.defaults` + overrides, `data.soap.element_agnostic: true` required, pyxtal only, per-stack seed `run_seed + index`); **Commands** (the table from the spec: `dimred-run`, `rerun`, `sweep`, `train-heads --heads-name`, `compare`, `benchmark`, `apply`); **On disk** (the layout tree from plan A, incl. that `.<x>.tmp` are interrupted writes and are ignored); **Predicting** (`FullStack.open(run).predict(...)`, missing expert → `None`, `device="cpu"` always works for GPU-trained runs); **Known quirks** (history keys `*_family_supcon` also in experts where they contrast spacegroups; plots relabel; `dimred-train-heads` reads only classifier/viz/seed; the family and expert datasets share no structures so end-to-end evaluation needs a held-out set, see `examples/evaluate_holdout_pyxtal.py`).
- [ ] **Step 2:** Add `fullstack` to the toctree in `docs/index.md`. In each of the other pages, remove or rewrite the supcon/`hierarchical_supcon`/`tails:`/`model: supcon` content: say that `supcon`/`supcon_mace` are `FullStack` and link to `fullstack.md`; keep cgcnn content. `docs/tails.md` becomes "cgcnn tails (`dimred-train-tail`)" plus a pointer to heads in `fullstack.md`. Find every stale mention with: `grep -rn "hierarchical_supcon\|model: supcon\|tails:\|dimred-train-tail\|sg_\|round19" docs/*.md`.
- [ ] **Step 3: CLAUDE.md files.** Root `CLAUDE.md`: replace the `supcon/`, `pipeline.tail_training`/`RunConfig.tails` and early-stopping paragraphs' references to the removed paths with a **FullStack** bullet (what it is, where it lives: `supcon/stack.py`, `pipeline/full_stack*.py`, `run_layout.py`, `featurize.py`) and a **traps** bullet: GPU-trained stacks need `device="cpu"` on CPU-only machines; `*_family_supcon` history keys in experts; readers must ignore `.*.tmp`; the strict YAML parser rejects old configs and `tests/test_example_configs.py` loads every shipped config; `mu2` is required. Early stopping now applies through `StackConfig`/`TrainConfig` (same fields). `src/dim_red/pipeline/CLAUDE.md` and `src/dim_red/supcon/CLAUDE.md`: delete sections for removed modules/paths, add `run_layout`/`featurize`/`predict`. Keep each file short (the point of this step is fewer, truer lines).
- [ ] **Step 4: `REFACTOR_TODO.md`:** mark §1 (single stack) and §2 (inference with experts) done by FullStack/predict, delete §5b's "single_run/train_tail monolithic" items that no longer exist, keep §2b/§3/§4/§6 items that still apply.
- [ ] **Step 5: Build the docs**

Run: `sphinx-build -W -b html docs docs/_build/html`
Expected: build succeeds with no warnings (`docs/_build` is gitignored; if not, do not add it).

- [ ] **Step 6: Commit**

```bash
git add docs CLAUDE.md src/dim_red/pipeline/CLAUDE.md src/dim_red/supcon/CLAUDE.md REFACTOR_TODO.md
git commit -m "Document FullStack: new docs page, trimmed CLAUDE.md files, TODO update

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Final verification (after Task 13)

Run only the files this plan touched (never the full suite unless the user asks):

```bash
pytest tests/test_pipeline_run_layout.py tests/test_pipeline_featurize.py \
  tests/test_pipeline_full_stack_predict.py tests/test_pipeline_full_stack.py \
  tests/test_pipeline_full_stack_config.py tests/test_pipeline_stack_data.py \
  tests/test_pipeline_compare.py tests/test_pipeline_benchmark.py \
  tests/test_pipeline_sweep.py tests/test_pipeline_cli.py \
  tests/test_example_configs.py tests/test_supcon_stack.py tests/test_supcon_stack_golden.py \
  tests/test_pipeline_single_run.py tests/test_pipeline_tail_training.py \
  tests/test_pipeline_tail_config.py tests/test_pipeline_inference.py \
  tests/test_pipeline_config.py tests/test_pipeline_dataset_cache.py -v
pytest tests/test_pipeline_full_stack_e2e.py --runslow -v   # one real training, opt-in
```

## Plan self-review

- **Spec coverage:** B1 (no legacy) → Task 3 has one layout, Task 10 deletes old paths; B2 (per-stack compare/benchmark) → Tasks 6–7; B3 (cgcnn out of compare/benchmark, kept elsewhere) → Tasks 7, 9, 10; B4 (`mu2`) → Tasks 1, 4; B5 (sweep FullStack-only) → Task 8; B6 (history keys unchanged, labels from role) → Task 6 (`display_metric`); problems 1–4 → Tasks 2, 6, 3, 11; spec sections 3 (predict/featurize/device/apply) → Tasks 2, 4, 5; section 4 (commands, deletions) → Tasks 9, 10; section 5 (migration, config test, docs) → Tasks 11–13; section 6 order → matches task order (mu2 → device → layout → predict → compare/benchmark → commands → deletions → migration → docs).
- **Placeholders:** Tasks 10 and 12 describe deletions/rewrites of large bespoke files by named constructs plus mechanical `grep` acceptance checks instead of full diffs: the exact lines depend on code that must be re-read at execution time (shared cgcnn branches; analysis scripts without tests). Every new module, function, test and config has full code.
- **Type consistency:** `RunData` fields (`run_dir, stack, model_kind, flat_config, loss_history, embeddings, viz_embeddings, heads_name`) are used identically in Tasks 3, 6, 7; `open_stack(run_dir, stack, heads_name=None, allow_no_heads=False)` in Tasks 3, 6, 7; `Prediction`/`StackPrediction` in Tasks 4, 5, 12; `DEFAULT_HEADS_NAME` in Tasks 4, 8, 9; `load_stack(name, heads_name=None, device=None)` in Tasks 2, 4.
- **Review Focus coverage:** items 1→T3, 2→T2, 3→T3/T4/T6/T7, 4→T4, 5→T4, 6→T6, 7→T11.
