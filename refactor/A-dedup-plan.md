# Refactor A (deduplicazioni sicure) — Piano di implementazione

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** eliminare le duplicazioni sicure di `dataset_cache.py`, delle config/costanti di training SupCon e delle chiavi di history del tail di classificazione, senza cambiare alcun risultato.

**Architecture:** prima si fissano con test di caratterizzazione (che passano sul codice attuale) chiavi di cache, file, ritorni e log delle sei `get_or_build_*`; poi le sei funzioni diventano wrapper sottili di un'unica `_get_or_build(featurizer, source, ...)`. Poi `TailTrainConfig` diventa alias di `TrainConfig` e le costanti ripetute hanno una sola fonte. Per ultimo si rinominano le chiavi di history del tail di classificazione.

**Tech Stack:** Python 3.11, numpy, ase, jax/flax/optax (solo nei test di supcon), pytest. Ambiente conda `dmred`.

**Spec:** `refactor/A-dedup-design.md` (da leggere insieme a questo piano).

## Global Constraints

- Ambiente: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate dmred` prima di ogni comando Python/pytest.
- Le chiavi di cache **non cambiano**: `_cache_key`, `_pyxtal_cache_key`, `_graph_cache_key`, `_pyxtal_graph_cache_key`, `_mace_cache_key`, `_pyxtal_mace_cache_key` e `_augmentation_payload`/`_soap_kwargs_payload` non si toccano.
- Le firme e i valori di ritorno delle 6 funzioni pubbliche `get_or_build_*` e delle 3 `build_*_for_run` non cambiano; i messaggi di log restano identici.
- `pipeline/config.py` resta importabile senza jax: non importare `dim_red.supcon` da lì.
- Non toccare le chiavi di history di `cgcnn` (`train_family_ce`, `train_spacegroup_ce`, ...) né `single_run._EPOCH_COMPONENT_LABELS`.
- Test: eseguire **solo** i file di test indicati in ogni task, mai l'intera suite; lanciare i test pesanti uno alla volta (non in parallelo: OOM su GPU). Dopo una passata di sola formattazione (hook black/isort) non rilanciare i test.
- Commit: un commit per task, messaggio che termina con `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Se l'hook `pre-commit` riformatta dei file e il commit si interrompe: `git add -u` e ripetere lo stesso commit. Niente `git push`.
- I file di documentazione del refactor stanno in `refactor/`, non in `docs/` (Sphinx con `-W` leggerebbe i `.md` di `docs/`).

## Review Focus

Casi che la specifica implica ma che nessun test esistente esercita; ognuno ha il suo test nel task indicato.

1. Cache hit senza `pyxtal` installato: oggi `get_or_build_pyxtal_*` importa `dim_red.generate` anche su una hit; dopo il refactor non deve più farlo (Task 2, Step 1).
2. Fetch vuoto o generazione vuota: `ValueError` e nessun file lasciato in `cache_dir` (Task 1).
3. Cache senza il file `.extxyz` o con schema vecchio, per grafo e MACE (finora testato solo per SOAP): viene ricostruita (Task 1).
4. Sistemi cristallini in maiuscolo (`"CUBIC"`): label capitalizzata, chiave normalizzata (Task 1).
5. `dataclasses.replace(TailTrainConfig(), ...)` dopo l'alias restituisce un `TrainConfig` valido (Task 3).
6. `available_loss_metrics` ordina insieme le chiavi nuove (`train_ce`/`val_ce`) e quelle vecchie di `cgcnn` (Task 4).

## File Structure

- `tests/test_pipeline_dataset_cache_characterization.py` (nuovo, Task 1): fissa il comportamento osservabile della cache.
- `src/dim_red/pipeline/dataset_cache.py` (Task 2): aggiunge `_Featurizer`, `_StructureSource`, `_get_or_build`; le sei funzioni diventano wrapper.
- `src/dim_red/supcon/training.py`, `src/dim_red/supcon/tail_training.py` (Task 3): alias `TailTrainConfig`, costanti con fonte unica.
- `src/dim_red/supcon/tail_training.py`, `src/dim_red/pipeline/compare.py` e test (Task 4): rename chiavi.
- `refactor/A-dedup-design.md`, `REFACTOR_TODO.md`, `src/dim_red/supcon/CLAUDE.md`, `src/dim_red/pipeline/CLAUDE.md` (Task 5): documentazione e verifica finale.

---

### Task 1: Test di caratterizzazione della cache

Passano sul codice **attuale**; devono restare verdi per tutto il resto del lavoro.

**Files:**
- Create: `tests/test_pipeline_dataset_cache_characterization.py`

**Interfaces:**
- Consumes: le sei `get_or_build_*` e le sei funzioni di chiave di `dim_red.pipeline.dataset_cache` (firme attuali).
- Produces: la rete di sicurezza per il Task 2.

- [ ] **Step 1: Creare il file di test**

```python
"""Characterization tests for ``dim_red.pipeline.dataset_cache``.

They pin the observable behavior of the six ``get_or_build_*`` functions
(cache keys, cache file names and contents, return shapes, log lines) so the
internals can be restructured without invalidating existing caches or
changing any result. Expected values were recorded from the code before the
refactor.
"""

import logging
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from dim_red.augmentation import AugmentationConfig
from dim_red.pipeline import dataset_cache as dc
from dim_red.pipeline.config import PyxtalConfig

AUG = AugmentationConfig(n_augmented=1, jitter_probability=1.0, jitter_std=0.05, seed=7)
PYXTAL = PyxtalConfig(spacegroups=[225], structures_per_spacegroup=2)
SOAP_KW = {"r_cut": 3.0, "n_max": 2, "l_max": 2}
GRAPH_KW = {"radius": 3.0, "max_num_nbr": 4}
MACE_KW = {"checkpoint_path": "x.npz", "r_max": 6.0, "pooling": "mean"}

# (function, args, key with augmentation, key without augmentation)
KEY_CASES = {
    "soap_fetch": (dc._cache_key, (["cubic"], 2, SOAP_KW), "3ffce6cf603d9752", "ea9f81daed72cf06"),
    "soap_pyxtal": (dc._pyxtal_cache_key, (PYXTAL, 0, SOAP_KW), "pyxtal-31fd2343feb45e6c", "pyxtal-b2a947f6a4f60ea9"),
    "graph_fetch": (dc._graph_cache_key, (["cubic"], 2, GRAPH_KW), "cgcnn-69b0745d9560b06d", "cgcnn-ffb0ff20517873c9"),
    "graph_pyxtal": (dc._pyxtal_graph_cache_key, (PYXTAL, 0, GRAPH_KW), "cgcnn-pyxtal-e72d0d941da0384d", "cgcnn-pyxtal-63d7ed469d4b7650"),
    "mace_fetch": (dc._mace_cache_key, (["cubic"], 2, MACE_KW), "mace-22b2e5b06a39e187", "mace-3d2614f589d1cc69"),
    "mace_pyxtal": (dc._pyxtal_mace_cache_key, (PYXTAL, 0, MACE_KW), "mace-pyxtal-02c79b69134102bd", "mace-pyxtal-3204c36da14dfa9f"),
}


@pytest.mark.parametrize("name", sorted(KEY_CASES))
def test_cache_keys_are_pinned(name):
    fn, args, with_aug, without_aug = KEY_CASES[name]
    assert fn(*args, AUG) == with_aug
    assert fn(*args) == without_aug


def _atoms(symbol, material_id, spacegroup, family=None):
    atoms = Atoms(
        symbol * 2,
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    atoms.info["material_id"] = material_id
    atoms.info["spacegroup"] = spacegroup
    if family is not None:
        atoms.info["family"] = family
    return atoms


FETCHED = lambda: [_atoms("Cu", "mp-1", 225), _atoms("Fe", "mp-2", 229)]  # noqa: E731
GENERATED = lambda: [  # noqa: E731
    _atoms("Cu", "pyxtal-225-0", 225, "Cubic"),
    _atoms("Fe", "pyxtal-225-1", 225, "Cubic"),
]
FAKE_X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
FAKE_MEAN = np.array([2.0, 3.0], dtype=np.float32)
FAKE_STD = np.array([1.0, 1.0], dtype=np.float32)
SOAP_FILE_KEYS = ["X", "labels", "material_ids", "spacegroups", "feature_mean", "feature_std"]
GRAPH_FILE_KEYS = [
    "local_species_idx", "nbr_idx", "nbr_fea", "nbr_mask", "atom_mask",
    "labels", "material_ids", "spacegroups",
]  # fmt: skip


def _run(fn, tmp_path, caplog, **kwargs):
    """Call ``fn`` twice: a cache miss then a hit. Returns both results."""
    with caplog.at_level(logging.INFO, logger="dim_red.pipeline"):
        first = fn(cache_dir=tmp_path, **kwargs)
        second = fn(cache_dir=tmp_path, **kwargs)
    return first, second


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


def test_soap_fetch(tmp_path, caplog):
    with patch.object(dc, "fetch_structures_by_crystal_system", side_effect=lambda **kw: FETCHED()) as fetch, \
         patch.object(dc, "_compute_soap_and_standardize", return_value=(FAKE_X, FAKE_MEAN, FAKE_STD)) as feat:  # fmt: skip
        first, second = _run(
            dc.get_or_build_dataset, tmp_path, caplog,
            crystal_systems=["cubic"], soap_kwargs=SOAP_KW, limit_per_system=2, augmentation=AUG,
        )  # fmt: skip
    assert fetch.call_count == 1 and feat.call_count == 1
    key = KEY_CASES["soap_fetch"][2]
    assert len(first) == 7 and len(second) == 7
    X, labels, ids, sgs, path, mean, std = first
    np.testing.assert_array_equal(X, FAKE_X)
    assert labels == ["Cubic"] * 4
    assert ids == ["mp-1", "mp-1", "mp-2", "mp-2"]
    assert sgs == [225, 225, 229, 229]
    assert path == tmp_path / f"{key}.extxyz" and path.exists()
    assert sorted(np.load(tmp_path / f"{key}.npz").files) == sorted(SOAP_FILE_KEYS)
    assert second[1:4] == first[1:4] and second[4] == path
    msgs = _messages(caplog)
    assert any(m.startswith(f"Dataset cache miss ({key}); fetching structures") for m in msgs)
    assert any(m.startswith(f"Dataset cache hit ({key}) for crystal_systems=") for m in msgs)


def test_soap_pyxtal(tmp_path, caplog):
    with patch("dim_red.generate.generate_structures", side_effect=lambda cfg: GENERATED()) as gen, \
         patch.object(dc, "_compute_soap_and_standardize", return_value=(FAKE_X, FAKE_MEAN, FAKE_STD)) as feat:  # fmt: skip
        first, second = _run(
            dc.get_or_build_pyxtal_dataset, tmp_path, caplog,
            pyxtal_config=PYXTAL, seed=0, soap_kwargs=SOAP_KW, augmentation=AUG,
        )  # fmt: skip
    assert gen.call_count == 1 and feat.call_count == 1
    key = KEY_CASES["soap_pyxtal"][2]
    assert len(first) == 7
    X, labels, ids, sgs, path, mean, std = first
    assert labels == ["Cubic"] * 4
    assert ids == ["pyxtal-225-0", "pyxtal-225-0", "pyxtal-225-1", "pyxtal-225-1"]
    assert sgs == [225] * 4
    assert path == tmp_path / f"{key}.extxyz" and path.exists()
    assert sorted(np.load(tmp_path / f"{key}.npz").files) == sorted(SOAP_FILE_KEYS)
    msgs = _messages(caplog)
    assert any(m.startswith(f"Dataset cache miss ({key}); generating structures with pyxtal") for m in msgs)
    assert any(m == f"Dataset cache hit ({key}) for pyxtal generation" for m in msgs)


def test_graph_fetch(tmp_path, caplog):
    with patch.object(dc, "fetch_structures_by_crystal_system", side_effect=lambda **kw: FETCHED()) as fetch:
        first, second = _run(
            dc.get_or_build_cgcnn_dataset, tmp_path, caplog,
            crystal_systems=["cubic"], graph_kwargs=GRAPH_KW, limit_per_system=2, augmentation=AUG,
        )  # fmt: skip
    assert fetch.call_count == 1
    key = KEY_CASES["graph_fetch"][2]
    assert len(first) == 5 and len(first[0]) == 5
    arrays, labels, ids, sgs, path = first
    assert path == tmp_path / f"{key}.extxyz" and path.exists()
    assert sorted(np.load(tmp_path / f"{key}.npz").files) == sorted(GRAPH_FILE_KEYS)
    for a, b in zip(arrays, second[0]):
        np.testing.assert_array_equal(a, b)
    msgs = _messages(caplog)
    assert any(m.startswith(f"CGCNN dataset cache miss ({key}); fetching structures") for m in msgs)
    assert any(m.startswith(f"CGCNN dataset cache hit ({key}) for crystal_systems=") for m in msgs)


def test_graph_pyxtal(tmp_path, caplog):
    with patch("dim_red.generate.generate_structures", side_effect=lambda cfg: GENERATED()) as gen:
        first, second = _run(
            dc.get_or_build_pyxtal_cgcnn_dataset, tmp_path, caplog,
            pyxtal_config=PYXTAL, seed=0, graph_kwargs=GRAPH_KW, augmentation=AUG,
        )  # fmt: skip
    assert gen.call_count == 1
    key = KEY_CASES["graph_pyxtal"][2]
    assert len(first) == 5
    path = first[4]
    assert path == tmp_path / f"{key}.extxyz" and path.exists()
    assert sorted(np.load(tmp_path / f"{key}.npz").files) == sorted(GRAPH_FILE_KEYS)
    assert set(first[1]) == {"Cubic"}


def test_mace_fetch(tmp_path, caplog):
    with patch.object(dc, "fetch_structures_by_crystal_system", side_effect=lambda **kw: FETCHED()) as fetch, \
         patch.object(dc, "_compute_mace_and_standardize", return_value=(FAKE_X, FAKE_MEAN, FAKE_STD)) as feat:  # fmt: skip
        first, second = _run(
            dc.get_or_build_mace_dataset, tmp_path, caplog,
            crystal_systems=["cubic"], mace_kwargs=MACE_KW, limit_per_system=2, augmentation=AUG,
        )  # fmt: skip
    assert fetch.call_count == 1 and feat.call_count == 1
    key = KEY_CASES["mace_fetch"][2]
    assert len(first) == 7
    assert first[4] == tmp_path / f"{key}.extxyz" and first[4].exists()
    assert sorted(np.load(tmp_path / f"{key}.npz").files) == sorted(SOAP_FILE_KEYS)
    msgs = _messages(caplog)
    assert any(m.startswith(f"MACE dataset cache miss ({key}); fetching structures") for m in msgs)
    assert any(m.startswith(f"MACE dataset cache hit ({key}) for crystal_systems=") for m in msgs)


def test_mace_pyxtal(tmp_path, caplog):
    with patch("dim_red.generate.generate_structures", side_effect=lambda cfg: GENERATED()) as gen, \
         patch.object(dc, "_compute_mace_and_standardize", return_value=(FAKE_X, FAKE_MEAN, FAKE_STD)) as feat:  # fmt: skip
        first, second = _run(
            dc.get_or_build_pyxtal_mace_dataset, tmp_path, caplog,
            pyxtal_config=PYXTAL, seed=0, mace_kwargs=MACE_KW, augmentation=AUG,
        )  # fmt: skip
    assert gen.call_count == 1 and feat.call_count == 1
    key = KEY_CASES["mace_pyxtal"][2]
    assert len(first) == 7
    assert first[4] == tmp_path / f"{key}.extxyz"
    assert sorted(np.load(tmp_path / f"{key}.npz").files) == sorted(SOAP_FILE_KEYS)


# --- Edge cases pinned from the pre-refactor behavior ----------------------


def test_fetch_labels_are_capitalized_whatever_the_input_case(tmp_path):
    with patch.object(dc, "fetch_structures_by_crystal_system", side_effect=lambda **kw: FETCHED()), \
         patch.object(dc, "_compute_soap_and_standardize", return_value=(FAKE_X, FAKE_MEAN, FAKE_STD)):  # fmt: skip
        labels = dc.get_or_build_dataset(
            crystal_systems=["CUBIC"], soap_kwargs=SOAP_KW, limit_per_system=2, cache_dir=tmp_path,
        )[1]  # fmt: skip
    assert labels == ["Cubic", "Cubic"]


def test_nothing_fetched_raises_and_leaves_no_cache_files(tmp_path):
    with patch.object(dc, "fetch_structures_by_crystal_system", return_value=[]):
        with pytest.raises(ValueError, match="No structures fetched"):
            dc.get_or_build_cgcnn_dataset(
                crystal_systems=["cubic"], graph_kwargs=GRAPH_KW, limit_per_system=2, cache_dir=tmp_path,
            )  # fmt: skip
    assert list(tmp_path.iterdir()) == []


def test_nothing_generated_raises_and_leaves_no_cache_files(tmp_path):
    with patch("dim_red.generate.generate_structures", return_value=[]):
        with pytest.raises(ValueError, match="pyxtal generated no structures"):
            dc.get_or_build_pyxtal_mace_dataset(
                pyxtal_config=PYXTAL, seed=0, mace_kwargs=MACE_KW, cache_dir=tmp_path,
            )  # fmt: skip
    assert list(tmp_path.iterdir()) == []


def test_graph_cache_without_structures_file_is_rebuilt(tmp_path):
    with patch.object(dc, "fetch_structures_by_crystal_system", side_effect=lambda **kw: FETCHED()) as fetch:
        kwargs = dict(crystal_systems=["cubic"], graph_kwargs=GRAPH_KW, limit_per_system=2, cache_dir=tmp_path)  # fmt: skip
        dc.get_or_build_cgcnn_dataset(**kwargs)
        key = dc._graph_cache_key(["cubic"], 2, GRAPH_KW)
        (tmp_path / f"{key}.extxyz").unlink()
        dc.get_or_build_cgcnn_dataset(**kwargs)
    assert fetch.call_count == 2
    assert (tmp_path / f"{key}.extxyz").exists()


def test_mace_cache_with_stale_schema_is_rebuilt(tmp_path):
    with patch("dim_red.generate.generate_structures", side_effect=lambda cfg: GENERATED()) as gen, \
         patch.object(dc, "_compute_mace_and_standardize", return_value=(FAKE_X, FAKE_MEAN, FAKE_STD)):  # fmt: skip
        kwargs = dict(pyxtal_config=PYXTAL, seed=0, mace_kwargs=MACE_KW, cache_dir=tmp_path)
        dc.get_or_build_pyxtal_mace_dataset(**kwargs)
        key = dc._pyxtal_mace_cache_key(PYXTAL, 0, MACE_KW)
        np.savez(tmp_path / f"{key}.npz", X=FAKE_X)  # drops the other fields
        dc.get_or_build_pyxtal_mace_dataset(**kwargs)
    assert gen.call_count == 2
```

- [ ] **Step 2: Eseguire i test sul codice attuale**

Run: `pytest tests/test_pipeline_dataset_cache_characterization.py -q`
Expected: `17 passed`. Se un test fallisce NON modificare `dataset_cache.py`: o l'attesa nel test è sbagliata (correggerla in base al comportamento attuale) o il file è stato incollato male.

- [ ] **Step 3: Controllare che i test esistenti restino verdi**

Run: `pytest tests/test_pipeline_dataset_cache.py -q`
Expected: tutti passano (36).

- [ ] **Step 4: Commit**

```bash
git add tests/test_pipeline_dataset_cache_characterization.py
git commit -m "Add characterization tests for dataset_cache (keys, files, returns, logs)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Una sola sequenza `get_or_build` per le sei funzioni

**Files:**
- Modify: `src/dim_red/pipeline/dataset_cache.py`
- Test: `tests/test_pipeline_dataset_cache_characterization.py`, `tests/test_pipeline_dataset_cache.py`

**Interfaces:**
- Consumes: Task 1 (i test di caratterizzazione) e le helper già presenti nel modulo: `_load_cached_dataset`, `_save_dataset_cache`, `_load_cached_graph_dataset`, `_save_graph_dataset_cache`, `_compute_soap_and_standardize`, `_compute_graphs`, `_compute_mace_and_standardize`, `_save_structures_cache`, `_structures_cache_path`, `_UNKNOWN_SPACEGROUP`.
- Produces: `_Featurizer`, `_StructureSource`, `_soap_featurizer()`, `_graph_featurizer()`, `_mace_featurizer()`, `_fetch_source(crystal_systems, limit_per_system, api_key)`, `_pyxtal_source(pyxtal_config, seed)`, `_get_or_build(featurizer, source, key, kwargs, cache_dir, augmentation) -> tuple`. Le sei funzioni pubbliche mantengono nome, firma e ritorno.

- [ ] **Step 1: Scrivere il test rosso (Review Focus 1)**

Aggiungere in fondo a `tests/test_pipeline_dataset_cache_characterization.py`:

```python
def test_pyxtal_cache_hit_does_not_import_pyxtal(tmp_path):
    """A cache hit must not need ``dim_red.generate`` (hence pyxtal)."""
    import sys

    kwargs = dict(pyxtal_config=PYXTAL, seed=0, soap_kwargs=SOAP_KW, cache_dir=tmp_path)
    with patch("dim_red.generate.generate_structures", side_effect=lambda cfg: GENERATED()), \
         patch.object(dc, "_compute_soap_and_standardize", return_value=(FAKE_X, FAKE_MEAN, FAKE_STD)):  # fmt: skip
        dc.get_or_build_pyxtal_dataset(**kwargs)  # miss: fills the cache
    with patch.dict(sys.modules, {"dim_red.generate": None}):
        result = dc.get_or_build_pyxtal_dataset(**kwargs)  # hit: must not import it
    assert len(result) == 7
```

Run: `pytest tests/test_pipeline_dataset_cache_characterization.py::test_pyxtal_cache_hit_does_not_import_pyxtal -v`
Expected: FAIL con `ModuleNotFoundError` (sottoclasse di `ImportError`): oggi l'import avviene prima del controllo della cache.

- [ ] **Step 2: Estendere gli import di tipo**

In `src/dim_red/pipeline/dataset_cache.py` sostituire

```python
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union
```

con

```python
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
```

- [ ] **Step 3: Inserire la sequenza condivisa**

Inserire il blocco seguente **subito prima** di `def _resolve_augmentation(` (le funzioni helper del grafo e di MACE sono definite più sotto nel file: va bene, vengono risolte al momento della chiamata):

```python
# --- Shared get-or-build sequence ------------------------------------------
#
# Every ``get_or_build_*`` function below follows the same steps (compute the
# cache key, try to load, otherwise obtain structures, augment, cache the
# structures, featurize, cache the arrays, return); only the featurizer and
# the structure source differ. Both are small objects looked up through module
# globals at call time (the lambdas), so tests patching e.g.
# ``dataset_cache.compute_soap`` or ``_compute_mace_and_standardize`` keep
# working.


@dataclasses.dataclass(frozen=True)
class _Featurizer:
    """How one feature kind (SOAP / CGCNN graph / MACE) is computed, cached
    and returned.

    Attributes:
        log_prefix: Start of the cache hit/miss log lines (e.g. ``"Dataset"``).
        load: ``cache_path -> (features, labels, material_ids, spacegroups)``
            or ``None`` on a cache miss/stale schema. ``features`` is opaque
            to the shared sequence.
        compute: ``(atoms_list, kwargs) -> features``.
        save: ``(cache_path, features, labels, material_ids, spacegroups)``.
        result: ``(features, labels, material_ids, spacegroups,
            structures_path) -> the tuple the public function returns``.
    """

    log_prefix: str
    load: Callable[[Path], Optional[tuple]]
    compute: Callable[[list, Dict[str, Any]], Any]
    save: Callable[..., None]
    result: Callable[..., tuple]


def _soap_load(cache_path: Path) -> Optional[tuple]:
    cached = _load_cached_dataset(cache_path)
    if cached is None:
        return None
    X, labels, material_ids, spacegroups, mean, std = cached
    return (X, mean, std), labels, material_ids, spacegroups


def _soap_save(cache_path, features, labels, material_ids, spacegroups) -> None:
    X_std, feature_mean, feature_std = features
    _save_dataset_cache(
        cache_path, X_std, labels, material_ids, spacegroups, feature_mean, feature_std
    )


def _soap_result(features, labels, material_ids, spacegroups, structures_path):
    X_std, feature_mean, feature_std = features
    return (
        X_std,
        labels,
        material_ids,
        spacegroups,
        structures_path,
        feature_mean,
        feature_std,
    )


def _graph_load(cache_path: Path) -> Optional[tuple]:
    cached = _load_cached_graph_dataset(cache_path)
    if cached is None:
        return None
    graph_arrays, labels, material_ids, spacegroups = cached
    return graph_arrays, labels, material_ids, spacegroups


def _graph_save(cache_path, features, labels, material_ids, spacegroups) -> None:
    _save_graph_dataset_cache(
        cache_path, *features, labels, material_ids, spacegroups
    )


def _graph_result(features, labels, material_ids, spacegroups, structures_path):
    return tuple(features), labels, material_ids, spacegroups, structures_path


def _soap_featurizer() -> _Featurizer:
    return _Featurizer(
        log_prefix="Dataset",
        load=_soap_load,
        compute=lambda atoms, kw: _compute_soap_and_standardize(atoms, kw),
        save=_soap_save,
        result=_soap_result,
    )


def _graph_featurizer() -> _Featurizer:
    return _Featurizer(
        log_prefix="CGCNN dataset",
        load=_graph_load,
        compute=lambda atoms, kw: _compute_graphs(atoms, kw),
        save=_graph_save,
        result=_graph_result,
    )


def _mace_featurizer() -> _Featurizer:
    return _Featurizer(
        log_prefix="MACE dataset",
        load=_soap_load,  # same array schema as SOAP
        compute=lambda atoms, kw: _compute_mace_and_standardize(atoms, kw),
        save=_soap_save,
        result=_soap_result,
    )


@dataclasses.dataclass(frozen=True)
class _StructureSource:
    """Where a dataset's structures come from.

    Attributes:
        hit_detail: Tail of the cache-hit log line.
        miss_detail: Tail of the cache-miss log line.
        produce: ``augmentation -> (atoms_list, labels, material_ids,
            spacegroups)``, with augmentation already applied.
    """

    hit_detail: Tuple[str, tuple]
    miss_detail: Tuple[str, tuple]
    produce: Callable[[Optional[AugmentationConfig]], tuple]


def _fetch_source(
    crystal_systems: Sequence[str], limit_per_system: int, api_key: Optional[str]
) -> _StructureSource:
    def produce(augmentation):
        all_atoms = []
        labels: List[str] = []
        for cs in crystal_systems:
            atoms_list = fetch_structures_by_crystal_system(
                crystal_system=cs, api_key=api_key, limit=limit_per_system
            )
            n_fetched = len(atoms_list)
            if augmentation is not None:
                atoms_list = augment_structures(atoms_list, augmentation)
                logger.info(
                    "Fetched %d structures for crystal_system=%s (%d after augmentation)",
                    n_fetched,
                    cs,
                    len(atoms_list),
                )
            else:
                logger.info(
                    "Fetched %d structures for crystal_system=%s", n_fetched, cs
                )
            all_atoms.extend(atoms_list)
            labels.extend([cs.capitalize()] * len(atoms_list))

        if not all_atoms:
            raise ValueError("No structures fetched for the requested crystal systems.")

        material_ids = [a.info.get("material_id", "unknown") for a in all_atoms]
        spacegroups = [a.info.get("spacegroup", _UNKNOWN_SPACEGROUP) for a in all_atoms]
        return all_atoms, labels, material_ids, spacegroups

    systems = list(crystal_systems)
    return _StructureSource(
        hit_detail=("for crystal_systems=%s", (systems,)),
        miss_detail=("fetching structures for crystal_systems=%s", (systems,)),
        produce=produce,
    )


def _pyxtal_source(pyxtal_config: "PyxtalConfig", seed: int) -> _StructureSource:
    def produce(augmentation):
        # Lazy: dim_red.generate requires pyxtal, not a hard dim_red dependency.
        from dim_red.generate import GenerationConfig, generate_structures

        generation_kwargs = dataclasses.asdict(pyxtal_config)
        generation_kwargs.pop("seed", None)  # the resolved "seed" arg wins
        if generation_kwargs.get("candidate_num_ions") is None:
            generation_kwargs.pop("candidate_num_ions")
        all_atoms = generate_structures(
            GenerationConfig(seed=seed, **generation_kwargs)
        )

        if not all_atoms:
            raise ValueError(
                "pyxtal generated no structures for the requested configuration."
            )

        n_generated = len(all_atoms)
        if augmentation is not None:
            all_atoms = augment_structures(all_atoms, augmentation)
            logger.info(
                "Generated %d structures with pyxtal (%d after augmentation)",
                n_generated,
                len(all_atoms),
            )

        labels = [a.info["family"] for a in all_atoms]
        material_ids = [a.info["material_id"] for a in all_atoms]
        spacegroups = [a.info["spacegroup"] for a in all_atoms]
        return all_atoms, labels, material_ids, spacegroups

    return _StructureSource(
        hit_detail=("for pyxtal generation", ()),
        miss_detail=("generating structures with pyxtal", ()),
        produce=produce,
    )


def _get_or_build(
    featurizer: _Featurizer,
    source: _StructureSource,
    key: str,
    kwargs: Dict[str, Any],
    cache_dir: Union[str, Path],
    augmentation: Optional[AugmentationConfig],
) -> tuple:
    """The shared cache-or-build sequence behind every ``get_or_build_*``."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.npz"
    structures_path = _structures_cache_path(cache_path)

    cached = featurizer.load(cache_path)
    if cached is not None:
        fmt, args = source.hit_detail
        logger.info(f"{featurizer.log_prefix} cache hit (%s) " + fmt, key, *args)
        features, labels, material_ids, spacegroups = cached
        return featurizer.result(
            features, labels, material_ids, spacegroups, structures_path
        )

    fmt, args = source.miss_detail
    logger.info(f"{featurizer.log_prefix} cache miss (%s); " + fmt, key, *args)

    all_atoms, labels, material_ids, spacegroups = source.produce(augmentation)
    _save_structures_cache(cache_path, all_atoms)
    features = featurizer.compute(all_atoms, kwargs)
    featurizer.save(cache_path, features, labels, material_ids, spacegroups)
    return featurizer.result(
        features, labels, material_ids, spacegroups, structures_path
    )
```

- [ ] **Step 4: Sostituire le due funzioni SOAP**

Sostituire tutto da `def get_or_build_dataset(` fino a (escluso) il commento `# --- Shared get-or-build sequence` aggiunto allo Step 3 — cioè le due funzioni `get_or_build_dataset` e `get_or_build_pyxtal_dataset` con i loro docstring — con:

```python
def get_or_build_dataset(
    crystal_systems: Sequence[str],
    soap_kwargs: Dict[str, Any],
    limit_per_system: int,
    cache_dir: Union[str, Path],
    api_key: Optional[str] = None,
    augmentation: Optional[AugmentationConfig] = None,
) -> Tuple[np.ndarray, List[str], List[str], List[int], Path, np.ndarray, np.ndarray]:
    """Fetch structures, compute a global SOAP descriptor per structure, and
    standardize the result -- reusing a cached copy on disk when available.

    Args:
        crystal_systems: Crystal systems to include (as accepted by
            ``fetch_structures_by_crystal_system``).
        soap_kwargs: Keyword arguments for ``compute_soap`` minus ``average``
            (the pipeline always requests ``average="outer"``).
        limit_per_system: Max structures fetched per crystal system.
        cache_dir: Directory where cached datasets (``<hash>.npz``/
            ``<hash>.extxyz``) live.
        api_key: Materials Project API key (falls back to ``MP_API_KEY``).
        augmentation: Optional ``dim_red.augmentation.AugmentationConfig``
            applied to each crystal system's fetched structures. ``None``
            (default) disables augmentation.

    Returns:
        ``(X_std, labels, material_ids, spacegroups, structures_path,
        feature_mean, feature_std)``: ``X_std`` has shape
        ``(n_samples, n_features)``, ``spacegroups`` holds the MP spacegroup
        number (``-1`` when unavailable), ``structures_path`` is the cached
        extended-XYZ file with the exact ``Atoms`` (same order), and
        ``feature_mean``/``feature_std`` are the standardization statistics
        ``X_std`` was derived from, cached so
        ``dim_red.pipeline.inference.load_trained_run`` never needs to
        recompute SOAP.
    """
    key = _cache_key(crystal_systems, limit_per_system, soap_kwargs, augmentation)
    return _get_or_build(
        _soap_featurizer(),
        _fetch_source(crystal_systems, limit_per_system, api_key),
        key,
        soap_kwargs,
        cache_dir,
        augmentation,
    )


def get_or_build_pyxtal_dataset(
    pyxtal_config: "PyxtalConfig",
    seed: int,
    soap_kwargs: Dict[str, Any],
    cache_dir: Union[str, Path],
    augmentation: Optional[AugmentationConfig] = None,
) -> Tuple[np.ndarray, List[str], List[str], List[int], Path, np.ndarray, np.ndarray]:
    """The ``pyxtal`` counterpart to ``get_or_build_dataset``: generate a
    synthetic structure database with ``dim_red.generate``, compute and
    standardize SOAP descriptors, reusing a cached copy when available.

    Args:
        pyxtal_config: Generation settings (``dim_red.pipeline.config.PyxtalConfig``).
        seed: The *effective* seed to generate with, taking priority over
            ``pyxtal_config.seed`` (resolving ``pyxtal_config.seed or
            RunConfig.seed`` is the caller's job, see ``build_dataset_for_run``).
        soap_kwargs: Keyword arguments for ``compute_soap`` minus ``average``.
        cache_dir: Directory where cached datasets live.
        augmentation: Optional ``dim_red.augmentation.AugmentationConfig``
            applied to the generated structures before SOAP.

    Returns:
        Same 7-element shape as ``get_or_build_dataset``, with ``labels``
        holding each structure's crystal family.
    """
    key = _pyxtal_cache_key(pyxtal_config, seed, soap_kwargs, augmentation)
    return _get_or_build(
        _soap_featurizer(),
        _pyxtal_source(pyxtal_config, seed),
        key,
        soap_kwargs,
        cache_dir,
        augmentation,
    )
```

Nota sull'ordine: nel file il blocco dello Step 3 viene dopo le due funzioni SOAP; queste usano quei nomi solo a runtime, quindi l'ordine non conta.

- [ ] **Step 5: Sostituire le due funzioni del grafo**

Sostituire tutto da `def get_or_build_cgcnn_dataset(` fino a (escluso) `def build_graph_dataset_for_run(` — le due funzioni `get_or_build_cgcnn_dataset` e `get_or_build_pyxtal_cgcnn_dataset` — con:

```python
def get_or_build_cgcnn_dataset(
    crystal_systems: Sequence[str],
    graph_kwargs: Dict[str, Any],
    limit_per_system: int,
    cache_dir: Union[str, Path],
    api_key: Optional[str] = None,
    augmentation: Optional[AugmentationConfig] = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    List[str],
    List[str],
    List[int],
    Path,
]:
    """The graph-based counterpart to ``get_or_build_dataset``: fetch
    structures and build CGCNN graph arrays, reusing a cached copy when
    available.

    Args:
        crystal_systems: Crystal systems to include.
        graph_kwargs: Keyword arguments for ``atoms_list_to_graph_arrays``
            (``dim_red.pipeline.config.GraphConfig.graph_kwargs()``).
        limit_per_system: Max structures fetched per crystal system.
        cache_dir: Directory where cached datasets live.
        api_key: Materials Project API key (falls back to ``MP_API_KEY``).
        augmentation: Optional ``dim_red.augmentation.AugmentationConfig``,
            applied before graph construction.

    Returns:
        ``(graph_arrays, labels, material_ids, spacegroups, structures_path)``
        where ``graph_arrays`` is the 5-tuple ``(local_species_idx, nbr_idx,
        nbr_fea, nbr_mask, atom_mask)`` -- 5 elements, not 7: no
        ``feature_mean``/``feature_std`` for graph features.
    """
    key = _graph_cache_key(
        crystal_systems, limit_per_system, graph_kwargs, augmentation
    )
    return _get_or_build(
        _graph_featurizer(),
        _fetch_source(crystal_systems, limit_per_system, api_key),
        key,
        graph_kwargs,
        cache_dir,
        augmentation,
    )


def get_or_build_pyxtal_cgcnn_dataset(
    pyxtal_config: "PyxtalConfig",
    seed: int,
    graph_kwargs: Dict[str, Any],
    cache_dir: Union[str, Path],
    augmentation: Optional[AugmentationConfig] = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    List[str],
    List[str],
    List[int],
    Path,
]:
    """The ``pyxtal`` counterpart to ``get_or_build_cgcnn_dataset``; same
    5-element return, with ``labels`` holding each structure's crystal
    family. ``seed`` is the *effective* generation seed (see
    ``get_or_build_pyxtal_dataset``).
    """
    key = _pyxtal_graph_cache_key(pyxtal_config, seed, graph_kwargs, augmentation)
    return _get_or_build(
        _graph_featurizer(),
        _pyxtal_source(pyxtal_config, seed),
        key,
        graph_kwargs,
        cache_dir,
        augmentation,
    )
```

- [ ] **Step 6: Sostituire le due funzioni MACE**

Sostituire tutto da `def get_or_build_mace_dataset(` fino a (escluso) `def build_mace_dataset_for_run(` con:

```python
def get_or_build_mace_dataset(
    crystal_systems: Sequence[str],
    mace_kwargs: Dict[str, Any],
    limit_per_system: int,
    cache_dir: Union[str, Path],
    api_key: Optional[str] = None,
    augmentation: Optional[AugmentationConfig] = None,
) -> Tuple[np.ndarray, List[str], List[str], List[int], Path, np.ndarray, np.ndarray]:
    """The MACE-embedding counterpart to ``get_or_build_dataset``: fetch
    structures, embed them with a frozen MACE model and standardize,
    reusing a cached copy when available. Same 7-element return and cache
    array schema as the SOAP path; only the featurization differs.

    Args:
        crystal_systems: Crystal systems to include.
        mace_kwargs: Keyword arguments for ``dim_red.mace.model.MaceEncoder``
            (``dim_red.pipeline.config.MaceConfig.mace_kwargs()``).
        limit_per_system: Max structures fetched per crystal system.
        cache_dir: Directory where cached datasets live.
        api_key: Materials Project API key (falls back to ``MP_API_KEY``).
        augmentation: Optional ``dim_red.augmentation.AugmentationConfig``.
    """
    key = _mace_cache_key(crystal_systems, limit_per_system, mace_kwargs, augmentation)
    return _get_or_build(
        _mace_featurizer(),
        _fetch_source(crystal_systems, limit_per_system, api_key),
        key,
        mace_kwargs,
        cache_dir,
        augmentation,
    )


def get_or_build_pyxtal_mace_dataset(
    pyxtal_config: "PyxtalConfig",
    seed: int,
    mace_kwargs: Dict[str, Any],
    cache_dir: Union[str, Path],
    augmentation: Optional[AugmentationConfig] = None,
) -> Tuple[np.ndarray, List[str], List[str], List[int], Path, np.ndarray, np.ndarray]:
    """The ``pyxtal`` counterpart to ``get_or_build_mace_dataset``; ``seed``
    is the *effective* generation seed (see ``get_or_build_pyxtal_dataset``).
    """
    key = _pyxtal_mace_cache_key(pyxtal_config, seed, mace_kwargs, augmentation)
    return _get_or_build(
        _mace_featurizer(),
        _pyxtal_source(pyxtal_config, seed),
        key,
        mace_kwargs,
        cache_dir,
        augmentation,
    )
```

- [ ] **Step 7: Aggiornare i docstring del modulo**

Nel docstring in cima a `dataset_cache.py` aggiungere in fondo, prima delle virgolette di chiusura:

```
All six ``get_or_build_*`` functions are thin wrappers around the single
``_get_or_build`` sequence, parameterized by a ``_Featurizer`` (SOAP, CGCNN
graph or MACE) and a ``_StructureSource`` (Materials Project fetch or pyxtal
generation).
```

Inoltre nel docstring di `build_mace_dataset_for_run` sostituire `only when\n    ``config.model_kind == "mace"``` con `only when\n    ``config.model_kind == "supcon_mace"``` (il model_kind `mace` non esiste più).

- [ ] **Step 8: Eseguire i test**

Run, uno alla volta:
`pytest tests/test_pipeline_dataset_cache_characterization.py -q` → Expected: `18 passed` (incluso il test dello Step 1, ora verde).
`pytest tests/test_pipeline_dataset_cache.py -q` → Expected: 36 passati.
`pytest tests/test_pipeline_single_run.py -q` → Expected: 24 passati.
`pytest tests/test_pipeline_inference.py -q` → Expected: 19 passati.

- [ ] **Step 9: Verificare che non ci siano import inutilizzati**

Run: `python -c "import ast,sys; t=ast.parse(open('src/dim_red/pipeline/dataset_cache.py').read()); print('ok')"` e controllare a occhio che `Callable` sia usato (lo è, in `_Featurizer`/`_StructureSource`).

- [ ] **Step 10: Commit**

```bash
git add -u src tests
git commit -m "Replace the six get_or_build_* bodies with one shared _get_or_build sequence

Same cache keys, files, returns and log lines; dim_red.generate is now only
imported when structures actually have to be generated.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Alias `TailTrainConfig` e costanti con fonte unica

**Files:**
- Modify: `src/dim_red/supcon/training.py` (docstring di `TrainConfig`)
- Modify: `src/dim_red/supcon/tail_training.py`
- Test: `tests/test_supcon_tail_training.py`, `tests/test_supcon_training.py`

**Interfaces:**
- Consumes: `supcon.training.TrainConfig`, `_BATCHING_STRATEGIES`, `_DISTANCE_METRICS`.
- Produces: `supcon.tail_training.TailTrainConfig is supcon.training.TrainConfig`; `supcon.tail_training._BATCHING_STRATEGIES is supcon.training._BATCHING_STRATEGIES` (e lo stesso per `_DISTANCE_METRICS`).

- [ ] **Step 1: Scrivere i test rossi**

In `tests/test_supcon_tail_training.py` aggiungere in fondo:

```python
def test_tail_train_config_is_the_phase1_train_config():
    import dataclasses

    from dim_red.supcon.training import TrainConfig

    assert TailTrainConfig is TrainConfig
    replaced = dataclasses.replace(TailTrainConfig(), tau=0.2, epochs=3)
    assert isinstance(replaced, TrainConfig)
    assert (replaced.tau, replaced.epochs) == (0.2, 3)
```

In `tests/test_supcon_training.py` aggiungere in fondo:

```python
def test_batching_and_distance_constants_have_a_single_source():
    from dim_red.pipeline import config
    from dim_red.supcon import tail_training, training

    assert tail_training._BATCHING_STRATEGIES is training._BATCHING_STRATEGIES
    assert tail_training._DISTANCE_METRICS is training._DISTANCE_METRICS
    # pipeline.config must stay importable without jax, so it keeps its own
    # copy; make sure it can't drift.
    assert config._BATCHING_STRATEGIES == training._BATCHING_STRATEGIES
```

Run: `pytest tests/test_supcon_tail_training.py::test_tail_train_config_is_the_phase1_train_config tests/test_supcon_training.py::test_batching_and_distance_constants_have_a_single_source -v`
Expected: 2 FAIL (`TailTrainConfig is TrainConfig` falso; le tuple non sono lo stesso oggetto).

- [ ] **Step 2: Modificare `supcon/tail_training.py`**

(a) Rimuovere la riga `from dataclasses import dataclass`.

(b) Sostituire `from dim_red.supcon.training import norm_penalty, supcon_loss` con:

```python
from dim_red.supcon.training import (
    _BATCHING_STRATEGIES,
    _DISTANCE_METRICS,
    TrainConfig,
    norm_penalty,
    supcon_loss,
)
```

(c) Eliminare le due righe `_BATCHING_STRATEGIES = ...` e `_DISTANCE_METRICS = ...`, e tutta la classe `@dataclass(frozen=True) class TailTrainConfig: ...` (dal decoratore fino alla riga `early_stopping_restore_best: bool = True` inclusa). Al suo posto:

```python
# Phase 2 trains with exactly the same loop settings as phase 1 (same fields
# and defaults), so the config is one class. ``tau``/``distance`` only matter
# for the SupCon loss (``train_visualization_tail``); ``train_classification_tail``
# ignores them.
TailTrainConfig = TrainConfig
```

- [ ] **Step 3: Aggiornare il docstring di `TrainConfig`**

In `src/dim_red/supcon/training.py` sostituire la prima riga del docstring

```
    """Training configuration for ``SupConEncoder`` optimization.
```

con

```
    """Training-loop configuration for SupCon, shared by phase 1
    (``training_first_phase``) and phase 2 (``train_classification_tail``/
    ``train_visualization_tail`` -- exported there as
    ``dim_red.supcon.tail_training.TailTrainConfig``).
```

e aggiungere, prima di `Attributes:`, la riga: `Phase 2's classification tail ignores ``tau``/``distance`` (no SupCon loss).`

- [ ] **Step 4: Eseguire i test**

Run, uno alla volta:
`pytest tests/test_supcon_tail_training.py -q` → tutti passano (i due nuovi inclusi nei rispettivi file).
`pytest tests/test_supcon_training.py -q` → tutti passano.
`pytest tests/test_pipeline_tail_config.py -q` → tutti passano.

- [ ] **Step 5: Commit**

```bash
git add -u src tests
git commit -m "Make TailTrainConfig an alias of TrainConfig; one source for batching/distance constants

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Chiavi di history `train_ce` / `val_ce` per il tail di classificazione

**Files:**
- Modify: `src/dim_red/supcon/tail_training.py`
- Modify: `src/dim_red/pipeline/compare.py:231-238` (`_LOSS_METRIC_ORDER`)
- Test: `tests/test_supcon_tail_training.py`, `tests/test_pipeline_tail_training.py`, `tests/test_pipeline_compare.py`

**Interfaces:**
- Consumes: Task 3 (nessuna dipendenza funzionale, ma lo stesso file).
- Produces: `train_classification_tail` restituisce una history con le chiavi `train_loss`, `val_loss`, `train_ce`, `val_ce`.

- [ ] **Step 1: Aggiornare i test (rossi)**

`tests/test_supcon_tail_training.py`, righe 63-66 circa: sostituire `train_family_ce`/`val_family_ce` con `train_ce`/`val_ce` nel ciclo e nel `zip`:

```python
    for key in ("train_loss", "val_loss", "train_ce", "val_ce"):
```

e

```python
    for total, ce in zip(history["train_loss"], history["train_ce"]):
```

`tests/test_pipeline_tail_training.py`, righe 125-132: l'`assert header == [...]` diventa

```python
    assert header == [
        "epoch",
        "train_loss",
        "val_loss",
        "train_ce",
        "val_ce",
    ]
```

In `tests/test_pipeline_compare.py` aggiungere in fondo (Review Focus 6):

```python
def test_available_loss_metrics_orders_new_and_legacy_cross_entropy_keys():
    from types import SimpleNamespace

    run = SimpleNamespace(
        loss_history={
            "val_family_ce": [1.0],
            "val_ce": [1.0],
            "train_ce": [1.0],
            "val_loss": [1.0],
            "train_family_ce": [1.0],
            "train_loss": [1.0],
        }
    )
    assert available_loss_metrics([run]) == [
        "train_loss",
        "val_loss",
        "train_ce",
        "val_ce",
        "train_family_ce",
        "val_family_ce",
    ]
```

Run: `pytest tests/test_supcon_tail_training.py tests/test_pipeline_compare.py::test_available_loss_metrics_orders_new_and_legacy_cross_entropy_keys -q`
Expected: FAIL (chiavi non ancora rinominate; ordine non ancora previsto).

- [ ] **Step 2: Rinominare le chiavi in `supcon/tail_training.py`**

Nel dizionario `history` di `train_classification_tail`:

```python
        "train_ce": [],
        "val_ce": [],
```

e nel ciclo delle epoche:

```python
        history["train_ce"].append(train_loss)
        history["val_ce"].append(val_loss)
```

Nel docstring `Returns:` sostituire il blocco sulle chiavi con:

```
        training early): ``"train_loss"``/``"val_loss"`` (the cross-entropy)
        and ``"train_ce"``/``"val_ce"`` (the same value, kept as separate
        columns of the saved ``loss_history.csv``; older runs wrote
        ``train_family_ce``/``val_family_ce`` here).
```

- [ ] **Step 3: Aggiornare `compare._LOSS_METRIC_ORDER`**

```python
_LOSS_METRIC_ORDER = [
    "train_loss",
    "val_loss",
    "train_ce",
    "val_ce",
    "train_family_ce",
    "val_family_ce",
    "train_spacegroup_ce",
    "val_spacegroup_ce",
]
```

- [ ] **Step 4: Eseguire i test**

Run, uno alla volta:
`pytest tests/test_supcon_tail_training.py -q`
`pytest tests/test_pipeline_compare.py -q`
`pytest tests/test_pipeline_tail_training.py -q` (lento: ~2 minuti)
Expected: tutti passano.

- [ ] **Step 5: Commit**

```bash
git add -u src tests
git commit -m "Rename classification-tail history keys to train_ce/val_ce

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Documentazione, parità end-to-end e chiusura

**Files:**
- Modify: `refactor/A-dedup-design.md`, `REFACTOR_TODO.md`, `src/dim_red/supcon/CLAUDE.md`, `src/dim_red/pipeline/CLAUDE.md`

- [ ] **Step 1: Aggiornare `src/dim_red/supcon/CLAUDE.md`**

Nella riga di `tail_training.py` sostituire `` `TailTrainConfig` (same fields as `TrainConfig`; unification is in `REFACTOR_TODO.md`) `` con `` `TailTrainConfig` (an alias of `training.TrainConfig`) `` e `` history keys `train_family_ce`/`val_family_ce` kept for compat `` con `` history keys `train_ce`/`val_ce` ``.

- [ ] **Step 2: Aggiornare `REFACTOR_TODO.md`**

Nel §3 eliminare: l'elemento su `TrainConfig`/`TailTrainConfig` gemelle, l'elemento sulle chiavi di history `train_family_ce`/`val_family_ce`, l'elemento su `_BATCHING_STRATEGIES`/`_DISTANCE_METRICS`; nel §5b l'elemento sulle sei `get_or_build_*`; nel §5 la prima voce (meccanismo unico di chiavi deprecate: già `_DEPRECATED_KEYS`). Aggiungere in testa al file la riga: `Refactor A (dedup sicure) completato: vedi refactor/A-dedup-design.md.`

- [ ] **Step 3: Build della documentazione**

Run: `sphinx-build -b html -W --keep-going docs /tmp/docs_html_check`
Expected: exit 0, nessun warning.

- [ ] **Step 4: Parità end-to-end prima/dopo**

Creare `/tmp/cmp_a/cfg.yaml` con il contenuto seguente (`output_dir` verrà impostato per lato):

```yaml
seed: 42
output_dir: OUTDIR
name: cmp
model: supcon
data_source: pyxtal
pyxtal: {families: [cubic, hexagonal, tetragonal], structures_per_family: 40, distribution: uniform, n_species: 2, seed: 42}
soap: {r_cut: 5.0, n_max: 4, l_max: 3, sigma: 0.5, element_agnostic: false, species: null}
augmentation: {n_augmented: 1, keep_original: true, jitter_probability: 1.0, jitter_std: 0.05, seed: 7}
encoder: {encoder_hidden_dim: [64, 32], latent_dim: 8}
supcon: {mode: family_only, lambda_family: 1.0, tau: 0.05, distance: cosine, projection_dim: 32}
train: {epochs: 15, batch_size: 32, learning_rate: 0.001, val_ratio: 0.2, device: cpu}
tails:
  classification: {head_hidden_dim: 16}
  visualization: {viz_dim: 2, mode: family_only, hidden_dim: [16], tau: 0.1, distance: euclidean}
  train: {epochs: 15, batch_size: 32, learning_rate: 0.001, device: cpu}
```

Eseguire il lato "prima" su un worktree del commit precedente al refactor (`e3f8218`) e il lato "dopo" sul working tree, uno dopo l'altro (non in parallelo):

```bash
mkdir -p /tmp/cmp_a
git worktree add /tmp/cmp_a/before e3f8218
for side in before after; do
  sed "s#OUTDIR#/tmp/cmp_a/out_$side#" /tmp/cmp_a/cfg.yaml > /tmp/cmp_a/cfg_$side.yaml
  if [ $side = before ]; then ROOT=/tmp/cmp_a/before; else ROOT=$PWD; fi
  (cd $ROOT && PYTHONPATH=$ROOT/src python -c "
from dim_red.pipeline.cli import run_command
run_command(['/tmp/cmp_a/cfg_$side.yaml'])")
done
git worktree remove --force /tmp/cmp_a/before && git worktree prune
```

Confrontare:

```bash
python - <<'PY'
import hashlib, pathlib
import numpy as np, pandas as pd
b, a = pathlib.Path("/tmp/cmp_a/out_before/cmp"), pathlib.Path("/tmp/cmp_a/out_after/cmp")
md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
assert md5(b / "dataset.extxyz") == md5(a / "dataset.extxyz")
x, y = pd.read_csv(b / "loss_history.csv"), pd.read_csv(a / "loss_history.csv")
assert np.abs(x.values - y.values).max() == 0.0
eb, ea = np.load(b / "embeddings.npz", allow_pickle=True), np.load(a / "embeddings.npz", allow_pickle=True)
for k in eb.files:
    if eb[k].dtype.kind in "fi":
        assert np.abs(eb[k].astype(float) - ea[k].astype(float)).max() == 0.0, k
vb = np.load(b / "tails/visualization/tail_embeddings.npz", allow_pickle=True)["embeddings"]
va = np.load(a / "tails/visualization/tail_embeddings.npz", allow_pickle=True)["embeddings"]
assert np.abs(vb - va).max() == 0.0
cb = pd.read_csv(b / "tails/classification/loss_history.csv")
ca = pd.read_csv(a / "tails/classification/loss_history.csv")
assert list(cb.columns) == ["epoch", "train_loss", "val_loss", "train_family_ce", "val_family_ce"]
assert list(ca.columns) == ["epoch", "train_loss", "val_loss", "train_ce", "val_ce"]
assert np.abs(cb.values - ca.values).max() == 0.0
print("PARITY OK")
PY
```

Expected: `PARITY OK`. (A differenza del confronto con `master` fatto in precedenza, qui anche il tail di classificazione deve coincidere: la sua inizializzazione non è stata toccata.)

- [ ] **Step 5: Verifica finale e commit**

Run: `git status --short` (solo i file di documentazione modificati) e poi:

```bash
git add -u
git commit -m "Document refactor A: update CLAUDE.md and REFACTOR_TODO

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Aggiornare la memoria**

Aggiornare `project_repo_prune_2026_10.md` (cartella memoria del progetto) con una riga: refactor A completato, con il commit finale, e aggiornare il file di piano `~/.claude/plans/vorrei-che-passassimo-in-joyful-elephant.md` con lo stato.

---

## Self-review

- **Copertura della spec:** A1 → Task 3; A2 → Task 3; A3 → Task 1+2; A4 → Task 4; verifica 1 → Task 1; 2 → Task 3 Step 1; 3 → Task 4; 4 → test per file toccato in ogni task; 5 → Task 5 Step 4; 6 → Task 5 Step 3.
- **Placeholder:** nessuno; ogni blocco di codice è completo ed è stato eseguito su un prototipo (le funzioni dei Task 1 e 2 con 91 test verdi tra `characterization`, `dataset_cache`, `single_run`, `inference`).
- **Coerenza dei tipi:** `_Featurizer`/`_StructureSource`/`_get_or_build` hanno lo stesso nome e la stessa firma nel blocco CORE e nei wrapper.
- **Rischio residuo:** gli `Expected` con numeri di test passati (36, 24, 19) valgono per lo stato di `master` a `e3f8218`; se nel frattempo il numero cambia, conta che non ci siano fallimenti.
