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
    "soap_fetch": (
        dc._cache_key,
        (["cubic"], 2, SOAP_KW),
        "3ffce6cf603d9752",
        "ea9f81daed72cf06",
    ),
    "soap_pyxtal": (
        dc._pyxtal_cache_key,
        (PYXTAL, 0, SOAP_KW),
        "pyxtal-31fd2343feb45e6c",
        "pyxtal-b2a947f6a4f60ea9",
    ),
    "graph_fetch": (
        dc._graph_cache_key,
        (["cubic"], 2, GRAPH_KW),
        "cgcnn-69b0745d9560b06d",
        "cgcnn-ffb0ff20517873c9",
    ),
    "graph_pyxtal": (
        dc._pyxtal_graph_cache_key,
        (PYXTAL, 0, GRAPH_KW),
        "cgcnn-pyxtal-e72d0d941da0384d",
        "cgcnn-pyxtal-63d7ed469d4b7650",
    ),
    "mace_fetch": (
        dc._mace_cache_key,
        (["cubic"], 2, MACE_KW),
        "mace-22b2e5b06a39e187",
        "mace-3d2614f589d1cc69",
    ),
    "mace_pyxtal": (
        dc._pyxtal_mace_cache_key,
        (PYXTAL, 0, MACE_KW),
        "mace-pyxtal-02c79b69134102bd",
        "mace-pyxtal-3204c36da14dfa9f",
    ),
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
SOAP_FILE_KEYS = [
    "X",
    "labels",
    "material_ids",
    "spacegroups",
    "feature_mean",
    "feature_std",
]
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
    assert any(
        m.startswith(f"Dataset cache miss ({key}); fetching structures") for m in msgs
    )
    assert any(
        m.startswith(f"Dataset cache hit ({key}) for crystal_systems=") for m in msgs
    )


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
    assert any(
        m.startswith(f"Dataset cache miss ({key}); generating structures with pyxtal")
        for m in msgs
    )
    assert any(m == f"Dataset cache hit ({key}) for pyxtal generation" for m in msgs)


def test_graph_fetch(tmp_path, caplog):
    with patch.object(
        dc, "fetch_structures_by_crystal_system", side_effect=lambda **kw: FETCHED()
    ) as fetch:
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
    assert any(
        m.startswith(f"CGCNN dataset cache miss ({key}); fetching structures")
        for m in msgs
    )
    assert any(
        m.startswith(f"CGCNN dataset cache hit ({key}) for crystal_systems=")
        for m in msgs
    )


def test_graph_pyxtal(tmp_path, caplog):
    with patch(
        "dim_red.generate.generate_structures", side_effect=lambda cfg: GENERATED()
    ) as gen:
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
    assert any(
        m.startswith(f"MACE dataset cache miss ({key}); fetching structures")
        for m in msgs
    )
    assert any(
        m.startswith(f"MACE dataset cache hit ({key}) for crystal_systems=")
        for m in msgs
    )


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
    with patch.object(
        dc, "fetch_structures_by_crystal_system", side_effect=lambda **kw: FETCHED()
    ) as fetch:
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
        kwargs = dict(
            pyxtal_config=PYXTAL, seed=0, mace_kwargs=MACE_KW, cache_dir=tmp_path
        )
        dc.get_or_build_pyxtal_mace_dataset(**kwargs)
        key = dc._pyxtal_mace_cache_key(PYXTAL, 0, MACE_KW)
        np.savez(tmp_path / f"{key}.npz", X=FAKE_X)  # drops the other fields
        dc.get_or_build_pyxtal_mace_dataset(**kwargs)
    assert gen.call_count == 2
