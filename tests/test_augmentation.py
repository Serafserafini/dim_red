"""
Unit tests for structure data augmentation (positional jitter and vacancies).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from dim_red.augmentation import (
    AugmentationConfig,
    augment_structures,
    fetch_and_augment_structures,
    jitter_positions,
    remove_random_atoms,
)


def _make_atoms(n=6, material_id="mp-1"):
    positions = [[float(i), 0.0, 0.0] for i in range(n)]
    atoms = Atoms("H" * n, positions=positions)
    atoms.info["material_id"] = material_id
    return atoms


def test_jitter_positions_zero_std_leaves_positions_unchanged():
    atoms = _make_atoms()
    rng = np.random.default_rng(0)
    jittered = jitter_positions(atoms, std=0.0, rng=rng)
    np.testing.assert_array_equal(jittered.get_positions(), atoms.get_positions())
    assert jittered is not atoms


def test_jitter_positions_perturbs_by_expected_magnitude():
    atoms = _make_atoms(n=200)
    rng = np.random.default_rng(0)
    jittered = jitter_positions(atoms, std=0.1, rng=rng)
    diffs = jittered.get_positions() - atoms.get_positions()
    assert not np.allclose(diffs, 0.0)
    # Std of a large sample of Gaussian noise should be close to the configured std.
    assert 0.05 < np.std(diffs) < 0.2


def test_remove_random_atoms_respects_min_atoms():
    atoms = _make_atoms(n=10)
    rng = np.random.default_rng(0)
    reduced = remove_random_atoms(atoms, atom_probability=1.0, rng=rng, min_atoms=3)
    assert len(reduced) == 3
    assert reduced.info["material_id"] == "mp-1"


def test_remove_random_atoms_no_op_when_probability_zero():
    atoms = _make_atoms(n=10)
    rng = np.random.default_rng(0)
    reduced = remove_random_atoms(atoms, atom_probability=0.0, rng=rng, min_atoms=1)
    assert len(reduced) == len(atoms)


def test_remove_random_atoms_no_op_when_at_or_below_min_atoms():
    atoms = _make_atoms(n=2)
    rng = np.random.default_rng(0)
    reduced = remove_random_atoms(atoms, atom_probability=1.0, rng=rng, min_atoms=2)
    assert len(reduced) == 2


def test_augmentation_config_validates_probabilities():
    with pytest.raises(ValueError):
        AugmentationConfig(jitter_probability=1.5)
    with pytest.raises(ValueError):
        AugmentationConfig(vacancy_probability=-0.1)
    with pytest.raises(ValueError):
        AugmentationConfig(jitter_std=-1.0)
    with pytest.raises(ValueError):
        AugmentationConfig(min_atoms=0)
    with pytest.raises(ValueError):
        AugmentationConfig(n_augmented=-1)


def test_augment_structures_keeps_original_and_generates_copies():
    atoms_list = [
        _make_atoms(n=10, material_id="mp-1"),
        _make_atoms(n=10, material_id="mp-2"),
    ]
    config = AugmentationConfig(
        n_augmented=3,
        keep_original=True,
        jitter_probability=1.0,
        jitter_std=0.05,
        vacancy_probability=0.0,
        seed=42,
    )
    result = augment_structures(atoms_list, config)

    assert len(result) == len(atoms_list) * (1 + config.n_augmented)

    originals = [a for a in result if not a.info["augmented"]]
    augmented = [a for a in result if a.info["augmented"]]
    assert len(originals) == 2
    assert len(augmented) == 6
    for a in originals:
        assert a.info["augmentations_applied"] == []
    for a in augmented:
        assert a.info["augmentations_applied"] == ["jitter"]
        assert a.info["source_material_id"] in ("mp-1", "mp-2")


def test_augment_structures_without_keeping_original():
    atoms_list = [_make_atoms(n=10)]
    config = AugmentationConfig(n_augmented=2, keep_original=False, seed=1)
    result = augment_structures(atoms_list, config)
    assert len(result) == 2


def test_augment_structures_jitter_probability_zero_never_applies_jitter():
    atoms_list = [_make_atoms(n=10)]
    config = AugmentationConfig(
        n_augmented=20,
        keep_original=False,
        jitter_probability=0.0,
        jitter_std=0.1,
        vacancy_probability=0.0,
        seed=7,
    )
    result = augment_structures(atoms_list, config)
    assert all(a.info["augmentations_applied"] == [] for a in result)
    for a in result:
        np.testing.assert_array_equal(a.get_positions(), atoms_list[0].get_positions())


def test_augment_structures_vacancy_probability_one_removes_atoms():
    atoms_list = [_make_atoms(n=20)]
    config = AugmentationConfig(
        n_augmented=5,
        keep_original=False,
        jitter_probability=0.0,
        vacancy_probability=1.0,
        vacancy_atom_probability=0.5,
        min_atoms=5,
        seed=3,
    )
    result = augment_structures(atoms_list, config)
    assert all(a.info["augmentations_applied"] == ["vacancy"] for a in result)
    assert all(5 <= len(a) < 20 for a in result)


def test_augment_structures_is_reproducible_with_same_seed():
    atoms_list = [_make_atoms(n=10)]
    config = AugmentationConfig(
        n_augmented=5,
        jitter_probability=0.5,
        vacancy_probability=0.5,
        vacancy_atom_probability=0.3,
        seed=123,
    )
    result_a = augment_structures(atoms_list, config)
    result_b = augment_structures(atoms_list, config)

    assert [a.info["augmentations_applied"] for a in result_a] == [
        b.info["augmentations_applied"] for b in result_b
    ]
    for a, b in zip(result_a, result_b):
        np.testing.assert_array_equal(a.get_positions(), b.get_positions())


def test_augment_structures_does_not_mutate_input():
    atoms_list = [_make_atoms(n=10)]
    original_positions = atoms_list[0].get_positions().copy()
    config = AugmentationConfig(
        n_augmented=5,
        jitter_probability=1.0,
        jitter_std=0.2,
        vacancy_probability=1.0,
        vacancy_atom_probability=0.5,
        min_atoms=3,
        seed=9,
    )
    augment_structures(atoms_list, config)
    np.testing.assert_array_equal(atoms_list[0].get_positions(), original_positions)


@patch("dim_red.augmentation.fetch_structures_by_crystal_system")
def test_fetch_and_augment_structures_chains_fetch_and_augment(mock_fetch):
    mock_fetch.return_value = [_make_atoms(n=10, material_id="mp-1")]
    config = AugmentationConfig(n_augmented=2, keep_original=True, seed=5)

    result = fetch_and_augment_structures(
        "cubic", augmentation_config=config, api_key="dummy", limit=1
    )

    mock_fetch.assert_called_once_with(crystal_system="cubic", api_key="dummy", limit=1)
    assert len(result) == 3
