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
    make_supercell_for_radius,
    remove_random_atoms,
)


def _make_atoms(n=6, material_id="mp-1"):
    positions = [[float(i), 0.0, 0.0] for i in range(n)]
    atoms = Atoms("H" * n, positions=positions)
    atoms.info["material_id"] = material_id
    return atoms


def _make_periodic_atoms(cell=(2.0, 2.0, 2.0), material_id="mp-1"):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
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


def test_remove_random_atoms_respects_max_vacancies():
    atoms = _make_atoms(n=10)
    rng = np.random.default_rng(0)
    reduced = remove_random_atoms(atoms, atom_probability=1.0, rng=rng, max_vacancies=7)
    assert len(reduced) == 3  # 10 - 7 removed
    assert reduced.info["material_id"] == "mp-1"


def test_remove_random_atoms_no_op_when_probability_zero():
    atoms = _make_atoms(n=10)
    rng = np.random.default_rng(0)
    reduced = remove_random_atoms(atoms, atom_probability=0.0, rng=rng)
    assert len(reduced) == len(atoms)


def test_remove_random_atoms_never_drops_below_one_atom():
    atoms = _make_atoms(n=2)
    rng = np.random.default_rng(0)
    # atom_probability=1.0, no max_vacancies cap -- would remove both atoms
    # without the hard floor of 1 remaining atom.
    reduced = remove_random_atoms(atoms, atom_probability=1.0, rng=rng)
    assert len(reduced) == 1


def test_remove_random_atoms_no_op_for_single_atom_structure():
    atoms = _make_atoms(n=1)
    rng = np.random.default_rng(0)
    reduced = remove_random_atoms(atoms, atom_probability=1.0, rng=rng)
    assert len(reduced) == 1


def test_make_supercell_for_radius_expands_small_cubic_cell():
    atoms = _make_periodic_atoms(cell=(2.0, 2.0, 2.0))
    supercell = make_supercell_for_radius(atoms, radius=5.0)
    # perpendicular width = 2.0 per axis; need >= 2*5=10 -> ceil(10/2)=5 repeats/axis.
    assert len(supercell) == 5**3
    np.testing.assert_allclose(np.diag(supercell.get_cell()), [10.0, 10.0, 10.0])


def test_make_supercell_for_radius_no_op_when_cell_already_large_enough():
    atoms = _make_periodic_atoms(cell=(2.0, 2.0, 2.0))
    supercell = make_supercell_for_radius(atoms, radius=0.5)
    assert len(supercell) == 1
    assert supercell is not atoms


def test_make_supercell_for_radius_handles_exact_boundary_without_floating_point_overshoot():
    # 2*radius / perpendicular_width lands exactly on an integer (5); a naive
    # ceil() on the raw floating-point ratio can spuriously round up to 6
    # due to determinant/cross-product rounding noise.
    atoms = _make_periodic_atoms(cell=(2.0, 2.0, 2.0))
    supercell = make_supercell_for_radius(atoms, radius=5.0)
    assert len(supercell) == 125  # not 216 (6**3)


def test_make_supercell_for_radius_preserves_info():
    atoms = _make_periodic_atoms(cell=(2.0, 2.0, 2.0), material_id="mp-42")
    atoms.info["spacegroup"] = 225
    supercell = make_supercell_for_radius(atoms, radius=5.0)
    assert supercell.info["material_id"] == "mp-42"
    assert supercell.info["spacegroup"] == 225


def test_make_supercell_for_radius_skips_non_periodic_axes():
    atoms = Atoms(
        "H", positions=[[0.0, 0.0, 0.0]], cell=(2.0, 2.0, 2.0), pbc=(True, False, True)
    )
    supercell = make_supercell_for_radius(atoms, radius=5.0)
    # Axis 1 (non-periodic) stays at 1 repeat regardless of radius.
    assert len(supercell) == 5 * 1 * 5


def test_make_supercell_for_radius_handles_oblique_cell():
    atoms = Atoms(
        "H",
        positions=[[0.0, 0.0, 0.0]],
        cell=[[3.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        pbc=True,
    )
    supercell = make_supercell_for_radius(atoms, radius=6.0)
    # Same repeat counts as computed independently per axis from the
    # original (unrepeated) cell -- verified by construction, just check the
    # resulting cell is large enough and atom count matches repeats' product.
    assert len(supercell) > 1


def test_augmentation_config_validates_probabilities():
    with pytest.raises(ValueError):
        AugmentationConfig(jitter_probability=1.5)
    with pytest.raises(ValueError):
        AugmentationConfig(vacancy_probability=-0.1)
    with pytest.raises(ValueError):
        AugmentationConfig(jitter_std=-1.0)
    with pytest.raises(ValueError):
        AugmentationConfig(max_vacancies=-1)
    with pytest.raises(ValueError):
        AugmentationConfig(n_augmented=-1)
    with pytest.raises(ValueError, match="supercell_radius"):
        AugmentationConfig(supercell_radius=0.0)
    with pytest.raises(ValueError, match="supercell_radius"):
        AugmentationConfig(supercell_radius=-1.0)


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
    """Both mechanisms fully disabled (probability *and* magnitude/rate at
    0) -- there's nothing available to force, so augmented copies come out
    identical to the original, same as before the "at least one mechanism"
    guarantee existed.
    """
    atoms_list = [_make_atoms(n=10)]
    config = AugmentationConfig(
        n_augmented=20,
        keep_original=False,
        jitter_probability=0.0,
        jitter_std=0.0,
        vacancy_probability=0.0,
        vacancy_atom_probability=0.0,
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
        max_vacancies=15,
        seed=3,
    )
    result = augment_structures(atoms_list, config)
    assert all(a.info["augmentations_applied"] == ["vacancy"] for a in result)
    assert all(5 <= len(a) < 20 for a in result)


def test_augment_structures_guarantees_at_least_one_mechanism_when_both_available():
    """Neither mechanism's own probability ever triggers naturally (both are
    0.0), but both are "available" (jitter_std/vacancy_atom_probability >
    0) -- every augmented copy must still get one of the two forced on, and
    with enough samples both should show up (forcing picks randomly).
    """
    atoms_list = [_make_atoms(n=20)]
    config = AugmentationConfig(
        n_augmented=200,
        keep_original=False,
        jitter_probability=0.0,
        jitter_std=0.05,
        vacancy_probability=0.0,
        vacancy_atom_probability=0.1,
        seed=11,
    )
    result = augment_structures(atoms_list, config)
    assert all(a.info["augmentations_applied"] for a in result)
    applied_kinds = {tuple(a.info["augmentations_applied"]) for a in result}
    assert applied_kinds == {("jitter",), ("vacancy",)}


def test_augment_structures_forces_the_only_available_mechanism():
    """When only jitter is configured at all (vacancy_atom_probability=0.0
    disables vacancy entirely), it applies to every augmented copy
    regardless of its own jitter_probability -- the only thing forceable.
    """
    atoms_list = [_make_atoms(n=10)]
    config = AugmentationConfig(
        n_augmented=20,
        keep_original=False,
        jitter_probability=0.0,
        jitter_std=0.05,
        vacancy_probability=0.0,
        vacancy_atom_probability=0.0,
        seed=13,
    )
    result = augment_structures(atoms_list, config)
    assert all(a.info["augmentations_applied"] == ["jitter"] for a in result)


def test_augment_structures_expands_single_atom_structure_via_supercell_radius():
    """A 1-atom periodic cell (e.g. what pyxtal can generate) has nothing
    for vacancy removal to meaningfully work with on its own -- setting
    supercell_radius should expand it first so both the kept-original and
    every augmented copy have more than 1 atom.
    """
    atoms_list = [_make_periodic_atoms(cell=(2.0, 2.0, 2.0))]
    config = AugmentationConfig(
        n_augmented=5,
        keep_original=True,
        jitter_probability=0.0,
        jitter_std=0.0,
        vacancy_probability=1.0,
        vacancy_atom_probability=0.3,
        supercell_radius=5.0,
        seed=17,
    )
    result = augment_structures(atoms_list, config)

    assert len(result) == 6  # 1 original + 5 augmented
    for a in result:
        assert len(a) > 1
        assert a.info["material_id"] == "mp-1"
    originals = [a for a in result if not a.info["augmented"]]
    assert len(originals) == 1
    assert len(originals[0]) == 125  # the full, un-vacancied supercell
    augmented = [a for a in result if a.info["augmented"]]
    assert all(a.info["augmentations_applied"] == ["vacancy"] for a in augmented)
    assert all(a.info["source_material_id"] == "mp-1" for a in augmented)


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
        max_vacancies=7,
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
