"""
Unit tests for pyxtal-based synthetic structure generation (the alternative
to fetching from Materials Project).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("pyxtal")

from dim_red.generate import (
    GenerationConfig,
    _family_for_spacegroup,
    _pick_species,
    _resolve_counts,
    _target_spacegroups,
    generate_structures,
)

# --- GenerationConfig validation ---------------------------------------------


def test_config_requires_exactly_one_of_per_spacegroup_or_per_family():
    with pytest.raises(ValueError, match="exactly one"):
        GenerationConfig()
    with pytest.raises(ValueError, match="exactly one"):
        GenerationConfig(structures_per_spacegroup=1, structures_per_family=1)


def test_config_rejects_both_families_and_spacegroups():
    with pytest.raises(ValueError, match="only one"):
        GenerationConfig(
            structures_per_spacegroup=1, families=["Cubic"], spacegroups=[225]
        )


def test_config_rejects_unknown_family():
    with pytest.raises(ValueError, match="Unknown crystal families"):
        GenerationConfig(structures_per_spacegroup=1, families=["Nonagonal"])


def test_config_normalizes_family_casing():
    config = GenerationConfig(structures_per_spacegroup=1, families=["cubic"])
    assert config.families == ["Cubic"]


def test_config_rejects_out_of_range_spacegroup():
    with pytest.raises(ValueError, match=r"\[1, 230\]"):
        GenerationConfig(structures_per_spacegroup=1, spacegroups=[0, 300])


def test_config_rejects_invalid_distribution():
    with pytest.raises(ValueError, match="distribution"):
        GenerationConfig(structures_per_family=1, distribution="gaussian")


def test_config_rejects_non_positive_n_species():
    with pytest.raises(ValueError, match="n_species"):
        GenerationConfig(structures_per_spacegroup=1, n_species=0)


def test_config_rejects_n_species_larger_than_pool():
    with pytest.raises(ValueError, match="exceeds species_pool size"):
        GenerationConfig(
            structures_per_spacegroup=1, n_species=3, species_pool=["C", "N"]
        )


def test_config_rejects_empty_candidate_num_ions():
    with pytest.raises(ValueError, match="candidate_num_ions"):
        GenerationConfig(structures_per_spacegroup=1, candidate_num_ions=[])


# --- internal helpers ---------------------------------------------------------


def test_family_for_spacegroup_matches_known_values():
    assert _family_for_spacegroup(1) == "Triclinic"
    assert _family_for_spacegroup(225) == "Cubic"


def test_target_spacegroups_from_explicit_spacegroups():
    config = GenerationConfig(structures_per_spacegroup=1, spacegroups=[225, 196, 1])
    by_family = _target_spacegroups(config)
    assert by_family == {"Cubic": [196, 225], "Triclinic": [1]}


def test_target_spacegroups_from_families_filter():
    config = GenerationConfig(structures_per_spacegroup=1, families=["Triclinic"])
    by_family = _target_spacegroups(config)
    assert by_family == {"Triclinic": [1, 2]}


def test_target_spacegroups_defaults_to_all_families():
    config = GenerationConfig(structures_per_spacegroup=1)
    by_family = _target_spacegroups(config)
    assert set(by_family) == {
        "Triclinic",
        "Monoclinic",
        "Orthorhombic",
        "Tetragonal",
        "Trigonal",
        "Hexagonal",
        "Cubic",
    }
    assert sum(len(sgs) for sgs in by_family.values()) == 230


def test_resolve_counts_per_spacegroup_mode_applies_uniformly():
    config = GenerationConfig(structures_per_spacegroup=3, spacegroups=[1, 2, 225])
    by_family = _target_spacegroups(config)
    counts = _resolve_counts(config, by_family, np.random.default_rng(0))
    assert counts == {1: 3, 2: 3, 225: 3}


def test_resolve_counts_uniform_distribution_splits_as_evenly_as_possible():
    config = GenerationConfig(
        structures_per_family=5, families=["Triclinic"], distribution="uniform"
    )
    by_family = _target_spacegroups(config)  # {"Triclinic": [1, 2]}
    counts = _resolve_counts(config, by_family, np.random.default_rng(0))
    assert sum(counts.values()) == 5
    assert set(counts) == {1, 2}
    # divmod(5, 2) = (2, 1): the first (sorted) spacegroup gets the remainder.
    assert counts[1] == 3
    assert counts[2] == 2


def test_resolve_counts_random_distribution_sums_to_requested_and_is_reproducible():
    config = GenerationConfig(
        structures_per_family=20, families=["Triclinic"], distribution="random"
    )
    by_family = _target_spacegroups(config)
    counts_a = _resolve_counts(config, by_family, np.random.default_rng(0))
    counts_b = _resolve_counts(config, by_family, np.random.default_rng(0))
    assert sum(counts_a.values()) == 20
    assert set(counts_a) == {1, 2}
    assert counts_a == counts_b  # same seed -> same split


def test_pick_species_returns_distinct_elements_from_pool():
    rng = np.random.default_rng(0)
    species = _pick_species(3, ["C", "N", "O", "Si", "Fe"], rng)
    assert len(species) == 3
    assert len(set(species)) == 3
    assert set(species) <= {"C", "N", "O", "Si", "Fe"}


# --- generate_structures (real pyxtal calls, scoped small for speed) ----------


def test_generate_structures_per_spacegroup_mode():
    config = GenerationConfig(
        spacegroups=[225, 1], structures_per_spacegroup=2, n_species=1, seed=0
    )
    atoms_list = generate_structures(config)

    assert len(atoms_list) == 4
    spacegroups = {a.info["spacegroup"] for a in atoms_list}
    assert spacegroups == {225, 1}
    for a in atoms_list:
        expected_family = "Cubic" if a.info["spacegroup"] == 225 else "Triclinic"
        assert a.info["family"] == expected_family
    material_ids = [a.info["material_id"] for a in atoms_list]
    assert len(set(material_ids)) == len(material_ids)  # all unique


def test_generate_structures_uses_n_species_distinct_elements():
    config = GenerationConfig(
        spacegroups=[225], structures_per_spacegroup=3, n_species=2, seed=1
    )
    atoms_list = generate_structures(config)

    assert len(atoms_list) == 3
    for a in atoms_list:
        assert len(set(a.get_chemical_symbols())) == 2


def test_generate_structures_is_reproducible_with_same_seed():
    config = GenerationConfig(
        spacegroups=[225, 1], structures_per_spacegroup=2, n_species=1, seed=42
    )
    atoms_a = generate_structures(config)
    atoms_b = generate_structures(config)

    assert [a.get_chemical_symbols() for a in atoms_a] == [
        b.get_chemical_symbols() for b in atoms_b
    ]
    for a, b in zip(atoms_a, atoms_b):
        np.testing.assert_allclose(a.get_positions(), b.get_positions())


# --- failure handling / warnings (mocked pyxtal) -------------------------------


@patch("dim_red.generate.pyxtal")
def test_generate_structures_warns_and_continues_on_failures(mock_pyxtal_cls, caplog):
    from pyxtal.msg import Comp_CompatibilityError

    mock_instance = MagicMock()
    mock_instance.from_random.side_effect = Comp_CompatibilityError("always fails")
    mock_pyxtal_cls.return_value = mock_instance

    config = GenerationConfig(spacegroups=[225], structures_per_spacegroup=2, seed=0)
    with caplog.at_level("WARNING", logger="dim_red.generate"):
        atoms_list = generate_structures(config)

    assert atoms_list == []
    messages = [rec.message for rec in caplog.records]
    assert any(
        "Spacegroup 225" in m and "failed to generate 2/2" in m for m in messages
    )
    assert any("pyxtal generation:" in m for m in messages)


def test_generate_structures_reports_only_failing_spacegroups(caplog):
    from dim_red import generate as generate_module

    real_try_generate_one = generate_module._try_generate_one

    def fake_try_generate_one(spacegroup, species, config, rng):
        if spacegroup == 1:
            return None
        return real_try_generate_one(spacegroup, species, config, rng)

    config = GenerationConfig(
        spacegroups=[225, 1], structures_per_spacegroup=1, n_species=1, seed=0
    )
    with patch("dim_red.generate._try_generate_one", side_effect=fake_try_generate_one):
        with caplog.at_level("WARNING", logger="dim_red.generate"):
            atoms_list = generate_structures(config)

    assert len(atoms_list) == 1
    assert atoms_list[0].info["spacegroup"] == 225
    messages = [rec.message for rec in caplog.records]
    assert any("Spacegroup 1" in m for m in messages)
    assert not any("Spacegroup 225" in m for m in messages)
