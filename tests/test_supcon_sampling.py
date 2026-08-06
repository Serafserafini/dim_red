"""
Unit tests for the SupCon balanced batch sampler (no jax dependency -- pure
numpy index logic).
"""

import logging

import numpy as np
import pytest

from dim_red.supcon.sampling import balanced_batch_indices, iter_balanced_batches


def test_s_none_uses_every_spacegroup_present_and_splits_uniformly():
    family_ids = np.array([0] * 20 + [1] * 20)
    spacegroup_ids = np.array(([0] * 10 + [1] * 10) + ([2] * 10 + [3] * 10))
    rng = np.random.default_rng(0)

    idx = balanced_batch_indices(
        family_ids, spacegroup_ids, P=None, K=6, S=None, rng=rng
    )

    assert len(idx) == 2 * 6  # 2 families x K=6
    for family, expected_spacegroups in ((0, {0, 1}), (1, {2, 3})):
        family_mask = family_ids[idx] == family
        assert family_mask.sum() == 6
        # S=None -> every spacegroup present for that family is used.
        assert set(spacegroup_ids[idx][family_mask].tolist()) == expected_spacegroups


def test_spacegroup_stratified_batch_has_expected_size():
    family_ids = np.array([0] * 20 + [1] * 20)
    spacegroup_ids = np.array(([0] * 10 + [1] * 10) + ([2] * 10 + [3] * 10))
    rng = np.random.default_rng(0)

    idx = balanced_batch_indices(family_ids, spacegroup_ids, P=None, K=6, S=2, rng=rng)

    assert len(idx) == 2 * 6  # 2 families x K=6
    for family, expected_spacegroups in ((0, {0, 1}), (1, {2, 3})):
        family_mask = family_ids[idx] == family
        assert family_mask.sum() == 6
        assert set(spacegroup_ids[idx][family_mask].tolist()) == expected_spacegroups


def test_k_not_divisible_by_s_distributes_remainder():
    # 1 family, 3 spacegroups, K=7 -> base=2, remainder=1: first spacegroup
    # chosen gets 3, the other two get 2 each (2+2+3=7).
    family_ids = np.zeros(30, dtype=int)
    spacegroup_ids = np.array([0] * 10 + [1] * 10 + [2] * 10)
    rng = np.random.default_rng(0)

    idx = balanced_batch_indices(family_ids, spacegroup_ids, P=1, K=7, S=3, rng=rng)

    assert len(idx) == 7
    counts = sorted(int((spacegroup_ids[idx] == sg).sum()) for sg in (0, 1, 2))
    assert counts == [2, 2, 3]


def test_p_larger_than_available_families_is_clamped_with_warning(caplog):
    family_ids = np.array([0] * 5 + [1] * 5)
    spacegroup_ids = np.zeros(10, dtype=int)
    rng = np.random.default_rng(0)

    with caplog.at_level(logging.WARNING, logger="dim_red.pipeline"):
        idx = balanced_batch_indices(
            family_ids, spacegroup_ids, P=5, K=2, S=None, rng=rng
        )

    assert len(idx) == 2 * 2  # clamped to the 2 families actually present
    assert any("clamping to 2" in r.message for r in caplog.records)


def test_s_larger_than_available_spacegroups_is_clamped_with_warning(caplog):
    family_ids = np.zeros(10, dtype=int)
    spacegroup_ids = np.array([0] * 5 + [1] * 5)  # only 2 distinct spacegroups
    rng = np.random.default_rng(0)

    with caplog.at_level(logging.WARNING, logger="dim_red.pipeline"):
        idx = balanced_batch_indices(family_ids, spacegroup_ids, P=1, K=4, S=5, rng=rng)

    assert len(idx) == 4
    assert any("clamping to 2" in r.message for r in caplog.records)


def test_sampling_without_replacement_when_pool_is_large_enough():
    family_ids = np.zeros(1000, dtype=int)
    spacegroup_ids = np.zeros(1000, dtype=int)
    rng = np.random.default_rng(0)

    idx = balanced_batch_indices(family_ids, spacegroup_ids, P=1, K=50, S=None, rng=rng)

    assert len(idx) == len(set(idx.tolist()))  # no duplicates


def test_sampling_with_replacement_when_pool_is_too_small():
    family_ids = np.zeros(3, dtype=int)  # pool of 3, need 10
    spacegroup_ids = np.zeros(3, dtype=int)
    rng = np.random.default_rng(0)

    idx = balanced_batch_indices(family_ids, spacegroup_ids, P=1, K=10, S=None, rng=rng)

    assert len(idx) == 10
    assert set(idx.tolist()) <= {0, 1, 2}
    assert len(set(idx.tolist())) < 10  # necessarily repeats


def test_iter_balanced_batches_matches_requested_count_and_shapes():
    family_ids = np.array([0] * 20 + [1] * 20)
    spacegroup_ids = np.array(([0] * 10 + [1] * 10) * 2)
    X = np.arange(40 * 3, dtype=np.float32).reshape(40, 3)
    rng = np.random.default_rng(0)

    batches = iter_balanced_batches(
        (X, family_ids, spacegroup_ids),
        family_ids,
        spacegroup_ids,
        P=None,
        K=4,
        S=2,
        n_batches=5,
        rng=rng,
    )

    assert len(batches) == 5
    for batch_x, batch_family, batch_spacegroup in batches:
        assert batch_x.shape == (8, 3)  # 2 families x K=4
        assert batch_family.shape == (8,)
        assert batch_spacegroup.shape == (8,)


def test_iter_balanced_batches_slices_arrays_consistently_with_chosen_indices():
    family_ids = np.arange(20) % 2
    spacegroup_ids = np.zeros(20, dtype=int)
    X = np.arange(20).reshape(20, 1).astype(np.float32)
    rng = np.random.default_rng(0)

    batches = iter_balanced_batches(
        (X, family_ids),
        family_ids,
        spacegroup_ids,
        P=None,
        K=3,
        S=None,
        n_batches=1,
        rng=rng,
    )
    batch_x, batch_family = batches[0]
    # X was built so that X[i, 0] == i and family_ids[i] == i % 2 -> the
    # sliced family values must match what X's own row index implies.
    np.testing.assert_array_equal(batch_family, batch_x[:, 0].astype(int) % 2)
