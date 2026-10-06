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
