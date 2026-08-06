"""
Unit tests for the fetch -> SOAP dataset cache used by the pipeline sweep to
avoid repeating expensive network + SOAP work across hidden-layer configs
that share the same crystal-system subset.
"""

from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.io import read as read_atoms

from dim_red.augmentation import AugmentationConfig
from dim_red.pipeline.dataset_cache import _cache_key, get_or_build_dataset


def _fake_atoms(symbol: str, material_id: str, spacegroup: int) -> Atoms:
    atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
    atoms.info["material_id"] = material_id
    atoms.info["spacegroup"] = spacegroup
    return atoms


def test_get_or_build_dataset_cache_miss_then_hit(tmp_path):
    fake_atoms = [_fake_atoms("Cu", "mp-1", 225), _fake_atoms("Fe", "mp-2", 229)]
    fake_soap = np.array([[1.0, 2.0], [3.0, 4.0]])

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            return_value=fake_atoms,
        ) as mock_fetch,
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap
        ) as mock_soap,
    ):
        X1, labels1, ids1, sg1, structures_path1 = get_or_build_dataset(
            crystal_systems=["cubic"],
            soap_kwargs={"r_cut": 3.0, "n_max": 2, "l_max": 2},
            limit_per_system=2,
            cache_dir=tmp_path,
        )

        assert mock_fetch.call_count == 1
        assert mock_soap.call_count == 1
        assert X1.shape == (2, 2)
        assert labels1 == ["Cubic", "Cubic"]
        assert ids1 == ["mp-1", "mp-2"]
        assert sg1 == [225, 229]

        # The exact structures are cached as extended XYZ alongside the .npz.
        assert structures_path1.suffix == ".extxyz"
        assert structures_path1.exists()
        cached_structures = read_atoms(structures_path1, index=":")
        assert len(cached_structures) == 2
        assert cached_structures[0].info["material_id"] == "mp-1"
        assert cached_structures[1].info["material_id"] == "mp-2"

        # compute_soap should always be called with average="outer" and
        # auto-detected species when none were configured.
        _, soap_call_kwargs = mock_soap.call_args
        assert soap_call_kwargs["average"] == "outer"
        assert soap_call_kwargs["species"] == ["Cu", "Fe"]

        # Second call with identical parameters should hit the cache and not
        # call fetch/compute_soap again.
        X2, labels2, ids2, sg2, structures_path2 = get_or_build_dataset(
            crystal_systems=["cubic"],
            soap_kwargs={"r_cut": 3.0, "n_max": 2, "l_max": 2},
            limit_per_system=2,
            cache_dir=tmp_path,
        )

        assert mock_fetch.call_count == 1
        assert mock_soap.call_count == 1
        np.testing.assert_allclose(X1, X2)
        assert labels2 == labels1
        assert ids2 == ids1
        assert sg2 == sg1
        assert structures_path2 == structures_path1


def test_get_or_build_dataset_rebuilds_stale_cache_missing_spacegroups(tmp_path):
    """A cache .npz written before the "spacegroups" field existed should be
    treated as a miss and rebuilt, not raise a KeyError.
    """
    fake_atoms = [_fake_atoms("Cu", "mp-1", 225)]
    fake_soap = np.array([[1.0, 2.0]])
    soap_kwargs = {"r_cut": 3.0, "n_max": 2, "l_max": 2}

    from dim_red.pipeline.dataset_cache import _cache_key

    key = _cache_key(["cubic"], 2, soap_kwargs)
    tmp_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        tmp_path / f"{key}.npz",
        X=np.array([[9.0, 9.0]]),
        labels=np.array(["Stale"]),
        material_ids=np.array(["mp-old"]),
        # "spacegroups" intentionally omitted to simulate a pre-existing cache.
    )

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            return_value=fake_atoms,
        ) as mock_fetch,
        patch("dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap),
    ):
        X, labels, ids, sg, structures_path = get_or_build_dataset(
            crystal_systems=["cubic"],
            soap_kwargs=soap_kwargs,
            limit_per_system=2,
            cache_dir=tmp_path,
        )

    assert mock_fetch.call_count == 1
    assert labels == ["Cubic"]
    assert ids == ["mp-1"]
    assert sg == [225]
    assert structures_path.exists()


def test_get_or_build_dataset_rebuilds_stale_cache_missing_structures(tmp_path):
    """A cache .npz whose companion .extxyz is missing (e.g. pre-existing
    cache from before that artifact existed) should also be treated as a
    miss and rebuilt, not silently return without any structures file.
    """
    fake_atoms = [_fake_atoms("Cu", "mp-1", 225)]
    fake_soap = np.array([[1.0, 2.0]])
    soap_kwargs = {"r_cut": 3.0, "n_max": 2, "l_max": 2}

    from dim_red.pipeline.dataset_cache import _cache_key

    key = _cache_key(["cubic"], 2, soap_kwargs)
    tmp_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        tmp_path / f"{key}.npz",
        X=np.array([[9.0, 9.0]]),
        labels=np.array(["Stale"]),
        material_ids=np.array(["mp-old"]),
        spacegroups=np.array([1], dtype=np.int64),
        # ".extxyz" companion intentionally not written.
    )

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            return_value=fake_atoms,
        ) as mock_fetch,
        patch("dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap),
    ):
        X, labels, ids, sg, structures_path = get_or_build_dataset(
            crystal_systems=["cubic"],
            soap_kwargs=soap_kwargs,
            limit_per_system=2,
            cache_dir=tmp_path,
        )

    assert mock_fetch.call_count == 1
    assert labels == ["Cubic"]
    assert structures_path.exists()


def test_get_or_build_dataset_different_params_miss_cache(tmp_path):
    fake_atoms = [_fake_atoms("Cu", "mp-1", 225)]
    fake_soap = np.array([[1.0, 2.0]])

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            return_value=fake_atoms,
        ) as mock_fetch,
        patch("dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap),
    ):
        get_or_build_dataset(
            crystal_systems=["cubic"],
            soap_kwargs={"r_cut": 3.0, "n_max": 2, "l_max": 2},
            limit_per_system=2,
            cache_dir=tmp_path,
        )
        get_or_build_dataset(
            crystal_systems=["hexagonal"],
            soap_kwargs={"r_cut": 3.0, "n_max": 2, "l_max": 2},
            limit_per_system=2,
            cache_dir=tmp_path,
        )

        # Different crystal_systems -> different cache key -> fetch called twice.
        assert mock_fetch.call_count == 2
        assert len(list(tmp_path.glob("*.npz"))) == 2
        assert len(list(tmp_path.glob("*.extxyz"))) == 2


def test_get_or_build_dataset_raises_when_nothing_fetched(tmp_path):
    with patch(
        "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
        return_value=[],
    ):
        with pytest.raises(ValueError, match="No structures fetched"):
            get_or_build_dataset(
                crystal_systems=["cubic"],
                soap_kwargs={},
                limit_per_system=2,
                cache_dir=tmp_path,
            )


# --- augmentation ------------------------------------------------------------


def test_get_or_build_dataset_applies_augmentation(tmp_path):
    fake_atoms = [_fake_atoms("Cu", "mp-1", 225), _fake_atoms("Fe", "mp-2", 229)]
    # 2 fetched structures x (1 original + 1 augmented copy) = 4 rows.
    fake_soap = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    augmentation = AugmentationConfig(
        n_augmented=1,
        keep_original=True,
        jitter_probability=1.0,
        jitter_std=0.01,
        vacancy_probability=0.0,
        seed=0,
    )

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            return_value=fake_atoms,
        ) as mock_fetch,
        patch("dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap),
    ):
        X, labels, ids, sg, structures_path = get_or_build_dataset(
            crystal_systems=["cubic"],
            soap_kwargs={"r_cut": 3.0, "n_max": 2, "l_max": 2},
            limit_per_system=2,
            cache_dir=tmp_path,
            augmentation=augmentation,
        )

    assert mock_fetch.call_count == 1
    assert X.shape == (4, 2)
    assert labels == ["Cubic"] * 4
    assert len(ids) == 4
    assert len(read_atoms(structures_path, index=":")) == 4


def test_get_or_build_dataset_augmentation_changes_cache_key(tmp_path):
    fake_atoms = [_fake_atoms("Cu", "mp-1", 225)]
    augmentation = AugmentationConfig(n_augmented=1, jitter_probability=1.0, seed=0)

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            return_value=fake_atoms,
        ) as mock_fetch,
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap",
            side_effect=[np.array([[1.0, 2.0]]), np.array([[1.0, 2.0], [3.0, 4.0]])],
        ),
    ):
        get_or_build_dataset(
            crystal_systems=["cubic"],
            soap_kwargs={"r_cut": 3.0, "n_max": 2, "l_max": 2},
            limit_per_system=2,
            cache_dir=tmp_path,
        )
        get_or_build_dataset(
            crystal_systems=["cubic"],
            soap_kwargs={"r_cut": 3.0, "n_max": 2, "l_max": 2},
            limit_per_system=2,
            cache_dir=tmp_path,
            augmentation=augmentation,
        )

    # Different augmentation settings -> different cache key -> fetch twice,
    # two distinct cache files on disk.
    assert mock_fetch.call_count == 2
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_cache_key_differs_with_and_without_augmentation():
    soap_kwargs = {"r_cut": 3.0, "n_max": 2, "l_max": 2}
    key_plain = _cache_key(["cubic"], 2, soap_kwargs)
    key_aug = _cache_key(["cubic"], 2, soap_kwargs, AugmentationConfig(n_augmented=2))
    assert key_plain != key_aug


# --- pyxtal data source ------------------------------------------------------

pytest.importorskip("pyxtal")

from dim_red.pipeline.config import AugmentationConfig as PipelineAugmentationConfig
from dim_red.pipeline.config import (
    FetchConfig,
    PyxtalConfig,
    RunConfig,
    SoapConfig,
    TrainSettings,
    VAEArchConfig,
)
from dim_red.pipeline.dataset_cache import (
    build_dataset_for_run,
    get_or_build_pyxtal_dataset,
)


def _fake_generated_atoms(symbol, material_id, spacegroup, family):
    atoms = _fake_atoms(symbol, material_id, spacegroup)
    atoms.info["family"] = family
    return atoms


def test_get_or_build_pyxtal_dataset_cache_miss_then_hit(tmp_path):
    fake_atoms = [
        _fake_generated_atoms("Cu", "pyxtal-225-0", 225, "Cubic"),
        _fake_generated_atoms("Fe", "pyxtal-225-1", 225, "Cubic"),
    ]
    fake_soap = np.array([[1.0, 2.0], [3.0, 4.0]])
    pyxtal_config = PyxtalConfig(spacegroups=[225], structures_per_spacegroup=2)

    with (
        patch(
            "dim_red.generate.generate_structures", return_value=fake_atoms
        ) as mock_generate,
        patch(
            "dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap
        ) as mock_soap,
    ):
        X1, labels1, ids1, sg1, structures_path1 = get_or_build_pyxtal_dataset(
            pyxtal_config=pyxtal_config,
            seed=0,
            soap_kwargs={"r_cut": 3.0, "n_max": 2, "l_max": 2},
            cache_dir=tmp_path,
        )

        assert mock_generate.call_count == 1
        assert mock_soap.call_count == 1
        assert X1.shape == (2, 2)
        assert labels1 == ["Cubic", "Cubic"]
        assert ids1 == ["pyxtal-225-0", "pyxtal-225-1"]
        assert sg1 == [225, 225]
        assert structures_path1.exists()
        assert len(read_atoms(structures_path1, index=":")) == 2

        # Second call with identical parameters should hit the cache.
        X2, labels2, ids2, sg2, structures_path2 = get_or_build_pyxtal_dataset(
            pyxtal_config=pyxtal_config,
            seed=0,
            soap_kwargs={"r_cut": 3.0, "n_max": 2, "l_max": 2},
            cache_dir=tmp_path,
        )

        assert mock_generate.call_count == 1
        assert mock_soap.call_count == 1
        np.testing.assert_allclose(X1, X2)
        assert labels2 == labels1
        assert ids2 == ids1
        assert sg2 == sg1
        assert structures_path2 == structures_path1


def test_get_or_build_pyxtal_dataset_different_seed_misses_cache(tmp_path):
    fake_atoms = [_fake_generated_atoms("Cu", "pyxtal-225-0", 225, "Cubic")]
    fake_soap = np.array([[1.0, 2.0]])
    pyxtal_config = PyxtalConfig(spacegroups=[225], structures_per_spacegroup=1)

    with (
        patch(
            "dim_red.generate.generate_structures", return_value=fake_atoms
        ) as mock_generate,
        patch("dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap),
    ):
        get_or_build_pyxtal_dataset(pyxtal_config, 0, {}, tmp_path)
        get_or_build_pyxtal_dataset(pyxtal_config, 1, {}, tmp_path)

        assert mock_generate.call_count == 2
        assert len(list(tmp_path.glob("*.npz"))) == 2
        assert len(list(tmp_path.glob("*.extxyz"))) == 2


def test_get_or_build_pyxtal_dataset_applies_augmentation(tmp_path):
    fake_atoms = [_fake_generated_atoms("Cu", "pyxtal-225-0", 225, "Cubic")]
    # 1 generated structure x (1 original + 1 augmented copy) = 2 rows.
    fake_soap = np.array([[1.0, 2.0], [3.0, 4.0]])
    pyxtal_config = PyxtalConfig(spacegroups=[225], structures_per_spacegroup=1)
    augmentation = AugmentationConfig(n_augmented=1, jitter_probability=1.0, seed=0)

    with (
        patch(
            "dim_red.generate.generate_structures", return_value=fake_atoms
        ) as mock_generate,
        patch("dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap),
    ):
        X, labels, ids, sg, structures_path = get_or_build_pyxtal_dataset(
            pyxtal_config=pyxtal_config,
            seed=0,
            soap_kwargs={"r_cut": 3.0, "n_max": 2, "l_max": 2},
            cache_dir=tmp_path,
            augmentation=augmentation,
        )

    assert mock_generate.call_count == 1
    assert X.shape == (2, 2)
    assert labels == ["Cubic", "Cubic"]
    assert len(read_atoms(structures_path, index=":")) == 2


def test_get_or_build_pyxtal_dataset_raises_when_nothing_generated(tmp_path):
    pyxtal_config = PyxtalConfig(spacegroups=[225], structures_per_spacegroup=1)
    with patch("dim_red.generate.generate_structures", return_value=[]):
        with pytest.raises(ValueError, match="pyxtal generated no structures"):
            get_or_build_pyxtal_dataset(pyxtal_config, 0, {}, tmp_path)


def _run_config(data_source="fetch", pyxtal_config=None, augmentation_config=None):
    return RunConfig(
        fetch=(
            FetchConfig(crystal_systems=["cubic"], limit_per_system=2)
            if data_source == "fetch"
            else None
        ),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=2),
        train=TrainSettings(),
        seed=0,
        data_source=data_source,
        pyxtal=pyxtal_config,
        augmentation=augmentation_config,
    )


def test_build_dataset_for_run_dispatches_to_fetch(tmp_path):
    fake_atoms = [_fake_atoms("Cu", "mp-1", 225)]
    fake_soap = np.array([[1.0, 2.0]])
    config = _run_config(data_source="fetch")

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            return_value=fake_atoms,
        ) as mock_fetch,
        patch("dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap),
        patch("dim_red.generate.generate_structures") as mock_generate,
    ):
        build_dataset_for_run(config, cache_dir=tmp_path)

    assert mock_fetch.call_count == 1
    mock_generate.assert_not_called()


def test_build_dataset_for_run_dispatches_to_pyxtal(tmp_path):
    fake_atoms = [_fake_generated_atoms("Cu", "pyxtal-225-0", 225, "Cubic")]
    fake_soap = np.array([[1.0, 2.0]])
    config = _run_config(
        data_source="pyxtal",
        pyxtal_config=PyxtalConfig(spacegroups=[225], structures_per_spacegroup=1),
    )

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system"
        ) as mock_fetch,
        patch("dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap),
        patch(
            "dim_red.generate.generate_structures", return_value=fake_atoms
        ) as mock_generate,
    ):
        X, labels, ids, sg, structures_path = build_dataset_for_run(
            config, cache_dir=tmp_path
        )

    assert mock_generate.call_count == 1
    mock_fetch.assert_not_called()
    assert labels == ["Cubic"]
    assert ids == ["pyxtal-225-0"]
    assert sg == [225]
    assert structures_path.exists()


def test_get_or_build_pyxtal_dataset_accepts_pyxtal_config_with_seed_set(tmp_path):
    """PyxtalConfig.seed must not collide with the "seed" kwarg passed to
    GenerationConfig (regression test: dataclasses.asdict(pyxtal_config)
    includes "seed" too, which used to raise "multiple values for 'seed'").
    """
    fake_atoms = [_fake_generated_atoms("Cu", "pyxtal-225-0", 225, "Cubic")]
    fake_soap = np.array([[1.0, 2.0]])
    pyxtal_config = PyxtalConfig(
        spacegroups=[225], structures_per_spacegroup=1, seed=999
    )

    with (
        patch("dim_red.generate.generate_structures", return_value=fake_atoms),
        patch("dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap),
    ):
        X, labels, ids, sg, structures_path = get_or_build_pyxtal_dataset(
            pyxtal_config=pyxtal_config, seed=999, soap_kwargs={}, cache_dir=tmp_path
        )
    assert labels == ["Cubic"]


def test_build_dataset_for_run_pyxtal_seed_falls_back_to_run_seed(tmp_path):
    config = _run_config(
        data_source="pyxtal",
        pyxtal_config=PyxtalConfig(spacegroups=[225], structures_per_spacegroup=1),
    )
    assert config.pyxtal.seed is None

    with patch(
        "dim_red.pipeline.dataset_cache.get_or_build_pyxtal_dataset"
    ) as mock_get:
        mock_get.return_value = (
            np.zeros((1, 1)),
            ["Cubic"],
            ["id"],
            [225],
            tmp_path / "fake.extxyz",
        )
        build_dataset_for_run(config, cache_dir=tmp_path)

    _, kwargs = mock_get.call_args
    assert kwargs["seed"] == config.seed


def test_build_dataset_for_run_pyxtal_seed_overrides_run_seed(tmp_path):
    config = _run_config(
        data_source="pyxtal",
        pyxtal_config=PyxtalConfig(
            spacegroups=[225], structures_per_spacegroup=1, seed=999
        ),
    )

    with patch(
        "dim_red.pipeline.dataset_cache.get_or_build_pyxtal_dataset"
    ) as mock_get:
        mock_get.return_value = (
            np.zeros((1, 1)),
            ["Cubic"],
            ["id"],
            [225],
            tmp_path / "fake.extxyz",
        )
        build_dataset_for_run(config, cache_dir=tmp_path)

    _, kwargs = mock_get.call_args
    assert kwargs["seed"] == 999
    assert kwargs["seed"] != config.seed


def test_build_dataset_for_run_augmentation_defaults_to_none(tmp_path):
    config = _run_config(data_source="fetch")
    assert config.augmentation is None

    with patch("dim_red.pipeline.dataset_cache.get_or_build_dataset") as mock_get:
        mock_get.return_value = (
            np.zeros((1, 1)),
            ["Cubic"],
            ["id"],
            [225],
            tmp_path / "fake.extxyz",
        )
        build_dataset_for_run(config, cache_dir=tmp_path)

    _, kwargs = mock_get.call_args
    assert kwargs["augmentation"] is None


def test_build_dataset_for_run_forwards_augmentation_to_fetch(tmp_path):
    fake_atoms = [_fake_atoms("Cu", "mp-1", 225)]
    fake_soap = np.array([[1.0, 2.0], [3.0, 4.0]])
    config = _run_config(
        data_source="fetch",
        augmentation_config=PipelineAugmentationConfig(
            n_augmented=1, jitter_probability=1.0, seed=5
        ),
    )

    with (
        patch(
            "dim_red.pipeline.dataset_cache.fetch_structures_by_crystal_system",
            return_value=fake_atoms,
        ) as mock_fetch,
        patch("dim_red.pipeline.dataset_cache.compute_soap", return_value=fake_soap),
    ):
        X, labels, ids, sg, structures_path = build_dataset_for_run(
            config, cache_dir=tmp_path
        )

    assert mock_fetch.call_count == 1
    assert X.shape == (2, 2)  # 1 fetched x (1 original + 1 augmented copy)


def test_build_dataset_for_run_augmentation_seed_falls_back_to_run_seed(tmp_path):
    config = _run_config(
        data_source="fetch",
        augmentation_config=PipelineAugmentationConfig(n_augmented=1),
    )
    assert config.augmentation.seed is None

    with patch("dim_red.pipeline.dataset_cache.get_or_build_dataset") as mock_get:
        mock_get.return_value = (
            np.zeros((1, 1)),
            ["Cubic"],
            ["id"],
            [225],
            tmp_path / "fake.extxyz",
        )
        build_dataset_for_run(config, cache_dir=tmp_path)

    _, kwargs = mock_get.call_args
    assert kwargs["augmentation"].seed == config.seed


def test_build_dataset_for_run_augmentation_seed_overrides_run_seed(tmp_path):
    config = _run_config(
        data_source="fetch",
        augmentation_config=PipelineAugmentationConfig(n_augmented=1, seed=999),
    )

    with patch("dim_red.pipeline.dataset_cache.get_or_build_dataset") as mock_get:
        mock_get.return_value = (
            np.zeros((1, 1)),
            ["Cubic"],
            ["id"],
            [225],
            tmp_path / "fake.extxyz",
        )
        build_dataset_for_run(config, cache_dir=tmp_path)

    _, kwargs = mock_get.call_args
    assert kwargs["augmentation"].seed == 999


def test_build_dataset_for_run_forwards_supercell_radius(tmp_path):
    config = _run_config(
        data_source="fetch",
        augmentation_config=PipelineAugmentationConfig(
            n_augmented=1, supercell_radius=5.0
        ),
    )

    with patch("dim_red.pipeline.dataset_cache.get_or_build_dataset") as mock_get:
        mock_get.return_value = (
            np.zeros((1, 1)),
            ["Cubic"],
            ["id"],
            [225],
            tmp_path / "fake.extxyz",
        )
        build_dataset_for_run(config, cache_dir=tmp_path)

    _, kwargs = mock_get.call_args
    assert kwargs["augmentation"].supercell_radius == 5.0
