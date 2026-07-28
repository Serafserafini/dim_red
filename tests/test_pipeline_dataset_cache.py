"""
Unit tests for the fetch -> SOAP dataset cache used by the pipeline sweep to
avoid repeating expensive network + SOAP work across hidden-layer configs
that share the same crystal-system subset.
"""

from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from dim_red.pipeline.dataset_cache import get_or_build_dataset


def _fake_atoms(symbol: str, material_id: str) -> Atoms:
    atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
    atoms.info["material_id"] = material_id
    return atoms


def test_get_or_build_dataset_cache_miss_then_hit(tmp_path):
    fake_atoms = [_fake_atoms("Cu", "mp-1"), _fake_atoms("Fe", "mp-2")]
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
        X1, labels1, ids1 = get_or_build_dataset(
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

        # compute_soap should always be called with average="outer" and
        # auto-detected species when none were configured.
        _, soap_call_kwargs = mock_soap.call_args
        assert soap_call_kwargs["average"] == "outer"
        assert soap_call_kwargs["species"] == ["Cu", "Fe"]

        # Second call with identical parameters should hit the cache and not
        # call fetch/compute_soap again.
        X2, labels2, ids2 = get_or_build_dataset(
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


def test_get_or_build_dataset_different_params_miss_cache(tmp_path):
    fake_atoms = [_fake_atoms("Cu", "mp-1")]
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
