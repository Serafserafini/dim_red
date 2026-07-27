"""
Unit tests for the integrated analysis workflow.
"""

from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from ase import Atoms
from dim_red.analysis.workflow import run_pca_reduction


@patch("dim_red.analysis.workflow.fetch_structures_by_crystal_system")
def test_run_pca_reduction(mock_fetch):
    # Create mock Atoms objects for Cubic
    atoms_cubic_1 = Atoms("Cu", positions=[[0, 0, 0]], pbc=True, cell=[[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    atoms_cubic_1.info["material_id"] = "mp-30"
    atoms_cubic_2 = Atoms("Cu", positions=[[0, 0, 0]], pbc=True, cell=[[3.1, 0, 0], [0, 3.1, 0], [0, 0, 3.1]])
    atoms_cubic_2.info["material_id"] = "mp-31"
    
    # Create mock Atoms objects for Hexagonal
    atoms_hex_1 = Atoms("Mg", positions=[[0, 0, 0]], pbc=True, cell=[[3, 0, 0], [-1.5, 2.6, 0], [0, 0, 5]])
    atoms_hex_1.info["material_id"] = "mp-76"
    atoms_hex_2 = Atoms("Mg", positions=[[0, 0, 0]], pbc=True, cell=[[3.1, 0, 0], [-1.55, 2.68, 0], [0, 0, 5.1]])
    atoms_hex_2.info["material_id"] = "mp-77"
    
    # Setup mock behavior: first call returns Cubic, second returns Hexagonal
    mock_fetch.side_effect = [
        [atoms_cubic_1, atoms_cubic_2],
        [atoms_hex_1, atoms_hex_2]
    ]
    
    # Run PCA reduction on Cubic and Hexagonal
    X_reduced, labels, material_ids = run_pca_reduction(
        crystal_systems=["cubic", "hexagonal"],
        n_components=2,
        api_key="dummy_api_key",
        limit_per_system=2,
        soap_kwargs={"n_max": 2, "l_max": 2}
    )
    
    # Check outputs
    assert X_reduced.shape == (4, 2)
    assert labels == ["Cubic", "Cubic", "Hexagonal", "Hexagonal"]
    assert material_ids == ["mp-30", "mp-31", "mp-76", "mp-77"]
