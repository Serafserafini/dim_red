"""
Unit tests for fetching crystal structures from Materials Project.
"""

from unittest.mock import MagicMock, patch
import pytest
from ase import Atoms
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from dim_red.fetch import fetch_structures_by_crystal_system


def test_fetch_structures_no_api_key(monkeypatch):
    # Clear MP_API_KEY if set in env
    monkeypatch.delenv("MP_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is required"):
        fetch_structures_by_crystal_system(crystal_system="cubic")


@patch("dim_red.fetch.MPRester")
def test_fetch_structures_success(mock_mp_rester):
    # Create a mock pymatgen Structure
    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Cu"], [[0.0, 0.0, 0.0]])
    
    # Create a mock search document
    mock_doc = MagicMock()
    mock_doc.material_id = "mp-30"
    mock_doc.structure = structure
    
    # Setup mock MPRester context manager and search method
    mock_mpr_instance = mock_mp_rester.return_value.__enter__.return_value
    mock_mpr_instance.materials.summary.search.return_value = [mock_doc]
    
    # Call fetch function with dummy key
    atoms_list = fetch_structures_by_crystal_system(
        crystal_system="cubic",
        api_key="dummy_api_key",
        limit=1
    )
    
    # Verify correct parameters were used in search
    mock_mpr_instance.materials.summary.search.assert_called_once_with(
        crystal_system=["Cubic"],
        fields=["material_id", "structure"],
        chunk_size=1,
        num_chunks=1
    )
    
    # Check returned ASE Atoms properties
    assert len(atoms_list) == 1
    atoms = atoms_list[0]
    assert isinstance(atoms, Atoms)
    assert atoms.info["material_id"] == "mp-30"
    assert atoms.get_chemical_symbols() == ["Cu"]
