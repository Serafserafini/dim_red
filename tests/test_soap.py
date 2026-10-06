"""
Unit tests for SOAP descriptor generation.
"""

import numpy as np
from ase.build import molecule

from dim_red.soap import compute_soap


def test_compute_soap_single_molecule():
    # Build a simple water molecule
    water = molecule("H2O")

    # Compute SOAP without specifying species (auto-detection)
    soap_vec = compute_soap(water, r_cut=3.0, n_max=2, l_max=2, average="off")

    # average="off" returns a descriptor for each atom
    # water molecule has 3 atoms
    assert isinstance(soap_vec, np.ndarray)
    assert soap_vec.ndim == 2
    assert soap_vec.shape[0] == 3
    assert soap_vec.shape[1] > 0


def test_compute_soap_average():
    water = molecule("H2O")

    # Compute averaged SOAP
    soap_vec = compute_soap(water, r_cut=3.0, n_max=2, l_max=2, average="outer")

    # average="outer" returns a single vector for the entire system
    assert isinstance(soap_vec, np.ndarray)
    assert soap_vec.ndim == 1 or (soap_vec.ndim == 2 and soap_vec.shape[0] == 1)
    # dscribe might return 2D array of shape (1, n_features) or 1D array of shape (n_features,)
    # let's flatten or check the last dimension
    assert soap_vec.shape[-1] > 0


def test_compute_soap_list():
    water = molecule("H2O")
    co2 = molecule("CO2")

    # Compute SOAP for a list of molecules
    # CO2 has C and O, H2O has H and O -> total species: C, H, O
    soap_vecs = compute_soap([water, co2], r_cut=3.0, n_max=2, l_max=2, average="outer")

    # Result should have 2 configurations
    assert isinstance(soap_vecs, np.ndarray)
    assert soap_vecs.shape[0] == 2
    assert soap_vecs.shape[1] > 0
