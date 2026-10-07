import numpy as np
import pytest
from ase.build import bulk

pytest.importorskip("dscribe")
pytest.importorskip("jax")

from dim_red.pipeline.config import MaceConfig, PyxtalConfig, SoapConfig
from dim_red.pipeline.featurize import featurize_structures, standardize
from dim_red.pipeline.full_stack_config import StackDataConfig


def _data(**soap):
    base = dict(r_cut=4.0, n_max=3, l_max=2, sigma=0.5, element_agnostic=True)
    base.update(soap)
    return StackDataConfig(pyxtal=PyxtalConfig(), soap=SoapConfig(**base))


def _nacl():
    return bulk("NaCl", "rocksalt", a=5.6, cubic=True)


def _si():
    return bulk("Si", "diamond", a=5.43, cubic=True)


def test_featurization_ignores_other_structures_in_the_batch():
    alone = featurize_structures([_nacl()], "supcon", _data())
    in_batch = featurize_structures([_si(), _nacl()], "supcon", _data())
    np.testing.assert_allclose(alone[0], in_batch[1], atol=1e-6)


def test_elements_do_not_change_the_vector():
    a = _nacl()
    b = _nacl()
    b.set_chemical_symbols(
        ["K" if s == "Na" else "Br" for s in b.get_chemical_symbols()]
    )
    fa = featurize_structures([a], "supcon", _data())
    fb = featurize_structures([b], "supcon", _data())
    np.testing.assert_allclose(fa, fb, atol=1e-6)


def test_output_is_float32_matrix():
    out = featurize_structures([_nacl(), _si()], "supcon", _data())
    assert out.dtype == np.float32 and out.ndim == 2 and out.shape[0] == 2


def test_non_agnostic_soap_is_refused():
    with pytest.raises(ValueError, match="element_agnostic"):
        featurize_structures([_nacl()], "supcon", _data(element_agnostic=False))


def test_empty_input_is_refused():
    with pytest.raises(ValueError, match="empty"):
        featurize_structures([], "supcon", _data())


def test_standardize_matches_the_training_formula():
    raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mean = np.array([2.0, 3.0], dtype=np.float32)
    std = np.array([1.0, 2.0], dtype=np.float32)
    out = standardize(raw, mean, std)
    np.testing.assert_allclose(out, [[-1.0, -0.5], [1.0, 0.5]])
    assert out.dtype == np.float32
