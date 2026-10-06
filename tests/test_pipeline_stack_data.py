from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("jax")

from dim_red.pipeline.full_stack_config import FAMILY, full_stack_config_from_dict
from dim_red.pipeline.stack_data import StackDataset, build_stack_dataset


def _spec(name=FAMILY, model_kind="supcon", extra_data=None, seed=5):
    data = {"pyxtal": {"structures_per_spacegroup": 2}}
    data.update(extra_data or {})
    block = {
        "data": data,
        "encoder": {"encoder_hidden_dim": [4], "latent_dim": 2},
    }
    d = {"seed": seed, "model_kind": model_kind}
    if model_kind == "supcon_mace":
        data["mace"] = {"checkpoint_path": "ckpt"}
    if name == FAMILY:
        d["family"] = block
    else:
        d["experts"] = {name: block}
    return full_stack_config_from_dict(d).stacks[name]


def _fake_result():
    return (
        np.zeros((3, 4), dtype=np.float32),
        ["Cubic"] * 3,
        ["a", "b", "c"],
        [195, 195, 196],
        Path("s.extxyz"),
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
    )


def test_soap_dataset_uses_the_stack_seed_and_soap_kwargs():
    spec = _spec("cubic", seed=5)
    with patch(
        "dim_red.pipeline.stack_data.get_or_build_pyxtal_dataset",
        return_value=_fake_result(),
    ) as soap_builder:
        ds = build_stack_dataset(spec, "supcon", "cache")
    kwargs = soap_builder.call_args.kwargs
    assert kwargs["seed"] == spec.data.pyxtal.seed == 5 + 7
    assert kwargs["pyxtal_config"].families == ["Cubic"]
    assert kwargs["soap_kwargs"] == spec.data.soap.as_kwargs()
    assert kwargs["augmentation"] is None
    assert isinstance(ds, StackDataset) and ds.X.shape == (3, 4)
    assert ds.spacegroups == [195, 195, 196]


def test_mace_model_kind_routes_to_the_mace_builder():
    spec = _spec(model_kind="supcon_mace")
    with patch(
        "dim_red.pipeline.stack_data.get_or_build_pyxtal_mace_dataset",
        return_value=_fake_result(),
    ) as mace_builder:
        build_stack_dataset(spec, "supcon_mace", "cache")
    assert mace_builder.call_args.kwargs["mace_kwargs"] == spec.data.mace.mace_kwargs()


def test_augmentation_seed_defaults_to_the_stack_seed():
    spec = _spec(
        extra_data={"augmentation": {"n_augmented": 1, "jitter_std": 0.01}}, seed=3
    )
    with patch(
        "dim_red.pipeline.stack_data.get_or_build_pyxtal_dataset",
        return_value=_fake_result(),
    ) as soap_builder:
        build_stack_dataset(spec, "supcon", "cache")
    augmentation = soap_builder.call_args.kwargs["augmentation"]
    assert augmentation.n_augmented == 1 and augmentation.seed == 3
