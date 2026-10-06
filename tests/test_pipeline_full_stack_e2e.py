"""Real (unmocked) FullStack run on a tiny pyxtal configuration. Slow: run
with ``pytest tests/test_pipeline_full_stack_e2e.py --runslow``."""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("pyxtal")
pytest.importorskip("dscribe")

from dim_red.pipeline.full_stack import FullStack
from dim_red.pipeline.full_stack_config import full_stack_config_from_dict


def _block(spacegroups, per_sg):
    return {
        "data": {
            "pyxtal": {"spacegroups": spacegroups, "structures_per_spacegroup": per_sg},
            "soap": {"r_cut": 3.0, "n_max": 2, "l_max": 2},
        },
        "encoder": {"encoder_hidden_dim": [16], "latent_dim": 4},
        "projection": {"projection_dim": 8},
        "train": {"epochs": 30, "batch_size": 8, "learning_rate": 1e-2},
        "classifier": {"hidden_dim": 8},
        "viz": {"hidden_dim": [8]},
        "min_train_rows": 6,
    }


@pytest.mark.slow
def test_full_stack_trains_end_to_end_on_tiny_pyxtal(tmp_path):
    config = full_stack_config_from_dict(
        {
            "name": "e2e",
            "seed": 0,
            "output_dir": str(tmp_path / "runs"),
            "family": _block([16, 75, 195], 8),
            "experts": {
                "cubic": _block([195, 200], 8),
                "tetragonal": _block([75, 81], 8),
            },
        }
    )
    fs = FullStack.create(config)
    bodies = fs.fit_body()
    heads = fs.fit_heads("h")

    assert set(bodies) == {"family", "cubic", "tetragonal"}
    for name, history in bodies.items():
        assert len(history["train_loss"]) > 0, name
        assert np.isfinite(history["train_loss"]).all(), name
    for name, result in heads.items():
        clf = result["classifier"]["train_loss"]
        assert np.isfinite(clf).all() and clf[-1] < clf[0], name
        preds = np.load(
            fs.run_dir / "stacks" / name / "heads" / "h" / "predictions.npz"
        )
        np.testing.assert_allclose(preds["probs"].sum(axis=1), 1.0, atol=1e-4)
        assert preds["probs"].shape[0] == len(preds["label_ids"]), name
