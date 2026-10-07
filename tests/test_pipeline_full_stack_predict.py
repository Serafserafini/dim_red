import csv

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

pytest.importorskip("jax")

from dim_red.pipeline import full_stack as fs_mod
from dim_red.pipeline.full_stack import (
    FullStack,
    Prediction,
    StackPrediction,
    apply_to_structures,
)
from dim_red.pipeline.run_layout import FAMILY
from tests.fullstack_helpers import SYSTEM_SPACEGROUPS, make_run


def _structures(n):
    return [Atoms("Cu", positions=[[0, 0, 0]]) for _ in range(n)]


@pytest.fixture
def featurize_calls(monkeypatch):
    calls = []

    def fake(atoms_list, model_kind, data):
        calls.append(len(atoms_list))
        return np.stack(
            [np.arange(6, dtype=np.float32) + i for i in range(len(atoms_list))]
        )

    monkeypatch.setattr(fs_mod, "featurize_structures", fake)
    return calls


def test_stack_names_lists_trained_stacks(tmp_path, monkeypatch):
    fs = make_run(tmp_path, monkeypatch)
    assert fs.stack_names() == ["family", "tetragonal", "cubic"]


def test_predict_routes_each_structure_to_its_systems_expert(
    tmp_path, monkeypatch, featurize_calls
):
    fs = FullStack.open(make_run(tmp_path, monkeypatch).run_dir)
    real = FullStack._predict_stack

    def stub(self, name, structures, heads_name, device, cache, rows=None):
        if name == FAMILY:  # force the routing: Cubic, Hexagonal, Tetragonal, Cubic
            proba = np.eye(3)[[0, 1, 2, 0]]
            return StackPrediction(
                classes=["Cubic", "Hexagonal", "Tetragonal"],
                proba=proba,
                viz=np.zeros((4, 2)),
            )
        return real(self, name, structures, heads_name, device, cache, rows)

    monkeypatch.setattr(FullStack, "_predict_stack", stub)
    pred = fs.predict(_structures(4))

    assert isinstance(pred, Prediction)
    assert pred.family == ["Cubic", "Hexagonal", "Tetragonal", "Cubic"]
    assert pred.expert == ["cubic", None, "tetragonal", "cubic"]
    assert pred.spacegroup[1] is None and pred.spacegroup_proba[1] is None
    assert pred.viz_expert[1] is None
    assert pred.spacegroup[0] in SYSTEM_SPACEGROUPS["cubic"]
    assert pred.spacegroup[3] in SYSTEM_SPACEGROUPS["cubic"]
    assert pred.spacegroup[2] in SYSTEM_SPACEGROUPS["tetragonal"]
    assert pred.viz_expert[0].shape == (2,)
    assert pred.viz_family.shape == (4, 2)
    assert pred.family_proba.shape == (4,)
    assert 0.0 < pred.spacegroup_proba[0] <= 1.0


def test_features_are_computed_once_per_distinct_data_config(
    tmp_path, monkeypatch, featurize_calls
):
    fs = FullStack.open(make_run(tmp_path, monkeypatch).run_dir)
    real = FullStack._predict_stack

    def stub(self, name, structures, heads_name, device, cache, rows=None):
        if name == FAMILY:  # force both experts to run
            proba = np.eye(3)[[0, 2, 0, 2, 0]]
            return StackPrediction(
                classes=["Cubic", "Hexagonal", "Tetragonal"],
                proba=proba,
                viz=np.zeros((5, 2)),
            )
        return real(self, name, structures, heads_name, device, cache, rows)

    monkeypatch.setattr(FullStack, "_predict_stack", stub)
    pred = fs.predict(_structures(5))
    assert set(pred.expert) == {"cubic", "tetragonal"}
    assert featurize_calls == [5]  # both experts share one featurization


def test_predict_stack_works_alone_and_without_a_family_stack(
    tmp_path, monkeypatch, featurize_calls
):
    fs = FullStack.open(make_run(tmp_path, monkeypatch, with_family=False).run_dir)
    assert fs.stack_names() == ["tetragonal", "cubic"]
    out = fs.predict_stack("cubic", _structures(3))
    assert out.classes == SYSTEM_SPACEGROUPS["cubic"]
    assert out.proba.shape == (3, 2) and out.viz.shape == (3, 2)
    np.testing.assert_allclose(out.proba.sum(axis=1), 1.0, rtol=1e-5)
    with pytest.raises(ValueError, match="family stack.*predict_stack"):
        fs.predict(_structures(3))


def test_predict_stack_rejects_an_untrained_stack(
    tmp_path, monkeypatch, featurize_calls
):
    fs = FullStack.open(make_run(tmp_path, monkeypatch).run_dir)
    with pytest.raises(ValueError, match=r"'hexagonal'.*trained stacks"):
        fs.predict_stack("hexagonal", _structures(2))


def test_predict_requires_a_heads_choice_when_ambiguous(
    tmp_path, monkeypatch, featurize_calls
):
    fs = FullStack.open(make_run(tmp_path, monkeypatch, heads=("a", "b")).run_dir)
    with pytest.raises(ValueError, match="several heads"):
        fs.predict(_structures(2))
    assert len(fs.predict(_structures(2), heads_name="a").family) == 2


def test_predict_rejects_empty_input(tmp_path, monkeypatch, featurize_calls):
    fs = FullStack.open(make_run(tmp_path, monkeypatch).run_dir)
    with pytest.raises(ValueError, match="empty"):
        fs.predict([])


def test_apply_writes_predictions_csv_and_viz(tmp_path, monkeypatch, featurize_calls):
    run = make_run(tmp_path, monkeypatch).run_dir
    path = tmp_path / "new.extxyz"
    atoms = _structures(3)
    atoms[0].info["material_id"] = "mine-0"
    write(str(path), atoms, format="extxyz")

    out = apply_to_structures(run, path)

    assert out == run / "applied"
    rows = list(csv.DictReader(open(out / "new_predictions.csv")))
    assert [r["material_id"] for r in rows] == ["mine-0", "new-1", "new-2"]
    assert set(rows[0]) == {
        "material_id",
        "family",
        "family_proba",
        "expert",
        "spacegroup",
        "spacegroup_proba",
    }
    for r in rows:
        assert r["family"] in {"Cubic", "Tetragonal", "Hexagonal"}
        if r["expert"] == "":
            assert r["spacegroup"] == "" and r["spacegroup_proba"] == ""
        else:
            assert int(r["spacegroup"]) in SYSTEM_SPACEGROUPS[r["expert"]]
    with np.load(out / "new_viz.npz") as npz:
        assert npz["viz_family"].shape == (3, 2)
        assert npz["viz_expert"].shape == (3, 2)
        assert list(npz["material_ids"]) == ["mine-0", "new-1", "new-2"]


def test_apply_rejects_an_empty_structure_file(tmp_path, monkeypatch, featurize_calls):
    run = make_run(tmp_path, monkeypatch).run_dir
    empty = tmp_path / "empty.extxyz"
    empty.write_text("")
    with pytest.raises(ValueError, match="No structures"):
        apply_to_structures(run, empty)
