import numpy as np
import pytest

from dim_red.pipeline.compare import classification_accuracies_from_npz
from dim_red.pipeline.run_layout import (
    FAMILY,
    STACK_ORDER,
    discover_full_stack_runs,
    head_names,
    is_full_stack_run,
    is_hidden_tmp,
    open_stack,
    resolve_heads_name,
    trained_stack_names,
)
from tests.fullstack_helpers import write_fake_run


def test_constants_are_the_canonical_stack_order():
    assert STACK_ORDER[0] == FAMILY
    assert STACK_ORDER[-1] == "cubic"
    assert len(STACK_ORDER) == 8


def test_hidden_tmp_dirs_are_never_stacks_or_heads(tmp_path):
    run = write_fake_run(tmp_path / "r", stacks=("family", "cubic"))
    # an interrupted atomic write leaves these behind
    (run / "stacks" / ".tetragonal.tmp" / "body").mkdir(parents=True)
    (run / "stacks" / ".tetragonal.tmp" / "body" / "stack.yaml").write_text("{}")
    heads = run / "stacks" / "family" / "heads"
    (heads / ".viz2.tmp").mkdir()
    (heads / ".viz2.tmp" / "heads.yaml").write_text("{}")
    assert is_hidden_tmp(heads / ".viz2.tmp")
    assert not is_hidden_tmp(heads / "default")
    assert trained_stack_names(run) == ["family", "cubic"]
    assert head_names(run / "stacks" / "family") == ["default"]


def test_trained_stack_names_use_canonical_order(tmp_path):
    run = write_fake_run(tmp_path / "r", stacks=("cubic", "family", "triclinic"))
    assert trained_stack_names(run) == ["family", "triclinic", "cubic"]


def test_a_stack_without_a_body_is_not_trained(tmp_path):
    run = write_fake_run(tmp_path / "r", stacks=("family", "cubic"))
    (run / "stacks" / "cubic" / "body" / "stack.yaml").unlink()
    assert trained_stack_names(run) == ["family"]


def test_discover_full_stack_runs(tmp_path):
    write_fake_run(tmp_path / "s" / "a")
    write_fake_run(tmp_path / "s" / "b")
    (tmp_path / "s" / "not_a_run").mkdir()
    (tmp_path / "s" / ".c.tmp").mkdir()
    assert [p.name for p in discover_full_stack_runs(tmp_path / "s")] == ["a", "b"]
    assert is_full_stack_run(tmp_path / "s" / "a")
    assert not is_full_stack_run(tmp_path / "s" / "not_a_run")


def test_discover_full_stack_runs_raises_when_empty(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No completed FullStack runs"):
        discover_full_stack_runs(tmp_path / "empty")


def test_resolve_heads_name_requires_a_choice_when_ambiguous(tmp_path):
    run = write_fake_run(tmp_path / "r", heads=("a", "b"))
    stack_dir = run / "stacks" / "family"
    with pytest.raises(ValueError, match=r"several heads.*\['a', 'b'\]"):
        resolve_heads_name(stack_dir)
    assert resolve_heads_name(stack_dir, "b") == "b"
    with pytest.raises(ValueError, match="no heads named 'zzz'"):
        resolve_heads_name(stack_dir, "zzz")


def test_resolve_heads_name_with_none_trained(tmp_path):
    run = write_fake_run(tmp_path / "r", heads=())
    with pytest.raises(ValueError, match="no heads trained"):
        resolve_heads_name(run / "stacks" / "family")


def test_open_family_stack_exposes_classification_payload(tmp_path):
    run = write_fake_run(tmp_path / "r", families=("Cubic", "Hexagonal"), n=8)
    data = open_stack(run, FAMILY)
    assert data.stack == FAMILY and data.run_dir == run
    assert data.model_kind == "supcon" and data.heads_name == "default"
    assert data.label == "r"
    assert data.viz_embeddings.shape == (8, 2)
    assert data.flat_config["model.latent_dim"] == 2
    assert data.loss_history["val_loss"].shape == (3,)
    accs = classification_accuracies_from_npz(data.embeddings)
    assert accs["family"] == 1.0 and "spacegroup" not in accs


def test_open_expert_stack_exposes_spacegroup_payload(tmp_path):
    run = write_fake_run(tmp_path / "r", stacks=("family", "cubic"), n=8)
    data = open_stack(run, "cubic")
    accs = classification_accuracies_from_npz(data.embeddings)
    assert accs["spacegroup"] == 1.0 and "family" not in accs
    assert set(data.embeddings["labels"]) == {"Cubic"}
    assert data.embeddings["spacegroup_classes"].dtype == np.int64


def test_open_stack_without_heads_requires_opt_in(tmp_path):
    run = write_fake_run(tmp_path / "r", heads=())
    with pytest.raises(ValueError, match="no heads trained"):
        open_stack(run, FAMILY)
    data = open_stack(run, FAMILY, allow_no_heads=True)
    assert data.heads_name is None and data.viz_embeddings is None
    assert "family_probs" not in data.embeddings


def test_open_untrained_stack_names_the_trained_ones(tmp_path):
    run = write_fake_run(tmp_path / "r")
    with pytest.raises(ValueError, match=r"'cubic' is not trained.*\['family'\]"):
        open_stack(run, "cubic")
