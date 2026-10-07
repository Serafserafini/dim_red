import pytest
import yaml

pytest.importorskip("jax")

from dim_red.pipeline.config import SweepConfig
from dim_red.pipeline.run_layout import discover_full_stack_runs, trained_stack_names
from dim_red.pipeline.sweep import run_sweep
from tests.fullstack_helpers import config_dict, fake_builder


@pytest.fixture
def patched(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dim_red.pipeline.full_stack.build_stack_dataset",
        fake_builder(tmp_path, []),
    )


def _sweep(tmp_path, grid):
    base = config_dict(tmp_path / "runs", experts=("cubic",))
    return SweepConfig(base=base, grid=grid)


def test_one_run_per_grid_combination(tmp_path, patched):
    sweep = _sweep(
        tmp_path,
        {"family.encoder.latent_dim": [3, 5], "experts.defaults.train.epochs": [2]},
    )
    run_dirs = run_sweep(sweep, cache_dir=tmp_path / "cache")
    assert len(run_dirs) == 2
    assert {p.parent for p in run_dirs} == {tmp_path / "runs" / run_dirs[0].parent.name}
    assert sorted(p.name for p in run_dirs) == [
        "fs_latent_dim-3_epochs-2",
        "fs_latent_dim-5_epochs-2",
    ]
    for d in run_dirs:
        assert trained_stack_names(d) == ["family", "cubic"]
        assert (d / "stacks" / "family" / "heads" / "default" / "heads.yaml").exists()


def test_sweep_dir_records_base_and_grid(tmp_path, patched):
    sweep = _sweep(tmp_path, {"family.encoder.latent_dim": [3]})
    (run_dir,) = run_sweep(sweep, cache_dir=tmp_path / "cache")
    saved = yaml.safe_load((run_dir.parent / "sweep.yaml").read_text())
    assert saved["grid"] == {"family.encoder.latent_dim": [3]}
    assert saved["base"]["name"] == "fs"


def test_discover_finds_the_sweeps_runs(tmp_path, patched):
    sweep = _sweep(tmp_path, {"family.encoder.latent_dim": [3, 5]})
    run_dirs = run_sweep(sweep, cache_dir=tmp_path / "cache")
    assert discover_full_stack_runs(run_dirs[0].parent) == sorted(run_dirs)


def test_second_sweep_gets_a_new_numbered_dir(tmp_path, patched):
    sweep = _sweep(tmp_path, {"family.encoder.latent_dim": [3]})
    (a,) = run_sweep(sweep, cache_dir=tmp_path / "cache")
    (b,) = run_sweep(sweep, cache_dir=tmp_path / "cache")
    assert a.parent != b.parent and b.parent.name.endswith("-2")


def test_invalid_combination_fails_before_anything_is_trained(tmp_path, patched):
    sweep = _sweep(tmp_path, {"family.encoder.latent_dimm": [3]})
    with pytest.raises(ValueError, match="unknown keys"):
        run_sweep(sweep, cache_dir=tmp_path / "cache")
    assert not (tmp_path / "runs").exists()


def test_cgcnn_style_base_is_refused_with_a_hint(tmp_path):
    sweep = SweepConfig(base={"model": "cgcnn", "output_dir": str(tmp_path)}, grid={})
    with pytest.raises(ValueError, match="dimred-run"):
        run_sweep(sweep)


def test_empty_grid_runs_the_base_once(tmp_path, patched):
    sweep = _sweep(tmp_path, {})
    run_dirs = run_sweep(sweep, cache_dir=tmp_path / "cache")
    assert [p.name for p in run_dirs] == ["fs"]
