"""
Unit tests for the sweep comparison suite: given a directory of already
completed run folders (config.yaml + loss_history.csv + embeddings.npz, the
same artifacts ``run_single`` writes), ``dim_red.pipeline.compare`` should
discover them and render the comparison plots without needing jax/VAE
training to run again.
"""

import csv

import numpy as np
import pytest
import yaml

from dim_red.pipeline.compare import (
    LatentUmapParams,
    _abbreviate_hyperparam_key,
    available_loss_metrics,
    compute_embedding_baselines,
    discover_runs,
    final_metric_groups,
    generate_comparison_report,
    latent_grid_axes,
    load_runs,
    plot_aux_accuracy_comparison,
    plot_final_metric_vs_hyperparam,
    plot_latent_space_grid,
    plot_loss_curves,
    plot_spacegroup_family_histogram,
    run_labels,
    varying_hyperparams,
)
from dim_red.pipeline.config import (
    AuxHeadsConfig,
    FetchConfig,
    RunConfig,
    SoapConfig,
    TrainSettings,
    VAEArchConfig,
    run_config_to_dict,
)

matplotlib = pytest.importorskip("matplotlib")


def _write_loss_history(path, epochs=3, val_loss_final=1.0, aux=False):
    fieldnames = [
        "epoch",
        "train_loss",
        "train_recon",
        "train_kl",
        "val_loss",
        "val_recon",
        "val_kl",
    ]
    if aux:
        fieldnames += ["train_family_ce", "val_family_ce"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            val_loss = val_loss_final + (epochs - epoch) * 0.1
            row = {
                "epoch": epoch,
                "train_loss": val_loss + 0.05,
                "train_recon": val_loss * 0.7,
                "train_kl": val_loss * 0.3,
                "val_loss": val_loss,
                "val_recon": val_loss * 0.7,
                "val_kl": val_loss * 0.3,
            }
            if aux:
                row["train_family_ce"] = 0.2
                row["val_family_ce"] = 0.25
            writer.writerow(row)


def _write_run(
    run_dir,
    hidden_dim,
    crystal_systems,
    val_loss_final,
    with_aux=False,
    n=6,
    learning_rate=1e-3,
    n_features=None,
    latent_dim=2,
):
    run_dir.mkdir(parents=True)

    aux_heads = (
        AuxHeadsConfig(mode="family_only", lambda_family=1.0, head_hidden_dim=4)
        if with_aux
        else AuxHeadsConfig()
    )
    config = RunConfig(
        fetch=FetchConfig(crystal_systems=crystal_systems, limit_per_system=8),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=hidden_dim, latent_dim=latent_dim),
        train=TrainSettings(
            epochs=3, batch_size=4, val_ratio=0.25, learning_rate=learning_rate
        ),
        aux_heads=aux_heads,
        seed=0,
        output_dir=str(run_dir.parent),
        name=run_dir.name,
    )
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(run_config_to_dict(config), f, sort_keys=False)

    _write_loss_history(
        run_dir / "loss_history.csv", val_loss_final=val_loss_final, aux=with_aux
    )

    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(n, latent_dim)).astype(np.float32)
    labels = np.array([crystal_systems[0].capitalize()] * n)
    payload = dict(
        embeddings=embeddings,
        labels=labels,
        material_ids=np.array([f"mp-{i}" for i in range(n)]),
        spacegroups=np.array([1] * n, dtype=np.int64),
        split=np.array(["train"] * (n - 1) + ["val"]),
    )
    if with_aux:
        family_probs = np.zeros((n, 1), dtype=np.float32)
        family_probs[:, 0] = 1.0
        payload["family_probs"] = family_probs
        payload["family_classes"] = np.array([crystal_systems[0].capitalize()])
    if n_features is not None:
        payload["features"] = rng.normal(size=(n, n_features)).astype(np.float32)
    np.savez(run_dir / "embeddings.npz", **payload)
    return run_dir


@pytest.fixture
def sweep_dir(tmp_path):
    d = tmp_path / "20260729-1"
    _write_run(d / "hd-4_cs-cubic", [4], ["cubic"], val_loss_final=1.0)
    _write_run(d / "hd-8_cs-cubic", [8], ["cubic"], val_loss_final=0.5)
    return d


def test_discover_runs_finds_completed_run_dirs(sweep_dir):
    dirs = discover_runs(sweep_dir)
    assert [d.name for d in dirs] == ["hd-4_cs-cubic", "hd-8_cs-cubic"]


def test_discover_runs_raises_when_empty(tmp_path):
    empty = tmp_path / "empty-sweep"
    empty.mkdir()
    with pytest.raises(ValueError, match="No completed runs"):
        discover_runs(empty)


def test_varying_hyperparams_detects_hidden_dim_only(sweep_dir):
    runs = load_runs(sweep_dir)
    assert varying_hyperparams(runs) == ["vae.encoder_hidden_dim"]


def test_varying_hyperparams_is_not_limited_to_named_axes(tmp_path):
    """Any dotted config path that differs across runs is detected -- not
    just hidden dims/crystal systems/lambda weights -- since the sweep grid
    itself can vary any RunConfig field.
    """
    d = tmp_path / "20260729-1"
    _write_run(d / "lr-a", [4], ["cubic"], val_loss_final=1.0, learning_rate=0.01)
    _write_run(d / "lr-b", [4], ["cubic"], val_loss_final=0.5, learning_rate=0.001)
    runs = load_runs(d)
    assert varying_hyperparams(runs) == ["train.learning_rate"]


def test_crystal_systems_hyperparam_is_abbreviated(tmp_path):
    d = tmp_path / "20260729-1"
    _write_run(
        d / "run-a", [4], ["cubic", "hexagonal", "monoclinic", "orthorhombic"], 1.0
    )
    runs = load_runs(d)
    groups = final_metric_groups(runs, "fetch.crystal_systems")
    # Long crystal-system names truncated to 3 letters each, not the full
    # names, so axis labels/titles built from this value stay plot-safe.
    (key,) = groups.keys()
    assert key == "cub-hex-mon-ort"


def test_run_labels_abbreviates_crystal_systems_and_stays_short(tmp_path):
    d = tmp_path / "20260729-1"
    long_systems = ["cubic", "hexagonal", "monoclinic", "orthorhombic", "tetragonal"]
    run_dir_a = _write_run(d / "run-a", [128, 64, 32], long_systems, 1.0)
    run_dir_b = _write_run(d / "run-b", [128, 64], ["cubic"], 0.5)
    runs = load_runs(d)
    labels = run_labels(runs)
    assert set(labels) == {run_dir_a, run_dir_b}
    assert "monoclinic" not in labels[run_dir_a]  # full names not used
    assert "mon" in labels[run_dir_a]  # abbreviation is
    # Hyperparameter *names* are also abbreviated (hd, cs), same short forms
    # as run-directory names, not the full config field names.
    assert "encoder_hidden_dim" not in labels[run_dir_a]
    assert "crystal_systems" not in labels[run_dir_a]
    assert "hd=128-64-32" in labels[run_dir_a]
    assert "cs=" in labels[run_dir_a]
    for label in labels.values():
        assert len(label) < 80


def test_abbreviate_hyperparam_key_uses_short_forms():
    assert _abbreviate_hyperparam_key("vae.encoder_hidden_dim") == "hd"
    assert _abbreviate_hyperparam_key("fetch.crystal_systems") == "cs"
    assert _abbreviate_hyperparam_key("aux_heads.lambda_family") == "lf"
    assert _abbreviate_hyperparam_key("train.learning_rate") == "lr"


def test_abbreviate_hyperparam_key_falls_back_to_full_name_when_unknown():
    assert _abbreviate_hyperparam_key("some.unmapped_field") == "unmapped_field"


def test_run_labels_falls_back_to_directory_name_when_nothing_varies(tmp_path):
    d = tmp_path / "20260729-1"
    run_dir = _write_run(d / "only-run", [4], ["cubic"], 1.0)
    runs = load_runs(d)
    labels = run_labels(runs)
    assert labels == {run_dir: "only-run"}


def test_plot_loss_curves_writes_file(sweep_dir, tmp_path):
    runs = load_runs(sweep_dir)
    out = tmp_path / "loss_curves.png"
    plot_loss_curves(runs, save_path=out)
    assert out.exists()


def test_plot_loss_curves_writes_csv(sweep_dir, tmp_path):
    runs = load_runs(sweep_dir)
    out = tmp_path / "loss_curves.csv"
    plot_loss_curves(runs, csv_path=out)
    assert out.exists()
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0]) == {"run", "epoch", "metric", "value"}
    # 2 runs x 3 epochs x 2 default metrics (train_loss, val_loss) -- calling
    # plot_loss_curves directly, with no explicit `metrics`, stays just the
    # total: generate_comparison_report is the one that now asks for every
    # available metric explicitly (see below).
    assert len(rows) == 2 * 3 * 2
    assert {row["metric"] for row in rows} == {"train_loss", "val_loss"}


def test_generate_comparison_report_loss_curves_include_all_available_metrics(
    sweep_dir,
):
    """The `compare` command's loss_curves.png/.csv must plot every loss
    component common to all runs (recon/kl here), not just the total.
    """
    report_dir = generate_comparison_report(sweep_dir, write_data_files=True)
    with open(report_dir / "loss_curves.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {row["metric"] for row in rows} == {
        "train_loss",
        "val_loss",
        "train_recon",
        "val_recon",
        "train_kl",
        "val_kl",
    }


def test_generate_comparison_report_loss_curves_include_aux_ce_terms(tmp_path):
    d = tmp_path / "20260729-3"
    _write_run(d / "hd-4_cs-cubic", [4], ["cubic"], val_loss_final=1.0, with_aux=True)
    _write_run(d / "hd-8_cs-cubic", [8], ["cubic"], val_loss_final=0.5, with_aux=True)

    report_dir = generate_comparison_report(d, write_data_files=True)
    with open(report_dir / "loss_curves.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {"train_family_ce", "val_family_ce"} <= {row["metric"] for row in rows}


def test_final_metric_groups_groups_multiple_runs_per_value(tmp_path):
    # Two crystal-system sets, three runs each (mirrors the real sweep shape
    # that surfaced the "only 2 bars" overlap bug: several runs sharing the
    # same crystal_systems value must all land in one group, not collapse).
    d = tmp_path / "20260729-1"
    for i, val_loss in enumerate([1.0, 1.2, 1.4]):
        _write_run(
            d / f"hd-{4+i}_cs-cubic", [4 + i], ["cubic"], val_loss_final=val_loss
        )
    for i, val_loss in enumerate([2.0, 2.2]):
        _write_run(
            d / f"hd-{4+i}_cs-hexagonal",
            [4 + i],
            ["hexagonal"],
            val_loss_final=val_loss,
        )

    runs = load_runs(d)
    groups = final_metric_groups(runs, "fetch.crystal_systems")

    assert set(groups) == {"cub", "hex"}
    assert len(groups["cub"]) == 3
    assert len(groups["hex"]) == 2
    # No averaging/collapsing at grouping time -- every run's own final
    # val_loss survives in its group, in the order runs were loaded.
    np.testing.assert_allclose(
        sorted(groups["cub"]), sorted([1.0, 1.2, 1.4]), atol=1e-6
    )
    np.testing.assert_allclose(sorted(groups["hex"]), sorted([2.0, 2.2]), atol=1e-6)


def test_plot_final_metric_vs_hyperparam_writes_file(sweep_dir, tmp_path):
    runs = load_runs(sweep_dir)
    out = tmp_path / "final_vs_hd.png"
    plot_final_metric_vs_hyperparam(runs, "vae.encoder_hidden_dim", save_path=out)
    assert out.exists()


def test_plot_final_metric_vs_hyperparam_writes_csv(sweep_dir, tmp_path):
    runs = load_runs(sweep_dir)
    out = tmp_path / "final_vs_hd.csv"
    plot_final_metric_vs_hyperparam(
        runs, "vae.encoder_hidden_dim", metric="val_loss", csv_path=out
    )
    assert out.exists()
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0]) == {"run", "vae.encoder_hidden_dim", "val_loss"}
    assert len(rows) == len(runs)


def test_plot_final_metric_vs_hyperparam_rejects_unknown_key(sweep_dir):
    runs = load_runs(sweep_dir)
    with pytest.raises(ValueError, match="Unknown hyperparam"):
        plot_final_metric_vs_hyperparam(runs, "not_a_real_key")


def test_plot_spacegroup_family_histogram_writes_file(sweep_dir, tmp_path):
    runs = load_runs(sweep_dir)
    out = tmp_path / "spacegroup_histogram.png"
    plot_spacegroup_family_histogram(runs, save_path=out)
    assert out.exists()


def test_plot_spacegroup_family_histogram_writes_csv(sweep_dir, tmp_path):
    runs = load_runs(sweep_dir)
    out = tmp_path / "spacegroup_histogram.csv"
    plot_spacegroup_family_histogram(runs, csv_path=out)
    assert out.exists()
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0]) == {"spacegroup", "family"}
    # Computed once, from the first run's dataset only (6 points, fixture's
    # default n), not once per run.
    assert len(rows) == 6
    assert {row["family"] for row in rows} == {"Cubic"}


def test_plot_latent_space_grid_writes_file(sweep_dir, tmp_path):
    runs = load_runs(sweep_dir)
    out = tmp_path / "latent_grid.png"
    plot_latent_space_grid(runs, save_path=out)
    assert out.exists()


def test_plot_latent_space_grid_writes_csv(sweep_dir, tmp_path):
    runs = load_runs(sweep_dir)
    out = tmp_path / "latent_grid.csv"
    plot_latent_space_grid(runs, csv_path=out)
    assert out.exists()
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0]) == {
        "source",
        "row",
        "col",
        "is_baseline",
        "projected_umap",
        "dim_0",
        "dim_1",
        "label",
    }
    # 2 runs x 6 points each; no baseline (fixture doesn't save "features").
    assert len(rows) == 2 * 6
    assert all(row["is_baseline"] == "False" for row in rows)
    # Fixture runs are already 2D -- no UMAP projection needed.
    assert all(row["projected_umap"] == "False" for row in rows)


def test_latent_grid_axes_returns_none_with_fewer_than_two_varying(sweep_dir):
    runs = load_runs(sweep_dir)  # only vae.encoder_hidden_dim varies
    assert latent_grid_axes(runs) is None


def test_latent_grid_axes_picks_two_largest_cardinality_axes(tmp_path):
    d = tmp_path / "20260729-1"
    # vae.encoder_hidden_dim: 3 distinct values; fetch.crystal_systems: 2.
    for hd in ([4], [8], [16]):
        for cs in (["cubic"], ["hexagonal"]):
            _write_run(d / f"hd-{hd[0]}_cs-{cs[0]}", hd, cs, 1.0)
    runs = load_runs(d)
    assert latent_grid_axes(runs) == ("vae.encoder_hidden_dim", "fetch.crystal_systems")


def test_plot_latent_space_grid_lays_out_two_axes_as_rows_and_cols(tmp_path):
    d = tmp_path / "20260729-1"
    for hd in ([4], [8]):
        for lr in (0.01, 0.001, 0.0001):
            _write_run(d / f"hd-{hd[0]}_lr-{lr}", hd, ["cubic"], 1.0, learning_rate=lr)
    runs = load_runs(d)
    assert latent_grid_axes(runs) == ("train.learning_rate", "vae.encoder_hidden_dim")
    out = tmp_path / "latent_grid.png"
    plot_latent_space_grid(runs, save_path=out)
    assert out.exists()


def test_plot_latent_space_grid_drops_extra_runs_sharing_a_cell(tmp_path, caplog):
    d = tmp_path / "20260729-1"
    # 3 axes vary, all tied at 2 distinct values each -> only 2 of them
    # become the grid's rows/cols, so the 3rd axis's extra runs collide
    # into already-occupied cells and get dropped (with a warning).
    for hd in ([4], [8]):
        for cs in (["cubic"], ["hexagonal"]):
            for lr in (0.01, 0.001):
                _write_run(
                    d / f"hd-{hd[0]}_cs-{cs[0]}_lr-{lr}", hd, cs, 1.0, learning_rate=lr
                )
    runs = load_runs(d)
    out = tmp_path / "latent_grid.png"
    with caplog.at_level("WARNING", logger="dim_red.pipeline"):
        plot_latent_space_grid(runs, save_path=out)
    assert out.exists()
    assert any("shared a" in rec.message for rec in caplog.records)


def test_compute_embedding_baselines_returns_none_without_saved_features(
    sweep_dir, caplog
):
    runs = load_runs(sweep_dir)  # fixture's _write_run doesn't save "features"
    with caplog.at_level("WARNING", logger="dim_red.pipeline"):
        result = compute_embedding_baselines(runs)
    assert result is None
    assert any("saved 'features'" in rec.message for rec in caplog.records)


def test_compute_embedding_baselines_fits_pca_on_first_available_run(tmp_path):
    d = tmp_path / "20260729-1"
    _write_run(d / "hd-4_cs-cubic", [4], ["cubic"], 1.0)  # no features
    _write_run(d / "hd-8_cs-cubic", [8], ["cubic"], 0.5, n=10, n_features=6)
    runs = load_runs(d)

    result = compute_embedding_baselines(runs)
    assert result is not None
    source_name, labels, baselines = result
    assert source_name == "hd-8_cs-cubic"  # the only one with saved features
    assert labels.shape[0] == 10
    assert "PCA" in baselines
    assert baselines["PCA"].shape == (10, 2)


def test_plot_latent_space_grid_includes_baseline_row_when_features_present(tmp_path):
    d = tmp_path / "20260729-1"
    _write_run(d / "hd-4_cs-cubic", [4], ["cubic"], 1.0, n=10, n_features=6)
    _write_run(d / "hd-8_cs-cubic", [8], ["cubic"], 0.5, n=10, n_features=6)
    runs = load_runs(d)
    out = tmp_path / "latent_grid.png"

    plot_latent_space_grid(runs, save_path=out)
    assert out.exists()


def test_plot_latent_space_grid_csv_includes_baseline_rows_when_features_present(
    tmp_path,
):
    d = tmp_path / "20260729-1"
    _write_run(d / "hd-4_cs-cubic", [4], ["cubic"], 1.0, n=10, n_features=6)
    _write_run(d / "hd-8_cs-cubic", [8], ["cubic"], 0.5, n=10, n_features=6)
    runs = load_runs(d)
    out = tmp_path / "latent_grid.csv"

    plot_latent_space_grid(runs, csv_path=out)
    assert out.exists()
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    baseline_sources = {row["source"] for row in rows if row["is_baseline"] == "True"}
    assert "PCA baseline" in baseline_sources
    non_baseline_rows = [row for row in rows if row["is_baseline"] == "False"]
    assert len(non_baseline_rows) == 2 * 10


def test_plot_latent_space_grid_projects_non_2d_embeddings_via_umap(tmp_path):
    d = tmp_path / "20260729-1"
    _write_run(d / "hd-4_cs-cubic", [4], ["cubic"], 1.0, n=30, latent_dim=5)
    runs = load_runs(d)
    out_png = tmp_path / "latent_grid.png"
    out_csv = tmp_path / "latent_grid.csv"

    plot_latent_space_grid(
        runs,
        save_path=out_png,
        csv_path=out_csv,
        umap_params=LatentUmapParams(n_neighbors=5),
    )
    assert out_png.exists()

    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    non_baseline_rows = [row for row in rows if row["is_baseline"] == "False"]
    assert len(non_baseline_rows) == 30
    assert all(row["projected_umap"] == "True" for row in non_baseline_rows)
    # Projected down to exactly 2D regardless of the original 5D embedding.
    for row in non_baseline_rows:
        float(row["dim_0"])
        float(row["dim_1"])


def test_plot_latent_space_grid_mixed_2d_and_non_2d_runs(tmp_path):
    d = tmp_path / "20260729-1"
    _write_run(d / "hd-4_cs-cubic", [4], ["cubic"], 1.0, n=15, latent_dim=2)
    _write_run(d / "hd-8_cs-cubic", [8], ["cubic"], 0.5, n=15, latent_dim=4)
    runs = load_runs(d)
    out_csv = tmp_path / "latent_grid.csv"

    plot_latent_space_grid(runs, csv_path=out_csv)

    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    by_source = {}
    for row in rows:
        if row["is_baseline"] == "False":
            by_source.setdefault(row["source"], set()).add(row["projected_umap"])
    # Exactly one run needed projecting, the other stayed as-is.
    assert sorted(v.pop() for v in by_source.values()) == ["False", "True"]


def test_plot_latent_space_grid_skips_run_when_umap_unavailable(
    tmp_path, monkeypatch, caplog
):
    d = tmp_path / "20260729-1"
    _write_run(d / "hd-4_cs-cubic", [4], ["cubic"], 1.0, n=10, latent_dim=5)
    runs = load_runs(d)

    import dim_red.pipeline.compare as compare_module

    def _raise_import_error(*args, **kwargs):
        raise ImportError("umap-learn not installed")

    monkeypatch.setattr(compare_module, "_make_umap", _raise_import_error)

    out = tmp_path / "latent_grid.png"
    with caplog.at_level("WARNING", logger="dim_red.pipeline"):
        plot_latent_space_grid(runs, save_path=out)

    assert not out.exists()
    assert any("No runs could be shown in 2D" in rec.message for rec in caplog.records)


def test_compute_embedding_baselines_forwards_umap_params(tmp_path, monkeypatch):
    d = tmp_path / "20260729-1"
    _write_run(d / "hd-4_cs-cubic", [4], ["cubic"], 1.0, n=10, n_features=6)
    runs = load_runs(d)

    captured = {}
    import dim_red.umap as umap_module

    original_init = umap_module.UMAP.__init__

    def _spy_init(self, *args, **kwargs):
        captured.update(kwargs)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(umap_module.UMAP, "__init__", _spy_init)

    params = LatentUmapParams(
        n_neighbors=3, min_dist=0.4, metric="cosine", random_state=7
    )
    compute_embedding_baselines(runs, umap_params=params)

    assert captured["n_neighbors"] == 3
    assert captured["min_dist"] == 0.4
    assert captured["metric"] == "cosine"
    assert captured["random_state"] == 7


def test_generate_comparison_report_forwards_umap_params(tmp_path):
    d = tmp_path / "20260729-1"
    _write_run(d / "hd-4_cs-cubic", [4], ["cubic"], 1.0, n=20, latent_dim=3)
    _write_run(d / "hd-8_cs-cubic", [8], ["cubic"], 0.5, n=20, latent_dim=3)

    report_dir = generate_comparison_report(
        d, write_data_files=True, umap_params=LatentUmapParams(n_neighbors=5)
    )
    assert (report_dir / "latent_space_grid.png").exists()
    with open(report_dir / "latent_space_grid.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    non_baseline_rows = [row for row in rows if row["is_baseline"] == "False"]
    assert all(row["projected_umap"] == "True" for row in non_baseline_rows)


def test_plot_aux_accuracy_skipped_without_aux_heads(sweep_dir, tmp_path):
    runs = load_runs(sweep_dir)
    out = tmp_path / "aux_acc.png"
    plot_aux_accuracy_comparison(runs, save_path=out)
    assert not out.exists()


def test_plot_aux_accuracy_written_when_aux_heads_present(tmp_path):
    d = tmp_path / "20260729-2"
    _write_run(d / "hd-4_cs-cubic", [4], ["cubic"], val_loss_final=1.0, with_aux=True)
    runs = load_runs(d)
    out = tmp_path / "aux_acc.png"
    plot_aux_accuracy_comparison(runs, save_path=out)
    assert out.exists()


def test_plot_aux_accuracy_csv_skipped_without_aux_heads(sweep_dir, tmp_path):
    runs = load_runs(sweep_dir)
    out = tmp_path / "aux_acc.csv"
    plot_aux_accuracy_comparison(runs, csv_path=out)
    assert not out.exists()


def test_plot_aux_accuracy_csv_written_when_aux_heads_present(tmp_path):
    d = tmp_path / "20260729-2"
    _write_run(d / "hd-4_cs-cubic", [4], ["cubic"], val_loss_final=1.0, with_aux=True)
    runs = load_runs(d)
    out = tmp_path / "aux_acc.csv"
    plot_aux_accuracy_comparison(runs, csv_path=out)
    assert out.exists()
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0]) == {"run", "metric", "accuracy"}
    assert rows[0]["metric"] == "family"


def test_generate_comparison_report_writes_expected_files(sweep_dir):
    report_dir = generate_comparison_report(sweep_dir, write_data_files=True)
    assert report_dir == sweep_dir / "comparison"
    assert (report_dir / "loss_curves.png").exists()
    assert (report_dir / "loss_curves.csv").exists()
    # Every available loss metric (not just val_loss) gets its own
    # final-metric-vs-hyperparameter plot, with a matching CSV of the data.
    for metric in [
        "train_loss",
        "val_loss",
        "train_recon",
        "val_recon",
        "train_kl",
        "val_kl",
    ]:
        assert (report_dir / f"final_{metric}_vs_vae_encoder_hidden_dim.png").exists()
        assert (report_dir / f"final_{metric}_vs_vae_encoder_hidden_dim.csv").exists()
    assert (report_dir / "spacegroup_histogram.png").exists()
    assert (report_dir / "spacegroup_histogram.csv").exists()
    assert (report_dir / "latent_space_grid.png").exists()
    assert (report_dir / "latent_space_grid.csv").exists()
    # No aux heads in this fixture -> no accuracy comparison written (PNG or CSV).
    assert not (report_dir / "aux_heads_accuracy.png").exists()
    assert not (report_dir / "aux_heads_accuracy.csv").exists()


def test_generate_comparison_report_skips_csvs_by_default(sweep_dir):
    report_dir = generate_comparison_report(sweep_dir)
    assert report_dir == sweep_dir / "comparison"
    # PNGs are always written...
    assert (report_dir / "loss_curves.png").exists()
    assert (report_dir / "spacegroup_histogram.png").exists()
    assert (report_dir / "latent_space_grid.png").exists()
    # ...but no CSV data files unless write_data_files=True is passed.
    assert not any(report_dir.glob("*.csv"))


def test_generate_comparison_report_includes_aux_ce_metrics_when_active(tmp_path):
    d = tmp_path / "20260729-1"
    _write_run(d / "hd-4_cs-cubic", [4], ["cubic"], val_loss_final=1.0, with_aux=True)
    _write_run(d / "hd-8_cs-cubic", [8], ["cubic"], val_loss_final=0.5, with_aux=True)

    report_dir = generate_comparison_report(d)

    for metric in ["train_family_ce", "val_family_ce"]:
        assert (report_dir / f"final_{metric}_vs_vae_encoder_hidden_dim.png").exists()
    assert (report_dir / "aux_heads_accuracy.png").exists()


def test_available_loss_metrics_orders_known_metrics_first(sweep_dir):
    runs = load_runs(sweep_dir)
    assert available_loss_metrics(runs) == [
        "train_loss",
        "val_loss",
        "train_recon",
        "val_recon",
        "train_kl",
        "val_kl",
    ]


def test_available_loss_metrics_includes_aux_ce_when_present(tmp_path):
    d = tmp_path / "20260729-1"
    _write_run(d / "hd-4_cs-cubic", [4], ["cubic"], val_loss_final=1.0, with_aux=True)
    runs = load_runs(d)
    assert available_loss_metrics(runs) == [
        "train_loss",
        "val_loss",
        "train_recon",
        "val_recon",
        "train_kl",
        "val_kl",
        "train_family_ce",
        "val_family_ce",
    ]


def test_available_loss_metrics_intersects_across_runs_with_different_columns(tmp_path):
    d = tmp_path / "20260729-1"
    _write_run(d / "hd-4_cs-cubic", [4], ["cubic"], val_loss_final=1.0, with_aux=True)
    _write_run(d / "hd-8_cs-cubic", [8], ["cubic"], val_loss_final=0.5, with_aux=False)
    runs = load_runs(d)
    # Only the metrics common to *every* run are usable for a fair comparison.
    assert available_loss_metrics(runs) == [
        "train_loss",
        "val_loss",
        "train_recon",
        "val_recon",
        "train_kl",
        "val_kl",
    ]
