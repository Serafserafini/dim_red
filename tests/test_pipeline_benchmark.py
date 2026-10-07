"""
Unit tests for the cross-architecture benchmark table
(``dim_red.pipeline.benchmark``): given already-completed run folders (the
same artifacts ``run_single``/``train_tail`` write), it should assemble one
CSV comparing them on standardized, dimensionality-agnostic metrics without
needing any real training to run again.
"""

import csv
import dataclasses
import math

import numpy as np
import pytest

from dim_red.analysis.metrics import embedding_quality_metrics
from dim_red.pipeline.benchmark import (
    _resolve_2d_embedding,
    benchmark_row,
    collect_run_dirs,
    generate_benchmark_plots,
    generate_benchmark_table,
    plot_2d_quality_by_model_kind,
    plot_classification_accuracy_by_model_kind,
    run_classification_accuracies,
)
from dim_red.pipeline.run_layout import open_stack
from tests.fullstack_helpers import write_fake_run

pytest.importorskip("sklearn")
matplotlib = pytest.importorskip("matplotlib")


# --- dim_red.analysis.metrics.embedding_quality_metrics -------------------


def test_embedding_quality_metrics_separated_clusters_score_high():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0.0, scale=0.1, size=(25, 2))
    cluster_b = rng.normal(loc=10.0, scale=0.1, size=(25, 2))
    embeddings = np.concatenate([cluster_a, cluster_b]).astype(np.float32)
    labels = np.array([0] * 25 + [1] * 25)

    metrics = embedding_quality_metrics(embeddings, {"family": labels})

    assert metrics["family_silhouette"] > 0.9
    assert metrics["family_kmeans_ari"] > 0.9
    assert metrics["family_kmeans_nmi"] > 0.9
    assert metrics["family_knn_accuracy"] > 0.9


def test_embedding_quality_metrics_single_class_returns_nan_no_raise():
    embeddings = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    labels = np.zeros(10, dtype=np.int64)

    metrics = embedding_quality_metrics(embeddings, {"family": labels})

    assert math.isnan(metrics["family_silhouette"])
    assert math.isnan(metrics["family_kmeans_ari"])
    assert math.isnan(metrics["family_kmeans_nmi"])
    assert math.isnan(metrics["family_knn_accuracy"])


def test_embedding_quality_metrics_excludes_negative_sentinel_labels():
    rng = np.random.default_rng(1)
    cluster_a = rng.normal(loc=0.0, scale=0.1, size=(20, 2))
    cluster_b = rng.normal(loc=10.0, scale=0.1, size=(20, 2))
    embeddings = np.concatenate([cluster_a, cluster_b]).astype(np.float32)
    # Every point has an unknown ("-1") spacegroup label -- that label set
    # alone should degrade to NaN, independently of the well-separated
    # family labels.
    family = np.array([0] * 20 + [1] * 20)
    spacegroup = np.full(40, -1, dtype=np.int64)

    metrics = embedding_quality_metrics(
        embeddings, {"family": family, "spacegroup": spacegroup}
    )

    assert metrics["family_knn_accuracy"] > 0.9
    assert math.isnan(metrics["spacegroup_knn_accuracy"])


# --- helpers ------------------------------------------------------------------


def _open(run_dir, stack="family"):
    return open_stack(run_dir, stack, allow_no_heads=True)


def _run(path, model_kind="supcon", **kw):
    kw.setdefault("families", ("Cubic", "Hexagonal"))
    return write_fake_run(path, model_kind=model_kind, **kw)


# --- run_classification_accuracies -------------------------------------------


def test_run_classification_accuracies_reads_heads_predictions(tmp_path):
    run = _open(_run(tmp_path / "run"))

    accs = run_classification_accuracies(run)

    assert accs["family"] == pytest.approx(1.0)


def test_run_classification_accuracies_empty_when_no_heads(tmp_path):
    run = _open(_run(tmp_path / "run", heads=()))

    assert run_classification_accuracies(run) == {}


# --- collect_run_dirs ---------------------------------------------------------


def test_collect_run_dirs_mixes_and_dedups_run_and_sweep_dirs(tmp_path):
    sweep_dir = tmp_path / "sweep"
    run_a = _run(sweep_dir / "run-a")
    run_b = _run(sweep_dir / "run-b")
    standalone = _run(tmp_path / "standalone_run", model_kind="cgcnn")

    result = collect_run_dirs([sweep_dir, standalone, sweep_dir])

    assert result == sorted({run_a.resolve(), run_b.resolve(), standalone.resolve()})


# --- _resolve_2d_embedding ---------------------------------------------------


def test_resolve_2d_embedding_prefers_viz_head_when_present(tmp_path):
    run = _open(_run(tmp_path / "run", latent_dim=8))

    result = _resolve_2d_embedding(run)

    assert result.shape == (8, 2)
    np.testing.assert_array_equal(result, run.viz_embeddings)


def test_resolve_2d_embedding_passthrough_when_native_is_2d(tmp_path):
    run = _open(_run(tmp_path / "run", latent_dim=2, heads=()))

    result = _resolve_2d_embedding(run)

    np.testing.assert_array_equal(result, run.embeddings["embeddings"])


def test_resolve_2d_embedding_pca_fallback_for_higher_dim(tmp_path):
    run = _open(_run(tmp_path / "run", latent_dim=8, heads=()))

    assert _resolve_2d_embedding(run).shape == (8, 2)


def test_resolve_2d_embedding_none_when_native_is_1d(tmp_path):
    run = _open(_run(tmp_path / "run", latent_dim=1, heads=()))

    assert _resolve_2d_embedding(run) is None


def test_resolve_2d_embedding_ignores_non_2d_viz_embeddings(tmp_path):
    run = _open(_run(tmp_path / "run", latent_dim=8))
    run = dataclasses.replace(run, viz_embeddings=np.zeros((8, 3), dtype=np.float32))

    # Falls through to the PCA fallback on the native (8D) embedding.
    assert _resolve_2d_embedding(run).shape == (8, 2)


# --- benchmark_row / generate_benchmark_table --------------------------------


def test_benchmark_row_leads_with_headline_quality_columns(tmp_path):
    row = benchmark_row(_open(_run(tmp_path / "run", model_kind="cgcnn", latent_dim=8)))

    keys = list(row.keys())
    assert keys[0] == "family"
    assert keys.index("family_2d_silhouette") < keys.index("run_dir")
    assert keys.index("run_dir") < keys.index("family_silhouette")


def test_benchmark_row_has_model_kind_and_metrics(tmp_path):
    row = benchmark_row(_open(_run(tmp_path / "run", model_kind="cgcnn", n=10)))

    assert row["model_kind"] == "cgcnn"
    assert row["stack"] == "family"
    assert row["n_samples"] == 10
    assert "family_silhouette" in row
    assert row["family"] == pytest.approx(1.0)


def test_benchmark_row_expert_has_no_family_metrics(tmp_path):
    run_dir = _run(tmp_path / "run", stacks=("family", "cubic"))
    row = benchmark_row(_open(run_dir, "cubic"))

    assert row["stack"] == "cubic"
    assert "spacegroup" in row
    assert not any(k.startswith("family") for k in row)


def test_generate_benchmark_table_end_to_end_mixed_model_kinds(tmp_path):
    dirs = [
        _run(tmp_path / "supcon_run", model_kind="supcon"),
        _run(tmp_path / "cgcnn_run", model_kind="cgcnn"),
        _run(tmp_path / "supcon_mace_run", model_kind="supcon_mace"),
    ]

    output_csv = tmp_path / "benchmark.csv"
    result_path = generate_benchmark_table(dirs, output_csv)

    assert result_path == output_csv
    with open(output_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    assert {row["model_kind"] for row in rows} == {"supcon", "cgcnn", "supcon_mace"}
    assert "family_silhouette" in rows[0]
    assert "family" in rows[0]


# --- benchmark plots ----------------------------------------------------------


def _rows_for_plots(tmp_path):
    dirs = [
        _run(tmp_path / "supcon_a", model_kind="supcon"),
        _run(tmp_path / "supcon_b", model_kind="supcon"),
        _run(tmp_path / "cgcnn_run", model_kind="cgcnn"),
    ]
    return [benchmark_row(_open(d)) for d in dirs]


def test_plot_classification_accuracy_by_model_kind_writes_png_and_csv(tmp_path):
    rows = _rows_for_plots(tmp_path)
    save_path = tmp_path / "accuracy.png"
    csv_path = tmp_path / "accuracy.csv"

    plot_classification_accuracy_by_model_kind(
        rows, save_path=save_path, csv_path=csv_path
    )

    assert save_path.exists()
    with open(csv_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert {r["model_kind"] for r in csv_rows} == {"supcon", "cgcnn"}
    assert {r["metric"] for r in csv_rows} == {"family"}


def test_plot_classification_accuracy_by_model_kind_skips_when_no_accuracy(tmp_path):
    rows = [benchmark_row(_open(_run(tmp_path / "run", heads=())))]
    save_path = tmp_path / "accuracy.png"

    plot_classification_accuracy_by_model_kind(rows, save_path=save_path)

    assert not save_path.exists()


def test_plot_2d_quality_by_model_kind_writes_png_and_csv(tmp_path):
    rows = _rows_for_plots(tmp_path)
    save_path = tmp_path / "quality2d.png"
    csv_path = tmp_path / "quality2d.csv"

    plot_2d_quality_by_model_kind(rows, save_path=save_path, csv_path=csv_path)

    assert save_path.exists()
    with open(csv_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert "family_2d" in {r["label_set"] for r in csv_rows}
    assert {r["model_kind"] for r in csv_rows} == {"supcon", "cgcnn"}


def test_generate_benchmark_plots_writes_both_pngs(tmp_path):
    rows = _rows_for_plots(tmp_path)
    output_dir = tmp_path / "benchmark_plots"

    result_dir = generate_benchmark_plots(rows, output_dir, write_data_files=True)

    assert result_dir == output_dir
    assert (output_dir / "accuracy_by_model_kind.png").exists()
    assert (output_dir / "accuracy_by_model_kind.csv").exists()
    assert (output_dir / "embedding_quality_2d_by_model_kind.png").exists()
    assert (output_dir / "embedding_quality_2d_by_model_kind.csv").exists()


def test_generate_benchmark_table_plot_false_skips_plots(tmp_path):
    output_csv = tmp_path / "benchmark.csv"

    generate_benchmark_table([_run(tmp_path / "supcon_run")], output_csv)

    assert not (tmp_path / "benchmark_plots").exists()


# --- per-(run, stack) rows and plots ------------------------------------------


def _sweep(tmp_path):
    sweep = tmp_path / "s"
    write_fake_run(
        sweep / "a", stacks=("family", "cubic"), families=("Cubic", "Hexagonal")
    )
    write_fake_run(
        sweep / "b",
        stacks=("family", "cubic"),
        families=("Cubic", "Hexagonal"),
        learning_rate=2e-3,
    )
    return sweep


def test_table_has_one_row_per_run_and_stack(tmp_path):
    out = generate_benchmark_table([_sweep(tmp_path)], tmp_path / "t.csv")
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 4
    assert {r["stack"] for r in rows} == {"family", "cubic"}
    family = [r for r in rows if r["stack"] == "family"]
    cubic = [r for r in rows if r["stack"] == "cubic"]
    assert all(r["family"] != "" for r in family)
    assert all(r["spacegroup"] != "" for r in cubic)
    assert all(r["family"] == "" for r in cubic)  # experts have no family metrics
    assert all(r["model_kind"] == "supcon" for r in rows)
    assert "wall_clock_seconds" not in rows[0]


def test_default_hyperparams_are_the_new_dotted_paths(tmp_path):
    out = generate_benchmark_table([_sweep(tmp_path)], tmp_path / "t.csv")
    header = next(csv.reader(open(out)))
    for key in (
        "model.latent_dim",
        "model.encoder_hidden_dim",
        "model.body_train.learning_rate",
    ):
        assert key in header


def test_collect_run_dirs_accepts_runs_and_sweeps(tmp_path):
    sweep = _sweep(tmp_path)
    assert collect_run_dirs([sweep]) == sorted(p.resolve() for p in sweep.iterdir())
    assert collect_run_dirs([sweep / "a", sweep]) == collect_run_dirs([sweep])


def test_plots_are_written_per_stack(tmp_path):
    generate_benchmark_table([_sweep(tmp_path)], tmp_path / "t.csv", plot=True)
    plots = tmp_path / "benchmark_plots"
    assert any((plots / "family").iterdir())
    assert any((plots / "cubic").iterdir())


def test_heads_name_is_forwarded(tmp_path):
    sweep = tmp_path / "s"
    write_fake_run(sweep / "a", heads=("x", "y"))
    with pytest.raises(ValueError, match="several heads"):
        generate_benchmark_table([sweep], tmp_path / "t.csv")
    generate_benchmark_table([sweep], tmp_path / "t.csv", heads_name="x")
