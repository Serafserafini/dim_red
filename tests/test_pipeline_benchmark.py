"""
Unit tests for the cross-architecture benchmark table
(``dim_red.pipeline.benchmark``): given already-completed run folders (the
same artifacts ``run_single``/``train_tail`` write), it should assemble one
CSV comparing them on standardized, dimensionality-agnostic metrics without
needing any real training to run again.
"""

import csv
import math
from datetime import datetime, timedelta

import numpy as np
import pytest
import yaml

from dim_red.analysis.metrics import embedding_quality_metrics
from dim_red.pipeline.benchmark import (
    _resolve_2d_embedding,
    benchmark_row,
    collect_run_dirs,
    generate_benchmark_table,
    run_classification_accuracies,
    run_wall_clock_seconds,
)
from dim_red.pipeline.compare import load_run
from dim_red.pipeline.config import (
    AuxHeadsConfig,
    FetchConfig,
    RunConfig,
    SoapConfig,
    TrainSettings,
    VAEArchConfig,
    run_config_to_dict,
)

pytest.importorskip("sklearn")


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


# --- helpers to fabricate on-disk run/tail directories ---------------------


def _write_loss_history(path, epochs=3):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            writer.writerow(
                {"epoch": epoch, "train_loss": 1.0 / epoch, "val_loss": 1.2 / epoch}
            )


def _write_run(
    run_dir,
    model_kind="vae",
    n=8,
    n_classes=2,
    latent_dim=2,
    with_aux=False,
):
    run_dir.mkdir(parents=True)
    aux_heads = (
        AuxHeadsConfig(mode="family_only", lambda_family=1.0, head_hidden_dim=4)
        if with_aux
        else AuxHeadsConfig()
    )
    config = RunConfig(
        model_kind=model_kind,
        fetch=FetchConfig(crystal_systems=["cubic"], limit_per_system=8),
        soap=SoapConfig(r_cut=3.0, n_max=2, l_max=2),
        vae=VAEArchConfig(encoder_hidden_dim=[4], latent_dim=latent_dim),
        train=TrainSettings(epochs=3, batch_size=4, val_ratio=0.25),
        aux_heads=aux_heads,
        seed=0,
        output_dir=str(run_dir.parent),
        name=run_dir.name,
    )
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(run_config_to_dict(config), f, sort_keys=False)
    _write_loss_history(run_dir / "loss_history.csv")

    rng = np.random.default_rng(0)
    class_labels = np.arange(n) % n_classes
    centers = rng.normal(scale=10.0, size=(n_classes, latent_dim))
    embeddings = centers[class_labels] + rng.normal(scale=0.1, size=(n, latent_dim))
    payload = dict(
        embeddings=embeddings.astype(np.float32),
        labels=class_labels.astype(np.int64),
        material_ids=np.array([f"mp-{i}" for i in range(n)]),
        spacegroups=np.full(n, -1, dtype=np.int64),
        split=np.array(["train"] * (n - 1) + ["val"]),
    )
    if with_aux:
        family_probs = np.eye(n_classes, dtype=np.float32)[class_labels]
        payload["family_probs"] = family_probs
        payload["family_classes"] = np.arange(n_classes, dtype=np.int64)
    np.savez(run_dir / "embeddings.npz", **payload)
    return run_dir


def _write_tail_predictions(run_dir, n=8, n_classes=2):
    tail_dir = run_dir / "tails" / "classification"
    tail_dir.mkdir(parents=True)
    class_labels = np.arange(n) % n_classes
    family_probs = np.eye(n_classes, dtype=np.float32)[class_labels]
    np.savez(
        tail_dir / "tail_predictions.npz",
        material_ids=np.array([f"mp-{i}" for i in range(n)]),
        split=np.array(["train"] * (n - 1) + ["val"]),
        family_probs=family_probs,
        family_classes=np.arange(n_classes, dtype=np.int64),
        labels=class_labels.astype(np.int64),
    )
    return tail_dir


def _write_visualization_tail(run_dir, n=8, n_classes=2, viz_dim=2, separated=True):
    tail_dir = run_dir / "tails" / "visualization"
    tail_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    class_labels = np.arange(n) % n_classes
    if separated:
        centers = rng.normal(scale=10.0, size=(n_classes, viz_dim))
        embeddings = centers[class_labels] + rng.normal(scale=0.1, size=(n, viz_dim))
    else:
        embeddings = rng.normal(size=(n, viz_dim))
    np.savez(tail_dir / "tail_embeddings.npz", embeddings=embeddings.astype(np.float32))
    return tail_dir


def _write_log(path, timestamps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ts in timestamps:
            f.write(f"{ts.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} [INFO] tick\n")


# --- run_classification_accuracies -----------------------------------------


def test_run_classification_accuracies_reads_embeddings_npz(tmp_path):
    run_dir = _write_run(tmp_path / "run", model_kind="vae", with_aux=True)
    run = load_run(run_dir)

    accs = run_classification_accuracies(run)

    assert accs["family"] == pytest.approx(1.0)


def test_run_classification_accuracies_falls_back_to_tail_predictions(tmp_path):
    run_dir = _write_run(tmp_path / "run", model_kind="supcon", with_aux=False)
    _write_tail_predictions(run_dir)
    run = load_run(run_dir)

    accs = run_classification_accuracies(run)

    assert accs["family"] == pytest.approx(1.0)


def test_run_classification_accuracies_empty_when_neither_present(tmp_path):
    run_dir = _write_run(tmp_path / "run", model_kind="supcon", with_aux=False)
    run = load_run(run_dir)

    assert run_classification_accuracies(run) == {}


# --- run_wall_clock_seconds --------------------------------------------------


def test_run_wall_clock_seconds_from_log_span(tmp_path):
    run_dir = tmp_path / "run"
    start = datetime(2026, 1, 1, 12, 0, 0)
    _write_log(run_dir / "run.log", [start, start + timedelta(seconds=30)])

    assert run_wall_clock_seconds(run_dir) == pytest.approx(30.0, abs=0.01)


def test_run_wall_clock_seconds_subtracts_nested_tail_span(tmp_path):
    run_dir = tmp_path / "run"
    start = datetime(2026, 1, 1, 12, 0, 0)
    _write_log(run_dir / "run.log", [start, start + timedelta(seconds=100)])
    _write_log(
        run_dir / "tails" / "classification" / "run.log",
        [start + timedelta(seconds=60), start + timedelta(seconds=100)],
    )

    assert run_wall_clock_seconds(run_dir) == pytest.approx(60.0, abs=0.01)


def test_run_wall_clock_seconds_none_when_log_missing(tmp_path):
    assert run_wall_clock_seconds(tmp_path / "nonexistent_run") is None


# --- collect_run_dirs --------------------------------------------------------


def test_collect_run_dirs_mixes_and_dedups_run_and_sweep_dirs(tmp_path):
    sweep_dir = tmp_path / "sweep"
    run_a = _write_run(sweep_dir / "run-a", model_kind="vae")
    run_b = _write_run(sweep_dir / "run-b", model_kind="vae")
    standalone = _write_run(
        tmp_path / "standalone_run", model_kind="cgcnn", with_aux=True
    )

    result = collect_run_dirs([sweep_dir, standalone, sweep_dir])

    assert result == sorted({run_a.resolve(), run_b.resolve(), standalone.resolve()})


# --- _resolve_2d_embedding ---------------------------------------------------


def test_resolve_2d_embedding_prefers_visualization_tail_when_present(tmp_path):
    run_dir = _write_run(tmp_path / "run", model_kind="supcon", latent_dim=8)
    _write_visualization_tail(run_dir, viz_dim=2)
    run = load_run(run_dir)

    result = _resolve_2d_embedding(run)

    assert result.shape == (8, 2)
    tail_embeddings = np.load(
        run_dir / "tails" / "visualization" / "tail_embeddings.npz"
    )["embeddings"]
    np.testing.assert_array_equal(result, tail_embeddings)


def test_resolve_2d_embedding_passthrough_when_native_is_2d(tmp_path):
    run_dir = _write_run(tmp_path / "run", model_kind="vae", latent_dim=2)
    run = load_run(run_dir)

    result = _resolve_2d_embedding(run)

    np.testing.assert_array_equal(result, run.embeddings["embeddings"])


def test_resolve_2d_embedding_pca_fallback_for_higher_dim(tmp_path):
    run_dir = _write_run(tmp_path / "run", model_kind="vae", latent_dim=8)
    run = load_run(run_dir)

    result = _resolve_2d_embedding(run)

    assert result.shape == (8, 2)


def test_resolve_2d_embedding_none_when_native_is_1d(tmp_path):
    run_dir = _write_run(tmp_path / "run", model_kind="vae", latent_dim=1)
    run = load_run(run_dir)

    assert _resolve_2d_embedding(run) is None


def test_resolve_2d_embedding_ignores_non_2d_visualization_tail(tmp_path):
    run_dir = _write_run(tmp_path / "run", model_kind="supcon", latent_dim=8)
    _write_visualization_tail(run_dir, viz_dim=3)
    run = load_run(run_dir)

    result = _resolve_2d_embedding(run)

    # Falls through to the PCA fallback on the native (8D) embedding, since
    # the tail's own embeddings are 3D, not the requested 2D view.
    assert result.shape == (8, 2)


# --- benchmark_row / generate_benchmark_table --------------------------------


def test_benchmark_row_leads_with_headline_quality_columns(tmp_path):
    run_dir = _write_run(
        tmp_path / "run", model_kind="cgcnn", with_aux=True, latent_dim=8
    )
    row = benchmark_row(load_run(run_dir))

    keys = list(row.keys())
    assert keys[0] == "family"
    assert keys.index("family_2d_silhouette") < keys.index("run_dir")
    assert keys.index("run_dir") < keys.index("family_silhouette")


def test_benchmark_row_has_model_kind_and_metrics(tmp_path):
    run_dir = _write_run(tmp_path / "run", model_kind="cgcnn", with_aux=True, n=10)
    row = benchmark_row(load_run(run_dir))

    assert row["model_kind"] == "cgcnn"
    assert row["n_samples"] == 10
    assert "family_silhouette" in row
    assert row["family"] == pytest.approx(1.0)


def test_generate_benchmark_table_end_to_end_mixed_model_kinds(tmp_path):
    vae_dir = _write_run(tmp_path / "vae_run", model_kind="vae", with_aux=True)
    cgcnn_dir = _write_run(tmp_path / "cgcnn_run", model_kind="cgcnn", with_aux=True)
    supcon_dir = _write_run(
        tmp_path / "supcon_run", model_kind="supcon", with_aux=False
    )
    _write_tail_predictions(supcon_dir)

    output_csv = tmp_path / "benchmark.csv"
    result_path = generate_benchmark_table([vae_dir, cgcnn_dir, supcon_dir], output_csv)

    assert result_path == output_csv
    with open(output_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    model_kinds = {row["model_kind"] for row in rows}
    assert model_kinds == {"vae", "cgcnn", "supcon"}
    fieldnames = rows[0].keys()
    assert "family_silhouette" in fieldnames
    assert "family" in fieldnames
    # supcon has no built-in aux heads in embeddings.npz -- its "family"
    # accuracy column comes from the tail_predictions.npz fallback, and is
    # still present (not missing) in the shared column set.
    supcon_row = next(r for r in rows if r["model_kind"] == "supcon")
    assert float(supcon_row["family"]) == pytest.approx(1.0)
