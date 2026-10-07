"""
Applies an already-trained cgcnn model -- as saved by
``dim_red.pipeline.single_run.run_single`` -- to new structures, and plots
the result in that model's latent space alongside the original dataset it
was trained on. supcon/supcon_mace stacks are applied through
``dim_red.pipeline.full_stack`` instead (``dimred-apply`` dispatches on the
run directory's layout).

A completed run directory always has, written together: ``config.yaml``
(architecture + graph settings), ``dataset.extxyz`` (the exact structures
used for training), ``model_params.msgpack`` (trained weights) and
``embeddings.npz`` (the original dataset's latent embeddings/labels, and the
family/spacegroup class vocabularies when the classification heads were
active). ``load_trained_run`` rebuilds the model architecture from
``config.yaml``/``embeddings.npz`` and loads its trained weights;
``encode_structures`` then builds graphs for new structures with the run's
own graph settings, so they land in a latent space comparable to the
training set's, and ``plot_applied_structures`` overlays the two (projecting
both through one shared UMAP fit when the latent space isn't already 2D, so
old and new points don't end up in two unrelated coordinate systems).

``load_run_embeddings`` is a lighter-weight counterpart for callers that
only need ``config.yaml``/``embeddings.npz`` and never the reconstructed
model itself -- e.g. ``dim_red.pipeline.tail_training.train_tail``, which
trains a tail directly on a frozen body's already-saved representations.
It requires only those two files (not the full four).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from ase import Atoms
from ase.io import read as read_atoms
from flax import serialization

from dim_red.analysis.plotting import plot_applied_structures
from dim_red.pipeline.compare import LatentUmapParams, make_umap
from dim_red.pipeline.config import RunConfig, load_run_config

logger = logging.getLogger("dim_red.pipeline")

_REQUIRED_RUN_FILES = (
    "config.yaml",
    "dataset.extxyz",
    "model_params.msgpack",
    "embeddings.npz",
)


@dataclass
class LoadedRun:
    """Everything needed to apply a trained cgcnn run's model to new structures."""

    run_dir: Path
    config: RunConfig
    model: Any  # cgcnn.model.CGCNNEncoder
    embeddings: Dict[
        str, np.ndarray
    ]  # this run's own embeddings.npz (original dataset)


_EMBEDDINGS_REQUIRED_RUN_FILES = ("config.yaml", "embeddings.npz")


@dataclass
class RunEmbeddings:
    """Lightweight counterpart to ``LoadedRun`` for callers that only need a
    completed run's config and ``embeddings.npz`` contents -- e.g.
    ``dim_red.pipeline.tail_training.train_tail``, which trains a tail on
    representations ``dim_red.pipeline.single_run.run_single`` already
    computed and saved, and never needs the model ``LoadedRun``/
    ``load_trained_run`` reconstructs. Use ``load_trained_run`` instead when
    the reconstructed model itself is actually needed (e.g. to encode new
    structures).
    """

    run_dir: Path
    config: RunConfig
    embeddings: Dict[str, np.ndarray]


def load_run_embeddings(run_dir: Union[str, Path]) -> RunEmbeddings:
    """Load just a completed run's config and ``embeddings.npz`` -- the
    minimal subset ``dim_red.pipeline.tail_training.train_tail`` actually
    needs (see ``RunEmbeddings``). Unlike ``load_trained_run``, this never
    reconstructs the model, and ``dataset.extxyz``/``model_params.msgpack``
    aren't required to exist at all.

    Args:
        run_dir: Path to a run directory written by
            ``dim_red.pipeline.single_run.run_single`` -- must contain at
            least ``config.yaml`` and ``embeddings.npz``.

    Returns:
        A ``RunEmbeddings``.

    Raises:
        FileNotFoundError: If either expected file is missing.
    """
    run_dir = Path(run_dir)
    missing = [f for f in _EMBEDDINGS_REQUIRED_RUN_FILES if not (run_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{run_dir} is missing {missing} -- not a completed "
            "dim_red.pipeline.single_run.run_single run directory"
        )
    config = load_run_config(run_dir / "config.yaml")
    with np.load(run_dir / "embeddings.npz") as npz:
        embeddings = dict(npz.items())
    return RunEmbeddings(run_dir=run_dir, config=config, embeddings=embeddings)


def _build_model(config: RunConfig, embeddings: Dict[str, np.ndarray]):
    """Reconstruct a cgcnn run's model architecture (with freshly-initialized,
    soon-to-be-overwritten weights) from its ``RunConfig`` -- mirrors the
    construction in ``dim_red.pipeline.single_run.run_single``. Family/
    spacegroup classifier-head sizes come from
    ``embeddings['family_classes']``/``['spacegroup_classes']``, since
    ``RunConfig`` itself doesn't carry the resolved class counts.
    """
    from dim_red.cgcnn.model import CGCNNEncoder

    aux_mode = config.aux_heads.mode
    use_family = aux_mode != "none"
    use_spacegroup = aux_mode == "family_and_spacegroup"
    n_family_classes = (
        len(embeddings["family_classes"])
        if use_family and "family_classes" in embeddings
        else None
    )
    n_spacegroup_classes = (
        len(embeddings["spacegroup_classes"])
        if use_spacegroup and "spacegroup_classes" in embeddings
        else None
    )
    return CGCNNEncoder(
        atom_fea_len=config.graph.atom_fea_len,
        n_conv=config.graph.n_conv,
        h_fea_len=config.graph.h_fea_len,
        n_h=config.graph.n_h,
        latent_dim=config.encoder.latent_dim,
        n_gaussian=config.graph.n_gaussian,
        max_species=config.graph.max_species,
        n_family_classes=n_family_classes,
        n_spacegroup_classes=n_spacegroup_classes,
        head_hidden_dim=config.aux_heads.head_hidden_dim,
        seed=config.seed,
    )


def load_trained_run(run_dir: Union[str, Path]) -> LoadedRun:
    """Load a completed cgcnn run directory's trained model, ready to encode
    new structures.

    Args:
        run_dir: Path to a run directory written by
            ``dim_red.pipeline.single_run.run_single`` -- must contain
            ``config.yaml``, ``dataset.extxyz``, ``model_params.msgpack`` and
            ``embeddings.npz`` (always written together by that function).

    Returns:
        A ``LoadedRun`` ready for ``encode_structures``/``plot_applied_in_latent_space``.

    Raises:
        FileNotFoundError: If any of the four expected files is missing.
        ValueError: If the run's ``model_kind`` is not ``"cgcnn"``.
    """
    run_dir = Path(run_dir)
    missing = [f for f in _REQUIRED_RUN_FILES if not (run_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{run_dir} is missing {missing} -- not a completed "
            "dim_red.pipeline.single_run.run_single run directory"
        )

    config = load_run_config(run_dir / "config.yaml")
    if config.model_kind != "cgcnn":
        raise ValueError(
            f"{run_dir} is a model_kind={config.model_kind!r} run -- "
            "load_trained_run only loads cgcnn runs; supcon/supcon_mace stacks "
            "are applied through dim_red.pipeline.full_stack.FullStack"
        )
    with np.load(run_dir / "embeddings.npz") as npz:
        embeddings = dict(npz.items())

    # No species vocabulary (the body's nn.Embed is keyed by a per-structure
    # LOCAL species slot, not a real chemical species -- see
    # dim_red.cgcnn.graph) and no standardization stats (Gaussian-expanded
    # bond distances are already bounded to [0, 1] by construction) --
    # dataset.extxyz doesn't even need to be read here.
    model = _build_model(config, embeddings)
    with open(run_dir / "model_params.msgpack", "rb") as f:
        model.params = serialization.from_bytes(model.params, f.read())

    logger.info(
        "Loaded cgcnn model from %s (latent_dim=%d)",
        run_dir,
        config.encoder.latent_dim,
    )
    return LoadedRun(
        run_dir=run_dir,
        config=config,
        model=model,
        embeddings=embeddings,
    )


def encode_structures(loaded: LoadedRun, atoms_list: List[Atoms]) -> np.ndarray:
    """Encode new structures into a ``LoadedRun``'s latent space.

    Args:
        loaded: Output of ``load_trained_run``.
        atoms_list: New ASE ``Atoms`` to encode.

    Returns:
        Latent coordinates, shape ``(len(atoms_list), latent_dim)``.

    Raises:
        ValueError: If ``atoms_list`` is empty.
    """
    if not atoms_list:
        raise ValueError("atoms_list is empty -- nothing to encode")

    from dim_red.cgcnn.graph import atoms_list_to_graph_arrays

    # max_atoms is resolved fresh here, across just atoms_list -- it is NOT
    # required to match whatever max_atoms training used (see
    # dim_red.cgcnn.graph.atoms_list_to_graph_arrays's docstring: no trained
    # parameter depends on max_atoms).
    graph_batch = atoms_list_to_graph_arrays(
        atoms_list, **loaded.config.graph.graph_kwargs()
    )
    return np.asarray(loaded.model.encode(graph_batch))


def plot_applied_in_latent_space(
    loaded: LoadedRun,
    new_z: np.ndarray,
    new_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    umap_params: Optional[LatentUmapParams] = None,
) -> None:
    """Plot new structures' latent coordinates alongside a run's original
    dataset (``loaded.embeddings['embeddings']``/``['labels']``).

    When the latent space isn't already 2D, both the original embeddings and
    ``new_z`` are projected into one shared 2D space by fitting
    ``dim_red.umap.UMAP`` on the original embeddings and then *transforming*
    ``new_z`` through that same fit -- not a second, independent UMAP fit,
    which would place old and new points in unrelated coordinate systems.

    Args:
        loaded: Output of ``load_trained_run``.
        new_z: Latent coordinates from ``encode_structures``, shape
            ``(m, latent_dim)`` -- ``latent_dim`` must match
            ``loaded.embeddings['embeddings']``'s.
        new_labels: See ``dim_red.analysis.plotting.plot_applied_structures``.
        title: Plot title (default: ``"<run name>'s latent space"``, with a
            ``" (UMAP)"`` suffix when projected).
        save_path: If provided, saves the plot to this filepath; otherwise
            the figure is shown (interactively, or captured inline in a notebook).
        umap_params: UMAP hyperparameters for the projection, used only when
            the latent space isn't 2D. ``None`` uses ``umap-learn``'s own
            defaults.
    """
    original_z = loaded.embeddings["embeddings"]
    original_labels = loaded.embeddings["labels"].tolist()

    if original_z.shape[1] == 2 and new_z.shape[1] == 2:
        plot_original, plot_new, projected = original_z, new_z, False
    else:
        umap = make_umap(n_components=2, umap_params=umap_params)
        plot_original = umap.fit_transform(original_z)
        plot_new = umap.transform(new_z)
        projected = True

    default_title = f"{loaded.run_dir.name}'s latent space" + (
        " (UMAP)" if projected else ""
    )
    plot_applied_structures(
        plot_original,
        original_labels,
        plot_new,
        new_labels=new_labels,
        title=title or default_title,
        save_path=str(save_path) if save_path else None,
    )


def apply_model_to_structures(
    run_dir: Union[str, Path],
    structures_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    label_field: Optional[str] = None,
    umap_params: Optional[LatentUmapParams] = None,
) -> Path:
    """End-to-end: load a trained run, encode every structure in
    ``structures_path``, and write both the resulting latent coordinates
    (``<stem>_embeddings.npz``) and a latent-space plot overlaying them on
    the run's original dataset (``<stem>_latent_space.png``) to
    ``output_dir``.

    Args:
        run_dir: Path to a completed run directory (see ``load_trained_run``).
        structures_path: Path to an extended-XYZ file with the structures to
            apply the model to.
        output_dir: Where to write the outputs (default: ``<run_dir>/applied``).
        label_field: An ``atoms.info`` key to color/label the new structures
            by (e.g. ``"family"`` for pyxtal-generated structures that carry
            one). ``None`` (default) groups every new structure under a
            single "Applied structure" legend entry.
        umap_params: UMAP hyperparameters for the latent-space plot, used
            only when the run's latent space isn't already 2D.

    Returns:
        ``output_dir`` actually used.

    Raises:
        ValueError: If ``structures_path`` contains no structures.
    """
    run_dir = Path(run_dir)
    loaded = load_trained_run(run_dir)

    new_atoms = read_atoms(str(structures_path), index=":")
    if not new_atoms:
        raise ValueError(f"No structures found in {structures_path}")
    new_z = encode_structures(loaded, new_atoms)

    stem = Path(structures_path).stem
    output_dir = Path(output_dir) if output_dir else run_dir / "applied"
    output_dir.mkdir(parents=True, exist_ok=True)

    material_ids = [
        a.info.get("material_id", f"{stem}-{i}") for i, a in enumerate(new_atoms)
    ]
    new_labels = (
        [str(a.info.get(label_field, "Applied structure")) for a in new_atoms]
        if label_field
        else None
    )

    npz_path = output_dir / f"{stem}_embeddings.npz"
    payload = dict(embeddings=new_z, material_ids=np.array(material_ids))
    if new_labels is not None:
        payload["labels"] = np.array(new_labels)
    np.savez(npz_path, **payload)
    logger.info(
        "Encoded %d structures from %s; saved embeddings to %s",
        len(new_atoms),
        structures_path,
        npz_path,
    )

    plot_path = output_dir / f"{stem}_latent_space.png"
    plot_applied_in_latent_space(
        loaded,
        new_z,
        new_labels=new_labels,
        title=f"{stem} applied to {run_dir.name}",
        save_path=plot_path,
        umap_params=umap_params,
    )
    logger.info("Saved applied-structures latent-space plot to %s", plot_path)

    return output_dir
