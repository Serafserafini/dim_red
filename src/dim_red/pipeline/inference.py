"""
Applies an already-trained model -- as saved by
``dim_red.pipeline.single_run.run_single`` -- to new structures, and plots
the result in that model's latent space alongside the original dataset it
was trained on.

A completed run directory always has, written together: ``config.yaml``
(architecture + SOAP settings), ``dataset.extxyz`` (the exact structures used
to fit/standardize the model, same order as the SOAP features it saw),
``model_params.msgpack`` (trained weights) and ``embeddings.npz`` (the
original dataset's latent embeddings/labels, plus the family/spacegroup
class vocabularies when a VAE's/Autoencoder's auxiliary heads were active).
``load_trained_run`` reads all four: it recomputes SOAP on ``dataset.extxyz``
to recover the exact species list and mean/std standardization statistics
implied by training (neither is saved directly -- only the already-
standardized ``features`` are, in ``embeddings.npz``), then rebuilds the
model architecture and loads its trained weights. ``encode_structures`` then
featurizes/standardizes new structures through that *same* pipeline -- not
each their own -- so they land in a latent space that's actually comparable
to the training set's, and ``plot_applied_structures`` overlays the two
(projecting both through one shared UMAP fit when the latent space isn't
already 2D, so old and new points don't end up in two unrelated coordinate
systems).
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
from dim_red.soap import compute_soap
from dim_red.utils import apply_standardization, fit_standardization

logger = logging.getLogger("dim_red.pipeline")

# Recomputed-vs-saved standardized-feature mismatch above this is treated as
# a sign config.yaml/dataset.extxyz don't actually match what produced this
# run (stale/hand-edited files) -- logged as a warning, not a hard failure,
# since a small mismatch is expected from floating-point non-associativity.
_STANDARDIZATION_MISMATCH_TOL = 1e-4

_REQUIRED_RUN_FILES = (
    "config.yaml",
    "dataset.extxyz",
    "model_params.msgpack",
    "embeddings.npz",
)


@dataclass
class LoadedRun:
    """Everything needed to apply a trained run's model to new structures."""

    run_dir: Path
    config: RunConfig
    model: Any  # dim_red.vae.model.VAE | autoencoder.model.Autoencoder | supcon.model.SupConEncoder
    species: List[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    embeddings: Dict[
        str, np.ndarray
    ]  # this run's own embeddings.npz (original dataset)


def _resolve_species(config: RunConfig, atoms_list: List[Atoms]) -> List[str]:
    """The species list SOAP was actually computed against for this run --
    ``config.soap.species`` if it was set explicitly, otherwise
    auto-detected from ``atoms_list`` (``dataset.extxyz``, the exact training
    atoms), mirroring ``dim_red.pipeline.dataset_cache._compute_soap_and_standardize``.
    """
    if config.soap.species is not None:
        return list(config.soap.species)
    species = set()
    for a in atoms_list:
        species.update(a.get_chemical_symbols())
    return sorted(species)


def _raw_soap_matrix(
    atoms_list: List[Atoms], soap_kwargs: Dict[str, Any], species: List[str]
) -> np.ndarray:
    kwargs = dict(soap_kwargs)
    kwargs["species"] = species
    kwargs["average"] = "outer"
    vectors = compute_soap(atoms_list, **kwargs)
    return np.asarray(vectors).reshape(len(atoms_list), -1)


def _build_model(config: RunConfig, input_dim: int, embeddings: Dict[str, np.ndarray]):
    """Reconstruct a run's model architecture (with freshly-initialized,
    soon-to-be-overwritten weights) from its ``RunConfig`` -- mirrors the
    construction in ``dim_red.pipeline.single_run.run_single``. Family/
    spacegroup classifier-head sizes (vae/autoencoder only) come from
    ``embeddings['family_classes']``/``['spacegroup_classes']``, since
    ``RunConfig`` itself doesn't carry the resolved class counts.
    """
    if config.model_kind == "supcon":
        from dim_red.supcon.model import SupConEncoder

        return SupConEncoder(
            input_dim=input_dim,
            encoder_hidden_dim=config.vae.encoder_hidden_dim,
            latent_dim=config.vae.latent_dim,
            seed=config.seed,
        )

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

    if config.model_kind == "vae":
        from dim_red.vae.model import VAE as model_cls
    else:
        from dim_red.autoencoder.model import Autoencoder as model_cls

    return model_cls(
        input_dim=input_dim,
        encoder_hidden_dim=config.vae.encoder_hidden_dim,
        decoder_hidden_dim=config.vae.decoder_hidden_dim,
        latent_dim=config.vae.latent_dim,
        n_family_classes=n_family_classes,
        n_spacegroup_classes=n_spacegroup_classes,
        head_hidden_dim=config.aux_heads.head_hidden_dim,
        mirror=config.vae.mirror,
        seed=config.seed,
    )


def load_trained_run(run_dir: Union[str, Path]) -> LoadedRun:
    """Load a completed run directory's trained model, ready to encode new structures.

    Args:
        run_dir: Path to a run directory written by
            ``dim_red.pipeline.single_run.run_single`` -- must contain
            ``config.yaml``, ``dataset.extxyz``, ``model_params.msgpack`` and
            ``embeddings.npz`` (always written together by that function).

    Returns:
        A ``LoadedRun`` ready for ``encode_structures``/``plot_applied_in_latent_space``.

    Raises:
        FileNotFoundError: If any of the four expected files is missing.
    """
    run_dir = Path(run_dir)
    missing = [f for f in _REQUIRED_RUN_FILES if not (run_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{run_dir} is missing {missing} -- not a completed "
            "dim_red.pipeline.single_run.run_single run directory"
        )

    config = load_run_config(run_dir / "config.yaml")
    with np.load(run_dir / "embeddings.npz") as npz:
        embeddings = dict(npz.items())

    train_atoms = read_atoms(str(run_dir / "dataset.extxyz"), index=":")
    species = _resolve_species(config, train_atoms)
    soap_kwargs = config.soap.as_kwargs()
    X_train_raw = _raw_soap_matrix(train_atoms, soap_kwargs, species)
    mean, std = fit_standardization(X_train_raw)

    if "features" in embeddings:
        saved = embeddings["features"]
        reproduced = apply_standardization(X_train_raw, mean, std)
        if reproduced.shape != saved.shape:
            logger.warning(
                "Recomputed SOAP features for %s's dataset.extxyz have shape %s, "
                "but embeddings.npz['features'] has shape %s -- config.yaml may "
                "not match what actually produced this run.",
                run_dir,
                reproduced.shape,
                saved.shape,
            )
        else:
            max_diff = float(np.max(np.abs(reproduced - saved)))
            if max_diff > _STANDARDIZATION_MISMATCH_TOL:
                logger.warning(
                    "Recomputed SOAP features for %s's dataset.extxyz differ from "
                    "embeddings.npz['features'] by up to %.3g -- config.yaml's SOAP "
                    "settings may not match what actually produced this run (new "
                    "structures would then be featurized inconsistently with the "
                    "trained model).",
                    run_dir,
                    max_diff,
                )

    model = _build_model(config, X_train_raw.shape[1], embeddings)
    with open(run_dir / "model_params.msgpack", "rb") as f:
        model.params = serialization.from_bytes(model.params, f.read())

    logger.info(
        "Loaded %s model from %s (%d species, %d-dim SOAP input, latent_dim=%d)",
        config.model_kind,
        run_dir,
        len(species),
        X_train_raw.shape[1],
        config.vae.latent_dim,
    )
    return LoadedRun(
        run_dir=run_dir,
        config=config,
        model=model,
        species=species,
        feature_mean=mean,
        feature_std=std,
        embeddings=embeddings,
    )


def encode_structures(loaded: LoadedRun, atoms_list: List[Atoms]) -> np.ndarray:
    """Encode new structures into a ``LoadedRun``'s latent space.

    Args:
        loaded: Output of ``load_trained_run``.
        atoms_list: New ASE ``Atoms`` to encode. Every chemical species
            present must be among ``loaded.species`` (the species SOAP was
            computed against during training) -- otherwise SOAP computation
            raises.

    Returns:
        Latent coordinates, shape ``(len(atoms_list), latent_dim)``.

    Raises:
        ValueError: If ``atoms_list`` is empty.
    """
    if not atoms_list:
        raise ValueError("atoms_list is empty -- nothing to encode")
    X_raw = _raw_soap_matrix(atoms_list, loaded.config.soap.as_kwargs(), loaded.species)
    X_std = apply_standardization(X_raw, loaded.feature_mean, loaded.feature_std)
    is_vae = loaded.config.model_kind == "vae"
    z = loaded.model.encode(X_std)[0] if is_vae else loaded.model.encode(X_std)
    return np.asarray(z)


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
