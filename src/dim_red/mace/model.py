"""
Frozen, pretrained MACE feature extractor (wraps ``mace_jax``, an official
JAX/Flax port of MACE -- https://github.com/ACEsuit/mace-jax).

Unlike every other body in this codebase (``vae``/``autoencoder``/``supcon``/
``cgcnn``), ``MaceEncoder`` never trains: it loads a pretrained foundation
model's weights (converted from a Torch checkpoint via ``mace_jax``'s own
``mace-jax-from-torch`` CLI, entirely outside this codebase and this
package's runtime -- see ``src/dim_red/mace/CLAUDE.md``) and only ever runs a
forward pass. There is deliberately no ``training.py`` in this package.

============================================================================
VERIFICATION STATUS -- READ BEFORE RELYING ON THIS IN PRODUCTION
============================================================================
This was written against ``mace_jax``'s public GitHub source as read
remotely (no ``mace_jax`` install was available while writing this), because
the package has no documented low-level Python API for one-off ``ase.Atoms``
inference (no ``ASE`` ``Calculator``, no ``atomic_data_from_ase`` helper).
Confirmed directly from ``mace_jax/modules/models.py``'s ``MACE.__call__``:

- Its return dict has a ``"node_feats"`` key (per-atom features, concatenated
  across every interaction layer when ``compute_node_feats=True``).
- Input is a single ``dict[str, jnp.ndarray]`` graph-batch, with at least
  ``node_attrs``/``edge_index``/``batch`` among its keys.

NOT confirmed (inferred from MACE's general architecture/PyTorch-MACE
conventions, or assumed as a reasonable API design) -- verify each against
the actually-installed ``mace_jax`` version's source
(``mace_jax/modules/models.py``'s ``prepare_graph``, and whatever
``mace-jax-from-torch`` actually writes out) before trusting this in
production:

1. ``node_attrs`` is a one-hot encoding of atomic number against the
   checkpoint's own supported-species table (``z_table``/``atomic_numbers``,
   PyTorch-MACE's convention) -- ``_z_table_one_hot`` below assumes this.
2. The exact set of other keys ``data`` needs (``positions``/``shifts``/
   ``cell``/``ptr``) and their shapes/dtypes.
3. What ``mace-jax-from-torch --foundation mp --model-name small`` actually
   writes to disk (a single ``.npz`` of params? A sibling architecture-config
   file alongside it, e.g. JSON/gin?) -- ``load_frozen_checkpoint`` below
   assumes a ``<checkpoint_path>.npz`` (flax params, flattened with ``"/"``-
   joined keys, unflattened here via ``flax.traverse_util.unflatten_dict``)
   plus a sibling ``<checkpoint_path>.json`` (architecture hyperparameters +
   supported atomic numbers) -- adjust this loader once the real conversion
   output is inspected on the cluster.
============================================================================
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from flax import traverse_util

from dim_red.mace.graph import atoms_list_to_mace_batch

Array = jax.Array
Params = dict


def _z_table_one_hot(atomic_numbers: np.ndarray, z_table: List[int]) -> np.ndarray:
    """One-hot-encode real atomic numbers against the pretrained checkpoint's
    supported-species table.

    Unlike ``dim_red.cgcnn.graph._local_species_indices`` (a per-structure
    LOCAL remap with no fixed vocabulary), this uses a fixed GLOBAL
    vocabulary -- the checkpoint's own ``z_table`` -- since a pretrained
    foundation model's embedding table is keyed by real chemical species, not
    a per-structure-relative slot.

    Args:
        atomic_numbers: ``(n_atoms,)`` real atomic numbers.
        z_table: The checkpoint's supported atomic numbers, in the order its
            embedding table expects.

    Returns:
        ``(n_atoms, len(z_table))`` float32 one-hot array.

    Raises:
        ValueError: If any atomic number isn't in ``z_table`` -- this
            structure contains a chemical element the pretrained model was
            never trained on.
    """
    index = {z: i for i, z in enumerate(z_table)}
    n_atoms = len(atomic_numbers)
    one_hot = np.zeros((n_atoms, len(z_table)), dtype=np.float32)
    for row, z in enumerate(atomic_numbers):
        z = int(z)
        if z not in index:
            raise ValueError(
                f"Atomic number {z} is not among the pretrained checkpoint's "
                f"supported species {sorted(index)} -- this structure contains "
                "an element the foundation model was never trained on."
            )
        one_hot[row, index[z]] = 1.0
    return one_hot


def load_frozen_checkpoint(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a converted MACE-JAX checkpoint: frozen ``params`` plus the
    architecture config/species table needed to reconstruct the model that
    produced them.

    See this module's docstring -- the exact on-disk layout produced by
    ``mace-jax-from-torch`` needs confirming against the real tool; this
    assumes ``<checkpoint_path>.npz`` (flattened params, ``"/"``-joined keys)
    + a sibling ``<checkpoint_path>.json`` (``{"r_max": ..., "num_interactions":
    ..., "hidden_irreps": ..., "atomic_numbers": [...], ...}``).

    Returns:
        Dict with ``"params"`` (unflattened parameter pytree) and
        ``"arch_config"`` (the parsed JSON sidecar).
    """
    checkpoint_path = Path(checkpoint_path)
    npz_path = checkpoint_path.with_suffix(".npz")
    config_path = checkpoint_path.with_suffix(".json")

    flat = dict(np.load(npz_path))
    flat_params = {tuple(k.split("/")): jnp.asarray(v) for k, v in flat.items()}
    params = traverse_util.unflatten_dict(flat_params)

    with open(config_path, "r") as f:
        arch_config = json.load(f)

    return {"params": params, "arch_config": arch_config}


class MaceEncoder:
    """Frozen pretrained MACE model, used only for inference.

    Mirrors ``dim_red.cgcnn.model.CGCNNEncoder``'s external shape just enough
    to slot into the pipeline (``encode(atoms_list) -> (n, latent_dim)``,
    ``self.params`` for save/load uniformity with
    ``dim_red.pipeline.single_run``/``dim_red.pipeline.inference``) --
    everything else is different: there is no architecture-hyperparameter
    constructor (the architecture is whatever the checkpoint says), no
    ``classify_family``/``classify_spacegroup`` (no heads -- classification
    happens entirely in a separate tail, see
    ``dim_red.pipeline.tail_training``), and ``self.params`` never changes
    after construction (no training loop touches it).
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        r_max: float,
        pooling: str = "mean",
    ):
        """Load a frozen pretrained MACE model.

        Args:
            checkpoint_path: Path (without extension, or with -- both
                accepted via ``Path.with_suffix``) to a MACE-JAX checkpoint
                converted from a Torch foundation model -- see
                ``load_frozen_checkpoint``.
            r_max: Cutoff radius (Angstroms) for neighbor search -- MUST
                match the cutoff the checkpoint was trained with (found in
                the checkpoint's own architecture config; passed explicitly
                here rather than trusted implicitly so a mismatched
                ``dim_red.pipeline.config.MaceConfig.r_max`` fails loudly
                rather than silently producing wrong features).
            pooling: ``"mean"`` (default) or ``"sum"`` -- how per-atom
                ``node_feats`` are pooled into one per-structure vector.

        Raises:
            ValueError: If ``pooling`` isn't ``"mean"``/``"sum"``.
        """
        if pooling not in ("mean", "sum"):
            raise ValueError(f"pooling must be 'mean' or 'sum', got {pooling!r}")

        # Imported here, not at module top, so `dim_red.mace.model` stays
        # importable without `mace_jax` installed unless this class is
        # actually instantiated -- same lazy-heavy-dependency convention as
        # `dim_red.generate`'s pyxtal import inside `dataset_cache.py`.
        from mace_jax.modules.models import MACE

        checkpoint = load_frozen_checkpoint(checkpoint_path)
        arch_config = checkpoint["arch_config"]

        self.z_table: List[int] = list(arch_config["atomic_numbers"])
        self.r_max = r_max
        self.pooling = pooling

        # arch_config's remaining keys are whatever MACE(...) needs beyond
        # r_max/atomic_numbers -- forwarded as-is; see this module's
        # docstring, item 3.
        arch_kwargs = {k: v for k, v in arch_config.items() if k != "atomic_numbers"}
        arch_kwargs["r_max"] = r_max
        self.module = MACE(**arch_kwargs)
        self.params: Params = checkpoint["params"]

    def encode(self, atoms_list: List[Atoms]) -> np.ndarray:
        """Encode a list of structures into one pooled vector each.

        Args:
            atoms_list: Structures to encode (non-empty). Every chemical
                species present must be among ``self.z_table`` (the
                pretrained checkpoint's supported species) -- otherwise
                raises (see ``_z_table_one_hot``).

        Returns:
            ``(len(atoms_list), mace_dim)`` float32 array, one pooled
            representation per structure.
        """
        batch = atoms_list_to_mace_batch(atoms_list, self.r_max)
        n_graphs = batch["n_graphs"]

        node_attrs = _z_table_one_hot(batch["atomic_numbers"], self.z_table)
        ptr = np.concatenate(
            [
                np.array([0], dtype=np.int32),
                np.cumsum(np.bincount(batch["batch"], minlength=n_graphs)).astype(
                    np.int32
                ),
            ]
        )

        data = {
            "node_attrs": jnp.asarray(node_attrs),
            "positions": jnp.asarray(batch["positions"]),
            "edge_index": jnp.asarray(batch["edge_index"]),
            "shifts": jnp.asarray(batch["shifts"]),
            "batch": jnp.asarray(batch["batch"]),
            "ptr": jnp.asarray(ptr),
        }

        out = self.module.apply({"params": self.params}, data, compute_node_feats=True)
        node_feats = np.asarray(out["node_feats"])

        feat_dim = node_feats.shape[-1]
        pooled = np.zeros((n_graphs, feat_dim), dtype=np.float32)
        np.add.at(pooled, batch["batch"], node_feats)
        if self.pooling == "mean":
            counts = np.maximum(
                np.bincount(batch["batch"], minlength=n_graphs), 1
            ).astype(np.float32)
            pooled = pooled / counts[:, None]

        return pooled
