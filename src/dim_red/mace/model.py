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
Verified against an actually-installed ``mace_jax`` (0.2.0, 2026-09) --
this replaces an earlier version of this module written blind against
``mace_jax``'s source read remotely, which turned out to be wrong on several
points once a real install was available:

- ``mace_jax.modules.models.MACE`` is a Flax **NNX** module (``nnx.Module``),
  not Linen -- there is no ``.apply({"params": ...}, ...)`` call convention.
  A loaded checkpoint's ``graphdef``/``params`` are called as
  ``graphdef.apply(params)(data, ...)`` (``mace_jax.tools.bundle``'s own
  pattern, mirrored by ``mace_jax.cli.mace_jax_predict``).
- ``mace-jax-from-torch --output foo.npz`` does not write a real ``.npz``
  (numpy zip archive) despite the extension -- it's raw
  ``flax.serialization.to_bytes(...)`` (msgpack) bytes of an NNX state dict.
  Loading it requires rebuilding the exact same graph structure first
  (``mace_jax.tools.model_builder._build_jax_model``) and restoring state
  into it as a template (``mace_jax.tools.bundle.load_model_bundle`` does
  exactly this, and is reused here rather than reimplemented).
- The model's forward-pass input is a specific dict built by
  ``mace_jax.tools.gin_model._graph_to_data`` from a (possibly padded)
  ``jraph.GraphsTuple`` -- ``node_attrs`` (one-hot), ``node_attrs_index``,
  ``edge_index``, ``shifts``, ``unit_shifts``, ``batch``, ``ptr``, ``cell``,
  optionally ``head`` -- not the simpler ad-hoc dict this module's graph
  construction used to build directly.
- Padding a batch with ``jraph.pad_with_graphs`` before calling the model is
  not just a JIT-shape-stabilization nicety -- ``mace_jax``'s own
  ``MACEJAXCalculator``/``mace_jax_predict`` CLI both always pad (at least
  one spare node/edge/graph) even for a single structure, so this module
  does too rather than risk relying on unpadded-batch behavior nobody
  upstream actually exercises.

``dim_red.mace.graph`` (pure NumPy/ASE, no ``jax``/``mace_jax`` import) is
intentionally left untouched by this -- its ``atoms_to_mace_graph``/
``atoms_list_to_mace_batch`` dicts are not what ``mace_jax`` itself consumes
(this module doesn't feed them into ``mace_jax`` at all), but the shape is
still useful as a lightweight, dependency-free description of a structure's
neighbor graph, and existing tests cover it as such.
============================================================================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from ase import Atoms

Array = Any
Params = dict


def _sanitize_for_serialization(obj: Any) -> Any:
    """Recursively convert ``mace_jax``'s ``ConfigDict`` wrapper (used for a
    few static/config-like NNX state leaves, e.g. ``_normalize2mom_consts_var``)
    into a plain ``dict`` -- ``flax.serialization.to_bytes``'s msgpack packer
    (``strict_types=True``) rejects a ``ConfigDict`` outright even though it
    behaves like a dict. Mirrors ``mace_jax.nnx_utils.state_to_serializable_dict``'s
    own internal ``_convert`` helper (the same sanitization ``mace_jax`` itself
    applies before writing bytes via its CLI), just applied here to an
    already-``state_to_pure_dict``-shaped tree (``load_model_bundle``'s
    ``params``) rather than to a live ``nnx.State``.
    """
    from mace_jax.nnx_config import ConfigDict

    if isinstance(obj, (ConfigDict, dict)):
        return {k: _sanitize_for_serialization(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_sanitize_for_serialization(v) for v in obj)
    return obj


def load_frozen_checkpoint(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a converted MACE-JAX checkpoint via ``mace_jax``'s own loader.

    Args:
        checkpoint_path: Path to the ``mace-jax-from-torch``-produced
            ``<name>.npz`` (or the extension-less ``<name>``) -- its sibling
            ``<name>.json`` architecture/species config is discovered
            automatically (``mace_jax.tools.bundle.resolve_model_paths``).

    Returns:
        Dict with ``"params"`` (a flat pytree of arrays, ready for
        ``flax.serialization.to_bytes`` -- a *sanitized* copy safe to write
        to disk, see ``_sanitize_for_serialization``), ``"params_for_apply"``
        (the original, un-sanitized pytree -- MUST be used for the actual
        forward pass instead, see ``MaceEncoder._encode_chunk``: a handful of
        static/config-like leaves are wrapped in ``mace_jax``'s own
        ``ConfigDict``, a custom-registered single-leaf pytree node, and
        ``graphdef.apply`` expects that *exact* leaf structure back --
        replacing a ``ConfigDict`` leaf with an equivalent plain ``dict``
        changes how many leaves the pytree flattens to, so ``graphdef.apply``
        of the sanitized copy raises a leaf-count mismatch), ``"graphdef"``
        (the NNX ``GraphDef`` needed to turn either ``params`` pytree back
        into a callable model), and ``"arch_config"`` (parsed JSON sidecar:
        ``r_max``, ``atomic_numbers``, and the rest of the architecture
        hyperparameters ``mace_jax.tools.model_builder._build_jax_model``
        needs).
    """
    # Imported here, not at module top, so `dim_red.mace.model` stays
    # importable without `mace_jax` installed unless this function is
    # actually called -- same lazy-heavy-dependency convention as
    # `dim_red.generate`'s pyxtal import inside `dataset_cache.py`.
    from mace_jax.tools import bundle as mace_bundle

    checkpoint_path = Path(checkpoint_path)
    npz_path = checkpoint_path.with_suffix(".npz")
    bundle = mace_bundle.load_model_bundle(str(npz_path), "float32")

    return {
        "params": _sanitize_for_serialization(bundle.params),
        "params_for_apply": bundle.params,
        "graphdef": bundle.graphdef,
        "arch_config": bundle.config,
    }


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
        max_nodes_per_batch: int = 1024,
        max_edges_per_batch: int = 4096,
        max_graphs_per_batch: int = 64,
    ):
        """Load a frozen pretrained MACE model.

        Args:
            checkpoint_path: Path (without extension, or with -- both
                accepted) to a MACE-JAX checkpoint converted from a Torch
                foundation model -- see ``load_frozen_checkpoint``.
            r_max: Cutoff radius (Angstroms) for neighbor search -- MUST
                match the cutoff the checkpoint was trained with. Checked
                against the checkpoint's own recorded ``r_max`` and raises
                if they disagree, rather than silently using a mismatched
                cutoff (which would produce wrong features without any
                error).
            pooling: ``"mean"`` (default) or ``"sum"`` -- how per-atom
                ``node_feats`` are pooled into one per-structure vector.
            max_nodes_per_batch/max_edges_per_batch/max_graphs_per_batch:
                *Minimum* padded-batch budget (atoms/edges/structures) each
                forward pass is padded up to -- purely a memory/throughput
                knob, not a correctness one (results are identical
                regardless of these, `encode` grows the actual budget past
                this floor if a single structure needs more, see its own
                docstring). ``encode`` greedily bin-packs ``atoms_list``
                into chunks that fit under the (possibly grown) budget
                (variable number of structures per chunk, since e.g. a
                pathologically small periodic cell -- or a large augmented
                supercell -- can have far more neighbor-list edges within
                ``r_max`` than a typical structure in the same dataset),
                rather than a fixed structure count per chunk -- padding
                every chunk to a *count*-based batch size would pad to
                ``batch_size * worst_single_structure_size``, which blows up
                GPU memory the moment one outlier structure is much larger
                than the rest of the dataset. Every chunk is still padded to
                the *same* budget within one ``encode`` call, so the forward
                pass JIT-compiles once per call and is reused for every
                chunk regardless of how many real structures happen to be
                in it.

        Raises:
            ValueError: If ``pooling`` isn't ``"mean"``/``"sum"``, or if
                ``r_max`` doesn't match the checkpoint's own cutoff.
        """
        if pooling not in ("mean", "sum"):
            raise ValueError(f"pooling must be 'mean' or 'sum', got {pooling!r}")

        checkpoint = load_frozen_checkpoint(checkpoint_path)
        arch_config = checkpoint["arch_config"]

        ckpt_r_max = float(arch_config["r_max"])
        if abs(ckpt_r_max - r_max) > 1e-6:
            raise ValueError(
                f"r_max={r_max} does not match the checkpoint's own cutoff "
                f"({ckpt_r_max}) recorded in its architecture config -- use "
                f"r_max={ckpt_r_max} (mismatched cutoffs silently produce "
                "wrong features)."
            )

        self.z_table: List[int] = [int(z) for z in arch_config["atomic_numbers"]]
        self.r_max = r_max
        self.pooling = pooling
        self.max_nodes_per_batch = max_nodes_per_batch
        self.max_edges_per_batch = max_edges_per_batch
        self.max_graphs_per_batch = max_graphs_per_batch
        self.graphdef = checkpoint["graphdef"]
        self.params: Params = checkpoint["params"]
        self._params_for_apply: Params = checkpoint["params_for_apply"]

    def encode(self, atoms_list: List[Atoms]) -> np.ndarray:
        """Encode a list of structures into one pooled vector each.

        Args:
            atoms_list: Structures to encode (non-empty). Every chemical
                species present must be among the checkpoint's own supported
                species -- otherwise raises (``mace_jax.data.AtomicNumberTable``
                / ``z_to_index_map``).

        Returns:
            ``(len(atoms_list), mace_dim)`` float32 array, one pooled
            representation per structure, in the same order as ``atoms_list``.

        Raises:
            ValueError: If ``atoms_list`` is empty.
        """
        if not atoms_list:
            raise ValueError("atoms_list is empty -- nothing to encode")

        from mace_jax import data as mace_data

        z_table = mace_data.AtomicNumberTable(self.z_table)
        graphs = [
            mace_data.graph_from_configuration(
                mace_data.config_from_atoms(atoms), cutoff=self.r_max, z_table=z_table
            )
            for atoms in atoms_list
        ]
        sizes = [
            (int(np.asarray(g.n_node).sum()), int(np.asarray(g.n_edge).sum()))
            for g in graphs
        ]

        # max_nodes_per_batch/max_edges_per_batch are a *minimum* padding
        # budget (sized for decent multi-structure batching on typical
        # structures), not a hard cap: a single outlier structure (e.g. a
        # large augmented supercell -- pyxtal + augmentation.supercell_radius
        # can produce periodic cells with far more atoms/neighbor-list edges
        # within r_max than a typical one in the same dataset) grows the
        # actual padding budget to fit it, rather than raising. Every chunk
        # still shares this one grown shape, so the forward pass still only
        # JIT-compiles once.
        max_single_nodes = max(n for n, _e in sizes)
        max_single_edges = max(e for _n, e in sizes)
        n_node_pad = max(self.max_nodes_per_batch, max_single_nodes + 1)
        n_edge_pad = max(self.max_edges_per_batch, max_single_edges + 1)
        n_graph_pad = self.max_graphs_per_batch + 1

        # Greedily bin-pack into chunks that fit the (possibly grown)
        # padding budget -- see the constructor docstring for why this
        # isn't just a fixed structure count per chunk.
        chunks: List[List[int]] = []
        current: List[int] = []
        cur_nodes = cur_edges = 0
        for i, (n, e) in enumerate(sizes):
            if current and (
                cur_nodes + n >= n_node_pad
                or cur_edges + e >= n_edge_pad
                or len(current) >= self.max_graphs_per_batch
            ):
                chunks.append(current)
                current, cur_nodes, cur_edges = [], 0, 0
            current.append(i)
            cur_nodes += n
            cur_edges += e
        if current:
            chunks.append(current)

        results: List[Optional[np.ndarray]] = [None] * len(atoms_list)
        for chunk_indices in chunks:
            chunk_graphs = [graphs[i] for i in chunk_indices]
            chunk_pooled = self._encode_chunk(
                chunk_graphs, n_node_pad, n_edge_pad, n_graph_pad
            )
            for local_i, global_i in enumerate(chunk_indices):
                results[global_i] = chunk_pooled[local_i]
        return np.stack(results, axis=0)

    def _encode_chunk(
        self, graphs: List[Any], n_node_pad: int, n_edge_pad: int, n_graph_pad: int
    ) -> np.ndarray:
        import jraph
        from mace_jax.tools import gin_model

        n_graphs = len(graphs)
        batched = jraph.batch_np(graphs)
        padded = jraph.pad_with_graphs(
            batched, n_node=n_node_pad, n_edge=n_edge_pad, n_graph=n_graph_pad
        )

        data_dict = gin_model._graph_to_data(padded, num_species=len(self.z_table))
        outputs, _ = self.graphdef.apply(self._params_for_apply)(
            data_dict,
            compute_force=False,
            compute_stress=False,
            compute_node_feats=True,
        )
        node_feats = np.asarray(outputs["node_feats"])

        node_mask = np.asarray(jraph.get_node_padding_mask(padded))
        batch = np.asarray(data_dict["batch"])
        real = node_mask.astype(bool) & (batch < n_graphs)

        feat_dim = node_feats.shape[-1]
        pooled = np.zeros((n_graphs, feat_dim), dtype=np.float32)
        np.add.at(pooled, batch[real], node_feats[real])
        if self.pooling == "mean":
            counts = np.zeros(n_graphs, dtype=np.float32)
            np.add.at(counts, batch[real], 1.0)
            pooled = pooled / np.maximum(counts, 1.0)[:, None]

        return pooled
