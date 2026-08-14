"""
Container for whole-dataset CGCNN graph arrays.

Mirrors ``dim_red.vae.database.VAEDatabase``'s API shape (``train_val_split``)
generalized to 5 parallel arrays instead of one flat feature matrix -- a
crystal graph (variable atom/edge count per structure) can't be represented
as one row of a fixed-width matrix the way a SOAP descriptor can, so this
holds five arrays already padded to a common ``max_atoms`` (see
``dim_red.cgcnn.graph.atoms_list_to_graph_arrays``), each row-indexable in
lockstep.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class GraphDatabase:
    """Container for tabular graph arrays used for CGCNN training.

    Attributes:
        local_species_idx: ``(n_samples, max_atoms)`` int32 -- per-structure
            LOCAL species slot (``1..max_species``, ``0``=padding), NOT a
            real atomic number -- see
            ``dim_red.cgcnn.graph._local_species_indices``.
        nbr_idx: ``(n_samples, max_atoms, max_num_nbr)`` int32.
        nbr_fea: ``(n_samples, max_atoms, max_num_nbr, n_gaussian)`` float32.
        nbr_mask: ``(n_samples, max_atoms, max_num_nbr)`` float32.
        atom_mask: ``(n_samples, max_atoms)`` float32.
    """

    local_species_idx: np.ndarray
    nbr_idx: np.ndarray
    nbr_fea: np.ndarray
    nbr_mask: np.ndarray
    atom_mask: np.ndarray

    @classmethod
    def from_arrays(
        cls,
        local_species_idx: np.ndarray,
        nbr_idx: np.ndarray,
        nbr_fea: np.ndarray,
        nbr_mask: np.ndarray,
        atom_mask: np.ndarray,
    ) -> "GraphDatabase":
        """Build a :class:`GraphDatabase` from array-likes, validating shapes.

        Args:
            local_species_idx: ``(n, max_atoms)``.
            nbr_idx: ``(n, max_atoms, max_num_nbr)``.
            nbr_fea: ``(n, max_atoms, max_num_nbr, n_gaussian)``.
            nbr_mask: ``(n, max_atoms, max_num_nbr)``.
            atom_mask: ``(n, max_atoms)``.

        Returns:
            A new :class:`GraphDatabase` with appropriately dtype'd arrays.

        Raises:
            ValueError: If any array has the wrong rank, or the arrays'
                leading dimensions are inconsistent with each other.
        """
        local_species_idx = np.asarray(local_species_idx, dtype=np.int32)
        nbr_idx = np.asarray(nbr_idx, dtype=np.int32)
        nbr_fea = np.asarray(nbr_fea, dtype=np.float32)
        nbr_mask = np.asarray(nbr_mask, dtype=np.float32)
        atom_mask = np.asarray(atom_mask, dtype=np.float32)

        if local_species_idx.ndim != 2:
            raise ValueError(
                "local_species_idx must be 2D (n, max_atoms), got "
                f"{local_species_idx.ndim}D"
            )
        if nbr_idx.ndim != 3:
            raise ValueError(
                f"nbr_idx must be 3D (n, max_atoms, max_num_nbr), got {nbr_idx.ndim}D"
            )
        if nbr_fea.ndim != 4:
            raise ValueError(
                "nbr_fea must be 4D (n, max_atoms, max_num_nbr, n_gaussian), "
                f"got {nbr_fea.ndim}D"
            )
        if nbr_mask.ndim != 3:
            raise ValueError(
                f"nbr_mask must be 3D (n, max_atoms, max_num_nbr), got {nbr_mask.ndim}D"
            )
        if atom_mask.ndim != 2:
            raise ValueError(
                f"atom_mask must be 2D (n, max_atoms), got {atom_mask.ndim}D"
            )

        n, max_atoms = local_species_idx.shape
        max_num_nbr = nbr_idx.shape[2]
        expected_shapes = {
            "nbr_idx": (n, max_atoms, max_num_nbr),
            "nbr_fea": (n, max_atoms, max_num_nbr, nbr_fea.shape[3]),
            "nbr_mask": (n, max_atoms, max_num_nbr),
            "atom_mask": (n, max_atoms),
        }
        actual_shapes = {
            "nbr_idx": nbr_idx.shape,
            "nbr_fea": nbr_fea.shape,
            "nbr_mask": nbr_mask.shape,
            "atom_mask": atom_mask.shape,
        }
        for name, expected in expected_shapes.items():
            if actual_shapes[name] != expected:
                raise ValueError(
                    f"{name}.shape={actual_shapes[name]} is inconsistent with "
                    f"local_species_idx.shape={local_species_idx.shape} "
                    f"(expected {expected})"
                )

        return cls(
            local_species_idx=local_species_idx,
            nbr_idx=nbr_idx,
            nbr_fea=nbr_fea,
            nbr_mask=nbr_mask,
            atom_mask=atom_mask,
        )

    @property
    def n_samples(self) -> int:
        """Number of structures in this database."""
        return self.local_species_idx.shape[0]

    def as_tuple(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The five parallel arrays, in a fixed order.

        Feeds directly into ``dim_red.cgcnn.training``'s batch-iteration
        helpers, duplicated from ``dim_red.autoencoder.training`` and
        already rank-agnostic there (they operate generically on any tuple
        of same-leading-dim arrays), so no changes are needed to accept
        these ranks.
        """
        return (
            self.local_species_idx,
            self.nbr_idx,
            self.nbr_fea,
            self.nbr_mask,
            self.atom_mask,
        )

    def __getitem__(self, idx) -> "GraphDatabase":
        """Row-index all five arrays at once (fancy/boolean/slice indexing)."""
        return GraphDatabase(
            local_species_idx=self.local_species_idx[idx],
            nbr_idx=self.nbr_idx[idx],
            nbr_fea=self.nbr_fea[idx],
            nbr_mask=self.nbr_mask[idx],
            atom_mask=self.atom_mask[idx],
        )

    def train_val_split(
        self, val_ratio: float = 0.2, seed: int = 42
    ) -> Tuple["GraphDatabase", "GraphDatabase"]:
        """Split dataset into train/validation partitions.

        Identical split logic/semantics to
        ``dim_red.vae.database.VAEDatabase.train_val_split``: reproducible
        via a local RNG seeded with ``seed``, at least one sample guaranteed
        in both partitions.

        Args:
            val_ratio: Fraction of samples assigned to validation. Must be
                strictly in ``(0, 1)``.
            seed: Random seed for reproducibility.

        Returns:
            A tuple ``(train_db, val_db)``.

        Raises:
            ValueError: If ``val_ratio`` is not strictly between 0 and 1.
        """
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be in the open interval (0, 1)")

        n_samples = self.n_samples
        n_val = max(1, int(round(n_samples * val_ratio)))
        n_val = min(n_val, n_samples - 1)

        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        return self[train_idx], self[val_idx]
