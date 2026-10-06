"""One dataset per stack: every ``SingleStack`` of a ``FullStack`` gets its own
pyxtal-generated (and optionally augmented) structures, featurized with SOAP
or MACE through the existing dataset cache. Stacks never share structures:
their pyxtal seeds differ (see ``full_stack_config``)."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

from dim_red.pipeline.dataset_cache import (
    get_or_build_pyxtal_dataset,
    get_or_build_pyxtal_mace_dataset,
    resolve_augmentation,
)
from dim_red.pipeline.full_stack_config import StackSpec


@dataclass(frozen=True)
class StackDataset:
    """Standardized features plus the per-structure metadata of one stack."""

    X: np.ndarray
    labels: List[str]  # crystal family of each structure
    material_ids: List[str]
    spacegroups: List[int]
    structures_path: Path
    feature_mean: np.ndarray
    feature_std: np.ndarray


def build_stack_dataset(
    spec: StackSpec, model_kind: str, cache_dir: Union[str, Path]
) -> StackDataset:
    augmentation = resolve_augmentation(spec.data.augmentation, spec.seed)
    seed = spec.data.pyxtal.seed if spec.data.pyxtal.seed is not None else spec.seed
    if model_kind == "supcon_mace":
        result = get_or_build_pyxtal_mace_dataset(
            pyxtal_config=spec.data.pyxtal,
            seed=seed,
            mace_kwargs=spec.data.mace.mace_kwargs(),
            cache_dir=cache_dir,
            augmentation=augmentation,
        )
    else:
        result = get_or_build_pyxtal_dataset(
            pyxtal_config=spec.data.pyxtal,
            seed=seed,
            soap_kwargs=spec.data.soap.as_kwargs(),
            cache_dir=cache_dir,
            augmentation=augmentation,
        )
    X, labels, material_ids, spacegroups, structures_path, mean, std = result
    return StackDataset(
        X=X,
        labels=list(labels),
        material_ids=list(material_ids),
        spacegroups=[int(s) for s in spacegroups],
        structures_path=Path(structures_path),
        feature_mean=mean,
        feature_std=std,
    )
