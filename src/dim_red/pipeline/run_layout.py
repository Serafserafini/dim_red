"""Reading a ``FullStack`` run directory (jax-free, so ``dimred-compare`` and
``dimred-benchmark`` never need jax).

Layout: ``<run_dir>/config.yaml`` and ``<run_dir>/stacks/<name>/...`` (see
``dim_red.pipeline.full_stack``). Directories named ``.<x>.tmp`` are
half-written atomic writes of an interrupted run and are never listed.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from dim_red.pipeline.config import flatten_config_dict

FAMILY = "family"
EXPERT_NAMES: Tuple[str, ...] = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)
STACK_ORDER: Tuple[str, ...] = (FAMILY,) + EXPERT_NAMES

_HIDDEN_TMP = re.compile(r"^\..*\.tmp$")


def is_hidden_tmp(path: Union[str, Path]) -> bool:
    return bool(_HIDDEN_TMP.match(Path(path).name))


def _visible_dirs(parent: Path) -> List[Path]:
    if not parent.is_dir():
        return []
    return sorted(p for p in parent.iterdir() if p.is_dir() and not is_hidden_tmp(p))


def is_full_stack_run(path: Union[str, Path]) -> bool:
    path = Path(path)
    return (path / "config.yaml").is_file() and (path / "stacks").is_dir()


def discover_full_stack_runs(sweep_dir: Union[str, Path]) -> List[Path]:
    """Every immediate subdirectory of ``sweep_dir`` that is a FullStack run,
    sorted by name for reproducible plot ordering."""
    sweep_dir = Path(sweep_dir)
    runs = [p for p in _visible_dirs(sweep_dir) if is_full_stack_run(p)]
    if not runs:
        raise ValueError(
            f"No completed FullStack runs found directly under {sweep_dir}"
        )
    return runs


def trained_stack_names(run_dir: Union[str, Path]) -> List[str]:
    """Stacks whose body is trained, in canonical order."""
    present = {
        p.name
        for p in _visible_dirs(Path(run_dir) / "stacks")
        if (p / "body" / "stack.yaml").is_file()
    }
    return [n for n in STACK_ORDER if n in present]


def head_names(stack_dir: Union[str, Path]) -> List[str]:
    return [
        p.name
        for p in _visible_dirs(Path(stack_dir) / "heads")
        if (p / "heads.yaml").is_file()
    ]


def resolve_heads_name(
    stack_dir: Union[str, Path], heads_name: Optional[str] = None
) -> str:
    """The heads set to use: the named one, or the only one if exactly one
    exists. Never picks silently between several."""
    names = head_names(stack_dir)
    if heads_name is not None:
        if heads_name not in names:
            raise ValueError(
                f"{stack_dir}: no heads named {heads_name!r}; available: {names}"
            )
        return heads_name
    if len(names) == 1:
        return names[0]
    if not names:
        raise ValueError(
            f"{stack_dir}: no heads trained yet; run fit_heads / dimred-train-heads"
        )
    raise ValueError(
        f"{stack_dir}: several heads available {names}; choose one with "
        "heads_name / --heads-name"
    )


def load_loss_history(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {
        key: np.array([float(row[key]) for row in rows]) for key in reader.fieldnames
    }


@dataclass
class RunData:
    """One stack of one run, as ``compare``/``benchmark`` consume it.

    ``run_dir`` is the run root and ``stack`` names the stack, so the pair is
    unique inside one comparison pass (a pass covers one stack across runs).
    """

    run_dir: Path
    stack: str
    model_kind: str
    flat_config: Dict[str, Any]
    loss_history: Dict[str, np.ndarray]
    embeddings: Dict[str, np.ndarray]
    viz_embeddings: Optional[np.ndarray] = None
    heads_name: Optional[str] = None

    @property
    def label(self) -> str:
        """Run directory name -- unique within a pass, used for identification."""
        return self.run_dir.name


def open_stack(
    run_dir: Union[str, Path],
    stack: str,
    heads_name: Optional[str] = None,
    allow_no_heads: bool = False,
) -> RunData:
    run_dir = Path(run_dir)
    trained = trained_stack_names(run_dir)
    if stack not in trained:
        raise ValueError(
            f"{run_dir}: stack {stack!r} is not trained; trained stacks: {trained}"
        )
    stack_dir = run_dir / "stacks" / stack
    run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    spec = yaml.safe_load((stack_dir / "config.yaml").read_text())
    classes = yaml.safe_load((stack_dir / "classes.yaml").read_text())["classes"]
    with np.load(stack_dir / "embeddings.npz") as npz:
        embeddings = dict(npz.items())
    if stack != FAMILY:
        # an expert's own labels are spacegroups; family-colored plots get the
        # single crystal system it covers
        embeddings["labels"] = np.full(
            len(embeddings["spacegroups"]), stack.capitalize()
        )

    used: Optional[str] = None
    viz: Optional[np.ndarray] = None
    if not (heads_name is None and allow_no_heads and not head_names(stack_dir)):
        used = resolve_heads_name(stack_dir, heads_name)
    if used is not None:
        heads_dir = stack_dir / "heads" / used
        with np.load(heads_dir / "predictions.npz") as npz:
            probs = npz["probs"]
        if stack == FAMILY:
            embeddings["family_probs"] = probs
            embeddings["family_classes"] = np.asarray([str(c) for c in classes])
        else:
            embeddings["spacegroup_probs"] = probs
            embeddings["spacegroup_classes"] = np.asarray(classes, dtype=np.int64)
        with np.load(heads_dir / "viz_embeddings.npz") as npz:
            viz = npz["embeddings"]

    return RunData(
        run_dir=run_dir,
        stack=stack,
        model_kind=run_cfg["model_kind"],
        flat_config=flatten_config_dict(spec),
        loss_history=load_loss_history(stack_dir / "body" / "loss_history.csv"),
        embeddings=embeddings,
        viz_embeddings=viz,
        heads_name=used,
    )
