import copy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("jax")

from dim_red.pipeline.config import (
    _set_dotted,
    run_config_from_dict,
    tail_train_config_from_dict,
)
from dim_red.pipeline.full_stack_config import full_stack_config_from_dict

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIGS = sorted(CONFIGS_DIR.rglob("*.yaml"))


def _is_full_stack(d):
    return any(k in d for k in ("model_kind", "family", "experts"))


def _load(path):
    d = yaml.safe_load(path.read_text())
    if "base" in d and "grid" in d:
        base = d["base"]
        assert _is_full_stack(base), f"{path}: sweep base must be a FullStack config"
        full_stack_config_from_dict(base)
        first = copy.deepcopy(base)
        for key, values in d["grid"].items():
            _set_dotted(first, key, values[0])
        return full_stack_config_from_dict(first)
    if _is_full_stack(d):
        return full_stack_config_from_dict(d)
    if "tail_kind" in d:
        return tail_train_config_from_dict(d)
    return run_config_from_dict(d)


def test_there_are_example_configs():
    assert len(CONFIGS) >= 6


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p))
def test_example_configs_load(path):
    _load(path)


def test_full_stack_examples_cover_family_and_experts():
    cfg = full_stack_config_from_dict(
        yaml.safe_load((CONFIGS_DIR / "full_stack.example.yaml").read_text())
    )
    assert "family" in cfg.stacks and len(cfg.stacks) >= 3
    assert all(s.data.soap.element_agnostic for s in cfg.stacks.values())
