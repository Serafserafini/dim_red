import pytest
import yaml

pytest.importorskip("jax")

from dim_red.pipeline.full_stack_config import (
    FAMILY,
    full_stack_config_from_dict,
    full_stack_config_from_resolved_dict,
    full_stack_config_to_dict,
    load_full_stack_config,
)


def _block(**over):
    block = {
        "data": {"pyxtal": {"structures_per_spacegroup": 2}},
        "encoder": {"encoder_hidden_dim": [8], "latent_dim": 4},
        "train": {"epochs": 2, "batch_size": 8},
    }
    block.update(over)
    return block


def _config(**over):
    d = {
        "name": "t",
        "seed": 10,
        "family": _block(),
        "experts": {"defaults": _block(), "cubic": {}, "tetragonal": {}},
    }
    d.update(over)
    return d


def test_defaults_are_merged_under_each_expert_override():
    d = _config()
    d["experts"]["cubic"] = {"encoder": {"latent_dim": 6}}
    cfg = full_stack_config_from_dict(d)
    assert cfg.stacks["cubic"].model.latent_dim == 6
    assert list(cfg.stacks["cubic"].model.encoder_hidden_dim) == [8]  # from defaults
    assert cfg.stacks["tetragonal"].model.latent_dim == 4


def test_stack_order_is_family_then_canonical_experts():
    d = _config()
    d["experts"] = {"defaults": _block(), "cubic": {}, "triclinic": {}}
    cfg = full_stack_config_from_dict(d)
    assert list(cfg.stacks) == ["family", "triclinic", "cubic"]


def test_expert_pyxtal_is_restricted_to_its_system_and_family_is_not():
    cfg = full_stack_config_from_dict(_config())
    assert cfg.stacks["cubic"].data.pyxtal.families == ["Cubic"]
    assert cfg.stacks["tetragonal"].data.pyxtal.families == ["Tetragonal"]
    assert cfg.stacks[FAMILY].data.pyxtal.families is None


def test_expert_with_a_different_family_is_rejected():
    d = _config()
    d["experts"]["cubic"] = {"data": {"pyxtal": {"families": ["Hexagonal"]}}}
    with pytest.raises(ValueError, match="cubic"):
        full_stack_config_from_dict(d)


def test_expert_with_explicit_spacegroups_is_left_alone():
    d = _config()
    d["experts"]["cubic"] = {
        "data": {"pyxtal": {"spacegroups": [195, 200], "structures_per_spacegroup": 3}}
    }
    cfg = full_stack_config_from_dict(d)
    pyxtal = cfg.stacks["cubic"].data.pyxtal
    assert pyxtal.spacegroups == [195, 200]
    assert pyxtal.families is None


def test_unknown_expert_name_lists_valid_names():
    d = _config()
    d["experts"]["cubik"] = {}
    with pytest.raises(ValueError, match="cubik") as exc:
        full_stack_config_from_dict(d)
    assert "cubic" in str(exc.value)


def test_config_without_stacks_is_rejected():
    with pytest.raises(ValueError, match="no stacks"):
        full_stack_config_from_dict({"name": "t"})


def test_family_is_optional():
    d = _config()
    del d["family"]
    cfg = full_stack_config_from_dict(d)
    assert FAMILY not in cfg.stacks and "cubic" in cfg.stacks


def test_seeds_follow_the_canonical_stack_index_and_fill_pyxtal_seed():
    cfg = full_stack_config_from_dict(_config())
    assert cfg.stacks[FAMILY].seed == 10
    assert cfg.stacks["tetragonal"].seed == 14
    assert cfg.stacks["cubic"].seed == 17
    assert cfg.stacks["cubic"].data.pyxtal.seed == 17
    assert cfg.stacks["cubic"].model.seed == 17
    assert cfg.stacks["cubic"].model.body_train.seed == 17


def test_explicit_pyxtal_seed_is_kept():
    d = _config()
    d["family"]["data"]["pyxtal"]["seed"] = 99
    cfg = full_stack_config_from_dict(d)
    assert cfg.stacks[FAMILY].data.pyxtal.seed == 99


def test_head_training_settings_inherit_the_body_train_block():
    d = _config()
    d["family"]["train"] = {"epochs": 7, "batch_size": 16, "val_ratio": 0.3}
    d["family"]["viz"] = {"train": {"epochs": 3}}
    cfg = full_stack_config_from_dict(d)
    model = cfg.stacks[FAMILY].model
    assert model.body_train.epochs == 7
    assert (
        model.classifier_train.epochs == 7 and model.classifier_train.batch_size == 16
    )
    assert model.viz_train.epochs == 3
    assert cfg.stacks[FAMILY].val_ratio == 0.3


def test_contrastive_and_viz_defaults():
    model = full_stack_config_from_dict(_config()).stacks[FAMILY].model
    assert (model.body_train.tau, model.body_train.distance) == (0.05, "cosine")
    assert (model.viz_train.tau, model.viz_train.distance) == (0.1, "euclidean")
    assert model.viz_dim == 2 and model.projection_dim == 128


def test_invalid_distance_is_rejected():
    d = _config()
    d["family"]["contrastive"] = {"distance": "manhattan"}
    with pytest.raises(ValueError, match="distance"):
        full_stack_config_from_dict(d)


def test_missing_encoder_fields_are_rejected():
    d = _config()
    d["family"]["encoder"] = {"latent_dim": 4}
    with pytest.raises(ValueError, match="encoder_hidden_dim"):
        full_stack_config_from_dict(d)


def test_unknown_block_key_is_rejected():
    d = _config()
    d["family"]["encoder_hidden"] = [8]
    with pytest.raises(ValueError, match="encoder_hidden"):
        full_stack_config_from_dict(d)


def test_supcon_mace_requires_a_checkpoint():
    d = _config(model_kind="supcon_mace")
    with pytest.raises(ValueError, match="checkpoint_path"):
        full_stack_config_from_dict(d)


def test_resolved_dict_roundtrip():
    cfg = full_stack_config_from_dict(_config())
    assert full_stack_config_from_resolved_dict(full_stack_config_to_dict(cfg)) == cfg


def test_load_from_yaml_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(_config()))
    cfg = load_full_stack_config(path)
    assert set(cfg.stacks) == {"family", "cubic", "tetragonal"}
