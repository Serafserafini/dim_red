"""Constants and synthetic inputs shared by ``make_golden.py`` (which runs the
OLD training code once) and the tests that compare ``SingleStack`` to its output."""

from pathlib import Path

import numpy as np

GOLDEN_DIR = Path(__file__).parent / "data"

SEED = 0
N_FEATURES = 12
FAMILIES = ["Alpha", "Beta", "Gamma"]
SPACEGROUPS = {"Alpha": [1, 2], "Beta": [3, 4], "Gamma": [5, 6]}
PER_SPACEGROUP = 10  # 3 families x 2 spacegroups x 10 = 60 structures
VAL_RATIO = 0.25

ENCODER_HIDDEN = [8]
LATENT = 4
PROJ_DIM = 6
TAU = 0.05
DISTANCE = "cosine"
BODY_EPOCHS = 3
BATCH = 8
HEAD_EPOCHS = 2
CLF_HIDDEN = 8
VIZ_HIDDEN = [6]
VIZ_TAU = 0.1
VIZ_DISTANCE = "euclidean"
EXPERT_CLF_HIDDEN = 4
EXPERT_VIZ_HIDDEN = [4]
MIN_SAMPLES_PER_EXPERT = 5

FAMILY_FILES = [
    "inputs.npz",
    "family_body_params.msgpack",
    "family_projection_params.msgpack",
    "family_body_loss_history.csv",
    "family_embeddings.npz",
    "family_classifier_params.msgpack",
    "family_classifier_loss_history.csv",
    "family_viz_params.msgpack",
    "family_viz_loss_history.csv",
]


def expert_files(family: str) -> list:
    return [
        f"expert_{family}_{suffix}"
        for suffix in (
            "body_params.msgpack",
            "projection_params.msgpack",
            "classifier_params.msgpack",
            "viz_params.msgpack",
            "body_loss_history.csv",
            "classifier_loss_history.csv",
            "viz_loss_history.csv",
            "local_classes.yaml",
        )
    ]


def make_inputs():
    """Deterministic synthetic dataset: returns ``(X, labels, material_ids,
    spacegroups)`` with ``X`` float32 of shape ``(60, N_FEATURES)``."""
    rng = np.random.default_rng(1234)
    X, labels, material_ids, spacegroups = [], [], [], []
    for family in FAMILIES:
        for sg in SPACEGROUPS[family]:
            center = rng.normal(size=N_FEATURES) * 2.0
            for i in range(PER_SPACEGROUP):
                X.append(center + rng.normal(size=N_FEATURES) * 0.5)
                labels.append(family)
                material_ids.append(f"g-{sg}-{i}")
                spacegroups.append(sg)
    return np.asarray(X, dtype=np.float32), labels, material_ids, spacegroups
