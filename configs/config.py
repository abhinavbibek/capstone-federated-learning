# configs/config.py

NUM_CLIENTS = 10
ROUNDS = 10

LOCAL_EPOCHS = 8
BATCH_SIZE = 64
LEARNING_RATE = 0.001


# Adversarial clients
ATTACK_CLIENTS = [6, 7, 9, 10]

# Differential Privacy settings
NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM = 1.0

# Random seed for reproducibility
SEED = 42

EXPERIMENTS = {
    "baseline": {
        "dp": False,
        "attack": None,
        "robust": False,
    },
    "label_flip": {
        "dp": False,
        "attack": "label_flip",
        "robust": False,
    },
    "targeted_flip": {
        "dp": False,
        "attack": "targeted_flip",
        "robust": False,
    },
    "feature_poison": {
        "dp": False,
        "attack": "feature_poison",
        "robust": False,
    },
    "sign_flip": {
        "dp": False,
        "attack": "sign_flip",
        "robust": False,
    },
    "scaling": {
        "dp": False,
        "attack": "scaling",
        "robust": False,
    },
    "dp_only": {
        "dp": True,
        "attack": False,
        "robust": False,
    },
    "full_system": {
        "dp": True,
        "attack": True,
        "robust": True,
    }
}