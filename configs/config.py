# configs/config.py

NUM_CLIENTS = 10
ROUNDS = 10

LOCAL_EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.01


# Adversarial clients
ATTACK_CLIENTS = [7, 9]

# Differential Privacy settings
NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM = 1.0

# Random seed for reproducibility
SEED = 42

EXPERIMENTS = {
    "baseline": {
        "dp": False,
        "attack": False,
        "robust": False,
    },
    "attack_only": {
        "dp": False,
        "attack": True,
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