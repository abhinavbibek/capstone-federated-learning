# configs/config.py

NUM_CLIENTS = 10
ROUNDS = 10

LOCAL_EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 0.001


# Adversarial clients
ATTACK_CLIENTS = [6, 7, 9, 10]


# Random seed for reproducibility
SEED = 42


EXPERIMENTS = {
    "baseline": {"attack": None, "defense": None},
    # ================= ATTACK ONLY =================
    "label_flip_only": {"attack": "label_flip", "defense": None},
    "targeted_flip_only": {"attack": "targeted_flip", "defense": None},
    "feature_poison_only": {"attack": "feature_poison", "defense": None},
    "sign_flip_only": {"attack": "sign_flip", "defense": None},
    "scaling_only": {"attack": "scaling", "defense": None},

    # ================= DATA POISON DEFENSES =================
    "label_flip_median": {"attack": "label_flip", "defense": "median"},
    "label_flip_trimmed": {"attack": "label_flip", "defense": "trimmed_mean"},
    "label_flip_krum": {"attack": "label_flip", "defense": "krum"},
    "label_flip_clip": {"attack": "label_flip", "defense": "clipping"},

    "targeted_flip_median": {"attack": "targeted_flip", "defense": "median"},
    "targeted_flip_trimmed": {"attack": "targeted_flip", "defense": "trimmed_mean"},
    "targeted_flip_krum": {"attack": "targeted_flip", "defense": "krum"},
    "targeted_flip_clip": {"attack": "targeted_flip", "defense": "clipping"},

    "feature_poison_median": {"attack": "feature_poison", "defense": "median"},
    "feature_poison_trimmed": {"attack": "feature_poison", "defense": "trimmed_mean"},
    "feature_poison_krum": {"attack": "feature_poison", "defense": "krum"},
    "feature_poison_clip": {"attack": "feature_poison", "defense": "clipping"},

    # ================= MODEL POISON DEFENSES =================
    "sign_flip_median": {"attack": "sign_flip", "defense": "median"},
    "sign_flip_trimmed": {"attack": "sign_flip", "defense": "trimmed_mean"},
    "sign_flip_krum": {"attack": "sign_flip", "defense": "krum"},
    "sign_flip_clip": {"attack": "sign_flip", "defense": "clipping"},

    "scaling_median": {"attack": "scaling", "defense": "median"},
    "scaling_trimmed": {"attack": "scaling", "defense": "trimmed_mean"},
    "scaling_krum": {"attack": "scaling", "defense": "krum"},
    "scaling_clip": {"attack": "scaling", "defense": "clipping"},

    # ================= DP EXPERIMENTS =================

    "dp_local_eps1": {
        "attack": None,
        "dp": "local",
        "noise": 2.0,
        "clip": 1.0
    },

    "dp_local_eps2": {
        "attack": None,
        "dp": "local",
        "noise": 1.0,
        "clip": 1.0
    },

    "dp_local_eps5": {
        "attack": None,
        "dp": "local",
        "noise": 0.5,
        "clip": 1.0
    },

    "dp_server_fixed": {
        "attack": None,
        "defense": "dp_server_fixed",
        "dp": None,
        "noise": 1.0,
        "clip": 1.0
    },

    "dp_server_adaptive": {
        "attack": None,
        "defense": "dp_server_adaptive",
        "dp": None,
        "noise": 1.0,
        "clip": 1.0
    },

    "dp_local_adaptive": {
        "attack": None,
        "dp": "local_adaptive",
        "noise": 1.0,
        "clip": 1.0,
        "defense": None
    },

    "final_system": {
    "attack": "label_flip",
    "dp": "hybrid_adaptive",
    "defense": "trust",  
    "noise": 1.5,
    "clip": 1.0
}
}