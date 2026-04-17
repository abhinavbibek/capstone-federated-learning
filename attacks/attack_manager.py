#attacks/attack_manager.py
import numpy as np
from attacks.label_flipping import label_flip, targeted_label_flip
from attacks.feature_poisoning import feature_poison
from attacks.model_poisoning import sign_flipping, scaling_attack

def apply_data_poisoning(X, y, attack_type):
    if attack_type == "label_flip":
        y = label_flip(y, flip_ratio=0.3)
    elif attack_type == "targeted_flip":
        y = targeted_label_flip(y)
    elif attack_type == "feature_poison":
        X = feature_poison(X)
    return X, y


def apply_model_poisoning(weights, attack_type):
    if attack_type == "sign_flip":
        weights = sign_flipping(weights)
    elif attack_type == "scaling":
        weights = scaling_attack(weights)
    return weights