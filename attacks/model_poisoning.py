#attacks/model_poisoning.py
import numpy as np
import torch  


def sign_flipping(weights, scale=1.5):
    return {k: -scale * v for k, v in weights.items()}


def scaling_attack(weights, scale=2.0, max_norm=5.0):
    poisoned = {}

    for k, v in weights.items():
        w = v * scale
        if isinstance(w, torch.Tensor):
            norm = torch.norm(w).item() 
        else:
            norm = np.linalg.norm(w)

        if norm > max_norm:
            w = w * (max_norm / (norm + 1e-6))

        poisoned[k] = w

    return poisoned