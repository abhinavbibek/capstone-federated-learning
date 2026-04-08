#attacks/label_flipping.py
import numpy as np

def label_flip(y, flip_ratio=0.3):
    y = y.copy()
    mask = np.random.rand(len(y)) < flip_ratio
    y[mask] = 1 - y[mask]
    return y

def targeted_label_flip(y):
    y = y.copy()
    y[y == 1] = 0
    return y