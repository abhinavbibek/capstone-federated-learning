#attacks/label_flipping.py
import numpy as np

def label_flip(y, flip_ratio=0.6):
    y = y.copy()
    mask = np.random.rand(len(y)) < flip_ratio
    y[mask] = 1 - y[mask]
    return y

# def targeted_label_flip(y):
#     y = y.copy()
#     y[y == 1] = 0
#     return y

# def targeted_label_flip(y):
#     y = y.copy()
#     mask = (y == 1)
#     y[mask] = 0
#     return y

def targeted_label_flip(y, ratio=0.5):
    y = y.copy()
    idx = np.where(y == 1)[0]
    flip_idx = np.random.choice(idx, int(len(idx)*ratio), replace=False)
    y[flip_idx] = 0
    return y