#attacks/label_flipping.py
import numpy as np

# 1. Simple flipping (already)
def label_flip(y):
    return 1 - y


# 2. Targeted flipping (ONLY flip 1 → 0)
def targeted_label_flip(y):
    y = y.copy()
    y[y == 1] = 0
    return y