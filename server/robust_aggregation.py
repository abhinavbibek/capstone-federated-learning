#server/robust_aggregation.py
import numpy as np


def median_aggregation(weights_list):
    """Coordinate-wise median"""
    stacked = np.stack(weights_list, axis=0)
    return np.median(stacked, axis=0)


def trimmed_mean_aggregation(weights_list, trim_ratio=0.2):
    """Trim extremes and average remaining"""
    stacked = np.stack(weights_list, axis=0)

    n_clients = stacked.shape[0]
    trim_k = int(n_clients * trim_ratio)

    sorted_weights = np.sort(stacked, axis=0)

    trimmed = sorted_weights[trim_k: n_clients - trim_k]

    return np.mean(trimmed, axis=0)

def krum_aggregation(weights, f=2):
    n = len(weights)
    scores = []

    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                dist = sum(
                    np.linalg.norm(w1 - w2)**2
                    for w1, w2 in zip(weights[i], weights[j])
                )
                distances.append(dist)

        distances.sort()
        score = sum(distances[:n - f - 2])
        scores.append(score)

    selected_idx = np.argmin(scores)
    return weights[selected_idx]