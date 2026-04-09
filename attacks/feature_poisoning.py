#attacks/feature_poisoning.py
import numpy as np

def feature_poison(X, severity=0.8, flip_prob=0.6):
    X = X.copy()

    # Auto severity based on standardized data
    if severity is None:
        severity = 0.3   # better default for scaled features

    num_features = X.shape[1]
    selected = np.random.choice(num_features, size=int(num_features * 0.3), replace=False)

    for f in selected:
        mask = np.random.rand(X.shape[0]) < flip_prob
        noise = np.random.normal(0, severity, size=np.sum(mask))
        X[mask, f] += noise

    return X