#attacks/feature_poisoning.py
import numpy as np

def feature_poison(X, severity=10.0):
    X = X.copy()

    important_features = [0, 1, 2, 3, 4]

    for f in important_features:
        noise = np.random.normal(0, severity, size=X.shape[0])
        X[:, f] = X[:, f] + noise

    return X