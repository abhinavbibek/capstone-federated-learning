#attacks/feature_poisoning.py
import numpy as np

def feature_poison(X, severity=50.0):
    X = X.copy()

    # poison top important features aggressively
    important_features = list(range(20))  # more features

    for f in important_features:
        noise = np.random.normal(0, severity, size=X.shape[0])
        X[:, f] += noise

    # add distribution shift
    X *= np.random.uniform(1.5, 3.0)

    return X