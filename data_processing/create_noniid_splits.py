import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

print("="*60)
print("CREATING NON-IID CLIENT DATA SPLITS (FIXED)")
print("="*60)

# Load data
with open('data/train.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X'].values if hasattr(data['X'], "values") else data['X']
y = data['y']

n_clients = 10
samples_per_client = 2000

# Get indices
idx_class0 = np.where(y == 0)[0]
idx_class1 = np.where(y == 1)[0]

# Shuffle
np.random.seed(42)
np.random.shuffle(idx_class0)
np.random.shuffle(idx_class1)

# Split WITHOUT overlap
splits_class0 = np.array_split(idx_class0, n_clients)
splits_class1 = np.array_split(idx_class1, n_clients)

# distributions = [
#     (0.9, 0.1),
#     (0.8, 0.2),
#     (0.7, 0.3),
#     (0.6, 0.4),
#     (0.5, 0.5),
#     (0.4, 0.6),
#     (0.3, 0.7),
#     (0.2, 0.8),
#     (0.1, 0.9),
#     (0.5, 0.5),
# ]
distributions = [
    (0.7, 0.3),
    (0.65, 0.35),
    (0.6, 0.4),
    (0.55, 0.45),
    (0.5, 0.5),
    (0.45, 0.55),
    (0.4, 0.6),
    (0.35, 0.65),
    (0.3, 0.7),
    (0.5, 0.5),
]


for i, (ratio0, ratio1) in enumerate(distributions):

    client_id = i + 1

    n0 = int(samples_per_client * ratio0)
    n1 = int(samples_per_client * ratio1)

    # Take from split (NO overlap)
    def sample_with_replacement(arr, n):
        if len(arr) >= n:
            return arr[:n]
        else:
            return np.random.choice(arr, n, replace=True)

    client_idx0 = sample_with_replacement(splits_class0[i], n0)
    client_idx1 = sample_with_replacement(splits_class1[i], n1)

    client_indices = np.concatenate([client_idx0, client_idx1])
    np.random.shuffle(client_indices)

    client_X = X[client_indices]
    client_y = y[client_indices]
    client_X = client_X.astype('float32')

    os.makedirs('data', exist_ok=True)
    with open(f'data/client_{client_id}.pkl', 'wb') as f:
        pickle.dump({'X': client_X, 'y': client_y}, f)

    print(f"Client {client_id}: Class0={sum(client_y==0)}, Class1={sum(client_y==1)}")

print("\n FIXED NON-IID SPLIT COMPLETED")
