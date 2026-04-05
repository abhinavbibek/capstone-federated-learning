# non_iid_splits.py
import pickle
import numpy as np
import os

# ==============================
# SELECT DATASET
# ==============================
DATASET = "credit"   # "adult" or "credit"

print("="*60)
print(f"CREATING NON-IID CLIENT DATA SPLITS ({DATASET.upper()})")
print("="*60)

# ==============================
# LOAD DATA
# ==============================
if DATASET == "adult":
    with open('data/adult_train.pkl', 'rb') as f:
        data = pickle.load(f)
    save_path = "data"

elif DATASET == "credit":
    with open('data/credit_train.pkl', 'rb') as f:
        data = pickle.load(f)
    save_path = "data"


X = data['X'].values if hasattr(data['X'], "values") else data['X']
y = data['y']

n_clients = 10
samples_per_client = 2000

# ==============================
# CLASS SPLIT
# ==============================
idx_class0 = np.where(y == 0)[0]
idx_class1 = np.where(y == 1)[0]

np.random.seed(42)
np.random.shuffle(idx_class0)
np.random.shuffle(idx_class1)

splits_class0 = np.array_split(idx_class0, n_clients)
splits_class1 = np.array_split(idx_class1, n_clients)

# ==============================
# DISTRIBUTIONS
# ==============================
if DATASET == "adult":
    distributions = [
        (0.7, 0.3), (0.65, 0.35), (0.6, 0.4), (0.55, 0.45),
        (0.5, 0.5), (0.45, 0.55), (0.4, 0.6), (0.35, 0.65),
        (0.3, 0.7), (0.5, 0.5),
    ]

elif DATASET == "credit":
    # MUCH MORE EXTREME (fraud ≈ 0.17%)
    distributions = [
        (0.995, 0.005),
        (0.99, 0.01),
        (0.98, 0.02),
        (0.97, 0.03),
        (0.95, 0.05),
        (0.93, 0.07),
        (0.90, 0.10),
        (0.85, 0.15),
        (0.80, 0.20),
        (0.90, 0.10),
    ]

# ==============================
# CREATE CLIENT DATA
# ==============================
for i, (ratio0, ratio1) in enumerate(distributions):

    client_id = i + 1

    n0 = int(samples_per_client * ratio0)
    n1 = int(samples_per_client * ratio1)

    def sample_with_replacement(arr, n):
        if len(arr) == 0:
            return np.array([], dtype=int)

        if len(arr) >= n:
            return np.random.choice(arr, n, replace=False)
        else:
            return np.random.choice(arr, n, replace=True)

    client_idx0 = sample_with_replacement(splits_class0[i], n0)
    client_idx1 = sample_with_replacement(splits_class1[i], n1)

    client_indices = np.concatenate([client_idx0, client_idx1])
    np.random.shuffle(client_indices)

    client_X = X[client_indices].astype('float32')
    client_y = y[client_indices]

    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/{DATASET}_client_{client_id}.pkl', 'wb') as f:
        pickle.dump({'X': client_X, 'y': client_y}, f)
    

    print(f"Client {client_id}: Class0={sum(client_y==0)}, Class1={sum(client_y==1)}")

print("\n NON-IID SPLIT COMPLETED")