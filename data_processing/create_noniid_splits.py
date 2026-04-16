
# non_iid_splits.py 

import pickle
import numpy as np
import os

# ==============================
# SELECT DATASET
# ==============================
DATASET = "credit"   # "adult" or "credit"

print("="*60)
print(f"DIRICHLET NON-IID SPLIT ({DATASET.upper()})")
print("="*60)

# ==============================
# LOAD DATA
# ==============================
if DATASET == "adult":
    with open('data/adult_train.pkl', 'rb') as f:
        data = pickle.load(f)
    save_path = "data"
    alpha = 0.5   # moderate heterogeneity

elif DATASET == "credit":
    with open('data/credit_train.pkl', 'rb') as f:
        data = pickle.load(f)
    save_path = "data"
    alpha = 0.05  # strong heterogeneity (important)

X = data['X'].values if hasattr(data['X'], "values") else data['X']
y = data['y']

n_clients = 10
samples_per_client = 2000

np.random.seed(42)

# ==============================
# CLASS INDICES
# ==============================
class_indices = {
    0: np.where(y == 0)[0],
    1: np.where(y == 1)[0]
}

# Shuffle
for c in class_indices:
    np.random.shuffle(class_indices[c])

# ==============================
# DIRICHLET SPLIT FUNCTION
# ==============================
def dirichlet_partition(class_indices, n_clients, alpha):
    client_indices = {i: [] for i in range(n_clients)}

    for cls, indices in class_indices.items():

        proportions = np.random.dirichlet(alpha * np.ones(n_clients))

        # Convert proportions → counts
        counts = (proportions * len(indices)).astype(int)

        # Fix rounding issue
        diff = len(indices) - np.sum(counts)
        for i in range(diff):
            counts[i % n_clients] += 1

        start = 0
        for client_id in range(n_clients):
            client_indices[client_id].extend(
                indices[start:start + counts[client_id]]
            )
            start += counts[client_id]

    return client_indices


# ==============================
# GENERATE SPLIT (WITH VALIDATION)
# ==============================
def generate_valid_split():
    for attempt in range(20):  # retry if bad split

        client_indices = dirichlet_partition(class_indices, n_clients, alpha)

        valid = True

        for cid in range(n_clients):
            labels = y[client_indices[cid]]

            if len(labels) == 0:
                valid = False
                break

            fraud_ratio = np.mean(labels)

            if DATASET == "credit":
                # avoid degenerate clients
                if fraud_ratio < 0.0005 or fraud_ratio > 0.05:
                    valid = False
                    break

            else:
                # adult (balanced-ish)
                if fraud_ratio < 0.05 or fraud_ratio > 0.95:
                    valid = False
                    break

        if valid:
            print(f"Valid split found (attempt {attempt+1})")
            return client_indices

    raise RuntimeError("Failed to generate valid Dirichlet split")


client_indices = generate_valid_split()

# ==============================
# FIX SAMPLES PER CLIENT
# ==============================
def adjust_client_size(indices, target_size):
    indices = np.array(indices)

    if len(indices) >= target_size:
        return np.random.choice(indices, target_size, replace=False)
    else:
        extra = np.random.choice(indices, target_size - len(indices), replace=True)
        return np.concatenate([indices, extra])


# ==============================
# CREATE CLIENT DATA
# ==============================
os.makedirs(save_path, exist_ok=True)

for cid in range(n_clients):

    indices = adjust_client_size(client_indices[cid], samples_per_client)

    client_X = X[indices].astype('float32')
    client_y = y[indices]

    with open(f'{save_path}/{DATASET}_client_{cid+1}.pkl', 'wb') as f:
        pickle.dump({'X': client_X, 'y': client_y}, f)

    print(
        f"Client {cid+1}: "
        f"Class0={sum(client_y==0)}, "
        f"Class1={sum(client_y==1)}, "
        f"Ratio={np.mean(client_y):.4f}"
    )

print("\n DIRICHLET NON-IID SPLIT COMPLETED")