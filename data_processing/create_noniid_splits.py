# non_iid_splits.py 
import pickle
import numpy as np
import os

DATASET = "credit"   # adult or credit
print(f"Dirichket non-iid split ({DATASET.upper()})")
if DATASET == "adult":
    with open('data/adult_train.pkl', 'rb') as f:
        data = pickle.load(f)
    save_path = "data"
    alpha = 0.5  
elif DATASET == "credit":
    with open('data/credit_train.pkl', 'rb') as f:
        data = pickle.load(f)
    save_path = "data"
    alpha = 0.05 
X = data['X'].values if hasattr(data['X'], "values") else data['X']
y = data['y']
n_clients = 10
samples_per_client = 2000
np.random.seed(42)

class_indices = {
    0: np.where(y == 0)[0],
    1: np.where(y == 1)[0]
}
for c in class_indices:
    np.random.shuffle(class_indices[c])


def dirichlet_partition(class_indices, n_clients, alpha):
    client_indices = {i: [] for i in range(n_clients)}
    for cls, indices in class_indices.items():
        proportions = np.random.dirichlet(alpha * np.ones(n_clients))
        counts = (proportions * len(indices)).astype(int)
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


def generate_valid_split():
    for attempt in range(50): 
        client_indices = dirichlet_partition(class_indices, n_clients, alpha)
        valid = True
        for cid in range(n_clients):
            if len(client_indices[cid]) == 0:
                valid = False
                break
        if valid:
            print(f"Valid split found (attempt {attempt+1})")
            return client_indices

    raise RuntimeError("Failed to generate valid Dirichlet split")

client_indices = generate_valid_split()

def adjust_client_size(indices, target_size):
    indices = np.array(indices)
    if len(indices) >= target_size:
        return np.random.choice(indices, target_size, replace=False)
    else:
        extra = np.random.choice(indices, target_size - len(indices), replace=True)
        return np.concatenate([indices, extra])


os.makedirs(save_path, exist_ok=True)
for cid in range(n_clients):
    indices = adjust_client_size(client_indices[cid], samples_per_client)
    client_X = X[indices].astype('float32')
    client_y = y[indices]
    with open(f'{save_path}/{DATASET}_client_{cid+1}.pkl', 'wb') as f:
        pickle.dump({'X': client_X, 'y': client_y}, f)
    c0 = sum(client_y == 0)
    c1 = sum(client_y == 1)
    ratio = np.mean(client_y)
    print(
        f"Client {cid+1}: "
        f"Class0={c0}, Class1={c1}, Ratio={ratio:.5f}"
    )