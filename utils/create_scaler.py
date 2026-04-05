# utils/create_scaler.py
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==============================
# SELECT DATASET
# ==============================
DATASET = "credit"   # "adult" or "credit"

all_X = []

for i in range(1, 11):
    with open(f"data/{DATASET}_client_{i}.pkl", "rb") as f:
        data = pickle.load(f)
        all_X.append(data["X"])

all_X = np.vstack(all_X)

print(f"Total samples used for scaler: {all_X.shape}")

scaler = StandardScaler()
scaler.fit(all_X)

with open(f"data/{DATASET}_global_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"{DATASET.upper()} Global scaler saved.")