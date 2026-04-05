#utils/create_scaler.py
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

all_X = []

for i in range(1, 11):
    with open(f"data/client_{i}.pkl", "rb") as f:
        data = pickle.load(f)
        all_X.append(data["X"])

all_X = np.vstack(all_X)

scaler = StandardScaler()
scaler.fit(all_X)

with open("data/global_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Global scaler saved.")