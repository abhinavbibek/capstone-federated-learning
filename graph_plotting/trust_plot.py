# analysis/plots/plot_trust_heatmap.py
import json
import numpy as np
import matplotlib.pyplot as plt
import os
DATASET = "adult"
SAVE_DIR = f"results/plots/{DATASET}/trust"
os.makedirs(SAVE_DIR, exist_ok=True)
# ==============================
# LOAD TRUST LOG
# ==============================
with open(f"results/trust/{DATASET}/trust_log.json") as f:
    data = json.load(f)
# Sort by round
data = sorted(data, key=lambda x: x["round"])
rounds = [d["round"] for d in data]
client_ids = data[0]["client_ids"]
# ==============================
# BUILD MATRIX
# ==============================
trust_matrix = []
for d in data:
    trust_matrix.append(d["trust_scores"])
trust_matrix = np.array(trust_matrix)  # shape: (rounds, clients)
# ==============================
# PLOT HEATMAP
# ==============================
plt.figure(figsize=(9,4))
im = plt.imshow(trust_matrix, aspect='auto')
plt.colorbar(im, label="Trust Score")
plt.xlabel("Clients")
plt.ylabel("Rounds")
plt.xticks(
    ticks=np.arange(len(client_ids)),
    labels=[f"C{i}" for i in client_ids],
    rotation=0
)
plt.yticks(
    ticks=np.arange(len(rounds)),
    labels=rounds
)
plt.title("Client Trust Dynamics Across Rounds")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/trust_heatmap.png", dpi=300)
plt.close()
print("Trust heatmap saved.")