# # plot_adult_heatmap_final.py
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# DATASET = "adult"
# SAVE_DIR = f"results/plots/{DATASET}/trust"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # ==============================
# # LOAD
# # ==============================
# with open(f"results/trust/{DATASET}/trust_log.json") as f:
#     data = json.load(f)

# data = sorted(data, key=lambda x: x["round"])
# rounds = [d["round"] for d in data]
# client_ids = data[0]["client_ids"]

# trust_matrix = np.array([d["trust_scores"] for d in data])

# # ==============================
# # FIX 1: SORT CLIENTS (IMPORTANT)
# # ==============================
# avg_trust = trust_matrix.mean(axis=0)
# sorted_idx = np.argsort(-avg_trust)

# trust_matrix = trust_matrix[:, sorted_idx]
# client_ids = [client_ids[i] for i in sorted_idx]

# # ==============================
# # FIX 3: LARGE FONTS (DOUBLE COLUMN SAFE)
# # ==============================
# plt.rcParams.update({
#     "font.size": 20,
#     "axes.titlesize": 22,
#     "axes.labelsize": 20,
#     "xtick.labelsize": 18,
#     "ytick.labelsize": 18
# })

# # ==============================
# # FIX 2: GLOBAL COLOR SCALE
# # ==============================
# vmin, vmax = 0.04, 0.45

# # ==============================
# # PLOT
# # ==============================
# plt.figure(figsize=(10,5))
# im = plt.imshow(trust_matrix, aspect='auto', vmin=vmin, vmax=vmax)

# cbar = plt.colorbar(im)
# cbar.set_label("Trust Score", fontsize=20)

# plt.xlabel("Clients")
# plt.ylabel("Rounds")

# # FIX 4: FORMAL TITLE
# plt.title("Trust Distribution Across Clients (Adult)")

# plt.xticks(
#     ticks=np.arange(len(client_ids)),
#     labels=[f"C{i}" for i in client_ids]
# )

# plt.tight_layout()
# plt.savefig(f"{SAVE_DIR}/adult_heatmap_final.png", dpi=300)
# plt.close()

# print("Adult heatmap saved (A*-ready).")


# plot_credit_heatmap_final.py
import json
import numpy as np
import matplotlib.pyplot as plt
import os

DATASET = "credit"
SAVE_DIR = f"results/plots/{DATASET}/trust"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================
# LOAD
# ==============================
with open(f"results/trust/{DATASET}/trust_log.json") as f:
    data = json.load(f)

data = sorted(data, key=lambda x: x["round"])
rounds = [d["round"] for d in data]
client_ids = data[0]["client_ids"]

trust_matrix = np.array([d["trust_scores"] for d in data])

# ==============================
# FIX 1: SORT CLIENTS (CONSISTENT)
# ==============================
avg_trust = trust_matrix.mean(axis=0)
sorted_idx = np.argsort(-avg_trust)

trust_matrix = trust_matrix[:, sorted_idx]
client_ids = [client_ids[i] for i in sorted_idx]

# ==============================
# STYLE (LARGE FONTS)
# ==============================
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18
})

# ==============================
# FIX 2: SAME COLOR SCALE
# ==============================
vmin, vmax = 0.04, 0.45

# ==============================
# PLOT
# ==============================
plt.figure(figsize=(10,5))
im = plt.imshow(trust_matrix, aspect='auto', vmin=vmin, vmax=vmax)

cbar = plt.colorbar(im)
cbar.set_label("Trust Score", fontsize=20)

plt.xlabel("Clients")
plt.ylabel("Rounds")

# FIX 4: FORMAL TITLE
plt.title("Trust Distribution Across Clients (Credit)")

plt.xticks(
    ticks=np.arange(len(client_ids)),
    labels=[f"C{i}" for i in client_ids]
)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/credit_heatmap_final.png", dpi=300)
plt.close()

print("Credit heatmap saved (A*-ready).")