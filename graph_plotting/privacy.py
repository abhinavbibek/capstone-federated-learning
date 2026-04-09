# analysis/plots/plot_privacy_bubble.py

import json
import numpy as np
import matplotlib.pyplot as plt
import os

DATASET = "credit"

SAVE_DIR = f"results/plots/{DATASET}/privacy"
os.makedirs(SAVE_DIR, exist_ok=True)

EXPERIMENTS = [
    "dp_local_eps1",
    "dp_local_eps2",
    "dp_local_eps5",
    "dp_local_adaptive",
    "dp_server_fixed",
    "dp_server_adaptive",
    "final_system"
]

eps, f1, auc, labels = [], [], [], []

for exp in EXPERIMENTS:
    try:
        with open(f"results/{DATASET}_{exp}.json") as f:
            data = json.load(f)

        final = data[-1]

        eps.append(final.get("epsilon", 0))
        f1.append(final.get("f1", 0))
        auc.append(final.get("auc", 0))
        labels.append(exp)

    except:
        continue

eps = np.array(eps)
f1 = np.array(f1)
auc = np.array(auc)

# Normalize bubble size
sizes = (auc - auc.min()) / (auc.max() - auc.min() + 1e-6)
sizes = 200 + sizes * 800

# ==============================
# PLOT
# ==============================
plt.figure(figsize=(7,5))

scatter = plt.scatter(
    eps, f1,
    s=sizes,
    alpha=0.6
)

# Highlight final system
for i, label in enumerate(labels):
    if "final" in label:
        plt.scatter(
            eps[i], f1[i],
            s=400,
            edgecolor="black",
            linewidth=2
        )

# Labels
for i, label in enumerate(labels):
    plt.text(eps[i], f1[i], label, fontsize=7)

plt.xlabel("Privacy Budget (ε)")
plt.ylabel("F1 Score")
plt.title("Privacy–Utility–Performance Tradeoff (Bubble = AUC)")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(f"{SAVE_DIR}/bubble_tradeoff.png", dpi=300)
plt.close()

print("Bubble plot saved.")