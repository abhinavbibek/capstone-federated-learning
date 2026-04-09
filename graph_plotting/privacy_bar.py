# analysis/plots/plot_privacy_bar.py

import json
import numpy as np
import matplotlib.pyplot as plt
import os

DATASET = "credit"
SAVE_DIR = f"results/plots/{DATASET}/privacy"
os.makedirs(SAVE_DIR, exist_ok=True)

EXPERIMENTS = [
    "baseline",
    "label_flip_only",
    "label_flip_median",
    "dp_local_eps2",
    "final_system"
]

METRIC = "mia"  # change to entropy / leakage if needed

names = []
values = []

for exp in EXPERIMENTS:
    try:
        with open(f"results/{DATASET}_{exp}.json") as f:
            data = json.load(f)

        final = data[-1]

        names.append(exp)
        values.append(final[METRIC])

    except:
        continue

# Sort (worst → best)
sorted_idx = np.argsort(values)[::-1]

names = [names[i] for i in sorted_idx]
values = [values[i] for i in sorted_idx]

# ==============================
# PLOT
# ==============================
plt.figure(figsize=(7,4))

bars = plt.bar(names, values, alpha=0.7)

# Highlight final system
for i, name in enumerate(names):
    if "final" in name:
        bars[i].set_edgecolor("black")
        bars[i].set_linewidth(2)

plt.xticks(rotation=30, ha='right')
plt.ylabel(METRIC.upper())
plt.title(f"Privacy Leakage ({METRIC.upper()}) Comparison")

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig(f"{SAVE_DIR}/privacy_{METRIC}_bar.png", dpi=300)
plt.close()

print("Privacy bar plot saved.")