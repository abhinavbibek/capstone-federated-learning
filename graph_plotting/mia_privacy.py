# analysis/plots/plot_privacy_radar.py

import json
import numpy as np
import matplotlib.pyplot as plt
import os

DATASET = "credit"

SAVE_DIR = f"results/plots/{DATASET}/privacy"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================
# EXPERIMENTS TO COMPARE
# ==============================
EXPERIMENTS = [
    "baseline",
    "label_flip_only",
    "label_flip_median",
    "dp_local_eps2",
    "final_system"
]

METRICS = ["mia", "entropy", "confidence_gap", "leakage"]

# ==============================
# LOAD DATA
# ==============================
data_dict = {}

for exp in EXPERIMENTS:
    try:
        with open(f"results/{DATASET}_{exp}.json") as f:
            data = json.load(f)

        final = data[-1]

        data_dict[exp] = [final[m] for m in METRICS]

    except:
        print(f"Skipping {exp}")

# Convert to array
values = np.array(list(data_dict.values()))

# ==============================
# NORMALIZATION (0–1)
# ==============================
min_vals = values.min(axis=0)
max_vals = values.max(axis=0)

norm_values = (values - min_vals) / (max_vals - min_vals + 1e-6)

# ==============================
# RADAR SETUP
# ==============================
labels = METRICS
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)

# ==============================
# PLOT EACH EXPERIMENT
# ==============================
for i, (exp, vals) in enumerate(zip(data_dict.keys(), norm_values)):
    vals = vals.tolist()
    vals += vals[:1]

    linewidth = 2.5 if "final" in exp else 1.5
    alpha = 0.8 if "final" in exp else 0.5

    ax.plot(angles, vals, linewidth=linewidth, label=exp)
    ax.fill(angles, vals, alpha=0.1)

# Labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title("Privacy Leakage Comparison (Lower is Better)", fontsize=11)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/privacy_radar.png", dpi=300)
plt.close()

print("Radar plot saved.")