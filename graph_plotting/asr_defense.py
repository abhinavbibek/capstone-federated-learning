# analysis/plots/plot_asr_vs_accuracy.py

import json
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results/plots", exist_ok=True)

DATASET = "adult"

EXPERIMENTS = [
    "label_flip_only",
    "label_flip_median",
    "label_flip_trimmed",
    "label_flip_krum",
    "label_flip_clip",
    "final_system"
]

# ==============================
# LOAD DATA
# ==============================
results = []

for exp in EXPERIMENTS:
    with open(f"results/{DATASET}_{exp}.json") as f:
        data = json.load(f)

    final = data[-1]

    results.append({
        "exp": exp,
        "asr": final["asr"],
        "acc": final["accuracy"]
    })

# Sort by ASR
results = sorted(results, key=lambda x: x["asr"])

labels = [r["exp"] for r in results]
asr = [r["asr"] for r in results]
acc = [r["acc"] for r in results]

x = np.arange(len(labels))

# ==============================
# PLOT
# ==============================
fig, ax1 = plt.subplots(figsize=(9,5))

# BAR → ASR
bars = ax1.bar(x, asr, alpha=0.6)
ax1.set_ylabel("Attack Success Rate (ASR)", fontsize=11)

# Highlight final system
for i, label in enumerate(labels):
    if "final" in label:
        bars[i].set_alpha(1.0)
        bars[i].set_edgecolor("black")
        bars[i].set_linewidth(1.5)

# LINE → Accuracy
ax2 = ax1.twinx()
ax2.plot(x, acc, marker='o', linewidth=2)
ax2.set_ylabel("Accuracy", fontsize=11)

# X-axis
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)

# Grid
ax1.grid(axis='y', alpha=0.3)

plt.title("Security–Utility Tradeoff across Defenses", fontsize=12)
plt.tight_layout()

plt.savefig("results/plots/adult_asr_vs_accuracy.png", dpi=300)
plt.close()