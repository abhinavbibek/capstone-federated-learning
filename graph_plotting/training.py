# analysis/plots/plot_grouped_training.py

import json
import numpy as np
import matplotlib.pyplot as plt
import os

DATASET = "credit"  # change to "adult" when needed

# ==============================
# METRIC SELECTION
# ==============================
if DATASET == "credit":
    METRIC = "f1"   # 🔥 IMPORTANT
    ylabel = "F1 Score"
else:
    METRIC = "accuracy"
    ylabel = "Accuracy"

# ==============================
# SAVE PATH
# ==============================
SAVE_DIR = f"results/plots/{DATASET}/grouped"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================
# ATTACK GROUPS
# ==============================
ATTACKS = {
    "label_flip": [
        "baseline",
        "label_flip_only",
        "label_flip_median",
        "label_flip_trimmed",
        "label_flip_krum",
        "label_flip_clip",
        "final_system"
    ],
    "targeted_flip": [
        "baseline",
        "targeted_flip_only",
        "targeted_flip_median",
        "targeted_flip_trimmed",
        "targeted_flip_krum",
        "targeted_flip_clip",
        "final_system"
    ],
    "feature_poison": [
        "baseline",
        "feature_poison_only",
        "feature_poison_median",
        "feature_poison_trimmed",
        "feature_poison_krum",
        "feature_poison_clip",
        "final_system"
    ],
    "sign_flip": [
        "baseline",
        "sign_flip_only",
        "sign_flip_median",
        "sign_flip_trimmed",
        "sign_flip_krum",
        "sign_flip_clip",
        "final_system"
    ],
    "scaling": [
        "baseline",
        "scaling_only",
        "scaling_median",
        "scaling_trimmed",
        "scaling_krum",
        "scaling_clip",
        "final_system"
    ]
}

# ==============================
# UTIL FUNCTIONS
# ==============================
def load_metric(exp):
    with open(f"results/{DATASET}_{exp}.json") as f:
        data = json.load(f)
    return np.array([d[METRIC] for d in data])

def ema(x, alpha=0.6):
    out = []
    prev = x[0]
    for val in x:
        prev = alpha * val + (1 - alpha) * prev
        out.append(prev)
    return np.array(out)

# ==============================
# COLOR SCHEME
# ==============================
COLORS = {
    "baseline": "black",
    "only": "red",
    "median": "#1f77b4",
    "trimmed": "#2ca02c",
    "krum": "#ff7f0e",
    "clip": "#9467bd",
    "final": "#17becf"
}

# ==============================
# PLOT
# ==============================
for attack, exps in ATTACKS.items():

    plt.figure(figsize=(8,5))

    for exp in exps:
        y = load_metric(exp)
        rounds = np.arange(1, len(y)+1)

        y_smooth = ema(y)

        std = np.std(y) * 0.15  # replace later with multi-seed

        # COLOR
        if exp == "baseline":
            color = COLORS["baseline"]
        elif "only" in exp:
            color = COLORS["only"]
        elif "median" in exp:
            color = COLORS["median"]
        elif "trimmed" in exp:
            color = COLORS["trimmed"]
        elif "krum" in exp:
            color = COLORS["krum"]
        elif "clip" in exp:
            color = COLORS["clip"]
        elif "final" in exp:
            color = COLORS["final"]
        else:
            color = "gray"

        linestyle = "--" if "only" in exp else "-"
        linewidth = 2.5 if "final" in exp else 1.8

        plt.plot(
            rounds, y_smooth,
            label=exp,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth
        )

        plt.fill_between(
            rounds,
            y_smooth - std,
            y_smooth + std,
            color=color,
            alpha=0.12
        )

    plt.xlabel("Communication Rounds")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} under {attack.replace('_',' ').title()} Attack")

    plt.grid(alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()

    plt.savefig(f"{SAVE_DIR}/{attack}_{METRIC}.png", dpi=300)
    plt.close()

print("Grouped plots saved successfully.")