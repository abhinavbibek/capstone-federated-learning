import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# STYLE (UPGRADED - LARGE FONTS)
# =========================
sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({
    "font.family": "serif",

    # 🔥 Large, publication-ready fonts
    "font.size": 25,
    "axes.titlesize": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 19,

    # aesthetics
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.3,
})

# =========================
# LOAD FUNCTION
# =========================
def load_history(dataset, exp):
    path = f"results/{dataset}_{exp}.json"
    with open(path) as f:
        data = json.load(f)
    
    rounds = [d["round"] for d in data]
    acc = [d["accuracy"] for d in data]
    f1 = [d["f1"] for d in data]
    
    return rounds, acc, f1


# =========================
# SMOOTHING (EMA)
# =========================
def smooth_curve(values, alpha=0.3):
    smoothed = []
    for i, v in enumerate(values):
        if i == 0:
            smoothed.append(v)
        else:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


# =========================
# SELECT EXPERIMENTS
# =========================
experiments = {
    "Baseline": ("baseline", "solid", 2.8),
    "Worst Attack": ("sign_flip_only", "dashed", 2.5),
    "Best Defense": ("label_flip_median", "dashdot", 2.5),
    "TAP-FL": ("final_system", "solid", 4.0),
}

# =========================
# COLOR PALETTE
# =========================
colors = sns.color_palette("tab10", n_colors=len(experiments))

# =========================
# ADULT FIGURE (ACCURACY)
# =========================
fig, ax = plt.subplots(figsize=(8, 6))

for i, (label, (exp, linestyle, lw)) in enumerate(experiments.items()):
    rounds, acc, _ = load_history("adult", exp)
    acc_smooth = smooth_curve(acc)

    ax.plot(
        rounds,
        acc_smooth,
        label=label,
        linestyle=linestyle,
        linewidth=lw,
        color=colors[i],
        alpha=0.95
    )

# =========================
# AXIS + TITLE
# =========================

ax.set_title("Adult Dataset (Accuracy)", fontsize=25, weight="bold", pad=10)
ax.set_xlabel("Communication Rounds", weight="bold")
ax.set_ylabel("Accuracy", weight="bold")

ax.grid(True, linestyle="--", alpha=0.4)
ax.minorticks_on()

# =========================
# LEGEND
# =========================
# =========================
# LEGEND (TOP, HORIZONTAL)
# =========================
# =========================
# LEGEND (COMPACT, TOP)
# =========================
# =========================
# LEGEND (FIGURE LEVEL - CLEAN FIX)
# =========================
handles, labels = ax.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),   # 🔥 above everything (title included)
    ncol=2,
    frameon=True,
    fontsize=16,
    edgecolor="black",
    columnspacing=1.2,
    handlelength=2.0
)
# =========================
# SAVE
# =========================
plt.tight_layout(rect=[0, 0, 1, 0.92])  # leaves space on top

plt.savefig("results/adult_training_dynamics.pdf", dpi=600, bbox_inches="tight")
plt.savefig("results/adult_training_dynamics.png", dpi=600, bbox_inches="tight")

plt.show()