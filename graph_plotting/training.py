import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# STYLE (A* LEVEL)
# =========================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
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
    "Baseline": ("baseline", "solid", 2.5),
    "Worst Attack": ("sign_flip_only", "dashed", 2.0),
    "Best Defense": ("label_flip_median", "dashdot", 2.0),
    "Final System": ("final_system", "solid", 3.5),
}

# =========================
# CREATE FIGURE
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

colors = sns.color_palette("tab10", n_colors=len(experiments))
# =========================
# CREDIT FIGURE
# =========================
fig, ax = plt.subplots(figsize=(7, 5))

for i, (label, (exp, linestyle, lw)) in enumerate(experiments.items()):
    rounds, _, f1 = load_history("credit", exp)
    f1_smooth = smooth_curve(f1)

    ax.plot(
        rounds,
        f1_smooth,
        label=label,
        linestyle=linestyle,
        linewidth=lw,
        color=colors[i],
        alpha=0.95
    )

ax.set_title("Credit Dataset (F1 Score)", fontsize=14, weight="bold")
ax.set_xlabel("Communication Rounds")
ax.set_ylabel("F1 Score")
ax.grid(True, linestyle="--", alpha=0.4)
ax.minorticks_on()

# ✅ Legend per figure
ax.legend(frameon=False, fontsize=11)

plt.tight_layout()

# ✅ Save separately
plt.savefig("results/credit_training_dynamics.pdf", dpi=600, bbox_inches="tight")
plt.savefig("results/credit_training_dynamics.png", dpi=600, bbox_inches="tight")

plt.show()