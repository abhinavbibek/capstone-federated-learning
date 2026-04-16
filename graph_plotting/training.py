import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# =========================
# STYLE (PUBLICATION READY)
# =========================
sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({
    "font.family": "serif",

    # Main fonts
    "font.size": 25,
    "axes.titlesize": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,

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
# EXPERIMENTS (SHORT LABELS 🔥)
# =========================
experiments = {
    "FedAvg": ("baseline", "solid", 2.8),
    "Sign Flip Attack": ("sign_flip_only", "dashed", 2.5),
    "Krum Defense": ("label_flip_only", "dashdot", 2.5),
    "TAP-FL": ("dp_local_eps5", "solid", 4.0),
}

# =========================
# COLORS
# =========================
colors = sns.color_palette("tab10", n_colors=len(experiments))

# =========================
# FIGURE
# =========================
fig, ax = plt.subplots(figsize=(8, 6))

for i, (label, (exp, linestyle, lw)) in enumerate(experiments.items()):
    rounds, acc, _ = load_history("adult", exp)
    f1_smooth = smooth_curve(acc)

    ax.plot(
        rounds,
        f1_smooth,
        label=label,
        linestyle=linestyle,
        linewidth=lw,
        color=colors[i],
        alpha=0.95
    )

# =========================
# AXIS + TITLE
# =========================
ax.set_title("Adult Dataset (Accuracy)", weight="bold", pad=10)
ax.set_xlabel("Communication Rounds", weight="bold")
ax.set_ylabel("Accuracy", weight="bold")

ax.grid(True, linestyle="--", alpha=0.4)
ax.minorticks_on()

# =========================
# LEGEND (MAJOR FIX 🔥)
# =========================
handles, labels = ax.get_legend_handles_labels()

legend_font = FontProperties(
    family='sans-serif',   # 🔥 more readable
    size=17,
    weight='bold'
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=2,                     # try ncol=4 if you want single row
    frameon=True,
    prop=legend_font,
    edgecolor="black",
    columnspacing=1.5,
    handlelength=2.5,
    labelspacing=0.4,
    borderpad=0.5,
    handletextpad=0.6
)

# =========================
# LAYOUT + SAVE
# =========================
plt.tight_layout(rect=[0, 0, 1, 0.92])

plt.savefig("results/adult_training_dynamics.pdf", dpi=600, bbox_inches="tight")
plt.savefig("results/adult_training_dynamics.png", dpi=600, bbox_inches="tight")

plt.show()