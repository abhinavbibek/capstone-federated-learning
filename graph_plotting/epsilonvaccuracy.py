import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# STYLE (LARGER + CLEAN)
# =========================
sns.set_theme(style="ticks")

plt.rcParams.update({
    "font.family": "serif",

    "font.size": 26,
    "axes.titlesize": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 22,

    "axes.spines.top": False,
    "axes.spines.right": False,
})

# =========================
# LOAD
# =========================
adult_df = pd.read_csv("results_summary_adult_round_40.csv")
credit_df = pd.read_csv("results_summary_credit_round_40.csv")

# =========================
# NOISE MAP
# =========================
NOISE_MAP = {
    "dp_local_eps1": 2.0,
    "dp_local_eps2": 1.0,
    "dp_local_eps5": 0.5,
    "dp_local_adaptive": 1.0,
    "final_system": 1.0
}

# =========================
# PREP
# =========================
def prepare(df):

    baseline = df[df["experiment"].str.contains("baseline")].copy()
    baseline["group"] = "Baseline"
    baseline["noise"] = 0

    dp = df[
        df["experiment"].str.contains("dp_local") |
        df["experiment"].str.contains("final_system")
    ].copy()

    dp = dp[(dp["epsilon"] > 0) & (dp["epsilon"] < 100)]

    def get_noise(exp):
        for k in NOISE_MAP:
            if k in exp:
                return NOISE_MAP[k]
        return None

    dp["noise"] = dp["experiment"].apply(get_noise)

    def group(exp):
        if "eps" in exp:
            return "Local Fixed DP"
        elif "adaptive" in exp:
            return "Local Adaptive DP"
        elif "final_system" in exp:
            return "TAP-FL"
        return "Other"

    dp["group"] = dp["experiment"].apply(group)

    return pd.concat([baseline, dp])


adult = prepare(adult_df)
credit = prepare(credit_df)

# =========================
# PALETTE
# =========================
palette = {
    "Baseline": "#000000",
    "Local Fixed DP": "#4C72B0",
    "Local Adaptive DP": "#55A868",
    "TAP-FL": "#C44E52"
}

# =========================
# PLOT FUNCTION
# =========================
def plot(ax, df, metric, title):

    # ----- BASELINE -----
    base = df[df["group"] == "Baseline"]
    ax.scatter(base["epsilon"], base[metric],
               s=250, marker="X", color="black",
               label="Baseline", zorder=5)

    # ----- FIXED DP -----
    fixed = df[df["group"] == "Local Fixed DP"].sort_values("epsilon")

    x_vals = list(base["epsilon"]) + list(fixed["epsilon"])
    y_vals = list(base[metric]) + list(fixed[metric])

    ax.plot(
        x_vals, y_vals,
        marker="o",
        linewidth=4,
        markersize=10,
        color=palette["Local Fixed DP"],
        label="Local Fixed DP (σ varies)"
    )

    # =========================
    # 🔥 SMART ANNOTATIONS (SMALL + NON-OVERLAPPING)
    # =========================
    y_min, y_max = min(y_vals), max(y_vals)
    y_range = y_max - y_min

    for i, (_, row) in enumerate(fixed.iterrows()):

        x = row["epsilon"]
        y = row[metric]

        # alternate above/below
        if i % 2 == 0:
            y_offset = 0.08 * y_range
            va = "bottom"
        else:
            y_offset = -0.10 * y_range
            va = "top"

        ax.text(
            x,
            y + y_offset,
            f"ε={row['epsilon']:.2f}, σ={row['noise']}",
            ha="center",
            va=va,
            fontsize=17,            # 🔥 SMALL (as requested)
            fontweight="bold",

            # 🔥 prevents overlap visually
            bbox=dict(
                facecolor="white",
                alpha=0.85,
                edgecolor="none",
                pad=1
            ),

            zorder=10
        )

    # ----- ADAPTIVE DP -----
    adaptive = df[df["group"] == "Local Adaptive DP"]

    if not adaptive.empty:
        ax.scatter(
            adaptive["epsilon"], adaptive[metric],
            s=230, marker="s",
            color=palette["Local Adaptive DP"],
            label=f"Local Adaptive DP (ε={adaptive['epsilon'].values[0]:.2f})",
            zorder=4
        )

    # ----- OUR SYSTEM -----
    final = df[df["group"] == "TAP-FL"]

    if not final.empty:
        ax.scatter(
            final["epsilon"], final[metric],
            s=300, marker="D",
            color=palette["TAP-FL"],
            edgecolor="black", linewidth=2,
            label=f"TAP-FL (ε={final['epsilon'].values[0]:.2f})",
            zorder=6
        )

    # =========================
    # 🔥 AXIS (ADD MARGIN → FIX CUTTING)
    # =========================
    ax.set_ylim(y_min - 0.15*y_range, y_max + 0.20*y_range)

    ax.set_xlabel("Privacy Budget (ε)", fontsize=26, weight="bold")
    ax.set_ylabel(metric.capitalize(), fontsize=26, weight="bold")
    ax.set_title(title, fontsize=26, weight="bold", pad=30)

    ax.grid(True, linestyle="--", alpha=0.3)

    # =========================
    # 🔥 TOP LEGEND
    # =========================
    handles, labels = ax.get_legend_handles_labels()

    fig = ax.get_figure()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),   # 🔥 ABOVE EVERYTHING
        ncol=2,
        frameon=True,
        fontsize=20,
        edgecolor="black",
        columnspacing=1.2,
        handletextpad=0.6
    )


# =========================
# ADULT FIGURE
# =========================
fig, ax = plt.subplots(figsize=(10, 7))

plot(ax, adult, "accuracy", "Adult Dataset")

plt.tight_layout(rect=[0, 0, 1, 0.85])

plt.savefig("results/adult_privacy_tradeoff.pdf", dpi=600, bbox_inches="tight")
plt.savefig("results/adult_privacy_tradeoff.png", dpi=600, bbox_inches="tight")

plt.show()


# =========================
# CREDIT FIGURE
# =========================
fig, ax = plt.subplots(figsize=(10, 7))

plot(ax, credit, "f1", "Credit Dataset")

plt.tight_layout(rect=[0, 0, 1, 0.85])

plt.savefig("results/credit_privacy_tradeoff.pdf", dpi=600, bbox_inches="tight")
plt.savefig("results/credit_privacy_tradeoff.png", dpi=600, bbox_inches="tight")

plt.show()