import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# STYLE
# =========================
sns.set_theme(style="ticks", context="paper", font_scale=1.3)

plt.rcParams.update({
    "font.family": "serif",
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
            return "Our System"
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
    "Our System": "#C44E52"
}

# =========================
# PLOT FUNCTION (UNCHANGED LOGIC)
# =========================
def plot(ax, df, metric, title):

    # ----- BASELINE -----
    base = df[df["group"] == "Baseline"]
    ax.scatter(base["epsilon"], base[metric],
               s=150, marker="X", color="black", label="Baseline", zorder=5)

    # ----- FIXED DP -----
    fixed = df[df["group"] == "Local Fixed DP"].sort_values("epsilon")

    # ✅ IMPORTANT: keep baseline connected (same as before)
    x_vals = list(base["epsilon"]) + list(fixed["epsilon"])
    y_vals = list(base[metric]) + list(fixed[metric])

    ax.plot(x_vals, y_vals,
            marker="o", linewidth=2.8, markersize=7,
            color=palette["Local Fixed DP"],
            label="Local Fixed DP (σ varies)")

    # --- ANNOTATIONS (same) ---
    offsets = [(0,10), (0,-15), (10,10), (-10,-10)]
    for i, (_, row) in enumerate(fixed.iterrows()):
        dx, dy = offsets[i % len(offsets)]

        ax.annotate(
            f"ε={row['epsilon']:.2f}, σ={row['noise']}",
            (row["epsilon"], row[metric]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha='center',
            fontsize=8
        )

    # ----- ADAPTIVE DP -----
    adaptive = df[df["group"] == "Local Adaptive DP"]

    if not adaptive.empty:
        ax.scatter(adaptive["epsilon"], adaptive[metric],
                   s=140, marker="s",
                   color=palette["Local Adaptive DP"],
                   label=f"Local Adaptive DP (ε={adaptive['epsilon'].values[0]:.2f})",
                   zorder=4)

    # ----- OUR SYSTEM -----
    final = df[df["group"] == "Our System"]

    if not final.empty:
        ax.scatter(final["epsilon"], final[metric],
                   s=200, marker="D",
                   color=palette["Our System"],
                   edgecolor="black", linewidth=2,
                   label=f"Our System (ε={final['epsilon'].values[0]:.2f})",
                   zorder=6)

    # ----- AXIS (NO LOG SCALE) -----
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title, weight="bold")

    ax.grid(True, linestyle="--", alpha=0.3)

    # ✅ SAME LEGEND STYLE AS BEFORE
    ax.legend(frameon=False)


# =========================
# ADULT FIGURE
# =========================
fig, ax = plt.subplots(figsize=(7, 5))

plot(ax, adult, "accuracy", "Adult Dataset")

plt.tight_layout()

plt.savefig("results/adult_privacy_tradeoff.pdf", dpi=600, bbox_inches="tight")
plt.savefig("results/adult_privacy_tradeoff.png", dpi=600, bbox_inches="tight")

plt.show()


# =========================
# CREDIT FIGURE
# =========================
fig, ax = plt.subplots(figsize=(7, 5))

plot(ax, credit, "f1", "Credit Dataset")

plt.tight_layout()

plt.savefig("results/credit_privacy_tradeoff.pdf", dpi=600, bbox_inches="tight")
plt.savefig("results/credit_privacy_tradeoff.png", dpi=600, bbox_inches="tight")

plt.show()