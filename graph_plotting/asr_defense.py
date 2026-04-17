import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# STYLE
# =========================
sns.set_theme(style="whitegrid", context="paper", font_scale=2.3)

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# =========================
# LOAD CSV
# =========================
adult_df = pd.read_csv("results_summary_adult_round_40.csv")
credit_df = pd.read_csv("results_summary_credit_round_40.csv")

# =========================
# HELPER FUNCTION
# =========================
def extract_asr(df, attack_name):
    data = {}

    # =========================
    # ATTACK-ONLY BASELINE
    # =========================
    attack_row = df[df["experiment"].str.contains(f"{attack_name}_only")]
    attack_asr = attack_row["asr"].values[0]

    # =========================
    # DEFENSE METHODS
    # =========================
    for defense_key, label in {
        "median": "Median Aggregation",
        "trimmed": "Trimmed Mean",
        "krum": "Krum"
    }.items():

        row = df[df["experiment"].str.contains(f"{attack_name}_{defense_key}")]
        if len(row) > 0:
            data[label] = row["asr"].values[0]

    # =========================
    # TAP-FL (CORRECT MAPPING)
    # =========================
    tapfl_map = {
        "label_flip": "final_system",
        "sign_flip": "final_system_sign",
        "feature_poison": "final_system_feature",
        "targeted_flip": "final_system_targeted"
    }

    exp_name = tapfl_map.get(attack_name)

    if exp_name is not None:
        final_row = df[df["experiment"].str.contains(exp_name)]

        if len(final_row) > 0:
            data["TAP-FL"] = final_row["asr"].values[0]
        else:
            print(f"[WARNING] Missing {exp_name} in CSV")

    return data, attack_asr


# =========================
# ATTACK LIST
# =========================
attacks = ["label_flip", "feature_poison", "sign_flip", "targeted_flip"]


# =========================
# PLOT FUNCTION
# =========================
def plot_dataset(df, dataset_name):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    palette = {
        "Median Aggregation": "#4C72B0",
        "Trimmed Mean": "#55A868",
        "Krum": "#C44E52",
        "TAP-FL": "#000000"
    }

    for i, attack in enumerate(attacks):

        ax = axes[i]

        data, attack_asr = extract_asr(df, attack)

        methods = list(data.keys())
        values = list(data.values())

        x = range(len(methods))

        # =========================
        # BAR PLOT
        # =========================
        bars = ax.bar(
            x,
            values,
            width=0.5,
            color=[palette[m] for m in methods],
            alpha=0.9
        )

        # Highlight TAP-FL
        for j, m in enumerate(methods):
            if m == "TAP-FL":
                bars[j].set_linewidth(3)
                bars[j].set_edgecolor("black")

        # =========================
        # ATTACK REFERENCE LINE
        # =========================
        # ax.axhline(
        #     attack_asr,
        #     color="red",
        #     linestyle="--",
        #     linewidth=2
        # )

        # =========================
        # VALUE LABELS
        # =========================
        for j, v in enumerate(values):
            ax.text(
                j,
                v + (max(values) * 0.05),
                f"{v:.3f}",
                ha="center",
                fontsize=24,
                fontweight="bold"
            )

        # =========================
        # TITLES
        # =========================
        ax.set_title(
            attack.replace("_", " ").title(),
            fontsize=26,
            weight="bold"
        )

        # =========================
        # CLEAN X-AXIS (NO TEXT)
        # =========================
        ax.set_xticks(x)
        ax.set_xticklabels([])   # 🔥 removed long labels

        ax.set_ylabel("ASR", fontsize=26, weight="bold")

        # =========================
        # LIMITS + GRID
        # =========================
        ax.set_ylim(0, max(max(values), attack_asr) * 1.25)
        ax.grid(True, linestyle="--", alpha=0.4)

    # =========================
    # GLOBAL LEGEND
    # =========================
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[m])
        for m in palette
    ]

    labels = list(palette.keys())

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),   # 🔥 moved DOWN inside figure
        ncol=2,
        fontsize=22,
        frameon=True
    )

    # =========================
    # MAIN TITLE
    # =========================
    # fig.suptitle(
    #     f"{dataset_name.upper()} — ASR vs Defense",
    #     fontsize=32,
    #     weight="bold",
    #     y=0.98
    # )

    # =========================
    # LAYOUT
    # =========================
    plt.tight_layout(rect=[0, 0, 1, 0.87])  # 🔥 more usable space

    # =========================
    # SAVE
    # =========================
    plt.savefig(f"results/{dataset_name}_asr_vs_defense.pdf", dpi=600)
    plt.savefig(f"results/{dataset_name}_asr_vs_defense.png", dpi=600)

    plt.show()


# =========================
# RUN
# =========================
plot_dataset(adult_df, "adult")
plot_dataset(credit_df, "credit")