import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# STYLE
# =========================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

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

    # attack baseline
    attack_row = df[df["experiment"].str.contains(f"{attack_name}_only")]
    attack_asr = attack_row["asr"].values[0]

    # defenses
    for defense_key, label in {
        "median": "Median Aggregation",
        "trimmed": "Trimmed Mean",
        "krum": "Krum"
    }.items():

        row = df[df["experiment"].str.contains(f"{attack_name}_{defense_key}")]
        if len(row) > 0:
            data[label] = row["asr"].values[0]

    # final system → rename here
    final_row = df[df["experiment"].str.contains("final_system")]
    data["Our System"] = final_row["asr"].values[0]

    return data, attack_asr


# =========================
# ATTACK LIST
# =========================
attacks = ["label_flip", "feature_poison", "sign_flip", "targeted_flip"]

# =========================
# PLOT FUNCTION
# =========================
def plot_dataset(df, dataset_name):

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    palette = {
        "Median Aggregation": "#4C72B0",
        "Trimmed Mean": "#55A868",
        "Krum": "#C44E52",
        "Our System": "#000000"
    }

    for i, attack in enumerate(attacks):

        ax = axes[i]

        data, attack_asr = extract_asr(df, attack)

        methods = list(data.keys())
        values = list(data.values())

        x = range(len(methods))

        bars = ax.bar(
            x,
            values,
            color=[palette[m] for m in methods],
            alpha=0.9
        )

        # Highlight OUR SYSTEM
        for j, m in enumerate(methods):
            if m == "Our System":
                bars[j].set_linewidth(3)
                bars[j].set_edgecolor("black")

        # Attack reference line
        ax.axhline(
            attack_asr,
            color="red",
            linestyle="--",
            linewidth=2,
            label=""
        )

        # Annotate
        for j, v in enumerate(values):
            ax.text(j, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

        ax.set_title(
            attack.replace("_", " ").title(),
            fontsize=12,
            weight="bold"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15)
        ax.set_ylabel("ASR")

        ax.set_ylim(0, max(max(values), attack_asr) * 1.25)

    fig.suptitle(
        f"{dataset_name.upper()} — ASR vs Defense",
        fontsize=16,
        weight="bold"
    )

    # Global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    plt.savefig(f"results/{dataset_name}_asr_vs_defense.pdf", dpi=600)
    plt.savefig(f"results/{dataset_name}_asr_vs_defense.png", dpi=600)

    plt.show()


# =========================
# RUN
# =========================
plot_dataset(adult_df, "adult")
plot_dataset(credit_df, "credit")