import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================
# CONFIG
# ==============================
DATASETS = ["adult", "credit"]
NUM_CLIENTS = 10

os.makedirs("results/plots", exist_ok=True)


def load_distribution(dataset):
    class0 = []
    class1 = []

    for i in range(1, NUM_CLIENTS + 1):
        with open(f"data/{dataset}_client_{i}.pkl", "rb") as f:
            data = pickle.load(f)

        y = data["y"]

        c0 = np.sum(y == 0)
        c1 = np.sum(y == 1)

        total = len(y)

        # convert to percentage
        class0.append((c0 / total) * 100)
        class1.append((c1 / total) * 100)

    return np.array(class0), np.array(class1)


def plot_non_iid():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for idx, dataset in enumerate(DATASETS):
        c0, c1 = load_distribution(dataset)

        clients = np.arange(1, NUM_CLIENTS + 1)

        ax = axes[idx]

        # horizontal stacked bars
        ax.barh(clients, c0, label="Class 0")
        ax.barh(clients, c1, left=c0, label="Class 1")

        ax.set_title(dataset.upper())
        ax.set_xlabel("Percentage (%)")
        ax.set_xlim(0, 100)

        ax.set_yticks(clients)
        ax.set_ylabel("Client ID")

        ax.grid(axis='x', linestyle='--', alpha=0.5)

    # single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    save_path = "results/plots/non_iid_combined.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    plot_non_iid()


# # analysis/plots/plot_noniid_box.py

# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# DATASET = "credit"
# NUM_CLIENTS = 10

# SAVE_DIR = f"results/plots/{DATASET}/data"
# os.makedirs(SAVE_DIR, exist_ok=True)

# fraud_ratios = []

# for i in range(1, NUM_CLIENTS + 1):
#     with open(f"data/{DATASET}_client_{i}.pkl", "rb") as f:
#         data = pickle.load(f)

#     y = data["y"]
#     fraud_ratios.append(np.mean(y))

# fraud_ratios = np.array(fraud_ratios)

# # ==============================
# # PLOT
# # ==============================
# plt.figure(figsize=(5,4))

# plt.boxplot(fraud_ratios, vert=True)
# plt.scatter(np.ones_like(fraud_ratios), fraud_ratios, alpha=0.7)

# plt.ylabel("Fraud Ratio")
# plt.title("Distribution of Fraud Ratios Across Clients")

# plt.grid(alpha=0.3)
# plt.tight_layout()

# plt.savefig(f"{SAVE_DIR}/noniid_box.png", dpi=300)
# plt.close()

# print("Non-IID variability plot saved.")