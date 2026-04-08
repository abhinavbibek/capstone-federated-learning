#analysis/shap_analysis.py
import torch
import numpy as np
import shap
import pickle
import os
import json
import logging
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


os.makedirs("results/shap", exist_ok=True)

logger = logging.getLogger("shap_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fh = logging.FileHandler("results/shap/abc_shap.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# LOAD DATA
def load_test_data(dataset):
    with open(f"data/{dataset}_test.pkl", "rb") as f:
        data = pickle.load(f)

    with open(f"data/{dataset}_global_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    X_raw = data["X"]
    if hasattr(X_raw, "values"):
        X_raw = X_raw.values

    X = scaler.transform(X_raw).astype("float32")
    y = data["y"]

    return X, y


# SHAP COMPUTATION
def compute_shap(model, X, exp_name, sample_size=100):

    logger.info(f"[{exp_name}] Starting SHAP computation")

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    logger.info(f"[{exp_name}] Model moved to CPU and set to eval")

    X_sample = torch.tensor(X[:sample_size]).to(device)

    try:
        logger.info(f"[{exp_name}] Using DeepExplainer")

        explainer = shap.DeepExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

    except Exception as e:
        logger.warning(f"[{exp_name}] DeepExplainer failed → switching to KernelExplainer")

        def f(x):
            x_tensor = torch.tensor(x).float().to(device)
            with torch.no_grad():
                return torch.sigmoid(model(x_tensor)).cpu().numpy()

        explainer = shap.KernelExplainer(f, X[:50])
        shap_values = explainer.shap_values(X[:sample_size])

    shap_values = np.array(shap_values)

    if len(shap_values.shape) == 3:
        shap_values = shap_values.squeeze(-1)

    global_importance = np.mean(np.abs(shap_values), axis=0).reshape(-1)

    logger.info(f"[{exp_name}] SHAP computation completed")

    return shap_values, global_importance



def spearman_corr(a, b):
    return float(spearmanr(a, b)[0])


def topk_jaccard(a, b, k=10):
    top1 = set(np.argsort(a)[-k:])
    top2 = set(np.argsort(b)[-k:])
    return len(top1 & top2) / len(top1 | top2)


def shap_drift(a, b):
    return float(np.mean(np.abs(a - b)))


def faithfulness_test(model, X, shap_values, k=5):
    device = torch.device("cpu")
    model.eval()

    X_tensor = torch.tensor(X[:len(shap_values)]).to(device)

    with torch.no_grad():
        original = torch.sigmoid(model(X_tensor)).cpu().numpy().ravel()

    drops = []

    for i in range(len(shap_values)):
        x = X[i].copy()
        top_features = np.argsort(np.abs(shap_values[i]))[-k:]
        x[top_features] = 0

        with torch.no_grad():
            new_pred = torch.sigmoid(
                model(torch.tensor(x).float().unsqueeze(0))
            ).item()

        drops.append(abs(original[i] - new_pred))

    return float(np.mean(drops))


def perturbation_stability(model, X, shap_values):
    X_noisy = X + np.random.normal(0, 0.01, X.shape)
    new_shap, _ = compute_shap(model, X_noisy, "temp")

    return spearman_corr(
        np.mean(np.abs(shap_values), axis=0),
        np.mean(np.abs(new_shap), axis=0)
    )


def save_shap(exp, shap_vals, global_vals, dataset):
    np.save(f"results/shap/{dataset}_{exp}_shap.npy", shap_vals)
    np.save(f"results/shap/{dataset}_{exp}_global.npy", global_vals)


def load_shap(exp, dataset):
    return (
        np.load(f"results/shap/{dataset}_{exp}_shap.npy"),
        np.load(f"results/shap/{dataset}_{exp}_global.npy")
    )



# CORE SHAP ANALYSIS
def run_shap_analysis(exp_name, model, dataset):

    logger.info(f"\n========== START: {exp_name} ==========")

    X, _ = load_test_data(dataset)

    logger.info(f"[{exp_name}] Data loaded → shape={X.shape}")

    shap_vals, global_vals = compute_shap(model, X, exp_name)

    logger.info(f"[{exp_name}] Computing metrics")

    faithfulness = faithfulness_test(model, X, shap_vals)
    stability = perturbation_stability(model, X, shap_vals)

    logger.info(
        f"[{exp_name}] Metrics → Faithfulness={faithfulness:.4f}, Stability={stability:.4f}"
    )

    metrics = {
        "experiment": exp_name,
        "faithfulness": faithfulness,
        "stability": stability
    }

    logger.info(f"[{exp_name}] Saving SHAP arrays")

    save_shap(exp_name, shap_vals, global_vals, dataset)

    logger.info(f"[{exp_name}] Saving metrics JSON")

    with open(f"results/shap/{dataset}_{exp_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"[{exp_name}] Generating SHAP summary plot")

    plt.figure()
    shap.summary_plot(shap_vals, X[:shap_vals.shape[0]], show=False)
    plt.savefig(f"results/shap/{dataset}_{exp_name}_summary.png")
    plt.close()

    logger.info(f"[{exp_name}] Plot saved")

    logger.info(f"========== END: {exp_name} ==========\n")

    return shap_vals, global_vals


def compare_experiments(base, other, dataset):

    logger.info(f"[COMPARE] {other} vs {base}")

    _, base_g = load_shap(base, dataset)
    _, other_g = load_shap(other, dataset)

    spearman = spearman_corr(base_g, other_g)
    jaccard = topk_jaccard(base_g, other_g)
    drift = shap_drift(base_g, other_g)

    logger.info(
        f"[COMPARE] {other} → Spearman={spearman:.4f}, Jaccard={jaccard:.4f}, Drift={drift:.4f}"
    )

    return {
        "spearman": spearman,
        "topk_jaccard": jaccard,
        "drift": drift,
    }


def get_leakage(exp, dataset):
    with open(f"results/{dataset}_{exp}.json") as f:
        return json.load(f)[-1]["leakage"]