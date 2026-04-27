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


def compute_shap(model, X, exp_name, sample_size=100):
    logger.info(f"[{exp_name}] Starting SHAP")
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
        logger.warning(f"[{exp_name}] DeepExplainer failed now switching to KernelExplainer")
        def f(x):
            x_tensor = torch.tensor(x).float().to(device)
            with torch.no_grad():
                return torch.sigmoid(model(x_tensor)).cpu().numpy()
        explainer = shap.KernelExplainer(f, X[:50])
        shap_values = explainer.shap_values(X[:sample_size])
    shap_values = np.array(shap_values)

    if shap_values is None or len(shap_values) == 0:
        raise ValueError("SHAP returned empty values")

    if len(shap_values.shape) == 3:
        shap_values = shap_values.squeeze(-1)

    if len(shap_values.shape) != 2:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

    global_importance = np.mean(np.abs(shap_values), axis=0)
    logger.info(f"SHAP shape: {shap_values.shape}")
    logger.info(f"Global importance shape: {global_importance.shape}")
    global_importance = np.array(global_importance).flatten()
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


def perturbation_stability(model, X, shap_values):
    X_noisy = X + np.random.normal(0, 0.01, X.shape)
    try:
        new_shap, _ = compute_shap(model, X_noisy, "temp")
        if new_shap is None or len(new_shap) == 0:
            return 0.0
    except Exception as e:
        logger.warning(f"Perturbation stability failed: {e}")
        return 0.0
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


def run_shap_analysis(exp_name, model, dataset):
    logger.info(f"\nStart: {exp_name}")
    X, _ = load_test_data(dataset)
    logger.info(f"[{exp_name}] Data loaded : shape={X.shape}")
    shap_vals, global_vals = compute_shap(model, X, exp_name)
    if shap_vals is None or len(shap_vals) == 0:
        logger.error(f"[{exp_name}] SHAP values are empty, skipping")
        return None, None

    if global_vals is None or len(global_vals) == 0:
        logger.error(f"[{exp_name}] Global importance empty, skipping")
        return None, None

    if np.isnan(global_vals).any():
        logger.error(f"[{exp_name}] NaNs in global importance,skipping")
        global_vals = np.nan_to_num(global_vals)
    stability = perturbation_stability(model, X, shap_vals)
    logger.info(
        f"[{exp_name}] Metrics: Stability={stability:.4f}"
    )

    metrics = {
        "experiment": exp_name,
        "stability": stability
    }
    save_shap(exp_name, shap_vals, global_vals, dataset)
    logger.info(f"[{exp_name}] Extracting explanations")
    # load feature names
    try:
        with open(f"data/{dataset}_test.pkl", "rb") as f:
            data = pickle.load(f)
        feature_names = data.get("feature_names", None)

        if feature_names is None or len(feature_names) != X.shape[1]:
            logger.warning(f"[{exp_name}] Feature names mismatch now using fallback")
            feature_names = [f"f{i}" for i in range(X.shape[1])]
    except:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    logger.info(f"[{exp_name}] Saving metrics JSON")
    topk = 10
    top_global_idx = np.argsort(global_vals)[::-1][:topk]
    global_explanation = [
        {
            "feature": feature_names[i],
            "importance": float(global_vals[i])
        }
        for i in top_global_idx
    ]

    sample_idx = 0
    sample_shap = shap_vals[sample_idx]
    top_local_idx = np.argsort(np.abs(sample_shap))[::-1][:topk]
    local_explanation = [
        {
            "feature": feature_names[i],
            "contribution": float(sample_shap[i]),
            "effect": "increase" if sample_shap[i] > 0 else "decrease"
        }
        for i in top_local_idx
    ]
    explanation_data = {
        "experiment": exp_name,
        "global_top_features": global_explanation,
        "local_explanation_sample0": local_explanation
    }

    with open(f"results/shap/{dataset}_{exp_name}_explanations.json", "w") as f:
        json.dump(explanation_data, f, indent=4)

    logger.info(f"[{exp_name}] Explanations saved")
    with open(f"results/shap/{dataset}_{exp_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"[{exp_name}] Generating SHAP summary plot")
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({
        "font.size": 24,
        "axes.labelsize": 26,
    })

    shap.summary_plot(
        shap_vals,
        X[:shap_vals.shape[0]],
        show=False,
        plot_size=None
    )
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=26, pad=6)
    ax.tick_params(axis='y', labelsize=26, pad=4)
    vals = shap_vals.flatten()
    xmin = np.percentile(vals, 1)
    xmax = np.percentile(vals, 99)
    plt.xlim(xmin, xmax)
    plt.xlabel("SHAP value (impact on model output)", fontsize=26, labelpad=12)
    plt.ylabel("Features", fontsize=26, labelpad=14)
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=26)
    cbar.set_ylabel("Feature value", fontsize=24, labelpad=10)
    plt.gcf().subplots_adjust(
        left=0.30,  
        right=0.90, 
        top=0.95,
        bottom=0.14
    )

    plt.tight_layout(pad=0.5)
    plt.savefig(
        f"results/shap/{dataset}_{exp_name}_summary.pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.savefig(
        f"results/shap/{dataset}_{exp_name}_summary.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.close()
    logger.info(f"[{exp_name}] Plot saved")
    return shap_vals, global_vals


def compare_experiments(base, other, dataset):
    logger.info(f"COMPARE {other} vs {base}")
    _, base_g = load_shap(base, dataset)
    _, other_g = load_shap(other, dataset)
    spearman = spearman_corr(base_g, other_g)
    jaccard = topk_jaccard(base_g, other_g)
    drift = shap_drift(base_g, other_g)
    logger.info(
        f"Compare {other} : Spearman={spearman:.4f}, Jaccard={jaccard:.4f}, Drift={drift:.4f}"
    )
    return {
        "spearman": spearman,
        "topk_jaccard": jaccard,
        "drift": drift,
    }


def get_leakage(exp, dataset):
    with open(f"results/{dataset}_{exp}.json") as f:
        return json.load(f)[-1]["leakage"]