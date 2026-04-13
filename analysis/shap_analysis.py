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

    # 🔥 FIX: ensure proper shape
    if shap_values is None or len(shap_values) == 0:
        raise ValueError("SHAP returned empty values")

    # Handle shape safely
    if len(shap_values.shape) == 3:
        shap_values = shap_values.squeeze(-1)

    # 🔥 FINAL SAFETY
    if len(shap_values.shape) != 2:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

    global_importance = np.mean(np.abs(shap_values), axis=0)
    logger.info(f"SHAP shape: {shap_values.shape}")
    logger.info(f"Global importance shape: {global_importance.shape}")
    # 🔥 ensure 1D
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



# CORE SHAP ANALYSIS
def run_shap_analysis(exp_name, model, dataset):

    logger.info(f"\n========== START: {exp_name} ==========")

    X, _ = load_test_data(dataset)

    logger.info(f"[{exp_name}] Data loaded → shape={X.shape}")

    shap_vals, global_vals = compute_shap(model, X, exp_name)
    # =========================
    # 🔥 SAFETY CHECK (CRITICAL)
    # =========================
    if shap_vals is None or len(shap_vals) == 0:
        logger.error(f"[{exp_name}] SHAP values are empty → skipping")
        return None, None

    if global_vals is None or len(global_vals) == 0:
        logger.error(f"[{exp_name}] Global importance empty → skipping")
        return None, None

    # Check NaNs
    if np.isnan(global_vals).any():
        logger.error(f"[{exp_name}] NaNs in global importance → fixing")
        global_vals = np.nan_to_num(global_vals)

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

    # ==============================
    # 🔥 NEW: EXPLANATION EXTRACTION
    # ==============================

    logger.info(f"[{exp_name}] Extracting explanations")

    # Try to load feature names
    try:
        with open(f"data/{dataset}_test.pkl", "rb") as f:
            data = pickle.load(f)
        feature_names = data.get("feature_names", None)

        if feature_names is None or len(feature_names) != X.shape[1]:
            logger.warning(f"[{exp_name}] Feature names mismatch → using fallback")
            feature_names = [f"f{i}" for i in range(X.shape[1])]
    except:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    logger.info(f"[{exp_name}] Saving metrics JSON")
    # ==============================
    # 🔹 GLOBAL EXPLANATION
    # ==============================
    topk = 10
    top_global_idx = np.argsort(global_vals)[::-1][:topk]

    global_explanation = [
        {
            "feature": feature_names[i],
            "importance": float(global_vals[i])
        }
        for i in top_global_idx
    ]


    # ==============================
    # 🔹 LOCAL EXPLANATION (sample 0)
    # ==============================
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


    # ==============================
    # 🔹 SAVE EXPLANATIONS
    # ==============================
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

   # =========================
    # LARGE FONT SHAP PLOT (FIXED - NO GAPS)
    # =========================
    plt.figure(figsize=(10, 8))

    # 🔥 Strong font scaling (paper-ready)
    plt.rcParams.update({
        "font.size": 26,
        "axes.labelsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
    })

    # SHAP plot
    shap.summary_plot(
        shap_vals,
        X[:shap_vals.shape[0]],
        show=False,
        plot_size=None  # IMPORTANT
    )
    # =========================
    # 🔥 FIX OUTLIER STRETCH
    # =========================

    # Flatten SHAP values
    vals = shap_vals.flatten()

    # Compute robust limits (ignore extreme outliers)
    xmin = np.percentile(vals, 1)
    xmax = np.percentile(vals, 99)

    plt.xlim(xmin, xmax)
    plt.tight_layout(pad=0.5)
    # 🔥 Reduce LEFT GAP + bring COLORBAR closer
    plt.gcf().subplots_adjust(
        left=0.28,   # ↓ reduce empty left space (KEY FIX)
        right=0.92,  # ↓ bring colorbar closer
        top=0.95,
        bottom=0.12
    )

    # 🔥 Bigger labels (force override)
    plt.xlabel("SHAP value (impact on model output)", fontsize=26)
    plt.ylabel("Features", fontsize=26)

    # 🔥 Make colorbar text bigger
    cbar = plt.gcf().axes[-1]  # last axis is colorbar
    cbar.tick_params(labelsize=24)


    # SAVE (no extra padding)
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