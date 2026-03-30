# analysis/shap_analysis.py

import torch
import numpy as np
import shap
import pickle
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================
# LOAD MODEL + DATA
# =========================

def load_test_data():
    with open("data/test.pkl", "rb") as f:
        data = pickle.load(f)

    with open("data/global_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    X_raw = data["X"]
    if hasattr(X_raw, "values"):
        X_raw = X_raw.values

    X = scaler.transform(X_raw).astype("float32")
    y = data["y"]

    return X, y


# =========================
# SHAP COMPUTATION
# =========================

def compute_shap(model, X, exp_name, sample_size=100):
    """
    Returns:
        shap_values: (N, features)
        global_importance: (features,)
    """

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    X_sample = torch.tensor(X[:sample_size]).to(device)

    try:
        explainer = shap.DeepExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

        # handle binary case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

    except Exception as e:
        print("[WARNING] DeepExplainer failed, switching to KernelExplainer")

        def f(x):
            x_tensor = torch.tensor(x).float().to(device)
            with torch.no_grad():
                return torch.sigmoid(model(x_tensor)).cpu().numpy()

        explainer = shap.KernelExplainer(f, X[:50])
        shap_values = explainer.shap_values(X[:sample_size])

    shap_values = np.array(shap_values)

    global_importance = np.mean(np.abs(shap_values), axis=0)
    top_features = np.argsort(global_importance)[-10:][::-1]

    print(f"\n[{exp_name.upper()}] Top Features:")
    for f in top_features:
        print(f"Feature {f} → Importance: {global_importance[f]:.4f}")

    return shap_values, global_importance


# =========================
# METRICS
# =========================

def spearman_corr(shap1, shap2):
    corr, _ = spearmanr(shap1, shap2)
    return float(corr)


def topk_jaccard(shap1, shap2, k=10):
    top1 = np.argsort(shap1)[-k:]
    top2 = np.argsort(shap2)[-k:]

    intersection = len(set(top1).intersection(set(top2)))
    union = len(set(top1).union(set(top2)))

    return intersection / union if union > 0 else 0.0


def shap_drift(shap1, shap2):
    return float(np.mean(np.abs(shap1 - shap2)))


def prediction_explanation_consistency(model, X, shap_values):
    """
    correlation between prediction confidence and shap magnitude
    """

    device = torch.device("cpu")
    model.eval()

    X_tensor = torch.tensor(X[:len(shap_values)]).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(X_tensor)).cpu().numpy().ravel()

    shap_sum = np.sum(np.abs(shap_values), axis=1)

    corr, _ = spearmanr(probs, shap_sum)

    return float(corr)


# =========================
# SAVE / LOAD
# =========================

def save_shap(exp_name, shap_values, global_importance):
    os.makedirs("results/shap", exist_ok=True)

    np.save(f"results/shap/{exp_name}_shap.npy", shap_values)
    np.save(f"results/shap/{exp_name}_global.npy", global_importance)


def load_shap(exp_name):
    shap_values = np.load(f"results/shap/{exp_name}_shap.npy")
    global_importance = np.load(f"results/shap/{exp_name}_global.npy")

    return shap_values, global_importance


# =========================
# COMPARISON PIPELINE
# =========================

def compare_experiments(base_exp, other_exp, model, X):
    
    shap_vals, other_global = load_shap(other_exp)
    base_vals, base_global = load_shap(base_exp)
    

    results = {
        "spearman": spearman_corr(base_global, other_global),
        "topk_jaccard": topk_jaccard(base_global, other_global),
        "drift": shap_drift(base_global, other_global),
        "consistency": prediction_explanation_consistency(model, X, shap_vals),
    }
    
    
    return results


# =========================
# HUMAN-READABLE EXPLANATIONS
# =========================

def explain_single_prediction(model, X, shap_values, idx=0, feature_names=None):
    """
    Returns textual explanation for one sample
    """

    sample = X[idx]
    shap_val = shap_values[idx]

    top_features = np.argsort(np.abs(shap_val))[-5:]

    explanation = []

    for f in reversed(top_features):
        name = f"feature_{f}" if feature_names is None else feature_names[f]
        contribution = shap_val[f]

        direction = "increased" if contribution > 0 else "decreased"

        explanation.append(
            f"{name} {direction} prediction (impact={contribution:.4f})"
        )

    return explanation

def privacy_interpretability_tradeoff(exp_name):
    import json

    with open(f"results/{exp_name}.json") as f:
        history = json.load(f)

    final = history[-1]
    leakage = final["leakage"]

    shap_vals, global_vals = load_shap(exp_name)
    base_vals, base_global = load_shap("baseline")

    result = {
        "spearman": spearman_corr(base_global, global_vals),
        "drift": shap_drift(base_global, global_vals),
        "leakage": leakage
    }

    print("\n[TRADEOFF ANALYSIS]")
    print(f"Leakage: {leakage:.4f}")
    print(f"SHAP Stability (Spearman): {result['spearman']:.4f}")
    print(f"SHAP Drift: {result['drift']:.4f}")

    return result

# =========================
# MAIN RUN FUNCTION
# =========================

def run_shap_analysis(exp_name, model):
    

    X, y = load_test_data()

    shap_values, global_importance = compute_shap(model, X, exp_name)

    save_shap(exp_name, shap_values, global_importance)

    print(f"[SHAP] Saved for {exp_name}")
    
    plt.figure()
    X_sample = X[:shap_values.shape[0]]
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(f"results/shap/{exp_name}_summary.png")
    plt.close()
    explanation = explain_single_prediction(model, X, shap_values, idx=0)

    print(f"\n[{exp_name.upper()}] Example Explanation:")
    for line in explanation:
        print(line)

    return shap_values, global_importance