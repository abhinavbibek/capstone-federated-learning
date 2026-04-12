#analysis/unified_model.py
import torch
import numpy as np
import pickle
import os
import json
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from models.mlp_model import SimpleMLPModel


import joblib

def save_unified_system(meta_model, tag):

    os.makedirs("results/unified", exist_ok=True)

    joblib.dump(meta_model, f"results/unified/{tag}_meta.pkl")

    print(f"[SAVED] Unified model → {tag}")



def safe_predict(model, X):

    input_dim = model.network[0].in_features

    if X.shape[1] != input_dim:
        if X.shape[1] > input_dim:
            X_mod = X[:, :input_dim]
        else:
            pad = np.zeros((X.shape[0], input_dim - X.shape[1]))
            X_mod = np.hstack([X, pad])
    else:
        X_mod = X

    return get_predictions(model, X_mod)



def load_data(dataset):
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


def load_model(dataset, exp):
    X, _ = load_data(dataset)

    model = SimpleMLPModel(X.shape[1])

    path = f"results/{dataset}_{exp}_model.pt"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model: {path}")

    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    return model


def load_saved_metrics(dataset, exp):
    path = f"results/{dataset}_{exp}.json"

    if not os.path.exists(path):
        print(f"[WARNING] No JSON found for {dataset}-{exp}")
        return None

    with open(path) as f:
        history = json.load(f)

    final = history[-1]

    return {
        "auc": final.get("auc", None),
        "f1": final.get("f1", None),
        "accuracy": final.get("accuracy", None),
        "leakage": final.get("leakage", None)
    }


# GET PREDICTIONS
def get_predictions(model, X):
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).numpy().ravel()
    return probs


# EVALUATE SINGLE MODEL
def evaluate_single(model, X, y):

    probs = get_predictions(model, X)
    preds = (probs > 0.5).astype(int)

    auc = roc_auc_score(y, probs)
    f1 = f1_score(y, preds)
    acc = accuracy_score(y, preds)

    return {
        "auc": float(auc),
        "f1": float(f1),
        "accuracy": float(acc)
    }


# VERIFY INDIVIDUAL MODELS
def verify_individual_models(exp_name):

    print(f"\n VERIFYING INDIVIDUAL MODELS: {exp_name}")

    for dataset in ["adult", "credit"]:

        X, y = load_data(dataset)
        model = load_model(dataset, exp_name)

        computed = evaluate_single(model, X, y)
        saved = load_saved_metrics(dataset, exp_name)

        print(f"\n[{dataset.upper()} - {exp_name}]")

        print("Computed:")
        print(computed)

        print("Saved (JSON):")
        print(saved)


# BUILD META FEATURES
def build_meta_features(exp_name):

    X_adult, y_adult = load_data("adult")
    X_credit, y_credit = load_data("credit")

    model_adult = load_model("adult", exp_name)
    model_credit = load_model("credit", exp_name)

    p_adult = get_predictions(model_adult, X_adult)
    p_credit = get_predictions(model_credit, X_credit)

    # STACK DATASETS (NOT ALIGN)
    meta_X = np.concatenate([
        np.column_stack([p_adult, np.zeros_like(p_adult)]),
        np.column_stack([np.zeros_like(p_credit), p_credit])
    ])

    y = np.concatenate([y_adult, y_credit])

    return meta_X, y


# TRAIN META MODEL
def train_meta_model(meta_X, y):
    model = LogisticRegression()
    model.fit(meta_X, y)
    return model


# EVALUATE META MODEL
def evaluate(meta_model, meta_X, y):

    probs = meta_model.predict_proba(meta_X)[:, 1]
    preds = (probs > 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(y, probs)),
        "f1": float(f1_score(y, preds)),
        "accuracy": float(accuracy_score(y, preds))
    }

def strict_validation(exp_name):

    print("\n" + "="*50)
    print(f" STRICT VALIDATION: {exp_name}")
    print("="*50)

    for dataset in ["adult", "credit"]:

        X, y = load_data(dataset)
        model = load_model(dataset, exp_name)
        probs = get_predictions(model, X)
        if dataset == "credit":
            threshold = np.percentile(probs, 99.5)
        else:
            threshold = 0.5

        preds = (probs > threshold).astype(int)

        computed = {
            "auc": float(roc_auc_score(y, probs)),
            "f1": float(f1_score(y, preds)),
            "accuracy": float(accuracy_score(y, preds))
        }
        saved = load_saved_metrics(dataset, exp_name)

        print(f"\n [{dataset.upper()} - {exp_name}]")

        print("SAVED (JSON):")
        print(saved)

        print("RECOMPUTED:")
        print(computed)

        # DIFFERENCE CHECK
        if saved is not None and saved["auc"] is not None:

            auc_diff = abs(saved["auc"] - computed["auc"])
            f1_diff = abs(saved["f1"] - computed["f1"])

            print(f"Δ AUC: {auc_diff:.6f} | Δ F1: {f1_diff:.6f}")

            if auc_diff > 0.01:
                print("WARNING: AUC mismatch!")
            if f1_diff > 0.05:
                print("WARNING: F1 mismatch (threshold-sensitive)")


# MAIN PIPELINE
def run_unified_system(exp_name, tag):

    print(f"\n==============================")
    print(f"UNIFIED SYSTEM: {tag}")
    print(f"==============================")
    strict_validation(exp_name)

    print("\nValidation complete → proceeding to fusion")
    meta_X, y = build_meta_features(exp_name)

    print(f"\nMeta feature shape: {meta_X.shape}")
    meta_model = train_meta_model(meta_X, y)

    save_unified_system(meta_model, tag)
    metrics = evaluate(meta_model, meta_X, y)

    print(f"\nUnified Result:")
    print(metrics)
    os.makedirs("results/unified", exist_ok=True)

    with open(f"results/unified/{tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics