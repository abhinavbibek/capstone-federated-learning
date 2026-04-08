#server/fl_server.py
import flwr as fl
import pickle
import torch
import torch.nn as nn
import numpy as np
import json
import os
from sklearn.metrics import f1_score, roc_auc_score
import sys
from sklearn.preprocessing import StandardScaler
from configs.config import *
from models.mlp_model import SimpleMLPModel
from server.robust_strategy import RobustFedAvg
from configs.config import EXPERIMENTS
from utils.seed import set_seed
from flwr.server.strategy import (
    DifferentialPrivacyServerSideFixedClipping,
    DifferentialPrivacyServerSideAdaptiveClipping
)
import logging
from analysis.shap_analysis import run_shap_analysis
logging.getLogger("flwr").setLevel(logging.ERROR)
set_seed(SEED)

exp_name = sys.argv[1]
dataset = sys.argv[2]


def get_eval_fn(exp_name, dataset):
    history = []
    with open(f"data/{dataset}_global_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # ===== LOAD TRAIN DATA (for privacy metrics) =====
    with open(f'data/{dataset}_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    exp_config = EXPERIMENTS[exp_name]
    attack_type = exp_config.get("attack", None)
    X_train_raw = train_data['X']
    if hasattr(X_train_raw, "values"):
        X_train_raw = X_train_raw.values

    X_train_scaled = scaler.transform(X_train_raw)

    X_train = torch.tensor(X_train_scaled.astype('float32'))
    y_train = torch.FloatTensor(train_data['y']).view(-1, 1)
    with open(f'data/{dataset}_test.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_raw = data['X']
    if hasattr(X_raw, "values"):
        X_raw = X_raw.values

    X_scaled = scaler.transform(X_raw)

    X_test = torch.tensor(X_scaled.astype('float32'))
    y_test = torch.FloatTensor(data['y']).view(-1, 1)
    print(type(data['X']))
    print("Test shape:", X_test.shape)
    print("Test label distribution:", np.unique(y_test.numpy(), return_counts=True))

    def compute_privacy_metrics(model, X_test, y_test, X_train, y_train, device):
        with torch.no_grad():
            # ===== TEST =====
            logits_test = model(X_test.to(device))
            probs_test = torch.sigmoid(logits_test).cpu().numpy().ravel()

            # ===== TRAIN =====
            logits_train = model(X_train.to(device))
            probs_train = torch.sigmoid(logits_train).cpu().numpy().ravel()

        # =========================
        # 🔥 1. MEMBERSHIP INFERENCE ATTACK (MIA)
        # =========================
        threshold = np.mean(probs_train)

        mia_acc = (
            np.sum(probs_train > threshold) +
            np.sum(probs_test <= threshold)
        ) / (len(probs_train) + len(probs_test))

        # =========================
        # 🔥 2. CONFIDENCE GAP
        # =========================
        conf_gap = np.mean(probs_train) - np.mean(probs_test)

        # =========================
        # 🔥 3. ENTROPY
        # =========================
        entropy = -np.mean(
            probs_test * np.log(probs_test + 1e-8) +
            (1 - probs_test) * np.log(1 - probs_test + 1e-8)
        )

        return {
            "mia": float(mia_acc),
            "confidence_gap": float(conf_gap),
            "entropy": float(entropy)
        }

    def compute_asr(y_true, y_pred, attack_type):
    
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        if attack_type is None:
            return 0.0

        # ======================
        # LABEL FLIP ATTACK
        # ======================
        if attack_type == "label_flip":
            # attack tries to flip labels
            flipped = (y_true != y_pred)
            return np.mean(flipped)

        # ======================
        # TARGETED FLIP (1 → 0)
        # ======================
        elif attack_type == "targeted_flip":
            target_mask = (y_true == 1)
            success = (y_pred[target_mask] == 0)
            return np.mean(success) if np.sum(target_mask) > 0 else 0.0

        # ======================
        # FEATURE POISON
        # ======================
        elif attack_type == "feature_poison":
            # indirect → measure misclassification
            return 1.0 - np.mean(y_true == y_pred)

        # ======================
        # MODEL POISONING
        # ======================
        elif attack_type in ["sign_flip", "scaling"]:
            return 1.0 - np.mean(y_true == y_pred)

        return 0.0

    def evaluate(server_round, parameters, config):
        X_np = X_test.numpy()  
        feature_mean = np.mean(X_np[:, 0])

        def leakage_score(probs, X):
            feature = X[:, 0]
            pred = (probs > 0.5).astype(int)
            true = (feature > feature_mean).astype(int)
            return np.mean(pred == true)

        device = torch.device("cpu")
        model = SimpleMLPModel(X_test.shape[1]).to(device)
        # Load parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(device) for k, v in params_dict}
        model.load_state_dict(state_dict)
        os.makedirs("results", exist_ok=True)
        model.eval()
        criterion = nn.BCEWithLogitsLoss()
        epsilon = config.get("epsilon", 0.0)
        # Move data to same device
        X = X_test.to(device)
        y = y_test.to(device)

        with torch.no_grad():
            logits = model(X)

            loss = criterion(logits, y).item()

            probs = torch.sigmoid(logits)
            leakage = leakage_score(probs.cpu().numpy().ravel(), X.cpu().numpy())

            y_true = y.cpu().numpy().ravel()
            y_prob = probs.cpu().numpy().ravel()
            print("Mean prob:", np.mean(y_prob))
            #threshold = 0.3 if dataset == "credit" else 0.5
            if dataset == "credit":
                threshold = np.percentile(y_prob, 99.5)  # adaptive
            else:
                threshold = 0.5

            y_pred = (probs > threshold).cpu().numpy().ravel()
            print("Predicted positives:", np.sum(y_pred))
            asr = compute_asr(y_true, y_pred, attack_type)
            
            
            privacy_metrics = compute_privacy_metrics(
                model,
                X_test,
                y_test,
                X_train,
                y_train,
                device
            )
           
            acc = (y_pred == y_true).mean()
            f1 = f1_score(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = 0.0

        print(
            f"Round {server_round:02d} | "
            f"Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}\n"
            f"⚠️ ASR: {asr:.4f} → (higher = attack more successful)\n"
            f"🔒 Leakage: {leakage:.4f} → (higher = more feature leakage)\n"
            f"🔒 MIA: {privacy_metrics['mia']:.4f} → (~0.5 safe, >0.7 risky)\n"
            f"🔒 Epsilon: {epsilon:.4f} → (lower = stronger privacy)\n"
            f"🔒 Confidence Gap: {privacy_metrics['confidence_gap']:.4f} → (higher = overfitting leakage)\n"
            f"🔒 Entropy: {privacy_metrics['entropy']:.4f} → (lower = confident = risky)\n"
        )

        history.append({
            "round": server_round,
            "loss": loss,
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "asr": float(asr),
            "epsilon": float(epsilon),
            "leakage": float(leakage),

            # 🔥 NEW PRIVACY METRICS
            "mia": privacy_metrics["mia"],
            "confidence_gap": privacy_metrics["confidence_gap"],
            "entropy": privacy_metrics["entropy"]
        })
        with open(f"results/{dataset}_{exp_name}.json", "w") as f:
            json.dump(history, f, indent=4)


        if server_round == ROUNDS:

            print("\n[INFO] Running SHAP analysis...")
            torch.save(model.state_dict(), f"results/{dataset}_{exp_name}_model.pt")
            run_shap_analysis(exp_name, model, dataset)

            try:
                from analysis.shap_analysis import load_shap, shap_drift
                base_vals, base_global = load_shap("baseline", dataset)
                curr_vals, curr_global = load_shap(exp_name, dataset)

                drift = shap_drift(base_global, curr_global)

                print(f"[SHAP DRIFT] {drift:.4f}")

                if drift > 0.2:
                    print("[WARNING] High explanation drift detected → possible attack")
            except:
                pass       
        return loss, {"accuracy": acc}
    
    return evaluate

if __name__ == "__main__":
    exp_name = sys.argv[1]
    exp_config = EXPERIMENTS[exp_name]

    print(f"Starting server for experiment: {exp_name}")

    defense = exp_config.get("defense", None)

    if defense == "dp_server_fixed":
        print("Using DP Server Fixed (Flower)")

        base_strategy = fl.server.strategy.FedAvg(
            evaluate_fn=get_eval_fn(exp_name, dataset),
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            fraction_evaluate=1.0,
            min_evaluate_clients=NUM_CLIENTS,
        )

        strategy = DifferentialPrivacyServerSideFixedClipping(
            base_strategy,
            noise_multiplier=exp_config["noise"],
            clipping_norm=exp_config["clip"],
            num_sampled_clients=NUM_CLIENTS,
        )


    elif defense == "dp_server_adaptive":
        print("Using DP Server Adaptive (Flower)")

        base_strategy = fl.server.strategy.FedAvg(
            evaluate_fn=get_eval_fn(exp_name, dataset),
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            fraction_evaluate=1.0,
            min_evaluate_clients=NUM_CLIENTS,
        )

        strategy = DifferentialPrivacyServerSideAdaptiveClipping(
            base_strategy,
            noise_multiplier=exp_config["noise"],
            num_sampled_clients=NUM_CLIENTS,
            clipped_count_stddev=2.0,
        )


    elif defense in ["median", "trimmed_mean", "krum", "clipping", "trust"]:
        print(f"Using Robust Strategy: {defense}")

        strategy = RobustFedAvg(
            method=defense,
            dataset=dataset,
            evaluate_fn=get_eval_fn(exp_name, dataset),
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            fraction_evaluate=1.0,
            min_evaluate_clients=NUM_CLIENTS,
        )


    else:
        print("Using Standard FedAvg")

        strategy = fl.server.strategy.FedAvg(
            evaluate_fn=get_eval_fn(exp_name, dataset),
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            fraction_evaluate=1.0,
            min_evaluate_clients=NUM_CLIENTS,
        )


    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )
    