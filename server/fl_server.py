#server/fl_server.py
import flwr as fl
from flwr.server.strategy import (
    DifferentialPrivacyServerSideFixedClipping
)
from sklearn.model_selection import train_test_split
import pickle
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from opacus.accountants import RDPAccountant
from sklearn.metrics import mutual_info_score
import sys
from sklearn.preprocessing import StandardScaler
from configs.config import *
from models.mlp_model import SimpleMLPModel
from server.robust_strategy import RobustFedAvg
from configs.config import EXPERIMENTS
from utils.seed import set_seed
import logging
from analysis.shap_analysis import run_shap_analysis
logging.getLogger("flwr").setLevel(logging.ERROR)
set_seed(SEED)

exp_name = sys.argv[1]
dataset = sys.argv[2]

def get_eval_fn(exp_name, dataset, strategy):
    history = []
    with open(f"data/{dataset}_global_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    accountant = RDPAccountant()
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

    def compute_mia(probs_train, probs_test):
        noise = np.random.normal(0, 0.01, size=probs_train.shape)
        probs_train = probs_train + noise
        noise = np.random.normal(0, 0.01, size=probs_test.shape)
        probs_test = probs_test + noise
        probs_train = np.clip(probs_train, 1e-6, 1 - 1e-6)
        probs_test = np.clip(probs_test, 1e-6, 1 - 1e-6)
        entropy_train = - (
            probs_train * np.log(probs_train) +
            (1 - probs_train) * np.log(1 - probs_train)
        )
        entropy_test = - (
            probs_test * np.log(probs_test) +
            (1 - probs_test) * np.log(1 - probs_test)
        )
        conf_train = np.abs(probs_train - 0.5)
        conf_test = np.abs(probs_test - 0.5)
        X = np.concatenate([
            np.stack([probs_train, entropy_train, conf_train], axis=1),
            np.stack([probs_test, entropy_test, conf_test], axis=1)
        ])
        y = np.concatenate([
            np.ones(len(probs_train)),
            np.zeros(len(probs_test))
        ])
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.5, stratify=y, random_state=42
        )
        clf = LogisticRegression(max_iter=300)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        return float(np.mean(preds == y_te))

    def compute_privacy_metrics(model, X_test, y_test, X_train, y_train, device):
        with torch.no_grad():
            logits_test = model(X_test.to(device))
            probs_test = torch.sigmoid(logits_test).cpu().numpy().ravel()
            logits_train = model(X_train.to(device))
            probs_train = torch.sigmoid(logits_train).cpu().numpy().ravel()
        threshold = np.mean(probs_train)
        mia_acc = compute_mia(probs_train, probs_test)
        conf_gap = np.mean(probs_train) - np.mean(probs_test)
        entropy = -np.mean(
            probs_test * np.log(probs_test + 1e-8) +
            (1 - probs_test) * np.log(1 - probs_test + 1e-8)
        )
        return {
            "mia": float(mia_acc),
            "confidence_gap": float(conf_gap),
            "entropy": float(entropy)
        }

    def compute_asr(y_true, y_pred, attack_type, dataset=None):
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        if attack_type is None:
            return 0.0
        if dataset == "credit":
            positives = (y_true == 1)
            if np.sum(positives) == 0:
                return 0.0
            recall = np.sum(y_pred[positives] == 1) / np.sum(positives)
            return float(1.0 - recall) 

        if attack_type == "label_flip":
            return float(np.mean(y_true != y_pred))

        elif attack_type == "targeted_flip":
            mask = (y_true == 1)
            if np.sum(mask) == 0:
                return 0.0
            return float(np.mean(y_pred[mask] == 0))

        elif attack_type == "feature_poison":
            return float(np.mean(y_true != y_pred))

        elif attack_type in ["sign_flip", "scaling"]:
            return float(np.mean(y_true != y_pred))
        return 0.0

    def evaluate(server_round, parameters, config):
        X_np = X_test.numpy()  
        feature_mean = np.mean(X_np[:, 0])
        def leakage_score(probs, X, threshold):
            preds = (probs > threshold).astype(int)
            scores = []
            for i in range(min(10, X.shape[1])):
                feature = X[:, i]
                feature_bin = (feature > np.mean(feature)).astype(int)
                mi = mutual_info_score(feature_bin, preds)
                scores.append(mi)
            return float(np.mean(scores))
        device = torch.device("cpu")
        model = SimpleMLPModel(X_test.shape[1]).to(device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(device) for k, v in params_dict}
        model.load_state_dict(state_dict)
        os.makedirs("results", exist_ok=True)
        model.eval()
        criterion = nn.BCEWithLogitsLoss()

        # Sever side DP
        if exp_config.get("defense") in ["dp_server_fixed"]:
            sample_rate = 1.0
            noise = exp_config.get("noise", 0.0)
            if noise > 0:
                accountant.step(noise_multiplier=noise, sample_rate=sample_rate)
            try:
                epsilon = accountant.get_epsilon(delta=1e-5)
            except:
                epsilon = 0.0

        # Local DP 
        elif exp_config.get("dp") in ["local", "local_adaptive"]:
            if hasattr(strategy, "latest_fit_metrics") and "mean_epsilon" in strategy.latest_fit_metrics:
                epsilon = strategy.latest_fit_metrics["mean_epsilon"]
            else:
                epsilon = 0.0
        else:
            epsilon = 0.0
        X = X_test.to(device)
        y = y_test.to(device)
        with torch.no_grad():
            logits = model(X)
            loss = criterion(logits, y).item()
            probs = torch.sigmoid(logits)
            y_true = y.cpu().numpy().ravel()
            y_prob = probs.cpu().numpy().ravel()
            print("Mean prob:", np.mean(y_prob))
            #threshold = 0.3 if dataset == "credit" else 0.5
            if dataset == "credit":
                threshold = np.percentile(y_prob, 95)  # adaptive
            else:
                threshold = 0.5
            leakage = leakage_score(y_prob, X.cpu().numpy(), threshold)
            y_pred = (probs > threshold).cpu().numpy().ravel()
            print("Predicted positives:", np.sum(y_pred))
            asr = compute_asr(y_true, y_pred, attack_type, dataset)
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
            f"ASR: {asr:.4f} → (higher = attack more successful)\n"
            f"Leakage: {leakage:.4f} → (higher = more feature leakage)\n"
            f"MIA: {privacy_metrics['mia']:.4f} → (~0.5 safe, >0.7 risky)\n"
            f"Epsilon: {epsilon:.4f} → (lower = stronger privacy)\n"
            f"Confidence Gap: {privacy_metrics['confidence_gap']:.4f} → (higher = overfitting leakage)\n"
            f"Entropy: {privacy_metrics['entropy']:.4f} → (lower = confident = risky)\n"
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
            "mia": privacy_metrics["mia"],
            "confidence_gap": privacy_metrics["confidence_gap"],
            "entropy": privacy_metrics["entropy"]
        })
        with open(f"results/{dataset}_{exp_name}.json", "w") as f:
            json.dump(history, f, indent=4)


        if server_round == ROUNDS and exp_name in [
            "baseline",
            "sign_flip_only",
            "feature_poison_only",
            "dp_local_eps1",
            "dp_local_eps2",
            "dp_local_eps5",
            "final_system"
        ]:

            print("\nRunning SHAP analysis...")
            torch.save(model.state_dict(), f"results/{dataset}_{exp_name}_model.pt")
            run_shap_analysis(exp_name, model, dataset)
            try:
                from analysis.shap_analysis import load_shap, shap_drift
                base_vals, base_global = load_shap("baseline", dataset)
                curr_vals, curr_global = load_shap(exp_name, dataset)
                drift = shap_drift(base_global, curr_global)
                print(f"Shap drift {drift:.4f}")
                if drift > 0.2:
                    print("High explanation drift detected : possible attack")
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
        strategy.evaluate_fn = get_eval_fn(exp_name, dataset, strategy)


    elif defense in ["median", "trimmed_mean", "krum", "clipping", "trust", "fltrust"] \
        or exp_config.get("dp") in ["local", "local_adaptive"]:
        print(f"Using Robust Strategy: {defense}")
        strategy = RobustFedAvg(
            method=defense if defense else "none",
            dataset=dataset,
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            fraction_evaluate=1.0,
            min_evaluate_clients=NUM_CLIENTS,
        )
        strategy.evaluate_fn = get_eval_fn(exp_name, dataset, strategy)


    else:
        print("Using Standard FedAvg")

        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            fraction_evaluate=1.0,
            min_evaluate_clients=NUM_CLIENTS,
        )
        strategy.evaluate_fn = get_eval_fn(exp_name, dataset, strategy)
        
    print("Strategy Type:", type(strategy))
    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )
    

