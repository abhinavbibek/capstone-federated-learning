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
import logging
from analysis.shap_analysis import run_shap_analysis
logging.getLogger("flwr").setLevel(logging.ERROR)
set_seed(SEED)



def get_eval_fn(exp_name):
    history = []
    with open('data/test.pkl', 'rb') as f:
        data = pickle.load(f)

    with open("data/global_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    X_raw = data['X']
    if hasattr(X_raw, "values"):
        X_raw = X_raw.values

    X_scaled = scaler.transform(X_raw)

    X_test = torch.tensor(X_scaled.astype('float32'))
    y_test = torch.FloatTensor(data['y']).view(-1, 1)
    print(type(data['X']))
    print("Test shape:", X_test.shape)
    print("Test label distribution:", np.unique(y_test.numpy(), return_counts=True))

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
        
        # Move data to same device
        X = X_test.to(device)
        y = y_test.to(device)

        with torch.no_grad():
            logits = model(X)

            loss = criterion(logits, y).item()

            probs = torch.sigmoid(logits)
            leakage = leakage_score(probs.cpu().numpy().ravel(), X.cpu().numpy())

            y_true = y.cpu().numpy().ravel()
            y_pred = (probs > 0.5).cpu().numpy().ravel()
            y_prob = probs.cpu().numpy().ravel()

            acc = (y_pred == y_true).mean()
            f1 = f1_score(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = 0.0

        # print(f"[Round {server_round}] Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        print(f"Round {server_round:02d} | Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Leakage: {leakage:.4f}")

        history.append({
            "round": server_round,
            "loss": loss,
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "leakage": float(leakage)
        })
        with open(f"results/{exp_name}.json", "w") as f:
            json.dump(history, f, indent=4)

        # Detect last round
        if server_round == ROUNDS:

            print("\n[INFO] Running SHAP analysis...")

            # Save model temporarily
            torch.save(model.state_dict(), f"results/{exp_name}_model.pt")

            # Run SHAP
            run_shap_analysis(exp_name, model)
        return loss, {"accuracy": acc}
    
    return evaluate

if __name__ == "__main__":
    exp_name = sys.argv[1]
    exp_config = EXPERIMENTS[exp_name]

    print(f"Starting server for experiment: {exp_name}")

    defense = exp_config.get("defense", None)

    if defense in [
        "median", "trimmed_mean", "krum", "clipping",
        "dp_server_fixed", "dp_server_adaptive"
    ]:
        print(f"Using Robust Strategy: {defense}")

        strategy = RobustFedAvg(
            method=defense,
            evaluate_fn=get_eval_fn(exp_name),
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            fraction_evaluate=1.0,
            min_evaluate_clients=NUM_CLIENTS,
        )

    else:
        print("Using Standard FedAvg")
        strategy = fl.server.strategy.FedAvg(
            evaluate_fn=get_eval_fn(exp_name),
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            fraction_evaluate=1.0,
            min_evaluate_clients=NUM_CLIENTS,
        )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )
    