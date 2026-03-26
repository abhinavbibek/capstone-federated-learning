#server/fl_server.py
import flwr as fl
import pickle
import torch
import torch.nn as nn
import json
import sys
from sklearn.preprocessing import StandardScaler
from configs.config import *
from models.mlp_model import SimpleMLPModel
from server.robust_strategy import RobustFedAvg
from configs.config import EXPERIMENTS

from utils.seed import set_seed
set_seed(SEED)
history = []


def get_eval_fn():

    with open('data/test.pkl', 'rb') as f:
        data = pickle.load(f)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data['X'])

    X_test = torch.tensor(X_scaled.astype('float32'))
    y_test = torch.FloatTensor(data['y']).view(-1, 1)

    def evaluate(server_round, parameters, config):

        device = torch.device("cpu")
        model = SimpleMLPModel(X_test.shape[1]).to(device)
        model.to(device) 
        # Load parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(device) for k, v in params_dict}
        model.load_state_dict(state_dict)

        model.eval()
        criterion = nn.BCELoss()

        # Move data to same device
        X = X_test.to(device)
        y = y_test.to(device)

        with torch.no_grad():
            preds = model(X)

            loss = criterion(preds, y).item()
            acc = ((preds > 0.5).float() == y).float().mean().item()

        print(f"[Round {server_round}] Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        history.append({
            "round": server_round,
            "loss": loss,
            "accuracy": acc
        })

        return loss, {"accuracy": acc}

    return evaluate

if __name__ == "__main__":
    exp_name = sys.argv[1]
    exp_config = EXPERIMENTS[exp_name]

    print(f"Starting server for experiment: {exp_name}")

    strategy = RobustFedAvg(
        evaluate_fn=get_eval_fn(),
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