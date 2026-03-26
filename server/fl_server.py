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
    X_test = scaler.fit_transform(data['X'])
    X_test = data['X'].values.astype('float32')
    X_test = torch.tensor(X_test)
    y_test = torch.FloatTensor(data['y']).view(-1, 1)

    def evaluate(server_round, parameters, config):

        model = SimpleMLPModel(X_test.shape[1])

        params = parameters

        params_dict = zip(model.state_dict().keys(), params)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict)

        model.eval()
        criterion = nn.BCELoss()

        with torch.no_grad():
            preds = model(X_test)
            loss = criterion(preds, y_test).item()
            acc = ((preds > 0.5) == y_test).float().mean().item()

        print(f"[Round {server_round}] Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        history.append({
            "round": server_round,
            "loss": loss,
            "accuracy": acc
        })

        return loss, {"accuracy": acc}

    return evaluate


def start_server(exp_name):

    exp = EXPERIMENTS[exp_name]

    if exp["robust"]:
        strategy = RobustFedAvg(
            method="median",
            fraction_fit=1.0,
            min_fit_clients=2,
            min_available_clients=2,
            evaluate_fn=get_eval_fn(),
        )
    else:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            min_fit_clients=2,
            min_available_clients=2,
            evaluate_fn=get_eval_fn(),
        )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )

    # Save results AFTER training
    with open("results.json", "w") as f:
        json.dump(history, f, indent=4)


if __name__ == "__main__":
    exp_name = sys.argv[1]
    start_server(exp_name)


