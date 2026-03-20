#clients/fl_client.py
import flwr as fl
import torch
import numpy as np
import pickle

from models.mlp_model import SimpleMLPModel
from privacy.opacus_dp import train_with_opacus
from attacks.label_flipping import poison_labels
from configs.config import *

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")



class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id, exp_config):
        self.exp_config = exp_config
        self.client_id = client_id
        self.model = SimpleMLPModel(INPUT_DIM).to(device)
        self.X, self.y = self.load_data()

    def load_data(self):
        with open(f"data/client_{self.client_id}.pkl", "rb") as f:
            data = pickle.load(f)
        return data["X"], data["y"]

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(
            zip(
                self.model.state_dict().keys(),
                [torch.tensor(p) for p in parameters]
            )
        )
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        y = self.y.copy()

        # Apply attack
        if self.exp_config["attack"] and self.client_id in ATTACK_CLIENTS:
            y = poison_labels(y)
        if self.exp_config["dp"]:
            weights, loss, epsilon = train_with_opacus(
                self.model,
                self.X,
                y,
                LOCAL_EPOCHS,
                LEARNING_RATE,
                BATCH_SIZE,
                NOISE_MULTIPLIER,
                MAX_GRAD_NORM
            )
        else:
            from federated.client_training import train_local
            weights, loss = train_local(
                self.model,
                self.X,
                y,
                LOCAL_EPOCHS,
                LEARNING_RATE,
                BATCH_SIZE
            )
            epsilon = 0.0
        return [
            val.cpu().numpy() for val in weights.values()
        ], len(self.X), {
            "loss": loss,
            "epsilon": epsilon
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        X = torch.FloatTensor(self.X).to(device)
        y = torch.FloatTensor(self.y).reshape(-1, 1).to(device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X)
            acc = ((preds > 0.5) == y).float().mean().item()

        return float(acc), len(self.X), {"accuracy": acc}