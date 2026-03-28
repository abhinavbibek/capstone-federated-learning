#clients/fl_client.py
import flwr as fl
import torch
import numpy as np
import pickle
import torch.nn as nn

from models.mlp_model import SimpleMLPModel
from privacy.opacus_dp import train_with_opacus
from attacks.label_flipping import poison_labels
from sklearn.preprocessing import StandardScaler
from configs.config import *
from federated.client_training import train_local
from utils.seed import set_seed
import logging
logging.getLogger("flwr").setLevel(logging.ERROR)
set_seed(SEED)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")



class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id, exp_config):
        self.exp_config = exp_config
        self.client_id = client_id
        self.X, self.y = self.load_data()
        input_dim = self.X.shape[1]
        self.model = SimpleMLPModel(input_dim).to(device)
        print(f"[Client {self.client_id}] X shape: {self.X.shape}")
        print(f"[Client {self.client_id}] Feature count: {self.X.shape[1]}")
        print(f"[Client {self.client_id}] Label distribution: {np.unique(self.y, return_counts=True)}")
        print(f"[Client {self.client_id}] Samples: {len(self.X)}")


    def load_data(self):
        with open(f"data/client_{self.client_id}.pkl", "rb") as f:
            data = pickle.load(f)

        with open("data/global_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        X_raw = data["X"]
        if hasattr(X_raw, "values"):
            X_raw = X_raw.values

        X = scaler.transform(X_raw)

        return X.astype("float32"), data["y"]

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.model.to(device)
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
            weights, loss = train_local(
                self.model,
                self.X,
                y,
                LOCAL_EPOCHS,
                LEARNING_RATE,
                BATCH_SIZE
            )
            epsilon = 0.0

        print(f"[Client {self.client_id}] Loss: {loss:.4f}, Epsilon: {epsilon:.2f}")

        # ✅ Convert state_dict → list of numpy arrays
        weights_ndarrays = [val.cpu().numpy() for val in weights.values()]

        # ✅ RETURN CORRECT FORMAT
        return weights_ndarrays, len(self.X), {"loss": float(loss), "epsilon": float(epsilon)}


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(device)
        X = torch.FloatTensor(self.X).to(device)
        y = torch.FloatTensor(self.y).reshape(-1, 1).to(device)

        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()

        logits = self.model(X)
        loss = criterion(logits, y).item()

        probs = torch.sigmoid(logits)
        acc = ((probs > 0.5) == y).float().mean().item()

        return loss, len(self.X), {"accuracy": acc}