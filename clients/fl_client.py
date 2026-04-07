#clients/fl_client.py
import flwr as fl
import torch
import numpy as np
import pickle
import torch.nn as nn

from models.mlp_model import SimpleMLPModel
from attacks.label_flipping import label_flip, targeted_label_flip
from attacks.feature_poisoning import feature_poison
from attacks.model_poisoning import sign_flipping, scaling_attack
from sklearn.preprocessing import StandardScaler
from configs.config import *
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from federated.client_training import train_local
from privacy.opacus_dp import train_with_opacus
from utils.seed import set_seed
import logging
logging.getLogger("flwr").setLevel(logging.ERROR)
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")



class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id, exp_config, dataset):
        self.dataset = dataset
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

        with open(f"data/{self.dataset}_client_{self.client_id}.pkl", "rb") as f:
            data = pickle.load(f)

        with open(f"data/{self.dataset}_global_scaler.pkl", "rb") as f:
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

        attack_type = self.exp_config["attack"]

        # Clean copies
        X = self.X.copy()
        y = self.y.copy()

        is_attacker = attack_type and self.client_id in ATTACK_CLIENTS

        # ======================
        # DATA POISONING
        # ======================
        if is_attacker:

            if attack_type == "label_flip":
                y = label_flip(y, flip_ratio=0.3)

            elif attack_type == "targeted_flip":
                y = targeted_label_flip(y)

            elif attack_type == "feature_poison":
                X = feature_poison(X)

        # ======================
        # TRAINING
        # ======================
        
        dp_mode = self.exp_config.get("dp", None)

        if dp_mode in ["local", "local_adaptive"]:
            noise = self.exp_config.get("noise", 1.0)
            clip = self.exp_config.get("clip", 1.0)
            # hybrid = lighter noise


            # 🔥 Disable adaptive for final system
            if self.exp_config.get("defense") == "trust":
                is_adaptive = False
            else:
                is_adaptive = (dp_mode == "local_adaptive")

            weights, loss, epsilon = train_with_opacus(
                self.model,
                X,
                y,
                LOCAL_EPOCHS,
                LEARNING_RATE,
                BATCH_SIZE,
                noise,
                clip,
                adaptive=is_adaptive,   # 🔥 ONLY CHANGE
                dataset=self.dataset
            )

        else:
            weights, loss = train_local(
                self.model,
                X,
                y,
                LOCAL_EPOCHS,
                LEARNING_RATE,
                BATCH_SIZE,
                dataset=self.dataset
            )
            epsilon = 0.0

        # ======================
        # MODEL POISONING
        # ======================
        if is_attacker:

            if attack_type == "sign_flip":
                weights = sign_flipping(weights)

            elif attack_type == "scaling":
                weights = scaling_attack(weights)

        # ======================
        # LOGGING (clean + correct)
        # ======================
        if is_attacker:
            print(f"[Client {self.client_id}] ATTACK={attack_type} | Loss={loss:.4f}")
        else:
            print(f"[Client {self.client_id}] benign | Loss={loss:.4f}")

        weights_ndarrays = [val.cpu().numpy() for val in weights.values()]
        fraud_ratio = np.mean(y)
        return weights_ndarrays, len(self.X), {
            "loss": float(loss),
            "epsilon": float(epsilon),
            "fraud_ratio": float(fraud_ratio),
            "client_id": self.client_id   # 🔥 ADD
        }


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(device)

        X = torch.FloatTensor(self.X).to(device)
        y = torch.FloatTensor(self.y).reshape(-1, 1).to(device)

        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            logits = self.model(X)
            loss = criterion(logits, y).item()

            probs = torch.sigmoid(logits)

        y_true = y.cpu().numpy().ravel()
        y_pred = (probs > 0.5).cpu().numpy().ravel()
        print("Predicted positives:", np.sum(y_pred))
        y_prob = probs.cpu().numpy().ravel()

        acc = (y_pred == y_true).mean()
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.0

        return loss, len(self.X), {
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "precision" : precision,
            "recall" : recall
        }