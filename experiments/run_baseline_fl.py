#experiments/run_baseline_fl.py
import pickle
import numpy as np
import torch

from models.mlp_model import SimpleMLPModel
from federated.aggregation import federated_average
from privacy.dp_training import train_private
from attacks.label_flipping import poison_labels

from utils.metrics import evaluate, evaluate_with_f1
from utils.logger import log_message

from configs.config import *


LOGFILE = "results/logs/training_log.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_client(client_id):

    with open(f"data/client_{client_id}.pkl", "rb") as f:

        data = pickle.load(f)

    return data["X"], data["y"]


def run_experiment():

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    global_model = SimpleMLPModel(INPUT_DIM).to(device)

    log_message("Starting Federated Learning Training", LOGFILE)

    for round_num in range(ROUNDS):

        log_message(f"\nRound {round_num+1}/{ROUNDS}", LOGFILE)

        client_weights = []
        epsilons = []

        for client_id in range(1, NUM_CLIENTS+1):

            X, y = load_client(client_id)

            # Adversarial simulation
            if client_id in ATTACK_CLIENTS:

                log_message(
                    f"Client {client_id} performing label flipping attack",
                    LOGFILE
                )

                y = poison_labels(y)

            client_model = SimpleMLPModel(INPUT_DIM)

            client_model.load_state_dict(global_model.state_dict())

            weights, loss, epsilon = train_private(
                client_model,
                X,
                y,
                LOCAL_EPOCHS,
                LEARNING_RATE,
                BATCH_SIZE,
                NOISE_MULTIPLIER,
                MAX_GRAD_NORM
            )

            client_weights.append(weights)
            epsilons.append(epsilon)

            log_message(
                f"Client {client_id} loss {loss:.4f} | ε={epsilon:.2f}",
                LOGFILE
            )

        new_weights = federated_average(client_weights)

        global_model.load_state_dict(new_weights)

        X_test, y_test = load_client(5)

        acc = evaluate(global_model, X_test, y_test)
        f1 = evaluate_with_f1(global_model, X_test, y_test)

        log_message(
            f"Global Accuracy {acc*100:.2f}% | F1 Score {f1:.3f}",
            LOGFILE
        )

        log_message(
            f"Average Privacy Budget ε = {np.mean(epsilons):.2f}",
            LOGFILE
        )

    torch.save(
        global_model.state_dict(),
        "results/models/global_model_private.pth"
    )

    log_message("Training completed successfully", LOGFILE)


if __name__ == "__main__":

    run_experiment()