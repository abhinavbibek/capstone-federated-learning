#server/fl_server.py
import flwr as fl
from configs.config import *
from server.robust_strategy import RobustFedAvg

def start_server():
    

    strategy = RobustFedAvg(
        method="median",   # or "trimmed_mean"
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server()