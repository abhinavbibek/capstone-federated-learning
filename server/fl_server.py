#server/fl_server.py
import flwr as fl
from configs.config import *
from server.robust_strategy import RobustFedAvg
import sys
from configs.config import EXPERIMENTS

def start_server(exp_name):
    
    exp = EXPERIMENTS[exp_name]

    if exp["robust"]:
        strategy = RobustFedAvg(
            method="median",   # or "trimmed_mean"
            fraction_fit=1.0,
            min_fit_clients=2,
            min_available_clients=2,
        )
    else:
        strategy = fl.server.strategy.FedAvg(
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
    exp_name = sys.argv[1]
    start_server(exp_name)


