#clients/run_client.py
import flwr as fl
import sys

from clients.fl_client import FLClient
from configs.config import EXPERIMENTS

client_id = int(sys.argv[1])
exp_name = sys.argv[2]

exp_config = EXPERIMENTS[exp_name]

fl.client.start_numpy_client(
    server_address="127.0.0.1:8081",
    client=FLClient(client_id, exp_config)
)