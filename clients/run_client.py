#clients/run_client.py
import flwr as fl
import sys

from clients.fl_client import FLClient

if __name__ == "__main__":
    client_id = int(sys.argv[1])

    fl.client.start_numpy_client(
        server_address="0.0.0.0:8080",
        client=FLClient(client_id)
    )