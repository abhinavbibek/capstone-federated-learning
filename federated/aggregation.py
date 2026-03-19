import torch


def federated_average(client_weights):

    avg_weights = {}

    for key in client_weights[0].keys():

        avg_weights[key] = torch.stack(
            [client[key] for client in client_weights]
        ).mean(0)

    return avg_weights