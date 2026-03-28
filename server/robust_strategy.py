#server/robust_strategy.py
import flwr as fl
import numpy as np

from server.robust_aggregation import (
    median_aggregation,
    trimmed_mean_aggregation,
    krum_aggregation
)


def clip_updates(weights, threshold=5.0):
    clipped = []

    for client_weights in weights:
        total_norm = np.sqrt(
            sum(np.sum(w**2) for w in client_weights)
        )

        if total_norm > threshold:
            scale = threshold / (total_norm + 1e-6)
            client_weights = [w * scale for w in client_weights]

        clipped.append(client_weights)

    return clipped


class RobustFedAvg(fl.server.strategy.FedAvg):

    def __init__(self, method="median", **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def aggregate_fit(self, rnd, results, failures):

        if not results:
            return None, {}

        weights = [
            fl.common.parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        # ================= CLIPPING =================
        if self.method == "clipping":
            weights = clip_updates(weights, threshold=5.0)

        # ================= KRUM =================
        if self.method == "krum":
            selected = krum_aggregation(weights)
            parameters = fl.common.ndarrays_to_parameters(selected)
            return parameters, {}

        # ================= MEDIAN / TRIMMED =================
        aggregated = []

        for layer in zip(*weights):
            layer_stack = np.stack(layer, axis=0)

            if self.method == "median":
                agg_layer = np.median(layer_stack, axis=0)

            elif self.method == "trimmed_mean":
                trim_ratio = 0.2
                n = layer_stack.shape[0]
                k = int(n * trim_ratio)

                sorted_layer = np.sort(layer_stack, axis=0)
                trimmed = sorted_layer[k:n-k]

                agg_layer = np.mean(trimmed, axis=0)

            elif self.method == "clipping":
                # clipping + mean
                agg_layer = np.mean(layer_stack, axis=0)

            else:
                raise ValueError("Unknown defense method")

            aggregated.append(agg_layer)

        parameters = fl.common.ndarrays_to_parameters(aggregated)

        return parameters, {}