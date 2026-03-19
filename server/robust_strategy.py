#server/robust_strategy.py
import flwr as fl
import numpy as np

from server.robust_aggregation import (
    median_aggregation,
    trimmed_mean_aggregation
)


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

            else:
                raise ValueError("Unknown aggregation method")

            aggregated.append(agg_layer)

        parameters = fl.common.ndarrays_to_parameters(aggregated)

        return parameters, {}