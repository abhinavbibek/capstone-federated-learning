# #server/robust_strategy.py
# import flwr as fl
# import numpy as np

# from server.robust_aggregation import (
#     median_aggregation,
#     trimmed_mean_aggregation,
#     krum_aggregation
# )

# def compute_norm(weights):
#     return np.sqrt(sum(np.sum(w**2) for w in weights))


# def fixed_clip(weights, clip_norm=5.0):
#     clipped = []
#     for w in weights:
#         norm = compute_norm(w)
#         if norm > clip_norm:
#             scale = clip_norm / (norm + 1e-6)
#             w = [layer * scale for layer in w]
#         clipped.append(w)
#     return clipped


# def adaptive_clip(weights):
#     norms = [compute_norm(w) for w in weights]
#     threshold = np.median(norms)
#     return fixed_clip(weights, threshold)


# def compute_client_weights(results):
#     losses = np.array([
#         fit_res.metrics.get("loss", 1.0)
#         for _, fit_res in results
#     ])

#     # 🔥 STRONGER weighting (better separation of good vs bad clients)
#     weights = np.exp(-losses)

#     # normalize
#     weights = weights / np.sum(weights)

#     return weights


# def add_adaptive_noise(aggregated, weights, rnd):
#     flat_weights = [np.concatenate([w.flatten() for w in client]) for client in weights]
#     stacked = np.stack(flat_weights)

#     variance = np.var(stacked, axis=0).mean()

#     # 🔥 round-aware decay
#     round_factor = np.exp(-0.4 * rnd)

#     noise_std = np.sqrt(variance + 1e-6) * round_factor

#     noisy = []
#     for w in aggregated:
#         noise = np.random.normal(0, noise_std, size=w.shape)
#         noisy.append(w + noise)

#     return noisy


# def clip_updates(weights, threshold=5.0):
#     clipped = []

#     for client_weights in weights:
#         total_norm = np.sqrt(
#             sum(np.sum(w**2) for w in client_weights)
#         )

#         if total_norm > threshold:
#             scale = threshold / (total_norm + 1e-6)
#             client_weights = [w * scale for w in client_weights]

#         clipped.append(client_weights)

#     return clipped


# class RobustFedAvg(fl.server.strategy.FedAvg):

#     def __init__(self, method="median", **kwargs):
#         super().__init__(**kwargs)
#         self.method = method

#     def aggregate_fit(self, rnd, results, failures):

#         if not results:
#             return None, {}

#         weights = [
#             fl.common.parameters_to_ndarrays(fit_res.parameters)
#             for _, fit_res in results
#         ]

#         losses = np.array([
#             fit_res.metrics.get("loss", 1.0)
#             for _, fit_res in results
#         ])

#         # Select top 70% clients (lowest loss)
#         k = int(0.7 * len(results))
#         selected_idx = np.argsort(losses)[:k]

#         # Filter weights
#         weights = [weights[i] for i in selected_idx]
#         losses = losses[selected_idx]

#         # Recompute weights ONLY on selected clients
#         client_weights = np.exp(-losses)
#         client_weights /= np.sum(client_weights)

#         # ================= CLIPPING =================
#         if self.method == "clipping":
#             weights = clip_updates(weights, threshold=5.0)

#         # ================= KRUM =================
#         if self.method == "krum":
#             selected = krum_aggregation(weights)
#             parameters = fl.common.ndarrays_to_parameters(selected)
#             return parameters, {}

#         # ================= DP SERVER =================
#         if self.method == "dp_server_fixed":
#             weights = fixed_clip(weights, 5.0)

#         elif self.method == "dp_server_adaptive":
#             weights = adaptive_clip(weights)

#         elif self.method == "dp_hybrid":
#             weights = fixed_clip(weights, 3.0)

#         # ================= AGGREGATION =================
#         aggregated = []

#         for layer_idx, layer in enumerate(zip(*weights)):
#             layer_stack = np.stack(layer, axis=0)

#             if self.method in ["median", "trimmed_mean"]:
#                 # keep robust aggregation intact
#                 if self.method == "median":
#                     agg_layer = np.median(layer_stack, axis=0)

#                 else:
#                     trim_ratio = 0.2
#                     n = layer_stack.shape[0]
#                     k = int(n * trim_ratio)

#                     sorted_layer = np.sort(layer_stack, axis=0)
#                     trimmed = sorted_layer[k:n-k]

#                     agg_layer = np.mean(trimmed, axis=0)

#             elif self.method == "clipping":
#                 agg_layer = np.mean(layer_stack, axis=0)

#             else:
#                 # 🔥 TRUST-WEIGHTED AGGREGATION (MAIN IMPROVEMENT)

#                 reshape_dims = [len(client_weights)] + [1] * (layer_stack.ndim - 1)

#                 weighted_layer = np.sum(
#                     layer_stack * client_weights.reshape(reshape_dims),
#                     axis=0
#                 )

#                 agg_layer = weighted_layer

#             aggregated.append(agg_layer)

#         # ================= ADAPTIVE DP NOISE =================
#         if self.method in ["dp_server_fixed", "dp_server_adaptive", "dp_hybrid"]:
#             aggregated = add_adaptive_noise(aggregated, weights, rnd)

#         parameters = fl.common.ndarrays_to_parameters(aggregated)

#         epsilons = [
#             fit_res.metrics.get("epsilon", 0.0)
#             for _, fit_res in results
#         ]

#         avg_epsilon = float(np.mean(epsilons))

#         return parameters, {"epsilon": avg_epsilon}

import flwr as fl
import numpy as np

from server.robust_aggregation import (
    median_aggregation,
    trimmed_mean_aggregation,
    krum_aggregation
)

def compute_norm(weights):
    return np.sqrt(sum(np.sum(w**2) for w in weights))


def fixed_clip(weights, clip_norm=5.0):
    clipped = []
    for w in weights:
        norm = compute_norm(w)
        if norm > clip_norm:
            scale = clip_norm / (norm + 1e-6)
            w = [layer * scale for layer in w]
        clipped.append(w)
    return clipped


def adaptive_clip(weights):
    norms = [compute_norm(w) for w in weights]
    threshold = np.median(norms)
    print(f"[DEBUG] Adaptive clip threshold: {threshold:.4f}")
    return fixed_clip(weights, threshold)


def compute_client_weights(results):
    losses = np.array([
        fit_res.metrics.get("loss", 1.0)
        for _, fit_res in results
    ])

    weights = np.exp(-losses)
    weights = weights / np.sum(weights)

    return weights


def add_adaptive_noise(aggregated, weights, rnd):
    flat_weights = [np.concatenate([w.flatten() for w in client]) for client in weights]
    stacked = np.stack(flat_weights)

    variance = np.var(stacked, axis=0).mean()
    round_factor = np.exp(-0.4 * rnd)

    noise_std = np.sqrt(variance + 1e-6) * round_factor

    print(f"[DEBUG] Noise std: {noise_std:.6f} | Variance: {variance:.6f}")

    noisy = []
    for w in aggregated:
        noise = np.random.normal(0, noise_std, size=w.shape)
        noisy.append(w + noise)

    return noisy


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

        # ✅ ADD: momentum storage
        self.prev_weights = None

    def aggregate_fit(self, rnd, results, failures):

        if not results:
            return None, {}

        print(f"\n[ROUND {rnd}] =========================")

        weights = [
            fl.common.parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        # ================= ADAPTIVE TOP-K (REPLACED BLOCK) =================
        losses = np.array([
            fit_res.metrics.get("loss", 1.0)
            for _, fit_res in results
        ])

        print(f"[DEBUG] All client losses: {losses}")

        mean_loss = np.mean(losses)
        std_loss = np.std(losses)

        threshold = mean_loss + 0.5 * std_loss

        print(f"[DEBUG] mean={mean_loss:.4f}, std={std_loss:.4f}, threshold={threshold:.4f}")

        selected_idx = np.where(losses <= threshold)[0]

        # fallback safety
        if len(selected_idx) < len(losses) // 2:
            print("[DEBUG] Fallback triggered (too few clients selected)")
            selected_idx = np.argsort(losses)[:len(losses)//2]

        print(f"[DEBUG] Selected clients: {selected_idx}")

        # filter
        weights = [weights[i] for i in selected_idx]
        losses = losses[selected_idx]

        # recompute weights
        client_weights = np.exp(-losses)
        client_weights /= np.sum(client_weights)

        print(f"[DEBUG] Client weights: {client_weights}")

        # ================= CLIPPING =================
        if self.method == "clipping":
            weights = clip_updates(weights, threshold=5.0)

        # ================= KRUM =================
        if self.method == "krum":
            selected = krum_aggregation(weights)
            parameters = fl.common.ndarrays_to_parameters(selected)
            return parameters, {}

        # ================= DP SERVER =================
        if self.method == "dp_server_fixed":
            weights = fixed_clip(weights, 5.0)

        elif self.method == "dp_server_adaptive":
            weights = adaptive_clip(weights)

        # ================= AGGREGATION =================
        aggregated = []

        for layer_idx, layer in enumerate(zip(*weights)):
            layer_stack = np.stack(layer, axis=0)

            if self.method in ["median", "trimmed_mean"]:
                if self.method == "median":
                    agg_layer = np.median(layer_stack, axis=0)
                else:
                    trim_ratio = 0.2
                    n = layer_stack.shape[0]
                    k = int(n * trim_ratio)

                    sorted_layer = np.sort(layer_stack, axis=0)
                    trimmed = sorted_layer[k:n-k]

                    agg_layer = np.mean(trimmed, axis=0)

            elif self.method == "clipping":
                agg_layer = np.mean(layer_stack, axis=0)

            else:
                reshape_dims = [len(client_weights)] + [1] * (layer_stack.ndim - 1)

                weighted_layer = np.sum(
                    layer_stack * client_weights.reshape(reshape_dims),
                    axis=0
                )

                agg_layer = weighted_layer

            aggregated.append(agg_layer)

        # ================= MOMENTUM (ADDED) =================
        if self.prev_weights is not None:
            print("[DEBUG] Applying momentum")
            momentum = 0.3
            aggregated = [
                momentum * prev + (1 - momentum) * curr
                for prev, curr in zip(self.prev_weights, aggregated)
            ]

        self.prev_weights = aggregated

        # ================= ADAPTIVE DP NOISE =================
        if self.method in ["dp_server_fixed", "dp_server_adaptive"]:
            aggregated = add_adaptive_noise(aggregated, weights, rnd)

        parameters = fl.common.ndarrays_to_parameters(aggregated)

        epsilons = [
            fit_res.metrics.get("epsilon", 0.0)
            for _, fit_res in results
        ]

        avg_epsilon = float(np.mean(epsilons))

        return parameters, {"epsilon": avg_epsilon}