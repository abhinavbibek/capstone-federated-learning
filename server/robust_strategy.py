
#server/robust_strategy.py
import flwr as fl
import numpy as np
from server.trust_manager import TrustManager

from server.robust_aggregation import (
    median_aggregation,
    trimmed_mean_aggregation,
    krum_aggregation
)

# ================= UTILITIES =================

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


def add_adaptive_noise(aggregated, weights, rnd, trust_scores=None):
    flat_weights = [np.concatenate([w.flatten() for w in client]) for client in weights]
    stacked = np.stack(flat_weights)

    variance = np.var(stacked, axis=0).mean()
    round_factor = np.exp(-0.4 * rnd)

    avg_trust = np.mean(trust_scores) if trust_scores is not None else 1.0

    noise_std = np.sqrt(variance + 1e-6) * round_factor * (1 + (1 - avg_trust))

    print(f"[DEBUG] Noise std: {noise_std:.6f} | Variance: {variance:.6f}")

    noisy = []
    for w in aggregated:
        noise = np.random.normal(0, noise_std, size=w.shape)
        noisy.append(w + noise)

    return noisy


def clip_updates(weights, threshold=5.0):
    clipped = []
    for client_weights in weights:
        total_norm = np.sqrt(sum(np.sum(w**2) for w in client_weights))
        if total_norm > threshold:
            scale = threshold / (total_norm + 1e-6)
            client_weights = [w * scale for w in client_weights]
        clipped.append(client_weights)
    return clipped


# ================= MAIN STRATEGY =================

class RobustFedAvg(fl.server.strategy.FedAvg):

    def __init__(self, method="median", **kwargs):
        super().__init__(**kwargs)
        self.method = method

        # 🔥 TRUST FLAG (ONLY activates for final system)
        self.use_trust = (method == "trust")

        self.trust_manager = TrustManager()
        self.prev_weights = None  # momentum

    def aggregate_fit(self, rnd, results, failures):

        if not results:
            return None, {}

        print(f"\n[ROUND {rnd}] =========================")

        # ================= LOAD CLIENT WEIGHTS =================
        weights = [
            fl.common.parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        losses = np.array([
            fit_res.metrics.get("loss", 1.0)
            for _, fit_res in results
        ])

        print(f"[DEBUG] All client losses: {losses}")

        # ================= ADAPTIVE TOP-K =================
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        threshold = mean_loss + 0.5 * std_loss

        selected_idx = np.where(losses <= threshold)[0]

        if len(selected_idx) < len(losses) // 2:
            print("[DEBUG] Fallback triggered")
            selected_idx = np.argsort(losses)[:len(losses)//2]

        print(f"[DEBUG] Selected clients: {selected_idx}")

        weights = [weights[i] for i in selected_idx]
        losses = losses[selected_idx]

        # ================= TRUST =================
        if self.use_trust:
            trust_scores = self.trust_manager.compute_trust(weights)
        else:
            trust_scores = np.ones(len(weights)) / len(weights)
        
        # --- TRUST FILTERING ---
        trust_threshold = np.percentile(trust_scores, 30)  # bottom 30% removed

        mask = trust_scores > trust_threshold

        filtered_weights = [w for w, m in zip(weights, mask) if m]
        filtered_losses = [l for l, m in zip(losses, mask) if m]
        filtered_trust = trust_scores[mask]

        num_removed = len(weights) - len(filtered_weights)
        print(f"[TRUST] Filtered {num_removed} clients")

        # Fallback (avoid empty aggregation)
        if len(filtered_weights) == 0:
            print("[TRUST WARNING] All clients filtered → fallback to original")
            filtered_weights = weights
            filtered_trust = trust_scores

        # ================= CLIENT WEIGHTS =================
        loss_weights = np.exp(-losses)
        loss_weights /= np.sum(loss_weights)

        if self.use_trust:
            client_weights = 0.5 * loss_weights + 0.5 * trust_scores
        else:
            client_weights = loss_weights

        client_weights /= np.sum(client_weights)

        print(f"[DEBUG] Trust scores: {trust_scores}")
        print(f"[DEBUG] Client weights: {client_weights}")

        # ================= TRUST FILTERING (ONLY FINAL SYSTEM) =================
        if self.use_trust:
            trust_threshold = np.mean(trust_scores) * 0.7

            filtered_weights = []
            filtered_client_weights = []

            for i, w in enumerate(weights):
                if trust_scores[i] >= trust_threshold:
                    filtered_weights.append(w)
                    filtered_client_weights.append(client_weights[i])

            if len(filtered_weights) > 0:
                print(f"[TRUST] Filtered {len(weights) - len(filtered_weights)} clients")
                weights = filtered_weights
                client_weights = np.array(filtered_client_weights)
                client_weights /= np.sum(client_weights)

        # ================= DEFENSE METHODS =================

        if self.method == "clipping":
            weights = clip_updates(weights)

        if self.method == "krum":
            selected = krum_aggregation(weights)
            return fl.common.ndarrays_to_parameters(selected), {}

        if self.method == "dp_server_fixed":
            weights = fixed_clip(weights)

        elif self.method == "dp_server_adaptive":
            weights = adaptive_clip(weights)

        # ================= AGGREGATION =================
        aggregated = []

        for layer in zip(*weights):
            layer_stack = np.stack(layer, axis=0)

            if self.method == "median":
                agg_layer = np.median(layer_stack, axis=0)

            elif self.method == "trimmed_mean":
                n = layer_stack.shape[0]
                k = int(n * 0.2)
                sorted_layer = np.sort(layer_stack, axis=0)
                trimmed = sorted_layer[k:n-k]
                agg_layer = np.mean(trimmed, axis=0)

            else:
                reshape_dims = [len(client_weights)] + [1] * (layer_stack.ndim - 1)
                agg_layer = np.sum(
                    layer_stack * client_weights.reshape(reshape_dims),
                    axis=0
                )

            aggregated.append(agg_layer)

        # ================= MOMENTUM =================
        if self.prev_weights is not None:
            momentum = 0.3
            aggregated = [
                momentum * prev + (1 - momentum) * curr
                for prev, curr in zip(self.prev_weights, aggregated)
            ]

        self.prev_weights = aggregated

        # ================= NOISE =================

        if self.method in ["dp_server_fixed", "dp_server_adaptive"]:
            aggregated = add_adaptive_noise(aggregated, weights, rnd)

        elif self.use_trust:
            aggregated = add_adaptive_noise(aggregated, weights, rnd, trust_scores)
        parameters = fl.common.ndarrays_to_parameters(aggregated)

        epsilons = [
            fit_res.metrics.get("epsilon", 0.0)
            for _, fit_res in results
        ]

        return parameters, {"epsilon": float(np.mean(epsilons))}