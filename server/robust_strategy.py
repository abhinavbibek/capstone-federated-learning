#server/robust_strategy.py
import flwr as fl
import numpy as np
from server.trust_manager import TrustManager
import os, json

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

def normalize_updates(weights):
    normalized = []
    for client_weights in weights:
        total_norm = np.sqrt(sum(np.sum(w**2) for w in client_weights))
        scale = max(total_norm, 1e-6)
        normalized.append([w / scale for w in client_weights])
    return normalized

def add_adaptive_noise(aggregated, weights, rnd, trust_scores=None):
    flat_weights = [np.concatenate([w.flatten() for w in client]) for client in weights]
    stacked = np.stack(flat_weights)

    variance = np.var(stacked, axis=0).mean()
    round_factor = np.exp(-0.1 * rnd)

    avg_trust = np.mean(trust_scores) if trust_scores is not None else 1.0

    noise_std = np.sqrt(variance + 1e-6) * round_factor * (1 + 0.5*(1 - avg_trust))

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

    def __init__(self, method="median", dataset="adult", **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.dataset = dataset   # 🔥 ADD THIS

        # 🔥 TRUST FLAG (ONLY activates for final system)
        self.use_trust = (method == "trust")
        self.latest_fit_metrics = {}

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
        # 🔥 DP SERVER FIX — normalize updates
        if self.method and self.method.startswith("dp_server"):
            weights = normalize_updates(weights)
        # 🔥 GLOBAL CLIPPING (stability)
        weights = clip_updates(weights, threshold=5.0)
        
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        threshold = mean_loss + 0.5 * std_loss
        # 🔥 DO NOT REMOVE CLIENTS
        selected_idx = np.arange(len(losses))   # keep all clients
        print(f"[DEBUG] Soft threshold (not used for removal): {threshold:.4f}")



        # selected_idx = np.where(losses <= threshold)[0]

        # if len(selected_idx) < len(losses) // 2:
        #     print("[DEBUG] Fallback triggered")
        #     selected_idx = np.argsort(losses)[:len(losses)//2]

        # print(f"[DEBUG] Selected clients: {selected_idx}")

        # weights = [weights[i] for i in selected_idx]
        # losses = losses[selected_idx]

        # ================= TRUST =================
        if self.use_trust:
            client_ids = [
                fit_res.metrics["client_id"]
                for _, fit_res in results
            ]# 🔥 ADD THIS LINE
            client_ids = [client_ids[i] for i in selected_idx]  # 🔥 MATCH FILTERED CLIENTS

            # trust_scores = self.trust_manager.compute_trust(client_ids, weights)
            filtered_weights = [weights[i] for i in selected_idx]

            trust_scores = self.trust_manager.compute_trust(client_ids, filtered_weights)
            weights = filtered_weights
            # ================= SAVE TRUST SCORES =================

            # Ensure dataset attribute exists
            dataset = self.dataset if hasattr(self, "dataset") else "unknown"

            os.makedirs(f"results/trust/{dataset}", exist_ok=True)
            trust_log_path = f"results/trust/{dataset}/trust_log.json"

            round_data = {
                "round": int(rnd),
                "client_ids": [int(cid) for cid in client_ids],
                "trust_scores": trust_scores.tolist()
            }

            # Load existing data
            if os.path.exists(trust_log_path):
                try:
                    with open(trust_log_path, "r") as f:
                        all_data = json.load(f)
                except:
                    all_data = []
            else:
                all_data = []

            # Append new round
            all_data.append(round_data)

            # Save
            tmp_path = trust_log_path + ".tmp"

            with open(tmp_path, "w") as f:
                json.dump(all_data, f, indent=4)

            os.replace(tmp_path, trust_log_path)
        else:
            trust_scores = np.ones(len(weights)) / len(weights)
        
        # --- TRUST FILTERING ---
        # if self.use_trust and rnd > 2:   # 🔥 ADD CONDITION + WARMUP

        #     trust_threshold = np.percentile(trust_scores, 10)

        #     mask = trust_scores > trust_threshold

        #     filtered_weights = [w for w, m in zip(weights, mask) if m]
        #     filtered_losses = [l for l, m in zip(losses, mask) if m]
        #     filtered_trust = trust_scores[mask]

        #     num_removed = len(weights) - len(filtered_weights)
        #     print(f"[TRUST] Filtered {num_removed} clients")

        #     # Apply only if not empty
        #     if len(filtered_weights) > 0:
        #         weights = filtered_weights
        #         losses = np.array(filtered_losses)
        #         trust_scores = filtered_trust

        # ================= FLTRUST =================
        if self.method == "fltrust":

            # Normalize updates
            normalized_weights = []
            for client_weights in weights:
                norm = compute_norm(client_weights)
                normalized_weights.append([w / (norm + 1e-6) for w in client_weights])

            # Reference model (mean)
            ref = []
            for layer in zip(*normalized_weights):
                ref.append(np.mean(np.stack(layer, axis=0), axis=0))

            # Cosine similarity trust
            trust_scores = []
            for client_weights in normalized_weights:
                dot = sum(np.sum(w1 * w2) for w1, w2 in zip(client_weights, ref))
                norm1 = compute_norm(client_weights)
                norm2 = compute_norm(ref)
                trust = dot / (norm1 * norm2 + 1e-6)
                trust_scores.append(max(trust, 0))

            trust_scores = np.array(trust_scores)
            trust_scores /= (np.sum(trust_scores) + 1e-6)

            client_weights = trust_scores
        # ================= CLIENT WEIGHTS =================
        # ================= CLIENT WEIGHTS =================

        if self.method == "fltrust":
            # use only trust-based weights
            client_weights = trust_scores

        else:
            loss_weights = np.exp(-losses)
            loss_weights /= np.sum(loss_weights)

            if self.use_trust:
                client_weights = 0.9 * loss_weights + 0.1 * trust_scores
            else:
                client_weights = loss_weights

        client_weights /= np.sum(client_weights)
        print(f"[DEBUG] Trust scores: {trust_scores}")
        print(f"[DEBUG] Client weights: {client_weights}")
        # ================= SOFT TRUST-GUIDED FILTERING =================
        if self.use_trust:

            # ---- 1. Normalize loss (safe scaling) ----
            loss_norm = (losses - np.min(losses)) / (np.max(losses) - np.min(losses) + 1e-6)
            loss_score = np.exp(-loss_norm)

            # ---- 2. Combine trust + loss ----
            combined_score = 0.6 * loss_score + 0.4 * trust_scores

            # ---- 3. Adaptive threshold ----
            threshold_soft = np.percentile(combined_score, 30)

            # ---- 4. Soft weighting (NO removal) ----
            soft_weights = []

            for i in range(len(weights)):
                if combined_score[i] >= threshold_soft:
                    soft_weights.append(combined_score[i])
                else:
                    # ↓ Instead of removing → suppress
                    # soft_weights.append(combined_score[i] * 0.8)
                    soft_weights.append(combined_score[i] * 0.5)

            soft_weights = np.array(soft_weights)
            soft_weights /= (np.sum(soft_weights) + 1e-6)

            # ---- 5. Merge with existing weights ----
            client_weights = 0.7 * client_weights + 0.3 * soft_weights
            client_weights /= (np.sum(client_weights) + 1e-6)

            print(f"[DEBUG] Soft weights: {soft_weights}")
            print(f"[DEBUG] Final adjusted weights: {client_weights}")

        # # ================= TRUST FILTERING (ONLY FINAL SYSTEM) =================
        # if self.use_trust:
        #     trust_threshold = np.mean(trust_scores) * 0.7

        #     filtered_weights = []
        #     filtered_client_weights = []

        #     for i, w in enumerate(weights):
        #         if trust_scores[i] >= trust_threshold:
        #             filtered_weights.append(w)
        #             filtered_client_weights.append(client_weights[i])

        #     if len(filtered_weights) > 0:
        #         print(f"[TRUST] Filtered {len(weights) - len(filtered_weights)} clients")
        #         weights = filtered_weights
        #         client_weights = np.array(filtered_client_weights)
        #         client_weights /= np.sum(client_weights)

        # ================= DEFENSE METHODS =================

        if self.method == "clipping":
            weights = clip_updates(weights, threshold=3.0)

        if self.method == "krum":
            selected = krum_aggregation(weights)
            return fl.common.ndarrays_to_parameters(selected), {}

        

        # ================= AGGREGATION =================
        aggregated = []

        for layer in zip(*weights):
            layer_stack = np.stack(layer, axis=0)

            if self.method == "median":
                agg_layer = np.median(layer_stack, axis=0)

            elif self.method == "trimmed_mean":
                n = layer_stack.shape[0]
                k = min(int(n * 0.1), (n - 2) // 2)

                if n - 2*k <= 0:
                    agg_layer = np.mean(layer_stack, axis=0)
                else:
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
            momentum = 0.0
            aggregated = [
                momentum * prev + (1 - momentum) * curr
                for prev, curr in zip(self.prev_weights, aggregated)
            ]

        self.prev_weights = aggregated

        parameters = fl.common.ndarrays_to_parameters(aggregated)

        epsilons = [
            fit_res.metrics.get("epsilon", 0.0)
            for _, fit_res in results
        ]
        print("\n[DEBUG] CLIENT EPSILONS:")
        for i, (_, fit_res) in enumerate(results):
            print(f"Client {i} epsilon:", fit_res.metrics.get("epsilon"))

        # Store locally in strategy (NO global import)
        self.latest_fit_metrics = {
            "fit_metrics": {
                i: fit_res.metrics for i, (_, fit_res) in enumerate(results)
            },
            "mean_epsilon": float(np.mean(epsilons))   # 🔥 ADD THIS
        }

        return parameters, {
            "epsilon": float(np.mean(epsilons))
        }
    