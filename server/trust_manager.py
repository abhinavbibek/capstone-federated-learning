#server/trust_manager.py
import numpy as np

class TrustManager:
    def __init__(self):
        self.prev_updates = None
        self.trust_scores = None

    def compute_update_distance(self, weights):
        flat = [np.concatenate([w.flatten() for w in client]) for client in weights]
        stacked = np.stack(flat)

        mean_update = np.mean(stacked, axis=0)

        distances = np.linalg.norm(stacked - mean_update, axis=1)
        return distances

    def compute_consistency(self, weights):
        if self.prev_updates is None:
            return np.ones(len(weights))

        if len(weights) != len(self.prev_updates):
            print("[TRUST] Resetting consistency (client set changed)")
            return np.ones(len(weights))

        flat_current = [np.concatenate([w.flatten() for w in client]) for client in weights]
        flat_prev = [np.concatenate([w.flatten() for w in client]) for client in self.prev_updates]

        consistency = []
        for c, p in zip(flat_current, flat_prev):
            cos_sim = np.dot(c, p) / (np.linalg.norm(c) * np.linalg.norm(p) + 1e-6)
            consistency.append(cos_sim)

        return np.array(consistency)

    def compute_trust(self, weights):
        distances = self.compute_update_distance(weights)
        consistency = self.compute_consistency(weights)

        # Normalize
        dist_score = np.exp(-distances)

        if len(dist_score) != len(consistency):
            print("[TRUST WARNING] Size mismatch → resetting consistency")
            consistency = np.ones(len(dist_score))
        cons_score = (consistency + 1) / 2  # [-1,1] → [0,1]

        trust = 0.6 * dist_score + 0.4 * cons_score

        trust = trust / (np.sum(trust) + 1e-6)

        self.prev_updates = weights
        self.trust_scores = trust

        print(f"[TRUST] Scores: {trust}")

        return trust