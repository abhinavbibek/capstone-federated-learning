#server/trust_manager.py
#import numpy as np

# class TrustManager:
#     def __init__(self):
#         self.prev_updates = None
#         self.trust_scores = None

#     def compute_update_distance(self, weights):
#         flat = [np.concatenate([w.flatten() for w in client]) for client in weights]
#         stacked = np.stack(flat)

#         mean_update = np.mean(stacked, axis=0)

#         distances = np.linalg.norm(stacked - mean_update, axis=1)
#         return distances

#     def compute_consistency(self, weights):
#         if self.prev_updates is None:
#             return np.ones(len(weights))

#         if len(weights) != len(self.prev_updates):
#             print("[TRUST] Resetting consistency (client set changed)")
#             return np.ones(len(weights))

#         flat_current = [np.concatenate([w.flatten() for w in client]) for client in weights]
#         flat_prev = [np.concatenate([w.flatten() for w in client]) for client in self.prev_updates]

#         consistency = []
#         for c, p in zip(flat_current, flat_prev):
#             cos_sim = np.dot(c, p) / (np.linalg.norm(c) * np.linalg.norm(p) + 1e-6)
#             consistency.append(cos_sim)

#         return np.array(consistency)

#     def compute_trust(self, weights):
#         distances = self.compute_update_distance(weights)
#         consistency = self.compute_consistency(weights)

#         # Normalize
#         dist_score = np.exp(-distances)

#         if len(dist_score) != len(consistency):
#             print("[TRUST WARNING] Size mismatch → resetting consistency")
#             consistency = np.ones(len(dist_score))
#         cons_score = (consistency + 1) / 2  # [-1,1] → [0,1]

#         trust = 0.6 * dist_score + 0.4 * cons_score

#         trust = trust / (np.sum(trust) + 1e-6)

#         self.prev_updates = weights
#         self.trust_scores = trust

#         print(f"[TRUST] Scores: {trust}")

#         return trust


#server/trust_manager.py
import numpy as np
class TrustManager:
    def __init__(self):
        self.prev_updates = {}  # client_id → flattened weights

    def flatten(self, client_weights):
        return np.concatenate([w.flatten() for w in client_weights])

    def compute_update_distance(self, flat_updates):
        stacked = np.stack([flat_updates[cid] for cid in flat_updates.keys()])
        mean_update = np.mean(stacked, axis=0)

        distances = {
            cid: np.linalg.norm(update - mean_update)
            for cid, update in flat_updates.items()
        }
        return distances

    def compute_consistency(self, flat_updates):
        consistency = {}

        for cid, current in flat_updates.items():
            if cid not in self.prev_updates:
                consistency[cid] = 1.0
                continue

            prev = self.prev_updates[cid]

            cos_sim = np.dot(current, prev) / (
                np.linalg.norm(current) * np.linalg.norm(prev) + 1e-6
            )
            consistency[cid] = cos_sim

        return consistency

    def compute_trust(self, client_ids, weights):
        flat_updates = {
            cid: self.flatten(w)
            for cid, w in zip(client_ids, weights)
        }

        distances = self.compute_update_distance(flat_updates)
        consistency = self.compute_consistency(flat_updates)

        # Convert to arrays in same order
        dist_arr = np.array([distances[cid] for cid in client_ids])
        cons_arr = np.array([consistency[cid] for cid in client_ids])

        dist_arr = (dist_arr - np.mean(dist_arr)) / (np.std(dist_arr) + 1e-6)
        dist_score = np.exp(-dist_arr)
        cons_score = (cons_arr + 1) / 2

        trust = 0.6 * dist_score + 0.4 * cons_score
        trust = trust / (np.sum(trust) + 1e-6)

        # Update memory
        self.prev_updates = flat_updates.copy()

        print(f"[TRUST] Scores: {trust}")

        return trust