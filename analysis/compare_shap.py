# analysis/compare_shap.py

from analysis.shap_analysis import compare_experiments, load_shap, load_test_data
from models.mlp_model import SimpleMLPModel
import torch
from analysis.shap_analysis import privacy_interpretability_tradeoff

# =========================
# DEFINE GROUPS
# =========================

# ATTACK_EXPS = [
#     "label_flip_only",
#     "targeted_flip_only",
#     "feature_poison_only",
#     "sign_flip_only",
#     "scaling_only"
# ]

# DP_EXPS = [
#     "dp_local_eps1",
#     "dp_local_eps2",
#     "dp_local_eps5",
#     "dp_hybrid_adaptive"
# ]

# FULL_SYSTEM_EXPS = [
#     "label_flip_median",
#     "label_flip_trimmed",
#     "label_flip_krum",
#     "feature_poison_trimmed",
#     "sign_flip_trimmed",
#     "scaling_trimmed"
# ]

# =========================
# SELECTED EXPERIMENTS (MANUAL CONTROL)
# =========================

SELECTED_EXPERIMENTS = [
    "label_flip_only",      # attack
    "label_flip_median",    # attack + defense
    "dp_local_eps1",        # DP
]

# =========================
# LOAD DATA
# =========================

X, _ = load_test_data()

def interpret_attack(result):
    print("\n[ATTACK INTERPRETABILITY EFFECT]")

    print(f"Spearman: {result['spearman']:.4f} (↓ means ranking changed)")
    print(f"Top-K Jaccard: {result['topk_jaccard']:.4f} (↓ means important features changed)")
    print(f"Drift: {result['drift']:.4f} (↑ means explanation unstable)")


def interpret_dp(result):
    print("\n[DP EFFECT ON INTERPRETABILITY]")

    if result["spearman"] > 0.8:
        print("Minimal distortion")
    elif result["spearman"] > 0.5:
        print("Moderate distortion")
    else:
        print("High distortion")

    print(f"Drift: {result['drift']:.4f}")

def evaluate_group(group_name, experiments):
    print(f"\n\n========== {group_name} ==========")

    for exp in experiments:
        try:
            model = SimpleMLPModel(X.shape[1])
            model.load_state_dict(torch.load(f"results/{exp}_model.pt"))

            result = compare_experiments(
                base_exp="baseline",
                other_exp=exp,
                model=model,
                X=X
            )

            print(f"\n--- {exp} vs baseline ---")
            for k, v in result.items():
                print(f"{k}: {v:.4f}")
            
            # 🔥 INTERPRETATION
            if "flip" in exp or "poison" in exp or "scaling" in exp:
                print("[ATTACK EFFECT]")
                print("↓ Spearman = feature ranking changed")
                print("↓ Jaccard = important features changed")
                print("↑ Drift = unstable explanations")

            elif "dp" in exp:
                print("[DP EFFECT]")
                if result["spearman"] > 0.8:
                    print("Minimal distortion")
                elif result["spearman"] > 0.5:
                    print("Moderate distortion")
                else:
                    print("High distortion")
            

            tradeoff = privacy_interpretability_tradeoff(exp)

        except Exception as e:
            print(f"[ERROR] {exp} failed: {e}")
    

    
# =========================
# RUN ALL
# =========================

# evaluate_group("ATTACK IMPACT", ATTACK_EXPS)
# evaluate_group("DP IMPACT", DP_EXPS)
# evaluate_group("FULL SYSTEM", FULL_SYSTEM_EXPS)

print("\n\n========== SELECTED EXPERIMENTS ==========")

for exp in SELECTED_EXPERIMENTS:
    try:
        model = SimpleMLPModel(X.shape[1])
        model.load_state_dict(torch.load(f"results/{exp}_model.pt"))

        result = compare_experiments(
            base_exp="baseline",
            other_exp=exp,
            model=model,
            X=X
        )

        print(f"\n--- {exp} vs baseline ---")
        for k, v in result.items():
            print(f"{k}: {v:.4f}")

        # 🔥 INTERPRETATION
        if "flip" in exp or "poison" in exp or "scaling" in exp:
            interpret_attack(result)

        elif "dp" in exp:
            interpret_dp(result)

        # 🔥 TRADEOFF
        tradeoff = privacy_interpretability_tradeoff(exp)

    except Exception as e:
        print(f"[ERROR] {exp} failed: {e}")