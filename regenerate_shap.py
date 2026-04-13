import torch
from models.mlp_model import SimpleMLPModel
from analysis.shap_analysis import run_shap_analysis, load_test_data

DATASET = "adult"

EXPERIMENTS = [
    "sign_flip_only",
    "dp_local_eps2",
    "final_system"
]

X, _ = load_test_data(DATASET)

for exp in EXPERIMENTS:
    print(f"\nGenerating SHAP for: {exp}")

    model = SimpleMLPModel(X.shape[1])
    model.load_state_dict(torch.load(f"results/{DATASET}_{exp}_model.pt"))

    run_shap_analysis(exp, model, DATASET)

print("\n DONE: High-quality SHAP plots generated")