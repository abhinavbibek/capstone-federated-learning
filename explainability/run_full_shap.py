#analysis/run_full_shap.py
import os
import json
import torch
import sys
import pandas as pd
from models.mlp_model import SimpleMLPModel
from analysis.shap_analysis import (
    run_shap_analysis,
    load_test_data,
    compare_experiments,
    get_leakage
)

EXPERIMENTS = [
    "baseline",
    "feature_poison_only",
    "sign_flip_only",
    "dp_local_eps1", "dp_local_eps2", "dp_local_eps5",
    "final_system"
]

dataset = sys.argv[1]
results = []

# Ensure baseline exists
X, _ = load_test_data(dataset)
baseline_model = SimpleMLPModel(X.shape[1])
baseline_model.load_state_dict(torch.load(f"results/{dataset}_baseline_model.pt"))
if not os.path.exists(f"results/shap/{dataset}_baseline_metrics.json"):
    run_shap_analysis("baseline", baseline_model, dataset)

for exp in EXPERIMENTS:
    if exp == "baseline":
        continue
    try:
        expl_path = f"results/shap/{dataset}_{exp}_explanations.json"
        if not os.path.exists(expl_path):
            print(f"Generating SHAP for {exp}")
            model = SimpleMLPModel(X.shape[1])
            model.load_state_dict(torch.load(f"results/{dataset}_{exp}_model.pt"))
            run_shap_analysis(exp, model, dataset)

        # Load metrics
        metrics_path = f"results/shap/{dataset}_{exp}_metrics.json"
        with open(metrics_path) as f:
            metrics_data = json.load(f)

        # Load feature importance
        with open(expl_path) as f:
            expl = json.load(f)
        base_metrics = compare_experiments("baseline", exp, dataset)
        leakage = get_leakage(exp, dataset)
        combined = {
            "experiment": exp,
            "spearman": base_metrics["spearman"],
            "topk_jaccard": base_metrics["topk_jaccard"],
            "drift": base_metrics["drift"],  
            "stability": metrics_data["stability"],        
            "leakage": leakage
        }
        with open(f"results/shap/{dataset}_{exp}_explanations.json") as f:
            expl = json.load(f)

        # Take top 3 features
        top_feats = [f["feature"] for f in expl["global_top_features"][:3]]
        combined["top_features"] = ", ".join(top_feats)
        results.append(combined)
        print(f"Completed {exp}")

    except Exception as e:
        print(f"{exp}: {e}")

df = pd.DataFrame(results)
df.to_csv(f"results/shap/{dataset}_final_shap_results.csv", index=False)
