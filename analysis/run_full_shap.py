#analysis/run_full_shap.py
import os
import json
import torch
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

import sys
dataset = sys.argv[1]

results = []

# Ensure baseline exists
X, _ = load_test_data(dataset)

baseline_model = SimpleMLPModel(X.shape[1])
baseline_model.load_state_dict(torch.load(f"results/{dataset}_baseline_model.pt"))

if not os.path.exists(f"results/shap/{dataset}_baseline_metrics.json"):
    print("[INFO] Generating baseline SHAP...")
    run_shap_analysis("baseline", baseline_model, dataset)

for exp in EXPERIMENTS:

    if exp == "baseline":
        continue

    try:
        metrics_path = f"results/shap/{dataset}_{exp}_metrics.json"
        if not os.path.exists(metrics_path):
            print(f"[INFO] Generating SHAP for {exp}")

            model = SimpleMLPModel(X.shape[1])
            model.load_state_dict(torch.load(f"results/{dataset}_{exp}_model.pt"))

            run_shap_analysis(exp, model, dataset)

        with open(metrics_path) as f:
            extra = json.load(f)

        base_metrics = compare_experiments("baseline", exp, dataset)
        leakage = get_leakage(exp, dataset)

        combined = {
            "experiment": exp,
            "spearman": base_metrics["spearman"],
            "topk_jaccard": base_metrics["topk_jaccard"],
            "drift": base_metrics["drift"],
            "faithfulness": extra["faithfulness"],
            "stability": extra["stability"],
            "leakage": leakage
        }

        results.append(combined)

        print(f"Completed {exp}")

    except Exception as e:
        print(f"[ERROR] {exp}: {e}")


df = pd.DataFrame(results)
df.to_csv(f"results/shap/{dataset}_final_shap_results.csv", index=False)

print("\n FINAL SHAP ANALYSIS COMPLETE")