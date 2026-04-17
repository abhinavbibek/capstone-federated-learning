#analysis/run_unified.py
import pandas as pd
from analysis.unified_model import run_unified_system

results = []
baseline_metrics = run_unified_system("baseline", "baseline_combined")
results.append({
    "Model": "Baseline combined",
    "Dataset": "adult+credit",
    "AUC": baseline_metrics["auc"],
    "F1": baseline_metrics["f1"],
    "Privacy": "N/A",
    "Robustness": "N/A"
})

final_metrics = run_unified_system("final_system", "final_combined")

results.append({
    "Model": "Final System combined",
    "Dataset": "adult+credit",
    "AUC": final_metrics["auc"],
    "F1": final_metrics["f1"],
    "Privacy": "Done",
    "Robustness": "Done"
})

df = pd.DataFrame(results)
df.to_csv("results/unified/final_comparison.csv", index=False)
print(df)

