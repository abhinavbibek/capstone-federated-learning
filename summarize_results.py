import os
import json
import pandas as pd

RESULTS_DIR = "results"

rows = []

for file in os.listdir(RESULTS_DIR):
    if file.startswith("dp_") and file.endswith(".json"):
    # if file.endswith(".json"):
        path = os.path.join(RESULTS_DIR, file)

        with open(path, "r") as f:
            data = json.load(f)

        final_round = data[-1]  # last round

        rows.append({
            "experiment": file.replace(".json", ""),
            "accuracy": round(final_round["accuracy"], 4),
            "f1": round(final_round["f1"], 4),
            "auc": round(final_round["auc"], 4),
            "leakage": round(final_round["leakage"], 4),
        })

# Convert to DataFrame
df = pd.DataFrame(rows)

# Sort nicely
df = df.sort_values(by="experiment")

# Save table
df.to_csv("results_summary_dp.csv", index=False)

print(df)