import os
import json
import pandas as pd

RESULTS_DIR = "results"

# Target rounds
TARGET_ROUNDS = [0, 2, 5, 8, 10, 20, 30, 40]

# Store results per round
round_data = {r: [] for r in TARGET_ROUNDS}

for file in os.listdir(RESULTS_DIR):

    if file.startswith("credit") and file.endswith(".json"):
        path = os.path.join(RESULTS_DIR, file)

        with open(path, "r") as f:
            data = json.load(f)

        # Convert list → dict by round for fast lookup
        round_dict = {entry["round"]: entry for entry in data}

        for r in TARGET_ROUNDS:
            if r in round_dict:
                entry = round_dict[r]

                round_data[r].append({
                    "experiment": file.replace(".json", ""),
                    "accuracy": round(entry.get("accuracy", 0), 4),
                    "f1": round(entry.get("f1", 0), 4),
                    "auc": round(entry.get("auc", 0), 4),
                    "leakage": round(entry.get("leakage", 0), 4),
                    "loss": round(entry.get("loss", 0), 4),
                    "asr": round(entry.get("asr", 0), 4),
                    "epsilon": round(entry.get("epsilon", 0), 4),
                    "mia": round(entry.get("mia", 0), 4),
                    "confidence_gap": round(entry.get("confidence_gap", 0), 4),
                    "entropy": round(entry.get("entropy", 0), 4),
                })

# ==============================
# SAVE CSVs PER ROUND
# ==============================
for r in TARGET_ROUNDS:
    df = pd.DataFrame(round_data[r])

    if not df.empty:
        df = df.sort_values(by="experiment")

        filename = f"results_summary_credit_round_{r}.csv"
        df.to_csv(filename, index=False)

        print(f"\nSaved: {filename}")
        print(df)
    else:
        print(f"\nNo data found for round {r}")