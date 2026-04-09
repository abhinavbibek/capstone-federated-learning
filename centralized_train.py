import os

# Directory containing model files
RESULTS_DIR = "results"

# Allowed experiment names
allowed_experiments = {
    "baseline",
    "sign_flip_only",
    "dp_local_eps1",
    "dp_local_eps2",
    "dp_local_eps5",
    "final_system"
}

# Generate allowed filenames
allowed_files = {
    f"adult_{exp}_model.pt" for exp in allowed_experiments
}

# Iterate through files
for filename in os.listdir(RESULTS_DIR):
    file_path = os.path.join(RESULTS_DIR, filename)

    # Only consider .pt files
    if filename.endswith(".pt"):
        if filename not in allowed_files:
            print(f"Deleting: {filename}")
            os.remove(file_path)
        else:
            print(f"Keeping: {filename}")