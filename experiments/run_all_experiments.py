#experiments/run_all_experiments.py
import os

EXPERIMENTS = ["baseline", "attack_only", "dp_only", "full_system"]

for exp in EXPERIMENTS:

    print(f"\n=== Running {exp} ===")

    # Start server
    os.system(f"python -m server.fl_server {exp} &")

    # Start clients
    for i in range(1, 6):
        os.system(f"python -m clients.run_client {i} {exp} &")

    # Wait (simple version)
    os.system("sleep 60")

    os.system("pkill -f python")