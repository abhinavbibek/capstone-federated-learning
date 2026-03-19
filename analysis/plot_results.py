import re
import matplotlib.pyplot as plt

log_file = "results/logs/training_log.txt"

rounds = []
accuracies = []

with open(log_file, "r") as f:
    lines = f.readlines()

current_round = 0

for line in lines:

    if "Round" in line and "/" in line:
        current_round += 1

    if "Global Accuracy" in line:

        match = re.search(r'Global Accuracy (\d+\.\d+)', line)

        if match:
            acc = float(match.group(1))
            rounds.append(current_round)
            accuracies.append(acc)

plt.figure(figsize=(8,5))

plt.plot(rounds, accuracies, marker='o')

plt.title("Federated Learning Performance")

plt.xlabel("Communication Round")

plt.ylabel("Accuracy (%)")

plt.grid(True)

plt.savefig("results/accuracy_vs_round.png")

plt.show()

print("Graph saved to results/accuracy_vs_round.png")