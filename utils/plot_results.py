import json
import matplotlib.pyplot as plt

exp_name = "baseline"  # change later

with open(f"results/{exp_name}.json") as f:
    data = json.load(f)

rounds = [x["round"] for x in data]
acc = [x["accuracy"] for x in data]
f1 = [x["f1"] for x in data]
auc = [x["auc"] for x in data]

plt.figure()
plt.plot(rounds, acc, label="Accuracy")
plt.plot(rounds, f1, label="F1")
plt.plot(rounds, auc, label="AUC")

plt.xlabel("Rounds")
plt.ylabel("Score")
plt.title(f"{exp_name} Performance")
plt.legend()

plt.savefig(f"results/{exp_name}_plot.png")
plt.show()