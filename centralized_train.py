import pickle
import torch
import torch.nn as nn
from models.mlp_model import SimpleMLPModel
from sklearn.metrics import accuracy_score

# Load data
with open("data/train.pkl", "rb") as f:
    train_data = pickle.load(f)

with open("data/test.pkl", "rb") as f:
    test_data = pickle.load(f)

with open("data/global_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Prepare data
X_train = train_data["X"]
y_train = train_data["y"]

X_test = test_data["X"]
y_test = test_data["y"]

# Convert if pandas
if hasattr(X_train, "values"):
    X_train = X_train.values
    X_test = X_test.values

# Scale
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Torch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1)

X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# Model
model = SimpleMLPModel(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Class imbalance
num_pos = y_train.sum().item()
num_neg = len(y_train) - num_pos
pos_weight = torch.tensor([num_neg / (num_pos + 1e-8)])

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Training
epochs = 10
batch_size = 128

for epoch in range(epochs):
    perm = torch.randperm(len(X_train))

    for i in range(0, len(X_train), batch_size):
        idx = perm[i:i+batch_size]

        batch_x = X_train[idx]
        batch_y = y_train[idx]

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    acc = accuracy_score(y_test.numpy(), preds.numpy())

print(f"\n✅ Centralized Accuracy: {acc:.4f}")