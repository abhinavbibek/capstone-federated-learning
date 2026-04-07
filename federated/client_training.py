#federated/client_training.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np

import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce_loss).mean()

def train_local(model, X, y, epochs, lr, batch_size, dataset):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()

    num_pos = y.sum().item()
    num_neg = len(y) - num_pos

    # criterion = nn.BCEWithLogitsLoss()
    y_tensor = torch.FloatTensor(y).to(device)
    pos_weight = (len(y_tensor) - y_tensor.sum()) / (y_tensor.sum() + 1e-6)
    pos_weight = torch.clamp(pos_weight, min=5.0, max=50.0)
    pos_weight = pos_weight.to(device)

    if dataset == "credit":
        criterion = FocalLoss(alpha=0.25, gamma=2)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).reshape(-1, 1).to(device)

    dataset_size = len(X)

    train_dataset = TensorDataset(X, y)

    # =========================
    # 🔥 BALANCED SAMPLER (ONLY CREDIT)
    # =========================
    if dataset == "credit":
        y_np = y.cpu().numpy().astype(int).ravel()

        class_counts = np.bincount(y_np)
        weights = 1.0 / (class_counts + 1e-6)

        sample_weights = weights[y_np]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler
        )

        # 🔥 PRIOR FOR LOGIT ADJUSTMENT
        prior = torch.mean(y).item()

    else:
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        prior = None


    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(epochs):

        epoch_loss = 0
        num_batches = 0

        for batch_x, batch_y in dataloader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)

            loss = criterion(logits, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
    
    return model.state_dict(), epoch_loss / num_batches