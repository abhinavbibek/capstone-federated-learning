#privacy/opacus_dp.py
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import warnings
import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F

warnings.filterwarnings(
    "ignore",
    message="Secure RNG turned off"
)
warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing"
)

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
        pt = torch.clamp(pt, min=1e-4, max=1.0)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce_loss).mean()

def compute_entropy(probs):
    return - (probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))

def train_with_opacus(model, X, y, epochs, lr, batch_size, noise_multiplier, max_grad_norm, adaptive=False, dataset=None ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = copy.deepcopy(model)
    model.to(device)
    model.train()
    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).reshape(-1, 1).to(device)
    train_dataset = TensorDataset(X, y)
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
        prior = torch.mean(y).item()
        prior = max(min(prior, 0.95), 0.05)
    else:
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        prior = None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if dataset == "credit":
        criterion = FocalLoss(alpha=0.75, gamma=2)
    else:
        criterion = nn.BCEWithLogitsLoss()

    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    base_noise = noise_multiplier
    base_clip = max_grad_norm

    for epoch in range(epochs):
        # compute adaptive factors
        if adaptive:
            model.eval()
            with torch.no_grad():
                indices = torch.randperm(len(X))[:batch_size]
                sample_x = X[indices].to(device)
                sample_y = y[indices].to(device)
                preds = model(sample_x)
                probs = torch.sigmoid(preds)
                entropy = compute_entropy(probs)
                avg_entropy = torch.mean(entropy)
                confidence = 1 - avg_entropy.item()
                feature = sample_x[:, 0]
                feature_corr = torch.abs(torch.mean(feature * sample_y.squeeze())).item()
                leakage = feature_corr / (feature_corr + 1e-6)
            scale_factor = confidence * leakage
            current_noise = base_noise * (0.5 + scale_factor)
            current_clip = base_clip * (1.0 + scale_factor)
            current_noise = max(0.5, min(2.0, current_noise))
            current_clip = max(0.5, min(5.0, current_clip))
            optimizer.noise_multiplier = current_noise
            optimizer.max_grad_norm = current_clip

        else:
            current_noise = base_noise
            current_clip = base_clip
            confidence = 0.0
            leakage = 0.0
        model.train()

        # Normal training no adaptive loop 
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    clean_state_dict = {
        k: v.detach().cpu() for k, v in model.state_dict().items()
    }
    try:
        privacy_engine.detach()
    except:
        pass
    return clean_state_dict, loss.item(), epsilon