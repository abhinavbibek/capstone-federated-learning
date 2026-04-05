#privacy/opacus_dp.py

import torch
import copy
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import warnings
import numpy as np

# Suppress Opacus warning
warnings.filterwarnings(
    "ignore",
    message="Secure RNG turned off"
)

# Suppress PyTorch backward hook warning
warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing"
)

def compute_entropy(probs):
    return - (probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))


def train_with_opacus(model, X, y, epochs, lr, batch_size, noise_multiplier, max_grad_norm, adaptive=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = copy.deepcopy(model)
    model.to(device)
    model.train()

    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).reshape(-1, 1).to(device)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = torch.nn.BCEWithLogitsLoss()
    pos_weight = (len(y) - y.sum()) / (y.sum() + 1e-6)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    privacy_engine = PrivacyEngine()
    

    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    # ADD THESE PRINTS ONLY (no logic change)

    base_noise = noise_multiplier
    base_clip = max_grad_norm

    for epoch in range(epochs):

        epoch_conf_vals = []
        epoch_leak_vals = []

        # =========================
        # 🔥 STEP 1: COMPUTE ADAPTIVE FACTORS (ONCE PER EPOCH)
        # =========================
        if adaptive:
            model.eval()

            with torch.no_grad():
                # Use small subset for speed (important)
                indices = torch.randperm(len(X))[:batch_size]
                sample_x = X[indices].to(device)
                sample_y = y[indices].to(device)

                preds = model(sample_x)
                probs = torch.sigmoid(preds)

                # CONFIDENCE
                entropy = compute_entropy(probs)
                avg_entropy = torch.mean(entropy)
                confidence = 1 - avg_entropy.item()

                # LEAKAGE
                feature = sample_x[:, 0]
                feature_corr = torch.abs(torch.mean(feature * sample_y.squeeze())).item()
                leakage = feature_corr / (feature_corr + 1e-6)

            scale_factor = confidence * leakage

            current_noise = base_noise * (0.5 + scale_factor)
            current_clip = base_clip * (1.0 + scale_factor)

            # clamp
            current_noise = max(0.5, min(2.0, current_noise))
            current_clip = max(0.5, min(5.0, current_clip))

            # 🔥 APPLY ONCE PER EPOCH
            optimizer.noise_multiplier = current_noise
            optimizer.max_grad_norm = current_clip

        else:
            current_noise = base_noise
            current_clip = base_clip
            confidence = 0.0
            leakage = 0.0

        model.train()

        # =========================
        # 🔥 STEP 2: NORMAL TRAINING LOOP (NO ADAPTIVE INSIDE)
        # =========================
        for batch_x, batch_y in dataloader:

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            preds = model(batch_x)
            loss = criterion(preds, batch_y)

            loss.backward()
            optimizer.step()

        # =========================
        # 🔥 LOGGING (PER EPOCH)
        # =========================
        print(f"[DP][Epoch {epoch}] "
            f"Noise={current_noise:.4f} "
            f"Clip={current_clip:.4f} "
            f"Confidence={confidence:.4f} "
            f"Leakage={leakage:.4f}")
    epsilon = privacy_engine.get_epsilon(delta=1e-5)

    clean_state_dict = {
        k: v.detach().cpu() for k, v in model.state_dict().items()
    }

    try:
        privacy_engine.detach()
    except:
        pass

    return clean_state_dict, loss.item(), epsilon