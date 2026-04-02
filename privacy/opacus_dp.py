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

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).reshape(-1, 1)

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

    

    for epoch in range(epochs):

        # 🔥 ANNEALING FACTOR (critical fix)
        anneal_factor = 1 / (1 + epoch)
        epoch_noise_vals = []
        epoch_grad_vals = []
        epoch_adaptive_vals = []
        for batch_x, batch_y in dataloader:

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # 🔒 Leakage regularization (ADD HERE)
            leak_feature = batch_x[:, 0].unsqueeze(1)
            batch_x = batch_x - 0.1 * leak_feature

            optimizer.zero_grad()
            preds = model(batch_x)

            probs = torch.sigmoid(preds)

            # =========================
            # 🔥 CONFIDENCE (STABLE)
            # =========================
            entropy = compute_entropy(probs)
            avg_entropy = torch.mean(entropy)

            confidence = 1 - avg_entropy

            # 🔥 ANNEALED CONFIDENCE
            if adaptive:
                adaptive_noise = noise_multiplier * (0.5 + 0.5 * confidence.item()) * anneal_factor
            else:
                adaptive_noise = 0.0

            epoch_adaptive_vals.append(adaptive_noise)
            
          
            loss = criterion(preds, batch_y)
            loss.backward()
            # 🔒 Gradient stabilization 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 🔥 APPLY YOUR CUSTOM METHOD ONLY IF adaptive=True
            if adaptive:

                feature = batch_x[:, 0]
                feature_corr = torch.abs(torch.mean(feature * batch_y.squeeze()))
                feature_score = feature_corr / (feature_corr + 1e-6)
                leakage_weight = 1 + feature_score.item()

                for name, param in model.named_parameters():

                    if param.grad is None:
                        continue

                    grad_norm = torch.norm(param.grad)
                    scale = 1 / (1 + grad_norm)

                    noise_std = adaptive_noise * scale * leakage_weight

                    # 🔒 Stability bounds
                    noise_std = float(noise_std)
                    noise_std = max(0.001, min(0.05, noise_std))
                    epoch_noise_vals.append(noise_std)
                    epoch_grad_vals.append(grad_norm.item())
                    noise = torch.normal(
                        mean=0,
                        std=noise_std,
                        size=param.grad.shape,
                        device=param.grad.device
                    )

                    param.grad += noise 
                    
            optimizer.step()

        noise_mean = np.mean(epoch_noise_vals) if len(epoch_noise_vals) > 0 else 0.0
        grad_mean = np.mean(epoch_grad_vals) if len(epoch_grad_vals) > 0 else 0.0

        print(f"[DP][Epoch {epoch}] "
            f"Noise(mean)={noise_mean:.6f} "
            f"Grad(mean)={grad_mean:.6f} "
            f"AdaptiveNoise={adaptive_noise:.6f}")
    epsilon = privacy_engine.get_epsilon(delta=1e-5)

    clean_state_dict = {
        k: v.detach().cpu() for k, v in model.state_dict().items()
    }

    try:
        privacy_engine.detach()
    except:
        pass

    return clean_state_dict, loss.item(), epsilon
