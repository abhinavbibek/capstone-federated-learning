#privacy/opacus_dp.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import torch.nn as nn


def train_with_opacus(
    model,
    X,
    y,
    epochs,
    lr,
    batch_size,
    noise_multiplier,
    max_grad_norm,
):

    device = torch.device("cpu")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    privacy_engine = PrivacyEngine()

    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    for _ in range(epochs):
        for batch_x, batch_y in dataloader:

            optimizer.zero_grad()

            preds = model(batch_x)
            loss = criterion(preds, batch_y)

            loss.backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    raw_weights = model.state_dict()

    cleaned_weights = {}
    for key in raw_weights:
        new_key = key.replace("_module.", "")
        cleaned_weights[new_key] = raw_weights[key]

    return cleaned_weights, loss.item(), epsilon
