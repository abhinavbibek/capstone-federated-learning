#privacy/dp_training.py
import torch
import torch.nn as nn
from opacus import PrivacyEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_private(model, X, y, epochs, lr, batch_size,
                  noise_multiplier, max_grad_norm):

    model = model.to(device)

    model.train()

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).reshape(-1, 1).to(device)

    dataset = torch.utils.data.TensorDataset(X, y)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    privacy_engine = PrivacyEngine()

    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    for epoch in range(epochs):

        for batch_x, batch_y in dataloader:

            optimizer.zero_grad()

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)

            loss.backward()

            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=1e-5)

    # ---- CLEAN STATE DICT ----
    raw_weights = model.state_dict()

    cleaned_weights = {}

    for key in raw_weights:

        new_key = key.replace("_module.", "")
        cleaned_weights[new_key] = raw_weights[key]

    return cleaned_weights, loss.item(), epsilon