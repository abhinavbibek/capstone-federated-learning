#federated/client_training.py
import torch
import torch.nn as nn

def train_local(model, X, y, epochs=2, lr=0.01, batch_size=64):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).reshape(-1, 1).to(device)

    dataset_size = len(X)

    for epoch in range(epochs):

        perm = torch.randperm(dataset_size)
        X = X[perm]
        y = y[perm]

        epoch_loss = 0

        for i in range(0, dataset_size, batch_size):

            batch_x = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            optimizer.zero_grad()

            preds = model(batch_x)

            loss = criterion(preds, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    return model.state_dict(), epoch_loss / (dataset_size // batch_size)