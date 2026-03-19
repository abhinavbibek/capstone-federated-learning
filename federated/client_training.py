import torch
import torch.nn as nn


def train_local(model, X, y, epochs=2, lr=0.01, batch_size=64):

    model.train()

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).reshape(-1, 1)

    dataset_size = len(X)

    for epoch in range(epochs):

        for i in range(0, dataset_size, batch_size):

            batch_x = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            optimizer.zero_grad()

            preds = model(batch_x)

            loss = criterion(preds, batch_y)

            loss.backward()

            optimizer.step()

    return model.state_dict(), loss.item()