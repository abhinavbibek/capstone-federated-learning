#federated/client_training.py
import torch
import torch.nn as nn

def train_local(model, X, y, epochs, lr, batch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()

    num_pos = y.sum().item()
    num_neg = len(y) - num_pos

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).reshape(-1, 1).to(device)

    dataset_size = len(X)

    for epoch in range(epochs):

        perm = torch.randperm(dataset_size)
        X = X[perm]
        y = y[perm]

        epoch_loss = 0
        num_batches = 0

        for i in range(0, dataset_size, batch_size):

            batch_x = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            optimizer.zero_grad()

            preds = model(batch_x)

            loss = criterion(preds, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
    
    return model.state_dict(), epoch_loss / num_batches