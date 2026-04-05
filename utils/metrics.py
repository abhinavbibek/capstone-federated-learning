import torch
from sklearn.metrics import f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, X, y):

    model.eval()
    model = model.to(device)

    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).reshape(-1, 1).to(device)

    with torch.no_grad():

        preds = model(X)

        predicted = (preds > 0.5).float()

        acc = (predicted == y).float().mean().item()

    return acc


def evaluate_with_f1(model, X, y):

    model.eval()

    X = torch.FloatTensor(X).to(device)

    with torch.no_grad():

        preds = model(X)

    predicted = (preds.cpu().numpy() > 0.5).astype(int)

    f1 = f1_score(y, predicted)

    return f1