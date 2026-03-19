#experiments/baseline_fl_demo.py
"""
Baseline Federated Learning Simulation (No Privacy, No Attacks)
Demonstrates basic federated averaging across 10 clients
"""
import numpy as np
from models.mlp_model import SimpleMLPModel
import pickle
import torch.nn as nn
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_client_data(client_id):
    """Load data for a specific client"""
    with open(f'data/client_{client_id}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y']


def train_client(model, X, y, epochs=2, lr=0.01, batch_size=64):
    """Train model on client's local data"""
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)

    # Simple batch training
    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size

    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            batch_X = X_tensor[start_idx:end_idx]
            batch_y = y_tensor[start_idx:end_idx]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        total_loss = epoch_loss / n_batches

    return model.state_dict(), total_loss


def evaluate_model(model, X, y):
    """Evaluate model accuracy"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)

        outputs = model(X_tensor)
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y_tensor).float().mean().item()

    return accuracy


def federated_average(client_weights):
    """Aggregate client weights using FedAvg algorithm"""
    avg_weights = {}
    for key in client_weights[0].keys():
        avg_weights[key] = torch.stack(
            [w[key].float() for w in client_weights]).mean(0)
    return avg_weights


# Main Federated Learning Simulation
if __name__ == "__main__":
    print("="*70)
    print(" "*15 + "FEDERATED LEARNING BASELINE SIMULATION")
    print("="*70)

    # Configuration
    n_rounds = 5
    n_clients = 10
    clients_per_round = 10  # All clients participate
    local_epochs = 2
    learning_rate = 0.01

    print(f"\n⚙️  Configuration:")
    print(f"   Communication rounds: {n_rounds}")
    print(f"   Total clients: {n_clients}")
    print(f"   Clients per round: {clients_per_round}")
    print(f"   Local epochs: {local_epochs}")
    print(f"   Learning rate: {learning_rate}")

    # Initialize global model
    global_model = SimpleMLPModel(input_dim=5)
    print(
        f"\n🌍 Global model initialized ({sum(p.numel() for p in global_model.parameters())} parameters)")

    # Training loop
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70)

    for round_num in range(1, n_rounds + 1):
        print(f"\n📍 Round {round_num}/{n_rounds}")
        print("-"*70)

        client_weights = []
        client_losses = []

        # Select clients (in this demo, all clients participate)
        selected_clients = list(range(1, n_clients + 1))

        # Each client trains locally
        for client_id in selected_clients:
            # Load client data
            X, y = load_client_data(client_id)

            # Create local model copy
            client_model = SimpleMLPModel(input_dim=5)
            client_model.load_state_dict(global_model.state_dict())

            # Train locally
            weights, loss = train_client(
                client_model, X, y, epochs=local_epochs, lr=learning_rate)
            client_weights.append(weights)
            client_losses.append(loss)

            print(
                f"   Client {client_id:2d} | Loss: {loss:.4f} | Samples: {len(X)}")

        # Aggregate using FedAvg
        print(f"\n   🔄 Aggregating {len(client_weights)} client updates...")
        global_weights = federated_average(client_weights)
        global_model.load_state_dict(global_weights)

        avg_loss = np.mean(client_losses)
        print(f"   📊 Round {round_num} Summary | Avg Loss: {avg_loss:.4f}")

        # Optional: Evaluate on a test client
        if round_num == n_rounds:
            print(f"\n   🎯 Final Evaluation on Client 1:")
            X_test, y_test = load_client_data(1)
            accuracy = evaluate_model(global_model, X_test, y_test)
            print(f"   Accuracy: {accuracy*100:.2f}%")

    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)

    # Save final model
    os.makedirs('results/models', exist_ok=True)
    torch.save(global_model.state_dict(),
               'results/models/baseline_fl_model.pt')
    print(f"\n💾 Final model saved to: results/models/baseline_fl_model.pt")

    print("\n" + "="*70)
    print("✅ FEDERATED LEARNING SIMULATION SUCCESSFUL!")
    print("="*70)
    print("\n📝 Next Steps:")
    print("   1. Add differential privacy (privacy/dp_mechanism.py)")
    print("   2. Add adversarial clients (attacks/label_flipping.py)")
    print("   3. Add robust aggregation (server/aggregation.py)")
    print("   4. Run full experiments with GPU on DGX")
    print("="*70)
