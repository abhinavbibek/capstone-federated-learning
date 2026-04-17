#models/mlp_model.py
import torch
import torch.nn as nn

class SimpleMLPModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

    def get_weights(self):
        return self.state_dict()

    def set_weights(self, weights):
        self.load_state_dict(weights)

if __name__ == "__main__":
    print("Testing MLP model")
    model = SimpleMLPModel(input_dim=5)
    # Test forward pass
    test_input = torch.randn(10, 5)  # Batch of 10 samples
    output = model(test_input)
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print(f"\n Model Architecture:")
    print(f"   Input dimension: 5")
    print(f"   Hidden layer: 64 neurons (ReLU)")
    print(f"   Output: 1 neuron (Sigmoid)")

    print(f"\n Model Statistics:")
    print(f"   Total parameters: {total_params}")
    print(f"   Trainable parameters: {trainable_params}")

    print(f"\n Forward Pass Test:")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(
        f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
