"""
Create Non-IID data splits for 10 federated clients
Each client has skewed class distribution to simulate realistic heterogeneity
"""
import pickle
import numpy as np
import os

print("="*60)
print("CREATING NON-IID CLIENT DATA SPLITS")
print("="*60)

# Load preprocessed data
print("\n📥 Loading preprocessed dataset...")
with open('data/adult_processed.pkl', 'rb') as f:
    data = pickle.load(f)
X, y = data['X'], data['y']
print(f"✅ Loaded {len(X)} samples")

# Configuration
n_clients = 10
samples_per_client = 2000

# Get indices for each class
idx_class0 = np.where(y == 0)[0]
idx_class1 = np.where(y == 1)[0]

print(f"\n📊 Dataset Statistics:")
print(f"   Class 0 samples: {len(idx_class0)}")
print(f"   Class 1 samples: {len(idx_class1)}")
print(f"   Samples per client: {samples_per_client}")

# Shuffle
np.random.seed(42)  # For reproducibility
np.random.shuffle(idx_class0)
np.random.shuffle(idx_class1)

# Create skewed distributions (label skew)
# This simulates Non-IID data where clients have different class ratios
distributions = [
    (0.9, 0.1),  # Client 1: 90% class 0 (heavy skew)
    (0.8, 0.2),  # Client 2
    (0.7, 0.3),  # Client 3
    (0.6, 0.4),  # Client 4
    (0.5, 0.5),  # Client 5: Balanced
    (0.4, 0.6),  # Client 6
    (0.3, 0.7),  # Client 7
    (0.2, 0.8),  # Client 8
    (0.1, 0.9),  # Client 9: 90% class 1 (heavy skew)
    (0.5, 0.5),  # Client 10: Balanced
]

print(f"\n🎯 Creating {n_clients} clients with Non-IID distributions:")
print(f"{'Client':<10} {'Samples':<10} {'Class 0':<12} {'Class 1':<12} {'Ratio':<15}")
print("-"*60)

client_data_summary = []

for i, (ratio0, ratio1) in enumerate(distributions):
    client_id = i + 1
    n0 = int(samples_per_client * ratio0)
    n1 = int(samples_per_client * ratio1)

    # Sample indices
    start_idx0 = i * 1000 % len(idx_class0)
    start_idx1 = i * 1000 % len(idx_class1)

    client_idx0 = idx_class0[start_idx0:start_idx0 + n0]
    client_idx1 = idx_class1[start_idx1:start_idx1 + n1]

    # Handle wrap-around if needed
    if len(client_idx0) < n0:
        client_idx0 = np.concatenate(
            [client_idx0, idx_class0[:n0 - len(client_idx0)]])
    if len(client_idx1) < n1:
        client_idx1 = np.concatenate(
            [client_idx1, idx_class1[:n1 - len(client_idx1)]])

    client_indices = np.concatenate([client_idx0, client_idx1])
    np.random.shuffle(client_indices)

    client_X = X[client_indices]
    client_y = y[client_indices]

    # Save client data
    os.makedirs('data', exist_ok=True)
    with open(f'data/client_{client_id}.pkl', 'wb') as f:
        pickle.dump({'X': client_X, 'y': client_y, 'client_id': client_id}, f)

    # Summary
    actual_class0 = sum(client_y == 0)
    actual_class1 = sum(client_y == 1)
    ratio_str = f"{actual_class0}:{actual_class1}"

    client_data_summary.append({
        'client_id': client_id,
        'total': len(client_y),
        'class0': actual_class0,
        'class1': actual_class1
    })

    print(
        f"Client {client_id:<3}  {len(client_y):<10} {actual_class0:<12} {actual_class1:<12} {ratio_str:<15}")

print("-"*60)

# Calculate heterogeneity metric (standard deviation of class ratios)
class_ratios = [s['class0'] / s['total'] for s in client_data_summary]
heterogeneity = np.std(class_ratios)
print(f"\n📈 Non-IID Heterogeneity Metric:")
print(f"   Std Dev of Class 0 ratios: {heterogeneity:.3f}")
print(f"   (Higher value = more heterogeneous)")

print("\n💾 Saved files:")
for i in range(1, n_clients + 1):
    print(f"   ✓ data/client_{i}.pkl")

print("\n" + "="*60)
print("✅ NON-IID CLIENT DATA CREATION COMPLETED!")
print("="*60)
