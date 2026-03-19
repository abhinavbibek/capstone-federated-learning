"""
Download and preprocess UCI Adult dataset for federated learning
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

print("="*60)
print("DOWNLOADING AND PREPROCESSING UCI ADULT DATASET")
print("="*60)

# Download Adult dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'sex',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

print("\n📥 Downloading dataset from UCI repository...")
df = pd.read_csv(url, names=columns, sep=',\s*',
                 engine='python', na_values='?')
print(f"✅ Downloaded {len(df)} samples")

# Remove missing values
print("\n🧹 Cleaning data (removing missing values)...")
df = df.dropna()
print(f"✅ After cleaning: {len(df)} samples")

# Encode target (income >50K = 1, <=50K = 0)
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

# Select numerical features only for simplicity (can add categorical later)
numerical_cols = ['age', 'education_num',
                  'capital_gain', 'capital_loss', 'hours_per_week']
X = df[numerical_cols].values
y = df['income'].values

print(f"\n📊 Feature Selection:")
print(f"   Selected features: {', '.join(numerical_cols)}")
print(f"   Feature matrix shape: {X.shape}")

# Normalize
print("\n🔧 Normalizing features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Class distribution
class0_count = sum(y == 0)
class1_count = sum(y == 1)
print(f"\n📈 Class Distribution:")
print(
    f"   Income <=50K (Class 0): {class0_count} samples ({class0_count/len(y)*100:.1f}%)")
print(
    f"   Income >50K  (Class 1): {class1_count} samples ({class1_count/len(y)*100:.1f}%)")

# Save
os.makedirs('data', exist_ok=True)
with open('data/adult_processed.pkl', 'wb') as f:
    pickle.dump({'X': X, 'y': y, 'scaler': scaler,
                'feature_names': numerical_cols}, f)

print(f"\n💾 Saved to: data/adult_processed.pkl")
print("\n" + "="*60)
print("✅ DATASET PREPROCESSING COMPLETED SUCCESSFULLY!")
print("="*60)
