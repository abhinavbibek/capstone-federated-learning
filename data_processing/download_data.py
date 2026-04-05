import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

print("="*60)
print("RESEARCH-GRADE PREPROCESSING: UCI ADULT")
print("="*60)

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'sex',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df = pd.read_csv(url, names=columns, sep=',\s*',
                 engine='python', na_values='?')

print(f"Original samples: {len(df)}")

# Drop missing
df = df.dropna()
print(f"After cleaning: {len(df)}")

# Target encoding
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

# Separate features
X = df.drop('income', axis=1)
y = df['income'].values

# One-hot encoding (CRITICAL)
print("\n🔧 Applying one-hot encoding...")
X = pd.get_dummies(X)
X = X.astype('float32')

feature_names = X.columns.tolist()

print(f"Total features after encoding: {len(feature_names)}")

print("\nSkipping scaling at global level (will scale per client)")

print(f"Final shape: {X.shape}")

# Split BEFORE any scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Save
os.makedirs('data', exist_ok=True)

with open('data/train.pkl', 'wb') as f:
    pickle.dump({'X': X_train, 'y': y_train, 'feature_names': feature_names}, f)

with open('data/test.pkl', 'wb') as f:
    pickle.dump({'X': X_test, 'y': y_test, 'feature_names': feature_names}, f)
print("\nDATA READY")