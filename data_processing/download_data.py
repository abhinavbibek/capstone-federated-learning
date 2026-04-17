# download_data.py
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

DATASET = "credit"   # credit or adult"

# UCI Adult Dataset
if DATASET == "adult":
    print("Preprocessing : UCI Adult")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race', 'sex',
               'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    df = pd.read_csv(url, names=columns, sep=',\s*',
                     engine='python', na_values='?')
    print(f"Original samples: {len(df)}")
    df = df.dropna()
    print(f"After cleaning: {len(df)}")
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
    X = df.drop('income', axis=1)
    y = df['income'].values
    print("\nApplying one-hot encoding...")
    X = pd.get_dummies(X)
    X = X.astype('float32')
    feature_names = X.columns.tolist()
    print(f"Total features after encoding: {len(feature_names)}")
    print("\nSkipping scaling at global level (will scale per client)")
    print(f"Final shape: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    os.makedirs('data', exist_ok=True)

    with open('data/adult_train.pkl', 'wb') as f:
        pickle.dump({'X': X_train, 'y': y_train, 'feature_names': feature_names}, f)

    with open('data/adult_test.pkl', 'wb') as f:
        pickle.dump({'X': X_test, 'y': y_test, 'feature_names': feature_names}, f)


# Credit card fraud dataset
elif DATASET == "credit":
    print("Preprocessing: Credit card fraud")
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    print(f"Original samples: {len(df)}")
    df = df.dropna()
    print(f"After cleaning: {len(df)}")
    y = df['Class'].values
    X = df.drop('Class', axis=1)
    X = X.astype('float32')
    feature_names = X.columns.tolist()
    print(f"Total features: {len(feature_names)}")
    print("\nSkipping scaling at global level (will scale per client)")
    print(f"Final shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    print("\nClass distribution (Train):")
    print(f"Class 0: {sum(y_train==0)}")
    print(f"Class 1: {sum(y_train==1)}")

    print("\nClass distribution (Test):")
    print(f"Class 0: {sum(y_test==0)}")
    print(f"Class 1: {sum(y_test==1)}")

    os.makedirs('data', exist_ok=True)
    with open('data/credit_train.pkl', 'wb') as f:
        pickle.dump({
            'X': X_train,
            'y': y_train,
            'feature_names': feature_names
        }, f)
    with open('data/credit_test.pkl', 'wb') as f:
        pickle.dump({
            'X': X_test,
            'y': y_test,
            'feature_names': feature_names
        }, f)