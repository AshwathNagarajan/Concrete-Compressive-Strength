import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocessing:

    def __init__(self, data_path, target_column):
        self.data_path = data_path
        self.target_column = target_column
        self.scaler = None

    def load_data(self):
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nDataset statistics:")
        print(df.describe())
        return df

    def preprocess_data(self, X, fit=True):
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            print(f"Scaler fitted on training data")
        else:
            X_scaled = self.scaler.transform(X)
            print(f"Scaler applied to test data")
        
        return X_scaled

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"\nData split:")
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset statistics:")
    print(df.describe())
    return df


def preprocess_data(X, scaler=None, fit=True):
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Scaler fitted on training data")
    else:
        X_scaled = scaler.transform(X)
        print(f"Scaler applied to test data")
    
    return X_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\nData split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test