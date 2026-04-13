import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path):
    return pd.read_csv(path)


def feature_engineering(df):
    df = df.copy()

    df["water_cement_ratio"] = df["water"] / (df["cement"] + 1e-6)

    df["total_binder"] = df["cement"] + df["blastfurnaceslag"] + df["flyash"]

    df["agg_ratio"] = df["coarseaggregate"] / (df["fineaggregate"] + 1e-6)

    df["log_age"] = np.log1p(df["age"])

    df["binder_water_ratio"] = df["total_binder"] / (df["water"] + 1e-6)

    return df


def split_data(df):
    X = df.drop("strength", axis=1)
    y = df["strength"]
    return X, y


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test