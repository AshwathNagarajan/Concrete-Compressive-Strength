import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
)

from config import Config


class DataPreprocessor:
    """
    Handles loading, preprocessing, and splitting of numeric datasets
    using configurable sklearn pipelines.
    """

    def __init__(self):
        self.target_col = Config.TARGET_COL
        self.test_size = Config.TEST_SIZE
        self.random_state = Config.RANDOM_STATE
        self.imputation_strategy = Config.IMPUTATION_STRATEGY
        self.scaler_type = Config.SCALER_TYPE
        self.distribution_transform = Config.DISTRIBUTION_TRANSFORM

        self.pipeline = None
        self.feature_names = None
        self.processed_data_path = "data/processed/processed_concrete.csv"

    # -------------------------------------------------
    # Data Loading
    # -------------------------------------------------
    def load_data(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    # -------------------------------------------------
    # Feature / Target Split
    # -------------------------------------------------
    def split_feature_target(self, df: pd.DataFrame):
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        self.feature_names = X.columns.tolist()
        return X, y

    # -------------------------------------------------
    # Pipeline Construction
    # -------------------------------------------------
    def build_pipeline(self):
        imputer = SimpleImputer(strategy=self.imputation_strategy)

        # Distribution transformation
        if self.distribution_transform == "power":
            dist_transformer = PowerTransformer(method="yeo-johnson")
        elif self.distribution_transform == "quantile":
            dist_transformer = QuantileTransformer(
                output_distribution="normal",
                random_state=self.random_state
            )
        else:
            dist_transformer = "passthrough"

        # Scaling
        if self.scaler_type == "standard":
            scaler = StandardScaler()
        elif self.scaler_type == "robust":
            scaler = RobustScaler()
        elif self.scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = "passthrough"

        numeric_pipeline = Pipeline(steps=[
            ("imputer", imputer),
            ("distribution_transform", dist_transformer),
            ("scaler", scaler)
        ])

        self.pipeline = ColumnTransformer(transformers=[
            ("numeric_pipeline", numeric_pipeline, self.feature_names)
        ])

    # -------------------------------------------------
    # Fit / Transform
    # -------------------------------------------------
    def fit_transform(self, X):
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built. Call build_pipeline() first.")
        return self.pipeline.fit_transform(X)

    def transform(self, X):
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built. Call build_pipeline() first.")
        return self.pipeline.transform(X)

    # -------------------------------------------------
    # Train-Test Split
    # -------------------------------------------------
    def split_train_test(self, X, y):
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

    # -------------------------------------------------
    # Save Processed Dataset
    # -------------------------------------------------
    def save_processed_data(self, X_processed, y):
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)

        processed_df = pd.DataFrame(
            X_processed,
            columns=self.feature_names
        )
        processed_df[self.target_col] = y.values
        processed_df.to_csv(self.processed_data_path, index=False)

        print(f"✅ Processed dataset saved to: {self.processed_data_path}")

    # -------------------------------------------------
    # Full Preprocessing Workflow
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame):
        X, y = self.split_feature_target(df)
        self.build_pipeline()
        X_processed = self.fit_transform(X)

        # Save full processed dataset
        self.save_processed_data(X_processed, y)

        X_train, X_test, y_train, y_test = self.split_train_test(X_processed, y)
        return X_train, X_test, y_train, y_test
