import os
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
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

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles loading, preprocessing, and splitting of numeric datasets
    using configurable sklearn pipelines.
    """

    def __init__(self) -> None:
        self.target_col: str = Config.TARGET_COL
        self.test_size: float = Config.TEST_SIZE
        self.random_state: int = Config.RANDOM_STATE
        self.imputation_strategy: str = Config.IMPUTATION_STRATEGY
        self.scaler_type: str = Config.SCALER_TYPE
        self.distribution_transform: str = Config.DISTRIBUTION_TRANSFORM

        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names: Optional[list] = None
        self.processed_data_path: str = Config.PROCESSED_DATA_PATH

    # -------------------------------------------------
    # Data Loading
    # -------------------------------------------------
    def load_data(self, path: str) -> pd.DataFrame:
        """
        Load data from CSV file with error handling.
        
        Args:
            path: Path to CSV file
            
        Returns:
            DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV is malformed
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")
            
            df = pd.read_csv(path)
            logger.info(f"Loaded dataset from {path} with shape {df.shape}")
            return df
        except FileNotFoundError as e:
            logger.error(f"File not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    # -------------------------------------------------
    # Feature / Target Split
    # -------------------------------------------------
    def split_feature_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features and target.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X features, y target)
            
        Raises:
            KeyError: If target column not in DataFrame
        """
        try:
            if self.target_col not in df.columns:
                raise KeyError(f"Target column '{self.target_col}' not found in data")
            
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]
            self.feature_names = X.columns.tolist()
            
            logger.info(f"Split data into {len(self.feature_names)} features and target")
            return X, y
        except Exception as e:
            logger.error(f"Error splitting features and target: {str(e)}")
            raise

    # -------------------------------------------------
    # Pipeline Construction
    # -------------------------------------------------
    def build_pipeline(self) -> None:
        """
        Build sklearn preprocessing pipeline with configured transformers.
        
        Raises:
            ValueError: If invalid configuration values provided
        """
        try:
            imputer = SimpleImputer(strategy=self.imputation_strategy)

            # Distribution transformation
            if self.distribution_transform == "power":
                dist_transformer = PowerTransformer(method="yeo-johnson")
            elif self.distribution_transform == "quantile":
                dist_transformer = QuantileTransformer(
                    output_distribution="normal",
                    random_state=self.random_state
                )
            elif self.distribution_transform == "none":
                dist_transformer = "passthrough"
            else:
                raise ValueError(f"Unknown distribution transform: {self.distribution_transform}")

            # Scaling
            if self.scaler_type == "standard":
                scaler = StandardScaler()
            elif self.scaler_type == "robust":
                scaler = RobustScaler()
            elif self.scaler_type == "minmax":
                scaler = MinMaxScaler()
            elif self.scaler_type == "none":
                scaler = "passthrough"
            else:
                raise ValueError(f"Unknown scaler type: {self.scaler_type}")

            numeric_pipeline = Pipeline(steps=[
                ("imputer", imputer),
                ("distribution_transform", dist_transformer),
                ("scaler", scaler)
            ])

            self.pipeline = ColumnTransformer(transformers=[
                ("numeric_pipeline", numeric_pipeline, self.feature_names)
            ])
            
            logger.info("Pipeline built successfully")
        except Exception as e:
            logger.error(f"Error building pipeline: {str(e)}")
            raise

    # -------------------------------------------------
    # Fit / Transform
    # -------------------------------------------------
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform features using pipeline.
        
        Args:
            X: Input features
            
        Returns:
            Transformed feature array
            
        Raises:
            RuntimeError: If pipeline not built
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built. Call build_pipeline() first.")
        return self.pipeline.fit_transform(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted pipeline.
        
        Args:
            X: Input features
            
        Returns:
            Transformed feature array
            
        Raises:
            RuntimeError: If pipeline not fitted
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not built. Call build_pipeline() first.")
        return self.pipeline.transform(X)

    # -------------------------------------------------
    # Train-Test Split
    # -------------------------------------------------
    def split_train_test(
        self, 
        X: np.ndarray, 
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Args:
            X: Features array
            y: Target series
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=Config.SHUFFLE
        )
        
        logger.info(f"Train-test split: {len(X_train)} train, {len(X_test)} test")
        return X_train, X_test, y_train, y_test

    # -------------------------------------------------
    # Save Processed Dataset
    # -------------------------------------------------
    def save_processed_data(self, X_processed: np.ndarray, y: pd.Series) -> None:
        """
        Save processed dataset to CSV file.
        
        Args:
            X_processed: Transformed features
            y: Target values
        """
        try:
            os.makedirs(os.path.dirname(self.processed_data_path) or ".", exist_ok=True)

            processed_df = pd.DataFrame(
                X_processed,
                columns=self.feature_names
            )
            processed_df[self.target_col] = y.values
            processed_df.to_csv(self.processed_data_path, index=False)

            logger.info(f"Processed dataset saved to: {self.processed_data_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise

    # -------------------------------------------------
    # Full Preprocessing Workflow
    # -------------------------------------------------
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute complete preprocessing workflow.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X, y = self.split_feature_target(df)
        self.build_pipeline()
        X_processed = self.fit_transform(X)

        # Save full processed dataset
        self.save_processed_data(X_processed, y)

        X_train, X_test, y_train, y_test = self.split_train_test(X_processed, y)
        return X_train, X_test, y_train, y_test


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
