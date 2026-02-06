import pandas as pd
from typing import Optional
from config import Config


class FeatureEngineer:
    """
    Performs domain-driven feature engineering for concrete compressive strength prediction.
    """

    def __init__(self) -> None:
        self.cement_col: str = Config.CEMENT_COL
        self.slag_col: str = Config.SLAG_COL
        self.flyash_col: str = Config.FLYASH_COL
        self.water_col: str = Config.WATER_COL
        self.superplasticizer_col: str = Config.SUPERPLASTICIZER_COL
        self.coarse_agg_col: str = Config.COARSE_AGG_COL
        self.fine_agg_col: str = Config.FINE_AGG_COL
        self.age_col: str = Config.AGE_COL
        self.epsilon: float = Config.EPSILON

    # -------------------------------------------------
    # Domain-Driven Ratios
    # -------------------------------------------------
    def add_water_cement_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate water-cement ratio (fundamental concrete property)."""
        df = df.copy()
        df["Water_Cement_Ratio"] = df[self.water_col] / (df[self.cement_col] + self.epsilon)
        return df

    def add_binder_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate binder content and water-binder ratio."""
        df = df.copy()
        df["Binder_Content"] = (
            df[self.cement_col] +
            df[self.slag_col] +
            df[self.flyash_col]
        )
        df["Water_Binder_Ratio"] = df[self.water_col] / (df["Binder_Content"] + self.epsilon)
        return df

    # -------------------------------------------------
    # Interaction Terms
    # -------------------------------------------------
    def add_interaction_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key concrete ingredients."""
        df = df.copy()

        df["Cement_Water"] = df[self.cement_col] * df[self.water_col]
        df["Cement_Age"] = df[self.cement_col] * df[self.age_col]
        df["Water_Age"] = df[self.water_col] * df[self.age_col]
        df["Aggregate_Ratio"] = df[self.coarse_agg_col] / (df[self.fine_agg_col] + self.epsilon)
        df["Superplasticizer_Water"] = df[self.superplasticizer_col] * df[self.water_col]

        return df

    # -------------------------------------------------
    # Unified Transformation
    # -------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations."""
        df = self.add_water_cement_ratio(df)
        df = self.add_binder_ratio(df)
        df = self.add_interaction_terms(df)
        return df
